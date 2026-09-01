"""Multi-seed CAR-NK evaluation with validation-selected sampling noise.

Two things this fixes relative to the single-seed runs.

Every number so far came from one seed, which cannot support any claim about a
trend. Five models are trained here, independently, from seeds 13, 37, 73, 101
and 137.

The earlier noise-scale sweep picked its best value by looking at test-set
energy distance, which is test-set tuning and cannot be reported as a result.
Scale is selected here on a validation task built entirely from training data:

  validation   direction estimated pre-infusion -> D7, extrapolate one step,
               score against the real D14 cells, which the direction never saw
  test         direction estimated D7 -> D14, extrapolate one and two steps,
               score against the held-out D21 and D28 cells

Both are one-step-ahead extrapolations, so a scale chosen on the first is
meaningful for the second, and the held-out timepoints never influence it.

Two numbers are reported per seed. The upstream default of 0.7 is the primary
result, since it is what following the documentation produces. The
validation-selected scale is reported beside it to show what the method can do
once that constant is set appropriately.

Three metrics, since energy distance alone is scale-sensitive and the central
question here concerns scale:

  energy distance                scale-sensitive
  MMD, RBF, bandwidth fixed on training data    scale-sensitive, different kernel
  per-gene mean correlation      invariant to affine rescaling
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEEDS = [13, 37, 73, 101, 137]
CANDIDATE_SCALES = [0.7, 0.3, 0.1, 0.03, 0.0]
UPSTREAM_DEFAULT_SCALE = 0.7


def _generate(model, diffusion, anchor, direction, targets, gene_size, scale, device, rng):
    """Extrapolate along `direction` from `anchor` and sample each target."""
    import torch
    from carnk_latent_extrapolation import sample_around_point

    out = {}
    for label, (steps, n_cells) in targets.items():
        z_target = anchor + direction * steps
        z_batch = (
            np.tile(z_target, (n_cells, 1))
            if scale == 0.0
            else sample_around_point(z_target, n_cells, rng, scale=scale)
        )
        with torch.no_grad():
            samples = diffusion.ddim_sample_loop(
                model,
                (n_cells, gene_size),
                model_kwargs={"z_mod": torch.tensor(z_batch, dtype=torch.float32, device=device)},
                noise=None,
            )
        out[label] = samples.cpu().numpy()
    return out


def _score(real: np.ndarray, generated: np.ndarray, bandwidth: float) -> dict:
    from reuse_gate.metrics.distribution import (
        energy_distance_multivariate,
        mean_expression_correlation,
        mmd_rbf,
    )

    return {
        "energy_distance": float(energy_distance_multivariate(real, generated)),
        "mmd_rbf": float(mmd_rbf(real, generated, bandwidth=bandwidth)),
        "mean_expression_correlation": float(mean_expression_correlation(real, generated)),
        "generated_mean": float(generated.mean()),
        "generated_std": float(generated.std()),
    }


def run(sweep_dir: Path, output_root: Path, train_steps: int = 50000) -> dict:
    import anndata as ad
    import torch
    from carnk_latent_extrapolation import RELEASED_CONFIG, _dense, _encode, build_model
    from run_tier0_gpu import train_squidiff_gpu

    from reuse_gate.metrics.distribution import median_pairwise_distance
    from reuse_gate.models.temporal_baselines import (
        conditional_mean_sampler,
        last_observation,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ad = ad.read_h5ad(sweep_dir / "train.h5ad")
    test_ad = ad.read_h5ad(sweep_dir / "test.h5ad")
    train_mat, test_mat = _dense(train_ad), _dense(test_ad)
    train_tp = train_ad.obs["timepoint_numeric"].to_numpy()
    test_tp = test_ad.obs["timepoint_numeric"].to_numpy()
    gene_size = train_mat.shape[1]

    # Kernel bandwidth is fixed once, on training data only.
    bandwidth = median_pairwise_distance(train_mat)
    print(f"MMD bandwidth fixed on training data: {bandwidth:.4f}")

    test_targets = {
        int(tp): ((int(tp) - 14) / 7.0, int((test_tp == tp).sum()))
        for tp in sorted(np.unique(test_tp))
    }
    real_d14 = train_mat[train_tp == 14]
    validation_targets = {14: (1.0, int(real_d14.shape[0]))}

    baselines = {
        "last_observation": _score(
            test_mat, last_observation(train_mat, test_mat), bandwidth
        ),
        "conditional_mean": _score(
            test_mat,
            conditional_mean_sampler(train_mat, test_mat.shape[0], np.random.RandomState(13)),
            bandwidth,
        ),
    }
    print("baselines on the test split:")
    for name, s in baselines.items():
        print(f"  {name:<18} ED {s['energy_distance']:8.3f}  MMD {s['mmd_rbf']:.5f}  "
              f"meanCorr {s['mean_expression_correlation']:+.4f}")

    results: dict = {
        "protocol": "latent extrapolation, released configuration",
        "released_config": RELEASED_CONFIG,
        "train_steps": train_steps,
        "seeds": SEEDS,
        "candidate_scales": CANDIDATE_SCALES,
        "upstream_default_scale": UPSTREAM_DEFAULT_SCALE,
        "scale_selection": "validation task: pre -> D7 direction, predict D14",
        "mmd_bandwidth": bandwidth,
        "baselines": baselines,
        "per_seed": [],
    }

    for seed in SEEDS:
        print()
        print("=" * 66)
        print(f"SEED {seed}")
        print("=" * 66)
        t0 = time.time()
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.pt"

        if model_path.exists():
            print(f"  model already trained, reusing {model_path}")
        else:
            train_squidiff_gpu(
                train_path=sweep_dir / "train.h5ad",
                output_dir=run_dir,
                gene_size=gene_size,
                diffusion_steps=RELEASED_CONFIG["diffusion_steps"],
                lr_anneal_steps=train_steps,
                batch_size=64,
                class_cond=False,
                num_layers=RELEASED_CONFIG["num_layers"],
                num_channels=128,
            )

        model, diffusion = build_model(gene_size, device)
        state = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()

        z_pre = _encode(model, train_mat[train_tp == 0], device)
        z_d7 = _encode(model, train_mat[train_tp == 7], device)
        z_d14 = _encode(model, real_d14, device)

        # Validation: direction pre -> D7, predict D14.
        val_direction = z_d7.mean(axis=0) - z_pre.mean(axis=0)
        val_anchor = z_d7.mean(axis=0)
        validation = []
        for scale in CANDIDATE_SCALES:
            gen = _generate(
                model, diffusion, val_anchor, val_direction, validation_targets,
                gene_size, scale, device, np.random.RandomState(seed),
            )[14]
            s = _score(real_d14, gen, bandwidth)
            validation.append({"scale": scale, **s})
            print(f"  val scale {scale:<6} ED {s['energy_distance']:9.3f}  "
                  f"MMD {s['mmd_rbf']:.5f}  meanCorr {s['mean_expression_correlation']:+.4f}")

        selected = min(validation, key=lambda e: e["energy_distance"])["scale"]
        print(f"  -> validation selects scale {selected}")

        # Test: direction D7 -> D14, predict D21 and D28.
        test_direction = z_d14.mean(axis=0) - z_d7.mean(axis=0)
        test_anchor = z_d14.mean(axis=0)

        test_scores = {}
        for label, scale in (("upstream_default", UPSTREAM_DEFAULT_SCALE),
                             ("validation_selected", selected)):
            gen = _generate(
                model, diffusion, test_anchor, test_direction, test_targets,
                gene_size, scale, device, np.random.RandomState(seed),
            )
            pooled = np.concatenate([gen[tp] for tp in sorted(gen)], axis=0)
            test_scores[label] = {
                "scale": scale,
                "pooled": _score(test_mat, pooled, bandwidth),
                "per_timepoint": {
                    str(tp): _score(test_mat[test_tp == tp], gen[tp], bandwidth)
                    for tp in sorted(gen)
                },
            }
            p = test_scores[label]["pooled"]
            print(f"  test {label:<20} scale {scale:<5} ED {p['energy_distance']:9.3f}  "
                  f"MMD {p['mmd_rbf']:.5f}  meanCorr {p['mean_expression_correlation']:+.4f}")

        results["per_seed"].append({
            "seed": seed,
            "validation": validation,
            "selected_scale": selected,
            "test": test_scores,
            "wall_seconds": round(time.time() - t0, 1),
        })
        (output_root / "seed_study_metrics.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

    # ── Aggregate ──
    print()
    print("=" * 66)
    print("AGGREGATE ACROSS SEEDS")
    print("=" * 66)
    summary = {}
    for label in ("upstream_default", "validation_selected"):
        for metric in ("energy_distance", "mmd_rbf", "mean_expression_correlation"):
            vals = [s["test"][label]["pooled"][metric] for s in results["per_seed"]]
            summary[f"{label}.{metric}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals],
            }
    summary["selected_scales"] = [s["selected_scale"] for s in results["per_seed"]]
    results["summary"] = summary

    print(f"  selected scales across seeds: {summary['selected_scales']}")
    print()
    print(f"  {'':<22} | {'ED':>18} | {'MMD':>18} | {'mean corr':>18}")
    print(f"  {'-' * 22}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 18}")
    for label in ("upstream_default", "validation_selected"):
        cells = []
        for metric in ("energy_distance", "mmd_rbf", "mean_expression_correlation"):
            s = summary[f"{label}.{metric}"]
            cells.append(f"{s['mean']:>9.4f} ± {s['std']:<6.4f}")
        print(f"  {label:<22} | {cells[0]:>18} | {cells[1]:>18} | {cells[2]:>18}")
    for name, s in baselines.items():
        print(f"  {name:<22} | {s['energy_distance']:>9.4f}          | "
              f"{s['mmd_rbf']:>9.5f}         | {s['mean_expression_correlation']:>+9.4f}")

    out = output_root / "seed_study_metrics.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {out}")
    return results


if __name__ == "__main__":
    sweep = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/squidiff_sweep_lognorm")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/squidiff_seed_study")
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    run(sweep, out, train_steps=steps)
