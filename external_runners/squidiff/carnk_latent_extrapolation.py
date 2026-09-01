"""Predict late CAR-NK states using Squidiff's own extrapolation mechanism.

Our earlier CAR-NK runs generated with class-conditional sampling, conditioning
on the latest training label. That is not how the authors predict development.
`fig4_VO_reproducibility.ipynb` encodes two observed states, takes the mean
difference in latent space as a direction, steps along it, and samples with the
resulting latent through `z_mod`:

    z_sem     = model.encoder(X)
    direction = z_sem[end].mean(0) - z_sem[start].mean(0)
    z_target  = z_sem[start].mean(0) + direction * i
    z_target  = sample_around_point(z_target, n, scale=0.7)
    samples   = ddim_sample_loop(model, shape, model_kwargs={"z_mod": z_target})

This script applies that mechanism to the temporal split, and runs it under the
released configuration rather than ours: `class_cond=False`, `use_encoder=True`,
`num_layers=3`, matching the args dict that loads VO_diff_model.pt. That also
means the run never touches the conditional branch carrying the three patched
defects.

Direction is estimated from D7 to D14, both in the training set. D21 is one step
beyond D14 and D28 is two, so the held-out timepoints are reached by
extrapolation and never enter the estimate.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "Squidiff"

# From fig4_VO_reproducibility.ipynb, the notebook that loads the released
# checkpoint. num_channels and dropout are not in that dict; defaults are used.
RELEASED_CONFIG = {
    "num_layers": 3,
    "class_cond": False,
    "use_encoder": True,
    "diffusion_steps": 1000,
}
# sample_squidiff.sampler.sample_around_point uses this spread.
LATENT_NOISE_SCALE = 0.7


def _dense(adata) -> np.ndarray:  # noqa: ANN001
    x = adata.X
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def _encode(model, matrix: np.ndarray, device, batch: int = 2048) -> np.ndarray:
    """Run the encoder over a matrix in batches, with no label conditioning."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, matrix.shape[0], batch):
            chunk = torch.tensor(matrix[i : i + batch], dtype=torch.float32, device=device)
            z = model.encoder(chunk, label=None, drug_dose=None, control_feature=None)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def sample_around_point(
    point: np.ndarray, n_samples: int, rng: np.random.RandomState, scale: float = LATENT_NOISE_SCALE
) -> np.ndarray:
    """Spread `n_samples` latents around a point, as sample_squidiff does."""
    return point + scale * rng.randn(n_samples, point.shape[0])


def build_model(gene_size: int, device):
    """Construct the released architecture and return it on `device`."""
    sys.path.insert(0, str(VENDOR))
    from Squidiff.script_util import create_model_and_diffusion

    model, diffusion = create_model_and_diffusion(
        gene_size=gene_size,
        output_dim=gene_size,
        num_channels=128,
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,
        use_drug_structure=False,
        drug_dimension=1024,
        comb_num=1,
        learn_sigma=False,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        **RELEASED_CONFIG,
    )
    return model.to(device), diffusion


def extrapolate(
    model,
    diffusion,
    train_mat: np.ndarray,
    train_tp: np.ndarray,
    targets: dict[int, int],
    gene_size: int,
    device,
    seed: int = 13,
) -> dict[int, np.ndarray]:
    """Generate each target timepoint by stepping along the D7 to D14 direction.

    `targets` maps a held-out timepoint to how many cells to generate for it.
    """
    import torch

    rng = np.random.RandomState(seed)

    z_start = _encode(model, train_mat[train_tp == 7], device)
    z_end = _encode(model, train_mat[train_tp == 14], device)
    direction = z_end.mean(axis=0) - z_start.mean(axis=0)
    anchor = z_end.mean(axis=0)

    print(f"  latent dim {direction.shape[0]}, direction norm {np.linalg.norm(direction):.4f}")
    print(f"  D7 cells {int((train_tp == 7).sum())}, D14 cells {int((train_tp == 14).sum())}")

    generated: dict[int, np.ndarray] = {}
    for timepoint, n_cells in targets.items():
        steps = (timepoint - 14) / 7.0  # D21 is one step past D14, D28 is two
        z_target = anchor + direction * steps
        z_batch = sample_around_point(z_target, n_cells, rng)
        with torch.no_grad():
            out = diffusion.ddim_sample_loop(
                model,
                (n_cells, gene_size),
                model_kwargs={"z_mod": torch.tensor(z_batch, dtype=torch.float32, device=device)},
                noise=None,
            )
        generated[timepoint] = out.cpu().numpy()
        print(f"  D{timepoint}: {steps:.0f} step(s) past D14, generated {n_cells} cells")
    return generated


def run(
    sweep_dir: Path,
    output_root: Path,
    step_counts: list[int],
    batch_size: int = 64,
    seed: int = 13,
) -> dict:
    import anndata as ad
    import torch
    from run_tier0_gpu import train_squidiff_gpu

    from reuse_gate.metrics.distribution import energy_distance_multivariate
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

    targets = {int(tp): int((test_tp == tp).sum()) for tp in sorted(np.unique(test_tp))}
    print(f"Train {train_mat.shape}, test {test_mat.shape}, targets {targets}")

    baselines = {
        "last_observation": float(
            energy_distance_multivariate(test_mat, last_observation(train_mat, test_mat))
        ),
        "conditional_mean": float(
            energy_distance_multivariate(
                test_mat,
                conditional_mean_sampler(
                    train_mat, n_samples=test_mat.shape[0], rng=np.random.RandomState(seed)
                ),
            )
        ),
    }
    print(f"Baselines: { 'last_observation' }={baselines['last_observation']:.3f}, "
          f"conditional_mean={baselines['conditional_mean']:.3f}")

    results: dict = {
        "protocol": "latent extrapolation (fig4_VO_reproducibility.ipynb)",
        "released_config": RELEASED_CONFIG,
        "latent_noise_scale": LATENT_NOISE_SCALE,
        "direction_estimated_from": "D7 -> D14, both in training",
        "seed": seed,
        "baselines": baselines,
        "budgets": [],
    }

    for steps in step_counts:
        print()
        print("=" * 62)
        print(f"BUDGET: {steps} steps, class_cond=False (released configuration)")
        print("=" * 62)
        run_dir = output_root / f"steps_{steps}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.pt"

        t0 = time.time()
        if model_path.exists():
            print(f"Model already trained: {model_path} - reusing")
            train_seconds = None
        else:
            train_result = train_squidiff_gpu(
                train_path=sweep_dir / "train.h5ad",
                output_dir=run_dir,
                gene_size=gene_size,
                diffusion_steps=RELEASED_CONFIG["diffusion_steps"],
                lr_anneal_steps=steps,
                batch_size=batch_size,
                class_cond=False,
                num_layers=RELEASED_CONFIG["num_layers"],
                num_channels=128,
            )
            train_seconds = train_result.get("train_time_seconds")

        model, diffusion = build_model(gene_size, device)
        state = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()

        generated = extrapolate(
            model, diffusion, train_mat, train_tp, targets, gene_size, device, seed=seed
        )

        per_tp = []
        for tp, gen in generated.items():
            real = test_mat[test_tp == tp]
            per_tp.append(
                {
                    "timepoint": tp,
                    "n": int(real.shape[0]),
                    "energy_distance": float(energy_distance_multivariate(real, gen)),
                    "real_mean": float(real.mean()),
                    "generated_mean": float(gen.mean()),
                    "real_std": float(real.std()),
                    "generated_std": float(gen.std()),
                }
            )

        pooled = np.concatenate([generated[tp] for tp in sorted(generated)], axis=0)
        pooled_ed = float(energy_distance_multivariate(test_mat, pooled))
        np.save(run_dir / "latent_extrapolation_generated.npy", pooled)

        entry = {
            "steps": steps,
            "train_seconds": train_seconds,
            "pooled_energy_distance": pooled_ed,
            "per_timepoint": per_tp,
            "generated_finite": bool(np.isfinite(pooled).all()),
            "wall_seconds": round(time.time() - t0, 1),
        }
        results["budgets"].append(entry)

        for e in per_tp:
            print(f"  D{e['timepoint']}: ED={e['energy_distance']:8.3f}  "
                  f"real {e['real_mean']:.3f}/{e['real_std']:.3f}  "
                  f"gen {e['generated_mean']:.3f}/{e['generated_std']:.3f}")
        print(f"  pooled ED = {pooled_ed:.3f}")

        (output_root / "latent_extrapolation_metrics.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

    print()
    print("=" * 62)
    print("SUMMARY: latent extrapolation, released configuration")
    print("=" * 62)
    print(f"  {'steps':>7} | {'pooled ED':>10}")
    print(f"  {'-' * 7}-+-{'-' * 10}")
    for e in results["budgets"]:
        print(f"  {e['steps']:>7} | {e['pooled_energy_distance']:>10.3f}")
    print(f"\n  conditional_mean baseline : {baselines['conditional_mean']:.3f}")
    print(f"  last_observation baseline : {baselines['last_observation']:.3f}")

    best = min(results["budgets"], key=lambda e: e["pooled_energy_distance"])
    results["verdict"] = {
        "best_steps": best["steps"],
        "best_pooled_ed": best["pooled_energy_distance"],
        "beats_conditional_mean": best["pooled_energy_distance"] < baselines["conditional_mean"],
        "beats_last_observation": best["pooled_energy_distance"] < baselines["last_observation"],
    }
    v = results["verdict"]
    print(f"\n  best {v['best_steps']} steps, ED {v['best_pooled_ed']:.3f}")
    print(f"  beats conditional_mean: {v['beats_conditional_mean']}")
    print(f"  beats last_observation: {v['beats_last_observation']}")

    out = output_root / "latent_extrapolation_metrics.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {out}")
    return results


if __name__ == "__main__":
    sweep = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/squidiff_sweep_lognorm")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/squidiff_latent_extrap")
    budgets = (
        [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [5000, 20000, 50000]
    )
    run(sweep, out, budgets)
