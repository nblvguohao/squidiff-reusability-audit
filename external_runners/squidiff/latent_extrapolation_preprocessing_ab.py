"""Raw-vs-log-normalized A/B under the published latent-extrapolation protocol.

The Barrier 1 preprocessing result in the manuscript (train_step_sweep.py) uses
class-conditional sampling as a fixed probe, deliberately, to isolate the
preprocessing/training-scale effect from the Barrier 3 latent-noise-scale
confound (the published default of 0.7 is ~67x the extrapolation direction and
would otherwise dominate any comparison). That framing is correct but leaves a
reviewer able to ask why the headline preprocessing finding is not shown under
the same protocol used for the headline performance result.

This script closes that gap. Same split, same seed, same released
architecture (class_cond=False, use_encoder=True, num_layers=3,
diffusion_steps=1000), same latent noise scale, held fixed across both
preprocessing conditions and all three training budgets. Only preprocessing
(raw counts vs normalize_total(1e4)+log1p) varies. The noise scale is fixed at
0.03, the value validation selected for 3 of 5 seeds in the main seed study
(seed_study.py) -- not re-tuned here, to keep this a clean single-variable A/B
rather than reopening scale selection.

The already-trained log-normalized models at artifacts/squidiff_latent_extrap
(steps_5000/20000/50000, released config) are reused rather than retrained.
Only the raw-count side is trained here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

FIXED_NOISE_SCALE = 0.03
RELEASED_CONFIG = {
    "num_layers": 3,
    "class_cond": False,
    "use_encoder": True,
    "diffusion_steps": 1000,
}
BUDGETS = [5000, 20000, 50000]
SEED = 13


def _evaluate_one(model_path: Path, train_mat, train_tp, test_mat, test_tp, gene_size, device):
    import torch
    from carnk_latent_extrapolation import _encode, build_model, sample_around_point

    from reuse_gate.metrics.distribution import energy_distance_multivariate

    model, diffusion = build_model(gene_size, device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    rng = np.random.RandomState(SEED)
    z_d7 = _encode(model, train_mat[train_tp == 7], device)
    z_d14 = _encode(model, train_mat[train_tp == 14], device)
    direction = z_d14.mean(axis=0) - z_d7.mean(axis=0)
    anchor = z_d14.mean(axis=0)

    targets = {int(tp): int((test_tp == tp).sum()) for tp in sorted(np.unique(test_tp))}
    generated = {}
    for tp, n in targets.items():
        steps = (tp - 14) / 7.0
        z_target = anchor + direction * steps
        z_batch = (
            np.tile(z_target, (n, 1))
            if FIXED_NOISE_SCALE == 0.0
            else sample_around_point(z_target, n, rng, scale=FIXED_NOISE_SCALE)
        )
        with torch.no_grad():
            out = diffusion.ddim_sample_loop(
                model,
                (n, gene_size),
                model_kwargs={"z_mod": torch.tensor(z_batch, dtype=torch.float32, device=device)},
                noise=None,
            )
        generated[tp] = out.cpu().numpy()

    pooled = np.concatenate([generated[tp] for tp in sorted(generated)], axis=0)
    return {
        "direction_norm": float(np.linalg.norm(direction)),
        "pooled_energy_distance": float(energy_distance_multivariate(test_mat, pooled)),
        "generated_mean": float(pooled.mean()),
        "generated_std": float(pooled.std()),
        "per_timepoint": {
            str(tp): {
                "energy_distance": float(
                    energy_distance_multivariate(test_mat[test_tp == tp], generated[tp])
                ),
                "n": int((test_tp == tp).sum()),
            }
            for tp in sorted(generated)
        },
    }


def run(
    raw_source_h5ad: Path,
    lognorm_sweep_dir: Path,
    output_root: Path,
) -> dict:
    import anndata as ad
    import torch
    from run_tier0_gpu import prepare_train_data, train_squidiff_gpu

    from reuse_gate.metrics.distribution import energy_distance_multivariate
    from reuse_gate.models.temporal_baselines import conditional_mean_sampler, last_observation

    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Raw-count split, built once ──
    raw_dir = output_root / "raw_split"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("Preparing raw-count split")
    print("=" * 66)
    raw_split = prepare_train_data(raw_source_h5ad, raw_dir, n_genes=500, log_normalize=False)

    # ── Log-normalized split, already on disk from carnk_latent_extrapolation.py ──
    lognorm_train = ad.read_h5ad(lognorm_sweep_dir / "train.h5ad")
    lognorm_test = ad.read_h5ad(lognorm_sweep_dir / "test.h5ad")

    def _dense(a):
        x = a.X
        return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)

    conditions = {
        "raw": {
            "train_mat": raw_split["train_mat"],
            "test_mat": raw_split["test_mat"],
            "train_tp": ad.read_h5ad(raw_split["train_path"]).obs["timepoint_numeric"].to_numpy(),
            "test_tp": ad.read_h5ad(raw_split["test_path"]).obs["timepoint_numeric"].to_numpy(),
            "train_path": Path(raw_split["train_path"]),
            "gene_size": raw_split["n_genes"],
            "existing_models": None,
        },
        "lognorm": {
            "train_mat": _dense(lognorm_train),
            "test_mat": _dense(lognorm_test),
            "train_tp": lognorm_train.obs["timepoint_numeric"].to_numpy(),
            "test_tp": lognorm_test.obs["timepoint_numeric"].to_numpy(),
            "train_path": lognorm_sweep_dir / "train.h5ad",
            "gene_size": lognorm_train.shape[1],
            "existing_models": Path("artifacts/squidiff_latent_extrap"),
        },
    }

    results: dict = {
        "protocol": "latent extrapolation, released configuration, FIXED noise scale",
        "released_config": RELEASED_CONFIG,
        "fixed_noise_scale": FIXED_NOISE_SCALE,
        "seed": SEED,
        "budgets": BUDGETS,
        "conditions": {},
    }

    for cond_name, cond in conditions.items():
        print()
        print("=" * 66)
        print(f"CONDITION: {cond_name}")
        print("=" * 66)

        baselines = {
            "last_observation": float(
                energy_distance_multivariate(
                    cond["test_mat"], last_observation(cond["train_mat"], cond["test_mat"])
                )
            ),
            "conditional_mean": float(
                energy_distance_multivariate(
                    cond["test_mat"],
                    conditional_mean_sampler(
                        cond["train_mat"], cond["test_mat"].shape[0], np.random.RandomState(SEED)
                    ),
                )
            ),
        }
        print(f"  baselines: {baselines}")

        cond_results: dict = {"baselines": baselines, "per_budget": []}

        for steps in BUDGETS:
            run_dir = output_root / cond_name / f"steps_{steps}"
            run_dir.mkdir(parents=True, exist_ok=True)
            model_path = run_dir / "model.pt"

            existing = cond["existing_models"]
            reused_from = None
            if existing is not None and (existing / f"steps_{steps}" / "model.pt").exists():
                model_path = existing / f"steps_{steps}" / "model.pt"
                reused_from = str(model_path)
                print(f"  [{cond_name}] {steps} steps: reusing {model_path}")
            elif model_path.exists():
                print(f"  [{cond_name}] {steps} steps: reusing {model_path}")
            else:
                print(f"  [{cond_name}] {steps} steps: training")
                t0 = time.time()
                train_squidiff_gpu(
                    train_path=cond["train_path"],
                    output_dir=run_dir,
                    gene_size=cond["gene_size"],
                    diffusion_steps=RELEASED_CONFIG["diffusion_steps"],
                    lr_anneal_steps=steps,
                    batch_size=64,
                    class_cond=False,
                    num_layers=RELEASED_CONFIG["num_layers"],
                    num_channels=128,
                )
                print(f"    trained in {time.time() - t0:.1f}s")

            eval_result = _evaluate_one(
                model_path,
                cond["train_mat"],
                cond["train_tp"],
                cond["test_mat"],
                cond["test_tp"],
                cond["gene_size"],
                device,
            )
            eval_result["steps"] = steps
            eval_result["reused_from"] = reused_from
            cond_results["per_budget"].append(eval_result)
            print(f"    ED pooled = {eval_result['pooled_energy_distance']:.3f}  "
                  f"gen mean/std = {eval_result['generated_mean']:.3f}/{eval_result['generated_std']:.3f}")

            results["conditions"][cond_name] = cond_results
            (output_root / "preprocessing_ab_metrics.json").write_text(
                json.dumps(results, indent=2, default=str)
            )

    print()
    print("=" * 66)
    print("SUMMARY: raw vs log-normalized, released protocol, fixed scale 0.03")
    print("=" * 66)
    print(f"  {'steps':>7} | {'raw ED':>12} | {'lognorm ED':>12}")
    for i, steps in enumerate(BUDGETS):
        raw_ed = results["conditions"]["raw"]["per_budget"][i]["pooled_energy_distance"]
        ln_ed = results["conditions"]["lognorm"]["per_budget"][i]["pooled_energy_distance"]
        print(f"  {steps:>7} | {raw_ed:>12.3f} | {ln_ed:>12.3f}")

    out = output_root / "preprocessing_ab_metrics.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {out}")
    return results


if __name__ == "__main__":
    raw_source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "artifacts/squidiff_tier0_gpu/source_data/gse190976_combined.h5ad"
    )
    lognorm_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/squidiff_sweep_lognorm")
    out_root = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("artifacts/squidiff_latent_extrap_ab")
    run(raw_source, lognorm_dir, out_root)
