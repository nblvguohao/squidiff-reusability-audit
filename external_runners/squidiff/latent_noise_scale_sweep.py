"""Test whether the published sampler's hardcoded noise scale transfers.

`sample_squidiff.sampler.sample_around_point` spreads generated latents around
an extrapolated point with an absolute, hardcoded standard deviation:

    def sample_around_point(self, point, num_samples=None, scale=0.7):
        return point + scale * np.random.randn(num_samples, point.shape[0])

`scale` is not adapted to the latent geometry of the data at hand. On the CAR-NK
encoder the per-dimension latent standard deviation is about 0.105, so the
default injects noise of norm 5.42 into a space whose entire within-timepoint
spread has norm 0.81. The extrapolated point is buried, and no error is raised.

This sweeps the scale on a single trained model, holding everything else fixed,
to separate two explanations for the poor extrapolation result: the method not
transferring to this task, or one constant in the sampling helper not
transferring to this latent space.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run(
    sweep_dir: Path,
    model_path: Path,
    output_dir: Path,
    scales: list[float] | None = None,
    seed: int = 13,
) -> dict:
    import anndata as ad
    import torch
    from carnk_latent_extrapolation import (
        LATENT_NOISE_SCALE,
        _dense,
        _encode,
        build_model,
        sample_around_point,
    )

    from reuse_gate.metrics.distribution import energy_distance_multivariate
    from reuse_gate.models.temporal_baselines import (
        conditional_mean_sampler,
        last_observation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ad = ad.read_h5ad(sweep_dir / "train.h5ad")
    test_ad = ad.read_h5ad(sweep_dir / "test.h5ad")
    train_mat, test_mat = _dense(train_ad), _dense(test_ad)
    train_tp = train_ad.obs["timepoint_numeric"].to_numpy()
    test_tp = test_ad.obs["timepoint_numeric"].to_numpy()
    gene_size = train_mat.shape[1]

    model, diffusion = build_model(gene_size, device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    latents = _encode(model, train_mat, device)
    latent_std = float(latents.std())
    z7 = _encode(model, train_mat[train_tp == 7], device)
    z14 = _encode(model, train_mat[train_tp == 14], device)
    direction = z14.mean(axis=0) - z7.mean(axis=0)
    anchor = z14.mean(axis=0)
    within_spread = float(np.linalg.norm(z14.std(axis=0)))

    if scales is None:
        # The upstream default, the empirical latent scale, a value between
        # them, and no spread at all.
        scales = [LATENT_NOISE_SCALE, 0.3, latent_std, latent_std / 2, 0.0]

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

    print(f"latent per-dim std {latents.std(axis=0).mean():.4f}, overall std {latent_std:.4f}")
    print(f"direction norm {np.linalg.norm(direction):.4f}, within-D14 spread {within_spread:.4f}")
    print(f"upstream default scale {LATENT_NOISE_SCALE} injects norm "
          f"{LATENT_NOISE_SCALE * np.sqrt(latents.shape[1]):.3f}")
    print()

    targets = {int(tp): int((test_tp == tp).sum()) for tp in sorted(np.unique(test_tp))}
    results: dict = {
        "model": str(model_path),
        "latent_dim": int(latents.shape[1]),
        "latent_overall_std": latent_std,
        "latent_per_dim_std_mean": float(latents.std(axis=0).mean()),
        "direction_norm": float(np.linalg.norm(direction)),
        "within_d14_spread_norm": within_spread,
        "upstream_default_scale": LATENT_NOISE_SCALE,
        "baselines": baselines,
        "seed": seed,
        "scales": [],
    }

    print(f"  {'scale':>8} | {'noise norm':>10} | {'pooled ED':>10} | {'gen mean':>9} | {'gen std':>8}")
    print(f"  {'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 8}")

    for scale in scales:
        rng = np.random.RandomState(seed)
        generated = []
        for timepoint, n_cells in targets.items():
            steps = (timepoint - 14) / 7.0
            z_target = anchor + direction * steps
            z_batch = (
                np.tile(z_target, (n_cells, 1))
                if scale == 0.0
                else sample_around_point(z_target, n_cells, rng, scale=scale)
            )
            with torch.no_grad():
                out = diffusion.ddim_sample_loop(
                    model,
                    (n_cells, gene_size),
                    model_kwargs={
                        "z_mod": torch.tensor(z_batch, dtype=torch.float32, device=device)
                    },
                    noise=None,
                )
            generated.append(out.cpu().numpy())

        pooled = np.concatenate(generated, axis=0)
        ed = float(energy_distance_multivariate(test_mat, pooled))
        entry = {
            "scale": float(scale),
            "noise_norm": float(scale * np.sqrt(latents.shape[1])),
            "pooled_energy_distance": ed,
            "generated_mean": float(pooled.mean()),
            "generated_std": float(pooled.std()),
            "is_upstream_default": bool(abs(scale - LATENT_NOISE_SCALE) < 1e-9),
        }
        results["scales"].append(entry)
        marker = "  <- upstream default" if entry["is_upstream_default"] else ""
        print(f"  {scale:>8.4f} | {entry['noise_norm']:>10.3f} | {ed:>10.3f} | "
              f"{pooled.mean():>9.3f} | {pooled.std():>8.3f}{marker}")

    print(f"\n  real test population: mean {test_mat.mean():.3f}, std {test_mat.std():.3f}")
    print(f"  conditional_mean baseline: {baselines['conditional_mean']:.3f}")
    print(f"  last_observation baseline: {baselines['last_observation']:.3f}")

    best = min(results["scales"], key=lambda e: e["pooled_energy_distance"])
    default = next(e for e in results["scales"] if e["is_upstream_default"])
    results["verdict"] = {
        "best_scale": best["scale"],
        "best_ed": best["pooled_energy_distance"],
        "upstream_default_ed": default["pooled_energy_distance"],
        "improvement_from_rescaling_noise": default["pooled_energy_distance"]
        - best["pooled_energy_distance"],
        "best_beats_conditional_mean": best["pooled_energy_distance"]
        < baselines["conditional_mean"],
        "best_beats_last_observation": best["pooled_energy_distance"]
        < baselines["last_observation"],
    }
    v = results["verdict"]
    print(f"\n  upstream default scale gives ED {v['upstream_default_ed']:.3f}")
    print(f"  best scale {v['best_scale']:.4f} gives ED {v['best_ed']:.3f}")
    print(f"  beats conditional_mean: {v['best_beats_conditional_mean']}")
    print(f"  beats last_observation: {v['best_beats_last_observation']}")

    out = output_dir / "latent_noise_scale_sweep.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {out}")
    return results


if __name__ == "__main__":
    sweep = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/squidiff_sweep_lognorm")
    model = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("artifacts/squidiff_latent_extrap/steps_50000/model.pt")
    )
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("artifacts/squidiff_latent_extrap")
    run(sweep, model, out)
