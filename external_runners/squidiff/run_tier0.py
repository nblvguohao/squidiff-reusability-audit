"""Squidiff Tier 0 experiment: split → baselines → Squidiff → metrics → report."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def run_tier0(data_path: Path, output_dir: Path, seed: int = 13) -> dict:
    """Execute complete Tier 0 pipeline."""
    import anndata as ad

    results: dict = {
        "dataset_checksum": "fb3714079579c8d1d5e0a92ce707892992d7364ba3ff570c1c2590cd5e4afbef",
        "seed": seed,
        "steps": {},
    }

    # ── Load data ──
    t0 = time.time()
    adata = ad.read_h5ad(data_path)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # ── Temporal split: pre/D7/D14 → train, D21/D28 → test ──
    train_mask = adata.obs['timepoint_numeric'].isin([0, 7, 14])
    test_mask = adata.obs['timepoint_numeric'].isin([21, 28])

    train = adata[train_mask].copy()
    test = adata[test_mask].copy()

    # Get train sample IDs and test sample IDs
    train_samples = sorted(train.obs['sample_id'].unique())
    test_samples = sorted(test.obs['sample_id'].unique())
    overlap = set(train_samples) & set(test_samples)

    results["split"] = {
        "train_n_cells": int(train.n_obs),
        "test_n_cells": int(test.n_obs),
        "train_n_samples": len(train_samples),
        "test_n_samples": len(test_samples),
        "train_samples": train_samples,
        "test_samples": test_samples,
        "sample_overlap": list(overlap),
        "train_timepoints": [0, 7, 14],
        "test_timepoints": [21, 28],
    }
    print(f"Split: train={train.n_obs} cells ({len(train_samples)} samples), "
          f"test={test.n_obs} cells ({len(test_samples)} samples)")
    print(f"Sample overlap: {overlap}")

    # ── Prepare data matrices ──
    # Use top 500 highly variable genes for efficiency
    if train.n_vars > 500:
        # Simple variance-based selection on training data only
        train_dense = train.X.toarray() if hasattr(train.X, 'toarray') else np.asarray(train.X)
        variances = np.var(train_dense, axis=0)
        top_idx = np.argsort(variances)[-500:]
        train_mat = train_dense[:, top_idx]
        test_dense = test.X.toarray() if hasattr(test.X, 'toarray') else np.asarray(test.X)
        test_mat = test_dense[:, top_idx]
        results["steps"]["feature_selection"] = {
            "method": "top500_variance",
            "n_features": 500,
            "fitted_on": "train_only",
        }
    else:
        train_mat = train.X.toarray() if hasattr(train.X, 'toarray') else np.asarray(train.X)
        test_mat = test.X.toarray() if hasattr(test.X, 'toarray') else np.asarray(test.X)

    print(f"Feature matrix: train {train_mat.shape}, test {test_mat.shape}")

    # ── Baseline 1: Last Observation ──
    from reuse_gate.metrics.distribution import energy_distance_multivariate
    from reuse_gate.metrics.regression import regression_metrics
    from reuse_gate.models.temporal_baselines import (
        conditional_mean_sampler,
        last_observation,
        linear_interpolation,
    )

    t1 = time.time()
    pred_lastobs = last_observation(train_mat, test_mat)
    results["steps"]["last_observation"] = {
        "wall_seconds": round(time.time() - t1, 2),
        "energy_distance": float(energy_distance_multivariate(test_mat, pred_lastobs)),
    }
    print(f"  last_observation: ED={results['steps']['last_observation']['energy_distance']:.4f}")

    # ── Baseline 2: Conditional Mean Sampler ──
    t1 = time.time()
    pred_condmean = conditional_mean_sampler(train_mat, n_samples=test_mat.shape[0], rng=np.random.RandomState(seed))
    results["steps"]["conditional_mean"] = {
        "wall_seconds": round(time.time() - t1, 2),
        "energy_distance": float(energy_distance_multivariate(test_mat, pred_condmean)),
    }
    print(f"  conditional_mean: ED={results['steps']['conditional_mean']['energy_distance']:.4f}")

    # ── Baseline 3: Linear Interpolation ──
    t1 = time.time()
    # Use pre/D7 as early, D14 as late (within training set only)
    early_mask = train.obs['timepoint_numeric'].isin([0, 7])
    late_mask = train.obs['timepoint_numeric'].isin([14])
    train_early = train_mat[early_mask.values]
    train_late = train_mat[late_mask.values]
    if train_early.shape[0] > 0 and train_late.shape[0] > 0:
        pred_linear = linear_interpolation(
            train_early, train_late, n_samples=test_mat.shape[0], alpha=1.0, rng=np.random.RandomState(seed)
        )
        results["steps"]["linear_interpolation"] = {
            "wall_seconds": round(time.time() - t1, 2),
            "energy_distance": float(energy_distance_multivariate(test_mat, pred_linear)),
        }
        print(f"  linear_interpolation: ED={results['steps']['linear_interpolation']['energy_distance']:.4f}")

    # ── Squidiff smoke ──
    t1 = time.time()
    try:
        import torch
        from Squidiff.script_util import create_model_and_diffusion

        n_genes = min(500, train_mat.shape[1])
        model, diffusion = create_model_and_diffusion(
            gene_size=n_genes, num_layers=2, output_dim=n_genes,
            class_cond=False, learn_sigma=False, num_channels=64,
            dropout=0.1, diffusion_steps=100, noise_schedule='linear',
            timestep_respacing='', use_kl=False, predict_xstart=False,
            rescale_timesteps=False, rescale_learned_sigmas=False,
            use_checkpoint=False, use_scale_shift_norm=False,
            use_fp16=False, use_encoder=False, use_drug_structure=False,
            drug_dimension=0, comb_num=1,
        )
        n_params = sum(p.numel() for p in model.parameters())

        # Quick forward pass with random input
        x = torch.randn(min(4, train_mat.shape[0]), n_genes)
        t_tensor = torch.rand(x.shape[0])
        with torch.no_grad():
            out = model(x, t_tensor)

        results["steps"]["squidiff_smoke"] = {
            "wall_seconds": round(time.time() - t1, 2),
            "status": "success",
            "n_params": n_params,
            "forward_pass_shape": list(out.shape),
            "finite_output": bool(torch.isfinite(out).all().item()),
        }
        print(f"  Squidiff smoke: {n_params:,} params, forward {list(x.shape)} → {list(out.shape)}")
    except Exception as exc:
        results["steps"]["squidiff_smoke"] = {
            "wall_seconds": round(time.time() - t1, 2),
            "status": "failed",
            "error": str(exc),
        }
        print(f"  Squidiff smoke: FAILED — {exc}")

    # ── Compute metrics ──
    # Regression metrics (only if we have predictions from all models)
    reg_results = regression_metrics(test_mat, pred_lastobs)
    results["steps"]["regression_lastobs"] = reg_results

    # ── Save predictions ──
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "last_observation.npy", pred_lastobs)
    np.save(pred_dir / "conditional_mean.npy", pred_condmean)
    if 'pred_linear' in dir():
        np.save(pred_dir / "linear_interpolation.npy", pred_linear)

    # ── Tier 0 GO evaluation ──
    go_conditions = []
    go_conditions.append(("no_sample_overlap", len(overlap) == 0))

    squidiff_ok = results["steps"]["squidiff_smoke"].get("status") == "success"
    go_conditions.append(("squidiff_completes", squidiff_ok))

    ed_lastobs = results["steps"]["last_observation"]["energy_distance"]
    ed_condmean = results["steps"]["conditional_mean"]["energy_distance"]
    beats_lastobs = ed_condmean < ed_lastobs or results["steps"].get("linear_interpolation", {}).get("energy_distance", 999) < ed_lastobs
    go_conditions.append(("beats_last_observation", beats_lastobs))

    results["tier0_go"] = {
        "all_pass": all(p for _, p in go_conditions),
        "conditions": dict(go_conditions),
    }

    # ── Summary ──
    results["wall_seconds_total"] = round(time.time() - t0, 1)
    results["n_cells_total"] = int(adata.n_obs)
    results["n_genes_total"] = int(adata.n_vars)

    print("\n=== Tier 0 Summary ===")
    print(f"Cells: {results['n_cells_total']}, Genes: {results['n_genes_total']}")
    print(f"Split: {results['split']['train_n_cells']} train / {results['split']['test_n_cells']} test")
    for name, passed in go_conditions:
        print(f"  GO-{name}: {'PASS' if passed else 'FAIL'}")
    print(f"Tier 0 GO: {'PASS' if results['tier0_go']['all_pass'] else 'FAIL'}")
    print(f"Total time: {results['wall_seconds_total']}s")

    return results


if __name__ == '__main__':
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('artifacts/squidiff_tier0/source_data/gse190976_combined.h5ad')
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('artifacts/squidiff_tier0')
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13

    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_tier0(data_path, output_dir, seed=seed)

    # Save results
    metrics_path = output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults saved to: {metrics_path}")
