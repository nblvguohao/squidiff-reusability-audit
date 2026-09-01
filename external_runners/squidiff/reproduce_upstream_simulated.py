"""Reproduce Squidiff on the upstream simulated dataset.

A Reusability Report should first establish that the method behaves as published
on the authors' own data, before drawing conclusions from new data. Our CAR-NK
results are only interpretable once that baseline exists: without it there is no
way to separate a property of the method from a mistake in our usage.

The simulated dataset is the one reproduction target that needs no downloads.
`prep_simu_data.ipynb` in Squidiff_reproducibility generates it inline under
`np.random.seed(42)`, so it can be regenerated exactly:

  3,000 cells, 200 genes, 3 cell types
  Gaussian per type, means [5, 8, 10], standard deviations [1, 1, 1]
  values clipped at zero

Two details of the upstream cell are preserved deliberately rather than
corrected, because the object is to reproduce what was run:

  - `noise_factor = 0.5` is declared but never applied; the line that would add
    the noise is commented out in the source notebook.
  - The clip at zero is a no-op here, since the smallest mean is five standard
    deviations above zero.

Preprocessing follows the same notebook: normalize_total(1e4) then log1p, then
highly variable gene selection.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM_SEED = 42
N_CELLS = 3000
N_GENES = 200
N_CELL_TYPES = 3
MEAN_EXPRESSION = [5, 8, 10]
STD_EXPRESSION = [1, 1, 1]


def build_simulated_adata(n_top_genes: int = 100):
    """Regenerate the upstream simulated dataset and apply their preprocessing."""
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    rng_state = np.random.get_state()
    np.random.seed(UPSTREAM_SEED)

    cells_per_type = N_CELLS // N_CELL_TYPES
    data = np.zeros((N_CELLS, N_GENES))
    for cell_type in range(N_CELL_TYPES):
        start = cell_type * cells_per_type
        end = (cell_type + 1) * cells_per_type
        data[start:end, :] = np.random.normal(
            loc=MEAN_EXPRESSION[cell_type],
            scale=STD_EXPRESSION[cell_type],
            size=(cells_per_type, N_GENES),
        )
    # Upstream leaves the noise term commented out; kept faithful.
    data = np.maximum(data, 0)
    np.random.set_state(rng_state)

    cell_types = np.repeat(np.arange(N_CELL_TYPES), cells_per_type)
    adata = ad.AnnData(
        X=data.astype(np.float32),
        obs=pd.DataFrame(
            {"Group": cell_types.astype(int)},
            index=[f"Cell_{i + 1}" for i in range(N_CELLS)],
        ),
        var=pd.DataFrame(index=[f"Gene_{i + 1}" for i in range(N_GENES)]),
    )

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, N_GENES - 1))
    adata = adata[:, adata.var.highly_variable].copy()
    return adata


def separability(matrix: np.ndarray, labels: np.ndarray) -> dict:
    """How cleanly the three cell types separate.

    The upstream simulation is built so that cell type is the dominant axis of
    variation, so a model that has learned the data at all should reproduce that
    structure. Silhouette score on the real data gives the target; the same
    score on generated cells, assigned to the nearest real centroid, gives the
    comparison.
    """
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2:
        return {"error": "need at least two groups"}
    return {"silhouette": float(silhouette_score(matrix, labels))}


def preprocessing_degeneracy() -> dict:
    """Track cell-type separability through each preprocessing step.

    The three simulated types differ only by a global expression level, applied
    uniformly across all genes. Library-size normalization is precisely the
    operation that removes a global multiplicative factor, so the step the
    upstream notebook prescribes is expected to annihilate the only signal the
    dataset carries. This measures whether it does.
    """
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    from sklearn.metrics import silhouette_score

    rng_state = np.random.get_state()
    np.random.seed(UPSTREAM_SEED)
    cells_per_type = N_CELLS // N_CELL_TYPES
    data = np.zeros((N_CELLS, N_GENES))
    for cell_type in range(N_CELL_TYPES):
        data[cell_type * cells_per_type : (cell_type + 1) * cells_per_type, :] = (
            np.random.normal(
                MEAN_EXPRESSION[cell_type],
                STD_EXPRESSION[cell_type],
                (cells_per_type, N_GENES),
            )
        )
    data = np.maximum(data, 0)
    np.random.set_state(rng_state)
    labels = np.repeat(np.arange(N_CELL_TYPES), cells_per_type)

    def summarise(matrix: np.ndarray) -> dict:
        return {
            "per_type_mean": [float(matrix[labels == c].mean()) for c in range(N_CELL_TYPES)],
            "silhouette": float(silhouette_score(matrix, labels)),
        }

    stages = {"raw simulation": summarise(data)}

    adata = ad.AnnData(
        X=data.astype(np.float32),
        obs=pd.DataFrame({"Group": labels}, index=[f"c{i}" for i in range(N_CELLS)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)]),
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    stages["after normalize_total"] = summarise(np.asarray(adata.X))
    sc.pp.log1p(adata)
    stages["after normalize + log1p"] = summarise(np.asarray(adata.X))
    return stages


def run(output_dir: Path, train_steps: int = 20000, batch_size: int = 64, seed: int = 13) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "Squidiff"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from run_tier0_gpu import train_squidiff_gpu

    from reuse_gate.metrics.distribution import energy_distance_multivariate

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Regenerate the upstream simulated dataset")
    print("=" * 60)
    adata = build_simulated_adata()
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    labels = adata.obs["Group"].to_numpy()
    print(f"Regenerated: {adata.n_obs} cells x {adata.n_vars} genes after HVG selection")
    print(f"  value range {X.min():.3f} to {X.max():.3f}, mean {X.mean():.3f}, std {X.std():.3f}")
    print(f"  cell types {np.bincount(labels).tolist()}")

    real_sep = separability(X, labels)
    print(f"  silhouette on real data: {real_sep.get('silhouette'):.4f}")

    degeneracy = preprocessing_degeneracy()
    print()
    print("  effect of the prescribed preprocessing on cell-type separability:")
    for stage, v in degeneracy.items():
        means = ", ".join(f"{m:.3f}" for m in v["per_type_mean"])
        print(f"    {stage:<28} means [{means}]   silhouette {v['silhouette']:>7.4f}")

    train_path = output_dir / "simulated_train.h5ad"
    adata.write_h5ad(train_path)

    results: dict = {
        "upstream_seed": UPSTREAM_SEED,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "data_moments": {"mean": float(X.mean()), "std": float(X.std())},
        "real_separability": real_sep,
        "preprocessing_degeneracy": degeneracy,
        "train_steps": train_steps,
        "seed": seed,
    }

    print()
    print("=" * 60)
    print("STEP 2: Train Squidiff on it")
    print("=" * 60)
    t0 = time.time()
    model_path = output_dir / "model.pt"
    if model_path.exists():
        print(f"Model already trained: {model_path} - reusing")
        train_result = {"status": "success", "model_path": str(model_path)}
    else:
        train_result = train_squidiff_gpu(
            train_path=train_path,
            output_dir=output_dir,
            gene_size=adata.n_vars,
            diffusion_steps=100,
            lr_anneal_steps=train_steps,
            batch_size=batch_size,
            class_cond=True,
        )
    results["training"] = {
        k: train_result.get(k) for k in ("status", "n_params", "train_time_seconds", "losses")
    }

    print()
    print("=" * 60)
    print("STEP 3: Generate each cell type and compare against its real population")
    print("=" * 60)
    from run_tier0_gpu import sample_from_model

    # The simulation puts the three types at clearly separated expression levels,
    # so conditioning on each label in turn is the sharpest available check: a
    # model that learned the conditioning should place each generated population
    # at its own type's level rather than at the pooled average.
    per_class = []
    print(f"  {'class':>5} | {'n':>5} | {'real mean':>9} | {'gen mean':>9} | "
          f"{'real std':>8} | {'gen std':>8} | {'ED':>8}")
    print("  " + "-" * 5 + "-+-" + "-" * 5 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-"
          + "-" * 8 + "-+-" + "-" * 8 + "-+-" + "-" * 8)

    for cls in range(N_CELL_TYPES):
        real_cls = X[labels == cls]
        gen_cls = sample_from_model(
            model_path=Path(train_result["model_path"]),
            train_adata_path=train_path,
            output_dir=output_dir,
            n_samples=int(real_cls.shape[0]),
            gene_size=adata.n_vars,
            seed=seed,
            conditioning_class=cls,
            output_name=f"generated_class{cls}.npy",
        )
        entry = {
            "class": cls,
            "n": int(real_cls.shape[0]),
            "real_mean": float(real_cls.mean()),
            "generated_mean": float(gen_cls.mean()),
            "real_std": float(real_cls.std()),
            "generated_std": float(gen_cls.std()),
            "energy_distance": float(energy_distance_multivariate(real_cls, gen_cls)),
        }
        per_class.append(entry)
        print(f"  {cls:>5} | {entry['n']:>5} | {entry['real_mean']:>9.3f} | "
              f"{entry['generated_mean']:>9.3f} | {entry['real_std']:>8.3f} | "
              f"{entry['generated_std']:>8.3f} | {entry['energy_distance']:>8.3f}")

    results["per_class"] = per_class

    # Does the conditioning actually separate the three generated populations?
    gen_means = [e["generated_mean"] for e in per_class]
    real_means = [e["real_mean"] for e in per_class]
    results["conditioning_check"] = {
        "generated_means_ordered": bool(
            all(gen_means[i] < gen_means[i + 1] for i in range(len(gen_means) - 1))
        ),
        "real_means_ordered": bool(
            all(real_means[i] < real_means[i + 1] for i in range(len(real_means) - 1))
        ),
        "generated_mean_spread": float(max(gen_means) - min(gen_means)),
        "real_mean_spread": float(max(real_means) - min(real_means)),
    }
    c = results["conditioning_check"]
    print()
    print(f"  real per-class means      {[round(m, 3) for m in real_means]}"
          f"  spread {c['real_mean_spread']:.3f}")
    print(f"  generated per-class means {[round(m, 3) for m in gen_means]}"
          f"  spread {c['generated_mean_spread']:.3f}")
    print(f"  conditioning preserves the ordering: {c['generated_means_ordered']}")
    print(f"\n  total wall time {time.time() - t0:.1f}s")

    out = output_dir / "reproduction_metrics.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {out}")
    return results


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/squidiff_reproduction")
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    run(out_dir, train_steps=steps)
