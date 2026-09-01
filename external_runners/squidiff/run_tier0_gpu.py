"""Squidiff Tier 0 GPU Training Pipeline.

Full pipeline: data prep → split → GPU training → sampling → metrics → report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def ensure_combined_adata(raw_dir: Path, output_path: Path) -> Path:
    """Build combined AnnData from raw MTX files if not already cached."""
    if output_path.exists():
        print(f"Combined AnnData already exists: {output_path}")
        return output_path

    print("Building combined AnnData from raw MTX files...")
    import gzip
    import re

    import anndata as ad
    import pandas as pd
    from scipy.io import mmread
    from scipy.sparse import issparse

    TIMEPOINT_ORDER = {"pre": 0, "D7": 7, "D14": 14, "D21": 21, "D28": 28}

    adatas = []
    raw_files = sorted(raw_dir.glob("GSM*_barcodes.tsv.gz"))

    for barcode_file in raw_files:
        prefix = str(barcode_file).replace("_barcodes.tsv.gz", "")
        matrix_file = Path(prefix + "_matrix.mtx.gz")
        feature_file = Path(prefix + "_features.tsv.gz")

        if not matrix_file.exists() or not feature_file.exists():
            continue

        # Parse sample name for metadata
        name = re.sub(r"^GSM\d+_", "", Path(prefix).name)
        parts = name.split("-")
        if parts[0].startswith("CB"):
            timepoint = "pre"
            construct = "NT"
            if "CARCD19" in name and "IL15" in name:
                construct = "CAR19_IL15"
            elif "CARCD19" in name:
                construct = "CAR19"
            elif "IL15" in name:
                construct = "IL15"
            context = "cord_blood"
        elif re.match(r"D\d+", parts[0]):
            timepoint = parts[0]
            rest = "-".join(parts[1:]).upper()
            if "CARCD19IL15" in rest:
                construct = "CAR19_IL15"
            elif "CARCD19" in rest:
                construct = "CAR19"
            elif "NT" in rest:
                construct = "NT"
            else:
                construct = "unknown"
            if "RAJI" in rest:
                context = "raji"
            elif "IL2" in rest:
                context = "IL2"
            else:
                context = "tumor"
        else:
            continue

        # Read MTX
        with gzip.open(matrix_file, "rb") as fh:
            mat = mmread(fh)
        if issparse(mat):
            mat = mat.tocsc().T.tocsr()
        else:
            mat = mat.T

        barcodes = pd.read_csv(barcode_file, sep="\t", header=None, names=["barcode"])
        features = pd.read_csv(feature_file, sep="\t", header=None, names=["gene_id", "gene_name", "feature_type"])

        gene_mask = (features["feature_type"] == "Gene Expression").values
        if gene_mask.sum() > 0:
            mat = mat[:, gene_mask]
            features = features.loc[gene_mask]

        adata = ad.AnnData(
            X=mat,
            obs=pd.DataFrame(index=barcodes["barcode"].values),
            var=pd.DataFrame(index=features["gene_name"].values),
        )
        adata.obs["sample_id"] = Path(prefix).name
        adata.obs["timepoint_raw"] = timepoint
        adata.obs["timepoint_numeric"] = TIMEPOINT_ORDER.get(timepoint, -1)
        adata.obs["construct"] = construct
        adata.obs["context"] = context
        adata.obs["species"] = "mus_musculus"
        adata.var_names_make_unique()
        adatas.append(adata)
        print(f"  {Path(prefix).name}: {adata.n_obs} cells × {adata.n_vars} genes")

    if not adatas:
        raise RuntimeError("No samples loaded!")

    combined = ad.concat(adatas, join="inner", index_unique="-")
    combined.obs["cell_id"] = combined.obs.index.values
    combined.obs["dataset_id"] = "GSE190976"
    combined.obs["donor_id"] = "mouse_pool"
    combined.obs["engineering_state"] = combined.obs["construct"]
    combined.obs["stimulation_state"] = "NA"
    combined.obs["tumor_context"] = combined.obs["context"].apply(
        lambda c: "tumor" if "raji" in str(c).lower() else "no_tumor"
    )
    combined.obs["split_group"] = "unassigned"
    combined.obs["endpoint_label"] = "NA"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(output_path)
    print(f"\nCombined: {combined.n_obs} cells × {combined.n_vars} genes → {output_path}")
    return output_path


def prepare_train_data(
    adata_path: Path,
    output_dir: Path,
    n_genes: int = 500,
    log_normalize: bool = True,
) -> dict:
    """Create temporal split and prepare Squidiff-compatible training data.

    `log_normalize` reproduces the preprocessing used in the upstream
    reproducibility notebooks (`sc.pp.normalize_total(target_sum=1e4)` followed
    by `sc.pp.log1p`), which the released examples apply before training. Both
    steps act per cell or elementwise, so neither carries information across the
    train/test boundary. Feature selection remains fitted on training cells only.

    Set it to False only to reproduce the raw-count behaviour of earlier runs.
    """
    import anndata as ad
    import scanpy as sc

    adata = ad.read_h5ad(adata_path)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    if log_normalize:
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print("Preprocessing: normalize_total(1e4) + log1p (per upstream notebooks)")
    else:
        print("Preprocessing: none (raw counts)")

    # Temporal split
    train_mask = adata.obs["timepoint_numeric"].isin([0, 7, 14])
    test_mask = adata.obs["timepoint_numeric"].isin([21, 28])

    train = adata[train_mask].copy()
    test = adata[test_mask].copy()

    train_samples = sorted(train.obs["sample_id"].unique())
    test_samples = sorted(test.obs["sample_id"].unique())
    overlap = set(train_samples) & set(test_samples)

    print(f"Train: {train.n_obs} cells ({len(train_samples)} samples)")
    print(f"Test:  {test.n_obs} cells ({len(test_samples)} samples)")
    print(f"Sample overlap: {overlap}")

    # Select top HVGs on training data only
    if train.n_vars > n_genes:
        train_dense = train.X.toarray() if hasattr(train.X, "toarray") else np.asarray(train.X)
        variances = np.var(train_dense, axis=0)
        top_idx = np.argsort(variances)[-n_genes:]
        train = train[:, top_idx].copy()
        test = test[:, top_idx].copy()
        print(f"Feature selection: top {n_genes} HVG (fitted on train only)")

    # Squidiff expects .obs['Group'] column as integer class labels
    # Map timepoint_numeric to sequential class IDs
    timepoints = sorted(train.obs["timepoint_numeric"].unique())
    tp_to_class = {tp: i for i, tp in enumerate(timepoints)}
    train.obs["Group"] = train.obs["timepoint_numeric"].map(tp_to_class).astype(int)

    # Also for test (for evaluation)
    test_tp = sorted(test.obs["timepoint_numeric"].unique())
    for tp in test_tp:
        if tp not in tp_to_class:
            tp_to_class[tp] = len(tp_to_class)
    test.obs["Group"] = test.obs["timepoint_numeric"].map(tp_to_class).astype(int)

    # Save
    train_path = output_dir / "train.h5ad"
    test_path = output_dir / "test.h5ad"
    train.write_h5ad(train_path)
    test.write_h5ad(test_path)
    print(f"Train saved: {train_path} ({train.n_obs} × {train.n_vars})")
    print(f"Test saved:  {test_path} ({test.n_obs} × {test.n_vars})")

    train_dense = train.X.toarray() if hasattr(train.X, "toarray") else np.asarray(train.X)
    test_dense = test.X.toarray() if hasattr(test.X, "toarray") else np.asarray(test.X)

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_n_cells": int(train.n_obs),
        "test_n_cells": int(test.n_obs),
        "n_genes": int(train.n_vars),
        "train_samples": train_samples,
        "test_samples": test_samples,
        "sample_overlap": list(overlap),
        "n_classes": len(tp_to_class),
        "timepoint_mapping": {str(k): v for k, v in tp_to_class.items()},
        "train_mat": train_dense,
        "test_mat": test_dense,
    }


def train_squidiff_gpu(
    train_path: Path,
    output_dir: Path,
    gene_size: int = 500,
    diffusion_steps: int = 100,
    lr_anneal_steps: int = 5000,
    batch_size: int = 64,
    class_cond: bool = True,
    num_layers: int = 2,
    num_channels: int = 64,
) -> dict:
    """Train Squidiff on GPU with training data."""
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "Squidiff"))

    from Squidiff import dist_util, logger
    from Squidiff.resample import create_named_schedule_sampler
    from Squidiff.script_util import create_model_and_diffusion
    from Squidiff.scrna_datasets import prepared_data
    from Squidiff.train_util import TrainLoop

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Monkey-patch dist_util.dev() to return our device
    dist_util.dev = lambda: device

    # Setup logger
    log_dir = output_dir / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(log_dir))

    # Create model
    print(f"\nCreating model: gene_size={gene_size}, num_layers={num_layers}, "
          f"num_channels={num_channels}, class_cond={class_cond}")
    model, diffusion = create_model_and_diffusion(
        gene_size=gene_size,
        output_dim=gene_size,
        num_layers=num_layers,
        num_channels=num_channels,
        dropout=0.1,
        class_cond=class_cond,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,
        use_encoder=True,
        use_drug_structure=False,
        drug_dimension=0,
        comb_num=1,
        learn_sigma=False,
        diffusion_steps=diffusion_steps,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    model.to(device)

    # Prepare data loader
    print(f"Loading training data from: {train_path}")
    data_loader = prepared_data(
        data_dir=str(train_path),
        control_data_dir=None,
        batch_size=batch_size,
        use_drug_structure=False,
        comb_num=1,
    )

    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    # Train
    print(f"\n{'='*60}")
    print(f"Starting GPU training: {lr_anneal_steps} steps, batch_size={batch_size}")
    print(f"{'='*60}")

    t_start = time.time()

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_loader,
        batch_size=batch_size,
        microbatch=batch_size,  # no micro-batching
        lr=1e-4,
        ema_rate="0.9999",
        log_interval=max(1, lr_anneal_steps // 10),
        save_interval=max(1, lr_anneal_steps // 5),
        resume_checkpoint=str(checkpoint_dir),
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=0.0,
        lr_anneal_steps=lr_anneal_steps,
        use_drug_structure=False,
        comb_num=1,
    )

    train_loop.run_loop()

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Save final model
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "gene_size": gene_size,
            "class_cond": class_cond,
            "n_params": n_params,
        },
        model_path,
    )
    print(f"Model saved: {model_path}")

    return {
        "status": "success",
        "model_path": str(model_path),
        "n_params": n_params,
        "train_time_seconds": round(train_time, 1),
        "train_steps": lr_anneal_steps,
        "gene_size": gene_size,
        "diffusion_steps": diffusion_steps,
        "class_cond": class_cond,
        "device": str(device),
        "losses": [float(loss.detach().cpu().numpy()) for loss in train_loop.loss_list[-10:]],
    }


def sample_from_model(
    model_path: Path,
    train_adata_path: Path,
    output_dir: Path,
    n_samples: int,
    gene_size: int,
    seed: int = 13,
    conditioning_class: int | None = None,
    output_name: str = "squidiff_generated.npy",
) -> np.ndarray:
    """Generate cells from trained Squidiff model."""
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "Squidiff"))

    from Squidiff import dist_util
    from Squidiff.script_util import create_model_and_diffusion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist_util.dev = lambda: device

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_gene_size = checkpoint["gene_size"]
    class_cond = checkpoint.get("class_cond", True)

    gene_size = min(gene_size, model_gene_size)

    model, diffusion = create_model_and_diffusion(
        gene_size=gene_size,
        output_dim=gene_size,
        num_layers=2,
        num_channels=64,
        dropout=0.1,
        class_cond=class_cond,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,
        use_encoder=True,
        use_drug_structure=False,
        drug_dimension=0,
        comb_num=1,
        learn_sigma=False,
        diffusion_steps=100,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params on {device}")

    # Determine conditioning class and prepare x_start.
    # The conditioning class is the largest label seen in training. For the
    # temporal split that is the latest observed timepoint, which is the only
    # class available when extrapolating to unseen later ones. Selection is
    # done on `Group` rather than any dataset-specific column so the same code
    # serves the CAR-NK split and the upstream simulated dataset.
    if class_cond:
        import anndata as ad
        train_adata = ad.read_h5ad(train_adata_path)
        if conditioning_class is None:
            conditioning_class = int(train_adata.obs["Group"].max())
        print(f"Conditioning on class {conditioning_class} (largest training label)")

        class_mask = train_adata.obs["Group"].to_numpy() == conditioning_class
        x_start_data = train_adata[class_mask].X
        if hasattr(x_start_data, "toarray"):
            x_start_data = x_start_data.toarray()
        x_start = torch.tensor(x_start_data, dtype=torch.float32, device=device)
        print(f"  x_start reference: {tuple(x_start.shape)} cells of class {conditioning_class}")

    # Generate samples
    print(f"Generating {n_samples} samples...")
    t_start = time.time()

    with torch.no_grad():
        shape = (n_samples, gene_size)

        if class_cond:
            # Randomly select x_start rows to match n_samples
            idx = torch.randint(0, x_start.shape[0], (n_samples,))
            model_kwargs = {
                "x_start": x_start[idx],
                "group": torch.full(
                    (n_samples,), conditioning_class, dtype=torch.float32, device=device
                ),
                "drug_dose": None,
                "control_feature": None,
            }
        else:
            model_kwargs = {
                "drug_dose": None,
                "control_feature": None,
            }

        # Use DDIM for faster sampling
        sample_fn = diffusion.ddim_sample_loop
        generated = sample_fn(model, shape, model_kwargs=model_kwargs)

    generated_np = generated.cpu().numpy()
    gen_time = time.time() - t_start
    print(f"Generated {n_samples} cells in {gen_time:.1f}s")

    # Save
    pred_path = output_dir / output_name
    np.save(pred_path, generated_np)
    print(f"Saved: {pred_path}")

    return generated_np


def run_baselines(train_mat: np.ndarray, test_mat: np.ndarray, seed: int = 13) -> dict:
    """Run all Tier 0 baselines."""
    from reuse_gate.metrics.distribution import energy_distance_multivariate
    from reuse_gate.models.temporal_baselines import (
        conditional_mean_sampler,
        last_observation,
        linear_interpolation,
    )

    results = {}

    # Last observation (takes train + test arrays)
    t1 = time.time()
    pred = last_observation(train_mat, test_mat)
    results["last_observation"] = {
        "wall_seconds": round(time.time() - t1, 2),
        "energy_distance": float(energy_distance_multivariate(test_mat, pred)),
    }

    # Conditional mean
    t1 = time.time()
    pred = conditional_mean_sampler(train_mat, n_samples=test_mat.shape[0], rng=np.random.RandomState(seed))
    results["conditional_mean"] = {
        "wall_seconds": round(time.time() - t1, 2),
        "energy_distance": float(energy_distance_multivariate(test_mat, pred)),
    }

    # Linear interpolation
    t1 = time.time()
    pred = linear_interpolation(train_mat[:train_mat.shape[0]//2], train_mat[train_mat.shape[0]//2:],
                                 n_samples=test_mat.shape[0], alpha=1.0, rng=np.random.RandomState(seed))
    results["linear_interpolation"] = {
        "wall_seconds": round(time.time() - t1, 2),
        "energy_distance": float(energy_distance_multivariate(test_mat, pred)),
    }

    return results


def run_tier0_gpu(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    n_genes: int = 500,
    diffusion_steps: int = 100,
    train_steps: int = 5000,
    batch_size: int = 64,
    seed: int = 13,
) -> dict:
    """Execute complete Tier 0 pipeline with GPU training."""
    if raw_dir is None:
        raw_dir = Path("data/raw")
    if output_dir is None:
        output_dir = Path("artifacts/squidiff_tier0_gpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"seed": seed, "pipeline": "tier0_gpu"}

    # Step 1: Ensure combined AnnData
    print("=" * 60)
    print("STEP 1: Data preparation")
    print("=" * 60)
    combined_path = output_dir / "source_data" / "gse190976_combined.h5ad"
    ensure_combined_adata(raw_dir, combined_path)

    # Step 2: Prepare train/test split
    print("\n" + "=" * 60)
    print("STEP 2: Temporal split + feature selection")
    print("=" * 60)
    split_info = prepare_train_data(combined_path, output_dir, n_genes=n_genes)
    results["split"] = {k: v for k, v in split_info.items() if k not in ("train_mat", "test_mat")}

    # Step 3: Run baselines
    print("\n" + "=" * 60)
    print("STEP 3: Baselines")
    print("=" * 60)
    baselines = run_baselines(split_info["train_mat"], split_info["test_mat"], seed=seed)
    results["baselines"] = baselines
    for name, r in baselines.items():
        print(f"  {name}: ED={r['energy_distance']:.2f} ({r['wall_seconds']}s)")

    # Step 4: GPU Training (skip if model already exists)
    print("\n" + "=" * 60)
    print("STEP 4: Squidiff GPU Training")
    print("=" * 60)
    model_path = output_dir / "model.pt"
    if model_path.exists():
        print(f"Model already exists: {model_path}")
        print("Skipping training, loading existing model...")
        train_result = {"status": "success", "model_path": str(model_path)}
    else:
        train_result = train_squidiff_gpu(
            train_path=Path(split_info["train_path"]),
            output_dir=output_dir,
            gene_size=n_genes,
            diffusion_steps=diffusion_steps,
            lr_anneal_steps=train_steps,
            batch_size=batch_size,
            class_cond=True,
        )
    results["training"] = train_result

    # Step 5: Generate predictions
    print("\n" + "=" * 60)
    print("STEP 5: Generate predictions")
    print("=" * 60)
    try:
        generated = sample_from_model(
            model_path=Path(train_result["model_path"]),
            train_adata_path=Path(split_info["train_path"]),
            output_dir=output_dir,
            n_samples=split_info["test_n_cells"],
            gene_size=n_genes,
            seed=seed,
        )
        results["generation"] = {"status": "success", "n_cells": int(generated.shape[0])}

        # Step 6: Evaluate Squidiff predictions
        print("\n" + "=" * 60)
        print("STEP 6: Squidiff metrics")
        print("=" * 60)
        from reuse_gate.metrics.distribution import energy_distance_multivariate

        squidiff_ed = float(energy_distance_multivariate(split_info["test_mat"], generated))
        results["squidiff"] = {"energy_distance": squidiff_ed}
        print(f"  Squidiff ED: {squidiff_ed:.2f}")
    except Exception as exc:
        results["generation"] = {"status": "failed", "error": str(exc)}
        results["squidiff"] = {"energy_distance": None, "error": str(exc)}
        print(f"  Generation failed: {exc}")

    # Step 7: Tier 0 GO evaluation
    print("\n" + "=" * 60)
    print("STEP 7: Tier 0 GO evaluation")
    print("=" * 60)

    ed_lastobs = results["baselines"]["last_observation"]["energy_distance"]
    ed_squidiff = results["squidiff"].get("energy_distance")

    conditions = [
        ("no_sample_overlap", len(split_info["sample_overlap"]) == 0),
        ("squidiff_trained", train_result["status"] == "success"),
        ("squidiff_generates", results.get("generation", {}).get("status") == "success"),
    ]
    if ed_squidiff is not None:
        conditions.append(("beats_last_observation", ed_squidiff < ed_lastobs))

    results["tier0_go"] = {
        "all_pass": all(p for _, p in conditions),
        "conditions": dict(conditions),
    }

    for name, passed in conditions:
        print(f"  GO-{name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Tier 0 GO: {'PASS' if results['tier0_go']['all_pass'] else 'FAIL'}")

    # Save results
    metrics_path = output_dir / "tier0_gpu_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {metrics_path}")

    return results


if __name__ == "__main__":
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/squidiff_tier0_gpu")
    genes = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 13

    run_tier0_gpu(
        raw_dir=raw,
        output_dir=out,
        n_genes=genes,
        train_steps=steps,
        batch_size=batch,
        seed=seed,
    )
