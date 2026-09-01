"""Run leakage-safe, five-seed Squidiff temporal-cutoff studies.

The configuration fixes every information boundary and training hyperparameter
before target-time data are scored. Preprocessing is per cell, feature
selection is fitted on training cells only, and late-cutoff scale selection
uses only the configured training-window validation triplet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@dataclass(frozen=True)
class CutoffRunConfig:
    """Immutable experiment specification loaded from YAML."""

    name: str
    train_times: tuple[int, ...]
    test_times: tuple[int, ...]
    primary_test_times: tuple[int, ...]
    exploratory_test_times: tuple[int, ...]
    direction_times: tuple[int, int]
    validation_triplet: tuple[int, int, int] | None
    fixed_scales: tuple[float, ...]
    candidate_scales: tuple[float, ...]
    seeds: tuple[int, ...]
    steps: int
    batch_size: int
    class_cond: bool
    use_encoder: bool
    num_layers: int
    diffusion_steps: int
    num_channels: int
    n_genes: int
    log_normalize: bool
    structure_cluster_counts: tuple[int, ...]
    rare_thresholds: tuple[float, ...]
    factor_component_grid: tuple[int, ...]
    factor_variance_target: float


@dataclass(frozen=True)
class PreparedCutoff:
    """Paths and dimensions produced by training-only data preparation."""

    train_path: Path
    test_path: Path
    manifest_path: Path
    n_genes: int


def _tuple(values: Any, cast: type = int) -> tuple[Any, ...]:
    return tuple(cast(value) for value in (values or ()))


def load_cutoff_config(path: Path, name: str) -> CutoffRunConfig:
    """Load and validate one named study from a declarative YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if name not in raw.get("studies", {}):
        raise KeyError(f"unknown cutoff study: {name}")
    model = raw["model"]
    evaluation = raw["evaluation"]
    study = raw["studies"][name]
    validation = study.get("validation_triplet")
    config = CutoffRunConfig(
        name=name,
        train_times=_tuple(study["train_times"]),
        test_times=_tuple(study["test_times"]),
        primary_test_times=_tuple(study["primary_test_times"]),
        exploratory_test_times=_tuple(study["exploratory_test_times"]),
        direction_times=_tuple(study["direction_times"]),
        validation_triplet=_tuple(validation) if validation is not None else None,
        fixed_scales=_tuple(study.get("fixed_scales"), float),
        candidate_scales=_tuple(study.get("candidate_scales"), float),
        seeds=_tuple(study["seeds"]),
        steps=int(model["steps"]),
        batch_size=int(model["batch_size"]),
        class_cond=bool(model["class_cond"]),
        use_encoder=bool(model["use_encoder"]),
        num_layers=int(model["num_layers"]),
        diffusion_steps=int(model["diffusion_steps"]),
        num_channels=int(model["num_channels"]),
        n_genes=int(model["n_genes"]),
        log_normalize=bool(model["log_normalize"]),
        structure_cluster_counts=_tuple(evaluation["structure_cluster_counts"]),
        rare_thresholds=_tuple(evaluation["rare_thresholds"], float),
        factor_component_grid=_tuple(evaluation["factor_component_grid"]),
        factor_variance_target=float(evaluation["factor_variance_target"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: CutoffRunConfig) -> None:
    if set(config.train_times) & set(config.test_times):
        raise ValueError("training and test timepoints overlap")
    if not set(config.direction_times).issubset(config.train_times):
        raise ValueError("direction timepoints must be in the training window")
    if config.validation_triplet is not None:
        if not set(config.validation_triplet).issubset(config.train_times):
            raise ValueError("validation triplet must be inside the training window")
        if set(config.validation_triplet) & set(config.test_times):
            raise ValueError("validation triplet overlaps the test window")
    if set(config.primary_test_times) | set(config.exploratory_test_times) != set(
        config.test_times
    ):
        raise ValueError("primary and exploratory targets must partition test_times")
    if config.validation_triplet is None and not config.fixed_scales:
        raise ValueError("a study without validation must predeclare fixed scales")
    if config.validation_triplet is not None and not config.candidate_scales:
        raise ValueError("a validation study must declare candidate scales")
    if config.class_cond or not config.use_encoder:
        raise ValueError("cutoff studies require class_cond=False and use_encoder=True")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dense(adata: Any) -> npt.NDArray[Any]:
    matrix = adata.X
    return np.asarray(
        matrix.toarray() if hasattr(matrix, "toarray") else matrix,
        dtype=np.float32,
    )


def prepare_cutoff_data(
    full_adata_path: Path,
    output_dir: Path,
    config: CutoffRunConfig,
    *,
    n_genes: int | None = None,
    log_normalize: bool | None = None,
) -> PreparedCutoff:
    """Normalize per cell, split, and fit feature selection on training only."""
    import anndata as ad
    import scanpy as sc

    from reuse_gate.splits.temporal_cutoffs import CutoffSpec, build_temporal_cutoff

    output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(full_adata_path)
    normalize = config.log_normalize if log_normalize is None else log_normalize
    if normalize:
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    spec = CutoffSpec(
        name=config.name,
        train_times=config.train_times,
        test_times=config.test_times,
        direction_times=config.direction_times,
        validation_triplet=config.validation_triplet,
        fixed_scale_sensitivity=config.fixed_scales,
    )
    cutoff = build_temporal_cutoff(adata, spec)
    train, test = cutoff.train, cutoff.test

    requested_genes = config.n_genes if n_genes is None else int(n_genes)
    if train.n_vars > requested_genes:
        variance = np.var(_dense(train), axis=0)
        selected_indices = np.argsort(variance, kind="stable")[-requested_genes:]
        train = train[:, selected_indices].copy()
        test = test[:, selected_indices].copy()

    mapping = {timepoint: index for index, timepoint in enumerate(config.train_times)}
    for timepoint in config.test_times:
        mapping.setdefault(timepoint, len(mapping))
    train.obs["Group"] = train.obs["timepoint_numeric"].map(mapping).astype(int)
    test.obs["Group"] = test.obs["timepoint_numeric"].map(mapping).astype(int)

    train_path = output_dir / "train.h5ad"
    test_path = output_dir / "test.h5ad"
    train.write_h5ad(train_path)
    test.write_h5ad(test_path)

    manifest = {
        **cutoff.manifest.to_dict(),
        "config": asdict(config),
        "input_file": full_adata_path.name,
        "input_sha256": _sha256(full_adata_path),
        "git_commit": _git_commit(),
        "preprocessing": (
            "per-cell normalize_total(target_sum=1e4), then log1p" if normalize else "none"
        ),
        "feature_selection": "top variance genes fitted on training cells only",
        "selected_features": [str(value) for value in train.var_names],
        "train_h5ad_sha256": _sha256(train_path),
        "test_h5ad_sha256": _sha256(test_path),
        "timepoint_mapping": {str(key): value for key, value in mapping.items()},
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedCutoff(
        train_path=train_path,
        test_path=test_path,
        manifest_path=manifest_path,
        n_genes=int(train.n_vars),
    )


def _target_geometry(
    config: CutoffRunConfig, test_times: npt.NDArray[Any]
) -> dict[int, tuple[float, int]]:
    start, end = config.direction_times
    interval = end - start
    if interval <= 0:
        raise ValueError("direction timepoints must be strictly increasing")
    return {
        int(timepoint): (
            (int(timepoint) - end) / interval,
            int((test_times == timepoint).sum()),
        )
        for timepoint in sorted(np.unique(test_times))
    }


def _generate(
    model: Any,
    diffusion: Any,
    anchor: npt.NDArray[Any],
    direction: npt.NDArray[Any],
    targets: dict[int, tuple[float, int]],
    gene_size: int,
    scale: float,
    device: Any,
    rng: np.random.RandomState,
) -> dict[int, npt.NDArray[Any]]:
    import torch

    generated = {}
    for timepoint, (steps, n_cells) in targets.items():
        center = anchor + direction * steps
        latent = np.tile(center, (n_cells, 1))
        if scale > 0:
            latent = latent + scale * rng.randn(n_cells, center.shape[0])
        with torch.no_grad():
            samples = diffusion.ddim_sample_loop(
                model,
                (n_cells, gene_size),
                model_kwargs={"z_mod": torch.tensor(latent, dtype=torch.float32, device=device)},
                noise=None,
            )
        generated[timepoint] = samples.cpu().numpy()
    return generated


def _score_population(
    real: npt.NDArray[Any],
    generated: npt.NDArray[Any],
    bandwidth: float,
    config: CutoffRunConfig,
) -> dict[str, Any]:
    from reuse_gate.metrics.distribution import (
        energy_distance_multivariate,
        mean_expression_correlation,
        mmd_rbf,
    )
    from reuse_gate.metrics.structure import (
        cluster_mass_metrics,
        correlation_frobenius,
        structure_sensitivity_grid,
    )

    return {
        "n_real_cells": int(real.shape[0]),
        "n_generated_cells": int(generated.shape[0]),
        "energy_distance": energy_distance_multivariate(real, generated),
        "mmd_rbf": mmd_rbf(real, generated, bandwidth=bandwidth),
        "mean_expression_correlation": mean_expression_correlation(real, generated),
        "correlation_frobenius_raw": correlation_frobenius(real, generated, normalized=False),
        "correlation_frobenius_normalized": correlation_frobenius(real, generated, normalized=True),
        "cluster_mass": cluster_mass_metrics(
            real,
            generated,
            n_clusters=8,
            rare_below=0.10,
            random_state=13,
        ),
        "cluster_mass_sensitivity": structure_sensitivity_grid(
            real,
            generated,
            cluster_counts=config.structure_cluster_counts,
            rare_thresholds=config.rare_thresholds,
            random_state=13,
        ),
    }


def _score_generated(
    real: npt.NDArray[Any],
    real_times: npt.NDArray[Any],
    generated: dict[int, npt.NDArray[Any]],
    bandwidth: float,
    config: CutoffRunConfig,
) -> dict[str, Any]:
    per_timepoint = {
        str(timepoint): _score_population(
            real[real_times == timepoint],
            generated[timepoint],
            bandwidth,
            config,
        )
        for timepoint in sorted(generated)
    }
    primary_real = np.concatenate(
        [real[real_times == timepoint] for timepoint in config.primary_test_times],
        axis=0,
    )
    primary_generated = np.concatenate(
        [generated[timepoint] for timepoint in config.primary_test_times],
        axis=0,
    )
    return {
        "primary": _score_population(
            primary_real,
            primary_generated,
            bandwidth,
            config,
        ),
        "per_timepoint": per_timepoint,
    }


def _baseline_populations(
    train: npt.NDArray[Any],
    train_times: npt.NDArray[Any],
    test_times: npt.NDArray[Any],
    config: CutoffRunConfig,
    seed: int,
) -> dict[str, dict[int, npt.NDArray[Any]]]:
    """Generate all baseline populations using only the configured train window."""
    from reuse_gate.models.temporal_baselines import (
        conditional_mean_sampler,
        fit_temporal_factor_gaussian,
        temporal_diagonal_gaussian,
    )

    previous_time, latest_time = config.direction_times
    previous = train[train_times == previous_time]
    latest = train[train_times == latest_time]
    targets = _target_geometry(config, test_times)
    candidates = tuple(
        value
        for value in config.factor_component_grid
        if value <= min(previous.shape[0] + latest.shape[0], latest.shape[1])
    )
    if not candidates:
        candidates = (1,)
    factor = fit_temporal_factor_gaussian(
        previous,
        latest,
        component_grid=candidates,
        variance_target=config.factor_variance_target,
    )

    generated: dict[str, dict[int, npt.NDArray[Any]]] = {
        "last_observation_resample": {},
        "pooled_diagonal_gaussian": {},
        "temporal_diagonal_gaussian": {},
        "temporal_factor_gaussian": {},
    }
    for offset, (timepoint, (steps, n_cells)) in enumerate(targets.items()):
        rng = np.random.RandomState(seed + offset)
        chosen = rng.choice(latest.shape[0], n_cells, replace=True)
        generated["last_observation_resample"][timepoint] = latest[chosen]
        generated["pooled_diagonal_gaussian"][timepoint] = conditional_mean_sampler(
            train,
            n_cells,
            np.random.RandomState(seed + 100 + offset),
        )
        generated["temporal_diagonal_gaussian"][timepoint] = temporal_diagonal_gaussian(
            previous,
            latest,
            steps=steps,
            n_samples=n_cells,
            rng=np.random.RandomState(seed + 200 + offset),
        )
        generated["temporal_factor_gaussian"][timepoint] = factor.sample(
            steps=steps,
            n_samples=n_cells,
            rng=np.random.RandomState(seed + 300 + offset),
        )
    return generated


def _build_baselines(
    train: npt.NDArray[Any],
    train_times: npt.NDArray[Any],
    test: npt.NDArray[Any],
    test_times: npt.NDArray[Any],
    bandwidth: float,
    config: CutoffRunConfig,
    seed: int,
) -> dict[str, Any]:
    generated = _baseline_populations(
        train,
        train_times,
        test_times,
        config,
        seed,
    )
    return {
        name: _score_generated(
            test,
            test_times,
            population,
            bandwidth,
            config,
        )
        for name, population in generated.items()
    }


def train_one_seed(
    prepared: PreparedCutoff,
    config: CutoffRunConfig,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Train, select any training-only scale, generate, and score one seed."""
    import anndata as ad
    import torch
    from carnk_latent_extrapolation import _encode, build_model
    from run_tier0_gpu import train_squidiff_gpu

    from reuse_gate.metrics.distribution import median_pairwise_distance

    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not predeclared")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ad = ad.read_h5ad(prepared.train_path)
    test_ad = ad.read_h5ad(prepared.test_path)
    train, test = _dense(train_ad), _dense(test_ad)
    train_times = train_ad.obs["timepoint_numeric"].to_numpy()
    test_times = test_ad.obs["timepoint_numeric"].to_numpy()
    bandwidth = median_pairwise_distance(train)
    model_path = output_dir / "model.pt"

    started = time.time()
    if not model_path.exists():
        train_squidiff_gpu(
            train_path=prepared.train_path,
            output_dir=output_dir,
            gene_size=prepared.n_genes,
            diffusion_steps=config.diffusion_steps,
            lr_anneal_steps=config.steps,
            batch_size=config.batch_size,
            class_cond=config.class_cond,
            num_layers=config.num_layers,
            num_channels=config.num_channels,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion = build_model(prepared.n_genes, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    direction_start, direction_end = config.direction_times
    z_start = _encode(model, train[train_times == direction_start], device)
    z_end = _encode(model, train[train_times == direction_end], device)
    test_direction = z_end.mean(axis=0) - z_start.mean(axis=0)
    test_anchor = z_end.mean(axis=0)
    test_targets = _target_geometry(config, test_times)

    validation: list[dict[str, float]] = []
    test_scales: tuple[float, ...]
    if config.validation_triplet is not None:
        validation_start, validation_end, validation_target = config.validation_triplet
        z_validation_start = _encode(
            model,
            train[train_times == validation_start],
            device,
        )
        z_validation_end = _encode(
            model,
            train[train_times == validation_end],
            device,
        )
        validation_direction = z_validation_end.mean(axis=0) - z_validation_start.mean(axis=0)
        validation_anchor = z_validation_end.mean(axis=0)
        validation_real = train[train_times == validation_target]
        validation_targets = {
            validation_target: (
                (validation_target - validation_end) / (validation_end - validation_start),
                int(validation_real.shape[0]),
            )
        }
        for scale in config.candidate_scales:
            population = _generate(
                model,
                diffusion,
                validation_anchor,
                validation_direction,
                validation_targets,
                prepared.n_genes,
                scale,
                device,
                np.random.RandomState(seed),
            )[validation_target]
            from reuse_gate.metrics.distribution import energy_distance_multivariate

            validation.append(
                {
                    "scale": scale,
                    "energy_distance": energy_distance_multivariate(
                        validation_real,
                        population,
                    ),
                }
            )
        test_scales = (min(validation, key=lambda row: row["energy_distance"])["scale"],)
        selection = "minimum validation energy distance within training window"
    else:
        test_scales = config.fixed_scales
        selection = "predeclared fixed-scale sensitivity; no target-based selection"

    squidiff = {}
    for scale in test_scales:
        generated = _generate(
            model,
            diffusion,
            test_anchor,
            test_direction,
            test_targets,
            prepared.n_genes,
            scale,
            device,
            np.random.RandomState(seed),
        )
        squidiff[f"scale_{scale:g}"] = _score_generated(
            test,
            test_times,
            generated,
            bandwidth,
            config,
        )
        np.savez_compressed(
            output_dir / f"generated_scale_{scale:g}.npz",
            **{f"d{timepoint}": values for timepoint, values in generated.items()},
        )

    result = {
        "cutoff": config.name,
        "seed": seed,
        "training": {
            "steps": config.steps,
            "batch_size": config.batch_size,
            "class_cond": config.class_cond,
            "use_encoder": config.use_encoder,
            "num_layers": config.num_layers,
            "diffusion_steps": config.diffusion_steps,
        },
        "split_manifest_sha256": _sha256(prepared.manifest_path),
        "mmd_bandwidth_fitted_on_training": bandwidth,
        "scale_selection": selection,
        "validation": validation,
        "selected_or_fixed_scales": list(test_scales),
        "squidiff": squidiff,
        "baselines": _build_baselines(
            train,
            train_times,
            test,
            test_times,
            bandwidth,
            config,
            seed,
        ),
        "wall_seconds": round(time.time() - started, 1),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def consolidate_cutoff(output_dir: Path, config: CutoffRunConfig) -> dict[str, Any]:
    """Collect all completed seed metrics without inventing missing results."""
    per_seed = []
    for seed in config.seeds:
        path = output_dir / f"seed_{seed}" / "metrics.json"
        if path.exists():
            per_seed.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {
        "cutoff": config.name,
        "expected_seeds": list(config.seeds),
        "completed_seeds": [entry["seed"] for entry in per_seed],
        "complete": len(per_seed) == len(config.seeds),
        "per_seed": per_seed,
    }
    (output_dir / "cutoff_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def run_cutoff(
    config: CutoffRunConfig,
    full_adata_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    seeds: tuple[int, ...] | None = None,
) -> Path:
    """Prepare a cutoff and optionally execute selected predeclared seeds."""
    prepared = prepare_cutoff_data(full_adata_path, output_dir, config)
    if dry_run:
        return prepared.manifest_path
    selected_seeds = config.seeds if seeds is None else seeds
    for seed in selected_seeds:
        train_one_seed(prepared, config, seed, output_dir / f"seed_{seed}")
    consolidate_cutoff(output_dir, config)
    return output_dir / "cutoff_summary.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "cutoff_studies.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO
        / "artifacts"
        / "squidiff_tier0_gpu"
        / "source_data"
        / "gse190976_combined.h5ad",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_cutoff_config(args.config, args.name)
    output = args.output or REPO / "artifacts" / "cutoff_studies" / config.name
    if args.consolidate_only:
        consolidate_cutoff(output, config)
        return
    run_cutoff(
        config,
        args.data,
        output,
        dry_run=args.dry_run,
        seeds=tuple(args.seed) if args.seed else None,
    )


if __name__ == "__main__":
    main()
