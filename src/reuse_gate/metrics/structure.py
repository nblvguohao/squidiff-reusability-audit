"""Structure-aware metrics for generated single-cell populations."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans


def _validated_populations(
    real: npt.NDArray[Any],
    generated: npt.NDArray[Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    real_array = np.asarray(real, dtype=np.float64)
    generated_array = np.asarray(generated, dtype=np.float64)
    if real_array.ndim != 2 or generated_array.ndim != 2:
        raise ValueError("real and generated populations must be two-dimensional")
    if real_array.shape[1] != generated_array.shape[1]:
        raise ValueError("real and generated populations must contain the same features")
    if real_array.shape[0] < 2 or generated_array.shape[0] < 2:
        raise ValueError("each population must contain at least two cells")
    return real_array, generated_array


def correlation_frobenius(
    real: npt.NDArray[Any],
    generated: npt.NDArray[Any],
    *,
    normalized: bool = True,
) -> float:
    """Distance between gene-correlation matrices.

    Constant genes in the real comparison population are removed because their
    Pearson correlation is undefined. The normalized form divides the
    Frobenius norm by the number of retained genes, making values comparable
    across feature sets of different size.
    """
    real_array, generated_array = _validated_populations(real, generated)
    keep = real_array.std(axis=0) > 0
    retained = int(keep.sum())
    if retained < 2:
        raise ValueError("fewer than two non-constant real genes remain")
    real_corr = np.corrcoef(real_array[:, keep], rowvar=False)
    generated_corr = np.corrcoef(generated_array[:, keep], rowvar=False)
    raw = float(np.linalg.norm(real_corr - generated_corr, ord="fro"))
    return raw / retained if normalized else raw


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "values": [float(value) for value in array],
    }


def same_distribution_structure_reference(
    real: npt.NDArray[Any],
    *,
    n_splits: int,
    rng: np.random.RandomState,
) -> dict[str, Any]:
    """Correlation-distance reference from random disjoint halves of real data."""
    real_array = np.asarray(real, dtype=np.float64)
    if real_array.ndim != 2 or real_array.shape[0] < 4:
        raise ValueError("real must be a two-dimensional population with at least four cells")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")

    raw_values: list[float] = []
    normalized_values: list[float] = []
    half = real_array.shape[0] // 2
    for _ in range(n_splits):
        order = rng.permutation(real_array.shape[0])
        left = real_array[order[:half]]
        right = real_array[order[half : 2 * half]]
        raw_values.append(correlation_frobenius(left, right, normalized=False))
        normalized_values.append(correlation_frobenius(left, right, normalized=True))
    return {
        "n_splits": n_splits,
        "raw": _summary(raw_values),
        "normalized": _summary(normalized_values),
    }


def cluster_mass_metrics(
    real: npt.NDArray[Any],
    generated: npt.NDArray[Any],
    *,
    n_clusters: int,
    rare_below: float,
    random_state: int,
) -> dict[str, float]:
    """Measure cluster coverage and the fidelity of generated cluster masses."""
    real_array, generated_array = _validated_populations(real, generated)
    if not 2 <= n_clusters <= real_array.shape[0]:
        raise ValueError("n_clusters must be between 2 and the number of real cells")
    if not 0 < rare_below < 1:
        raise ValueError("rare_below must be in (0, 1)")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    ).fit(real_array)
    real_mass = np.bincount(kmeans.labels_, minlength=n_clusters).astype(np.float64)
    real_mass /= real_mass.sum()
    generated_labels = kmeans.predict(generated_array)
    generated_mass = np.bincount(generated_labels, minlength=n_clusters).astype(np.float64)
    generated_mass /= generated_mass.sum()

    rare = real_mass < rare_below
    if rare.any():
        overlap = np.minimum(real_mass[rare], generated_mass[rare]).sum()
        real_rare_mass = real_mass[rare].sum()
        generated_rare_mass = generated_mass[rare].sum()
        rare_coverage = float(np.mean(generated_mass[rare] > 0))
        rare_recall = float(overlap / real_rare_mass)
        rare_precision = (
            float(overlap / generated_rare_mass) if generated_rare_mass > 0 else 0.0
        )
    else:
        rare_coverage = 1.0
        rare_recall = 1.0
        rare_precision = 1.0

    return {
        "rare_coverage": rare_coverage,
        "rare_mass_recall": rare_recall,
        "rare_mass_precision": rare_precision,
        "cluster_mass_mae": float(np.abs(real_mass - generated_mass).mean()),
        "cluster_mass_jsd": float(jensenshannon(real_mass, generated_mass) ** 2),
    }


def structure_sensitivity_grid(
    real: npt.NDArray[Any],
    generated: npt.NDArray[Any],
    *,
    cluster_counts: tuple[int, ...],
    rare_thresholds: tuple[float, ...],
    random_state: int,
) -> list[dict[str, float | int]]:
    """Evaluate cluster-mass conclusions across pre-specified settings."""
    rows: list[dict[str, float | int]] = []
    for n_clusters in cluster_counts:
        for rare_below in rare_thresholds:
            metrics = cluster_mass_metrics(
                real,
                generated,
                n_clusters=n_clusters,
                rare_below=rare_below,
                random_state=random_state,
            )
            rows.append(
                {
                    "n_clusters": int(n_clusters),
                    "rare_below": float(rare_below),
                    **metrics,
                }
            )
    return rows
