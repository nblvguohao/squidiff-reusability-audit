"""Distribution-level metrics for comparing generated and real sample populations."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist


def energy_distance_multivariate(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
    """Compute the multivariate energy distance between two sample sets.

    E(x, y) = 2 * mean(||x_i - y_j||) - mean(||x_i - x_j||) - mean(||y_i - y_j||)

    Returns 0 when x and y are identically distributed (in expectation).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Pairwise distances
    d_xy = cdist(x, y, metric="euclidean")
    d_xx = cdist(x, x, metric="euclidean")
    d_yy = cdist(y, y, metric="euclidean")

    a = np.mean(d_xy)  # cross-distance
    b = np.mean(d_xx)  # within-x
    c = np.mean(d_yy)  # within-y

    return float(max(0.0, float(2.0 * a - b - c)))


def median_pairwise_distance(
    samples: npt.NDArray[Any], max_points: int = 2000, seed: int = 13
) -> float:
    """Median pairwise Euclidean distance, the usual RBF bandwidth heuristic.

    Intended to be evaluated once on training data and then held fixed, so that
    the kernel does not adapt to whatever is being scored.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.shape[0] > max_points:
        idx = np.random.RandomState(seed).choice(samples.shape[0], max_points, replace=False)
        samples = samples[idx]
    distances = cdist(samples, samples, metric="euclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    return float(np.median(upper))


def mmd_rbf(x: npt.NDArray[Any], y: npt.NDArray[Any], *, bandwidth: float) -> float:
    """Squared maximum mean discrepancy under an RBF kernel.

    `bandwidth` is required rather than defaulted. Deriving it from the two
    samples being compared would let the evaluation data influence the metric,
    so the caller must fix it in advance, normally with
    `median_pairwise_distance` on training data.
    """
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gamma = 1.0 / (2.0 * bandwidth**2)

    k_xx = np.exp(-gamma * cdist(x, x, "sqeuclidean"))
    k_yy = np.exp(-gamma * cdist(y, y, "sqeuclidean"))
    k_xy = np.exp(-gamma * cdist(x, y, "sqeuclidean"))

    return float(max(0.0, k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()))


def mean_expression_correlation(
    real: npt.NDArray[Any], generated: npt.NDArray[Any]
) -> float:
    """Pearson correlation between the per-feature means of two populations.

    Unlike energy distance and MMD this is invariant to any affine rescaling of
    the generated values, so it separates "are the right genes high and low"
    from "is the output on the right scale". The two questions come apart
    sharply for a generator whose output scale has drifted.
    """
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if real.shape[1] != generated.shape[1]:
        raise ValueError(
            f"both populations need the same features, got {real.shape[1]} and {generated.shape[1]}"
        )

    real_means = real.mean(axis=0)
    generated_means = generated.mean(axis=0)
    if np.std(real_means) == 0 or np.std(generated_means) == 0:
        return 0.0
    return float(np.corrcoef(real_means, generated_means)[0, 1])
