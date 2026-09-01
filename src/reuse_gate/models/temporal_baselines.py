"""Simple temporal baselines for CAR-NK state prediction.

All baselines must fit only on training data. No test data used for fitting,
early stopping, normalization, or feature selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA


def last_observation(
    train: npt.NDArray[Any], test: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    """Predict the training mean for every test cell.

    This represents the simplest baseline: the most recent observed state
    is assumed to persist unchanged.
    """
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True)
    return np.tile(mean, (test.shape[0], 1))


def conditional_mean_sampler(
    train: npt.NDArray[Any],
    n_samples: int,
    rng: np.random.RandomState | None = None,
) -> npt.NDArray[np.float64]:
    """Sample from a Gaussian centered at the training mean with train variance.

    Adds diagonal Gaussian noise scaled by per-feature training variance.
    Fits only on train data.
    """
    train = np.asarray(train, dtype=np.float64)
    if rng is None:
        rng = np.random.RandomState(13)

    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std = np.maximum(std, 1e-8)  # avoid zero std

    samples = np.asarray(rng.randn(n_samples, train.shape[1]) * std + mean, dtype=np.float64)
    return samples


def linear_interpolation(
    train_early: npt.NDArray[Any],
    train_late: npt.NDArray[Any],
    n_samples: int,
    alpha: float = 1.0,
    rng: np.random.RandomState | None = None,
) -> npt.NDArray[np.float64]:
    """Extrapolate via linear interpolation between early and late timepoints.

    direction = mean(late) - mean(early)
    prediction = mean(late) + alpha * direction + small noise

    alpha=0 → stay at late mean
    alpha=1 → extrapolate one step forward
    """
    train_early = np.asarray(train_early, dtype=np.float64)
    train_late = np.asarray(train_late, dtype=np.float64)
    if rng is None:
        rng = np.random.RandomState(13)

    early_mean = train_early.mean(axis=0)
    late_mean = train_late.mean(axis=0)
    direction = late_mean - early_mean
    base = late_mean + alpha * direction

    # Add small noise scaled by late std
    late_std = train_late.std(axis=0, ddof=1)
    late_std = np.maximum(late_std, 1e-8)
    noise = rng.randn(n_samples, train_late.shape[1]) * late_std * 0.1

    return np.asarray(base[np.newaxis, :] + noise, dtype=np.float64)


def temporal_diagonal_gaussian(
    previous: npt.NDArray[Any],
    latest: npt.NDArray[Any],
    *,
    steps: float,
    n_samples: int,
    rng: np.random.RandomState,
) -> npt.NDArray[np.float64]:
    """Extrapolate per-gene means and retain latest-time diagonal variance."""
    previous_array = np.asarray(previous, dtype=np.float64)
    latest_array = np.asarray(latest, dtype=np.float64)
    if previous_array.ndim != 2 or latest_array.ndim != 2:
        raise ValueError("previous and latest must be two-dimensional")
    if previous_array.shape[1] != latest_array.shape[1]:
        raise ValueError("previous and latest must contain the same features")
    if previous_array.shape[0] < 2 or latest_array.shape[0] < 2:
        raise ValueError("each time point must contain at least two cells")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    previous_mean = previous_array.mean(axis=0)
    latest_mean = latest_array.mean(axis=0)
    target_mean = latest_mean + float(steps) * (latest_mean - previous_mean)
    latest_std = np.maximum(latest_array.std(axis=0, ddof=1), 1e-8)
    generated = rng.normal(
        loc=target_mean,
        scale=latest_std,
        size=(n_samples, latest_array.shape[1]),
    )
    return np.asarray(generated, dtype=np.float64)


@dataclass(frozen=True)
class TemporalFactorGaussian:
    """Training-only factor Gaussian with linearly extrapolated factor means."""

    pca: PCA
    previous_factor_mean: npt.NDArray[np.float64]
    latest_factor_mean: npt.NDArray[np.float64]
    factor_std: npt.NDArray[np.float64]
    residual_std: npt.NDArray[np.float64]
    n_components: int
    explained_variance_ratio: float

    def sample(
        self,
        *,
        steps: float,
        n_samples: int,
        rng: np.random.RandomState,
    ) -> npt.NDArray[np.float64]:
        """Sample a future population without using target-time information."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        factor_mean = self.latest_factor_mean + float(steps) * (
            self.latest_factor_mean - self.previous_factor_mean
        )
        factor_samples = rng.normal(
            loc=factor_mean,
            scale=self.factor_std,
            size=(n_samples, self.n_components),
        )
        reconstructed = self.pca.inverse_transform(factor_samples)
        residual = rng.normal(
            loc=0.0,
            scale=self.residual_std,
            size=(n_samples, self.pca.n_features_in_),
        )
        return np.asarray(reconstructed + residual, dtype=np.float64)


def fit_temporal_factor_gaussian(
    previous: npt.NDArray[Any],
    latest: npt.NDArray[Any],
    *,
    component_grid: tuple[int, ...],
    variance_target: float = 0.9,
) -> TemporalFactorGaussian:
    """Fit the smallest candidate factor model reaching a variance target."""
    previous_array = np.asarray(previous, dtype=np.float64)
    latest_array = np.asarray(latest, dtype=np.float64)
    if previous_array.ndim != 2 or latest_array.ndim != 2:
        raise ValueError("previous and latest must be two-dimensional")
    if previous_array.shape[1] != latest_array.shape[1]:
        raise ValueError("previous and latest must contain the same features")
    if not component_grid:
        raise ValueError("component_grid must not be empty")
    if not 0 < variance_target <= 1:
        raise ValueError("variance_target must be in (0, 1]")

    pooled = np.concatenate([previous_array, latest_array], axis=0)
    max_allowed = min(pooled.shape)
    candidates = tuple(sorted({int(value) for value in component_grid}))
    if candidates[0] <= 0 or candidates[-1] > max_allowed:
        raise ValueError(f"component_grid values must be between 1 and {max_allowed}")

    full_pca = PCA(n_components=candidates[-1], svd_solver="full").fit(pooled)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    selected = candidates[-1]
    for candidate in candidates:
        if cumulative[candidate - 1] >= variance_target:
            selected = candidate
            break

    pca = PCA(n_components=selected, svd_solver="full").fit(pooled)
    previous_scores = np.asarray(pca.transform(previous_array), dtype=np.float64)
    latest_scores = np.asarray(pca.transform(latest_array), dtype=np.float64)
    pooled_scores = np.concatenate([previous_scores, latest_scores], axis=0)
    factor_std = np.maximum(pooled_scores.std(axis=0, ddof=1), 1e-8)

    reconstruction = pca.inverse_transform(pooled_scores)
    residual_std = np.maximum((pooled - reconstruction).std(axis=0, ddof=1), 1e-8)
    return TemporalFactorGaussian(
        pca=pca,
        previous_factor_mean=previous_scores.mean(axis=0),
        latest_factor_mean=latest_scores.mean(axis=0),
        factor_std=factor_std,
        residual_std=residual_std,
        n_components=selected,
        explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
    )
