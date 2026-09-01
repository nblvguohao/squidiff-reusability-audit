"""Tests for task-aligned marginal and structure-capable temporal baselines."""

from __future__ import annotations

import numpy as np

from reuse_gate.models.temporal_baselines import (
    fit_temporal_factor_gaussian,
    temporal_diagonal_gaussian,
)


def test_temporal_diagonal_extrapolates_feature_means() -> None:
    previous = np.tile(np.array([[0.0, 2.0]]), (100, 1))
    latest = np.tile(np.array([[1.0, 4.0]]), (100, 1))

    generated = temporal_diagonal_gaussian(
        previous,
        latest,
        steps=1.0,
        n_samples=4000,
        rng=np.random.RandomState(1),
    )

    np.testing.assert_allclose(generated.mean(axis=0), [2.0, 6.0], atol=0.03)
    assert generated.shape == (4000, 2)


def test_temporal_diagonal_uses_latest_training_variance() -> None:
    rng = np.random.RandomState(2)
    previous = rng.normal(0, 1, size=(3000, 2))
    latest = rng.normal([1, 2], [0.5, 2.0], size=(3000, 2))

    generated = temporal_diagonal_gaussian(
        previous,
        latest,
        steps=0.0,
        n_samples=5000,
        rng=np.random.RandomState(3),
    )

    np.testing.assert_allclose(generated.std(axis=0), latest.std(axis=0, ddof=1), rtol=0.08)


def test_factor_gaussian_reproduces_training_correlation() -> None:
    rng = np.random.RandomState(4)
    previous_factor = rng.normal(size=(2500, 1))
    latest_factor = rng.normal(loc=1.0, size=(2500, 1))
    previous = np.c_[
        previous_factor[:, 0],
        previous_factor[:, 0] + rng.normal(0, 0.05, 2500),
    ]
    latest = np.c_[
        latest_factor[:, 0],
        latest_factor[:, 0] + rng.normal(0, 0.05, 2500),
    ]

    model = fit_temporal_factor_gaussian(
        previous,
        latest,
        component_grid=(1,),
        variance_target=0.9,
    )
    generated = model.sample(
        steps=1.0,
        n_samples=3000,
        rng=np.random.RandomState(5),
    )

    assert generated.shape == (3000, 2)
    assert np.corrcoef(generated, rowvar=False)[0, 1] > 0.9
    assert model.n_components == 1


def test_factor_gaussian_selects_smallest_grid_value_reaching_target() -> None:
    rng = np.random.RandomState(6)
    latent = rng.normal(size=(1000, 2))
    loading = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    previous = latent @ loading + rng.normal(0, 0.01, size=(1000, 3))
    latest = previous + np.array([1.0, 1.0, 2.0])

    model = fit_temporal_factor_gaussian(
        previous,
        latest,
        component_grid=(1, 2, 3),
        variance_target=0.9,
    )

    assert model.n_components == 2


def test_new_baselines_do_not_modify_training_arrays() -> None:
    previous = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    latest = previous + 1.0
    previous_copy = previous.copy()
    latest_copy = latest.copy()

    temporal_diagonal_gaussian(
        previous,
        latest,
        steps=1.0,
        n_samples=3,
        rng=np.random.RandomState(7),
    )
    fit_temporal_factor_gaussian(
        previous,
        latest,
        component_grid=(1,),
        variance_target=0.8,
    )

    np.testing.assert_array_equal(previous, previous_copy)
    np.testing.assert_array_equal(latest, latest_copy)
