"""Unit tests for temporal baselines — RED phase."""

import numpy as np

from reuse_gate.models.temporal_baselines import (
    conditional_mean_sampler,
    last_observation,
    linear_interpolation,
)


def test_last_observation_returns_correct_shape():
    """Last observation baseline must return same shape as test data."""
    train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    test = np.array([[0.0, 0.0], [0.0, 0.0]])
    result = last_observation(train, test)
    assert result.shape == test.shape


def test_last_observation_uses_train_mean():
    """Last observation baseline predicts the train mean for all test cells."""
    train = np.array([[0.0, 10.0], [10.0, 20.0]])
    test = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    result = last_observation(train, test)
    expected = np.tile(train.mean(axis=0), (3, 1))
    np.testing.assert_array_almost_equal(result, expected)


def test_conditional_mean_sampler_adds_noise():
    """Conditional mean sampler must add Gaussian noise to train mean."""
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    rng = np.random.RandomState(13)
    result1 = conditional_mean_sampler(train, n_samples=5, rng=rng)
    rng2 = np.random.RandomState(13)
    result2 = conditional_mean_sampler(train, n_samples=5, rng=rng2)
    # Same seed must produce same result
    np.testing.assert_array_equal(result1, result2)
    assert result1.shape == (5, 2)


def test_linear_interpolation_extrapolates():
    """Linear interpolation must extrapolate to future timepoints."""
    # Early: [1, 2, 3], Late: [6, 7, 8] — should extrapolate beyond
    train_early = np.array([[0.0], [1.0], [2.0]])
    train_late = np.array([[3.0], [4.0], [5.0]])
    result = linear_interpolation(
        train_early, train_late, n_samples=3, alpha=0.5
    )
    assert result.shape == (3, 1)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_baselines_do_not_modify_input():
    """Baselines must not mutate input arrays."""
    train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    train_copy = train.copy()
    test = np.array([[0.0, 0.0]], dtype=np.float64)
    test_copy = test.copy()

    last_observation(train, test)
    conditional_mean_sampler(train, n_samples=5)
    linear_interpolation(train, train, n_samples=3)

    np.testing.assert_array_equal(train, train_copy)
    np.testing.assert_array_equal(test, test_copy)
