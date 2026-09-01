"""Unit tests for distribution metrics — RED phase."""

import numpy as np
import pytest

from reuse_gate.metrics.distribution import (
    energy_distance_multivariate,
    mean_expression_correlation,
    mmd_rbf,
)


def test_identical_distributions_have_zero_energy_distance():
    """Identical samples must yield zero energy distance."""
    x = np.array([[0.0], [1.0], [2.0]])
    assert energy_distance_multivariate(x, x) == 0.0


def test_different_distributions_have_positive_energy_distance():
    """Different samples must yield positive energy distance."""
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[5.0], [6.0], [7.0]])
    assert energy_distance_multivariate(x, y) > 0.0


def test_energy_distance_symmetric():
    """Energy distance must be symmetric."""
    x = np.array([[0.0, 1.0], [1.0, 2.0]])
    y = np.array([[3.0, 4.0], [5.0, 6.0]])
    assert energy_distance_multivariate(x, y) == pytest.approx(
        energy_distance_multivariate(y, x)
    )


# ── MMD with a fixed kernel bandwidth ─────────────────────────────────────────


def test_mmd_of_identical_samples_is_zero():
    """A sample against itself must give zero within numerical tolerance."""
    x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert mmd_rbf(x, x, bandwidth=1.0) == pytest.approx(0.0, abs=1e-12)


def test_mmd_is_positive_for_separated_samples():
    """Well-separated samples must give a positive value."""
    x = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0]])
    y = np.array([[5.0, 5.0], [5.1, 5.1], [5.2, 5.0]])
    assert mmd_rbf(x, y, bandwidth=1.0) > 0.0


def test_mmd_is_symmetric():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[2.0, 2.0], [3.0, 3.0]])
    assert mmd_rbf(x, y, bandwidth=1.0) == pytest.approx(mmd_rbf(y, x, bandwidth=1.0))


def test_mmd_requires_an_explicit_bandwidth():
    """The bandwidth is a policy choice and must be fixed by the caller.

    Deriving it from the samples being compared would let test data influence
    the metric, so there is deliberately no default.
    """
    x = np.array([[0.0], [1.0]])
    with pytest.raises(TypeError):
        mmd_rbf(x, x)  # type: ignore[call-arg]


def test_mmd_rejects_non_positive_bandwidth():
    x = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="bandwidth"):
        mmd_rbf(x, x, bandwidth=0.0)


# ── Scale-invariant agreement of per-gene means ───────────────────────────────


def test_mean_expression_correlation_is_one_for_identical_samples():
    x = np.array([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0]])
    assert mean_expression_correlation(x, x) == pytest.approx(1.0)


def test_mean_expression_correlation_ignores_affine_rescaling():
    """The point of this metric: it must not move when only scale changes.

    Every other metric here is scale-sensitive, which is why a rescaled
    generator scores badly. This one answers the separate question of whether
    the right genes are high and low.
    """
    x = np.array([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0]])
    assert mean_expression_correlation(x, 3.0 * x + 7.0) == pytest.approx(1.0)


def test_mean_expression_correlation_detects_reordered_genes():
    """Correlation must drop when the per-gene ordering is wrong."""
    x = np.array([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0]])
    reversed_genes = x[:, ::-1]
    assert mean_expression_correlation(x, reversed_genes) == pytest.approx(-1.0)


def test_mean_expression_correlation_rejects_shape_mismatch():
    x = np.zeros((4, 3))
    y = np.zeros((4, 5))
    with pytest.raises(ValueError, match="features"):
        mean_expression_correlation(x, y)
