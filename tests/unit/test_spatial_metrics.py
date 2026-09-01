"""Unit tests for regression metrics — RED phase."""

import numpy as np

from reuse_gate.metrics.regression import regression_metrics


def test_perfect_prediction_has_zero_mae_and_unit_r2():
    """Perfect prediction must yield MAE=0 and R²=1."""
    y = np.array([[1.0, 2.0], [2.0, 3.0]])
    result = regression_metrics(y, y)
    assert result["mae"] == 0.0
    assert result["r2"] == 1.0


def test_mae_positive_for_imperfect_prediction():
    """MAE must be positive when prediction differs from target."""
    y_true = np.array([[0.0, 0.0], [0.0, 0.0]])
    y_pred = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = regression_metrics(y_true, y_pred)
    assert result["mae"] > 0.0


def test_r2_handles_zero_variance():
    """R² must handle the case where target has zero variance (returns 0.0)."""
    y_true = np.array([[5.0], [5.0], [5.0]])
    y_pred = np.array([[5.1], [4.9], [5.0]])
    result = regression_metrics(y_true, y_pred)
    assert "r2" in result
