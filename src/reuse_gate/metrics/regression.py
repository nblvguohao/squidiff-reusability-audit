"""Point-wise regression metrics for spatial and single-cell prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def regression_metrics(
    y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]
) -> dict[str, float]:
    """Compute MAE and R² (feature-wise then averaged).

    Args:
        y_true: Ground truth (n_samples, n_features).
        y_pred: Predictions of the same shape.

    Returns:
        Dict with 'mae' and 'r2' keys.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R²: 1 - SS_res / SS_tot, computed feature-wise then averaged
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)

    # Handle zero-variance features: if ss_tot == 0, R² is undefined → set to 0
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_per_feature = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    r2 = float(np.mean(r2_per_feature))

    return {"mae": mae, "r2": r2}
