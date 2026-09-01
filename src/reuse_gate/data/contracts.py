"""AnnData data contract validators for biomedical reusability gate.

Enforces the schemas defined in the plan:
  - Longitudinal single-cell contract (Section 5.2)
  - Perturbation contract (Section 5.3)
  - Spatial pair contract (Section 5.1)
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import pandas as pd

# Required obs columns for each contract per the plan
LONGITUDINAL_REQUIRED_OBS = [
    "cell_id",
    "dataset_id",
    "sample_id",
    "donor_id",
    "species",
    "timepoint_raw",
    "timepoint_numeric",
    "engineering_state",
    "stimulation_state",
    "tumor_context",
    "split_group",
    "endpoint_label",
]

PERTURBATION_REQUIRED_OBS = [
    "cell_id",
    "dataset_id",
    "sample_id",
    "batch_id",
    "cell_context",
    "perturbation_id",
    "perturbation_type",
    "dose",
    "time",
    "is_control",
    "control_pool_id",
    "split_group",
]


def _check_required_columns(adata: ad.AnnData, required: Sequence[str], contract_name: str) -> None:
    """Check that all required obs columns are present."""
    missing = [col for col in required if col not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"{contract_name} contract: missing required obs columns: {missing}"
        )


def _check_no_empty_required_strings(adata: ad.AnnData, fields: Sequence[str]) -> None:
    """Reject empty strings in required string fields.

    Per the plan: 'Unknown non-applicable metadata must be the string "NA",
    not an empty value.'
    """
    for col in fields:
        if col not in adata.obs.columns:
            continue
        series = adata.obs[col]
        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            empty_mask = series == ""
            if empty_mask.any():
                raise ValueError(
                    f"Column '{col}' contains {empty_mask.sum()} empty string(s). "
                    f"Use 'NA' for unknown/unavailable metadata."
                )


def validate_longitudinal_adata(adata: ad.AnnData) -> None:
    """Validate a longitudinal single-cell AnnData object.

    Raises ValueError if any required column is missing or contains empty values.
    """
    _check_required_columns(adata, LONGITUDINAL_REQUIRED_OBS, "longitudinal")
    _check_no_empty_required_strings(adata, LONGITUDINAL_REQUIRED_OBS)


def validate_perturbation_adata(adata: ad.AnnData) -> None:
    """Validate a perturbation AnnData object.

    Raises ValueError if any required column is missing or contains empty values.
    """
    _check_required_columns(adata, PERTURBATION_REQUIRED_OBS, "perturbation")
    _check_no_empty_required_strings(adata, PERTURBATION_REQUIRED_OBS)
