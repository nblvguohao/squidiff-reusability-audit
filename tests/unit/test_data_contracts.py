"""Unit tests for AnnData data contracts — RED phase."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from reuse_gate.data.contracts import validate_longitudinal_adata


def test_longitudinal_contract_requires_sample_id():
    """validate_longitudinal_adata must reject AnnData without sample_id."""
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(
            {"cell_id": ["c1", "c2"]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["GZMB", "PRF1"]),
    )
    with pytest.raises(ValueError, match="sample_id"):
        validate_longitudinal_adata(adata)


def test_longitudinal_contract_requires_required_fields():
    """All required obs columns from the contract must be present."""
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(
            {
                "cell_id": ["c1", "c2"],
                "sample_id": ["S1", "S1"],
            },
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["GZMB", "PRF1"]),
    )
    with pytest.raises(ValueError):
        validate_longitudinal_adata(adata)


def test_longitudinal_contract_accepts_complete_metadata():
    """Complete metadata with all required fields must pass."""
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(
            {
                "cell_id": ["c1", "c2"],
                "dataset_id": ["D1", "D1"],
                "sample_id": ["S1", "S1"],
                "donor_id": ["P1", "P1"],
                "species": ["human", "human"],
                "timepoint_raw": ["D0", "D7"],
                "timepoint_numeric": [0, 7],
                "engineering_state": ["WT", "WT"],
                "stimulation_state": ["none", "none"],
                "tumor_context": ["tumor", "tumor"],
                "split_group": ["train", "train"],
                "endpoint_label": ["naive", "activated"],
            },
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["GZMB", "PRF1"]),
    )
    # Must not raise
    validate_longitudinal_adata(adata)


def test_na_values_are_rejected_in_required_fields():
    """Required string fields must not contain empty strings or NA."""
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(
            {
                "cell_id": ["c1", "c2"],
                "dataset_id": ["", "D1"],
                "sample_id": ["S1", "S1"],
                "donor_id": ["P1", "P1"],
                "species": ["human", "human"],
                "timepoint_raw": ["D0", "D7"],
                "timepoint_numeric": [0, 7],
                "engineering_state": ["WT", "WT"],
                "stimulation_state": ["none", "none"],
                "tumor_context": ["tumor", "tumor"],
                "split_group": ["train", "train"],
                "endpoint_label": ["naive", "activated"],
            },
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["GZMB", "PRF1"]),
    )
    with pytest.raises(ValueError, match="empty"):
        validate_longitudinal_adata(adata)
