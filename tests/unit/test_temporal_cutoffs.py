"""Tests for declarative, leakage-safe temporal cutoff studies."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from reuse_gate.splits.temporal_cutoffs import (
    CutoffSpec,
    build_temporal_cutoff,
    early_cutoff_spec,
    late_cutoff_spec,
)


def _toy_adata() -> ad.AnnData:
    times = [0, 0, 7, 7, 14, 14, 21, 21, 28, 28]
    obs = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(len(times))],
            "timepoint_numeric": times,
        },
        index=[f"cell_{i}" for i in range(len(times))],
    )
    return ad.AnnData(X=np.arange(30, dtype=np.float32).reshape(10, 3), obs=obs)


def test_late_cutoff_uses_d28_only_for_test() -> None:
    split = build_temporal_cutoff(_toy_adata(), late_cutoff_spec())

    assert set(split.train.obs["timepoint_numeric"]) == {0, 7, 14, 21}
    assert set(split.test.obs["timepoint_numeric"]) == {28}
    assert set(split.train.obs["sample_id"]).isdisjoint(split.test.obs["sample_id"])
    assert split.manifest.direction_times == (14, 21)
    assert split.manifest.validation_triplet == (7, 14, 21)


def test_early_cutoff_predeclares_two_noise_scales() -> None:
    split = build_temporal_cutoff(_toy_adata(), early_cutoff_spec())

    assert set(split.train.obs["timepoint_numeric"]) == {0, 7}
    assert set(split.test.obs["timepoint_numeric"]) == {14, 21, 28}
    assert split.manifest.validation_triplet is None
    assert split.manifest.fixed_scale_sensitivity == (0.0, 0.03)


def test_manifest_records_sample_and_cell_counts() -> None:
    split = build_temporal_cutoff(_toy_adata(), late_cutoff_spec())

    assert split.manifest.train_cells == 8
    assert split.manifest.test_cells == 2
    assert split.manifest.train_samples == 8
    assert split.manifest.test_samples == 2
    assert split.manifest.train_sample_ids == tuple(f"S{i}" for i in range(8))
    assert split.manifest.test_sample_ids == ("S8", "S9")


def test_cutoff_rejects_overlapping_train_and_test_times() -> None:
    spec = CutoffSpec(
        name="invalid",
        train_times=(0, 7, 14),
        test_times=(14, 21),
        direction_times=(7, 14),
        validation_triplet=None,
    )

    try:
        build_temporal_cutoff(_toy_adata(), spec)
    except ValueError as exc:
        assert "timepoint" in str(exc).lower()
    else:
        raise AssertionError("overlapping train/test timepoints must be rejected")
