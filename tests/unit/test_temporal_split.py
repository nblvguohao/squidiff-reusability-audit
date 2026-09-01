"""Unit tests for temporal split — RED phase."""

import pandas as pd
import pytest

from reuse_gate.splits.temporal import build_temporal_holdout


@pytest.fixture
def longitudinal_obs():
    """Minimal longitudinal obs table."""
    return pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(8)],
            "sample_id": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"],
            "donor_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
            "timepoint_numeric": [0, 0, 7, 7, 14, 14, 21, 21],
            "timepoint_raw": ["pre", "pre", "D7", "D7", "D14", "D14", "D21", "D21"],
        }
    )


def test_temporal_holdout_places_late_samples_only_in_test(longitudinal_obs):
    """Samples at time >= test_min_time go to test only."""
    split = build_temporal_holdout(
        longitudinal_obs,
        train_max_time=14,
        test_min_time=21,
        group_col="sample_id",
    )
    assert split.train_max_time == 14
    assert split.test_min_time == 21
    assert set(split.train_groups).isdisjoint(split.test_groups)


def test_temporal_holdout_retains_exact_group_ids(longitudinal_obs):
    """Splits must record exact group identifiers for auditability."""
    split = build_temporal_holdout(
        longitudinal_obs,
        train_max_time=14,
        test_min_time=21,
        group_col="sample_id",
    )
    assert split.train_groups
    assert split.test_groups
    assert all(isinstance(g, str) for g in split.train_groups)
