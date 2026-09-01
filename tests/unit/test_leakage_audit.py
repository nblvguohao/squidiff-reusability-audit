"""Unit tests for leakage audit — RED phase."""

import pytest

from reuse_gate.splits.audit import assert_no_overlap


def test_overlap_is_rejected():
    """Overlapping train/test values must raise ValueError."""
    with pytest.raises(ValueError, match="overlap"):
        assert_no_overlap(
            train_values={"S1", "S2"},
            test_values={"S2", "S3"},
            field_name="sample_id",
        )


def test_disjoint_sets_pass():
    """Disjoint train/test values must pass silently."""
    assert_no_overlap(
        train_values={"S1", "S2"},
        test_values={"S3", "S4"},
        field_name="sample_id",
    )


def test_empty_test_set_is_rejected():
    """Empty test set must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        assert_no_overlap(
            train_values={"S1", "S2"},
            test_values=set(),
            field_name="sample_id",
        )
