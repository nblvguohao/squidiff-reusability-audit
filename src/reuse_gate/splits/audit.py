"""Leakage audit: verify no overlap between train and test group identifiers."""

from __future__ import annotations


def assert_no_overlap(
    train_values: set[str],
    test_values: set[str],
    field_name: str,
) -> None:
    """Assert train and test group sets are disjoint and non-empty.

    Raises:
        ValueError: if test is empty or if overlap exists.
    """
    if not test_values:
        raise ValueError(f"Test set is empty for field '{field_name}'")
    if not train_values:
        raise ValueError(f"Train set is empty for field '{field_name}'")

    overlap = train_values & test_values
    if overlap:
        raise ValueError(
            f"Leakage detected: {len(overlap)} group(s) overlap in '{field_name}': "
            f"{sorted(overlap)}"
        )
