"""Temporal holdout splits for longitudinal single-cell data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TemporalSplit:
    """A time-extrapolation split with no sample overlap."""

    train_groups: list[str]
    test_groups: list[str]
    validation_groups: list[str] = field(default_factory=list)
    train_max_time: float = 0.0
    test_min_time: float = 0.0
    group_col: str = "sample_id"
    time_col: str = "timepoint_numeric"


def build_temporal_holdout(
    obs: pd.DataFrame,
    train_max_time: float,
    test_min_time: float,
    group_col: str = "sample_id",
    time_col: str = "timepoint_numeric",
) -> TemporalSplit:
    """Build a temporal holdout split.

    Groups with max time <= train_max_time go to train.
    Groups with min time >= test_min_time go to test.

    Groups with timepoints between the two boundaries go to validation.
    No group appears in more than one split.
    """
    grouped = obs.groupby(group_col)[time_col]

    train_groups: list[str] = []
    test_groups: list[str] = []
    validation_groups: list[str] = []

    for group_id, times in grouped:
        max_t = times.max()
        min_t = times.min()

        if max_t <= train_max_time:
            train_groups.append(str(group_id))
        elif min_t >= test_min_time:
            test_groups.append(str(group_id))
        else:
            validation_groups.append(str(group_id))

    return TemporalSplit(
        train_groups=train_groups,
        test_groups=test_groups,
        validation_groups=validation_groups,
        train_max_time=train_max_time,
        test_min_time=test_min_time,
        group_col=group_col,
        time_col=time_col,
    )
