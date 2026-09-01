"""Perturbation cold-holdout splits for perturbation prediction evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PerturbationSplit:
    """A cold-perturbation split with no perturbation/context overlap."""

    train_groups: list[str]
    test_groups: list[str]
    validation_groups: list[str] = field(default_factory=list)
    holdout_col: str = "perturbation_id"
    seed: int = 13


def split_perturbations(
    obs: pd.DataFrame,
    holdout_col: str,
    n_test: int,
    n_val: int = 0,
    seed: int = 13,
) -> PerturbationSplit:
    """Split unique perturbation IDs into cold train/test/validation.

    Test perturbations never appear in training.
    """
    unique_ids = sorted(obs[holdout_col].unique().tolist())
    rng = random.Random(seed)
    shuffled = unique_ids[:]
    rng.shuffle(shuffled)

    test_groups = shuffled[:n_test]
    val_start = n_test
    val_end = n_test + n_val
    validation_groups = shuffled[val_start:val_end]
    train_groups = shuffled[val_end:]

    return PerturbationSplit(
        train_groups=train_groups,
        test_groups=test_groups,
        validation_groups=validation_groups,
        holdout_col=holdout_col,
        seed=seed,
    )
