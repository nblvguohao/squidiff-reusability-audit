"""Unit tests for perturbation split — RED phase."""

import pandas as pd
import pytest

from reuse_gate.splits.perturbation import split_perturbations


@pytest.fixture
def perturbation_obs():
    """Minimal perturbation obs table."""
    return pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(8)],
            "sample_id": ["S1"] * 8,
            "batch_id": ["B1", "B1", "B1", "B1", "B2", "B2", "B2", "B2"],
            "perturbation_id": ["CRISPR_A", "CRISPR_A", "CRISPR_B", "CRISPR_B",
                                "CRISPR_C", "CRISPR_C", "CRISPR_D", "CRISPR_D"],
            "control_pool_id": ["CP1", "CP1", "CP2", "CP2", "CP3", "CP3", "CP4", "CP4"],
            "is_control": [False] * 8,
        }
    )


def test_perturbation_split_is_cold(perturbation_obs):
    """Perturbation IDs in test must not appear in train."""
    split = split_perturbations(
        perturbation_obs,
        holdout_col="perturbation_id",
        n_test=2,
        seed=13,
    )
    train_perturbations = set(perturbation_obs.loc[
        perturbation_obs["perturbation_id"].isin(split.train_groups),
        "perturbation_id",
    ])
    test_perturbations = set(perturbation_obs.loc[
        perturbation_obs["perturbation_id"].isin(split.test_groups),
        "perturbation_id",
    ])
    assert train_perturbations.isdisjoint(test_perturbations)


def test_perturbation_split_respects_seed(perturbation_obs):
    """Same seed must produce same split."""
    s1 = split_perturbations(perturbation_obs, "perturbation_id", n_test=2, seed=13)
    s2 = split_perturbations(perturbation_obs, "perturbation_id", n_test=2, seed=13)
    assert s1.train_groups == s2.train_groups
    assert s1.test_groups == s2.test_groups
