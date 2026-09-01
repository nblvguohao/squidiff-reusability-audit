"""Unit tests for reuse_gate schemas — RED phase: these must fail before implementation exists."""

import pytest
from pydantic import ValidationError

from reuse_gate.schemas import RunRequest, SelectionDecision


def test_run_request_rejects_identical_train_and_test_paths(tmp_path):
    """Train and test paths must be different to prevent data leakage."""
    same = tmp_path / "same.h5ad"
    with pytest.raises(ValidationError, match="train_path and test_path"):
        RunRequest(
            run_id="r1",
            candidate="squidiff",
            model_id="squidiff",
            train_path=same,
            validation_path=tmp_path / "val.h5ad",
            test_path=same,
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path / "out",
            seed=13,
        )


def test_selection_decision_rejects_unknown_candidate():
    """SelectionDecision must reject candidate values not in the allowed set."""
    with pytest.raises(ValidationError):
        SelectionDecision(
            evaluation_date="2026-07-22",
            selected_candidate="unknown",
            decision_rule="invalid",
            candidate_results={},
            repository_commit="abc",
        )


def test_run_request_seed_must_be_positive():
    """Seed must be a positive integer."""
    with pytest.raises(ValidationError):
        RunRequest(
            run_id="r1",
            candidate="nichetrans",
            model_id="test",
            train_path="/tmp/train.h5ad",
            validation_path="/tmp/val.h5ad",
            test_path="/tmp/test.h5ad",
            config_path="/tmp/config.yaml",
            output_dir="/tmp/out",
            seed=-1,
        )


def test_run_request_rejects_empty_run_id():
    """Run ID must not be empty."""
    with pytest.raises(ValidationError):
        RunRequest(
            run_id="",
            candidate="nichetrans",
            model_id="test",
            train_path="/tmp/train.h5ad",
            validation_path="/tmp/val.h5ad",
            test_path="/tmp/test.h5ad",
            config_path="/tmp/config.yaml",
            output_dir="/tmp/out",
            seed=13,
        )
