"""Pydantic schemas for the biomedical reusability gate.

All models use Pydantic v2 with strict validation. No default values that could
mask configuration errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

CandidateName = Literal["nichetrans", "squidiff", "cmonge"]


class RunRequest(BaseModel):
    """Immutable request to run a candidate model on a specific data split."""

    run_id: str
    candidate: CandidateName
    model_id: str
    train_path: Path
    validation_path: Path
    test_path: Path
    config_path: Path
    output_dir: Path
    seed: int

    @model_validator(mode="after")
    def _validate_paths(self) -> RunRequest:
        if self.train_path == self.test_path:
            raise ValueError(
                f"train_path and test_path must differ; both are {self.train_path}"
            )
        if self.run_id == "":
            raise ValueError("run_id must not be empty")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0; got {self.seed}")
        return self


class RunResult(BaseModel):
    """Result of a model run, preserving stdout/stderr and metadata."""

    run_id: str
    status: Literal["success", "failed"]
    predictions_path: Path | None = None
    metrics_path: Path | None = None
    checkpoint_path: Path | None = None
    stdout_path: Path
    stderr_path: Path
    error_type: str | None = None
    error_message: str | None = None
    git_commit: str
    container_digest: str
    peak_cuda_allocated_mb: float | None = None
    train_seconds: float | None = None


class GateResult(BaseModel):
    """Result of evaluating a candidate's hard gates."""

    candidate: CandidateName
    hard_gate_pass: bool
    failed_gates: list[str] = []
    evidence_paths: list[Path] = []


class CandidateResults(BaseModel):
    """Aggregate results for one candidate."""

    hard_gate_pass: bool
    failed_gates: list[str] = []
    evidence_paths: list[Path] = []


class SelectionDecision(BaseModel):
    """Immutable machine-readable gate decision.

    Once written to reports/selection_decision.json, this file must not be
    overwritten. A changed decision requires an amendment file.
    """

    evaluation_date: str
    selected_candidate: Literal["nichetrans", "squidiff", "cmonge", "NO_GO"]
    decision_rule: str
    candidate_results: dict[str, CandidateResults]
    repository_commit: str

    @model_validator(mode="after")
    def _validate_candidate_results_keys(self) -> SelectionDecision:
        allowed = {"nichetrans", "squidiff", "cmonge"}
        actual = set(self.candidate_results.keys())
        if not actual.issubset(allowed):
            extra = actual - allowed
            raise ValueError(f"candidate_results keys must be subset of {allowed}; got extra: {extra}")
        return self
