"""Protocol defining the model-runner interface per Section 6 of the plan."""

from __future__ import annotations

from typing import Protocol

from reuse_gate.schemas import RunRequest, RunResult


class ModelRunner(Protocol):
    """Protocol that every upstream runner must satisfy."""

    model_id: str

    def run(self, request: RunRequest) -> RunResult:
        """Execute a run. Must return raw predictions/generated samples."""
        ...
