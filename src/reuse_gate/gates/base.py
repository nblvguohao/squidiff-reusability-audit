"""Base types for candidate gate evaluation."""

from __future__ import annotations

from typing import Any

from reuse_gate.schemas import CandidateName, GateResult


def evaluate_gates(
    candidate: CandidateName,
    gate_names: list[str],
    evidence: dict[str, Any],
    thresholds: dict[str, Any],
) -> GateResult:
    """Generic gate evaluator. Checks each gate against a threshold.

    Each gate is defined by a name and a threshold function that receives the
    evidence dict and returns (passed: bool, reason: str | None).
    """
    failed_gates: list[str] = []
    for gate in gate_names:
        checker = thresholds.get(gate)
        if checker is None:
            failed_gates.append(gate)
            continue
        passed, _reason = checker(evidence)
        if not passed:
            failed_gates.append(gate)

    return GateResult(
        candidate=candidate,
        hard_gate_pass=len(failed_gates) == 0,
        failed_gates=failed_gates,
    )
