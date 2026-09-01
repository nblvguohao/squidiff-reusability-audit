"""Squidiff hard gate evaluation (Section 2.2 of the plan)."""

from __future__ import annotations

from typing import Any

from reuse_gate.gates.base import evaluate_gates
from reuse_gate.schemas import GateResult

SQUIDIFF_GATES = [
    "SQ-G1", "SQ-G2", "SQ-G3", "SQ-G4", "SQ-G5",
    "SQ-G6", "SQ-G7", "SQ-G8", "SQ-G9", "SQ-G10",
]


def evaluate_squidiff_gate(evidence: dict[str, Any]) -> GateResult:
    """Evaluate all 10 Squidiff hard gates.

    Note: STUB — full thresholds implemented when Squidiff evaluation begins.
    """
    thresholds: dict[str, Any] = {
        f"SQ-G{i}": lambda ev, i=i: (bool(ev.get(f"sq_g{i}_pass", False)), None)
        for i in range(1, 11)
    }
    return evaluate_gates(
        candidate="squidiff",
        gate_names=SQUIDIFF_GATES,
        evidence=evidence,
        thresholds=thresholds,
    )
