"""CMonge hard gate evaluation (Section 2.3 of the plan)."""

from __future__ import annotations

from typing import Any

from reuse_gate.gates.base import evaluate_gates
from reuse_gate.schemas import GateResult

CMONGE_GATES = [
    "CM-G1", "CM-G2", "CM-G3", "CM-G4", "CM-G5",
    "CM-G6", "CM-G7", "CM-G8", "CM-G9", "CM-G10",
]


def evaluate_cmonge_gate(evidence: dict[str, Any]) -> GateResult:
    """Evaluate all 10 CMonge hard gates.

    Note: STUB — full thresholds implemented when CMonge evaluation begins.
    """
    thresholds: dict[str, Any] = {
        f"CM-G{i}": lambda ev, i=i: (bool(ev.get(f"cm_g{i}_pass", False)), None)
        for i in range(1, 11)
    }
    return evaluate_gates(
        candidate="cmonge",
        gate_names=CMONGE_GATES,
        evidence=evidence,
        thresholds=thresholds,
    )
