"""NicheTrans hard gate evaluation (Section 2.1 of the plan)."""

from __future__ import annotations

from typing import Any

from reuse_gate.gates.base import evaluate_gates
from reuse_gate.schemas import GateResult

NICHE_TRANS_GATES = [
    "NT-G1",  # Official repo and archived version can be pinned and built
    "NT-G2",  # One official example completes end-to-end
    "NT-G3",  # At least 2 independent biological units
    "NT-G4",  # Each unit has >= 500 locations after QC
    "NT-G5",  # >= 30 target features pass filters in every unit
    "NT-G6",  # Registration/matching >= 90% of source locations
    "NT-G7",  # Patient/slice-held-out split with no biological-unit overlap
    "NT-G8",  # At least 3 simple baselines can run on the same split
    "NT-G9",  # One-epoch or reduced-data smoke run completes in GPU memory
    "NT-G10",  # Full data available without manual-access dependency
]


def _check_nt_g1(ev: dict[str, Any]) -> tuple[bool, str | None]:
    ok = bool(ev.get("official_repo_pinned"))
    return ok, None if ok else "Official repository not pinned"


def _check_nt_g2(ev: dict[str, Any]) -> tuple[bool, str | None]:
    ok = bool(ev.get("official_smoke_pass"))
    return ok, None if ok else "Official example did not complete"


def _check_nt_g3(ev: dict[str, Any]) -> tuple[bool, str | None]:
    n = int(ev.get("biological_units", 0))
    ok = n >= 2
    return ok, None if ok else f"Only {n} biological unit(s); need >= 2"


def _check_nt_g4(ev: dict[str, Any]) -> tuple[bool, str | None]:
    n = int(ev.get("min_locations_per_unit", 0))
    ok = n >= 500
    return ok, None if ok else f"Min locations per unit is {n}; need >= 500"


def _check_nt_g5(ev: dict[str, Any]) -> tuple[bool, str | None]:
    n = int(ev.get("target_features", 0))
    ok = n >= 30
    return ok, None if ok else f"Only {n} target features; need >= 30"


def _check_nt_g6(ev: dict[str, Any]) -> tuple[bool, str | None]:
    rate = float(ev.get("match_rate", 0))
    ok = rate >= 0.90
    return ok, None if ok else f"Match rate {rate:.2f} < 0.90"


def _check_nt_g7(ev: dict[str, Any]) -> tuple[bool, str | None]:
    ok = bool(ev.get("group_split_pass"))
    return ok, None if ok else "Group-held-out split failed or has overlap"


def _check_nt_g8(ev: dict[str, Any]) -> tuple[bool, str | None]:
    n = int(ev.get("baseline_count", 0))
    ok = n >= 3
    return ok, None if ok else f"Only {n} baselines; need >= 3"


def _check_nt_g9(ev: dict[str, Any]) -> tuple[bool, str | None]:
    ok = bool(ev.get("gpu_smoke_pass"))
    return ok, None if ok else "GPU smoke run did not complete or OOMed"


def _check_nt_g10(ev: dict[str, Any]) -> tuple[bool, str | None]:
    blocked = bool(ev.get("manual_access_blocked", False))
    ok = not blocked
    return ok, None if ok else "Manual-access dependency blocks execution"


NICHE_TRANS_THRESHOLDS: dict[str, Any] = {
    "NT-G1": _check_nt_g1,
    "NT-G2": _check_nt_g2,
    "NT-G3": _check_nt_g3,
    "NT-G4": _check_nt_g4,
    "NT-G5": _check_nt_g5,
    "NT-G6": _check_nt_g6,
    "NT-G7": _check_nt_g7,
    "NT-G8": _check_nt_g8,
    "NT-G9": _check_nt_g9,
    "NT-G10": _check_nt_g10,
}


def evaluate_nichetrans_gate(evidence: dict[str, Any]) -> GateResult:
    """Evaluate all 10 NicheTrans hard gates against the provided evidence."""
    return evaluate_gates(
        candidate="nichetrans",
        gate_names=NICHE_TRANS_GATES,
        evidence=evidence,
        thresholds=NICHE_TRANS_THRESHOLDS,
    )
