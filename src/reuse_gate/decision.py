"""Deterministic candidate selection engine.

Candidate priority is fixed: NicheTrans → Squidiff → CMonge → NO_GO.
This must never be reordered without updating the plan and all tests.
"""

from __future__ import annotations

from reuse_gate.schemas import CandidateName, CandidateResults, GateResult, SelectionDecision


def _gate_to_candidate_result(gate: GateResult) -> CandidateResults:
    """Convert a GateResult to CandidateResults."""
    return CandidateResults(
        hard_gate_pass=gate.hard_gate_pass,
        failed_gates=gate.failed_gates,
        evidence_paths=gate.evidence_paths,
    )


def choose_candidate(
    nichetrans: GateResult,
    squidiff: GateResult,
    cmonge: GateResult,
    evaluation_date: str,
    repository_commit: str,
) -> SelectionDecision:
    """Apply deterministic priority: NicheTrans → Squidiff → CMonge → NO_GO.

    The first candidate with hard_gate_pass=True is selected.
    """
    candidate_results: dict[str, CandidateResults] = {
        "nichetrans": _gate_to_candidate_result(nichetrans),
        "squidiff": _gate_to_candidate_result(squidiff),
        "cmonge": _gate_to_candidate_result(cmonge),
    }

    # Priority-ordered evaluation
    selected: CandidateName | str
    decision_rule: str

    if nichetrans.hard_gate_pass:
        selected = "nichetrans"
        decision_rule = "NicheTrans passed all hard gates (NT-G1 through NT-G10). Selected per priority rule 1."
    elif squidiff.hard_gate_pass:
        selected = "squidiff"
        decision_rule = (
            f"NicheTrans failed gates: {nichetrans.failed_gates}. "
            "Squidiff passed all hard gates (SQ-G1 through SQ-G10). Selected per priority rule 2."
        )
    elif cmonge.hard_gate_pass:
        selected = "cmonge"
        decision_rule = (
            f"NicheTrans failed gates: {nichetrans.failed_gates}. "
            f"Squidiff failed gates: {squidiff.failed_gates}. "
            "CMonge passed all hard gates (CM-G1 through CM-G10). Selected per priority rule 3."
        )
    else:
        selected = "NO_GO"
        decision_rule = (
            f"All candidates failed. "
            f"NicheTrans: {nichetrans.failed_gates}; "
            f"Squidiff: {squidiff.failed_gates}; "
            f"CMonge: {cmonge.failed_gates}. "
            "No reusable candidate found."
        )

    return SelectionDecision(
        evaluation_date=evaluation_date,
        selected_candidate=selected,
        decision_rule=decision_rule,
        candidate_results=candidate_results,
        repository_commit=repository_commit,
    )
