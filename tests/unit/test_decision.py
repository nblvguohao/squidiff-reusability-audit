"""Unit tests for deterministic decision engine — RED phase."""

import pytest

from reuse_gate.decision import choose_candidate
from reuse_gate.schemas import GateResult


@pytest.fixture
def pass_gate():
    """Factory for a passing GateResult."""
    def _pass(candidate: str) -> GateResult:
        return GateResult(candidate=candidate, hard_gate_pass=True, failed_gates=[])
    return _pass


@pytest.fixture
def fail_gate():
    """Factory for a failing GateResult."""
    def _fail(candidate: str, *failed: str) -> GateResult:
        return GateResult(
            candidate=candidate,
            hard_gate_pass=False,
            failed_gates=list(failed),
        )
    return _fail


def test_nichetrans_wins_when_all_candidates_pass(pass_gate):
    """NicheTrans has top priority. If all pass, NicheTrans wins."""
    decision = choose_candidate(
        nichetrans=pass_gate("nichetrans"),
        squidiff=pass_gate("squidiff"),
        cmonge=pass_gate("cmonge"),
        evaluation_date="2026-07-22",
        repository_commit="abc",
    )
    assert decision.selected_candidate == "nichetrans"


def test_squidiff_is_selected_when_nichetrans_fails(pass_gate, fail_gate):
    """When NicheTrans fails, Squidiff is evaluated next."""
    decision = choose_candidate(
        nichetrans=fail_gate("nichetrans", "NT-G1"),
        squidiff=pass_gate("squidiff"),
        cmonge=pass_gate("cmonge"),
        evaluation_date="2026-07-22",
        repository_commit="abc",
    )
    assert decision.selected_candidate == "squidiff"


def test_no_go_when_all_fail(fail_gate):
    """When all three candidates fail, NO_GO is selected."""
    decision = choose_candidate(
        nichetrans=fail_gate("nichetrans", "NT-G1"),
        squidiff=fail_gate("squidiff", "SQ-G4"),
        cmonge=fail_gate("cmonge", "CM-G6"),
        evaluation_date="2026-07-22",
        repository_commit="abc",
    )
    assert decision.selected_candidate == "NO_GO"


def test_cmonge_only_evaluated_after_both_fail(pass_gate, fail_gate):
    """CMonge wins only when both NicheTrans and Squidiff fail."""
    decision = choose_candidate(
        nichetrans=fail_gate("nichetrans", "NT-G1"),
        squidiff=fail_gate("squidiff", "SQ-G4"),
        cmonge=pass_gate("cmonge"),
        evaluation_date="2026-07-22",
        repository_commit="abc",
    )
    assert decision.selected_candidate == "cmonge"


def test_decision_includes_all_candidate_results(pass_gate):
    """The decision must contain results for all three candidates."""
    decision = choose_candidate(
        nichetrans=pass_gate("nichetrans"),
        squidiff=pass_gate("squidiff"),
        cmonge=pass_gate("cmonge"),
        evaluation_date="2026-07-22",
        repository_commit="abc",
    )
    assert "nichetrans" in decision.candidate_results
    assert "squidiff" in decision.candidate_results
    assert "cmonge" in decision.candidate_results
