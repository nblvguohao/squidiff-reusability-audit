"""Unit tests for candidate gate evaluation — RED phase."""

from reuse_gate.gates.nichetrans import evaluate_nichetrans_gate


def test_nichetrans_gate_fails_when_only_one_biological_unit():
    """NT-G3 requires >= 2 biological units. One unit must fail."""
    evidence: dict = {
        "official_repo_pinned": True,
        "official_smoke_pass": True,
        "biological_units": 1,
        "min_locations_per_unit": 1000,
        "target_features": 100,
        "match_rate": 0.95,
        "group_split_pass": True,
        "baseline_count": 3,
        "gpu_smoke_pass": True,
        "manual_access_blocked": False,
    }
    result = evaluate_nichetrans_gate(evidence)
    assert result.hard_gate_pass is False
    assert "NT-G3" in result.failed_gates


def test_nichetrans_gate_passes_when_all_conditions_met():
    """All NT gates must pass when evidence meets every threshold."""
    evidence: dict = {
        "official_repo_pinned": True,
        "official_smoke_pass": True,
        "biological_units": 3,
        "min_locations_per_unit": 600,
        "target_features": 50,
        "match_rate": 0.96,
        "group_split_pass": True,
        "baseline_count": 3,
        "gpu_smoke_pass": True,
        "manual_access_blocked": False,
    }
    result = evaluate_nichetrans_gate(evidence)
    assert result.hard_gate_pass is True
    assert len(result.failed_gates) == 0


def test_nichetrans_gate_fails_when_match_rate_below_threshold():
    """NT-G6 requires >= 90% match rate."""
    evidence: dict = {
        "official_repo_pinned": True,
        "official_smoke_pass": True,
        "biological_units": 3,
        "min_locations_per_unit": 600,
        "target_features": 50,
        "match_rate": 0.85,
        "group_split_pass": True,
        "baseline_count": 3,
        "gpu_smoke_pass": True,
        "manual_access_blocked": False,
    }
    result = evaluate_nichetrans_gate(evidence)
    assert result.hard_gate_pass is False
    assert "NT-G6" in result.failed_gates
