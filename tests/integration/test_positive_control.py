"""Integration test for the positive control (TDD Phase 1.2).

The positive control scores Squidiff — released checkpoint, released VO
training data, published latent-extrapolation mechanism, upstream default
noise scale — against the same baselines used on CAR-NK, on the authors' own
data. It exists to separate two explanations of the CAR-NK result:
(a) Squidiff fails to transfer, vs (b) the evaluation regime favours
moment-matched samplers on any data.

Requires the released artefacts (figshare, CC BY 4.0) and a GPU; marked
`integration` and `gpu`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = REPO_ROOT / "artifacts" / "positive_control" / "positive_control_metrics.json"

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture(scope="module")
def metrics() -> dict:
    if not METRICS.exists():
        pytest.skip(
            "artifacts/positive_control/positive_control_metrics.json is absent; "
            "run external_runners/squidiff/positive_control.py, or fetch the "
            "frozen record from the archived release"
        )
    return json.loads(METRICS.read_text())


def test_metrics_cover_squidiff_and_baselines(metrics: dict):
    conditions = metrics["conditions"]
    for name in (
        "squidiff_upstream_default_0.7",
        "conditional_mean_pooled",
        "last_observation_day0_resample",
        "oracle_gaussian_day1",
    ):
        assert name in conditions, f"condition {name} missing"
        for metric in ("energy_distance", "mmd_rbf", "mean_expression_correlation"):
            value = conditions[name][metric]
            assert value == pytest.approx(value), f"{name}.{metric} is NaN"
            assert value >= 0.0 or metric == "mean_expression_correlation"
            assert -1.0 <= conditions[name]["mean_expression_correlation"] <= 1.0


def test_fit_sets_are_recorded(metrics: dict):
    """Every baseline must name the cells it was fit on."""
    for name, info in metrics["conditions"].items():
        assert info.get("fit_set"), f"{name} lacks a fit_set statement"


def test_provenance_is_complete(metrics: dict):
    for key in ("input_sha256", "git_commit", "released_config", "noise_scales"):
        assert metrics.get(key), f"provenance key {key} missing"
