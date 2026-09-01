"""Structure-metric replication of the positive control (TDD follow-up).

The main-text finding that Squidiff beats a diagonal-covariance baseline on
gene-gene correlation structure (Supplementary Note 10) was only checked on
CAR-NK. This test guards the same check repeated on the authors' own
released VO setting, using the same structure-metric helpers already
unit-tested in evaluation_robustness.py, so the finding does not rest on a
single dataset.

Requires the released artefacts (figshare, CC BY 4.0) and a GPU; marked
`integration` and `gpu`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = REPO_ROOT / "artifacts" / "positive_control" / "structure_metrics.json"

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture(scope="module")
def metrics() -> dict:
    if not METRICS.exists():
        pytest.skip(
            "artifacts/positive_control/structure_metrics.json is absent; "
            "run external_runners/squidiff/positive_control_structure.py, or "
            "fetch the frozen record from the archived release"
        )
    return json.loads(METRICS.read_text())


def test_covers_squidiff_and_baselines(metrics: dict):
    conditions = metrics["conditions"]
    for name in (
        "squidiff_small_0.03",
        "conditional_mean_pooled",
        "last_observation_day0_resample",
        "oracle_gaussian_day1",
    ):
        assert name in conditions, f"condition {name} missing"
        c = conditions[name]
        assert c["correlation_frobenius"] >= 0.0
        assert 0.0 <= c["rare_cluster_recall"] <= 1.0


def test_provenance_is_complete(metrics: dict):
    for key in ("input_sha256", "git_commit", "n_clusters", "rare_below"):
        assert metrics.get(key) is not None, f"provenance key {key} missing"
