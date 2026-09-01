"""Tests for the single-source revision result manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external_runners" / "squidiff"))

from consolidate_revision_results import consolidate  # noqa: E402


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    _write(
        root / "artifacts/squidiff_seed_study/seed_study_metrics.json",
        {"seeds": [13, 37, 73, 101, 137], "summary": {"metric": "primary"}},
    )
    _write(
        root / "artifacts/evaluation_robustness/robustness.json",
        {"null_anchor": {"energy_distance": {"mean": 0.1}}},
    )
    _write(
        root / "artifacts/squidiff_sweep_lognorm/split_manifest.json",
        {"train_times": [0, 7, 14], "test_times": [21, 28]},
    )
    for name in ("early_d14", "late_d28"):
        _write(
            root / f"artifacts/cutoff_studies/{name}/cutoff_summary.json",
            {
                "cutoff": name,
                "expected_seeds": [13, 37, 73, 101, 137],
                "completed_seeds": [13, 37, 73, 101, 137],
                "complete": True,
                "per_seed": [
                    {
                        "seed": seed,
                        "baselines": {"pooled_diagonal_gaussian": {"source": "uncorrected"}},
                    }
                    for seed in (13, 37, 73, 101, 137)
                ],
            },
        )
    for name in ("primary_d21_d28", "early_d14", "late_d28"):
        _write(
            root / f"artifacts/cutoff_studies/{name}/posthoc_evaluation.json",
            {
                "cutoff": name,
                "expected_seeds": [13, 37, 73, 101, 137],
                "completed_seeds": [13, 37, 73, 101, 137],
                "complete": True,
                "same_distribution_reference": {"repeats": 50},
                "per_seed": [
                    {
                        "seed": seed,
                        "baselines": {"pooled_diagonal_gaussian": {"source": "corrected"}},
                    }
                    for seed in (13, 37, 73, 101, 137)
                ],
            },
        )
    _write(
        root / "artifacts/cutoff_studies/primary_d21_d28/posthoc_model_evaluation.json",
        {
            "cutoff": "primary_d21_d28",
            "expected_seeds": [13, 37, 73, 101, 137],
            "completed_seeds": [13, 37, 73, 101, 137],
            "complete": True,
            "per_seed": [
                {
                    "seed": seed,
                    "selected_or_fixed_scales": [0.03],
                    "squidiff": {"scale_0.03": {"primary": {"metric": seed}}},
                }
                for seed in (13, 37, 73, 101, 137)
            ],
        },
    )
    _write(
        root / "artifacts/positive_control/positive_control_metrics.json",
        {"task": "day-1 from day-0", "results": {}},
    )
    _write(
        root / "artifacts/positive_control/structure_metrics.json",
        {"task": "day-1 from day-0", "structure": {}},
    )
    return root


def test_revision_manifest_contains_all_cutoffs(tmp_path):
    result = consolidate(_synthetic_root(tmp_path))

    assert set(result["cutoffs"]) == {
        "primary_d21_d28",
        "early_d14",
        "late_d28",
    }
    assert all(len(result["cutoffs"][name]["per_seed"]) == 5 for name in result["cutoffs"])
    assert "squidiff" in result["cutoffs"]["primary_d21_d28"]["per_seed"][0]
    assert result["cutoffs"]["primary_d21_d28"]["split_manifest"] == {
        "train_times": [0, 7, 14],
        "test_times": [21, 28],
    }


def test_vo_is_explicitly_target_informed(tmp_path):
    result = consolidate(_synthetic_root(tmp_path))

    assert result["vo_sanity_check"]["interpretation"] == "target-informed"
    assert result["vo_sanity_check"]["supports_generalization"] is False
    assert "target" in result["vo_sanity_check"]["information_boundary"].lower()


def test_posthoc_baseline_correction_overwrites_remote_value(tmp_path):
    result = consolidate(_synthetic_root(tmp_path))

    early = result["cutoffs"]["early_d14"]
    assert early["per_seed"][0]["baselines"]["pooled_diagonal_gaussian"] == {"source": "corrected"}
    assert early["same_distribution_reference"]["repeats"] == 50


def test_incomplete_cutoff_is_rejected(tmp_path):
    root = _synthetic_root(tmp_path)
    path = root / "artifacts/cutoff_studies/early_d14/cutoff_summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["complete"] = False
    _write(path, value)

    with pytest.raises(ValueError, match="incomplete"):
        consolidate(root)
