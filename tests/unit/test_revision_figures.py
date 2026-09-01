"""Tests for the final NMI revision figures and their source data."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external_runners" / "squidiff"))

import make_revision_figures  # noqa: E402
from make_revision_figures import figure3_source_rows, make_all_figures  # noqa: E402


def _metrics(value: float) -> dict:
    return {
        "energy_distance": value,
        "mean_expression_correlation": 1 - value / 100,
        "correlation_frobenius_raw": value * 2,
        "correlation_frobenius_normalized": value / 100,
        "cluster_mass": {
            "cluster_mass_mae": value / 100,
            "cluster_mass_jsd": value / 200,
            "rare_mass_recall": 0.8,
            "rare_mass_precision": 0.7,
        },
        "cluster_mass_sensitivity": [
            {
                "n_clusters": 8,
                "rare_below": 0.1,
                "cluster_mass_mae": value / 100,
                "cluster_mass_jsd": value / 200,
                "rare_mass_recall": 0.8,
                "rare_mass_precision": 0.7,
                "rare_coverage": 1.0,
            }
        ],
    }


def _manifest() -> dict:
    cutoffs = {}
    for index, name in enumerate(("primary_d21_d28", "early_d14", "late_d28")):
        baselines = {
            baseline: {"primary": _metrics(2.0 + offset + index)}
            for offset, baseline in enumerate(
                (
                    "last_observation_resample",
                    "pooled_diagonal_gaussian",
                    "temporal_diagonal_gaussian",
                    "temporal_factor_gaussian",
                )
            )
        }
        cutoffs[name] = {
            "per_seed": [
                {
                    "seed": 13,
                    "squidiff": {"scale_0.03": {"primary": _metrics(4.0 + index)}},
                    "baselines": baselines,
                }
            ],
            "same_distribution_reference": {
                "summary": {
                    "energy_distance": {"mean": 0.1},
                    "mean_expression_correlation": {"mean": 0.99},
                    "correlation_frobenius_normalized": {"mean": 0.05},
                }
            },
            "split_manifest": {
                "n_train_cells": 100,
                "n_test_cells": 40,
                "train_sample_ids": ["a", "b"],
                "test_sample_ids": ["c"],
            },
        }
    return {
        "cutoffs": cutoffs,
        "release_audit": {
            "released_checkpoint": {
                "checkpoint_info": {"n_tensors": 62, "n_parameters": 54_565_522},
                "load_state_dict": {
                    "n_missing": 0,
                    "n_unexpected": 0,
                    "strict_would_pass": True,
                },
                "sampling": {"finite": True, "energy_distance": 2.10},
            },
            "preprocessing_ab": {
                "fixed_noise_scale": 0.03,
                "budgets": [5_000, 20_000, 50_000],
                "conditions": {
                    "raw": {
                        "per_budget": [
                            {"steps": 5_000, "pooled_energy_distance": 3.0},
                            {"steps": 20_000, "pooled_energy_distance": 4.0},
                            {"steps": 50_000, "pooled_energy_distance": 5.0},
                        ]
                    },
                    "lognorm": {
                        "per_budget": [
                            {"steps": 5_000, "pooled_energy_distance": 3.0},
                            {"steps": 20_000, "pooled_energy_distance": 2.0},
                            {"steps": 50_000, "pooled_energy_distance": 1.0},
                        ]
                    },
                },
            },
            "latent_noise_sensitivity": {
                "upstream_default_scale": 0.7,
                "scales": [
                    {"scale": 0.0, "pooled_energy_distance": 1.0},
                    {"scale": 0.03, "pooled_energy_distance": 1.1},
                    {"scale": 0.7, "pooled_energy_distance": 20.0},
                ],
            },
            "simulated_benchmark": {
                "preprocessing_degeneracy": {
                    "raw simulation": {"silhouette": 0.47},
                    "after normalize + log1p": {"silhouette": -0.06},
                }
            },
            "conditional_interface": {"defects": ["dtype", "device", "rank"]},
        },
        "vo_sanity_check": {
            "interpretation": "target-informed",
            "supports_generalization": False,
        },
    }


def test_figure3_source_rows_cover_all_cutoffs_and_baselines():
    rows = figure3_source_rows(_manifest())

    assert {row["cutoff"] for row in rows} == {
        "primary_d21_d28",
        "early_d14",
        "late_d28",
    }
    assert {row["method"] for row in rows} == {
        "Squidiff",
        "Last observation",
        "Pooled diagonal Gaussian",
        "Temporal diagonal Gaussian",
        "Temporal factor Gaussian",
    }


def test_final_figure_bundle_uses_current_claim_language(tmp_path):
    outputs = make_all_figures(_manifest(), tmp_path, dpi=72)

    assert len(outputs) == 13
    assert all(path.exists() for path in outputs)
    text = (tmp_path / "figure_source_data.json").read_text(encoding="utf-8").lower()
    for stale in ("positive control", "evaluation regime", "cannot win"):
        assert stale not in text
    payload = json.loads((tmp_path / "figure_source_data.json").read_text())
    assert payload["vo_interpretation"] == "target-informed"
    assert set(payload["figure1_cutoffs"]) == {
        "primary_d21_d28",
        "early_d14",
        "late_d28",
    }
    assert len(payload["figure2"]["preprocessing_rows"]) == 6
    assert len(payload["figure2"]["latent_noise_rows"]) == 3
    assert payload["figure2"]["released_checkpoint"]["energy_distance"] == 2.10
    svg_text = "\n".join(
        (tmp_path / f"Figure_{index}.svg").read_text(encoding="utf-8") for index in (1, 2, 3)
    )
    assert "鈥" not in svg_text
    assert "\ufffd" not in svg_text
    assert "鈥" not in inspect.getsource(make_revision_figures)
