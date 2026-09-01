"""Tests for correction-safe post-hoc cutoff evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external_runners" / "squidiff"))

from cutoff_study import load_cutoff_config  # noqa: E402
from posthoc_cutoff_evaluation import (  # noqa: E402
    evaluate_cutoff,
    evaluate_generated_models,
)

CONFIG = ROOT / "configs" / "cutoff_studies.yaml"


def _write_split(path: Path) -> tuple[Path, Path]:
    rng = np.random.RandomState(7)
    train = ad.AnnData(
        X=np.vstack(
            [
                rng.normal(0.0, 0.2, size=(48, 14)),
                rng.normal(4.0, 0.2, size=(48, 14)),
            ]
        ).astype(np.float32),
        obs=pd.DataFrame(
            {
                "timepoint_numeric": [0] * 48 + [7] * 48,
                "sample_id": ["d0"] * 48 + ["d7"] * 48,
            },
            index=[f"train_{i}" for i in range(96)],
        ),
    )
    test = ad.AnnData(
        X=rng.normal(8.0, 0.3, size=(48, 14)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "timepoint_numeric": [14] * 48,
                "sample_id": ["d14"] * 48,
            },
            index=[f"test_{i}" for i in range(48)],
        ),
    )
    train_path = path / "train.h5ad"
    test_path = path / "test.h5ad"
    path.mkdir(parents=True)
    train.write_h5ad(train_path)
    test.write_h5ad(test_path)
    return train_path, test_path


def test_posthoc_correction_scores_only_requested_baseline(tmp_path):
    train_path, test_path = _write_split(tmp_path / "split")
    config = load_cutoff_config(CONFIG, "early_d14")

    result = evaluate_cutoff(
        config,
        train_path,
        test_path,
        seeds=(13,),
        baseline_names=("pooled_diagonal_gaussian",),
        reference_splits=4,
    )

    assert result["correction_reason"].startswith("pooled_diagonal_gaussian")
    assert result["completed_seeds"] == [13]
    assert set(result["per_seed"][0]["baselines"]) == {
        "pooled_diagonal_gaussian"
    }
    assert result["same_distribution_reference"]["repeats"] == 4
    # Full-window mean is 2, whereas the latest-only mean is 4.
    generated_mean = result["per_seed"][0]["generated_summary"][
        "pooled_diagonal_gaussian"
    ]["14"]["mean"]
    assert 1.8 < generated_mean < 2.2


def test_posthoc_result_roundtrips_as_json(tmp_path):
    train_path, test_path = _write_split(tmp_path / "split")
    config = load_cutoff_config(CONFIG, "early_d14")
    result = evaluate_cutoff(
        config,
        train_path,
        test_path,
        seeds=(13,),
        baseline_names=("pooled_diagonal_gaussian",),
        reference_splits=2,
    )

    destination = tmp_path / "posthoc_evaluation.json"
    destination.write_text(json.dumps(result), encoding="utf-8")

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["input_sha256"]["train_h5ad"]
    assert loaded["input_sha256"]["test_h5ad"]


def test_cached_generated_populations_receive_current_metric_schema(tmp_path):
    train_path, test_path = _write_split(tmp_path / "split")
    config = load_cutoff_config(CONFIG, "early_d14")
    generated = np.random.RandomState(9).normal(7.5, 0.4, size=(48, 14))
    generated_path = tmp_path / "seed_13.npy"
    np.save(generated_path, generated)

    result = evaluate_generated_models(
        config,
        train_path,
        test_path,
        generated={13: (0.03, generated_path)},
    )

    metrics = result["per_seed"][0]["squidiff"]["scale_0.03"]["primary"]
    assert "correlation_frobenius_normalized" in metrics
    assert "cluster_mass_sensitivity" in metrics
    assert result["per_seed"][0]["generated_sha256"]
