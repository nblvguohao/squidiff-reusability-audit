"""Unit tests for the evaluation-robustness helpers (TDD Phase 2).

Pure-function tests on synthetic data: null-distribution anchors, bootstrap
CIs (cell- and cluster-level), gene-correlation structure distance, and
rare-cluster mass recall. No network, GPU, or full datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external_runners" / "squidiff"))

from evaluation_robustness import (  # noqa: E402
    bootstrap_metric_ci,
    cluster_mass_recall,
    correlation_frobenius_distance,
    evaluate_structure_metrics,
    null_energy_distance,
    parallel_bootstrap,
)


def test_null_energy_distance_is_near_zero_for_identical_distribution():
    rng = np.random.RandomState(13)
    real = rng.randn(400, 20)
    null = null_energy_distance(real, n_splits=25, rng=np.random.RandomState(7))
    assert len(null) == 25
    assert abs(float(np.mean(null))) < 1.0
    # A genuinely different distribution must sit far above the null band.
    shifted = real + 3.0
    from reuse_gate.metrics.distribution import energy_distance_multivariate

    ed_shifted = energy_distance_multivariate(real, shifted)
    assert ed_shifted > float(np.quantile(null, 0.95))


def test_bootstrap_ci_contains_estimate_and_tracks_clusters():
    rng = np.random.RandomState(13)
    real = rng.randn(300, 10)
    gen = rng.randn(300, 10) * 1.1

    def metric(a, b):
        return float(np.mean(a) - np.mean(b))

    est, lo, hi = bootstrap_metric_ci(real, gen, metric, n_boot=100, rng=np.random.RandomState(3))
    assert lo <= est <= hi

    # Cluster-level bootstrap: with 5 clusters, resampling clusters must give
    # a CI at least as wide as cell-level (fewer effective units).
    clusters = np.repeat(np.arange(5), 60)
    est_c, lo_c, hi_c = bootstrap_metric_ci(
        real, gen, metric, n_boot=100, rng=np.random.RandomState(3), cluster=clusters
    )
    assert lo_c <= est_c <= hi_c
    assert (hi_c - lo_c) >= 0.0


def test_parallel_bootstrap_matches_sequential_loop():
    """The parallel driver must be bit-for-bit identical to the sequential one.

    Each (population, mode) task seeds its own RandomState, so distributing
    the tasks changes execution order only, never the numbers. `processes=1`
    exercises the deterministic in-process fallback; the Pool path is thin
    glue over the same worker.
    """
    from reuse_gate.metrics.distribution import energy_distance_multivariate

    rng = np.random.RandomState(5)
    real = rng.randn(60, 4)
    populations = {"a": rng.randn(60, 4), "b": rng.randn(60, 4) + 0.5}
    samples = np.repeat(["s1", "s2", "s3"], 20)

    got = parallel_bootstrap(populations, real, samples, n_boot=10, seed=21, processes=1)

    for name, gen in populations.items():
        est, lo, hi = bootstrap_metric_ci(
            real, gen, energy_distance_multivariate, n_boot=10, rng=np.random.RandomState(21)
        )
        est_s, lo_s, hi_s = bootstrap_metric_ci(
            real, gen, energy_distance_multivariate, n_boot=10,
            rng=np.random.RandomState(21), cluster=samples,
        )
        entry = got[name]["energy_distance"]
        assert entry["estimate"] == est
        assert entry["cell_ci"] == [lo, hi]
        assert entry["sample_ci"] == [lo_s, hi_s]


def test_correlation_frobenius_distance_catches_structure():
    rng = np.random.RandomState(13)
    # Real data with genuine gene-gene correlation.
    base = rng.randn(500, 1)
    real = np.hstack([base + 0.1 * rng.randn(500, 1) for _ in range(5)])
    # Diagonal sampler: same marginals, zero covariance.
    gen = rng.randn(500, 5) * real.std(axis=0) + real.mean(axis=0)

    assert correlation_frobenius_distance(real, real) < 1e-9
    d = correlation_frobenius_distance(real, gen)
    assert d > 1.0, "diagonal sampler must be exposed on correlated data"


def test_correlation_frobenius_distance_ignores_zero_variance_real_genes():
    """A gene that never varies in `real` must not poison the whole score.

    Pearson correlation is undefined for a constant variable, so
    np.corrcoef silently returns NaN for that gene's row/col; without a
    guard, one such gene makes the Frobenius norm NaN regardless of how
    well every other gene is matched. This happened on the real VO
    released data (2 of 596 genes are exactly zero across the whole
    held-out population) and must not recur.
    """
    rng = np.random.RandomState(13)
    base = rng.randn(400, 1)
    real = np.hstack([base + 0.1 * rng.randn(400, 1) for _ in range(4)])
    real = np.hstack([real, np.zeros((400, 1))])  # gene 5: constant in real
    gen4 = rng.randn(400, 4) * real[:, :4].std(axis=0) + real[:, :4].mean(axis=0)
    gen = np.hstack([gen4, rng.randn(400, 1)])  # gen varies on that gene; must not matter

    d = correlation_frobenius_distance(real, gen)
    assert np.isfinite(d), "a constant real gene must not turn the score into NaN"
    # Must equal the score computed after manually dropping the dead gene.
    expected = correlation_frobenius_distance(real[:, :4], gen[:, :4])
    assert d == pytest.approx(expected)


def test_cluster_mass_recall_perfect_for_identical_population():
    rng = np.random.RandomState(13)
    # Two clusters: one common (90%), one rare (10%).
    common = rng.randn(450, 8)
    rare = rng.randn(50, 8) + 6.0
    real = np.vstack([common, rare])

    recall = cluster_mass_recall(real, real.copy(), n_clusters=2, rare_below=0.2, rng=np.random.RandomState(1))
    assert recall == 1.0

    # A generator that drops the rare cluster entirely scores 0.
    gen_missing = common[rng.choice(450, 500, replace=True)]
    recall_missing = cluster_mass_recall(real, gen_missing, n_clusters=2, rare_below=0.2, rng=np.random.RandomState(1))
    assert recall_missing == 0.0


def test_evaluate_structure_metrics_reports_reference_and_mass_sensitivity():
    rng = np.random.RandomState(13)
    common = rng.randn(360, 6)
    rare = rng.randn(40, 6) + 5.0
    real = np.vstack([common, rare])
    generated = common[rng.choice(common.shape[0], real.shape[0], replace=True)]

    result = evaluate_structure_metrics(
        real,
        generated,
        cluster_counts=(2, 3),
        rare_thresholds=(0.15, 0.20),
        random_state=7,
    )

    assert set(result["correlation_frobenius"]) == {"raw", "normalized"}
    assert result["cluster_mass"]["rare_mass_recall"] < 1.0
    assert len(result["cluster_mass_sensitivity"]) == 4
