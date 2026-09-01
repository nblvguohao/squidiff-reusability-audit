"""Tests for normalized structure distances and cluster-mass diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from reuse_gate.metrics.structure import (
    cluster_mass_metrics,
    correlation_frobenius,
    same_distribution_structure_reference,
    structure_sensitivity_grid,
)


def _correlated_population(seed: int, n: int = 500) -> np.ndarray:
    rng = np.random.RandomState(seed)
    latent = rng.normal(size=(n, 2))
    return np.c_[
        latent[:, 0],
        latent[:, 0] + rng.normal(0, 0.1, n),
        latent[:, 1],
        latent[:, 1] + rng.normal(0, 0.1, n),
    ]


def test_normalized_frobenius_divides_by_retained_gene_count() -> None:
    real = _correlated_population(1)
    generated = np.random.RandomState(2).normal(size=real.shape)

    raw = correlation_frobenius(real, generated, normalized=False)
    normalized = correlation_frobenius(real, generated, normalized=True)

    assert normalized == pytest.approx(raw / real.shape[1])


def test_correlation_distance_drops_constant_real_genes() -> None:
    real = _correlated_population(3)
    generated = _correlated_population(4)
    real = np.c_[real, np.zeros(real.shape[0])]
    generated = np.c_[generated, np.ones(generated.shape[0])]

    result = correlation_frobenius(real, generated, normalized=False)

    assert np.isfinite(result)


def test_same_distribution_reference_is_reproducible() -> None:
    real = _correlated_population(5, n=300)

    first = same_distribution_structure_reference(
        real,
        n_splits=12,
        rng=np.random.RandomState(6),
    )
    second = same_distribution_structure_reference(
        real,
        n_splits=12,
        rng=np.random.RandomState(6),
    )

    assert first == second
    assert first["n_splits"] == 12
    assert first["normalized"]["q95"] >= first["normalized"]["mean"] > 0


def test_cluster_mass_metrics_penalize_one_cell_coverage() -> None:
    rng = np.random.RandomState(7)
    major = rng.normal(loc=[0, 0], scale=0.2, size=(900, 2))
    rare_a = rng.normal(loc=[4, 0], scale=0.1, size=(50, 2))
    rare_b = rng.normal(loc=[0, 4], scale=0.1, size=(50, 2))
    real = np.concatenate([major, rare_a, rare_b])
    generated = np.concatenate([major[:998], rare_a[:1], rare_b[:1]])

    result = cluster_mass_metrics(
        real,
        generated,
        n_clusters=3,
        rare_below=0.10,
        random_state=0,
    )

    assert result["rare_coverage"] == 1.0
    assert result["cluster_mass_mae"] > 0.03
    assert result["cluster_mass_jsd"] > 0
    assert result["rare_mass_recall"] < 0.1
    assert result["rare_mass_precision"] == pytest.approx(1.0)


def test_structure_sensitivity_grid_covers_every_setting() -> None:
    real = _correlated_population(8, n=400)
    generated = _correlated_population(9, n=400)

    rows = structure_sensitivity_grid(
        real,
        generated,
        cluster_counts=(3, 4),
        rare_thresholds=(0.05, 0.10, 0.15),
        random_state=1,
    )

    assert len(rows) == 6
    assert {(row["n_clusters"], row["rare_below"]) for row in rows} == {
        (3, 0.05),
        (3, 0.10),
        (3, 0.15),
        (4, 0.05),
        (4, 0.10),
        (4, 0.15),
    }
