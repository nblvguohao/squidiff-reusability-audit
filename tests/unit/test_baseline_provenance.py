"""Unit tests for the energy-distance decomposition helper (TDD Phase 1.1).

The decomposition underlies the provenance explanation: a point-mass
prediction has a zero within-sample term, which is why energy distance
penalizes it relative to a variance-matched sampler.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external_runners" / "squidiff"))

from baseline_provenance import decompose_energy_distance  # noqa: E402

from reuse_gate.metrics.distribution import energy_distance_multivariate  # noqa: E402


def test_decomposition_sums_to_energy_distance():
    rng = np.random.RandomState(13)
    real = rng.randn(200, 5) * 2.0 + 1.0
    gen = rng.randn(150, 5) * 2.0 + 1.2

    parts = decompose_energy_distance(real, gen)
    ed = energy_distance_multivariate(real, gen)

    assert set(parts) == {"cross", "within_real", "within_generated", "energy_distance"}
    assert parts["energy_distance"] == np.clip(
        2.0 * parts["cross"] - parts["within_real"] - parts["within_generated"], 0.0, None
    )
    assert abs(parts["energy_distance"] - ed) < 1e-9


def test_point_mass_has_zero_within_term():
    rng = np.random.RandomState(13)
    real = rng.randn(200, 5)
    point = np.tile(real.mean(axis=0, keepdims=True), (150, 1))

    parts = decompose_energy_distance(real, point)
    assert parts["within_generated"] == 0.0


def test_matched_gaussian_beats_point_mass_via_within_term():
    """The mechanism behind the baseline ordering, on synthetic data.

    A Gaussian matched to the real marginals yields near-zero energy distance
    while a point mass at the same mean does not, because the point mass gets
    no credit from the within-generated term.
    """
    rng = np.random.RandomState(13)
    mean = rng.randn(50)
    std = np.abs(rng.randn(50)) + 0.5
    real = rng.randn(400, 50) * std + mean

    point = np.tile(mean, (400, 1))
    gauss = rng.randn(400, 50) * std + mean

    ed_point = decompose_energy_distance(real, point)["energy_distance"]
    ed_gauss = decompose_energy_distance(real, gauss)["energy_distance"]

    assert ed_gauss < ed_point
    assert ed_gauss < 1.0  # matched marginals -> near-zero ED at n=400
