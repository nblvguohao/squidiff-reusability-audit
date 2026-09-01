"""Baseline provenance audit: what each baseline is fit on, and why they rank.

Reviewer-revision TDD, Phase 1.1. The manuscript's headline comparison rests
on two baselines whose construction was never stated in text. This script
rebuilds them exactly as `seed_study.py` does, decomposes the energy
distance into its three terms for each, and adds a true last-observation
baseline (resampled real D14 cells) so the point-mass nature of the
implemented `last_observation` is explicit.

Outputs `artifacts/baseline_provenance/baseline_provenance.json`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def decompose_energy_distance(real: np.ndarray, gen: np.ndarray) -> dict:
    """Split energy distance into cross / within-real / within-generated.

    ED = 2*cross - within_real - within_generated (clipped at 0). A constant
    prediction has within_generated == 0 and is penalized accordingly.
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    cross = float(cdist(real, gen).mean())
    within_real = float(cdist(real, real).mean())
    within_gen = float(cdist(gen, gen).mean())
    return {
        "cross": cross,
        "within_real": within_real,
        "within_generated": within_gen,
        "energy_distance": max(0.0, 2.0 * cross - within_real - within_gen),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run(sweep_dir: Path, output_dir: Path) -> dict:
    import anndata as ad

    from reuse_gate.metrics.distribution import (
        energy_distance_multivariate,
        mean_expression_correlation,
        median_pairwise_distance,
        mmd_rbf,
    )
    from reuse_gate.models.temporal_baselines import (
        conditional_mean_sampler,
        last_observation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_ad = ad.read_h5ad(sweep_dir / "train.h5ad")
    test_ad = ad.read_h5ad(sweep_dir / "test.h5ad")
    train_mat = np.asarray(train_ad.X.todense() if hasattr(train_ad.X, "todense") else train_ad.X)
    test_mat = np.asarray(test_ad.X.todense() if hasattr(test_ad.X, "todense") else test_ad.X)
    train_tp = train_ad.obs["timepoint_numeric"].to_numpy()
    test_tp = test_ad.obs["timepoint_numeric"].to_numpy()

    bandwidth = median_pairwise_distance(train_mat)
    rng = np.random.RandomState(13)

    def full_scores(real: np.ndarray, gen: np.ndarray) -> dict:
        return {
            **decompose_energy_distance(real, gen),
            "mmd_rbf": float(mmd_rbf(real, gen, bandwidth=bandwidth)),
            "mean_expression_correlation": float(mean_expression_correlation(real, gen)),
        }

    # Exactly as seed_study.py constructs them.
    gen_point = last_observation(train_mat, test_mat)
    gen_gauss = conditional_mean_sampler(train_mat, test_mat.shape[0], np.random.RandomState(13))

    # True last-observation: resample the real D14 cells (training window's
    # last timepoint), with replacement, to held-out size.
    d14 = train_mat[train_tp == 14]
    gen_true_lo = d14[rng.choice(d14.shape[0], test_mat.shape[0], replace=True)]

    baselines = {
        "conditional_mean": {
            "fit_set": (
                "pooled training window (pre-infusion + D7 + D14, "
                f"{train_mat.shape[0]} cells); per-gene mean and variance; "
                "held-out cells never touched"
            ),
            "scores": full_scores(test_mat, gen_gauss),
        },
        "last_observation_as_implemented": {
            "fit_set": (
                "pooled training mean over the same window, tiled to every "
                "held-out cell: a constant point-mass prediction with zero "
                "variance (despite the name, it is not the last timepoint)"
            ),
            "scores": full_scores(test_mat, gen_point),
        },
        "last_observation_true_d14_resample": {
            "fit_set": (
                f"real D14 cells from the training window ({d14.shape[0]} cells), "
                "resampled with replacement to held-out size"
            ),
            "scores": full_scores(test_mat, gen_true_lo),
        },
    }

    sanity = {
        "energy_distance_check": {
            name: {
                "decomposed": b["scores"]["energy_distance"],
                "direct": float(energy_distance_multivariate(test_mat, gen)),
            }
            for name, b, gen in (
                ("conditional_mean", baselines["conditional_mean"], gen_gauss),
                ("last_observation_as_implemented", baselines["last_observation_as_implemented"], gen_point),
                ("last_observation_true_d14_resample", baselines["last_observation_true_d14_resample"], gen_true_lo),
            )
        },
        "train_cells": int(train_mat.shape[0]),
        "test_cells": int(test_mat.shape[0]),
        "test_timepoints": {int(tp): int((test_tp == tp).sum()) for tp in np.unique(test_tp)},
        "mmd_bandwidth": bandwidth,
    }

    result = {
        "purpose": "baseline fit-set provenance and ED decomposition (TDD Phase 1.1)",
        "sweep_dir": str(sweep_dir),
        "input_sha256": {
            "train_h5ad": _sha256(sweep_dir / "train.h5ad"),
            "test_h5ad": _sha256(sweep_dir / "test.h5ad"),
        },
        "git_commit": _git_commit(),
        "baselines": baselines,
        "sanity": sanity,
    }

    out = output_dir / "baseline_provenance.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["scores"] for k, v in baselines.items()}, indent=2))
    print(f"\nSaved: {out}")
    return result


if __name__ == "__main__":
    sweep = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/squidiff_sweep_lognorm")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/baseline_provenance")
    run(sweep, outdir)
