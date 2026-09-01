"""Structure-metric replication of the positive control (TDD follow-up).

The main-text finding that Squidiff beats a diagonal-covariance baseline on
gene-gene correlation structure (Fig. 3c, Supplementary Note 10) was
computed only on CAR-NK. This script repeats it on the authors' own
released VO setting, so the structure-metric finding does not rest on a
single dataset. Only the repaired noise scale (0.03) is decoded here: the
upstream default (0.7) already fails on every marginal metric on this same
data (Supplementary Note 8), so its structure score would not change the
reading.

Reuses the same helpers already unit-tested in evaluation_robustness.py
(correlation_frobenius_distance, cluster_mass_recall) and the same
encode/decode/baseline logic as positive_control.py.

Output: artifacts/positive_control/structure_metrics.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external_runners" / "squidiff"))
sys.path.insert(0, str(REPO / "vendor" / "Squidiff"))

from evaluation_robustness import cluster_mass_recall, correlation_frobenius_distance  # noqa: E402
from load_released_checkpoint import PUBLISHED_VO_ARGS  # noqa: E402
from positive_control import _build_model, _decode, _dense, _encode, _sample_around  # noqa: E402

SAMPLING_SEEDS = [13, 37, 73]
SCALE = 0.03
N_CLUSTERS = 8
RARE_BELOW = 0.10


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def run(model_path: Path, adata_path: Path, output_dir: Path) -> dict:
    import anndata as ad
    import torch
    from Squidiff import dist_util

    from reuse_gate.models.temporal_baselines import conditional_mean_sampler

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist_util.dev = lambda: device
    print(f"device: {device}")

    adata = ad.read_h5ad(adata_path)
    X = _dense(adata)
    days = adata.obs["day"].to_numpy()
    X_d0, X_d1 = X[days == 0], X[days == 1]
    gene_size = PUBLISHED_VO_ARGS["gene_size"]

    def structure(gen: np.ndarray) -> dict:
        return {
            "correlation_frobenius": correlation_frobenius_distance(X_d1, gen),
            "rare_cluster_recall": cluster_mass_recall(
                X_d1, gen, n_clusters=N_CLUSTERS, rare_below=RARE_BELOW,
                rng=np.random.RandomState(1),
            ),
        }

    conditions: dict = {}

    # ── Squidiff at the repaired scale, 3 sampling seeds ──
    model, diffusion = _build_model(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    z_d0 = _encode(model, X_d0, device)
    z_d1 = _encode(model, X_d1, device)
    direction = z_d1.mean(axis=0) - z_d0.mean(axis=0)
    target = z_d0.mean(axis=0) + direction

    per_seed = []
    for seed in SAMPLING_SEEDS:
        rng = np.random.RandomState(seed)
        z_batch = _sample_around(target, X_d1.shape[0], rng, scale=SCALE)
        gen = _decode(diffusion, model, z_batch, gene_size, device)
        s = structure(gen)
        s["seed"] = seed
        per_seed.append(s)
        print(f"  squidiff scale {SCALE} seed {seed}: {s}", flush=True)
    conditions["squidiff_small_0.03"] = {
        "correlation_frobenius": float(np.mean([s["correlation_frobenius"] for s in per_seed])),
        "rare_cluster_recall": float(np.mean([s["rare_cluster_recall"] for s in per_seed])),
        "fit_set": (
            "released checkpoint trained on all 6,838 VO cells (days 0+1); "
            f"direction E[z_day1]-E[z_day0] from the released encoder; noise scale {SCALE}"
        ),
        "per_seed": per_seed,
    }

    # ── Baselines (no GPU needed) ──
    rng = np.random.RandomState(13)

    gen = conditional_mean_sampler(X, X_d1.shape[0], rng)
    conditions["conditional_mean_pooled"] = {
        **structure(gen),
        "fit_set": "per-gene mean/variance fit on all 6,838 pooled training cells (days 0+1)",
    }

    gen = X_d0[rng.choice(X_d0.shape[0], X_d1.shape[0], replace=True)]
    conditions["last_observation_day0_resample"] = {
        **structure(gen),
        "fit_set": f"real day-0 cells ({X_d0.shape[0]}) resampled with replacement",
    }

    gen = conditional_mean_sampler(X_d1, X_d1.shape[0], rng)
    conditions["oracle_gaussian_day1"] = {
        **structure(gen),
        "fit_set": (
            "per-gene mean/variance fit on the day-1 cells themselves "
            "(oracle: not usable for prediction; upper bound for marginal moment-matching)"
        ),
    }

    for name, c in conditions.items():
        print(f"  {name}: frob {c['correlation_frobenius']:.2f} "
              f"recall {c['rare_cluster_recall']:.3f}")

    result = {
        "purpose": "structure-metric replication of the positive control on the VO setting",
        "task": "predict the day-1 VO population from the day-0 anchor via the published latent-extrapolation mechanism",
        "checkpoint": str(model_path),
        "adata": str(adata_path),
        "input_sha256": {"model": _sha256(model_path), "adata": _sha256(adata_path)},
        "git_commit": _git_commit(),
        "sampling_seeds": SAMPLING_SEEDS,
        "noise_scale": SCALE,
        "n_clusters": N_CLUSTERS,
        "rare_below": RARE_BELOW,
        "n_day0": int(X_d0.shape[0]),
        "n_day1": int(X_d1.shape[0]),
        "device": str(device),
        "conditions": conditions,
    }
    out = output_dir / "structure_metrics.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {out}")
    return result


if __name__ == "__main__":
    base = Path("data/raw/upstream_figshare")
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else base / "VO_diff_model.pt"
    adata = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "VO_trained_adata.h5ad"
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("artifacts/positive_control")
    run(model, adata, outdir)
