"""Positive control: the same baselines, on the authors' own successful setting.

Reviewer-revision TDD, Phase 1.2. The CAR-NK result admits two explanations:
(a) Squidiff fails to transfer to new data, or (b) the evaluation regime —
distributional metrics on a low-drift task — favours moment-matched samplers
regardless of model. This control separates them by replaying the comparison
on the authors' own released artefacts, where the method is reported to work:

  checkpoint   VO_diff_model.pt        (figshare 10.6084/m9.figshare.27948633)
  training ad  VO_trained_adata.h5ad   6,838 cells, 596 genes, days {0, 1}
  mechanism    published latent extrapolation: encode the two observed states,
               direction = E[z_day1] - E[z_day0], target = E[z_day0] + direction,
               sample around the target, decode with DDIM
  noise scale  0.7 (upstream default — the only value the released prediction
               path can produce, see reports/noise_constant_provenance.md)
               and 0.03 (the scale validation selected on CAR-NK), 3 seeds each

Baselines, all fit without touching the day-1 cells being predicted unless
explicitly labelled oracle:

  conditional_mean_pooled        per-gene Gaussian fit on all 6,838 training
                                 cells — the same information the checkpoint
                                 was trained on
  last_observation_day0_resample real day-0 cells resampled to day-1 size —
                                 the no-model prediction
  oracle_gaussian_day1           per-gene Gaussian fit on the day-1 cells
                                 themselves; not usable for prediction, but an
                                 upper bound on what marginal moment-matching
                                 can achieve — any gap below this bound is
                                 joint structure only a generative model can
                                 carry

Output: artifacts/positive_control/positive_control_metrics.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external_runners" / "squidiff"))
sys.path.insert(0, str(REPO / "vendor" / "Squidiff"))

from load_released_checkpoint import PUBLISHED_VO_ARGS  # noqa: E402

SAMPLING_SEEDS = [13, 37, 73]
NOISE_SCALES = {"upstream_default_0.7": 0.7, "small_0.03": 0.03}
DECODE_BATCH = 1024


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


def _dense(a):
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    return np.asarray(X, dtype=np.float64)


def _build_model(device):
    from Squidiff.script_util import create_model_and_diffusion

    model, diffusion = create_model_and_diffusion(
        **PUBLISHED_VO_ARGS,
        num_channels=128,
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,
        use_drug_structure=False,
        drug_dimension=1024,
        comb_num=1,
        learn_sigma=False,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    model.to(device).eval()
    return model, diffusion


def _encode(model, mat: np.ndarray, device) -> np.ndarray:
    import torch

    chunks = []
    with torch.no_grad():
        for i in range(0, mat.shape[0], DECODE_BATCH):
            batch = torch.tensor(
                mat[i : i + DECODE_BATCH], dtype=torch.float32, device=device
            )
            z = model.encoder(batch, label=None, drug_dose=None, control_feature=None)
            chunks.append(z.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _decode(diffusion, model, z_batch: np.ndarray, gene_size: int, device) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, z_batch.shape[0], DECODE_BATCH):
            z = torch.tensor(
                z_batch[i : i + DECODE_BATCH], dtype=torch.float32, device=device
            )
            sample = diffusion.ddim_sample_loop(
                model,
                (z.shape[0], gene_size),
                model_kwargs={"z_mod": z},
                noise=None,
            )
            out.append(sample.detach().cpu().numpy())
            print(f"    decoded {min(i + DECODE_BATCH, z_batch.shape[0])}/{z_batch.shape[0]}",
                  flush=True)
    return np.concatenate(out, axis=0)


def _sample_around(point: np.ndarray, n: int, rng: np.random.RandomState, scale: float):
    """Vendored semantics: point + scale * randn(n, dim), explicit RNG."""
    return point + scale * rng.randn(n, point.shape[0])


def run(model_path: Path, adata_path: Path, output_dir: Path) -> dict:
    import anndata as ad
    import torch
    from Squidiff import dist_util

    from reuse_gate.metrics.distribution import (
        energy_distance_multivariate,
        mean_expression_correlation,
        median_pairwise_distance,
        mmd_rbf,
    )
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
    print(f"day0 {X_d0.shape}, day1 {X_d1.shape}, genes {gene_size}")

    bandwidth = median_pairwise_distance(X)
    print(f"MMD bandwidth (median heuristic, pooled training data): {bandwidth:.4f}")

    def scores(gen: np.ndarray) -> dict:
        return {
            "energy_distance": float(energy_distance_multivariate(X_d1, gen)),
            "mmd_rbf": float(mmd_rbf(X_d1, gen, bandwidth=bandwidth)),
            "mean_expression_correlation": float(mean_expression_correlation(X_d1, gen)),
            "generated_mean": float(gen.mean()),
            "generated_std": float(gen.std()),
        }

    conditions: dict = {}

    # ── Squidiff: published latent-extrapolation mechanism ──
    model, diffusion = _build_model(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    z_d0 = _encode(model, X_d0, device)
    z_d1 = _encode(model, X_d1, device)
    direction = z_d1.mean(axis=0) - z_d0.mean(axis=0)
    target = z_d0.mean(axis=0) + direction  # == mean(z_d1), one step ahead
    latent_stats = {
        "direction_norm": float(np.linalg.norm(direction)),
        "within_day1_spread": float(np.linalg.norm(z_d1.std(axis=0))),
        "noise_norm_at_0.7": float(0.7 * np.sqrt(z_d1.shape[1])),
    }
    print(f"latent: {latent_stats}")

    for label, scale in NOISE_SCALES.items():
        per_seed = []
        for seed in SAMPLING_SEEDS:
            t0 = time.time()
            rng = np.random.RandomState(seed)
            z_batch = _sample_around(target, X_d1.shape[0], rng, scale=scale)
            gen = _decode(diffusion, model, z_batch, gene_size, device)
            s = scores(gen)
            s["seed"] = seed
            s["wall_seconds"] = round(time.time() - t0, 1)
            per_seed.append(s)
            print(f"  squidiff {label} seed {seed}: ED {s['energy_distance']:.3f} "
                  f"MMD {s['mmd_rbf']:.5f} corr {s['mean_expression_correlation']:+.4f}",
                  flush=True)
        pooled = {
            k: float(np.mean([s[k] for s in per_seed]))
            for k in ("energy_distance", "mmd_rbf", "mean_expression_correlation")
        }
        conditions[f"squidiff_{label}"] = {
            **pooled,
            "fit_set": (
                "released checkpoint trained on all 6,838 VO cells (days 0+1); "
                "direction E[z_day1]-E[z_day0] from the released encoder; "
                f"noise scale {scale}"
            ),
            "per_seed": per_seed,
        }

    # ── Baselines ──
    rng = np.random.RandomState(13)

    gen = conditional_mean_sampler(X, X_d1.shape[0], rng)
    conditions["conditional_mean_pooled"] = {
        **scores(gen),
        "fit_set": "per-gene mean/variance fit on all 6,838 pooled training cells (days 0+1)",
    }
    print(f"  conditional_mean_pooled: ED {conditions['conditional_mean_pooled']['energy_distance']:.3f}")

    gen = X_d0[rng.choice(X_d0.shape[0], X_d1.shape[0], replace=True)]
    conditions["last_observation_day0_resample"] = {
        **scores(gen),
        "fit_set": f"real day-0 cells ({X_d0.shape[0]}) resampled with replacement",
    }
    print(f"  last_observation_day0:  ED {conditions['last_observation_day0_resample']['energy_distance']:.3f}")

    gen = conditional_mean_sampler(X_d1, X_d1.shape[0], rng)
    conditions["oracle_gaussian_day1"] = {
        **scores(gen),
        "fit_set": (
            "per-gene mean/variance fit on the day-1 cells themselves "
            "(oracle: not usable for prediction; upper bound for marginal moment-matching)"
        ),
    }
    print(f"  oracle_gaussian_day1:   ED {conditions['oracle_gaussian_day1']['energy_distance']:.3f}")

    result = {
        "purpose": "positive control: baselines vs Squidiff on the authors' own released setting",
        "task": "predict the day-1 VO population from the day-0 anchor via the published latent-extrapolation mechanism",
        "checkpoint": str(model_path),
        "adata": str(adata_path),
        "input_sha256": {"model": _sha256(model_path), "adata": _sha256(adata_path)},
        "git_commit": _git_commit(),
        "released_config": PUBLISHED_VO_ARGS,
        "noise_scales": NOISE_SCALES,
        "sampling_seeds": SAMPLING_SEEDS,
        "n_day0": int(X_d0.shape[0]),
        "n_day1": int(X_d1.shape[0]),
        "mmd_bandwidth": bandwidth,
        "latent_stats": latent_stats,
        "device": str(device),
        "conditions": conditions,
    }
    out = output_dir / "positive_control_metrics.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {out}")
    return result


if __name__ == "__main__":
    base = Path("data/raw/upstream_figshare")
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else base / "VO_diff_model.pt"
    adata = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "VO_trained_adata.h5ad"
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("artifacts/positive_control")
    run(model, adata, outdir)
