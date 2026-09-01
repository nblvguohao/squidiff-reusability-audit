"""Evaluation robustness: null anchors, bootstrap CIs, MMD sensitivity, structure.

Reviewer-revision TDD, Phase 2 (tasks 2.1-2.4), one coherent pass over the
same data:

  2.1  null-distribution anchor: split the held-out population in half and
       compute the metric between halves — the "same distribution" scale.
  2.2  uncertainty beyond training seed: cell-level and sample-level
       bootstrap CIs; leave-one-held-out-sample-out re-scoring; baseline
       dispersion across resampling seeds.
  2.3  MMD bandwidth sensitivity: ordering across a bandwidth grid.
  2.4  structure metrics a diagonal-Gaussian baseline cannot win:
       gene-gene correlation Frobenius distance and rare-cluster mass recall.

Regenerates the five seed models' validation-selected populations once (GPU)
and caches them under artifacts/evaluation_robustness/generated/, so later
audits are CPU-only. Writes artifacts/evaluation_robustness/robustness.json.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external_runners" / "squidiff"))


# ── Pure helpers (unit-tested on synthetic data) ─────────────────────────────


def null_energy_distance(real: np.ndarray, n_splits: int, rng: np.random.RandomState) -> np.ndarray:
    """ED between random halves of `real`: the same-distribution reference."""
    from reuse_gate.metrics.distribution import energy_distance_multivariate

    real = np.asarray(real, dtype=np.float64)
    out = []
    for _ in range(n_splits):
        idx = rng.permutation(real.shape[0])
        half = real.shape[0] // 2
        out.append(energy_distance_multivariate(real[idx[:half]], real[idx[half : 2 * half]]))
    return np.asarray(out)


def bootstrap_metric_ci(real, gen, metric_fn, n_boot: int, rng, cluster=None, alpha: float = 0.05):
    """Percentile bootstrap CI for metric_fn(real, gen).

    With `cluster=None`, resample cells in both populations. With `cluster`
    (an array of per-cell group labels for `real`), resample clusters in
    `real` (cells of a resampled cluster move together) and cells in `gen`.
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    estimate = float(metric_fn(real, gen))
    stats = []
    if cluster is None:
        for _ in range(n_boot):
            r = real[rng.randint(0, real.shape[0], real.shape[0])]
            g = gen[rng.randint(0, gen.shape[0], gen.shape[0])]
            stats.append(float(metric_fn(r, g)))
    else:
        cluster = np.asarray(cluster)
        labels, inverse = np.unique(cluster, return_inverse=True)
        for _ in range(n_boot):
            picked = rng.randint(0, len(labels), len(labels))
            # Rebuild the real sample cluster-by-cluster (with replacement).
            rows = [real[inverse == p] for p in picked]
            r = np.concatenate(rows, axis=0)
            g = gen[rng.randint(0, gen.shape[0], gen.shape[0])]
            stats.append(float(metric_fn(r, g)))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return estimate, float(lo), float(hi)


def correlation_frobenius_distance(real: np.ndarray, gen: np.ndarray) -> float:
    """Frobenius distance between the two gene-gene correlation matrices.

    A diagonal (per-gene independent) sampler has zero off-diagonal
    covariance by construction, so it cannot win this metric on data with
    real gene-gene correlation.

    Genes with zero variance in `real` are dropped before computing either
    matrix: Pearson correlation is undefined for a constant variable, so
    `np.corrcoef` returns NaN for that gene's row and column regardless of
    `gen`, which would silently turn the whole score into NaN. The released
    VO data has two such genes (exactly zero across the entire held-out
    population) out of 596; CAR-NK's HVG-selected genes have none.
    """
    from reuse_gate.metrics.structure import correlation_frobenius

    return correlation_frobenius(real, gen, normalized=False)


def cluster_mass_recall(real, gen, n_clusters: int, rare_below: float, rng) -> float:
    """Fraction of rare-cluster real cells whose cluster is covered by `gen`.

    Clusters are k-means fitted on `real`. A cluster is "rare" if it holds
    less than `rare_below` of the real cells, and "covered" if at least one
    generated cell is assigned to it (assignment by nearest centroid).
    """
    from sklearn.cluster import KMeans

    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(real)
    real_labels = km.labels_
    gen_labels = km.predict(gen)
    mass = np.bincount(real_labels, minlength=n_clusters) / real.shape[0]
    rare = np.where(mass < rare_below)[0]
    if rare.size == 0:
        return 1.0
    covered = set(np.unique(gen_labels))
    hits = sum(mass[c] for c in rare if c in covered)
    total = float(mass[rare].sum())
    return float(hits / total) if total > 0 else 1.0


def evaluate_structure_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    cluster_counts: tuple[int, ...] = (6, 8, 10, 12),
    rare_thresholds: tuple[float, ...] = (0.05, 0.10, 0.15),
    random_state: int = 13,
) -> dict:
    """Return pre-specified correlation and cluster-mass structure metrics."""
    from reuse_gate.metrics.structure import (
        cluster_mass_metrics,
        correlation_frobenius,
        structure_sensitivity_grid,
    )

    primary_cluster_count = 8 if 8 in cluster_counts else cluster_counts[0]
    primary_rare_threshold = 0.10 if 0.10 in rare_thresholds else rare_thresholds[0]
    return {
        "correlation_frobenius": {
            "raw": correlation_frobenius(real, generated, normalized=False),
            "normalized": correlation_frobenius(real, generated, normalized=True),
        },
        "cluster_mass": cluster_mass_metrics(
            real,
            generated,
            n_clusters=primary_cluster_count,
            rare_below=primary_rare_threshold,
            random_state=random_state,
        ),
        "cluster_mass_sensitivity": structure_sensitivity_grid(
            real,
            generated,
            cluster_counts=cluster_counts,
            rare_thresholds=rare_thresholds,
            random_state=random_state,
        ),
    }


def _bootstrap_one(task: tuple) -> tuple:
    """One (population, mode) bootstrap. Top-level so spawn workers can pickle it.

    Each task builds its own RandomState from the shared seed, so a task's
    numbers are independent of how many siblings run alongside it.
    """
    from reuse_gate.metrics.distribution import energy_distance_multivariate

    name, mode, real, gen, cluster, n_boot, seed = task
    est, lo, hi = bootstrap_metric_ci(
        real, gen, energy_distance_multivariate,
        n_boot=n_boot, rng=np.random.RandomState(seed), cluster=cluster,
    )
    return name, mode, est, lo, hi


def parallel_bootstrap(populations, real, cluster, n_boot: int, seed: int,
                       processes: int | None = None) -> dict:
    """Bootstrap ED CIs for every population, cell- and cluster-level.

    The (population × mode) tasks are independent and each re-seeds its own
    RandomState, so the result is identical to the sequential loop it
    replaces; only wall-clock changes. `processes=1` stays in-process (used
    by the determinism test); otherwise a spawn Pool fans the tasks out.
    """
    tasks = []
    for name, gen in populations.items():
        tasks.append((name, "cell", real, gen, None, n_boot, seed))
        tasks.append((name, "sample", real, gen, cluster, n_boot, seed))

    if processes is None:
        processes = min(8, os.cpu_count() or 1)
    if processes <= 1:
        results = [_bootstrap_one(t) for t in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(processes=processes) as pool:
            results = pool.map(_bootstrap_one, tasks)

    out: dict = {}
    for name, mode, est, lo, hi in results:
        entry = out.setdefault(name, {}).setdefault("energy_distance", {})
        entry["estimate"] = est
        entry["cell_ci" if mode == "cell" else "sample_ci"] = [lo, hi]
    return out


# ── Provenance ───────────────────────────────────────────────────────────────


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


# ── Population (re)generation ────────────────────────────────────────────────


def regenerate_populations(sweep_dir: Path, seed_study_dir: Path, out_dir: Path) -> dict:
    """Re-decode each seed model's validation-selected population once."""
    import anndata as ad
    import torch
    from carnk_latent_extrapolation import _dense, _encode, build_model

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ad = ad.read_h5ad(sweep_dir / "train.h5ad")
    train_mat, train_tp = _dense(train_ad), train_ad.obs["timepoint_numeric"].to_numpy()
    test_ad = ad.read_h5ad(sweep_dir / "test.h5ad")
    test_tp = test_ad.obs["timepoint_numeric"].to_numpy()
    gene_size = train_mat.shape[1]

    metrics = json.loads((seed_study_dir / "seed_study_metrics.json").read_text())
    paths = {}
    for entry in metrics["per_seed"]:
        seed = entry["seed"]
        scale = entry["selected_scale"]
        out = out_dir / f"seed_{seed}_scale_{scale}.npy"
        if out.exists():
            print(f"seed {seed}: cached {out.name}")
            paths[str(seed)] = str(out)
            continue
        model, diffusion = build_model(gene_size, device)
        state = torch.load(seed_study_dir / f"seed_{seed}" / "model.pt",
                           map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()

        z_d7 = _encode(model, train_mat[train_tp == 7], device)
        z_d14 = _encode(model, train_mat[train_tp == 14], device)
        direction = z_d14.mean(axis=0) - z_d7.mean(axis=0)
        anchor = z_d14.mean(axis=0)

        rng = np.random.RandomState(seed)
        batches = []
        for tp in sorted(np.unique(test_tp)):
            steps = (int(tp) - 14) / 7.0
            n = int((test_tp == tp).sum())
            z_target = anchor + direction * steps
            z_batch = (np.tile(z_target, (n, 1)) if scale == 0.0
                       else z_target + scale * rng.randn(n, z_target.shape[0]))
            batches.append(z_batch)
        z_all = np.concatenate(batches, axis=0)

        gen = []
        with torch.no_grad():
            for i in range(0, z_all.shape[0], 1024):
                z = torch.tensor(z_all[i : i + 1024], dtype=torch.float32, device=device)
                s = diffusion.ddim_sample_loop(
                    model, (z.shape[0], gene_size),
                    model_kwargs={"z_mod": z}, noise=None)
                gen.append(s.cpu().numpy())
        np.save(out, np.concatenate(gen, axis=0))
        print(f"seed {seed}: generated {z_all.shape[0]} cells at scale {scale}")
        paths[str(seed)] = str(out)
    return paths


# ── Main ─────────────────────────────────────────────────────────────────────


def run(sweep_dir: Path, seed_study_dir: Path, output_dir: Path) -> dict:
    import anndata as ad

    from reuse_gate.metrics.distribution import (
        energy_distance_multivariate,
        mean_expression_correlation,
        mmd_rbf,
    )
    from reuse_gate.metrics.structure import same_distribution_structure_reference
    from reuse_gate.models.temporal_baselines import conditional_mean_sampler

    output_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = output_dir / "generated"

    train_ad = ad.read_h5ad(sweep_dir / "train.h5ad")
    test_ad = ad.read_h5ad(sweep_dir / "test.h5ad")
    train_mat = np.asarray(train_ad.X.todense() if hasattr(train_ad.X, "todense") else train_ad.X)
    test_mat = np.asarray(test_ad.X.todense() if hasattr(test_ad.X, "todense") else test_ad.X)
    samples = test_ad.obs["sample_id"].to_numpy()

    metrics = json.loads((seed_study_dir / "seed_study_metrics.json").read_text())
    bandwidth = float(metrics["mmd_bandwidth"])
    selected = {str(e["seed"]): e["selected_scale"] for e in metrics["per_seed"]}

    populations = {
        "conditional_mean": conditional_mean_sampler(train_mat, test_mat.shape[0], np.random.RandomState(13)),
    }
    d14 = train_mat[train_ad.obs["timepoint_numeric"].to_numpy() == 14]
    populations["last_observation_d14"] = d14[
        np.random.RandomState(13).choice(d14.shape[0], test_mat.shape[0], replace=True)
    ]
    for seed, path in regenerate_populations(sweep_dir, seed_study_dir, gen_dir).items():
        populations[f"squidiff_seed_{seed}"] = np.load(path)

    result: dict = {
        "purpose": "evaluation robustness (TDD Phase 2.1-2.4)",
        "git_commit": _git_commit(),
        "input_sha256": {
            "train_h5ad": _sha256(sweep_dir / "train.h5ad"),
            "test_h5ad": _sha256(sweep_dir / "test.h5ad"),
        },
        "mmd_bandwidth_train": bandwidth,
        "selected_scales": selected,
    }

    rng = np.random.RandomState(13)

    # ── 2.1 null anchors ──
    print("2.1 null anchors ...")
    null_ed = null_energy_distance(test_mat, n_splits=50, rng=np.random.RandomState(7))
    null_mmd = []
    null_corr = []
    for _ in range(50):
        idx = rng.permutation(test_mat.shape[0])
        half = test_mat.shape[0] // 2
        a, b = test_mat[idx[:half]], test_mat[idx[half : 2 * half]]
        null_mmd.append(mmd_rbf(a, b, bandwidth=bandwidth))
        null_corr.append(mean_expression_correlation(a, b))
    result["null_anchor"] = {
        "energy_distance": {"mean": float(np.mean(null_ed)),
                            "q95": float(np.quantile(null_ed, 0.95))},
        "mmd_rbf": {"mean": float(np.mean(null_mmd)),
                    "q95": float(np.quantile(null_mmd, 0.95))},
        "mean_expression_correlation": {"mean": float(np.mean(null_corr)),
                                        "q05": float(np.quantile(null_corr, 0.05))},
        "correlation_structure": same_distribution_structure_reference(
            test_mat,
            n_splits=50,
            rng=np.random.RandomState(7),
        ),
        "note": "metric between random halves of the held-out population; "
                "the same-distribution reference band",
    }

    # ── 2.2 bootstrap CIs + leave-one-sample-out + baseline dispersion ──
    print("2.2 bootstrap CIs ...")
    B = 200
    result["bootstrap"] = parallel_bootstrap(populations, test_mat, samples, n_boot=B, seed=21)

    print("2.2 leave-one-sample-out ...")
    loso = {}
    for name, gen in populations.items():
        folds = {}
        for s in np.unique(samples):
            mask = samples != s
            folds[str(s)] = float(energy_distance_multivariate(test_mat[mask], gen))
        loso[name] = folds
    result["leave_one_sample_out"] = loso

    print("2.2 baseline dispersion ...")
    disp = {}
    for rep in range(10):
        g = conditional_mean_sampler(train_mat, test_mat.shape[0], np.random.RandomState(100 + rep))
        disp.setdefault("conditional_mean", []).append(float(energy_distance_multivariate(test_mat, g)))
        g = d14[np.random.RandomState(100 + rep).choice(d14.shape[0], test_mat.shape[0], replace=True)]
        disp.setdefault("last_observation_d14", []).append(float(energy_distance_multivariate(test_mat, g)))
    result["baseline_dispersion"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v}
        for k, v in disp.items()
    }

    # ── 2.3 MMD bandwidth sensitivity ──
    print("2.3 MMD bandwidth sensitivity ...")
    grid = [bandwidth * f for f in (0.25, 0.5, 1.0, 2.0, 4.0)]
    sens = {}
    for name, gen in populations.items():
        sens[name] = {f"{bw:.2f}": float(mmd_rbf(test_mat, gen, bandwidth=bw)) for bw in grid}
    result["mmd_bandwidth_sensitivity"] = {"grid": grid, "values": sens}

    # ── 2.4 structure metrics ──
    print("2.4 structure metrics ...")
    structure = {}
    for name, gen in populations.items():
        structure[name] = evaluate_structure_metrics(test_mat, gen)
    result["structure"] = structure

    out = output_dir / "robustness.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {out}")
    return result


if __name__ == "__main__":
    sweep = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/squidiff_sweep_lognorm")
    seeds = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/squidiff_seed_study")
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("artifacts/evaluation_robustness")
    run(sweep, seeds, outdir)
