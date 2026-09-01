"""Training-step sweep for Squidiff.

Answers the question a reviewer will ask first: is Squidiff's underperformance
relative to the conditional-mean baseline an artefact of insufficient training?

Trains independent models at increasing step counts on the identical
train split, samples from each, and reports energy distance against the
held-out D21/D28 cells. All models share seed, architecture, data and
feature selection; only lr_anneal_steps varies.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tier0_gpu import (  # noqa: E402
    prepare_train_data,
    run_baselines,
    sample_from_model,
    train_squidiff_gpu,
)


def run_sweep(
    output_root: Path,
    step_counts: list[int],
    n_genes: int = 500,
    diffusion_steps: int = 100,
    batch_size: int = 64,
    seed: int = 13,
    log_normalize: bool = True,
) -> dict:
    """Train one model per step count and evaluate each identically."""
    from reuse_gate.metrics.distribution import energy_distance_multivariate

    output_root.mkdir(parents=True, exist_ok=True)

    # Reuse the cached combined AnnData produced by the Tier 0 GPU run.
    combined_path = output_root.parent / "squidiff_tier0_gpu" / "source_data" / "gse190976_combined.h5ad"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Combined AnnData not found at {combined_path}. "
            "Run run_tier0_gpu.py first to build it."
        )

    print("=" * 60)
    print("Preparing split (identical for every step count)")
    print("=" * 60)
    split_info = prepare_train_data(
        combined_path, output_root, n_genes=n_genes, log_normalize=log_normalize
    )

    train_mat = split_info["train_mat"]
    test_mat = split_info["test_mat"]

    print("\n" + "=" * 60)
    print("Baselines (computed once; identical across the sweep)")
    print("=" * 60)
    baselines = run_baselines(train_mat, test_mat, seed=seed)
    for name, r in baselines.items():
        print(f"  {name}: ED={r['energy_distance']:.2f}")

    results: dict = {
        "seed": seed,
        "n_genes": n_genes,
        "diffusion_steps": diffusion_steps,
        "batch_size": batch_size,
        "log_normalize": log_normalize,
        "split": {k: v for k, v in split_info.items() if k not in ("train_mat", "test_mat")},
        "baselines": baselines,
        "sweep": [],
    }

    for steps in step_counts:
        print("\n" + "=" * 60)
        print(f"TRAINING: {steps} steps")
        print("=" * 60)

        run_dir = output_root / f"steps_{steps}"
        run_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()
        model_path = run_dir / "model.pt"
        if model_path.exists():
            print(f"Model already trained: {model_path} — reusing")
            train_result = {"status": "success", "model_path": str(model_path)}
        else:
            train_result = train_squidiff_gpu(
                train_path=Path(split_info["train_path"]),
                output_dir=run_dir,
                gene_size=n_genes,
                diffusion_steps=diffusion_steps,
                lr_anneal_steps=steps,
                batch_size=batch_size,
                class_cond=True,
            )

        entry: dict = {
            "steps": steps,
            "train_seconds": train_result.get("train_time_seconds"),
            "n_params": train_result.get("n_params"),
            "final_losses": train_result.get("losses"),
        }

        try:
            generated = sample_from_model(
                model_path=Path(train_result["model_path"]),
                train_adata_path=Path(split_info["train_path"]),
                output_dir=run_dir,
                n_samples=split_info["test_n_cells"],
                gene_size=n_genes,
                seed=seed,
            )
            t_ed = time.time()
            ed = float(energy_distance_multivariate(test_mat, generated))
            entry["energy_distance"] = ed
            entry["ed_seconds"] = round(time.time() - t_ed, 1)
            entry["generated_finite"] = bool(np.isfinite(generated).all())
            entry["generated_mean"] = float(generated.mean())
            entry["generated_std"] = float(generated.std())
            entry["status"] = "success"
            print(f"\n  ED @ {steps} steps: {ed:.2f}")
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["energy_distance"] = None
            print(f"\n  Sampling failed at {steps} steps: {exc}")

        entry["wall_seconds_total"] = round(time.time() - t_start, 1)
        results["sweep"].append(entry)

        # Persist after every step count so a crash does not lose earlier runs.
        (output_root / "sweep_metrics.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

    # ── Verdict ──
    ed_cond_mean = baselines["conditional_mean"]["energy_distance"]
    ed_last_obs = baselines["last_observation"]["energy_distance"]
    successful = [e for e in results["sweep"] if e.get("energy_distance") is not None]

    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"  {'steps':>8} | {'ED':>9} | {'train s':>8} | vs cond_mean")
    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*8}-+-------------")
    for e in successful:
        delta = e["energy_distance"] - ed_cond_mean
        marker = "BEATS" if delta < 0 else f"+{delta:.0f}"
        secs = f"{e['train_seconds']:.1f}" if e["train_seconds"] is not None else "cached"
        print(f"  {e['steps']:>8} | {e['energy_distance']:>9.2f} | {secs:>8} | {marker:>12}")
    print(f"\n  conditional_mean baseline: {ed_cond_mean:.2f}")
    print(f"  last_observation baseline: {ed_last_obs:.2f}")

    if successful:
        eds = [e["energy_distance"] for e in successful]
        best = min(successful, key=lambda e: e["energy_distance"])
        monotone_improving = all(eds[i] >= eds[i + 1] for i in range(len(eds) - 1))
        results["verdict"] = {
            "best_steps": best["steps"],
            "best_ed": best["energy_distance"],
            "beats_conditional_mean": best["energy_distance"] < ed_cond_mean,
            "beats_last_observation": best["energy_distance"] < ed_last_obs,
            "monotone_improving_with_steps": monotone_improving,
            "ed_range": [min(eds), max(eds)],
        }
        print(f"\n  Best: {best['steps']} steps, ED={best['energy_distance']:.2f}")
        print(f"  Beats conditional_mean: {results['verdict']['beats_conditional_mean']}")
        print(f"  Monotone improvement with steps: {monotone_improving}")

    (output_root / "sweep_metrics.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"\nResults saved: {output_root / 'sweep_metrics.json'}")
    return results


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/squidiff_step_sweep")
    if len(sys.argv) > 2:
        steps_list = [int(s) for s in sys.argv[2].split(",")]
    else:
        steps_list = [5000, 20000, 50000]
    seed_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 13
    lognorm = sys.argv[4].lower() not in ("0", "false", "no") if len(sys.argv) > 4 else True

    run_sweep(
        output_root=out, step_counts=steps_list, seed=seed_arg, log_normalize=lognorm
    )
