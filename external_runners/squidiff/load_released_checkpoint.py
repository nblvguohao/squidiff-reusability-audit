"""Load the officially released Squidiff checkpoint with the official code.

Three defects already found on the conditional training path raise on the first
optimizer step, which means that path cannot have been run end to end before
release. That raises a sharper question, and this script answers it: can the
weights the authors published be loaded and sampled from using the code they
published?

Inputs come from the figshare record referenced in the Squidiff_reproducibility
README (DOI 10.6084/m9.figshare.27948633, CC BY 4.0):

  VO_diff_model.pt        released checkpoint
  VO_trained_adata.h5ad   the data it was trained on

The script deliberately makes no repairs. Every failure is recorded with its
exact exception so that "does the released artefact work as released" has an
evidenced answer either way.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "Squidiff"

# Taken verbatim from the args dict in fig4_VO_reproducibility.ipynb, the
# notebook that loads this checkpoint. class_cond is False and use_encoder is
# True, so the released configuration routes around the conditional branch that
# carries the three defects in vendor/patches/squidiff/.
PUBLISHED_VO_ARGS = {
    "gene_size": 596,
    "output_dim": 596,
    "num_layers": 3,
    "class_cond": False,
    "use_encoder": True,
    "diffusion_steps": 1000,
}

# Training settings the same notebook used, recorded for the record: the
# released model saw 2,400 steps at batch size 16.
PUBLISHED_VO_TRAINING = {"lr_anneal_steps": 2400, "batch_size": 16, "lr": 1e-4}


def _record(step: str, fn, results: dict):
    """Run one step, recording success or the full exception without repairing."""
    print(f"\n--- {step} ---")
    try:
        value = fn()
        results[step] = {"status": "ok"}
        print("  ok")
        return value
    except Exception as exc:  # noqa: BLE001
        results[step] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc().splitlines()[-6:],
        }
        print(f"  FAILED  {type(exc).__name__}: {exc}")
        return None


def inspect_checkpoint(model_path: Path) -> dict:
    """Report what the released file actually contains."""
    import torch

    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    info: dict = {"container_type": type(obj).__name__}

    state = obj
    if isinstance(obj, dict) and not all(
        hasattr(v, "shape") for v in list(obj.values())[:5]
    ):
        info["top_level_keys"] = list(obj.keys())[:20]
        for candidate in ("model_state_dict", "state_dict", "model"):
            if candidate in obj:
                state = obj[candidate]
                info["state_dict_key"] = candidate
                break

    if isinstance(state, dict):
        shapes = {k: tuple(v.shape) for k, v in state.items() if hasattr(v, "shape")}
        info["n_tensors"] = len(shapes)
        info["n_parameters"] = int(sum(np.prod(s) for s in shapes.values()))
        info["first_tensors"] = dict(list(shapes.items())[:8])

        # The two layers that pin the architecture: input width gives gene_size,
        # the label embedding reveals whether the release is class-conditional.
        for k, s in shapes.items():
            if k.endswith("input_layer.weight"):
                info["inferred_hidden_size"], info["inferred_gene_size"] = s
            if "label_embed" in k and k.endswith("weight"):
                info["has_label_embedding"] = True
                info["label_embed_shape"] = s
            if "encoder" in k:
                info["has_encoder"] = True
    return info


def run(model_path: Path, adata_path: Path, output_dir: Path, seed: int = 13) -> dict:
    sys.path.insert(0, str(VENDOR))
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "checkpoint": str(model_path),
        "adata": str(adata_path),
        "source": "figshare 10.6084/m9.figshare.27948633 (CC BY 4.0)",
        "published_model_args": PUBLISHED_VO_ARGS,
        "published_training_args": PUBLISHED_VO_TRAINING,
        "seed": seed,
    }

    print("=" * 62)
    print("STEP 1: What does the released checkpoint contain?")
    print("=" * 62)
    info = _record("inspect_checkpoint", lambda: inspect_checkpoint(model_path), results)
    if info:
        results["checkpoint_info"] = info
        for k, v in info.items():
            if k != "first_tensors":
                print(f"  {k}: {v}")

    print()
    print("=" * 62)
    print("STEP 2: What data was it trained on?")
    print("=" * 62)

    def _load_adata():
        import anndata as ad

        a = ad.read_h5ad(adata_path)
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        summary = {
            "n_obs": int(a.n_obs),
            "n_vars": int(a.n_vars),
            "obs_columns": list(a.obs.columns),
            "mean": float(X.mean()),
            "std": float(X.std()),
            "min": float(X.min()),
            "max": float(X.max()),
        }
        if "Group" in a.obs.columns:
            summary["group_values"] = sorted(
                {float(g) for g in np.unique(a.obs["Group"].to_numpy())}
            )[:12]
        return a, X, summary

    loaded = _record("load_adata", _load_adata, results)
    if loaded:
        adata, X, summary = loaded
        results["adata_summary"] = summary
        for k, v in summary.items():
            print(f"  {k}: {v}")
        # The scale settles whether the release was trained on log-normalized
        # values, which is the question our CAR-NK runs turned on.
        print(f"\n  scale verdict: {'log-normalized' if X.max() < 20 else 'raw counts'}"
              f" (max {X.max():.2f})")
        results["scale_verdict"] = "log-normalized" if X.max() < 20 else "raw counts"

    print()
    print("=" * 62)
    print("STEP 3: Load the weights into the released architecture")
    print("=" * 62)

    def _build_and_load():
        import torch
        from Squidiff.script_util import create_model_and_diffusion

        # Exactly the argument dict fig4_VO_reproducibility.ipynb uses for this
        # checkpoint, not values inferred from the tensors. Note class_cond is
        # False: the released configuration does not touch the conditional path
        # on which the three blocking defects sit.
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
        gene_size = PUBLISHED_VO_ARGS["gene_size"]
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
        state = obj
        key = results.get("checkpoint_info", {}).get("state_dict_key")
        if key:
            state = obj[key]
        missing, unexpected = model.load_state_dict(state, strict=False)
        results["load_state_dict"] = {
            "n_missing": len(missing),
            "n_unexpected": len(unexpected),
            "missing_sample": list(missing)[:8],
            "unexpected_sample": list(unexpected)[:8],
            "strict_would_pass": not missing and not unexpected,
        }
        print(f"  missing keys    : {len(missing)}")
        print(f"  unexpected keys : {len(unexpected)}")
        print(f"  strict=True would pass: {not missing and not unexpected}")
        if missing:
            print(f"    e.g. missing    {list(missing)[:3]}")
        if unexpected:
            print(f"    e.g. unexpected {list(unexpected)[:3]}")
        return model, diffusion, gene_size

    built = _record("build_and_load_weights", _build_and_load, results)

    print()
    print("=" * 62)
    print("STEP 4: Sample from the released weights")
    print("=" * 62)

    if built and loaded:
        model, diffusion, gene_size = built
        adata, X, _ = loaded

        def _sample():
            import torch
            from Squidiff import dist_util

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dist_util.dev = lambda: device
            model.to(device).eval()

            n = min(512, int(adata.n_obs))
            ref = torch.tensor(X[:n, :gene_size], dtype=torch.float32, device=device)
            with torch.no_grad():
                z_sem = model.encoder(ref, label=None, drug_dose=None, control_feature=None)
                out = diffusion.ddim_sample_loop(
                    model, (n, gene_size), model_kwargs={"z_mod": z_sem}, noise=None
                )
            gen = out.cpu().numpy()
            np.save(output_dir / "released_checkpoint_samples.npy", gen)

            from reuse_gate.metrics.distribution import energy_distance_multivariate

            ed = float(energy_distance_multivariate(X[:n, :gene_size], gen))
            results["sampling"] = {
                "n_samples": n,
                "finite": bool(np.isfinite(gen).all()),
                "generated_mean": float(gen.mean()),
                "generated_std": float(gen.std()),
                "reference_mean": float(X[:n, :gene_size].mean()),
                "reference_std": float(X[:n, :gene_size].std()),
                "energy_distance": ed,
            }
            print(f"  generated {n} cells, finite={np.isfinite(gen).all()}")
            print(f"  generated mean {gen.mean():.4f} std {gen.std():.4f}")
            print(f"  reference mean {X[:n, :gene_size].mean():.4f} "
                  f"std {X[:n, :gene_size].std():.4f}")
            print(f"  energy distance {ed:.4f}")
            return gen

        _record("sample_from_released_weights", _sample, results)
    else:
        print("  skipped: an earlier step failed")
        results["sample_from_released_weights"] = {"status": "skipped"}

    out = output_dir / "released_checkpoint_check.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {out}")

    print()
    print("=" * 62)
    failed = [k for k, v in results.items() if isinstance(v, dict) and v.get("status") == "failed"]
    print(f"VERDICT: {'all steps ran' if not failed else 'failures at: ' + ', '.join(failed)}")
    print("=" * 62)
    return results


if __name__ == "__main__":
    base = Path("data/raw/upstream_figshare")
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else base / "VO_diff_model.pt"
    adata = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "VO_trained_adata.h5ad"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("artifacts/released_checkpoint")
    run(model, adata, out)
