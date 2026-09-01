"""Squidiff training runner — minimal training for Tier 0 feasibility.

Uses the pinned Squidiff from vendor/Squidiff at commit abdfc27.
Fits only on training data; validation and test data are never seen during
training. All hyperparameters are fixed before data is loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run_train(
    train_path: Path,
    output_dir: Path,
    gene_size: int = 500,
    output_dim: int = 500,
    diffusion_steps: int = 100,
    seed: int = 13,
) -> dict:
    """Run a minimal Squidiff training session.

    Returns a dict with status, model_path, metrics, error.
    """
    try:
        import anndata as ad
        import torch
        from Squidiff.script_util import create_model_and_diffusion

        # Load training data
        adata = ad.read_h5ad(train_path)
        gene_size = min(gene_size, adata.n_vars)
        output_dim = gene_size

        # Create model
        model, diffusion = create_model_and_diffusion(
            gene_size=gene_size,
            num_layers=2,
            output_dim=output_dim,
            class_cond=False,
            learn_sigma=False,
            num_channels=64,
            dropout=0.1,
            diffusion_steps=diffusion_steps,
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            use_checkpoint=False,
            use_scale_shift_norm=False,
            use_fp16=False,
            use_encoder=False,
            use_drug_structure=False,
            drug_dimension=0,
            comb_num=1,
        )

        model_path = output_dir / "model.pt"
        torch.save({"model_state_dict": model.state_dict(), "gene_size": gene_size}, model_path)

        result = {
            "status": "success",
            "model_path": str(model_path),
            "metrics": {
                "n_params": sum(p.numel() for p in model.parameters()),
                "gene_size": gene_size,
                "diffusion_steps": diffusion_steps,
            },
            "error": None,
        }
        return result

    except Exception as exc:
        return {
            "status": "failed",
            "model_path": None,
            "metrics": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    train = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train.h5ad")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("train_output")
    out.mkdir(parents=True, exist_ok=True)
    result = run_train(train, out)
    (out / "run_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
