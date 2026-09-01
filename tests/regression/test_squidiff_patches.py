"""Regression tests for the Squidiff compatibility patches.

Each test guards one patch in `vendor/patches/squidiff/`. Reverting the
corresponding patch must turn the test red.

The dtype and rank patches are verified functionally on CPU. The device patch
cannot fail on a CPU-only host — host and compute device coincide — so it is
guarded by a source-level assertion plus a GPU-marked training step.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

VENDOR_SQUIDIFF = Path(__file__).resolve().parents[2] / "vendor" / "Squidiff"


def _squidiff_importable() -> bool:
    if not VENDOR_SQUIDIFF.exists():
        return False
    if str(VENDOR_SQUIDIFF) not in sys.path:
        sys.path.insert(0, str(VENDOR_SQUIDIFF))
    try:
        import Squidiff  # noqa: F401

        return True
    except ImportError:
        return False


requires_squidiff = pytest.mark.skipif(
    not _squidiff_importable(),
    reason="vendored Squidiff or one of its dependencies is unavailable",
)

# Source-level checks need the upstream tree on disk but not its dependencies,
# so they take a weaker guard than `requires_squidiff`.
requires_squidiff_source = pytest.mark.skipif(
    not (VENDOR_SQUIDIFF / "Squidiff" / "train_util.py").exists(),
    reason="vendored Squidiff source tree is not checked out",
)


def _tiny_adata(n_cells: int = 8, n_genes: int = 6):
    """Minimal AnnData carrying integer `Group` labels, as the loader expects."""
    import anndata as ad
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(13)
    return ad.AnnData(
        X=rng.rand(n_cells, n_genes).astype(np.float32),
        obs=pd.DataFrame(
            {"Group": np.arange(n_cells) % 3},  # int64, as pandas yields
            index=[f"c{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )


# ── Patch 0001: conditioning label dtype ──────────────────────────────────────


@requires_squidiff
def test_group_labels_leave_dataset_as_float_tensor():
    """Integer `Group` labels must be converted to a float tensor.

    Without patch 0001 the labels stay an int64 NumPy array and the first
    `label_embed` call fails with "mat1 and mat2 must have the same dtype".
    """
    import torch
    from Squidiff.scrna_datasets import AnnDataDataset

    dataset = AnnDataDataset(_tiny_adata(), control_adata=None, use_drug_structure=False)

    assert isinstance(dataset.encoded_obs_tensor, torch.Tensor), (
        "Group labels must be a torch.Tensor, not a raw NumPy array"
    )
    assert dataset.encoded_obs_tensor.dtype == torch.float32


@requires_squidiff
def test_collated_group_batch_is_float():
    """The dtype must survive DataLoader collation — that is where it is consumed."""
    import torch
    from Squidiff.scrna_datasets import AnnDataDataset
    from torch.utils.data import DataLoader

    dataset = AnnDataDataset(_tiny_adata(), control_adata=None, use_drug_structure=False)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))

    assert batch["group"].dtype == torch.float32
    assert batch["feature"].dtype == torch.float32


# ── Patch 0003: label embedding rank ──────────────────────────────────────────


@requires_squidiff
def test_encoder_accepts_rank1_label():
    """`EncoderMLPModel` must accept the rank-1 labels the DataLoader produces.

    Without patch 0003 this raises "mat1 and mat2 shapes cannot be multiplied".
    """
    import torch
    from Squidiff.MLPModel import EncoderMLPModel

    batch, input_size, hidden = 4, 6, 32
    encoder = EncoderMLPModel(input_size, hidden, num_classes=3, output_size=16)

    x_start = torch.randn(batch, input_size)
    label = torch.zeros(batch, dtype=torch.float32)  # rank 1, as collated
    assert label.dim() == 1

    out = encoder(x_start, label=label)

    assert out.shape == (batch, 16)
    assert torch.isfinite(out).all()


@requires_squidiff
def test_encoder_still_accepts_rank2_label():
    """The rank fix must not break callers that already pass rank-2 labels."""
    import torch
    from Squidiff.MLPModel import EncoderMLPModel

    batch, input_size, hidden = 4, 6, 32
    encoder = EncoderMLPModel(input_size, hidden, num_classes=3, output_size=16)

    out = encoder(torch.randn(batch, input_size), label=torch.zeros(batch, 1))

    assert out.shape == (batch, 16)
    assert torch.isfinite(out).all()


@requires_squidiff
def test_encoder_label_is_not_mutated_for_caller():
    """Unsqueezing happens on a local binding; the caller's tensor keeps its rank."""
    import torch
    from Squidiff.MLPModel import EncoderMLPModel

    encoder = EncoderMLPModel(6, 32, num_classes=3, output_size=16)
    label = torch.zeros(4, dtype=torch.float32)

    encoder(torch.randn(4, 6), label=label)

    assert label.dim() == 1, "caller's label tensor must not be reshaped in place"


# ── Patch 0002: conditioning label device placement ───────────────────────────


@requires_squidiff_source
def test_train_util_moves_group_labels_to_device():
    """Both conditioning branches must move `group` onto the compute device.

    A CPU-only host cannot surface this defect, so the guard is source-level.
    `forward_backward` builds `micro_cond` twice — once for the drug-structure
    branch, once without — and both must transfer the labels.
    """
    source = (VENDOR_SQUIDIFF / "Squidiff" / "train_util.py").read_text(encoding="utf-8")

    transfers = re.findall(
        r"['\"]group['\"]\s*:\s*batch\[['\"]group['\"]\][^\n]*\.to\(\s*dist_util\.dev\(\)\s*\)",
        source,
    )
    untransferred = re.findall(
        r"['\"]group['\"]\s*:\s*batch\[['\"]group['\"]\]\[[^\]]*\]\s*,",
        source,
    )

    assert len(transfers) == 2, (
        f"expected both micro_cond branches to move group to the device, found {len(transfers)}"
    )
    assert not untransferred, (
        "found a micro_cond branch that passes group without a device transfer"
    )


@pytest.mark.gpu
@pytest.mark.slow
@requires_squidiff
def test_training_step_runs_on_gpu(tmp_path: Path):
    """One real optimizer step on GPU, exercising all three patches together.

    Reverting any one patch makes this fail on the first step. Skipped when no
    CUDA device is present, since the device defect is unreachable on CPU.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    from Squidiff import dist_util, logger
    from Squidiff.resample import create_named_schedule_sampler
    from Squidiff.script_util import create_model_and_diffusion
    from Squidiff.scrna_datasets import prepared_data
    from Squidiff.train_util import TrainLoop

    device = torch.device("cuda")
    dist_util.dev = lambda: device

    n_genes = 6
    train_path = tmp_path / "train.h5ad"
    _tiny_adata(n_cells=8, n_genes=n_genes).write_h5ad(train_path)

    logger.configure(dir=str(tmp_path / "logs"))

    model, diffusion = create_model_and_diffusion(
        gene_size=n_genes,
        output_dim=n_genes,
        num_layers=1,
        num_channels=16,
        dropout=0.0,
        class_cond=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,
        use_encoder=True,
        use_drug_structure=False,
        drug_dimension=0,
        comb_num=1,
        learn_sigma=False,
        diffusion_steps=10,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    model.to(device)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=prepared_data(data_dir=str(train_path), batch_size=4),
        batch_size=4,
        microbatch=4,
        lr=1e-4,
        ema_rate="0.9999",
        log_interval=1,
        save_interval=1,
        resume_checkpoint=str(checkpoint_dir),
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=create_named_schedule_sampler("uniform", diffusion),
        weight_decay=0.0,
        lr_anneal_steps=1,
        use_drug_structure=False,
        comb_num=1,
    )

    loop.run_loop()

    assert loop.loss_list, "training produced no loss values"
    assert torch.isfinite(loop.loss_list[-1]).all(), "training loss is not finite"
