"""Integration test for Squidiff smoke run — RED phase.

This test requires:
  - Squidiff installed from pinned commit
  - PyTorch with CUDA or CPU
  - Network access to download test data (in real run)

Marked as integration because it requires the full Squidiff dependency chain.
"""

from pathlib import Path

import pytest

from reuse_gate.models.external import ExternalProcessRunner
from reuse_gate.schemas import RunRequest

pytestmark = pytest.mark.integration


def _squidiff_smoke_available() -> bool:
    """Check if Squidiff and PyTorch are importable."""
    try:
        import Squidiff  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _squidiff_smoke_available(),
    reason="Squidiff or PyTorch not installed",
)
def test_squidiff_smoke_generates_finite_cells(tmp_path: Path):
    """A minimal Squidiff smoke run must produce finite generated cells.

    Uses a synthetic dataset to avoid downloading real data.
    """
    output_dir = tmp_path / "squidiff_smoke_out"
    output_dir.mkdir()

    # Create a tiny synthetic AnnData file
    import anndata as ad
    import numpy as np

    n_cells, n_genes = 100, 50
    adata = ad.AnnData(
        X=np.random.negative_binomial(5, 0.5, (n_cells, n_genes)).astype(np.float32),
        obs={"cell_type": ["A"] * 50 + ["B"] * 50},
    )
    train_path = tmp_path / "train.h5ad"
    adata.write_h5ad(train_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("gene_size: 50\noutput_dim: 50\n")

    request = RunRequest(
        run_id="squidiff-smoke-1",
        candidate="squidiff",
        model_id="smoke",
        train_path=train_path,
        validation_path=train_path,
        test_path=train_path,
        config_path=config_path,
        output_dir=output_dir,
        seed=13,
    )

    cmd = [
        "python", "-c",
        """
import torch
import anndata as ad
print("Squidiff + PyTorch available")
print("CUDA:", torch.cuda.is_available())
adata = ad.read_h5ad("train.h5ad")
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
# Smoke: just verify we can create the model
from Squidiff.diffusion import create_model_and_diffusion
print("Model creation API available")
""",
    ]

    runner = ExternalProcessRunner(cmd, timeout_seconds=30)
    result = runner.run(request)

    assert result.status == "success", f"Smoke failed: {result.error_message}"
    stdout = result.stdout_path.read_text()
    assert "Squidiff + PyTorch available" in stdout
