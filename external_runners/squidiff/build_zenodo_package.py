"""Build a curated data deposit for Zenodo.

`artifacts/` holds everything every experiment produced, including per-step
optimizer checkpoints (`training_logs/opt*.pt`, hundreds of megabytes each,
saved every ~1,000-5,000 steps purely so a crashed run could resume) and a
duplicate EMA copy of every model. None of that is needed to reproduce a
number in the manuscript, and shipping it would turn a ~4 GB deposit into
several hundred GB.

This script copies only what a published number, figure, or table traces
back to (per manuscript/RESULTS.md), renamed into a structure that mirrors
the manuscript's own section order rather than the ad hoc experiment names
under artifacts/. It also writes a manifest that maps every included file to
the claim it supports.

Two h5ad pairs are deliberately deduplicated after verifying byte-identical
content (raw split: squidiff_step_sweep vs squidiff_latent_extrap_ab/raw_split;
both are the same deterministic split of the same source data through the same
code path). The two copies of the combined source AnnData under
squidiff_tier0/ and squidiff_tier0_gpu/ are NOT identical (verified by hash);
only the squidiff_tier0_gpu copy is included, because that is the one every
downstream script actually reads.

Run once, from the repository root:
    python external_runners/squidiff/build_zenodo_package.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
PACKAGE = REPO_ROOT / "zenodo_package"

# (source relative to artifacts/, destination relative to zenodo_package/)
FILES: list[tuple[str, str]] = [
    # ── Source data: the single combined AnnData every split derives from ──
    ("squidiff_tier0_gpu/source_data/gse190976_combined.h5ad",
     "01_source_data/gse190976_combined.h5ad"),

    # ── Positive control: released checkpoint verification ──
    ("released_checkpoint/released_checkpoint_check.json",
     "02_positive_control/released_checkpoint_check.json"),
    ("released_checkpoint/released_checkpoint_samples.npy",
     "02_positive_control/released_checkpoint_samples.npy"),

    # ── Upstream simulated benchmark reproduction ──
    ("squidiff_reproduction/simulated_train.h5ad",
     "03_simulated_benchmark/simulated_train.h5ad"),
    ("squidiff_reproduction/generated_class0.npy",
     "03_simulated_benchmark/generated_class0.npy"),
    ("squidiff_reproduction/generated_class1.npy",
     "03_simulated_benchmark/generated_class1.npy"),
    ("squidiff_reproduction/generated_class2.npy",
     "03_simulated_benchmark/generated_class2.npy"),
    ("squidiff_reproduction/model.pt",
     "03_simulated_benchmark/model.pt"),
    ("squidiff_reproduction/reproduction_metrics.json",
     "03_simulated_benchmark/reproduction_metrics.json"),

    # ── CAR-NK temporal splits (canonical copy of each preprocessing condition) ──
    ("squidiff_step_sweep/train.h5ad", "04_splits/raw/train.h5ad"),
    ("squidiff_step_sweep/test.h5ad", "04_splits/raw/test.h5ad"),
    ("squidiff_sweep_lognorm/train.h5ad", "04_splits/lognorm/train.h5ad"),
    ("squidiff_sweep_lognorm/test.h5ad", "04_splits/lognorm/test.h5ad"),

    # ── Barrier 1: preprocessing sweep, class-conditional probe (Fig. 3a) ──
    ("squidiff_step_sweep/steps_5000/model.pt", "05_barrier1_probe/raw/steps_5000/model.pt"),
    ("squidiff_step_sweep/steps_20000/model.pt", "05_barrier1_probe/raw/steps_20000/model.pt"),
    ("squidiff_step_sweep/steps_50000/model.pt", "05_barrier1_probe/raw/steps_50000/model.pt"),
    ("squidiff_step_sweep/sweep_metrics.json", "05_barrier1_probe/raw/sweep_metrics.json"),
    ("squidiff_step_sweep/calibration_metrics.json", "05_barrier1_probe/raw/calibration_metrics.json"),
    ("squidiff_sweep_lognorm/steps_5000/model.pt", "05_barrier1_probe/lognorm/steps_5000/model.pt"),
    ("squidiff_sweep_lognorm/steps_20000/model.pt", "05_barrier1_probe/lognorm/steps_20000/model.pt"),
    ("squidiff_sweep_lognorm/steps_50000/model.pt", "05_barrier1_probe/lognorm/steps_50000/model.pt"),
    ("squidiff_sweep_lognorm/sweep_metrics.json", "05_barrier1_probe/lognorm/sweep_metrics.json"),
    ("squidiff_sweep_lognorm/calibration_metrics.json", "05_barrier1_probe/lognorm/calibration_metrics.json"),

    # ── Barrier 1 confirmed under the published protocol ──
    ("squidiff_latent_extrap_ab/raw/steps_5000/model.pt", "06_barrier1_published_protocol/raw/steps_5000/model.pt"),
    ("squidiff_latent_extrap_ab/raw/steps_20000/model.pt", "06_barrier1_published_protocol/raw/steps_20000/model.pt"),
    ("squidiff_latent_extrap_ab/raw/steps_50000/model.pt", "06_barrier1_published_protocol/raw/steps_50000/model.pt"),
    ("squidiff_latent_extrap_ab/preprocessing_ab_metrics.json",
     "06_barrier1_published_protocol/preprocessing_ab_metrics.json"),

    # ── Published-protocol budget sweep, log-normalized (also feeds Fig. 3b) ──
    ("squidiff_latent_extrap/steps_5000/model.pt", "07_latent_extrapolation_budget_sweep/steps_5000/model.pt"),
    ("squidiff_latent_extrap/steps_20000/model.pt", "07_latent_extrapolation_budget_sweep/steps_20000/model.pt"),
    ("squidiff_latent_extrap/steps_50000/model.pt", "07_latent_extrapolation_budget_sweep/steps_50000/model.pt"),
    ("squidiff_latent_extrap/latent_extrapolation_metrics.json",
     "07_latent_extrapolation_budget_sweep/latent_extrapolation_metrics.json"),

    # ── Barrier 3: noise-scale sweep (Fig. 3b), reuses steps_50000 above ──
    ("squidiff_latent_extrap/latent_noise_scale_sweep.json",
     "08_barrier3_noise_scale/latent_noise_scale_sweep.json"),

    # ── Performance under the published protocol, five seeds (Fig. 3c) ──
    ("squidiff_seed_study/seed_13/model.pt", "09_seed_study/seed_13/model.pt"),
    ("squidiff_seed_study/seed_37/model.pt", "09_seed_study/seed_37/model.pt"),
    ("squidiff_seed_study/seed_73/model.pt", "09_seed_study/seed_73/model.pt"),
    ("squidiff_seed_study/seed_101/model.pt", "09_seed_study/seed_101/model.pt"),
    ("squidiff_seed_study/seed_137/model.pt", "09_seed_study/seed_137/model.pt"),
    ("squidiff_seed_study/seed_study_metrics.json", "09_seed_study/seed_study_metrics.json"),

    # ── Manuscript figures and their source data ──
    ("manuscript_figures", "10_manuscript_figures"),

    # ── Positive control on the authors' own released setting (Supp. Notes 8, 11) ──
    ("positive_control/positive_control_metrics.json",
     "11_vo_positive_control/positive_control_metrics.json"),
    ("positive_control/structure_metrics.json",
     "11_vo_positive_control/structure_metrics.json"),

    # ── Baseline fit-set provenance and energy-distance decomposition (Supp. Note 7) ──
    ("baseline_provenance/baseline_provenance.json",
     "12_baseline_provenance/baseline_provenance.json"),

    # ── Phase 2: null anchors, bootstrap CIs, LOSO, MMD grid, structure (Supp. Notes 9-10) ──
    ("evaluation_robustness/robustness.json",
     "13_evaluation_robustness/robustness.json"),
    ("evaluation_robustness/generated", "13_evaluation_robustness/generated"),
]

MANIFEST_HEADER = """\
# Zenodo data deposit — Squidiff Reusability Report

Companion data to the code at
https://github.com/nblvguohao/biomed-reusability-gate (this deposit's DOI is
cross-referenced from the GitHub release; see the repository's README).

Every file below is named after the manuscript section it supports, not the
internal experiment name used while running it. `manuscript/RESULTS.md` in
the code repository is the index of record: every published number names the
script that produced it, and every one of those scripts reads or writes a
path under here.

Excluded by design: per-step optimizer checkpoints (`training_logs/opt*.pt`)
and the redundant EMA copy of every model. Neither is needed to reproduce a
published number; each `model.pt` is the exact final weights used for the
generation and metrics reported in the paper. The released Squidiff
checkpoint itself is not re-hosted here — it is the authors' own artefact,
cited via its original figshare DOI (10.6084/m9.figshare.27948633).

## Structure

| Folder | Manuscript section | Producing script |
|---|---|---|
| `01_source_data/` | Methods (data source) | `run_tier0_gpu.py::ensure_combined_adata` |
| `02_positive_control/` | "The released artefacts reproduce" | `load_released_checkpoint.py` |
| `03_simulated_benchmark/` | "The provided benchmark cannot catch this" | `reproduce_upstream_simulated.py` |
| `04_splits/` | CAR-NK temporal split (raw and log-normalized) | `run_tier0_gpu.py::prepare_train_data` |
| `05_barrier1_probe/` | Barrier 1, Fig. 3a | `train_step_sweep.py` |
| `06_barrier1_published_protocol/` | Barrier 1, confirmed under published protocol | `latent_extrapolation_preprocessing_ab.py` |
| `07_latent_extrapolation_budget_sweep/` | Feeds Barrier 3 and the confirmation above | `carnk_latent_extrapolation.py` |
| `08_barrier3_noise_scale/` | Barrier 3, Fig. 3b | `latent_noise_scale_sweep.py` |
| `09_seed_study/` | Performance under the published protocol, Fig. 3c | `seed_study.py` |
| `10_manuscript_figures/` | Figures 1-3 and their source data | `make_manuscript_figures.py` |
| `11_vo_positive_control/` | "The same ordering holds on the authors' own released setting"; Supp. Notes 8, 11 | `positive_control.py`, `positive_control_structure.py` |
| `12_baseline_provenance/` | Baseline fit sets and the energy-distance decomposition; Supp. Note 7 | `baseline_provenance.py` |
| `13_evaluation_robustness/` | Null anchors, bootstrap CIs, leave-one-sample-out, MMD bandwidth grid, structure metrics; Supp. Notes 9, 10 | `evaluation_robustness.py` |

Note on naming: `02_positive_control/` predates the term's later, narrower use
in the manuscript — it holds the released-checkpoint verification ("The
released artefacts reproduce"), not the baseline comparison on the authors'
own VO setting. That analysis is `11_vo_positive_control/`.

## Regenerating from scratch

Every file here is reproducible from the code repository and the public
GSE190976 accession; nothing here is a primary data source in its own right.
`06_barrier1_published_protocol/` reuses the log-normalized budget sweep at
`07_latent_extrapolation_budget_sweep/` for its log-normalized arm rather than
duplicating it (see `latent_extrapolation_preprocessing_ab.py`).
`13_evaluation_robustness/generated/` caches the five seed models'
validation-selected populations, re-decoded once so later robustness checks
are CPU-only; see `evaluation_robustness.py::regenerate_populations`.

"""


def build(dry_run: bool = False) -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    missing: list[str] = []

    for src_rel, dst_rel in FILES:
        src = ARTIFACTS / src_rel
        dst = PACKAGE / dst_rel
        if not src.exists():
            missing.append(src_rel)
            continue

        if dry_run:
            size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) if src.is_dir() else src.stat().st_size
            total_bytes += size
            print(f"  {size / 1e6:8.1f} MB  {dst_rel}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) if dst.is_dir() else dst.stat().st_size
        total_bytes += size
        print(f"  copied {dst_rel} ({size / 1e6:.1f} MB)")

    if missing:
        print("\nMISSING (skipped):")
        for m in missing:
            print(f"  {m}")

    if not dry_run:
        (PACKAGE / "README.md").write_text(MANIFEST_HEADER, encoding="utf-8")

    print(f"\nTotal: {total_bytes / 1e9:.2f} GB across {len(FILES) - len(missing)} entries")
    if dry_run:
        print("(dry run — nothing copied)")
    else:
        print(f"Package staged at: {PACKAGE}")


if __name__ == "__main__":
    import sys
    build(dry_run="--dry-run" in sys.argv)
