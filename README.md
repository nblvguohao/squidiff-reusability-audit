# Squidiff reusability audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22234143.svg)](https://doi.org/10.5281/zenodo.22234143)

Audit code, corrective patches and frozen numerical evidence for a leakage-safe
reusability assessment of [Squidiff](https://doi.org/10.1038/s41592-025-02877-y),
a diffusion model for predicting cellular development and responses to
perturbations (He et al., *Nature Methods* **23**, 65–77, 2026).

This repository accompanies a Reusability Report. It contains what is needed to
re-run the audit and to check every number in it. It does not contain the report
text.

## What the audit found

The released Squidiff artefacts are operable: the checkpoint loads with zero
missing and zero unexpected keys, and sampling produces finite values. Reuse on
an independent dataset then exposed three boundaries that a successful sampling
command does not reveal.

1. **An implicit data-scale requirement.** The released training script accepts
   an expression matrix but neither applies nor validates the library-size
   normalisation and log transformation that the released data already carries.
   Training longer on raw counts made predictions worse while the optimisation
   loss fell — energy distance rose from 376.76 at 5,000 steps to 561.72 at
   50,000. Applying `normalize_total(target_sum=10_000)` then `log1p` reversed
   it: 312.45 to 27.68. A falling training loss cannot diagnose the omission.

2. **Three faults in the label-conditional interface.** With the encoder and
   class conditioning enabled, the first optimisation step hits three faults in
   sequence: an integer dtype into the linear embedding, a host-resident tensor
   against an accelerator-resident model, and a rank-one label where the
   embedding expects a column. Correcting one exposes the next. The three
   patches are in [`vendor/patches/squidiff/`](vendor/patches/squidiff) with
   regression tests. They touch input plumbing, not the diffusion objective or
   the sampler. The released development configuration uses `class_cond=False`,
   so this concerns reuse of that interface and is not a statement about the
   authors' own encoder-based results.

3. **An unpropagated sampling default.** Development prediction extrapolates
   along a latent direction and samples around the extrapolated point. The
   sampler defaults to a latent standard deviation of 0.7, which the released
   prediction wrapper never passes. In our primary model the D7→D14 direction
   norm was 0.081 against an expected injected-noise norm of 5.42 at that
   default; energy distance was 1,302.39 at 0.7 and 27.56 at scale 0.

The audit then evaluated the model for temporal prediction on an independent
longitudinal mouse CAR-NK dataset (GEO
[GSE190976](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190976),
16,256 cells from 18 samples) at three leakage-safe temporal cutoffs, against
four training-window-only baselines, with a same-distribution reference and
complementary marginal, dependence and population-composition metrics.

Results depend strongly on the cutoff and the estimand. Simple temporal
baselines are competitive on marginal distances, while the diffusion model
retains dependence structure in settings where diagonal samplers cannot.

## Layout

```
src/reuse_gate/          audit package: data contracts, splits, metrics,
                         temporal baselines, provenance and gates
tests/                   unit, regression and integration tests, including the
                         three patch regressions
external_runners/        scripts that drive the audit end to end
vendor/patches/squidiff/ the three corrective patches
vendor/manifests/        the pinned upstream revision the patches apply to
configs/                 candidate selection and cutoff-study configuration
artifacts/               frozen numerical evidence (see below)
```

## The frozen evidence

`artifacts/` holds the machine-readable records every reported number comes from.
They are small enough to read directly and to diff.

| Path | What it fixes |
|---|---|
| `artifacts/cutoff_studies/{early_d14,primary_d21_d28,late_d28}/posthoc_evaluation.json` | Per-seed metrics for the model and all four baselines at each temporal cutoff, plus the same-distribution reference |
| `artifacts/cutoff_studies/*/split_manifest.json` | The sample-disjoint train/test populations and their hashes |
| `artifacts/evaluation_robustness/robustness.json` | Null anchors, sample-level bootstrap intervals, leave-one-sample-out re-scoring, baseline dispersion |
| `artifacts/revision_results/revision_results.json` | Consolidated three-cutoff manifest, the released-checkpoint sanity check and the release audit |
| `artifacts/baseline_provenance/baseline_provenance.json` | Which rows each baseline was fitted on |
| `artifacts/positive_control/*.json` | The released-checkpoint verification on the authors' own data |
| `artifacts/manuscript_figures/source_data.json` | Figure source data |

The **same-distribution reference** is the one to read first. It is the metric
computed between random halves of the held-out population over 50 repeats, and
it sets the finite-sample scale on which every other number must be read. At the
primary cutoff it is an energy distance of 0.031 (5th–95th percentile
0.015–0.070). Every candidate, including every baseline, sits far above it.

## Reproducing

Python 3.10–3.12.

```bash
pip install -e ".[dev]"
pytest
```

114 tests pass and 8 skip on a clean checkout with no upstream model, no GPU and
no downloaded data. They cover the data contracts, the temporal splits, the
metrics, the training-window baselines, the leakage audit, and the released
records in `artifacts/`. The skips are the checks that need the upstream tree or
a CUDA device, and they say so when skipped.

To re-run the audit itself you need the upstream code and the data, neither of
which is redistributed here:

```bash
git clone https://github.com/siyuh/Squidiff vendor/Squidiff
git -C vendor/Squidiff checkout abdfc27d84947dcccd745d1067c0840a41d32eb8
git -C vendor/Squidiff apply ../../vendor/patches/squidiff/*.patch
```

That commit is Squidiff v1.0.8, pinned 2026-07-22, and is recorded with its
hashes in `vendor/manifests/squidiff.json`. The patches apply to it and to no
other revision.

GSE190976 is public. The released Squidiff checkpoint and training data are at
[10.6084/m9.figshare.27948633](https://doi.org/10.6084/m9.figshare.27948633).
The derived-data record — checkpoints, splits and figure source data — is at
[10.5281/zenodo.21510503](https://doi.org/10.5281/zenodo.21510503).

## Scope

This is an assessment of reuse, not a replication attempt of the original
study's own results, and not a claim that the software is broken. Every
corrective patch is paired with a regression test, and every performance
statement is scoped to a declared temporal information boundary. The released
VO analysis is target-informed, so it confirms that the mechanism operates but
does not support an out-of-sample generalisation claim, and we say so.

## Licence and citation

Code is MIT-licensed. See [`CITATION.cff`](CITATION.cff). Please cite both this
software record and the Reusability Report when it appears.
