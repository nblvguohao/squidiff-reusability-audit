"""Declarative temporal cutoffs for the NMI Squidiff reusability study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import anndata as ad

from reuse_gate.splits.audit import assert_no_overlap


@dataclass(frozen=True)
class CutoffSpec:
    """Information boundary and prediction geometry for one cutoff study."""

    name: str
    train_times: tuple[int, ...]
    test_times: tuple[int, ...]
    direction_times: tuple[int, int]
    validation_triplet: tuple[int, int, int] | None
    fixed_scale_sensitivity: tuple[float, ...] = ()


@dataclass(frozen=True)
class CutoffManifest:
    """Auditable description of the cells and samples assigned to a cutoff."""

    name: str
    train_times: tuple[int, ...]
    test_times: tuple[int, ...]
    direction_times: tuple[int, int]
    validation_triplet: tuple[int, int, int] | None
    fixed_scale_sensitivity: tuple[float, ...]
    train_cells: int
    test_cells: int
    train_samples: int
    test_samples: int
    train_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]
    group_col: str
    time_col: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""
        return asdict(self)


@dataclass
class TemporalCutoff:
    """Materialized AnnData views plus their immutable audit manifest."""

    train: ad.AnnData
    test: ad.AnnData
    manifest: CutoffManifest


def early_cutoff_spec() -> CutoffSpec:
    """Pre-infusion/D7 training with D14 as the primary future target."""
    return CutoffSpec(
        name="early_d14",
        train_times=(0, 7),
        test_times=(14, 21, 28),
        direction_times=(0, 7),
        validation_triplet=None,
        fixed_scale_sensitivity=(0.0, 0.03),
    )


def late_cutoff_spec() -> CutoffSpec:
    """Training through D21 with D28 held out."""
    return CutoffSpec(
        name="late_d28",
        train_times=(0, 7, 14, 21),
        test_times=(28,),
        direction_times=(14, 21),
        validation_triplet=(7, 14, 21),
    )


def _validate_spec(spec: CutoffSpec) -> None:
    overlap = set(spec.train_times) & set(spec.test_times)
    if overlap:
        raise ValueError(f"train/test timepoint overlap: {sorted(overlap)}")
    if not set(spec.direction_times).issubset(spec.train_times):
        raise ValueError("direction timepoints must be in the training window")
    if spec.validation_triplet is not None and not set(spec.validation_triplet).issubset(
        spec.train_times
    ):
        raise ValueError("validation timepoints must be in the training window")
    if spec.validation_triplet is not None and spec.fixed_scale_sensitivity:
        raise ValueError("use either training-only validation or fixed scale sensitivity")


def build_temporal_cutoff(
    adata: ad.AnnData,
    spec: CutoffSpec,
    *,
    group_col: str = "sample_id",
    time_col: str = "timepoint_numeric",
) -> TemporalCutoff:
    """Materialize a cutoff without sharing samples across train and test."""
    _validate_spec(spec)
    for column in (group_col, time_col):
        if column not in adata.obs:
            raise ValueError(f"required observation column is missing: {column}")

    train = adata[adata.obs[time_col].isin(spec.train_times)].copy()
    test = adata[adata.obs[time_col].isin(spec.test_times)].copy()
    train_ids = {str(value) for value in train.obs[group_col]}
    test_ids = {str(value) for value in test.obs[group_col]}
    assert_no_overlap(train_ids, test_ids, group_col)

    manifest = CutoffManifest(
        name=spec.name,
        train_times=spec.train_times,
        test_times=spec.test_times,
        direction_times=spec.direction_times,
        validation_triplet=spec.validation_triplet,
        fixed_scale_sensitivity=spec.fixed_scale_sensitivity,
        train_cells=int(train.n_obs),
        test_cells=int(test.n_obs),
        train_samples=len(train_ids),
        test_samples=len(test_ids),
        train_sample_ids=tuple(sorted(train_ids)),
        test_sample_ids=tuple(sorted(test_ids)),
        group_col=group_col,
        time_col=time_col,
    )
    return TemporalCutoff(train=train, test=test, manifest=manifest)
