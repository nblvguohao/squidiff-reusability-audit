"""Consolidate all manuscript evidence into one provenance-tracked manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_load(path: Path) -> dict[str, Any] | None:
    return _load(path) if path.exists() else None


def _source(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
    }


def _completed_cutoff(path: Path) -> dict[str, Any]:
    cutoff = _load(path)
    expected = cutoff.get("expected_seeds", [])
    completed = cutoff.get("completed_seeds", [])
    if not cutoff.get("complete") or sorted(expected) != sorted(completed):
        raise ValueError(f"incomplete cutoff result: {cutoff.get('cutoff', path.parent.name)}")
    if len(cutoff.get("per_seed", [])) != len(expected):
        raise ValueError(f"incomplete per-seed results: {cutoff.get('cutoff', path.parent.name)}")
    return cutoff


def _completed_posthoc(path: Path) -> dict[str, Any]:
    posthoc = _load(path)
    expected = posthoc.get("expected_seeds", [])
    completed = posthoc.get("completed_seeds", [])
    if not posthoc.get("complete") or sorted(expected) != sorted(completed):
        raise ValueError(f"incomplete post-hoc evaluation: {posthoc.get('cutoff')}")
    if len(posthoc.get("per_seed", [])) != len(expected):
        raise ValueError(f"incomplete post-hoc per-seed results: {posthoc.get('cutoff')}")
    return posthoc


def _apply_posthoc(cutoff: dict[str, Any], posthoc: dict[str, Any]) -> None:
    """Replace only re-scored fields while retaining model results and logs."""
    corrections = {entry["seed"]: entry for entry in posthoc["per_seed"]}
    for entry in cutoff.get("per_seed", []):
        correction = corrections.get(entry["seed"])
        if correction is None:
            raise ValueError(f"missing post-hoc result for seed {entry['seed']}")
        entry.setdefault("baselines", {}).update(correction.get("baselines", {}))
    cutoff["same_distribution_reference"] = posthoc["same_distribution_reference"]
    cutoff["posthoc_evaluation"] = posthoc


def _merge_primary(
    seed_study: dict[str, Any],
    robustness: dict[str, Any],
    baseline_posthoc: dict[str, Any],
    model_posthoc: dict[str, Any],
    split_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    baselines = {
        entry["seed"]: entry.get("baselines", {}) for entry in baseline_posthoc["per_seed"]
    }
    per_seed = model_posthoc["per_seed"]
    for entry in per_seed:
        if entry["seed"] not in baselines:
            raise ValueError(f"missing primary baselines for seed {entry['seed']}")
        entry["baselines"] = baselines[entry["seed"]]
    return {
        "cutoff": "primary_d21_d28",
        "train_times": [0, 7, 14],
        "test_times": [21, 28],
        "primary_test_times": [21, 28],
        "expected_seeds": model_posthoc["expected_seeds"],
        "completed_seeds": model_posthoc["completed_seeds"],
        "complete": model_posthoc["complete"],
        "per_seed": per_seed,
        "same_distribution_reference": baseline_posthoc["same_distribution_reference"],
        "posthoc_evaluation": baseline_posthoc,
        "posthoc_model_evaluation": model_posthoc,
        "legacy_seed_study": seed_study,
        "robustness": robustness,
        "split_manifest": split_manifest,
    }


def consolidate(
    root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Merge primary, early, late, and target-informed sanity-check artifacts."""
    primary_path = root / "artifacts/squidiff_seed_study/seed_study_metrics.json"
    robustness_path = root / "artifacts/evaluation_robustness/robustness.json"
    early_path = root / "artifacts/cutoff_studies/early_d14/cutoff_summary.json"
    late_path = root / "artifacts/cutoff_studies/late_d28/cutoff_summary.json"
    primary_posthoc_path = root / "artifacts/cutoff_studies/primary_d21_d28/posthoc_evaluation.json"
    primary_model_posthoc_path = (
        root / "artifacts/cutoff_studies/primary_d21_d28/posthoc_model_evaluation.json"
    )
    early_posthoc_path = root / "artifacts/cutoff_studies/early_d14/posthoc_evaluation.json"
    late_posthoc_path = root / "artifacts/cutoff_studies/late_d28/posthoc_evaluation.json"
    primary_split_path = root / "artifacts/squidiff_sweep_lognorm/split_manifest.json"
    early_split_path = root / "artifacts/cutoff_studies/early_d14/split_manifest.json"
    late_split_path = root / "artifacts/cutoff_studies/late_d28/split_manifest.json"
    vo_distribution_path = root / "artifacts/positive_control/positive_control_metrics.json"
    vo_structure_path = root / "artifacts/positive_control/structure_metrics.json"
    released_path = root / "artifacts/released_checkpoint/released_checkpoint_check.json"
    preprocessing_path = root / "artifacts/squidiff_latent_extrap_ab/preprocessing_ab_metrics.json"
    noise_path = root / "artifacts/squidiff_latent_extrap/latent_noise_scale_sweep.json"
    simulation_path = root / "artifacts/squidiff_reproduction/reproduction_metrics.json"

    primary = _load(primary_path)
    robustness = _load(robustness_path)
    early = _completed_cutoff(early_path)
    late = _completed_cutoff(late_path)
    primary_posthoc = _completed_posthoc(primary_posthoc_path)
    primary_model_posthoc = _completed_posthoc(primary_model_posthoc_path)
    early_posthoc = _completed_posthoc(early_posthoc_path)
    late_posthoc = _completed_posthoc(late_posthoc_path)
    _apply_posthoc(early, early_posthoc)
    _apply_posthoc(late, late_posthoc)
    early["split_manifest"] = _optional_load(early_split_path)
    late["split_manifest"] = _optional_load(late_split_path)
    vo_distribution = _load(vo_distribution_path)
    vo_structure = _load(vo_structure_path)

    result = {
        "schema_version": "1.0",
        "purpose": "single source of truth for the NMI revision",
        "cutoffs": {
            "primary_d21_d28": _merge_primary(
                primary,
                robustness,
                primary_posthoc,
                primary_model_posthoc,
                _optional_load(primary_split_path),
            ),
            "early_d14": early,
            "late_d28": late,
        },
        "vo_sanity_check": {
            "interpretation": "target-informed",
            "supports_generalization": False,
            "information_boundary": (
                "The released checkpoint was trained on both target days and the "
                "latent direction uses target-day cells; this is a mechanism sanity "
                "check, not an out-of-sample prediction or generalization test."
            ),
            "distribution_metrics": vo_distribution,
            "structure_metrics": vo_structure,
        },
        "release_audit": {
            "released_checkpoint": _optional_load(released_path),
            "preprocessing_ab": _optional_load(preprocessing_path),
            "latent_noise_sensitivity": _optional_load(noise_path),
            "simulated_benchmark": _optional_load(simulation_path),
            "conditional_interface": {
                "scope": (
                    "The label-conditional interface fails in the pinned release; "
                    "the released development configuration uses class_cond=False."
                ),
                "defects": [
                    "label dtype at the embedding input",
                    "label device transfer",
                    "rank expected by the label embedding",
                ],
            },
        },
        "provenance": {
            "sources": [
                _source(path, root)
                for path in (
                    primary_path,
                    robustness_path,
                    early_path,
                    late_path,
                    primary_posthoc_path,
                    primary_model_posthoc_path,
                    early_posthoc_path,
                    late_posthoc_path,
                    vo_distribution_path,
                    vo_structure_path,
                    released_path,
                    preprocessing_path,
                    noise_path,
                    simulation_path,
                    primary_split_path,
                    early_split_path,
                    late_split_path,
                )
                if path.exists()
            ]
        },
    }
    destination = (
        output_path
        if output_path is not None
        else root / "artifacts/revision_results/revision_results.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    consolidate(args.root, args.output)


if __name__ == "__main__":
    main()
