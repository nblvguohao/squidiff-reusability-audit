"""Create the final NMI revision figures from the consolidated result manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIGURE_WIDTH_MM = 180
WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FONT_SIZE = 6.5
RASTER_DPI = 600
EXPORT_SUFFIXES = (".svg", ".pdf", ".png", ".tiff")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.grid": False,
        "savefig.transparent": False,
    }
)

PALETTE = {
    "Squidiff": "#0072B2",
    "Last observation": "#777777",
    "Pooled diagonal Gaussian": "#E69F00",
    "Temporal diagonal Gaussian": "#CC79A7",
    "Temporal factor Gaussian": "#009E73",
    "Reference": "#BBBBBB",
}
BASELINE_NAMES = {
    "last_observation_resample": "Last observation",
    "pooled_diagonal_gaussian": "Pooled diagonal Gaussian",
    "temporal_diagonal_gaussian": "Temporal diagonal Gaussian",
    "temporal_factor_gaussian": "Temporal factor Gaussian",
}
CUTOFF_LABELS = {
    "early_d14": "D14\ntrain≤D7",
    "primary_d21_d28": "D21/D28\ntrain≤D14",
    "late_d28": "D28\ntrain≤D21",
}
CUTOFF_ORDER = ("early_d14", "primary_d21_d28", "late_d28")


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )


def _clean_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=2.5, width=0.6)


def _export(figure: Any, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    paths = [output_dir / f"{stem}{suffix}" for suffix in EXPORT_SUFFIXES]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    figure.savefig(paths[2], dpi=dpi, bbox_inches="tight")
    figure.savefig(paths[3], dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return paths


def _metric_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    mass = metrics["cluster_mass"]
    return {
        "energy_distance": float(metrics["energy_distance"]),
        "mean_expression_correlation": float(metrics["mean_expression_correlation"]),
        "correlation_frobenius_raw": float(metrics["correlation_frobenius_raw"]),
        "correlation_frobenius_normalized": float(metrics["correlation_frobenius_normalized"]),
        "cluster_mass_mae": float(mass["cluster_mass_mae"]),
        "cluster_mass_jsd": float(mass["cluster_mass_jsd"]),
        "rare_mass_recall": float(mass["rare_mass_recall"]),
        "rare_mass_precision": float(mass["rare_mass_precision"]),
        "cluster_mass_sensitivity": metrics["cluster_mass_sensitivity"],
    }


def figure3_source_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every cutoff, seed, scale and baseline without dropping rows."""
    rows = []
    for cutoff_name in CUTOFF_ORDER:
        cutoff = manifest["cutoffs"][cutoff_name]
        for seed_entry in cutoff["per_seed"]:
            seed = int(seed_entry["seed"])
            for scale_name, block in seed_entry["squidiff"].items():
                rows.append(
                    {
                        "cutoff": cutoff_name,
                        "seed": seed,
                        "method": "Squidiff",
                        "scale": float(scale_name.removeprefix("scale_")),
                        **_metric_fields(block["primary"]),
                    }
                )
            for baseline_name, block in seed_entry["baselines"].items():
                if baseline_name not in BASELINE_NAMES:
                    continue
                rows.append(
                    {
                        "cutoff": cutoff_name,
                        "seed": seed,
                        "method": BASELINE_NAMES[baseline_name],
                        "scale": None,
                        **_metric_fields(block["primary"]),
                    }
                )
    return rows


def _figure1(manifest: dict[str, Any]) -> Any:
    figure, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.65))
    ax_a, ax_b, ax_c = axes
    for axis in axes:
        axis.set_axis_off()

    _panel_label(ax_a, "a")
    ax_a.set_title("Evidence ladder", loc="left")
    stages = (
        ("Released artefacts", "strict load + finite sampling"),
        ("Executable interfaces", "tests of advertised code paths"),
        ("Predictive reuse", "sample-disjoint temporal cutoffs"),
    )
    for index, (title, detail) in enumerate(stages):
        y = 0.82 - index * 0.31
        ax_a.add_patch(
            mpl.patches.FancyBboxPatch(
                (0.05, y - 0.12),
                0.9,
                0.19,
                boxstyle="round,pad=0.015",
                facecolor="#F4F4F4",
                edgecolor="#333333",
                linewidth=0.7,
            )
        )
        ax_a.text(0.09, y, title, fontweight="bold", va="center")
        ax_a.text(0.09, y - 0.065, detail, va="center")
        if index < len(stages) - 1:
            ax_a.annotate(
                "",
                xy=(0.5, y - 0.20),
                xytext=(0.5, y - 0.13),
                arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#333333"},
            )

    _panel_label(ax_b, "b")
    ax_b.set_title("Temporal information boundaries", loc="left")
    days = (0, 7, 14, 21, 28)
    for row, cutoff_name in enumerate(CUTOFF_ORDER):
        cutoff = manifest["cutoffs"][cutoff_name]
        y = 0.80 - row * 0.29
        train_times = set(cutoff.get("train_times", ()))
        test_times = set(cutoff.get("test_times", ()))
        if not train_times and cutoff_name == "early_d14":
            train_times, test_times = {0, 7}, {14, 21, 28}
        elif not train_times and cutoff_name == "primary_d21_d28":
            train_times, test_times = {0, 7, 14}, {21, 28}
        elif not train_times:
            train_times, test_times = {0, 7, 14, 21}, {28}
        ax_b.plot([0.11, 0.91], [y, y], color="#888888", lw=0.7)
        for index, day in enumerate(days):
            x = 0.11 + index * 0.20
            color = (
                "#0072B2" if day in train_times else "#E69F00" if day in test_times else "#DDDDDD"
            )
            ax_b.scatter(x, y, s=28, color=color, edgecolor="black", linewidth=0.4)
            ax_b.text(x, y - 0.08, f"D{day}", ha="center")
        ax_b.text(
            0.11,
            y + 0.085,
            CUTOFF_LABELS[cutoff_name].replace("\n", " • "),
            va="bottom",
            ha="left",
            fontsize=5.8,
            fontweight="bold",
        )
    ax_b.text(0.11, 0.01, "blue, train only; ochre, scored targets", ha="left")

    _panel_label(ax_c, "c")
    ax_c.set_title("Operations by information set", loc="left")
    train_ops = (
        "normalization per cell",
        "feature selection",
        "direction / validation",
        "baseline fitting",
    )
    test_ops = ("generation count", "metric scoring", "reference splits")
    ax_c.add_patch(
        mpl.patches.Rectangle((0.03, 0.48), 0.94, 0.42, facecolor="#E6F2F8", edgecolor="#0072B2")
    )
    ax_c.text(0.07, 0.84, "Training information only", fontweight="bold")
    for index, item in enumerate(train_ops):
        ax_c.text(0.08, 0.76 - index * 0.075, f"• {item}")
    ax_c.add_patch(
        mpl.patches.Rectangle((0.03, 0.10), 0.94, 0.26, facecolor="#FFF2D8", edgecolor="#E69F00")
    )
    ax_c.text(0.07, 0.31, "Target information", fontweight="bold")
    for index, item in enumerate(test_ops):
        ax_c.text(0.08, 0.235 - index * 0.06, f"• {item}")
    return figure


def _figure2(manifest: dict[str, Any]) -> Any:
    audit = manifest["release_audit"]
    checkpoint = audit["released_checkpoint"]
    preprocessing = audit["preprocessing_ab"]
    simulation = audit["simulated_benchmark"]
    noise = audit["latent_noise_sensitivity"]
    figure, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 4.7))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    _panel_label(ax_a, "a")
    ax_a.set_axis_off()
    ax_a.text(
        0.03,
        0.96,
        "Released checkpoint audit",
        fontsize=7,
        fontweight="bold",
        va="top",
    )
    checkpoint_rows = (
        ("State tensors", checkpoint["checkpoint_info"]["n_tensors"]),
        ("Parameters", f"{checkpoint['checkpoint_info']['n_parameters'] / 1e6:.1f} M"),
        (
            "Missing / unexpected keys",
            (
                f"{checkpoint['load_state_dict']['n_missing']} / "
                f"{checkpoint['load_state_dict']['n_unexpected']}"
            ),
        ),
        ("Finite generated values", str(checkpoint["sampling"]["finite"])),
        ("Energy distance", f"{checkpoint['sampling']['energy_distance']:.2f}"),
    )
    for index, (label, value) in enumerate(checkpoint_rows):
        y = 0.76 - index * 0.16
        ax_a.text(0.03, y, label)
        ax_a.text(0.97, y, value, ha="right", fontweight="bold")
        ax_a.plot([0.03, 0.97], [y - 0.06, y - 0.06], color="#DDDDDD", lw=0.5)

    _panel_label(ax_b, "b")
    ax_b.set_title("Single-variable preprocessing comparison", loc="left")
    for condition, label, color in (
        ("raw", "raw counts", "#D55E00"),
        ("lognorm", "normalize_total + log1p", "#0072B2"),
    ):
        values = preprocessing["conditions"][condition]["per_budget"]
        ax_b.plot(
            [entry["steps"] for entry in values],
            [entry["pooled_energy_distance"] for entry in values],
            marker="o",
            ms=3,
            lw=1,
            label=label,
            color=color,
        )
    ax_b.set_xlabel("Training steps")
    ax_b.set_ylabel("Energy distance (lower is better)")
    ax_b.legend(frameon=False)
    _clean_axis(ax_b)

    _panel_label(ax_c, "c")
    ax_c.set_title("Simulation separability after preprocessing", loc="left")
    degeneracy = simulation["preprocessing_degeneracy"]
    labels = ("raw simulation", "after normalize + log1p")
    values = [float(degeneracy[label]["silhouette"]) for label in labels]
    ax_c.plot([0, 1], values, marker="o", color="#333333", lw=1)
    ax_c.axhline(0, color="#AAAAAA", lw=0.6)
    ax_c.set_xticks([0, 1], ["Raw", "Normalized\n+ log1p"])
    ax_c.set_ylabel("Silhouette coefficient (higher is better)")
    _clean_axis(ax_c)

    _panel_label(ax_d, "d")
    ax_d.set_title("Latent-noise sensitivity", loc="left")
    scales = noise["scales"]
    x = np.arange(len(scales))
    values = [entry["pooled_energy_distance"] for entry in scales]
    ax_d.plot(x, values, marker="o", color="#0072B2", lw=1)
    default = float(noise["upstream_default_scale"])
    default_indices = [
        index for index, entry in enumerate(scales) if float(entry["scale"]) == default
    ]
    if default_indices:
        index = default_indices[0]
        ax_d.scatter(index, values[index], s=32, facecolor="none", edgecolor="#D55E00")
    ax_d.set_yscale("log")
    ax_d.set_xticks(x, [f"{entry['scale']:.3g}" for entry in scales])
    ax_d.set_xlabel("Latent-noise scale")
    ax_d.set_ylabel("Energy distance (lower is better)")
    _clean_axis(ax_d)
    figure.tight_layout(w_pad=1.8, h_pad=2.1)
    return figure


def _display_group(row: dict[str, Any]) -> str:
    if row["method"] == "Squidiff" and row["cutoff"] == "early_d14":
        return f"Squidiff s={row['scale']:g}"
    return row["method"]


def _plot_metric(
    axis: Any,
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    *,
    higher: bool,
) -> None:
    groups = [
        "Squidiff",
        "Squidiff s=0",
        "Squidiff s=0.03",
        "Last observation",
        "Pooled diagonal Gaussian",
        "Temporal diagonal Gaussian",
        "Temporal factor Gaussian",
    ]
    offsets = np.linspace(-0.28, 0.28, len(groups))
    for cutoff_index, cutoff_name in enumerate(CUTOFF_ORDER):
        cutoff_rows = [row for row in rows if row["cutoff"] == cutoff_name]
        for group, offset in zip(groups, offsets, strict=True):
            values = [float(row[metric]) for row in cutoff_rows if _display_group(row) == group]
            if not values:
                continue
            method = "Squidiff" if group.startswith("Squidiff") else group
            marker = "D" if group == "Squidiff s=0.03" else "o"
            x = cutoff_index + offset
            jitter = np.linspace(-0.018, 0.018, num=len(values))
            axis.scatter(
                x + jitter,
                values,
                s=8,
                alpha=0.55,
                color=PALETTE[method],
                marker=marker,
                linewidth=0,
            )
            mean = float(np.mean(values))
            spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            axis.errorbar(
                x,
                mean,
                yerr=spread,
                color=PALETTE[method],
                marker=marker,
                markersize=3.4,
                markeredgecolor="black",
                markeredgewidth=0.3,
                capsize=1.5,
                lw=0.8,
            )
    axis.set_xticks(range(3), [CUTOFF_LABELS[value] for value in CUTOFF_ORDER])
    axis.set_ylabel(ylabel)
    direction = "higher is better" if higher else "lower is better"
    axis.set_title(direction, loc="right", fontsize=5.8, pad=2)
    _clean_axis(axis)


def _plot_rare_sensitivity(axis: Any, rows: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in rows
        if row["cutoff"] == "primary_d21_d28"
        and row["method"] in ("Squidiff", "Temporal factor Gaussian")
    ]
    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in selected:
        for setting in row["cluster_mass_sensitivity"]:
            values[(row["method"], float(setting["rare_below"]))].append(
                float(setting["rare_mass_recall"])
            )
    for method in ("Squidiff", "Temporal factor Gaussian"):
        thresholds = sorted(threshold for candidate, threshold in values if candidate == method)
        if not thresholds:
            continue
        means = [np.mean(values[(method, threshold)]) for threshold in thresholds]
        axis.plot(
            thresholds,
            means,
            marker="o",
            ms=3,
            lw=1,
            color=PALETTE[method],
            label=method,
        )
    axis.set_xlabel("Rare-state threshold")
    axis.set_ylabel("Rare-mass recall (higher is better)")
    axis.set_ylim(-0.03, 1.03)
    _clean_axis(axis)


def _figure3(rows: list[dict[str, Any]]) -> Any:
    figure = plt.figure(figsize=(WIDTH_IN, 4.95))
    grid = figure.add_gridspec(2, 3, hspace=0.48, wspace=0.52)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[0, 2])
    ax_d = figure.add_subplot(grid[1, :2])
    ax_e = figure.add_subplot(grid[1, 2])

    _panel_label(ax_a, "a")
    _plot_metric(
        ax_a,
        rows,
        "energy_distance",
        "Energy distance",
        higher=False,
    )
    _panel_label(ax_b, "b")
    _plot_metric(
        ax_b,
        rows,
        "mean_expression_correlation",
        "Mean-expression correlation",
        higher=True,
    )
    _panel_label(ax_c, "c")
    _plot_metric(
        ax_c,
        rows,
        "correlation_frobenius_normalized",
        "Normalized correlation distance",
        higher=False,
    )
    _panel_label(ax_d, "d")
    _plot_metric(
        ax_d,
        rows,
        "cluster_mass_mae",
        "Cluster-mass MAE",
        higher=False,
    )
    _plot_rare_sensitivity(ax_e, rows)
    ax_e.set_title("Primary cutoff sensitivity", loc="left")

    handles = []
    for method in PALETTE:
        if method == "Reference":
            continue
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                color=PALETTE[method],
                marker="o",
                lw=0,
                label=method,
            )
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(bottom=0.16)
    return figure


def make_all_figures(
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    dpi: int = RASTER_DPI,
) -> list[Path]:
    """Render all main figures and a complete machine-readable source file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = figure3_source_rows(manifest)
    audit = manifest["release_audit"]
    preprocessing_rows = []
    for condition, block in audit["preprocessing_ab"]["conditions"].items():
        for entry in block["per_budget"]:
            preprocessing_rows.append(
                {
                    "condition": condition,
                    "training_steps": entry["steps"],
                    "energy_distance": entry["pooled_energy_distance"],
                }
            )
    simulation = audit["simulated_benchmark"]["preprocessing_degeneracy"]
    figure1_cutoffs = {}
    for cutoff_name in CUTOFF_ORDER:
        cutoff = manifest["cutoffs"][cutoff_name]
        figure1_cutoffs[cutoff_name] = {
            "train_times": cutoff.get("train_times"),
            "test_times": cutoff.get("test_times"),
        }
    source = {
        "schema_version": "1.0",
        "source_manifest": "artifacts/revision_results/revision_results.json",
        "figure1_cutoffs": figure1_cutoffs,
        "figure2": {
            "released_checkpoint": {
                "n_tensors": audit["released_checkpoint"]["checkpoint_info"]["n_tensors"],
                "n_parameters": audit["released_checkpoint"]["checkpoint_info"]["n_parameters"],
                "n_missing_keys": audit["released_checkpoint"]["load_state_dict"]["n_missing"],
                "n_unexpected_keys": audit["released_checkpoint"]["load_state_dict"][
                    "n_unexpected"
                ],
                "finite_generated_values": audit["released_checkpoint"]["sampling"]["finite"],
                "energy_distance": audit["released_checkpoint"]["sampling"]["energy_distance"],
            },
            "preprocessing_rows": preprocessing_rows,
            "simulation_rows": [
                {
                    "stage": stage,
                    "silhouette": simulation[stage]["silhouette"],
                }
                for stage in ("raw simulation", "after normalize + log1p")
            ],
            "latent_noise_rows": [
                {
                    "scale": entry["scale"],
                    "energy_distance": entry["pooled_energy_distance"],
                    "is_default": float(entry["scale"])
                    == float(audit["latent_noise_sensitivity"]["upstream_default_scale"]),
                }
                for entry in audit["latent_noise_sensitivity"]["scales"]
            ],
        },
        "figure3_rows": rows,
        "vo_interpretation": manifest["vo_sanity_check"]["interpretation"],
        "vo_supports_generalization": manifest["vo_sanity_check"]["supports_generalization"],
        "statistics": {
            "n_definition": "five independently trained computational seeds",
            "center": "arithmetic mean across seeds",
            "spread": "sample standard deviation across seeds",
            "inference": "descriptive; no seed-based population hypothesis test",
        },
    }
    source_path = output_dir / "figure_source_data.json"
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    outputs = [source_path]
    outputs.extend(_export(_figure1(manifest), output_dir, "Figure_1", dpi))
    outputs.extend(_export(_figure2(manifest), output_dir, "Figure_2", dpi))
    outputs.extend(_export(_figure3(rows), output_dir, "Figure_3", dpi))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/revision_results/revision_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/revision_figures"),
    )
    parser.add_argument("--dpi", type=int, default=RASTER_DPI)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    make_all_figures(manifest, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
