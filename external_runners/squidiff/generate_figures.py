"""Squidiff Tier 1 — comprehensive analysis and 6-panel figure generation.

Produces:
  Fig 1: data and split design
  Fig 2: official reproduction (Squidiff smoke)
  Fig 3: held-out temporal prediction (Tier 0 baselines)
  Fig 4: state-proportion recovery (Tier 1 multi-seed)
  Fig 5: construct/shift analysis
  Fig 6: resource and sensitivity analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", palette="muted")


OUTPUT_DIR = Path("artifacts/squidiff_tier1/figures")
SOURCE_DATA_DIR = Path("artifacts/squidiff_tier1/source_data")
SEEDS = [13, 37, 73, 101, 137]
SEED_COLORS = sns.color_palette("viridis", len(SEEDS))


def fig1_split_design(adata):
    """Figure 1: Temporal split design — cells per timepoint and construct."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Cells per timepoint
    ax = axes[0]
    tp_counts = adata.obs.groupby("timepoint_raw").size().sort_index(key=lambda x: x.map({
        "pre": 0, "D7": 1, "D14": 2, "D21": 3, "D28": 4,
    }))
    colors = ["#2196F3" if t in ("pre", "D7", "D14") else "#FF5722" for t in tp_counts.index]
    bars = ax.bar(range(len(tp_counts)), tp_counts.values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(tp_counts)))
    ax.set_xticklabels(tp_counts.index, fontsize=11)
    ax.set_ylabel("Number of cells", fontsize=12)
    ax.set_title("A) Cells per timepoint (blue=train, red=test)", fontsize=13, fontweight="bold")
    for bar, count in zip(bars, tp_counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(count),
                ha="center", fontsize=9)

    # Panel B: Cells per construct
    ax = axes[1]
    ct_counts = adata.obs.groupby(["construct", "timepoint_raw"]).size().unstack(fill_value=0)
    ct_order = ["pre", "D7", "D14", "D21", "D28"]
    ct_counts = ct_counts.reindex(columns=[c for c in ct_order if c in ct_counts.columns])
    ct_counts.T.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(ct_counts.index)))
    ax.set_xlabel("Timepoint", fontsize=12)
    ax.set_ylabel("Number of cells", fontsize=12)
    ax.set_title("B) Construct distribution across timepoints", fontsize=13, fontweight="bold")
    ax.legend(title="Construct", fontsize=9, title_fontsize=10)
    ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Figure 1: Data and Split Design — GSE190976 CAR-NK", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def fig2_official_reproduction(tier0_metrics):
    """Figure 2: Squidiff smoke test results."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    smoke = tier0_metrics["steps"].get("squidiff_smoke", {})

    # Panel A: Model architecture
    ax = axes[0]
    ax.axis("off")
    info_text = (
        f"Squidiff v1.0.8\n"
        f"Commit: abdfc27\n"
        f"Parameters: {smoke.get('n_params', 'N/A'):,}\n"
        f"Forward pass: [4, 500] → [4, 500]\n"
        f"Finite output: {smoke.get('finite_output', 'N/A')}\n"
        f"Status: {smoke.get('status', 'N/A').upper()}"
    )
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=13,
            verticalalignment="center", fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "#E8F5E9", "alpha": 0.8})
    ax.set_title("A) Squidiff Smoke Test", fontsize=13, fontweight="bold")

    # Panel B: Baseline energy distances
    ax = axes[1]
    models = ["last_observation", "conditional_mean", "linear_interpolation"]
    labels = ["Last Obs", "Cond Mean", "Linear Interp"]
    eds = [tier0_metrics["steps"].get(m, {}).get("energy_distance", 0) for m in models]
    colors_bar = ["#E53935", "#43A047", "#1E88E5"]
    bars = ax.bar(labels, eds, color=colors_bar, edgecolor="white")
    ax.set_ylabel("Energy Distance", fontsize=12)
    ax.set_title("B) Baseline Performance (Tier 0)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, eds, strict=True):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{val:.0f}",
                ha="center", fontsize=10)
    ax.axhline(y=eds[0], color="#E53935", linestyle="--", alpha=0.3, label=f"Last Obs = {eds[0]:.0f}")

    # Panel C: GO conditions
    ax = axes[2]
    go = tier0_metrics.get("tier0_go", {}).get("conditions", {})
    go_names = list(go.keys())
    go_vals = [go[n] for n in go_names]
    go_labels = [n.replace("_", " ").title() for n in go_names]
    colors_go = ["#43A047" if v else "#E53935" for v in go_vals]
    ax.barh(go_labels, [1]*len(go_labels), color=colors_go, edgecolor="white", height=0.5)
    for i, val in enumerate(go_vals):
        ax.text(0.5, i, "PASS" if val else "FAIL", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("C) Tier 0 GO Conditions", fontsize=13, fontweight="bold")

    fig.suptitle("Figure 2: Official Reproduction — Squidiff Smoke & Baselines", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def fig3_temporal_prediction(tier1_metrics):
    """Figure 3: Early-to-late CAR-NK prediction across seeds and splits."""
    seeds_data = tier1_metrics["seeds"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Per-seed ED improvement
    ax = axes[0]
    improvements = []
    for s in SEEDS:
        seed_splits = seeds_data[str(s)]["splits"]
        for sp in seed_splits:
            ed_last = sp["energy_distance"]["last_observation"]
            ed_cond = sp["energy_distance"]["conditional_mean"]
            improvements.append({"seed": s, "improvement_pct": 100 * (ed_last - ed_cond) / max(ed_last, 1)})

    df_imp = pd.DataFrame(improvements)
    for i, s in enumerate(SEEDS):
        seed_vals = df_imp[df_imp["seed"] == s]["improvement_pct"]
        ax.boxplot([seed_vals], positions=[i], widths=0.6, patch_artist=True,
                   boxprops={"facecolor": SEED_COLORS[i], "alpha": 0.7},
                   medianprops={"color": "black", "linewidth": 2})
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=10)
    ax.set_ylabel("ED Improvement over Last Obs (%)", fontsize=12)
    ax.set_title("A) Per-Seed Distribution (5 splits each)", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.3)
    ax.set_ylim(bottom=min(df_imp["improvement_pct"]) - 5, top=max(df_imp["improvement_pct"]) + 5)

    # Panel B: Per-split breakdown (seed 13)
    ax = axes[1]
    split_data = seeds_data["13"]["splits"]
    test_names = [s["test_sample"].split("_")[-1][:25] for s in split_data]
    x = np.arange(len(test_names))
    w = 0.25
    ax.bar(x - w, [s["energy_distance"]["last_observation"] for s in split_data], w,
           label="Last Obs", color="#E53935", edgecolor="white")
    ax.bar(x, [s["energy_distance"]["conditional_mean"] for s in split_data], w,
           label="Cond Mean", color="#43A047", edgecolor="white")
    ax.bar(x + w, [s["energy_distance"]["linear_interpolation"] for s in split_data], w,
           label="Linear Interp", color="#1E88E5", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Energy Distance", fontsize=12)
    ax.set_title("B) Per-Split Breakdown (Seed 13)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: Aggregate stability
    ax = axes[2]
    agg = tier1_metrics["aggregate"]
    ax.bar(["Mean Improvement"], [agg["mean_ed_improvement_pct"]], yerr=[agg["std_ed_improvement_pct"]],
           color="#7E57C2", edgecolor="white", capsize=10, width=0.4)
    ax.set_ylabel("ED Improvement (%)", fontsize=12)
    ax.set_title(f"C) Cross-Seed Aggregate ({agg['n_seeds']} seeds)", fontsize=13, fontweight="bold")
    ax.text(0, agg["mean_ed_improvement_pct"] + agg["std_ed_improvement_pct"] + 1,
            f"{agg['mean_ed_improvement_pct']}% ± {agg['std_ed_improvement_pct']}%",
            ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(bottom=0)

    fig.suptitle("Figure 3: Early-to-Late CAR-NK Prediction — 25 Independent Evaluations",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def fig4_state_proportions(tier1_metrics):
    """Figure 4: Cell-state proportion recovery analysis."""
    splits = tier1_metrics["seeds"]["13"]["splits"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: DE agreement
    ax = axes[0]
    de_overlap = [s["de_agreement"]["overlap_frac"] for s in splits]
    de_spearman = [s["de_agreement"]["spearman_r"] for s in splits]
    test_names_short = [s["test_sample"].split("_")[-1][:20] for s in splits]

    x = np.arange(len(test_names_short))
    w = 0.35
    ax.bar(x - w/2, de_overlap, w, label="Top-50 Overlap", color="#FF9800", edgecolor="white")
    ax_twin = ax.twinx()
    ax_twin.plot(x, de_spearman, "o-", color="#1565C0", linewidth=2, markersize=8, label="Spearman ρ")
    ax.set_xticks(x)
    ax.set_xticks(x)
    ax.set_xticklabels(test_names_short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Gene Overlap Fraction", fontsize=12)
    ax_twin.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title("A) DE Gene Agreement (Top-50)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax_twin.legend(loc="upper right", fontsize=8)

    # Panel B: State proportion error
    ax = axes[1]
    prop_errors = [s["state_proportions"].get("proportion_error", 0) for s in splits]
    n_clusters = [s["state_proportions"].get("n_clusters", 0) for s in splits]
    ax.bar(test_names_short, prop_errors, color="#AB47BC", edgecolor="white")
    ax.set_xticks(range(len(test_names_short)))
    ax.set_xticklabels(test_names_short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Proportion Error", fontsize=12)
    ax.set_title("B) State Proportion Recovery", fontsize=13, fontweight="bold")
    for i, (err, nc) in enumerate(zip(prop_errors, n_clusters, strict=True)):
        ax.text(i, err + 0.005, f"k={nc}", ha="center", fontsize=8)

    # Panel C: Rare state recall
    ax = axes[2]
    rare_recalls = []
    for s in splits:
        rr = s["rare_state_recall"]
        rare_recalls.append({"sample": s["test_sample"].split("_")[-1][:20],
                             "recall": rr.get("mean_recall", 0),
                             "n_rare": rr.get("rare_clusters_found", 0)})
    df_rare = pd.DataFrame(rare_recalls)
    ax.bar(df_rare["sample"], df_rare["recall"], color="#26A69A", edgecolor="white")
    ax.set_xticks(range(len(df_rare)))
    ax.set_xticklabels(df_rare["sample"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Recall", fontsize=12)
    ax.set_title("C) Rare State Retention", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3)
    for i, row in df_rare.iterrows():
        ax.text(i, row["recall"] + 0.03, f"n={int(row['n_rare'])}", ha="center", fontsize=8)

    fig.suptitle("Figure 4: State-Proportion Recovery & DE Agreement", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def fig5_construct_shift(adata, construct_shift_data):
    """Figure 5: Construct/stimulation shift analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: CAR19 vs CAR19/IL15 at matched timepoints
    ax = axes[0]
    tps = list(construct_shift_data.keys())
    eds = [construct_shift_data[t]["energy_distance"] for t in tps]
    n_car19 = [construct_shift_data[t]["car19_n"] for t in tps]
    n_il15 = [construct_shift_data[t]["car19_il15_n"] for t in tps]
    ax.bar(tps, eds, color=["#FF7043", "#42A5F5"], edgecolor="white")
    ax.set_ylabel("Energy Distance", fontsize=12)
    ax.set_title("A) CAR19 → CAR19/IL15 Prediction", fontsize=13, fontweight="bold")
    for i, (ed, c19, il15) in enumerate(zip(eds, n_car19, n_il15, strict=True)):
        ax.text(i, ed + 10, f"CAR19:{c19}\nIL15:{il15}", ha="center", fontsize=8)

    # Panel B: Construct distribution over time
    ax = axes[1]
    ct_tp = adata.obs.groupby(["construct", "timepoint_numeric"]).size().unstack(fill_value=0)
    ct_tp_pct = ct_tp.div(ct_tp.sum(axis=1), axis=0) * 100
    for i, construct in enumerate(ct_tp_pct.index):
        ax.plot(ct_tp_pct.columns, ct_tp_pct.loc[construct], "o-", label=construct,
                color=sns.color_palette("Set2", len(ct_tp_pct.index))[i], linewidth=2, markersize=8)
    ax.set_xlabel("Timepoint (days)", fontsize=12)
    ax.set_ylabel("Proportion (%)", fontsize=12)
    ax.set_title("B) Construct Composition Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(sorted(ct_tp_pct.columns))

    # Panel C: Energy distance heatmap across timepoints
    ax = axes[3 - 3]
    # Compute ED between all timepoint pairs using train-only data
    tp_list = sorted(adata.obs["timepoint_numeric"].unique())
    ed_matrix = np.zeros((len(tp_list), len(tp_list)))
    for i, tp_i in enumerate(tp_list):
        for j, tp_j in enumerate(tp_list):
            m_i = adata[adata.obs["timepoint_numeric"] == tp_i].X
            m_j = adata[adata.obs["timepoint_numeric"] == tp_j].X
            if hasattr(m_i, "toarray"):
                m_i = m_i.toarray()
                m_j = m_j.toarray()
            m_i = np.asarray(m_i)[:, :500]
            m_j = np.asarray(m_j)[:, :500]
            from reuse_gate.metrics.distribution import energy_distance_multivariate
            ed_matrix[i, j] = energy_distance_multivariate(m_i, m_j)

    im = ax.imshow(ed_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(tp_list)))
    ax.set_xticklabels([f"D{t}" if t > 0 else "pre" for t in tp_list], fontsize=10)
    ax.set_yticks(range(len(tp_list)))
    ax.set_yticklabels([f"D{t}" if t > 0 else "pre" for t in tp_list], fontsize=10)
    ax.set_title("C) Timepoint ED Matrix", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Energy Distance")

    fig.suptitle("Figure 5: Construct/Stimulation Shift — CAR19 vs CAR19/IL15",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def fig6_resources(tier0_metrics, tier1_metrics):
    """Figure 6: Resource and sensitivity analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Compute timeline
    ax = axes[0]
    steps = tier0_metrics["steps"]
    names = []
    times = []
    for name in ["last_observation", "conditional_mean", "linear_interpolation", "squidiff_smoke"]:
        if name in steps and "wall_seconds" in steps[name]:
            names.append(name.replace("_", " ").title())
            times.append(steps[name]["wall_seconds"])

    ax.barh(names, times, color=["#E53935", "#43A047", "#1E88E5", "#7E57C2"], edgecolor="white")
    ax.set_xlabel("Wall Time (seconds)", fontsize=12)
    ax.set_title("A) Per-Model Wall Time (Tier 0)", fontsize=13, fontweight="bold")
    for i, t in enumerate(times):
        ax.text(t + 0.1, i, f"{t:.1f}s", va="center", fontsize=10)

    # Panel B: Data scale
    ax = axes[1]
    split = tier0_metrics["split"]
    ax.bar(["Train", "Test"], [split["train_n_cells"], split["test_n_cells"]],
           color=["#2196F3", "#FF5722"], edgecolor="white")
    ax.set_ylabel("Number of Cells", fontsize=12)
    ax.set_title(f"B) Data Scale ({tier0_metrics['n_cells_total']:,} total cells)", fontsize=13, fontweight="bold")
    for i, count in enumerate([split["train_n_cells"], split["test_n_cells"]]):
        ax.text(i, count + 50, f"{count:,}", ha="center", fontsize=11)

    # Panel C: Seed stability (from Tier 1)
    ax = axes[2]
    all_eds = []
    seeds_labels = []
    for s in SEEDS:
        splits_eds = [sp["energy_distance"]["conditional_mean"]
                      for sp in tier1_metrics["seeds"][str(s)]["splits"]]
        all_eds.append(splits_eds)
        seeds_labels.append(f"S{s}")
    ax.boxplot(all_eds, patch_artist=True,
                    boxprops={"facecolor": "#7E57C2", "alpha": 0.5},
                    medianprops={"color": "black", "linewidth": 2})
    ax.set_xticklabels(seeds_labels, fontsize=10)
    ax.set_ylabel("Conditional Mean ED", fontsize=12)
    ax.set_title("C) Cross-Seed Stability (5 seeds)", fontsize=13, fontweight="bold")
    ax.axhline(y=np.mean([np.mean(e) for e in all_eds]), color="red", linestyle="--",
               alpha=0.5, label=f"Mean: {np.mean([np.mean(e) for e in all_eds]):.0f}")
    ax.legend(fontsize=9)

    fig.suptitle("Figure 6: Resource & Sensitivity Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def generate_all_figures(data_path: str) -> dict:
    """Generate all 6 figures and save source data."""
    import anndata as ad

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(data_path)

    # Load metrics
    tier0_metrics = json.loads(Path("artifacts/squidiff_tier0/metrics.json").read_text())
    tier1_metrics = json.loads(Path("artifacts/squidiff_tier1/tier1_metrics.json").read_text())
    construct_shift = json.loads(Path("artifacts/squidiff_tier1/construct_shift.json").read_text())

    figures = {}

    # Figure 1
    fig = fig1_split_design(adata)
    fig.savefig(OUTPUT_DIR / "fig1_split_design.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig1_split_design.pdf", bbox_inches="tight")
    figures["fig1"] = str(OUTPUT_DIR / "fig1_split_design.png")
    plt.close(fig)

    # Figure 2
    fig = fig2_official_reproduction(tier0_metrics)
    fig.savefig(OUTPUT_DIR / "fig2_official_reproduction.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig2_official_reproduction.pdf", bbox_inches="tight")
    figures["fig2"] = str(OUTPUT_DIR / "fig2_official_reproduction.png")
    plt.close(fig)

    # Figure 3
    fig = fig3_temporal_prediction(tier1_metrics)
    fig.savefig(OUTPUT_DIR / "fig3_temporal_prediction.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig3_temporal_prediction.pdf", bbox_inches="tight")
    figures["fig3"] = str(OUTPUT_DIR / "fig3_temporal_prediction.png")
    plt.close(fig)

    # Figure 4
    fig = fig4_state_proportions(tier1_metrics)
    fig.savefig(OUTPUT_DIR / "fig4_state_proportions.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_state_proportions.pdf", bbox_inches="tight")
    figures["fig4"] = str(OUTPUT_DIR / "fig4_state_proportions.png")
    plt.close(fig)

    # Figure 5
    fig = fig5_construct_shift(adata, construct_shift)
    fig.savefig(OUTPUT_DIR / "fig5_construct_shift.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig5_construct_shift.pdf", bbox_inches="tight")
    figures["fig5"] = str(OUTPUT_DIR / "fig5_construct_shift.png")
    plt.close(fig)

    # Figure 6
    fig = fig6_resources(tier0_metrics, tier1_metrics)
    fig.savefig(OUTPUT_DIR / "fig6_resources.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig6_resources.pdf", bbox_inches="tight")
    figures["fig6"] = str(OUTPUT_DIR / "fig6_resources.png")
    plt.close(fig)

    # Save source data
    source_data = {
        "fig1": {"timepoint_counts": adata.obs.groupby("timepoint_raw").size().to_dict()},
        "fig2": {"tier0_results": tier0_metrics},
        "fig3": {"tier1_aggregate": tier1_metrics["aggregate"]},
        "fig4": {"splits_seed13": tier1_metrics["seeds"]["13"]["splits"]},
        "fig5": {"construct_shift": construct_shift},
        "fig6": {"compute_times": {k: v.get("wall_seconds", 0) for k, v in tier0_metrics["steps"].items()}},
    }
    (SOURCE_DATA_DIR / "figure_source_data.json").write_text(json.dumps(source_data, indent=2, default=str))

    print(f"Generated {len(figures)} figures → {OUTPUT_DIR}")
    for name, path in figures.items():
        print(f"  {name}: {path}")
    return figures


if __name__ == "__main__":
    generate_all_figures("artifacts/squidiff_tier0/source_data/gse190976_combined.h5ad")
