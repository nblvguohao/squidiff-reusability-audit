"""Build the three manuscript figures and their source data.

Core conclusion the figure set defends: Squidiff's released artefacts work as
released, but three independent and silent barriers each invert or invalidate
what a reuser concludes.

  Fig 1  schematic-led composite   the assessment, and where reuse fails
  Fig 2  quantitative grid         the release works; the benchmark cannot test it
  Fig 3  asymmetric mixed-modality the two silent barriers, and performance after both

Colour vocabulary is fixed once and reused across every panel: neutral grey for
baselines and context, blue for a correctly configured run, red for the
configuration a reuser lands on by following the documentation, green for the
authors' own released artefacts.

Two deliberate choices about integrity.

Quantities on a logarithmic axis are drawn as points with error bars, never as
bars. A bar encodes value by length from a baseline, and on a log axis that
baseline is arbitrary, so a log-scaled bar chart misstates the comparison it
appears to make.

MMD at the upstream noise scale is not plotted. All five independently trained
models return 0.638179 to six decimals, which is mean(k_xx) alone: the kernel
has saturated and the number measures nothing about generation quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

# ── Mandatory: editable text in SVG ──────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

MM = 1 / 25.4
WIDTH = 180 * MM

C = {
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#A8A8A8",
    "neutral_dark": "#606060",
    "ink": "#272727",
    "correct": "#0F4D92",
    "correct_soft": "#B4C0E4",
    "wrong": "#B64342",
    "wrong_soft": "#E9A6A1",
    "ok": "#2E9E44",
    "ok_soft": "#AADCA9",
}


def panel_label(ax, label, x=-0.085, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom", color=C["ink"])


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in ("svg", "pdf", "png"):
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, dpi=600 if fmt == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _box(ax, x, y, w, h, text, face, edge, fs=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C["ink"], linespacing=1.3)


def _arrow(ax, x1, y1, x2, y2, color=C["neutral_dark"]):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=6.5, linewidth=0.9, color=color))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1
# ─────────────────────────────────────────────────────────────────────────────

def figure1(out_dir: Path) -> dict:
    fig = plt.figure(figsize=(WIDTH, 76 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.10)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ── a: the assessment, with each barrier attached where it actually surfaced ──
    ax_a.set_xlim(0, 10.4)
    ax_a.set_ylim(0, 10)
    ax_a.axis("off")
    panel_label(ax_a, "a", x=-0.01, y=0.99)
    ax_a.text(0.15, 9.45, "Assessment, and where reuse failed", fontsize=7.2,
              fontweight="bold", color=C["ink"], va="bottom")

    # Barriers are numbered in the order a reuser meets them, which is also the
    # order of the workflow, rather than the order we happened to find them.
    steps = [
        (8.15, "Pin upstream  abdfc27", "#FFFFFF", C["neutral_dark"], None),
        (6.55, "Verify released artefacts", "#FFFFFF", C["ok"], None),
        (4.95, "Reproduce simulated\nbenchmark", "#FFFFFF", C["neutral_dark"], None),
        (3.35, "Prepare CAR-NK data", "#FFFFFF", C["neutral_dark"],
         "Barrier 1\npreprocessing not documented"),
        (1.75, "Train with conditioning", "#FFFFFF", C["neutral_dark"],
         "Barrier 2\nconditional branch cannot run"),
        (0.15, "Sample and evaluate", C["correct_soft"], C["correct"],
         "Barrier 3\nsampling constant fails silently"),
    ]
    for y, text, face, edge, barrier in steps:
        _box(ax_a, 0.15, y, 4.35, 1.15, text, face, edge)
        if barrier:
            _box(ax_a, 5.65, y, 4.65, 1.15, barrier, C["wrong_soft"], C["wrong"], fs=6.1)
            _arrow(ax_a, 4.6, y + 0.575, 5.55, y + 0.575, C["wrong"])
    for y in (8.15, 6.55, 4.95, 3.35, 1.75):
        _arrow(ax_a, 2.32, y, 2.32, y - 0.42)

    ax_a.text(5.65, 7.12, "loads, samples, reproduces", fontsize=6.0, color=C["ok"],
              va="center")
    ax_a.text(5.65, 5.52, "degenerate under its\nown preprocessing", fontsize=6.0,
              color=C["neutral_dark"], va="center", linespacing=1.3)

    # ── b: the three defect sites on the conditioning path ──
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis("off")
    panel_label(ax_b, "b", x=-0.02, y=0.99)
    ax_b.text(0.2, 9.45, "Conditioning path", fontsize=7.2,
              fontweight="bold", color=C["ink"], va="bottom")
    ax_b.text(0.2, 8.95, "use_encoder = True, class_cond = True", fontsize=6.0,
              color=C["neutral_dark"], va="bottom")

    stages = [
        (8.00, "AnnData  obs['Group']", None),
        (6.60, "DataLoader collate", "i   labels stay int64\n    Long vs Float"),
        (5.20, "microbatch to device", "ii  labels left on host\n    cpu vs cuda:0"),
        (3.80, "label_embed  Linear(1, h)", "iii rank-1 given,\n    rank-2 required"),
        (2.40, "first optimizer step", None),
    ]
    for y, text, defect in stages:
        face = C["wrong_soft"] if defect else "#FFFFFF"
        edge = C["wrong"] if defect else C["neutral_dark"]
        _box(ax_b, 0.2, y - 0.45, 4.3, 0.90, text, face, edge, fs=6.1)
        if defect:
            ax_b.text(4.85, y, defect, fontsize=5.9, color=C["wrong"],
                      va="center", ha="left", linespacing=1.3)
    for y in (7.55, 6.15, 4.75, 3.35):
        _arrow(ax_b, 2.35, y, 2.35, y - 0.5)

    ax_b.text(0.2, 0.15, "Each raises on the first optimizer step, in this fixed order:\n"
                         "correcting one exposes the next. The released configuration\n"
                         "sets class_cond = False, so this path was never exercised\n"
                         "upstream.",
              fontsize=5.9, color=C["neutral_dark"], va="bottom", linespacing=1.5)

    paths = save(fig, out_dir, "fig1_assessment_and_barriers")
    return {"figure": "fig1", "files": [str(p) for p in paths],
            "note": "schematic only, no measured values plotted"}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2
# ─────────────────────────────────────────────────────────────────────────────

def figure2(root: Path, out_dir: Path) -> dict:
    import anndata as ad

    released = json.loads(
        (root / "artifacts/released_checkpoint/released_checkpoint_check.json").read_text())
    repro = json.loads(
        (root / "artifacts/squidiff_reproduction/reproduction_metrics.json").read_text())

    def load(path):
        a = ad.read_h5ad(path)
        return np.asarray(a.X.toarray() if hasattr(a.X, "toarray") else a.X, dtype=np.float64)

    ref = load(root / "data/raw/upstream_figshare/VO_trained_adata.h5ad")
    gen = np.load(root / "artifacts/released_checkpoint/released_checkpoint_samples.npy")
    carnk_log = load(root / "artifacts/squidiff_sweep_lognorm/train.h5ad")
    carnk_raw = load(root / "artifacts/squidiff_step_sweep/train.h5ad")

    fig = plt.figure(figsize=(WIDTH, 60 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.50)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))

    # ── a: the released checkpoint reproduces its own reference ──
    bins = np.linspace(0, 12, 60)
    ax_a.hist(ref[:512].ravel(), bins=bins, density=True, color=C["neutral_light"],
              edgecolor="none", label="Reference cells")
    ax_a.hist(gen.ravel(), bins=bins, density=True, histtype="step",
              color=C["ok"], linewidth=1.3, label="Generated")
    ax_a.set_xlabel("Expression (log-normalized)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("Released checkpoint reproduces", fontsize=7, pad=14)
    ax_a.legend(loc="upper right", fontsize=6, handlelength=1.3)
    ld, s = released["load_state_dict"], released["sampling"]
    ax_a.text(0.97, 0.62, f"{ld['n_missing']} missing, {ld['n_unexpected']} unexpected keys\n"
                          f"strict load passes\nenergy distance {s['energy_distance']:.2f}",
              transform=ax_a.transAxes, fontsize=5.9, va="top", ha="right",
              color=C["neutral_dark"], linespacing=1.45)
    panel_label(ax_a, "a", x=-0.20, y=1.06)

    # ── b: the scale the method assumes, versus what we first fed it ──
    rows = [
        ("Released\ntraining data", ref, C["ok"]),
        ("CAR-NK\nlog-normalized", carnk_log, C["correct"]),
        ("CAR-NK\nraw counts", carnk_raw, C["wrong"]),
    ]
    ys = np.arange(len(rows))[::-1]
    for yi, (_lab, X, col) in zip(ys, rows, strict=True):
        mean, mx = float(X.mean()), float(X.max())
        ax_b.plot([mean, mx], [yi, yi], color=col, lw=2.2, solid_capstyle="round", zorder=2)
        ax_b.plot(mean, yi, "o", ms=3.6, color=col, zorder=3)
        ax_b.plot(mx, yi, "|", ms=6, mew=1.3, color=col, zorder=3)
        ax_b.text(mx * 1.6, yi + 0.02, f"{mx:,.0f}" if mx > 100 else f"{mx:.1f}",
                  fontsize=6.0, va="center", color=col)
    ax_b.set_xscale("log")
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels([r[0] for r in rows], fontsize=6.0, linespacing=1.3)
    ax_b.set_xlabel("Expression value")
    ax_b.set_xlim(0.4, 3e5)
    ax_b.set_ylim(-0.6, 2.6)
    ax_b.set_title("Scale the method assumes", fontsize=7, pad=6)
    ax_b.text(0.98, 0.06, "dot, mean\ntick, maximum", transform=ax_b.transAxes,
              fontsize=5.8, va="bottom", ha="right", color=C["neutral_dark"],
              linespacing=1.45)
    panel_label(ax_b, "b", x=-0.34, y=1.06)

    # ── c: the simulated benchmark loses its only signal ──
    deg = repro["preprocessing_degeneracy"]
    keys = ["raw simulation", "after normalize_total", "after normalize + log1p"]
    short = ["raw\nsimulation", "after\nnormalize_total", "after\n+ log1p"]
    sils = [deg[k]["silhouette"] for k in keys]
    cols = [C["ok"] if v > 0.1 else C["wrong"] for v in sils]
    xs = np.arange(len(keys))
    ax_c.bar(xs, sils, 0.5, color=cols, edgecolor=C["ink"], linewidth=0.7)
    ax_c.axhline(0, color=C["ink"], lw=0.8)
    # Per-type means ride along with the stage label, so nothing floats free
    # inside the axes where it could collide with a neighbouring column.
    ticks = []
    for k, lab in zip(keys, short, strict=True):
        means = deg[k]["per_type_mean"]
        ticks.append(lab + "\n" + " / ".join(f"{m:.2g}" for m in means))
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(ticks, fontsize=5.6, linespacing=1.5)
    ax_c.set_ylabel("Silhouette, cell-type separability")
    ax_c.set_ylim(-0.16, 0.60)
    ax_c.set_title("Upstream simulated benchmark", fontsize=7, pad=6)
    for x, v in zip(xs, sils, strict=True):
        ax_c.text(x, v + (0.025 if v > 0 else -0.025), f"{v:+.3f}", ha="center",
                  va="bottom" if v > 0 else "top", fontsize=6.0, color=C["ink"])
    ax_c.text(0.5, -0.30, "third line: mean expression of the three cell types",
              transform=ax_c.transAxes, fontsize=5.7, ha="center", color=C["neutral_dark"])
    panel_label(ax_c, "c", x=-0.22, y=1.06)

    paths = save(fig, out_dir, "fig2_release_works_benchmark_does_not")
    return {
        "figure": "fig2",
        "files": [str(p) for p in paths],
        "a_released_sampling": s,
        "a_load_state_dict": ld,
        "b_data_scale": [{"dataset": r[0].replace("\n", " "), "mean": float(r[1].mean()),
                          "max": float(r[1].max())} for r in rows],
        "c_preprocessing_degeneracy": deg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3
# ─────────────────────────────────────────────────────────────────────────────

def figure3(root: Path, out_dir: Path) -> dict:
    # Main panel: the published-protocol latent-extrapolation A/B. The
    # class-conditional probe (train_step_sweep) is retained only as labelled
    # corroboration — Barrier 2 argues that branch cannot run as released, so
    # the main-text evidence for Barrier 1 must not rest on it.
    ab = json.loads(
        (root / "artifacts/squidiff_latent_extrap_ab/preprocessing_ab_metrics.json").read_text())
    probe_raw = json.loads((root / "artifacts/squidiff_step_sweep/sweep_metrics.json").read_text())
    probe_logn = json.loads(
        (root / "artifacts/squidiff_sweep_lognorm/sweep_metrics.json").read_text())
    noise = json.loads(
        (root / "artifacts/squidiff_latent_extrap/latent_noise_scale_sweep.json").read_text())
    seeds = json.loads(
        (root / "artifacts/squidiff_seed_study/seed_study_metrics.json").read_text())

    fig = plt.figure(figsize=(WIDTH, 118 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.86], width_ratios=[1.3, 1.0, 1.0],
                          hspace=0.62, wspace=0.50)
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # ── a: preprocessing decides the direction of the trend (hero) ──
    steps = [e["steps"] for e in ab["conditions"]["raw"]["per_budget"]]
    ed_raw = [e["pooled_energy_distance"] for e in ab["conditions"]["raw"]["per_budget"]]
    ed_log = [e["pooled_energy_distance"] for e in ab["conditions"]["lognorm"]["per_budget"]]
    ax_a.plot(steps, ed_raw, "o-", color=C["wrong"], lw=1.7, ms=4.6,
              label="Raw counts, what the code path leaves you to guess")
    ax_a.plot(steps, ed_log, "o-", color=C["correct"], lw=1.7, ms=4.6,
              label="normalize_total + log1p, what the method expects")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Training steps")
    ax_a.set_ylabel("Energy distance to held-out cells")
    ax_a.set_xticks(steps)
    ax_a.set_xticklabels([f"{s // 1000}k" for s in steps])
    ax_a.set_ylim(4, 900)
    ax_a.legend(loc="lower left", fontsize=6, handlelength=1.6)
    ax_a.set_title("Preprocessing decides whether more training helps or hurts",
                   fontsize=7.4, pad=14)
    ax_a.text(0.0, 1.015, "published latent-extrapolation protocol, noise scale fixed at "
                          "0.03; identical code, data, split and seed",
              transform=ax_a.transAxes, fontsize=5.9, va="bottom", color=C["neutral_dark"])
    ax_a.annotate(f"{ed_raw[-1]:.0f}", (steps[-1], ed_raw[-1]), textcoords="offset points",
                  xytext=(7, 1), fontsize=6.2, color=C["wrong"])
    ax_a.annotate(f"{ed_log[-1]:.2f}", (steps[-1], ed_log[-1]), textcoords="offset points",
                  xytext=(7, -1), fontsize=6.2, color=C["correct"])
    panel_label(ax_a, "a", x=-0.075, y=1.10)

    # ── b: one hardcoded constant spans 47-fold ──
    sc = sorted(noise["scales"], key=lambda e: e["scale"])
    ZERO_POS = 0.008   # where a scale of exactly 0 is drawn on the log axis
    xs = [e["scale"] if e["scale"] > 0 else ZERO_POS for e in sc]
    ys = [e["pooled_energy_distance"] for e in sc]
    ax_b.plot(xs, ys, "o-", color=C["ink"], lw=1.4, ms=4, zorder=3)
    up = next(e for e in sc if e["is_upstream_default"])
    ax_b.plot([up["scale"]], [up["pooled_energy_distance"]], "o", ms=8,
              mfc="none", mec=C["wrong"], mew=1.6, zorder=4)
    ax_b.annotate("upstream\ndefault", (up["scale"], up["pooled_energy_distance"]),
                  textcoords="offset points", xytext=(-2, -26), fontsize=6.0,
                  color=C["wrong"], ha="center", linespacing=1.3)
    for s0 in sorted(set(seeds["summary"]["selected_scales"])):
        ax_b.axvline(s0 if s0 > 0 else ZERO_POS, color=C["correct"], ls=":", lw=1.0, zorder=1)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Latent noise scale")
    ax_b.set_ylabel("Energy distance")
    ax_b.set_xlim(0.005, 1.6)
    ax_b.set_title("One hardcoded constant", fontsize=7, pad=14)
    ax_b.text(0.0, 1.015, "dotted, chosen on validation", transform=ax_b.transAxes,
              fontsize=5.9, va="bottom", color=C["correct"])
    ax_b.set_xticks([ZERO_POS, 0.1, 1.0])
    ax_b.set_xticklabels(["0", "0.1", "1"])
    panel_label(ax_b, "b", x=-0.34, y=1.10)

    # ── c: five seeds, three metrics ──
    # Baselines come from the provenance audit (Phase 1.1), not the seed-study
    # metrics file: the manuscript text cites the true D14-resample
    # last-observation, not the stale point-mass variant.
    prov = json.loads(
        (root / "artifacts/baseline_provenance/baseline_provenance.json").read_text())
    S = seeds["summary"]
    base = {
        "conditional_mean": prov["baselines"]["conditional_mean"]["scores"],
        "last_observation": prov["baselines"]["last_observation_true_d14_resample"]["scores"],
    }
    # Same-distribution reference band and structure metrics (Phase 2). The
    # final panel c needs both: MMD is demoted to a supplementary sensitivity
    # analysis because its Squidiff-baseline ordering flips with bandwidth,
    # and the middle panel is the gene-gene correlation structure distance.
    rob = json.loads((root / "artifacts/evaluation_robustness/robustness.json").read_text())
    null_anchor = rob["null_anchor"]
    structure = rob["structure"]
    seed_order = [13, 37, 73, 101, 137]
    frob_vals = [structure[f"squidiff_seed_{s}"]["correlation_frobenius"] for s in seed_order]
    metrics = [
        ("energy_distance", "Energy distance", True, "lower is better"),
        ("correlation_frobenius", "Gene–gene correlation distance", False, "lower is better"),
        ("mean_expression_correlation", "Per-gene mean correlation", False, "higher is better"),
    ]
    jitter = np.random.RandomState(0).normal(0, 0.05, 5)
    for ax, (key, title, logscale, hint) in zip(ax_c, metrics, strict=True):
        if key == "correlation_frobenius":
            # Structure values live in the robustness pass, not the seed study.
            vals = frob_vals
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            base_pair = (structure["conditional_mean"]["correlation_frobenius"],
                         structure["last_observation_d14"]["correlation_frobenius"])
        else:
            vals = S[f"validation_selected.{key}"]["values"]
            mean, sd = S[f"validation_selected.{key}"]["mean"], S[f"validation_selected.{key}"]["std"]
            base_pair = (base["conditional_mean"][key], base["last_observation"][key])
        # Points, not bars: a bar on a log axis encodes length from an arbitrary floor.
        ax.errorbar([0], [mean], yerr=[sd], fmt="o", ms=6, color=C["correct"],
                    ecolor=C["correct"], elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=3)
        ax.plot(jitter, vals, "o", ms=2.8, mfc=C["correct_soft"], mec=C["correct"],
                mew=0.5, zorder=2)
        for value, col, ls in zip(base_pair, (C["ink"], C["neutral_mid"]), ("-", "--"), strict=True):
            ax.axhline(value, color=col, ls=ls, lw=1.0, zorder=1)
        if key in null_anchor:
            n = null_anchor[key]
            lo_band, hi_band = (n["q05"], 1.0) if key == "mean_expression_correlation" \
                else (0.0, n["q95"])
            ax.axhspan(lo_band, hi_band, color=C["neutral_light"], alpha=0.45, zorder=0)
            ax.axhline(n["mean"], color=C["neutral_dark"], ls=":", lw=0.9, zorder=1)
        if logscale:
            ax.set_yscale("log")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(title, fontsize=6.9, pad=6)
        ax.text(0.5, -0.13, hint, transform=ax.transAxes, fontsize=5.8,
                color=C["neutral_dark"], ha="center")
    ax_c[0].set_ylabel("Value across 5 seeds")
    ax_c[0].set_ylim(0.02, 40)
    ax_c[1].set_ylim(0, 300)
    ax_c[2].set_ylim(0.75, 1.0)
    # One shared key rather than three sets of colliding inline labels.
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C["correct"], ms=5,
                   label="Squidiff, 5 seeds"),
        plt.Line2D([], [], color=C["ink"], lw=1.0, label="conditional-mean"),
        plt.Line2D([], [], color=C["neutral_mid"], ls="--", lw=1.0,
                   label="last-observation (D14 resample)"),
    ]
    if null_anchor is not None:
        handles.append(Patch(facecolor=C["neutral_light"], alpha=0.45, edgecolor="none",
                             label="same-distribution band"))
    ax_c[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                   ncol=2, fontsize=6, handlelength=1.5, columnspacing=1.4)
    # The conditional-mean baseline has zero gene-gene covariance by
    # construction, so this is the one metric it cannot win regardless of
    # centering; last-observation wins by construction too, since it is real
    # resampled cells. The caveat sits beside the panel it applies to.
    ax_c[1].text(0.5, 0.92, "solid line: zero covariance by construction",
                 transform=ax_c[1].transAxes, fontsize=5.5, color=C["neutral_dark"],
                 ha="center", va="center", linespacing=1.4)
    panel_label(ax_c[0], "c", x=-0.38, y=1.08)

    paths = save(fig, out_dir, "fig3_barriers_and_performance")
    return {
        "figure": "fig3",
        "files": [str(p) for p in paths],
        "a_preprocessing_ab": {"steps": steps, "raw_counts": ed_raw, "log_normalized": ed_log},
        "a_protocol": ("latent extrapolation, released configuration, fixed noise "
                       f"scale {ab['fixed_noise_scale']} (single-variable A/B)"),
        "a_class_conditional_probe": {
            "note": ("corroboration only: class-conditional probe from "
                     "train_step_sweep.py, held fixed across preprocessing "
                     "conditions; not the main evidence, since Barrier 2 shows "
                     "this branch cannot run as released"),
            "steps": [e["steps"] for e in probe_logn["sweep"]],
            "raw_counts": [e["energy_distance"] for e in probe_raw["sweep"]],
            "log_normalized": [e["energy_distance"] for e in probe_logn["sweep"]],
        },
        "b_noise_scale": sc,
        "b_validation_selected_scales": seeds["summary"]["selected_scales"],
        "c_metrics": [m[0] for m in metrics],
        "c_seed_summary": {k: v for k, v in S.items() if k != "selected_scales"},
        "c_baselines": base,
        "c_null_anchor": null_anchor,
        "c_structure": {
            "squidiff_values": frob_vals,
            "conditional_mean": structure["conditional_mean"]["correlation_frobenius"],
            "last_observation_d14": structure["last_observation_d14"]["correlation_frobenius"],
            "rare_cluster_recall": {
                "conditional_mean": structure["conditional_mean"]["rare_cluster_recall"],
                "last_observation_d14": structure["last_observation_d14"]["rare_cluster_recall"],
                "squidiff_seeds": [structure[f"squidiff_seed_{s}"]["rare_cluster_recall"]
                                   for s in seed_order],
            },
        },
        "excluded": "MMD at scale 0.7 is saturated (0.638179 for all five seeds, equal to "
                    "mean(k_xx)) and is deliberately not plotted; MMD is demoted to a "
                    "supplementary bandwidth-sensitivity check (Supplementary Table 4) "
                    "because its Squidiff-vs-baseline ordering flips across the bandwidth grid",
    }


def main(root: Path, out_dir: Path) -> None:
    source = {
        "dataset": "GSE190976, mouse CAR-NK, 16,256 cells",
        "split": "train pre/D7/D14, 11,588 cells, 13 samples; "
                 "held out D21/D28, 4,668 cells, 5 samples; no sample on both sides",
        "preprocessing": "normalize_total(1e4) + log1p, top-500 HVG fitted on training cells only",
        "seeds": [13, 37, 73, 101, 137],
        "error_bars": "standard deviation across the five seeds",
        "released_artefacts": "figshare 10.6084/m9.figshare.27948633, CC BY 4.0",
        "figures": [],
    }
    for name, fn in (("Figure 1", lambda: figure1(out_dir)),
                     ("Figure 2", lambda: figure2(root, out_dir)),
                     ("Figure 3", lambda: figure3(root, out_dir))):
        print(f"{name} ...")
        source["figures"].append(fn())

    sd = out_dir / "source_data.json"
    sd.write_text(json.dumps(source, indent=2, default=str))
    print(f"\nSource data: {sd}")
    for f in source["figures"]:
        for p in f["files"]:
            print(f"  {p}")


if __name__ == "__main__":
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/manuscript_figures")
    main(repo, out)
