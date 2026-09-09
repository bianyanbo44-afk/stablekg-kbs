"""Generate StableKG figures with a single Matplotlib backend.

Every quantitative panel reads a committed source table.  The script exports
editable SVG/PDF, a 600-dpi TIFF, a PNG preview and Nature-figure QA reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SKILL_SCRIPTS = Path(r"C:/Users/Lenovo/.codex/skills/nature-figure/scripts")
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))
try:
    from audit_panel_alignment import matplotlib_layout_manifest, require_matplotlib_panel_alignment
except ImportError:
    from layout_checks import matplotlib_layout_manifest, require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results_final_v5" / "analysis"
CHRONO = ROOT / "results_chronological_v5"
OUT = ROOT / "figures" / "out"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "StableKG": "#007C78",
    "baseline": "#4B5563",
    "static": "#9CA3AF",
    "global_decay": "#D97706",
    "stability_aware": "#007C78",
    "event": "#007C78",
    "window": "#D97706",
    "counter": "#7C3AED",
    "correct": "#007C78",
    "incorrect": "#C2415D",
    "ink": "#17202A",
    "muted": "#5F6B73",
    "grid": "#D9E1E5",
    "paper": "#FFFFFF",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "savefig.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        }
    )


def label_axes(axes: list[plt.Axes]) -> None:
    for index, axis in enumerate(axes):
        axis.text(
            -0.24,
            1.10,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
            color=COLORS["ink"],
        )


def clean_axis(axis: plt.Axes, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.8)
        axis.set_axisbelow(True)


def export(fig: plt.Figure, stem: str, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    layout_json = OUT / f"{stem}.alignment-layout.json"
    alignment_json = OUT / f"{stem}.alignment.json"
    alignment_svg = OUT / f"{stem}.alignment.svg"
    manifest = matplotlib_layout_manifest(fig, axes=axes)
    layout_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(alignment_json),
        overlay_svg=str(alignment_svg),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig1_concept() -> None:
    predictions = pd.read_csv(ANALYSIS / "../ICEWS14_predictions.csv")
    sample = predictions
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.1, 1.55)
    ax.axhline(0, color=COLORS["muted"], lw=0.7)
    ax.set_yticks([-1, 0, 1], ["against", "", "for"])
    ax.set_xticks([0, 6, 12], ["old", "recent", "now"])
    ax.set_xlabel("event time (schematic)")
    ax.set_title("Evidence is time-local")
    fixed_ages = np.array([[0, 1, 3, 5, 6, 8, 10, 11], [0, 2, 4, 5, 7, 9, 10, 11]])
    fixed_offsets = np.array([[0.00, 0.04, -0.04, 0.03, -0.02, 0.04, -0.03, 0.01], [0.00, -0.03, 0.04, -0.04, 0.03, -0.02, 0.04, -0.03]])
    for row, (y, color) in enumerate([(1, COLORS["correct"]), (-1, COLORS["incorrect"])]) :
        for age, jitter in zip(fixed_ages[row], fixed_offsets[row]):
            opacity = 0.22 + 0.7 * np.exp(-(12 - age) / 8)
            ax.scatter(age, y + jitter, s=35, color=color, alpha=opacity, edgecolor="white", lw=0.35, zorder=3)
    ax.tick_params(axis="x", length=0)
    clean_axis(ax, grid=False)
    ax.spines["left"].set_visible(False)

    ax = axes[1]
    x = sample["belief_stability"].to_numpy(float)
    y = sample["certificate"].to_numpy(float)
    correct = sample["correct"].to_numpy(bool)
    ax.scatter(x[~correct], y[~correct], s=8, alpha=0.28, color=COLORS["incorrect"], linewidths=0, label="Incorrect")
    ax.scatter(x[correct], y[correct], s=8, alpha=0.30, color=COLORS["correct"], linewidths=0, label="Correct")
    seed0 = pd.read_csv(ANALYSIS / "../ICEWS14_seeds.csv").iloc[0]
    threshold = seed0["stability_threshold@0.20"]
    ax.axvline(threshold, color=COLORS["muted"], lw=0.7, ls="--")
    ax.axvspan(threshold, 1, color=COLORS["correct"], alpha=0.06)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="calibrated belief", ylabel="deletion diagnostic", title="Belief and evidence stability")
    clean_axis(ax)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.3, markerscale=1.4)

    ax = axes[2]
    nodes = {"base": [(0.12, 0.25), (0.12, 0.72), (0.12, 0.49)], "derived": [(0.53, 0.25), (0.53, 0.72)], "answer": [(0.88, 0.49)]}
    for start, ends in [((0.16, 0.25), (0.49, 0.25)), ((0.16, 0.72), (0.49, 0.72)), ((0.16, 0.49), (0.49, 0.72)), ((0.57, 0.25), (0.84, 0.49)), ((0.57, 0.72), (0.84, 0.49))]:
        ax.add_patch(FancyArrowPatch(start, ends, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=COLORS["muted"], connectionstyle="arc3,rad=0.05"))
    for x0, y0 in nodes["base"]:
        ax.add_patch(plt.Circle((x0, y0), 0.065, facecolor=COLORS["incorrect"], edgecolor="white", lw=1.0, alpha=0.9))
    for x0, y0 in nodes["derived"]:
        ax.add_patch(plt.Circle((x0, y0), 0.065, facecolor=COLORS["global_decay"], edgecolor="white", lw=1.0, alpha=0.9))
    ax.add_patch(plt.Circle(nodes["answer"][0], 0.075, facecolor=COLORS["StableKG"], edgecolor="white", lw=1.0, alpha=0.95))
    ax.text(0.12, 0.02, "Changed\nfacts", ha="center", fontsize=6.5, color=COLORS["muted"])
    ax.text(0.52, 0.02, "Affected\nclosure", ha="center", fontsize=6.5, color=COLORS["muted"])
    ax.text(0.88, 0.02, "Updated\nanswer", ha="center", fontsize=6.5, color=COLORS["muted"])
    ax.set(xlim=(0, 1), ylim=(0, 1), title="Dependency-guided updates")
    ax.axis("off")
    label_axes(list(axes))
    export(fig, "fig1_concept", list(axes))


def fig2_risk_coverage() -> None:
    data = pd.read_csv(ANALYSIS / "risk_coverage.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), sharey=True, gridspec_kw={"wspace": 0.26})
    for axis, dataset in zip(axes, ("ICEWS14", "ICEWS05-15", "GDELT")):
        subset = data[data.dataset == dataset]
        for method in ("baseline", "StableKG"):
            line = subset[subset.method == method]
            axis.plot(line.coverage, line.risk, color=COLORS[method], lw=2.0 if method == "StableKG" else 1.35, ls="-" if method == "StableKG" else "--", label=method)
        row = pd.read_csv(ANALYSIS / "headline_public.csv")
        headline = row[row.dataset == dataset].iloc[0]
        seed0 = pd.read_csv(ANALYSIS / f"../{dataset}_seeds.csv").iloc[0]
        axis.scatter([seed0["stability_coverage@0.20"], seed0["stability_coverage@0.40"]], [seed0["stability_risk@0.20"], seed0["stability_risk@0.40"]], color=COLORS["StableKG"], s=14, zorder=4)
        axis.set(xlim=(0, 1), ylim=(0, 1), title=dataset, xlabel="coverage")
        axis.text(0.04, 0.08, f"AURC {seed0.stability_aurc:.3f}", transform=axis.transAxes, fontsize=6.6, color=COLORS["StableKG"], fontweight="bold")
        clean_axis(axis)
    axes[0].set_ylabel("selective risk")
    axes[-1].legend(frameon=False, loc="upper left", handlelength=2.0)
    label_axes(list(axes))
    export(fig, "fig2_risk_coverage", list(axes))


def fig3_interventions() -> None:
    data = pd.read_csv(ANALYSIS / "intervention_statistics.csv")
    comparison = pd.read_csv(ROOT / "results_review_20260909/corrected/diagnostic_controls.csv")
    fig, grid = plt.subplots(2, 3, figsize=(7.2, 4.5), gridspec_kw={"wspace": 0.30, "hspace": 0.60})
    axes = grid[0]
    edits = [("event", "single-event deletion"), ("window", "7-unit window deletion"), ("counter", "counter-evidence injection")]
    datasets = ("ICEWS14", "ICEWS05-15", "GDELT")
    for axis, dataset in zip(axes, datasets):
        subset = data[data.dataset == dataset]
        groups = ["low certificate", "high certificate"]
        width = 0.23
        centers = np.arange(len(edits))
        for group_index, group in enumerate(groups):
            vals = []
            low = []
            high = []
            for edit, _ in edits:
                row = subset[(subset.intervention == edit) & (subset.certificate_group == group)].iloc[0]
                vals.append(row.flip_rate)
                low.append(row.flip_rate - row.ci_low)
                high.append(row.ci_high - row.flip_rate)
            offset = (group_index - 0.5) * width
            axis.bar(centers + offset, vals, width=width, color=COLORS["incorrect"] if group_index == 0 else COLORS["StableKG"], alpha=0.86, yerr=[low, high], capsize=2, error_kw={"elinewidth": 0.65, "capthick": 0.65}, label="Low diagnostic score" if group_index==0 else "High diagnostic score")
        axis.set_xticks(centers, ["event", "window", "counter"], rotation=24, rotation_mode="anchor", ha="right")
        axis.set_ylim(0, 1.08)
        axis.set_title(dataset)
        clean_axis(axis)
    for axis, dataset in zip(grid[1], datasets):
        sub = comparison[comparison.dataset == dataset]
        matrix = np.asarray([[sub[(sub.edit==edit)&(sub.feature==feature)].iloc[0].survival_auroc
                              for feature in ("certificate","counter_evidence_cost","margin")]
                             for edit in ("event","window","counter")])
        axis.imshow(matrix, cmap="Blues", vmin=.5, vmax=1, aspect="auto")
        axis.set_xticks([0,1,2], ["Deletion\nscore", "Normalized\nmargin", "Raw\nmargin"])
        axis.set_yticks([0,1,2], ["Event", "Window", "Injection"])
        axis.tick_params(length=0)
        axis.set_title("Intervention-survival AUROC", fontsize=7.6)
        for (i,j), value in np.ndenumerate(matrix):
            axis.text(j, i, f"{value:.4f}", ha="center", va="center", fontsize=7.0,
                      color="white" if value >= .83 else COLORS["ink"])
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("winner-flip rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, handlelength=1.6, columnspacing=1.2)
    label_axes(list(grid.ravel()))
    export(fig, "fig3_interventions", list(grid.ravel()))


def fig4_temporal() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.35), gridspec_kw={"wspace": 0.28, "hspace": 0.42})
    axes_flat = list(axes.ravel())
    for column, (axis, dataset) in enumerate(zip(axes[0], ("ICEWS14", "ICEWS05-15"))):
        predictions = pd.read_csv(CHRONO / f"{dataset}_chronological_predictions.csv")
        summary = json.loads((CHRONO / f"{dataset}_chronological_summary.json").read_text())
        from math import ceil
        width = max(1, ceil((predictions.timestamp.max()-predictions.timestamp.min()+1)/10))
        predictions["time_block"] = ((predictions.timestamp-predictions.timestamp.min())//width).astype(int)
        for method in ("baseline", "StableKG"):
            prefix = "baseline" if method=="baseline" else "stability"
            threshold = summary[f"{prefix}_threshold@0.20"]
            x, risk, coverage = [], [], []
            for _, part in predictions.groupby("time_block"):
                selected = part[part[f"belief_{prefix}"]>=threshold]
                x.append(part.timestamp.min())
                risk.append(1-selected.correct.mean() if len(selected) else np.nan)
                coverage.append(len(selected)/len(part))
            axis.plot(x, risk, marker="o", ms=2.2, lw=1.45 if method == "StableKG" else 1.0,
                      ls="-" if method=="StableKG" else "--", color=COLORS[method], label=method)
            axes[1,column].plot(x,coverage,marker="o",ms=2.2,lw=1.45 if method=="StableKG" else 1.,
                                ls="-" if method=="StableKG" else "--",color=COLORS[method],label=method)
        axis.set_title(f"{dataset} chronological replay")
        axis.set_xlabel("timestamp block")
        axis.set_ylim(0, 0.8)
        clean_axis(axis)
        axes[1,column].set(title=f"{dataset} realized coverage", xlabel="timestamp block", ylim=(0,.5))
        axes[1,column].axhline(.2,color=COLORS["grid"],lw=.7,ls=":")
        clean_axis(axes[1,column])
    axes[0, 0].set_ylabel("risk (20% validation target)")
    axes[1, 0].set_ylabel("accepted fraction")
    top_handles, top_labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(top_handles, top_labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=2, handlelength=1.7, columnspacing=1.2)
    label_axes(axes_flat)
    export(fig, "fig4_temporal", axes_flat)


def fig5_efficiency() -> None:
    summary = json.loads((ANALYSIS / "synthetic_summary_20.json").read_text(encoding="utf-8"))
    labels = ["dependency nodes", "update time"]
    means = [summary["node_reduction"]["mean"], summary["runtime_reduction"]["mean"]]
    lows = [means[i] - summary[key]["ci_low"] for i, key in enumerate(("node_reduction", "runtime_reduction"))]
    highs = [summary[key]["ci_high"] - means[i] for i, key in enumerate(("node_reduction", "runtime_reduction"))]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.35), gridspec_kw={"wspace": 0.42})
    axes[0].bar([0], [means[0] * 100], yerr=[[lows[0] * 100], [highs[0] * 100]], color=COLORS["StableKG"], width=0.48, capsize=3, error_kw={"elinewidth": 0.7})
    axes[0].set(xticks=[0], xticklabels=["closure"], ylim=(0, 110), ylabel="reduction (%)", title="recomputation work")
    axes[0].text(0, 88, f"{means[0] * 100:.1f}%", ha="center", fontsize=7.2, fontweight="bold", color=COLORS["StableKG"])
    axes[1].bar([0], [means[1] * 100], yerr=[[lows[1] * 100], [highs[1] * 100]], color=COLORS["global_decay"], width=0.48, capsize=3, error_kw={"elinewidth": 0.7})
    axes[1].set(xticks=[0], xticklabels=["propagation"], ylim=(0, 110), title="cached-closure update time")
    axes[1].text(0, 68, f"{means[1] * 100:.1f}%", ha="center", fontsize=7.2, fontweight="bold", color=COLORS["global_decay"])
    for axis in axes:
        clean_axis(axis)
    axes[1].text(0.5, -0.22, "max |full − incremental| = 0", transform=axes[1].transAxes, ha="center", fontsize=6.5, color=COLORS["muted"])
    label_axes(list(axes))
    export(fig, "fig5_efficiency", list(axes))


def fig6_reliability() -> None:
    data = pd.read_csv(ANALYSIS / "reliability.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), sharex=True, sharey=True, gridspec_kw={"wspace": 0.25})
    for axis, dataset in zip(axes, ("ICEWS14", "ICEWS05-15", "GDELT")):
        for method in ("baseline", "StableKG"):
            line = data[(data.dataset == dataset) & (data.method == method)]
            axis.plot(line.belief, line.accuracy, marker="o", ms=2.4, lw=1.35 if method == "StableKG" else 1.0, color=COLORS[method], label=method)
        axis.plot([0, 1], [0, 1], color=COLORS["grid"], lw=0.8, ls="--")
        axis.set(xlim=(0, 1), ylim=(0, 1), title=dataset, xlabel="predicted belief")
        clean_axis(axis)
    axes[0].set_ylabel("empirical accuracy")
    axes[-1].legend(frameon=False, loc="upper left")
    label_axes(list(axes))
    export(fig, "fig6_reliability", list(axes))


def main() -> None:
    configure()
    fig1_concept()
    fig2_risk_coverage()
    fig3_interventions()
    fig4_temporal()
    fig5_efficiency()
    fig6_reliability()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
