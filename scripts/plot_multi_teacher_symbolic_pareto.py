"""Plot the ten-split, six-budget symbolic teacher/direct comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from manuscript_plot_palette import GRID, INK, MODEL_COLORS, MUTED


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts" / "results" / "multi_teacher_symbolic_pareto_10split" / "aggregate_by_export.csv"
OUTPUT = ROOT / "paper" / "paper_figure_package" / "figure" / "fig05_multi_teacher_symbolic_pareto"
SUBMISSION_OUTPUT = ROOT / "paper" / "paper_figure_package" / "figure"

TASKS = ["I_dark", "I_photo", "AC_Response", "Capacitance"]
TASK_LABELS = {
    "I_dark": "Dark current",
    "I_photo": "Net photocurrent",
    "AC_Response": "Optical SSAC response",
    "Capacitance": "Apparent capacitance",
}
SOURCES = ["tcad_direct", "bkan_parameter_mean", "dkan", "mlp_l"]
SOURCE_LABELS = {
    "tcad_direct": "TCAD direct",
    "bkan_parameter_mean": "BKAN teacher",
    "dkan": "DKAN teacher",
    "mlp_l": "MLP teacher",
}
COLORS = {
    source: MODEL_COLORS[source] for source in SOURCES
}
MARKERS = {
    "tcad_direct": "o",
    "bkan_parameter_mean": "s",
    "dkan": "^",
    "mlp_l": "D",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.4,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "axes.linewidth": 0.75,
        }
    )


def main() -> None:
    configure_style()
    frame = pd.read_csv(INPUT)
    frame["pareto_mean_rmse_terms"] = frame["pareto_mean_rmse_terms"].astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=False)
    for panel, (ax, task) in enumerate(zip(axes.flat, TASKS, strict=True)):
        task_frame = frame.loc[frame["task"] == task].copy()
        scale = 1.0e16 if task == "Capacitance" else 1.0
        for source in SOURCES:
            rows = task_frame.loc[task_frame["source"] == source].sort_values("terms")
            x = rows["terms"].to_numpy()
            y = rows["mean_rmse"].to_numpy() * scale
            yerr = rows["sd_rmse"].to_numpy() * scale
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=COLORS[source],
                marker=MARKERS[source],
                markersize=4.2,
                linewidth=1.05,
                elinewidth=0.7,
                capsize=1.8,
                alpha=0.92,
                label=SOURCE_LABELS[source],
                zorder=2,
            )
        front = task_frame.loc[task_frame["pareto_mean_rmse_terms"]]
        ax.scatter(
            front["terms"],
            front["mean_rmse"] * scale,
            s=49,
            facecolors="none",
            edgecolors=INK,
            linewidths=1.0,
            zorder=4,
        )
        ax.set_yscale("log")
        ax.set_title(TASK_LABELS[task], pad=4)
        ax.set_xlabel("Exported terms")
        if panel % 2 == 0:
            ax.set_ylabel("Held-out RMSE")
        if task == "Capacitance":
            ax.set_ylabel(r"Held-out RMSE ($\times 10^{-16}$ F)")
        ax.grid(True, which="major", color=GRID, linewidth=0.55)
        ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.5,
            -0.32,
            f"({chr(ord('a') + panel)})",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="top",
            clip_on=False,
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    pareto_handle = plt.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor=INK,
        markeredgewidth=1.0,
        markersize=6.5,
        label="Pareto point",
    )
    panel_legends = []
    for ax in axes.flat:
        panel_legends.append(
            ax.legend(
                handles + [pareto_handle],
                labels + ["Pareto point"],
                loc="best",
                fontsize=7.2,
                ncol=1,
            )
        )
    SUBMISSION_OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    all_axes = list(axes.flat)
    original_visibility = [ax.get_visible() for ax in all_axes]
    for ax, stem in zip(
        all_axes,
        [
            "fig05a_symbolic_dark_current",
            "fig05b_symbolic_net_photocurrent",
            "fig05c_symbolic_optical_ssac",
            "fig05d_symbolic_capacitance",
        ],
        strict=True,
    ):
        for item in all_axes:
            item.set_visible(item is ax)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(
            SUBMISSION_OUTPUT / f"{stem}.pdf",
            bbox_inches=bbox,
            pad_inches=0.06,
            facecolor="white",
        )
    for ax, visible in zip(all_axes, original_visibility, strict=True):
        ax.set_visible(visible)
    for legend in panel_legends:
        legend.remove()
    fig.legend(
        handles + [pareto_handle],
        labels + ["Pareto point"],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.15,
        handletextpad=0.45,
    )
    fig.suptitle(
        "Matched symbolic export: fidelity–complexity trade-off",
        x=0.07,
        y=0.99,
        ha="left",
        fontsize=10.8,
        fontweight="semibold",
    )
    fig.text(
        0.07,
        0.955,
        "Mean ± SD over ten grouped splits; black rings mark the aggregate Pareto front.",
        ha="left",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.18, hspace=0.58, wspace=0.30)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(
        SUBMISSION_OUTPUT / "fig05_multi_teacher_symbolic_pareto.pdf",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
