"""Build the compact APD paired-evidence figure from frozen results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from manuscript_plot_palette import BLUE, GREEN, GRID, INK, MAGENTA, MUTED


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts/results/apd_second_device"
PREDICTIONS = RESULTS / "predictions/seed_42"
FIGURE_OUT = ROOT / "paper/paper_figure_package/figure"
SELECTION_OUT = RESULTS / "apd_paired_evidence_selection.csv"

BKAN_COLOR = GREEN
REFERENCE_COLOR = INK
POWER_COLORS = {1e-7: BLUE, 3e-7: MAGENTA}

TASKS = [
    ("APD_I_dark", "Dark current"),
    ("APD_I_photo", "Illuminated current"),
    ("APD_I_net", "Net photocurrent"),
    ("APD_M", "Multiplication gain"),
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.55,
            "savefig.dpi": 300,
        }
    )


def paired_split_differences() -> pd.DataFrame:
    metrics = pd.read_csv(RESULTS / "metrics_by_seed.csv")
    metrics = metrics[
        metrics["task"].isin([task for task, _ in TASKS])
        & metrics["model"].isin(["bkan", "spline_ridge"])
    ]
    wide = metrics.pivot(index=["task", "seed"], columns="model", values="rmse_target")
    wide["delta"] = wide["bkan"] - wide["spline_ridge"]
    return wide.reset_index()


def plot_paired_panel(ax: plt.Axes) -> None:
    split_deltas = paired_split_differences()
    stats = pd.read_csv(RESULTS / "paired_statistics.csv")
    stats = stats[
        stats["task"].isin([task for task, _ in TASKS])
        & stats["baseline"].eq("spline_ridge")
    ].set_index("task")

    for y, (task, _) in enumerate(TASKS):
        values = (
            split_deltas.loc[split_deltas["task"].eq(task)]
            .sort_values("seed")["delta"]
            .to_numpy(float)
        )
        jitter = np.linspace(-0.13, 0.13, len(values))
        ax.scatter(
            values,
            y + jitter,
            s=17,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.75,
            zorder=2,
        )

        row = stats.loc[task]
        mean = float(row["paired_delta_mean"])
        low = float(row["corrected_ci95_low"])
        high = float(row["corrected_ci95_high"])
        ax.plot([low, high], [y, y], color=REFERENCE_COLOR, lw=1.6, zorder=3)
        ax.plot([low, low], [y - 0.08, y + 0.08], color=REFERENCE_COLOR, lw=1.0)
        ax.plot([high, high], [y - 0.08, y + 0.08], color=REFERENCE_COLOR, lw=1.0)
        ax.scatter(
            [mean],
            [y],
            marker="D",
            s=31,
            color=BKAN_COLOR,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
        ax.text(
            0.013,
            y,
            f"{int(row['wins'])}/10 wins",
            va="center",
            ha="left",
            fontsize=7.0,
            color=MUTED,
        )

    ax.axvline(0, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.set_yticks(range(len(TASKS)), [label for _, label in TASKS])
    ax.invert_yaxis()
    ax.set_xlim(-0.075, 0.032)
    ax.set_xlabel("Paired RMSE difference\n" + r"BKAN-VI $-$ Spline-Ridge")
    ax.set_title("Matched-split advantage")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=BLUE,
                markeredgewidth=0.75,
                markersize=4.5,
                label="Individual split",
            ),
            Line2D(
                [0],
                [0],
                color=REFERENCE_COLOR,
                lw=1.6,
                marker="D",
                markerfacecolor=BKAN_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=4.8,
                label="Mean + corrected 95% CI",
            ),
            Line2D(
                [0],
                [0],
                color=MUTED,
                lw=1.0,
                ls="--",
                label="Zero difference",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.52),
        columnspacing=0.9,
        handlelength=2.0,
        fontsize=6.8,
    )


def read_predictions(model: str) -> pd.DataFrame:
    return pd.read_csv(
        PREDICTIONS / "net_photocurrent" / model / "test_predictions.csv"
    )


def representative_net_group(bkan: pd.DataFrame) -> tuple[str, float, float]:
    work = bkan.copy()
    work["sqerr"] = (
        work["prediction_model_space"] - work["actual_model_space"]
    ) ** 2
    errors = work.groupby("structure_group_id")["sqerr"].mean().pow(0.5)
    median_rmse = float(errors.median())
    group_id = str((errors - median_rmse).abs().idxmin())
    return group_id, float(errors.loc[group_id]), median_rmse


def plot_response_panel(ax: plt.Axes) -> dict[str, object]:
    bkan_all = read_predictions("bkan")
    spline_all = read_predictions("spline_ridge")
    group_id, group_rmse, median_rmse = representative_net_group(bkan_all)
    bkan = bkan_all[bkan_all["structure_group_id"].astype(str).eq(group_id)]
    spline = spline_all[spline_all["structure_group_id"].astype(str).eq(group_id)]

    for power in sorted(bkan["optical_power_W"].unique()):
        color = POWER_COLORS[float(power)]
        b = bkan[np.isclose(bkan["optical_power_W"], power)].sort_values("bias_v")
        s = spline[np.isclose(spline["optical_power_W"], power)].sort_values("bias_v")
        x = b["bias_v"].to_numpy(float)
        actual = b["net_photocurrent_A"].to_numpy(float)
        pred_bkan = 10 ** b["prediction_model_space"].to_numpy(float)
        pred_spline = 10 ** s["prediction_model_space"].to_numpy(float)

        ax.plot(x, pred_spline, color=color, lw=1.2, ls="--", zorder=1)
        ax.plot(x, pred_bkan, color=color, lw=1.6, zorder=2)
        ax.scatter(
            x,
            actual,
            s=12,
            facecolor="white",
            edgecolor=color,
            linewidth=0.7,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Bias (V)")
    ax.set_ylabel(r"$I_{\rm photo,net}$ (A)")
    ax.set_title("Representative complete response")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=REFERENCE_COLOR,
                markeredgewidth=0.7,
                markersize=4.5,
                label="TCAD",
            ),
            Line2D([0], [0], color=REFERENCE_COLOR, lw=1.6, label="BKAN-VI"),
            Line2D(
                [0],
                [0],
                color=REFERENCE_COLOR,
                lw=1.2,
                ls="--",
                label="Spline-Ridge",
            ),
            Line2D([0], [0], color=POWER_COLORS[1e-7], lw=2, label="100 nW"),
            Line2D([0], [0], color=POWER_COLORS[3e-7], lw=2, label="300 nW"),
        ],
        frameon=False,
        ncol=2,
        loc="lower left",
        columnspacing=0.9,
        handlelength=2.0,
    )

    return {
        "seed": 42,
        "task": "net_photocurrent",
        "structure_group_id": group_id,
        "bkan_group_rmse_model_space": group_rmse,
        "median_bkan_group_rmse_model_space": median_rmse,
    }


def main() -> None:
    style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.05, 3.05),
        gridspec_kw={"width_ratios": [1.05, 1.25]},
    )
    plot_paired_panel(axes[0])
    selection = plot_response_panel(axes[1])
    for label, ax in zip("ab", axes):
        ax.text(
            0.5,
            -0.36,
            f"({label})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=9.5,
            clip_on=False,
        )
    fig.tight_layout(pad=0.8)
    fig.subplots_adjust(bottom=0.36)

    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    original_visibility = [ax.get_visible() for ax in axes]
    for ax, stem in zip(
        axes,
        ["fig03a_apd_paired_advantage", "fig03b_apd_complete_response"],
        strict=True,
    ):
        for item in axes:
            item.set_visible(item is ax)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(
            FIGURE_OUT / f"{stem}.pdf",
            bbox_inches=bbox,
            pad_inches=0.06,
            facecolor="white",
        )
    for ax, visible in zip(axes, original_visibility, strict=True):
        ax.set_visible(visible)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_OUT / f"fig03_apd_paired_evidence.{suffix}",
            bbox_inches="tight",
        )
    plt.close(fig)
    pd.DataFrame([selection]).to_csv(SELECTION_OUT, index=False)
    print(
        f"net_photocurrent: {selection['structure_group_id']} "
        f"(RMSE={selection['bkan_group_rmse_model_space']:.6g}, "
        f"median={selection['median_bkan_group_rmse_model_space']:.6g})"
    )


if __name__ == "__main__":
    main()
