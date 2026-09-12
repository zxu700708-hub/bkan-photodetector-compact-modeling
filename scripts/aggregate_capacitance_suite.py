from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text

from manuscript_plot_palette import GRID, INK, MODEL_COLORS, MUTED


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
PAPER_ROBUSTNESS = (
    ROOT / "paper" / "paper_figure_package" / "selected_results" / "robustness"
)
PAPER_SUMMARIES = ROOT / "paper" / "paper_figure_package" / "summaries"
ARTIFACT_ROBUSTNESS = (
    ROOT / "artifacts" / "paper_figure_package" / "selected_results" / "robustness"
)
ARTIFACT_SUMMARIES = ROOT / "artifacts" / "paper_figure_package" / "summaries"
SUBMISSION_FIGURES = ROOT / "paper" / "paper_figure_package" / "figure"

MODEL_ORDER = ["mlp_l", "mlp_n", "dkan", "bkan"]
MODEL_LABEL = {
    "mlp_l": "MLP-L",
    "mlp_n": "MLP-N",
    "dkan": "DKAN",
    "bkan": "BKAN-VI",
}
COLORS = {
    "mlp_l": MODEL_COLORS["mlp_l"],
    "mlp_n": MODEL_COLORS["curve_lut"],
    "dkan": MODEL_COLORS["dkan"],
    "bkan": MODEL_COLORS["bkan"],
}

FIGURE_TEXT_SCALE = 1.18


def enlarge_figure_text(fig: plt.Figure) -> None:
    for artist in fig.findobj(match=Text):
        artist.set_fontsize(artist.get_fontsize() * FIGURE_TEXT_SCALE)


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    enlarge_figure_text(fig)
    for root in (PAPER_ROBUSTNESS, ARTIFACT_ROBUSTNESS):
        root.mkdir(parents=True, exist_ok=True)
        fig.savefig(root / f"{name}.png")
        fig.savefig(root / f"{name}.svg")
    if name == "capacitance_structured_generalization":
        SUBMISSION_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            SUBMISSION_FIGURES / "fig06_structured_extrapolation.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def save_summary(fig: plt.Figure, name: str) -> None:
    enlarge_figure_text(fig)
    for root in (PAPER_SUMMARIES, ARTIFACT_SUMMARIES):
        root.mkdir(parents=True, exist_ok=True)
        fig.savefig(root / f"{name}.png")
        fig.savefig(root / f"{name}.svg")
    plt.close(fig)


def aggregate_structured() -> pd.DataFrame:
    sources = [
        RESULTS / "structured_generalization_capacitance_nonbkan" / "metrics_by_split.csv",
        RESULTS / "structured_generalization_capacitance_bkan_a" / "metrics_by_split.csv",
        RESULTS / "structured_generalization_capacitance_bkan_b" / "metrics_by_split.csv",
    ]
    raw = pd.concat([pd.read_csv(path) for path in sources], ignore_index=True)
    out = RESULTS / "structured_generalization_capacitance"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "metrics_by_split.csv", index=False)

    key = ["task", "scenario", "holdout_column", "holdout_side", "model"]
    numeric = [
        column
        for column in raw.columns
        if column not in key + ["seed"] and pd.api.types.is_numeric_dtype(raw[column])
    ]
    rows: list[dict[str, object]] = []
    for values, frame in raw.groupby(key, dropna=False):
        row = dict(zip(key, values))
        row["n_runs"] = len(frame)
        for column in numeric:
            row[f"{column}_mean"] = frame[column].mean()
            row[f"{column}_std"] = frame[column].std(ddof=0)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    return summary


def plot_data_efficiency() -> None:
    base = pd.read_csv(RESULTS / "generalization_sweep" / "metrics_summary.csv")
    cap = pd.read_csv(
        RESULTS / "generalization_sweep_capacitance" / "metrics_summary.csv"
    )
    data = pd.concat([base, cap], ignore_index=True)
    tasks = [
        ("I_dark", r"$I_{\mathrm{dark}}$"),
        ("I_photo", r"$I_{\mathrm{photo}}$"),
        ("AC_Response", "AC response"),
        ("Capacitance", r"$C_{\mathrm{app}}$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8), sharex=True)
    for ax, (task, label) in zip(axes.ravel(), tasks):
        subset = data[data["task"].eq(task)]
        for model in MODEL_ORDER:
            frame = subset[subset["model"].eq(model)].sort_values("fraction")
            if frame.empty:
                continue
            x = frame["fraction"].to_numpy() * 100
            y = frame["r2_target_mean"].to_numpy()
            e = frame["r2_target_std"].fillna(0).to_numpy()
            ax.plot(x, y, marker="o", linewidth=1.7, color=COLORS[model], label=MODEL_LABEL[model])
            ax.fill_between(x, y - e, y + e, color=COLORS[model], alpha=0.12)
        ax.set_title(label)
        ax.set_xscale("log")
        ax.set_xlabel("training data (%)")
        ax.set_ylabel(r"test $R^2$")
        if task == "I_dark":
            ax.set_ylim(-0.05, 1.02)
            ax.text(
                1.15,
                0.08,
                r"DKAN at 1%: $R^2=-470.9$",
                fontsize=7.5,
                color=COLORS["dkan"],
            )
        else:
            lower = min(0.55, float(subset["r2_target_mean"].min()) - 0.03)
            ax.set_ylim(lower, 1.01)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Data-efficiency sweep across four device-modeling tasks", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    save(fig, "generalization_rmse_vs_fraction_four_tasks")


def plot_structured(summary: pd.DataFrame) -> None:
    scenarios = [
        "voltage_extreme_reverse",
        "frequency_high",
        "length_high",
        "temperature_high",
        "trap_high",
    ]
    labels = ["reverse bias", "high frequency", "long length", "high temperature", "high trap"]
    models = ["dkan", "bkan", "mlp_l", "spline_ridge", "poly3_ridge"]
    colors = {
        **COLORS,
        "spline_ridge": MODEL_COLORS["spline_ridge"],
        "poly3_ridge": MODEL_COLORS["poly3_ridge"],
    }
    display = {**MODEL_LABEL, "spline_ridge": "Spline-Ridge", "poly3_ridge": "Poly3-Ridge"}
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    width = 0.15
    x = np.arange(len(scenarios))
    for offset, model in enumerate(models):
        values = []
        errors = []
        for scenario in scenarios:
            row = summary[
                summary["scenario"].eq(scenario) & summary["model"].eq(model)
            ]
            values.append(float(row["r2_target_mean"].iloc[0]))
            errors.append(float(row["r2_target_std"].iloc[0]))
        ax.bar(
            x + (offset - 2) * width,
            values,
            width,
            yerr=errors,
            capsize=2,
            color=colors[model],
            label=display[model],
        )
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"structured-holdout $R^2$")
    ax.set_ylim(-1.0, 1.08)
    ax.set_title("Apparent-capacitance structured extrapolation")
    ax.legend(ncol=3, frameon=False, loc="lower right")
    ax.text(
        0.02,
        0.03,
        r"Poly3-Ridge reverse-bias result ($R^2=-13.55$) is clipped.",
        transform=ax.transAxes,
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout()
    save(fig, "capacitance_structured_generalization")


def write_summary(structured: pd.DataFrame) -> None:
    point = pd.read_csv(
        RESULTS / "traditional_vs_kan_capacitance" / "metrics_summary.csv"
    )
    grouped = pd.read_csv(
        RESULTS / "traditional_vs_kan_grouped_capacitance" / "metrics_summary.csv"
    )
    uq = pd.read_csv(
        RESULTS / "uq_repeated_grouped_capacitance" / "uq_metrics_summary.csv"
    )
    symbolic = pd.read_csv(
        RESULTS / "device_modeling_capacitance_bayesian_symbolic" / "summary.csv"
    )
    dropout = pd.read_csv(
        RESULTS / "device_modeling_capacitance_dropout" / "summary.csv"
    )
    hmc = pd.read_csv(
        RESULTS / "device_modeling_capacitance_hmc_multichain" / "summary.csv"
    )
    lines = [
        "# Apparent-capacitance experiment coverage",
        "",
        "- Point-wise benchmark: 8 model families, 3 seeds.",
        "- Grouped sample-ID benchmark: 7 model families, 3 seeds, zero sample-ID overlap.",
        "- Repeated grouped VI/conformal UQ: 10 seeds.",
        "- Data-efficiency sweep: 1--100%, 4 neural models, 3 sampling seeds.",
        "- Structured extrapolation: 5 held-out boundaries.",
        "- MC dropout: completed on the maintained grouped split.",
        "- Restricted multi-chain HMC: completed; convergence diagnostics failed, so posterior comparison is excluded.",
        "- Bayesian symbolic-gated distillation: completed.",
        "",
        "## Key machine-readable sources",
        "",
        f"- Point-wise rows: {len(point)}",
        f"- Grouped rows: {len(grouped)}",
        f"- UQ rows: {len(uq)}",
        f"- Structured rows: {len(structured)}",
        f"- Symbolic rows: {len(symbolic)}",
        f"- MC-dropout rows: {len(dropout)}",
        f"- HMC rows: {len(hmc)}",
    ]
    out = RESULTS / "capacitance_full_suite"
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_value_map() -> None:
    rows = [
        ("BKAN-VI", 0.9933, 48544, 1872, True, "#009E73"),
        ("DKAN", 0.9924, 628, 1088, False, "#0072B2"),
        ("Spline-Ridge", 0.9905, 188, 50, False, "#56B4E9"),
        ("Poly3-Ridge", 0.9850, 101, 120, False, "#E69F00"),
        ("MLP-L", 0.9827, 75, 2625, False, "#CC79A7"),
        ("RBF-Nystroem", 0.9807, 258, 2433, False, "#7F7F7F"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for name, score, latency, params, uq, color in rows:
        marker = "*" if uq else "o"
        size = 90 if uq else 45 + 20 * np.log10(max(params, 10))
        ax.scatter(latency, score, s=size, marker=marker, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.annotate(name, (latency, score), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"batch-size-1 CPU latency ($\mu$s, log scale)")
    ax.set_ylabel(r"four-task mean point-wise $R^2$")
    ax.set_ylim(0.978, 0.995)
    ax.set_title("Four-task compact-model value map")
    ax.text(
        0.02,
        0.04,
        "Star: calibrated UQ implemented; marker area scales approximately with parameter count.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout()
    save_summary(fig, "fig11_compact_model_value_map")


def plot_symbolic_quality() -> None:
    tasks = [r"$I_{\mathrm{dark}}$", r"$I_{\mathrm{photo}}$", "AC response"]
    rmse = [0.02098, 0.00267, 0.09822]
    lengths = [4204, 4909, 1611]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    axes[0].bar(tasks, rmse, color=["#009E73", "#14833B", "#0072B2"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("formula-to-network RMSE")
    axes[0].set_title("Edge-distilled formulas")
    for index, value in enumerate(rmse):
        axes[0].text(index, value * 1.08, f"{value:.4g}", ha="center", fontsize=8)

    axes[1].bar(
        ["posterior-mean\nBKAN", "12-term symbolic"],
        [0.999714, 0.994868],
        color=["#009E73", "#E69F00"],
    )
    axes[1].set_ylim(0.99, 1.0)
    axes[1].set_ylabel(r"capacitance test $R^2$")
    axes[1].set_title("Apparent-capacitance symbolic fidelity")
    for index, value in enumerate([0.999714, 0.994868]):
        axes[1].text(index, value + 0.0002, f"{value:.6f}", ha="center", fontsize=8)

    axes[2].bar(tasks, lengths, color=["#0072B2", "#009E73", "#14833B"])
    axes[2].axhline(12, color="#E69F00", linewidth=2, label="capacitance: 12 active terms")
    axes[2].set_ylabel("exported characters")
    axes[2].set_title("Current/RF formula complexity")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Symbolic distillation quality across four tasks", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_summary(fig, "fig08_symbolic_formula_quality")


def main() -> None:
    configure()
    structured = aggregate_structured()
    plot_data_efficiency()
    plot_structured(structured)
    plot_value_map()
    plot_symbolic_quality()
    write_summary(structured)


if __name__ == "__main__":
    main()
