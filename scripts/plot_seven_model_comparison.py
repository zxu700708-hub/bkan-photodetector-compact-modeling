"""Create a cross-device relative-RMSE heatmap for the seven-model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "bkan",
    "dkan",
    "mlp_l",
    "spline_ridge",
    "gmls",
    "autopinn",
    "curve_lut",
)
MODEL_LABELS = (
    "BKAN-VI",
    "DKAN",
    "MLP-L",
    "Spline-Ridge",
    "GMLS-adapted",
    "AutoPINN-adapted",
    "Curve-LUT-PCHIP",
)
PD_TASK_LABELS = {
    "I_dark": "PD dark current",
    "I_photo": "PD net photocurrent",
    "AC_Response": "PD AC response",
    "Capacitance": "PD capacitance",
}
APD_TASK_LABELS = {
    "APD_I_dark": "APD dark current",
    "APD_I_photo": "APD photocurrent",
    "APD_I_net": "APD net photocurrent",
    "APD_M": "APD multiplication gain",
    "APD_V_M10": "APD gain-threshold voltage",
    "APD_V_br": "APD breakdown voltage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pd-root",
        type=Path,
        default=ROOT / "artifacts/results/compact_framework_seven_models_pd",
    )
    parser.add_argument(
        "--apd-root",
        type=Path,
        default=ROOT / "artifacts/results/compact_framework_seven_models_apd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/results/compact_framework_seven_models_comparison",
    )
    return parser.parse_args()


def load_matrix(pd_root: Path, apd_root: Path) -> tuple[np.ndarray, list[str]]:
    pd_summary = pd.read_csv(pd_root / "metrics_summary.csv").rename(
        columns={"rmse_mean": "rmse"}
    )
    apd_summary = pd.read_csv(apd_root / "metrics_summary.csv").rename(
        columns={"rmse_target_mean": "rmse"}
    )
    tasks = list(PD_TASK_LABELS) + list(APD_TASK_LABELS)
    labels = list(PD_TASK_LABELS.values()) + list(APD_TASK_LABELS.values())
    combined = pd.concat(
        [
            pd_summary[["task", "model", "rmse"]],
            apd_summary[["task", "model", "rmse"]],
        ],
        ignore_index=True,
    )
    pivot = combined.pivot(index="model", columns="task", values="rmse").reindex(
        index=MODELS, columns=tasks
    )
    if pivot.isna().any().any():
        raise ValueError("Seven-model comparison matrix is incomplete")
    values = pivot.to_numpy(dtype=np.float64)
    return values / values.min(axis=0, keepdims=True), labels


def plot(relative: np.ndarray, task_labels: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.2, 6.4))
    upper = max(float(np.max(relative)), 1.01)
    image = ax.imshow(
        relative,
        aspect="auto",
        cmap="YlGnBu",
        norm=LogNorm(vmin=1.0, vmax=upper),
    )
    ax.set_xticks(np.arange(len(task_labels)), task_labels, rotation=36, ha="right")
    ax.set_yticks(np.arange(len(MODEL_LABELS)), MODEL_LABELS)
    ax.set_xlabel("Device task")
    ax.set_ylabel("Model")
    ax.set_title("Photodetector construction-method comparison")
    for row in range(relative.shape[0]):
        for column in range(relative.shape[1]):
            value = relative[row, column]
            text_color = "white" if value > np.sqrt(upper) else "black"
            weight = "bold" if np.isclose(value, 1.0, rtol=1e-12) else "normal"
            ax.text(
                column,
                row,
                f"{value:.2f}x",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight=weight,
            )
    ax.axvline(len(PD_TASK_LABELS) - 0.5, color="white", linewidth=2.0)
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("RMSE / best RMSE within task (lower is better)")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    relative, labels = load_matrix(args.pd_root, args.apd_root)
    plot(relative, labels, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
