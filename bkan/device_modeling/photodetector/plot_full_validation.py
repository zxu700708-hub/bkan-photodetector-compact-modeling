"""Plot the full algorithm validation saved under main/final_algorithm_validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_LABELS = {
    "I_dark": "Dark current",
    "I_photo": "Net photocurrent",
    "AC_Response": "Optical SSAC response",
}

TASK_DIRS = {
    "I_dark": "dark_current",
    "I_photo": "photo_current",
    "AC_Response": "ac_response",
}

AXIS_COLUMNS = {
    "I_dark": "dark_voltage",
    "I_photo": "light_voltage",
    "AC_Response": "frequency_ghz",
}

TARGET_COLUMNS = {
    "I_dark": "dark_current",
    "I_photo": "net_photocurrent",
    "AC_Response": "ac_response_db",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "ac_response_db" not in frame and "bandwidth" in frame:
        frame = frame.rename(columns={"bandwidth": "ac_response_db"})
    return frame


def savefig(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / name, bbox_inches="tight")
    plt.close()


def full_metrics(root: Path) -> pd.DataFrame:
    rows = []
    det = read_csv(root / "deterministic" / "summary.csv")
    for _, row in det.iterrows():
        rows.append(
            {
                "task": row["task"],
                "method": "Deterministic KAN",
                "rmse": row["test_rmse_model_space"],
                "mae": row["test_mae_model_space"],
                "r2": row["test_r2_model_space"],
            }
        )

    bayes = read_csv(root / "bayesian_compare" / "summary.csv")
    for _, row in bayes[bayes["inference_method"].eq("hmc")].iterrows():
        rows.append(
            {
                "task": row["task"],
                "method": "Bayesian KAN HMC",
                "rmse": row["rmse_model_space"],
                "mae": row["mae_model_space"],
                "r2": row["r2_model_space"],
            }
        )

    sym = read_csv(root / "symbolic_gated" / "summary.csv")
    for _, row in sym.iterrows():
        rows.append(
            {
                "task": row["task"],
                "method": "Symbolic gated KAN",
                "rmse": row["test_rmse"],
                "mae": row["test_mae"],
                "r2": row["test_r2"],
            }
        )

    bs = read_csv(root / "bayesian_symbolic_hmc" / "summary.csv")
    for _, row in bs.iterrows():
        rows.append(
            {
                "task": row["task"],
                "method": "Bayesian symbolic HMC",
                "rmse": row["test_rmse"],
                "mae": row["test_mae"],
                "r2": row["test_r2"],
            }
        )
    return pd.DataFrame(rows)


def plot_overview(metrics: pd.DataFrame, out_dir: Path) -> None:
    tasks = list(TASK_LABELS)
    methods = [
        "Deterministic KAN",
        "Bayesian KAN HMC",
        "Symbolic gated KAN",
        "Bayesian symbolic HMC",
    ]
    colors = ["#2f6fdd", "#2ca02c", "#d95f02", "#9467bd"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    specs = [("r2", "Test R2", False), ("rmse", "Test RMSE", True), ("mae", "Test MAE", True)]
    x = np.arange(len(tasks))
    width = 0.18
    for ax, (metric, title, logy) in zip(axes, specs):
        for i, (method, color) in enumerate(zip(methods, colors)):
            vals = []
            for task in tasks:
                row = metrics[metrics["task"].eq(task) & metrics["method"].eq(method)]
                vals.append(row[metric].iloc[0] if not row.empty else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, color=color, alpha=0.88, label=method)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks], rotation=18, ha="right")
        if logy:
            ax.set_yscale("log")
        if metric == "r2":
            ax.set_ylim(max(0.85, metrics["r2"].min() - 0.04), 1.01)
        ax.legend(loc="best", fontsize=8)
    savefig(out_dir, "01_overall_algorithm_metrics.png")


def plot_bayesian_methods(root: Path, out_dir: Path) -> None:
    df = read_csv(root / "bayesian_compare" / "summary.csv")
    tasks = list(TASK_LABELS)
    methods = ["dropout", "hmc", "vi"]
    labels = {"dropout": "Dropout", "hmc": "HMC", "vi": "VI"}
    colors = {"dropout": "#2f6fdd", "hmc": "#2ca02c", "vi": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5))
    specs = [
        ("r2_model_space", "R2", False),
        ("rmse_model_space", "RMSE", True),
        ("picp_95", "95% PICP", False),
        ("mean_std_model_space", "Mean predictive std", True),
    ]
    x = np.arange(len(tasks))
    width = 0.24
    for ax, (metric, title, logy) in zip(axes.ravel(), specs):
        for i, method in enumerate(methods):
            vals = []
            for task in tasks:
                row = df[df["task"].eq(task) & df["inference_method"].eq(method)]
                vals.append(row[metric].iloc[0] if not row.empty else np.nan)
            ax.bar(x + (i - 1) * width, vals, width=width, label=labels[method], color=colors[method], alpha=0.88)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks], rotation=18, ha="right")
        if logy:
            ax.set_yscale("log")
        if metric == "picp_95":
            ax.axhline(95.0, color="#555555", linestyle="--", linewidth=1.0, label="Target 95%")
        ax.legend(loc="best")
    savefig(out_dir, "02_bayesian_method_comparison.png")


def select_curve(df: pd.DataFrame, task: str) -> pd.DataFrame:
    axis = AXIS_COLUMNS[task]
    if "_curve_id" in df.columns:
        curve_id = df["_curve_id"].value_counts().index[0]
        return df[df["_curve_id"].eq(curve_id)].sort_values(axis)
    excluded = {
        axis,
        TARGET_COLUMNS[task],
        "split",
        "actual",
        "prediction",
        "abs_error",
        "bayesian_precision_weight",
        "bayesian_model_std",
        "bayesian_teacher_mean",
        "actual_model_space",
        "prediction_model_space",
        "prediction_physical",
        "abs_error_model_space",
    }
    cond = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not cond:
        return df.sort_values(axis)
    key = df.groupby(cond, sort=False).size().idxmax()
    if not isinstance(key, tuple):
        key = (key,)
    mask = np.ones(len(df), dtype=bool)
    for col, value in zip(cond, key):
        mask &= np.isclose(df[col].to_numpy(dtype=float), float(value), rtol=1e-10, atol=1e-12)
    return df.loc[mask].sort_values(axis)


def plot_prediction_scatter(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 13.0))
    methods = [
        ("deterministic", "Deterministic KAN"),
        ("symbolic_gated", "Symbolic gated"),
        ("bayesian_symbolic_hmc", "Bayesian symbolic HMC"),
    ]
    for row, task in enumerate(TASK_LABELS):
        for col, (method_dir, label) in enumerate(methods):
            ax = axes[row, col]
            if method_dir == "deterministic":
                path = root / method_dir / f"{TASK_DIRS[task]}_deterministic_kan" / "predictions.csv"
                df = read_csv(path)
                test = df[df["split"].eq("test")]
                actual = test["actual_model_space"]
                pred = test["prediction_model_space"]
            else:
                path = root / method_dir / f"{TASK_DIRS[task]}_symbolic_gated_kan" / "predictions.csv"
                df = read_csv(path)
                test = df[df["split"].eq("test")]
                actual = test["actual"]
                pred = test["prediction"]
            ax.scatter(actual, pred, s=14, alpha=0.7, color="#2f6fdd", edgecolors="none")
            lo = min(float(actual.min()), float(pred.min()))
            hi = max(float(actual.max()), float(pred.max()))
            pad = 0.04 * (hi - lo + 1e-12)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#222222", linestyle="--", linewidth=1.0)
            ax.set_title(f"{TASK_LABELS[task]}: {label}")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Prediction")
    savefig(out_dir, "03_prediction_scatter.png")


def plot_axis_curves(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 13.0))
    colors = {
        "actual": "#111111",
        "det": "#2f6fdd",
        "sym": "#d95f02",
        "bs": "#9467bd",
    }
    for ax, task in zip(axes, TASK_LABELS):
        axis = AXIS_COLUMNS[task]
        det = read_csv(root / "deterministic" / f"{TASK_DIRS[task]}_deterministic_kan" / "predictions.csv")
        det_curve = select_curve(det[det["split"].eq("test")], task)
        sym = read_csv(root / "symbolic_gated" / f"{TASK_DIRS[task]}_symbolic_gated_kan" / "predictions.csv")
        sym_curve = select_curve(sym[sym["split"].eq("test")], task)
        bs = read_csv(root / "bayesian_symbolic_hmc" / f"{TASK_DIRS[task]}_symbolic_gated_kan" / "predictions.csv")
        bs_curve = select_curve(bs[bs["split"].eq("test")], task)

        ax.plot(det_curve[axis], det_curve[TARGET_COLUMNS[task]], color=colors["actual"], marker="o", markersize=3, label="Actual")
        ax.plot(det_curve[axis], det_curve["prediction_physical"], color=colors["det"], marker="s", markersize=3, label="Deterministic KAN")
        ax.plot(sym_curve[axis], sym_curve["prediction"], color=colors["sym"], marker="^", markersize=3, label="Symbolic gated")
        ax.plot(bs_curve[axis], bs_curve["prediction"], color=colors["bs"], marker="d", markersize=3, label="Bayesian symbolic HMC")
        if "bayesian_model_std" in bs_curve.columns:
            x = bs_curve[axis].to_numpy(dtype=float)
            y = bs_curve["prediction"].to_numpy(dtype=float)
            s = bs_curve["bayesian_model_std"].to_numpy(dtype=float)
            ax.fill_between(x, y - 2 * s, y + 2 * s, color=colors["bs"], alpha=0.13, label="Bayesian symbolic ±2 std")
        ax.set_title(f"{TASK_LABELS[task]} representative curve")
        ax.set_xlabel(axis)
        ax.set_ylabel("Physical value")
        ax.legend(loc="best", ncol=2)
    savefig(out_dir, "04_axis_curve_comparison.png")


def plot_training_curves(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 10.5))
    for row, task in enumerate(TASK_LABELS):
        for col, (method_dir, title) in enumerate(
            [
                ("symbolic_gated", "Symbolic gated"),
                ("bayesian_symbolic_hmc", "Bayesian symbolic HMC"),
            ]
        ):
            ax = axes[row, col]
            hist = read_csv(root / method_dir / f"{TASK_DIRS[task]}_symbolic_gated_kan" / "training_history.csv")
            ax.plot(hist["step"], hist["train_total"], label="train total", color="#2f6fdd")
            ax.plot(hist["step"], hist["validation_total"], label="validation total", color="#d95f02")
            ax.set_yscale("log")
            ax.set_title(f"{TASK_LABELS[task]}: {title}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend(loc="best")
    savefig(out_dir, "05_symbolic_training_curves.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("main/final_algorithm_validation"))
    args = parser.parse_args()

    setup_style()
    root = args.root
    out_dir = root / "visualizations"
    metrics = full_metrics(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "combined_metrics.csv", index=False)

    plot_overview(metrics, out_dir)
    plot_bayesian_methods(root, out_dir)
    plot_prediction_scatter(root, out_dir)
    plot_axis_curves(root, out_dir)
    plot_training_curves(root, out_dir)
    print(f"Saved validation visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
