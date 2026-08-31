from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
RESULTS = REPO_ROOT / "artifacts" / "results"
OUT = RESULTS / "current_visualization_report"

TASK_LABELS = {
    "I_dark": "Dark current",
    "I_photo": "Net photocurrent",
    "AC_Response": "Optical SSAC response",
}

SYMBOLIC_DIRS = {
    "I_dark": RESULTS / "final_symbolic_gated_validation" / "dark_current_symbolic_gated_kan",
    "I_photo": RESULTS / "final_symbolic_gated_validation" / "photo_current_symbolic_gated_kan",
    "AC_Response": RESULTS / "final_symbolic_gated_validation" / "ac_response_symbolic_gated_kan",
}

BAYES_SYMBOLIC_DIRS = {
    "I_dark": RESULTS / "validation_bayesian_symbolic_dark_ac_full" / "dark_current_symbolic_gated_kan",
    "I_photo": RESULTS / "validation_bayesian_symbolic_photo_full" / "photo_current_symbolic_gated_kan",
    "AC_Response": RESULTS / "validation_bayesian_symbolic_dark_ac_full" / "ac_response_symbolic_gated_kan",
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

PREDICTION_VALUE_COLUMNS = {
    "actual",
    "prediction",
    "abs_error",
    "bayesian_precision_weight",
    "bayesian_model_std",
    "bayesian_teacher_mean",
    "split",
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
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "ac_response_db" not in frame and "bandwidth" in frame:
        frame = frame.rename(columns={"bandwidth": "ac_response_db"})
    return frame


def savefig(name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def load_symbolic_summary() -> pd.DataFrame:
    df = read_csv(RESULTS / "final_symbolic_gated_validation" / "summary.csv")
    return pd.DataFrame(
        {
            "task": df["task"],
            "method": "Symbolic gated KAN",
            "rmse": df["test_rmse"],
            "mae": df["test_mae"],
            "r2": df["test_r2"],
            "derivative_rmse": np.nan,
            "active_terms": df["active_symbolic_terms"],
        }
    )


def load_bayes_symbolic_summary() -> pd.DataFrame:
    parts = [
        read_csv(RESULTS / "validation_bayesian_symbolic_photo_full" / "summary.csv"),
        read_csv(RESULTS / "validation_bayesian_symbolic_dark_ac_full" / "summary.csv"),
    ]
    df = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(
        {
            "task": df["task"],
            "method": "Bayesian symbolic KAN",
            "rmse": df["test_rmse"],
            "mae": df["test_mae"],
            "r2": df["test_r2"],
            "bayesian_rmse": df["bayesian_rmse_model_space"],
            "picp": df["bayesian_picp_95"],
            "mean_std": df["bayesian_mean_std_model_space"],
            "derivative_rmse": df.get("bayesian_derivative_rmse_model_space", np.nan),
            "active_terms": df["active_symbolic_terms"],
        }
    )


def load_bayes_methods_summary() -> pd.DataFrame:
    parts = [
        read_csv(RESULTS / "validation_bayes_methods_photo" / "summary.csv"),
        read_csv(RESULTS / "validation_bayes_methods_dark_ac" / "summary.csv"),
    ]
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["task", "inference_method"]).reset_index(drop=True)


def plot_metric_overview(symbolic: pd.DataFrame, bayes_symbolic: pd.DataFrame) -> None:
    combined = pd.concat([symbolic, bayes_symbolic], ignore_index=True)
    order = ["I_dark", "I_photo", "AC_Response"]
    methods = ["Symbolic gated KAN", "Bayesian symbolic KAN"]
    colors = {"Symbolic gated KAN": "#3b7ddd", "Bayesian symbolic KAN": "#d95f02"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = [("r2", "Test R2", False), ("rmse", "Test RMSE", True), ("active_terms", "Active symbolic terms", False)]
    x = np.arange(len(order))
    width = 0.36

    for ax, (metric, title, logy) in zip(axes, metrics):
        for offset, method in zip([-width / 2, width / 2], methods):
            values = [
                combined.loc[(combined["task"] == task) & (combined["method"] == method), metric].iloc[0]
                for task in order
            ]
            ax.bar(x + offset, values, width=width, label=method, color=colors[method], alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in order], rotation=18, ha="right")
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        if metric == "r2":
            ax.set_ylim(min(0.75, combined["r2"].min() - 0.03), 1.01)
        ax.legend(loc="best")
    savefig("01_symbolic_metric_overview.png")


def plot_bayes_method_comparison(bayes_methods: pd.DataFrame) -> None:
    tasks = ["I_dark", "I_photo", "AC_Response"]
    methods = ["dropout", "hmc", "vi"]
    colors = {"dropout": "#3b7ddd", "hmc": "#2ca02c", "vi": "#d62728"}
    labels = {"dropout": "Dropout", "hmc": "HMC", "vi": "VI"}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
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
            values = []
            for task in tasks:
                row = bayes_methods[
                    (bayes_methods["task"] == task)
                    & (bayes_methods["inference_method"] == method)
                ]
                values.append(row[metric].iloc[0] if not row.empty else np.nan)
            ax.bar(x + (i - 1) * width, values, width=width, label=labels[method], color=colors[method], alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks], rotation=18, ha="right")
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        if metric == "picp_95":
            ax.axhline(95, color="#555555", linewidth=1.0, linestyle="--", label="Target 95%")
        ax.legend(loc="best")
    savefig("02_bayesian_inference_method_comparison.png")


def plot_training_curves() -> None:
    rows = list(TASK_LABELS)
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.5), sharex=False)
    for row, task in enumerate(rows):
        op_hist = read_csv(SYMBOLIC_DIRS[task] / "training_history.csv")
        bs_hist = read_csv(BAYES_SYMBOLIC_DIRS[task] / "training_history.csv")

        ax = axes[row, 0]
        ax.plot(op_hist["step"], op_hist["train_total"], label="train total", color="#3b7ddd")
        ax.plot(op_hist["step"], op_hist["validation_total"], label="validation total", color="#d95f02")
        ax.set_yscale("log")
        ax.set_title(f"{TASK_LABELS[task]}: symbolic gated training")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")

        ax = axes[row, 1]
        ax.plot(bs_hist["step"], bs_hist["train_total"], label="train total", color="#3b7ddd")
        ax.plot(bs_hist["step"], bs_hist["validation_total"], label="validation total", color="#d95f02")
        if "active_terms" in bs_hist:
            ax2 = ax.twinx()
            ax2.plot(bs_hist["step"], bs_hist["active_terms"], label="active terms", color="#2ca02c", alpha=0.65)
            ax2.set_ylabel("Active terms")
            ax2.grid(False)
        ax.set_yscale("log")
        ax.set_title(f"{TASK_LABELS[task]}: Bayesian symbolic training")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
    savefig("03_training_curves.png")


def scatter_panel(ax, actual: np.ndarray, prediction: np.ndarray, title: str) -> None:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    ax.scatter(actual, prediction, s=14, alpha=0.7, color="#3b7ddd", edgecolors="none")
    lo = np.nanmin([actual.min(), prediction.min()])
    hi = np.nanmax([actual.max(), prediction.max()])
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#222222", linewidth=1.0, linestyle="--")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Prediction")


def condition_columns(frame: pd.DataFrame, task: str) -> list[str]:
    axis_col = AXIS_COLUMNS[task]
    excluded = PREDICTION_VALUE_COLUMNS | {axis_col, TARGET_COLUMNS[task]}
    return [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def select_representative_curve(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    axis_col = AXIS_COLUMNS[task]
    if "_curve_id" in frame.columns:
        curve_id = frame["_curve_id"].value_counts(sort=True).index[0]
        return frame[frame["_curve_id"].eq(curve_id)].sort_values(axis_col).copy()

    columns = condition_columns(frame, task)
    if not columns:
        return frame.sort_values(axis_col).copy()

    first_key = frame.groupby(columns, sort=False).size().idxmax()
    if not isinstance(first_key, tuple):
        first_key = (first_key,)
    mask = np.ones(len(frame), dtype=bool)
    for column, value in zip(columns, first_key):
        mask &= np.isclose(frame[column].to_numpy(dtype=float), float(value), rtol=1e-10, atol=1e-12)
    return frame.loc[mask].sort_values(axis_col).copy()


def matching_curve(frame: pd.DataFrame, reference: pd.DataFrame, task: str) -> pd.DataFrame:
    axis_col = AXIS_COLUMNS[task]
    columns = [
        column for column in condition_columns(reference, task)
        if column in frame.columns
    ]
    if not columns:
        return select_representative_curve(frame, task)
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        value = float(reference[column].iloc[0])
        mask &= np.isclose(frame[column].to_numpy(dtype=float), value, rtol=1e-10, atol=1e-12)
    matched = frame.loc[mask].copy()
    if matched.empty:
        return select_representative_curve(frame, task)
    return matched.sort_values(axis_col)


def exact_matching_curve(frame: pd.DataFrame, reference: pd.DataFrame, task: str) -> pd.DataFrame | None:
    axis_col = AXIS_COLUMNS[task]
    columns = [
        column for column in condition_columns(reference, task)
        if column in frame.columns
    ]
    if not columns:
        return None
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        value = float(reference[column].iloc[0])
        mask &= np.isclose(frame[column].to_numpy(dtype=float), value, rtol=1e-10, atol=1e-12)
    matched = frame.loc[mask].copy()
    if matched.empty:
        return None
    return matched.sort_values(axis_col)



def plot_prediction_scatter() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 13.0))
    for row, task in enumerate(TASK_LABELS):
        op = read_csv(SYMBOLIC_DIRS[task] / "predictions.csv")
        op = op[op["split"].eq("test")] if "split" in op else op
        scatter_panel(
            axes[row, 0],
            op["actual"],
            op["prediction"],
            f"{TASK_LABELS[task]}: symbolic gated",
        )
        bs = read_csv(BAYES_SYMBOLIC_DIRS[task] / "predictions.csv")
        test = bs[bs["split"].eq("test")] if "split" in bs else bs
        scatter_panel(
            axes[row, 1],
            test["actual"],
            test["prediction"],
            f"{TASK_LABELS[task]}: Bayesian symbolic",
        )
    savefig("04_prediction_scatter.png")


def plot_residual_histograms() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 10.5))
    for row, task in enumerate(TASK_LABELS):
        op = read_csv(SYMBOLIC_DIRS[task] / "predictions.csv")
        op = op[op["split"].eq("test")] if "split" in op else op
        op_resid = op["prediction"] - op["actual"]
        axes[row, 0].hist(op_resid, bins=32, color="#3b7ddd", alpha=0.82)
        axes[row, 0].set_title(f"{TASK_LABELS[task]}: symbolic gated residual")
        axes[row, 0].set_xlabel("Prediction - actual")

        bs = read_csv(BAYES_SYMBOLIC_DIRS[task] / "predictions.csv")
        test = bs[bs["split"].eq("test")] if "split" in bs else bs
        bs_resid = test["prediction"] - test["actual"]
        axes[row, 1].hist(bs_resid, bins=32, color="#d95f02", alpha=0.82)
        axes[row, 1].set_title(f"{TASK_LABELS[task]}: Bayesian symbolic residual")
        axes[row, 1].set_xlabel("Prediction - actual")
    savefig("05_residual_distributions.png")


def plot_axis_curves() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 11.0))
    for row, task in enumerate(TASK_LABELS):
        axis_col = AXIS_COLUMNS[task]

        op = read_csv(SYMBOLIC_DIRS[task] / "predictions.csv")
        op = op[op["split"].eq("test")] if "split" in op else op
        op_curve = select_representative_curve(op, task)
        axes[row, 0].plot(op_curve[axis_col], op_curve["actual"], marker="o", markersize=3, label="actual", color="#222222")
        axes[row, 0].plot(op_curve[axis_col], op_curve["prediction"], marker="s", markersize=3, label="prediction", color="#3b7ddd")
        axes[row, 0].set_title(f"{TASK_LABELS[task]}: symbolic gated curve")
        axes[row, 0].set_xlabel(axis_col)
        axes[row, 0].set_ylabel("Physical value")
        axes[row, 0].legend(loc="best")

        bs = read_csv(BAYES_SYMBOLIC_DIRS[task] / "predictions.csv")
        test = bs[bs["split"].eq("test")].copy() if "split" in bs else bs.copy()
        bs_curve = matching_curve(test, op_curve, task)
        axes[row, 1].plot(bs_curve[axis_col], bs_curve["actual"], marker="o", markersize=3, label="actual", color="#222222")
        axes[row, 1].plot(bs_curve[axis_col], bs_curve["prediction"], marker="s", markersize=3, label="prediction", color="#d95f02")
        if "bayesian_model_std" in bs_curve:
            std = bs_curve["bayesian_model_std"].to_numpy(dtype=float)
            pred = bs_curve["prediction"].to_numpy(dtype=float)
            axes[row, 1].fill_between(
                bs_curve[axis_col].to_numpy(dtype=float),
                pred - 2 * std,
                pred + 2 * std,
                color="#d95f02",
                alpha=0.16,
                label="±2 std",
            )
        axes[row, 1].set_title(f"{TASK_LABELS[task]}: Bayesian symbolic curve")
        axes[row, 1].set_xlabel(axis_col)
        axes[row, 1].set_ylabel("Physical value")
        axes[row, 1].legend(loc="best")
    savefig("06_axis_curve_examples.png")


def plot_uncertainty_quality() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for ax, task in zip(axes, TASK_LABELS):
        df = read_csv(BAYES_SYMBOLIC_DIRS[task] / "predictions.csv")
        test = df[df["split"].eq("test")] if "split" in df else df
        if "bayesian_model_std" not in test:
            ax.text(0.5, 0.5, "No uncertainty column", ha="center", va="center")
            continue
        err = np.abs(test["prediction"].to_numpy(dtype=float) - test["actual"].to_numpy(dtype=float))
        std = test["bayesian_model_std"].to_numpy(dtype=float)
        ax.scatter(std, err, s=16, alpha=0.72, color="#9467bd", edgecolors="none")
        ax.set_xscale("log" if np.nanmax(std) / max(np.nanmin(std[std > 0]), 1e-30) > 100 else "linear")
        ax.set_yscale("log" if np.nanmax(err) / max(np.nanmin(err[err > 0]), 1e-30) > 100 else "linear")
        corr = np.corrcoef(std, err)[0, 1] if len(std) > 2 and np.std(std) > 0 and np.std(err) > 0 else np.nan
        ax.set_title(f"{TASK_LABELS[task]}: std vs abs error\ncorr={corr:.3f}")
        ax.set_xlabel("Predictive std")
        ax.set_ylabel("Absolute error")
    savefig("07_bayesian_symbolic_uncertainty_quality.png")


def plot_curve_method_comparison() -> None:
    colors = {
        "actual": "#111111",
        "symbolic": "#3b7ddd",
        "bayes_symbolic": "#d95f02",
        "dropout": "#9467bd",
        "hmc": "#2ca02c",
        "vi": "#d62728",
    }
    bayes_curves = read_csv(OUT / "bayesian_method_curve_predictions.csv")
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 13.2))
    for ax, task in zip(axes, TASK_LABELS):
        axis_col = AXIS_COLUMNS[task]
        task_bayes_curves = bayes_curves[bayes_curves["task"].eq(task)].copy()
        reference_curve = task_bayes_curves[task_bayes_curves["method"].eq("hmc")].sort_values("axis_value")
        reference_for_matching = reference_curve.rename(
            columns={"axis_value": axis_col, "actual": TARGET_COLUMNS[task]}
        )

        ax.plot(
            reference_curve["axis_value"],
            reference_curve["actual"],
            color=colors["actual"],
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            label="Actual",
        )

        bs = read_csv(BAYES_SYMBOLIC_DIRS[task] / "predictions.csv")
        bs = bs[bs["split"].eq("test")].copy() if "split" in bs else bs
        bs_curve = exact_matching_curve(bs, reference_for_matching, task)
        if bs_curve is not None:
            ax.plot(
                bs_curve[axis_col],
                bs_curve["prediction"],
                color=colors["bayes_symbolic"],
                marker="^",
                markersize=3,
                linewidth=1.7,
                label="Bayesian symbolic",
            )

        for method, label in (("dropout", "Dropout KAN"), ("hmc", "HMC KAN"), ("vi", "VI KAN")):
            result = task_bayes_curves[task_bayes_curves["method"].eq(method)].sort_values("axis_value")
            ax.plot(
                result["axis_value"],
                result["mean"],
                color=colors[method],
                linewidth=1.55,
                alpha=0.95,
                label=label,
            )

        ax.set_title(f"{TASK_LABELS[task]}: same-curve method comparison")
        ax.set_xlabel(axis_col)
        ax.set_ylabel("Physical value")
        ax.legend(loc="best", ncol=2)
    savefig("08_same_curve_all_methods.png")


def plot_bayesian_method_uncertainty_curves() -> None:
    colors = {"dropout": "#9467bd", "hmc": "#2ca02c", "vi": "#d62728"}
    labels = {"dropout": "Dropout", "hmc": "HMC", "vi": "VI"}
    bayes_curves = read_csv(OUT / "bayesian_method_curve_predictions.csv")
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 12.0), sharex=False)
    for row, task in enumerate(TASK_LABELS):
        axis_col = AXIS_COLUMNS[task]
        for col, method in enumerate(("dropout", "hmc", "vi")):
            ax = axes[row, col]
            result = bayes_curves[
                bayes_curves["task"].eq(task) & bayes_curves["method"].eq(method)
            ].sort_values("axis_value")
            x = result["axis_value"].to_numpy(dtype=float)
            y = result["actual"].to_numpy(dtype=float)
            mean = result["mean"].to_numpy(dtype=float)
            std = result["std"].to_numpy(dtype=float)
            ax.plot(x, y, color="#111111", marker="o", markersize=3, linewidth=1.6, label="Actual")
            ax.plot(x, mean, color=colors[method], marker="s", markersize=3, linewidth=1.5, label=labels[method])
            ax.fill_between(x, mean - 2 * std, mean + 2 * std, color=colors[method], alpha=0.14, label="±2 std")
            ax.set_title(f"{TASK_LABELS[task]}: {labels[method]}")
            ax.set_xlabel(axis_col)
            ax.set_ylabel("Physical value")
            ax.legend(loc="best")
    savefig("09_bayesian_method_uncertainty_curves.png")


def write_tables(
    symbolic: pd.DataFrame,
    bayes_symbolic: pd.DataFrame,
    bayes_methods: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([symbolic, bayes_symbolic], ignore_index=True).to_csv(
        OUT / "symbolic_metric_table.csv", index=False
    )
    bayes_methods.to_csv(OUT / "bayesian_method_metric_table.csv", index=False)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)

    symbolic = load_symbolic_summary()
    bayes_symbolic = load_bayes_symbolic_summary()
    bayes_methods = load_bayes_methods_summary()

    plot_metric_overview(symbolic, bayes_symbolic)
    plot_bayes_method_comparison(bayes_methods)
    plot_training_curves()
    plot_prediction_scatter()
    plot_residual_histograms()
    plot_axis_curves()
    plot_uncertainty_quality()
    plot_curve_method_comparison()
    plot_bayesian_method_uncertainty_curves()
    write_tables(symbolic, bayes_symbolic, bayes_methods)

    print(f"Visualization report written to: {OUT}")


if __name__ == "__main__":
    main()
