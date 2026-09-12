"""
P0-5: TCAD-referenced compact-model evidence package.

This script fills three paper-facing gaps:

1. PDK/CML-style engineering baselines:
   LUT-like kNN interpolation, polynomial response surface, spline response
   surface, and scalable RBF-kernel ridge regression.

2. Structured generalization:
   hold out physically meaningful regions such as high active-layer length,
   high temperature, high trap coefficient, and extreme reverse bias.

4. Error-cost value table:
   combine accuracy, parameter/storage cost, inference time, Verilog-A readiness,
   uncertainty support, and interpretability into one compact table.

Default run:
  python -B bkan/simulation/paper_experiments/p0_5_tcad_reference_experiments.py

Faster smoke test:
  python -B bkan/simulation/paper_experiments/p0_5_tcad_reference_experiments.py --sections engineering value --seeds 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = PROJECT_ROOT.parent
ARTIFACT_RESULTS = REPO_ROOT / "artifacts" / "results"
DEFAULT_DATA = ARTIFACT_RESULTS / "device_modeling" / "cleaned_data.csv"
DEFAULT_ENGINEERING_OUT = ARTIFACT_RESULTS / "pdk_style_baselines"
DEFAULT_STRUCTURED_OUT = ARTIFACT_RESULTS / "structured_generalization"
DEFAULT_VALUE_OUT = ARTIFACT_RESULTS / "compact_model_value"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .p0_2_generalization_sweep import (  # type: ignore  # noqa: E402
        TASKS,
        MLP_ARCHES,
        compute_metrics,
        fixed_split,
        load_capacitance_dataframe,
        load_dataframe,
        make_scaled_arrays,
        prepare_task_dataframe,
        resolve_device,
        train_dkan,
        train_mlp,
        transformed_target,
    )
except ImportError:
    from p0_2_generalization_sweep import (  # noqa: E402
        TASKS,
        MLP_ARCHES,
        compute_metrics,
        fixed_split,
        load_capacitance_dataframe,
        load_dataframe,
        make_scaled_arrays,
        prepare_task_dataframe,
        resolve_device,
        train_dkan,
        train_mlp,
        transformed_target,
    )


ENGINEERING_MODELS = [
    "pdk_lut_knn",
    "poly3_ridge",
    "spline_ridge",
    "rbf_nystroem",
]

MODEL_LABELS = {
    "pdk_lut_knn": "LUT/kNN interpolation",
    "poly3_ridge": "Polynomial response surface",
    "spline_ridge": "Spline response surface",
    "rbf_nystroem": "RBF kernel ridge",
    "physics_simple": "Analytical physics",
    "physics_full": "Physics / Eq.-Circuit",
    "mlp_n": "MLP-N",
    "mlp_l": "MLP-L",
    "dkan": "D-KAN",
    "bkan": "B-KAN",
}

MODEL_COLORS = {
    "pdk_lut_knn": "#795548",
    "poly3_ridge": "#607D8B",
    "spline_ridge": "#009688",
    "rbf_nystroem": "#3F51B5",
    "physics_simple": "#E53935",
    "physics_full": "#FF7043",
    "mlp_n": "#FF9800",
    "mlp_l": "#9C27B0",
    "dkan": "#2196F3",
    "bkan": "#4CAF50",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_run_config(args: argparse.Namespace, output_dir: Path, section: str) -> None:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload["section"] = section
    (output_dir / "config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_engineering_model(
    model_name: str,
    n_train: int,
    n_features: int,
    seed: int,
    knn_neighbors: int,
    rbf_components: int,
) -> Pipeline:
    if model_name == "pdk_lut_knn":
        return Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=max(1, min(knn_neighbors, n_train)),
                weights="distance",
                p=2,
            )),
        ])

    if model_name == "poly3_ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("ridge", Ridge(alpha=1e-6)),
        ])

    if model_name == "spline_ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("spline", SplineTransformer(
                n_knots=6,
                degree=3,
                include_bias=False,
            )),
            ("ridge", Ridge(alpha=1e-5)),
        ])

    if model_name == "rbf_nystroem":
        n_components = max(16, min(rbf_components, n_train))
        return Pipeline([
            ("scale", StandardScaler()),
            ("rbf", Nystroem(
                kernel="rbf",
                gamma=1.0 / max(1, n_features),
                n_components=n_components,
                random_state=seed,
            )),
            ("ridge", Ridge(alpha=1e-4)),
        ])

    raise ValueError(f"Unknown engineering model: {model_name}")


def estimate_model_cost(model_name: str, model: Pipeline, n_train: int, n_features: int) -> int:
    if model_name == "pdk_lut_knn":
        return int(n_train * (n_features + 1))

    if "ridge" in model.named_steps:
        ridge = model.named_steps["ridge"]
        count = int(np.size(ridge.coef_) + np.size(ridge.intercept_))
        if "rbf" in model.named_steps:
            rbf = model.named_steps["rbf"]
            count += int(np.size(rbf.components_))
        return count

    return 0


def time_predict(model, x_test: np.ndarray, repeats: int = 30) -> Dict[str, float]:
    if len(x_test) == 0:
        return {"infer_bs1_s": float("nan"), "infer_per_sample_s": float("nan")}

    x1 = x_test[:1]
    xb = x_test[: min(len(x_test), 1024)]

    for _ in range(3):
        model.predict(x1)
        model.predict(xb)

    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(x1)
    bs1 = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(xb)
    per_sample = (time.perf_counter() - start) / (repeats * len(xb))

    return {"infer_bs1_s": bs1, "infer_per_sample_s": per_sample}


def fit_predict_engineering(
    model_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    spec,
    seed: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, float]]:
    x_cols = list(spec.input_cols)
    x_train = df_train[x_cols].to_numpy(dtype=np.float64)
    x_test = df_test[x_cols].to_numpy(dtype=np.float64)
    y_train = transformed_target(df_train, spec)

    model = build_engineering_model(
        model_name=model_name,
        n_train=len(df_train),
        n_features=len(x_cols),
        seed=seed,
        knn_neighbors=args.knn_neighbors,
        rbf_components=args.rbf_components,
    )

    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_time = time.perf_counter() - start
    y_pred = model.predict(x_test).reshape(-1)
    infer = time_predict(model, x_test, repeats=args.timing_repeats)

    info = {
        "param_count": estimate_model_cost(model_name, model, len(df_train), len(x_cols)),
        "train_time_s": fit_time,
        "infer_bs1_s": infer["infer_bs1_s"],
        "infer_per_sample_s": infer["infer_per_sample_s"],
        "storage_points": len(df_train) if model_name == "pdk_lut_knn" else 0,
    }
    return y_pred, info


def summarize_metrics(metrics_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        "rmse_target",
        "mae_target",
        "r2_target",
        "rmse_raw",
        "mae_raw",
        "medape_percent",
        "p95ape_percent",
        "param_count",
        "train_time_s",
        "infer_bs1_s",
        "infer_per_sample_s",
        "n_train",
        "n_test",
    ]
    rows = []
    for keys, group in metrics_df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_runs"] = len(group)
        for col in metric_cols:
            if col not in group.columns:
                continue
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_engineering_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    tasks = list(summary_df["task"].drop_duplicates())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.8 * len(tasks), 4.6), squeeze=False)

    for ax, task in zip(axes.flatten(), tasks):
        sub = summary_df[summary_df["task"] == task].copy()
        sub = sub.sort_values("rmse_target_mean")
        labels = [MODEL_LABELS.get(m, m) for m in sub["model"]]
        colors = [MODEL_COLORS.get(m, "#888888") for m in sub["model"]]
        ax.bar(np.arange(len(sub)), sub["rmse_target_mean"], color=colors, alpha=0.86)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("RMSE (target space)")
        ax.set_title(task, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("PDK/CML-Style Engineering Baselines vs TCAD Reference", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "engineering_baseline_rmse.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_engineering_report(summary_df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    display_cols = [
        "task",
        "model",
        "n_runs",
        "rmse_target_mean",
        "r2_target_mean",
        "param_count_mean",
        "train_time_s_mean",
        "infer_bs1_s_mean",
    ]
    show = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
    show["model"] = show["model"].map(lambda m: MODEL_LABELS.get(m, m))

    lines = [
        "# PDK/CML-Style Engineering Baselines",
        "",
        "## Purpose",
        "",
        "Evaluate non-neural engineering compact-model baselines commonly used as "
        "PDK/CML-style surrogates: LUT-like interpolation, polynomial response "
        "surface, spline response surface, and scalable RBF kernel regression.",
        "",
        "All metrics are measured against the same TCAD reference data used by the "
        "KAN/MLP experiments.",
        "",
        "## Configuration",
        "",
        f"- Data: `{args.data}`",
        f"- Tasks: `{', '.join(args.tasks)}`",
        f"- Models: `{', '.join(args.engineering_models)}`",
        f"- Random seeds: `{', '.join(map(str, args.seeds))}`",
        f"- Split: 70/30 train/test per seed",
        "",
        "## Summary",
        "",
        df_to_markdown(show),
        "",
        "## Output Files",
        "",
        "- `metrics_by_seed.csv`",
        "- `metrics_summary.csv`",
        "- `predictions.csv`",
        "- `engineering_baseline_rmse.png`",
    ]
    (output_dir / "pdk_style_baselines_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_engineering_baselines(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(args.engineering_output_dir)
    ensure_dir(output_dir)
    write_run_config(args, output_dir, "engineering")

    rows = []
    pred_rows = []

    for task_name in args.tasks:
        spec = TASKS[task_name]
        df_raw = (
            load_capacitance_dataframe(Path(args.capacitance_data))
            if task_name == "Capacitance"
            else load_dataframe(Path(args.data))
        )
        df_task = prepare_task_dataframe(df_raw, spec)
        for seed in args.seeds:
            df_train, df_test = fixed_split(df_task, seed=seed)
            y_true = transformed_target(df_test, spec)
            y_raw = df_test[spec.target_col].to_numpy(dtype=np.float64)

            for model_name in args.engineering_models:
                print(f"[engineering] task={task_name} seed={seed} model={model_name}")
                y_pred, info = fit_predict_engineering(model_name, df_train, df_test, spec, seed, args)
                metrics = compute_metrics(y_true, y_raw, y_pred, spec)
                rows.append({
                    "task": task_name,
                    "seed": seed,
                    "model": model_name,
                    **metrics,
                    **info,
                    "n_train": len(df_train),
                    "n_test": len(df_test),
                })
                pred_rows.extend({
                    "task": task_name,
                    "seed": seed,
                    "model": model_name,
                    "row_id": int(idx),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                } for idx, (yt, yp) in enumerate(zip(y_true, y_pred)))

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(metrics_df, ["task", "model"])
    metrics_df.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir / "predictions.csv", index=False)
    save_engineering_plot(summary_df, output_dir)
    write_engineering_report(summary_df, args, output_dir)
    return metrics_df, summary_df


def scenario_specs_for_task(task_name: str) -> List[Tuple[str, str, str, float]]:
    if task_name == "I_dark":
        return [
            ("voltage_extreme_reverse", "dark_voltage", "low", 0.20),
            ("length_high", "active_layer_length", "high", 0.20),
            ("temperature_high", "simulation_temperature", "high", 0.20),
            ("trap_high", "trap_assisted_recomb_A", "high", 0.20),
        ]
    if task_name == "I_photo":
        return [
            ("voltage_extreme_reverse", "light_voltage", "low", 0.20),
            ("length_high", "active_layer_length", "high", 0.20),
            ("temperature_high", "simulation_temperature", "high", 0.20),
            ("trap_high", "trap_assisted_recomb_A", "high", 0.20),
        ]
    if task_name == "AC_Response":
        return [
            ("frequency_high", "frequency_ghz", "high", 0.20),
            ("length_high", "active_layer_length", "high", 0.20),
            ("temperature_high", "simulation_temperature", "high", 0.20),
        ]
    if task_name == "Capacitance":
        return [
            ("voltage_extreme_reverse", "bias_v", "low", 0.20),
            ("frequency_high", "log_frequency_ghz", "high", 0.20),
            ("length_high", "active_layer_length", "high", 0.20),
            ("temperature_high", "simulation_temperature", "high", 0.20),
            ("trap_high", "trap_assisted_recomb_A", "high", 0.20),
        ]
    return []


def structured_split(
    df: pd.DataFrame,
    column: str,
    side: str,
    fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    values = pd.to_numeric(df[column], errors="coerce")
    if side == "low":
        threshold = float(values.quantile(fraction))
        test_mask = values <= threshold
    elif side == "high":
        threshold = float(values.quantile(1.0 - fraction))
        test_mask = values >= threshold
    else:
        raise ValueError(f"Unknown holdout side: {side}")

    df_test = df.loc[test_mask].reset_index(drop=True)
    df_train = df.loc[~test_mask].reset_index(drop=True)
    if len(df_train) < 30 or len(df_test) < 20:
        raise ValueError(
            f"Structured split too small for {column}/{side}: "
            f"train={len(df_train)}, test={len(df_test)}"
        )
    return df_train, df_test, threshold


def fit_predict_structured_model(
    model_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    spec,
    seed: int,
    device,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if model_name in ENGINEERING_MODELS:
        return fit_predict_engineering(model_name, df_train, df_test, spec, seed, args)

    arrays = make_scaled_arrays(df_train, df_test, spec)
    if model_name == "mlp_l":
        return train_mlp(
            arrays,
            MLP_ARCHES["mlp_l"],
            seed=seed,
            device=device,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            lamb=args.mlp_lamb,
            lamb_l1=args.mlp_lamb_l1,
            lamb_entropy=args.mlp_lamb_entropy,
        )
    if model_name == "dkan":
        return train_dkan(
            arrays,
            seed=seed,
            device=device,
            steps=args.kan_steps,
            grid=8,
            k=3,
            width=8,
        )
    if model_name == "bkan":
        try:
            from .p0_3_traditional_comparison import train_bkan  # type: ignore
        except ImportError:
            from p0_3_traditional_comparison import train_bkan
        return train_bkan(
            df_train,
            df_test,
            spec,
            seed=seed,
            device=device,
            output_dir=Path(args.structured_output_dir),
            epochs=args.bkan_epochs,
            mc_samples=args.bkan_mc_samples,
            grid=8,
            k=3,
            width=8,
            run_tag=args.structured_run_tag,
            batch_size=args.bkan_batch_size,
        )
    raise ValueError(f"Unknown structured model: {model_name}")


def save_structured_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    tasks = list(summary_df["task"].drop_duplicates())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(11.5, 3.7 * len(tasks)), squeeze=False)

    for ax, task in zip(axes.flatten(), tasks):
        sub = summary_df[summary_df["task"] == task].copy()
        scenarios = list(sub["scenario"].drop_duplicates())
        models = list(sub["model"].drop_duplicates())
        x = np.arange(len(scenarios))
        width = 0.78 / max(1, len(models))
        for i, model in enumerate(models):
            vals = []
            for scenario in scenarios:
                row = sub[(sub["scenario"] == scenario) & (sub["model"] == model)]
                vals.append(float(row["rmse_target_mean"].iloc[0]) if not row.empty else np.nan)
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=MODEL_COLORS.get(model, "#888888"),
                label=MODEL_LABELS.get(model, model),
                alpha=0.86,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
        ax.set_ylabel("RMSE (target space)")
        ax.set_title(task, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncols=3, fontsize=8)

    fig.suptitle("Structured Holdout Generalization vs TCAD Reference", fontweight="bold")
    fig.tight_layout()
    for artist in fig.findobj(match=matplotlib.text.Text):
        artist.set_fontsize(artist.get_fontsize() * 1.18)
    fig.savefig(output_dir / "structured_generalization_rmse.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_structured_report(summary_df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    display_cols = [
        "task",
        "scenario",
        "holdout_column",
        "holdout_side",
        "model",
        "n_runs",
        "rmse_target_mean",
        "r2_target_mean",
        "n_train_mean",
        "n_test_mean",
    ]
    show = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
    show["model"] = show["model"].map(lambda m: MODEL_LABELS.get(m, m))

    lines = [
        "# Structured Generalization",
        "",
        "## Purpose",
        "",
        "Test extrapolation-like TCAD development scenarios by holding out physically "
        "meaningful regions rather than using only random interpolation splits.",
        "",
        "## Configuration",
        "",
        f"- Data: `{args.data}`",
        f"- Tasks: `{', '.join(args.tasks)}`",
        f"- Models: `{', '.join(args.structured_models)}`",
        f"- Holdout fraction: `{args.structured_holdout_fraction}`",
        f"- Stochastic model seeds: `{', '.join(map(str, args.structured_seeds))}`",
        "- Deterministic models run once per holdout; stochastic models are summarized across seeds.",
        f"- B-KAN epochs / batch size: `{args.bkan_epochs}` / `{args.bkan_batch_size}`",
        "",
        "## Summary",
        "",
        df_to_markdown(show),
        "",
        "## Output Files",
        "",
        "- `metrics_by_split.csv`",
        "- `metrics_summary.csv`",
        "- `structured_generalization_rmse.png`",
    ]
    (output_dir / "structured_generalization_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_structured_generalization(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(args.structured_output_dir)
    ensure_dir(output_dir)
    write_run_config(args, output_dir, "structured")

    device = resolve_device(args.device)
    rows = []

    for task_name in args.tasks:
        spec = TASKS[task_name]
        df_raw = (
            load_capacitance_dataframe(Path(args.capacitance_data))
            if task_name == "Capacitance"
            else load_dataframe(Path(args.data))
        )
        df_task = prepare_task_dataframe(df_raw, spec)
        scenarios = scenario_specs_for_task(task_name)
        if args.structured_scenarios:
            wanted = set(args.structured_scenarios)
            scenarios = [item for item in scenarios if item[0] in wanted]
        for scenario, column, side, default_fraction in scenarios:
            fraction = args.structured_holdout_fraction or default_fraction
            df_train, df_test, threshold = structured_split(df_task, column, side, fraction)
            y_true = transformed_target(df_test, spec)
            y_raw = df_test[spec.target_col].to_numpy(dtype=np.float64)

            for model_name in args.structured_models:
                stochastic = model_name in {"rbf_nystroem", "mlp_l", "dkan", "bkan"}
                model_seeds = args.structured_seeds if stochastic else [None]
                for seed in model_seeds:
                    effective_seed = args.structured_seeds[0] if seed is None else seed
                    args.structured_run_tag = f"{task_name}_{scenario}_seed_{effective_seed}"
                    print(
                        f"[structured] task={task_name} scenario={scenario} "
                        f"model={model_name} seed={effective_seed} "
                        f"train={len(df_train)} test={len(df_test)}"
                    )
                    y_pred, info = fit_predict_structured_model(
                        model_name=model_name,
                        df_train=df_train,
                        df_test=df_test,
                        spec=spec,
                        seed=effective_seed,
                        device=device,
                        args=args,
                    )
                    metrics = compute_metrics(y_true, y_raw, y_pred, spec)
                    rows.append({
                        "task": task_name,
                        "scenario": scenario,
                        "holdout_column": column,
                        "holdout_side": side,
                        "threshold": threshold,
                        "model": model_name,
                        "seed": effective_seed if stochastic else "deterministic",
                        **metrics,
                        **info,
                        "n_train": len(df_train),
                        "n_test": len(df_test),
                    })

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(
        metrics_df,
        ["task", "scenario", "holdout_column", "holdout_side", "model"],
    )
    metrics_df.to_csv(output_dir / "metrics_by_split.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    save_structured_plot(summary_df, output_dir)
    write_structured_report(summary_df, args, output_dir)
    return metrics_df, summary_df


def read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def model_metadata() -> Dict[str, Dict[str, str]]:
    return {
        "physics_simple": {
            "family": "classical analytical baseline",
            "verilog_a_ready": "manual",
            "uncertainty": "no",
            "interpretability": "high",
            "notes": "First-order compact physics baseline.",
        },
        "physics_full": {
            "family": "classical physics / equivalent-circuit baseline",
            "verilog_a_ready": "manual",
            "uncertainty": "no",
            "interpretability": "high",
            "notes": "Mechanism-based baseline, not a BSIM-like standard model.",
        },
        "pdk_lut_knn": {
            "family": "PDK/CML-style table interpolation",
            "verilog_a_ready": "table/manual",
            "uncertainty": "no",
            "interpretability": "low",
            "notes": "Stores train table; strong interpolation baseline.",
        },
        "poly3_ridge": {
            "family": "response surface",
            "verilog_a_ready": "yes",
            "uncertainty": "no",
            "interpretability": "medium",
            "notes": "Polynomial compact response surface.",
        },
        "spline_ridge": {
            "family": "response surface",
            "verilog_a_ready": "manual",
            "uncertainty": "no",
            "interpretability": "medium",
            "notes": "Spline compact response surface.",
        },
        "rbf_nystroem": {
            "family": "kernel surrogate",
            "verilog_a_ready": "no",
            "uncertainty": "no",
            "interpretability": "low",
            "notes": "Scalable approximation to RBF kernel regression.",
        },
        "mlp_n": {
            "family": "neural baseline",
            "verilog_a_ready": "no",
            "uncertainty": "no",
            "interpretability": "low",
            "notes": "Parameter-matched neural baseline.",
        },
        "mlp_l": {
            "family": "neural baseline",
            "verilog_a_ready": "no",
            "uncertainty": "no",
            "interpretability": "low",
            "notes": "Larger MLP baseline.",
        },
        "dkan": {
            "family": "KAN compact surrogate",
            "verilog_a_ready": "via symbolic distillation",
            "uncertainty": "no",
            "interpretability": "medium-high",
            "notes": "Spline KAN; symbolic extraction path exists.",
        },
        "bkan": {
            "family": "Bayesian KAN compact surrogate",
            "verilog_a_ready": "via posterior-mean symbolic distillation",
            "uncertainty": "yes",
            "interpretability": "medium-high",
            "notes": "Adds epistemic uncertainty and OOD diagnostics.",
        },
    }


def load_inference_times() -> Dict[str, float]:
    path = ARTIFACT_RESULTS / "inference_benchmark" / "inference_benchmark_data.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    times = data.get("inference_time", {})
    formula = data.get("formula_complexity", {}).get("inference_times", {})
    return {
        "dkan": float(times.get("bs1_kan_deterministic", np.nan)),
        "bkan": float(times.get("bs1_bayeskan_mc100", np.nan)),
        "mlp_n": float(times.get("bs1_mlp_s", np.nan)),
        "mlp_l": float(times.get("bs1_mlp_m", np.nan)),
        "symbolic_veriloga": float(formula.get("1", np.nan)),
    }


def default_param_count(model: str, task: str) -> float:
    defaults = {
        "dkan": 952.0,
        "bkan": 1664.0,
        "mlp_n": 769.0,
        "mlp_l": 2561.0,
    }
    if model in defaults:
        return defaults[model]
    if model == "physics_simple":
        return 3.0 if task == "I_photo" else 4.0
    if model == "physics_full":
        if task == "I_dark":
            return 9.0
        if task == "I_photo":
            return 3.0
        if task == "AC_Response":
            return 5.0
    return np.nan


def collect_accuracy_tables(args: argparse.Namespace) -> pd.DataFrame:
    frames = []

    p0_3 = ARTIFACT_RESULTS / "traditional_vs_kan" / "metrics_summary.csv"
    df_trad = read_optional_csv(p0_3)
    if not df_trad.empty:
        df_trad = df_trad.rename(columns={
            "n_seeds": "n_runs",
            "param_count_mean": "param_count_mean",
        })
        keep = [
            "task",
            "model",
            "n_runs",
            "rmse_target_mean",
            "rmse_target_std",
            "r2_target_mean",
            "r2_target_std",
        ]
        frames.append(df_trad[[c for c in keep if c in df_trad.columns]].copy())

    engineering_path = Path(args.engineering_output_dir) / "metrics_summary.csv"
    df_eng = read_optional_csv(engineering_path)
    if not df_eng.empty:
        df_eng = df_eng.rename(columns={"n_runs": "n_runs"})
        keep = [
            "task",
            "model",
            "n_runs",
            "rmse_target_mean",
            "rmse_target_std",
            "r2_target_mean",
            "r2_target_std",
            "param_count_mean",
            "train_time_s_mean",
            "infer_bs1_s_mean",
        ]
        frames.append(df_eng[[c for c in keep if c in df_eng.columns]].copy())

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["task", "model"], keep="last")
    return combined


def make_value_tables(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(args.value_output_dir)
    ensure_dir(output_dir)

    by_task = collect_accuracy_tables(args)
    if by_task.empty:
        raise RuntimeError("No accuracy tables available. Run engineering or P0-3 first.")

    metadata = model_metadata()
    inference = load_inference_times()

    for col in ["param_count_mean", "train_time_s_mean", "infer_bs1_s_mean"]:
        if col not in by_task.columns:
            by_task[col] = np.nan

    by_task["param_count_mean"] = by_task.apply(
        lambda r: r["param_count_mean"]
        if pd.notna(r["param_count_mean"])
        else default_param_count(str(r["model"]), str(r["task"])),
        axis=1,
    )

    by_task["infer_bs1_s_mean"] = by_task.apply(
        lambda r: r["infer_bs1_s_mean"]
        if pd.notna(r["infer_bs1_s_mean"])
        else inference.get(str(r["model"]), np.nan),
        axis=1,
    )

    by_task["rmse_rank_in_task"] = by_task.groupby("task")["rmse_target_mean"].rank(
        method="average",
        ascending=True,
    )

    for key in ["family", "verilog_a_ready", "uncertainty", "interpretability", "notes"]:
        by_task[key] = by_task["model"].map(lambda m: metadata.get(m, {}).get(key, ""))

    agg_rows = []
    for model, group in by_task.groupby("model", sort=False):
        meta = metadata.get(model, {})
        agg_rows.append({
            "model": model,
            "label": MODEL_LABELS.get(model, model),
            "family": meta.get("family", ""),
            "tasks_covered": ",".join(sorted(group["task"].astype(str).unique())),
            "mean_r2": float(group["r2_target_mean"].mean()),
            "mean_rmse_rank": float(group["rmse_rank_in_task"].mean()),
            "mean_param_or_storage": float(group["param_count_mean"].mean())
            if group["param_count_mean"].notna().any() else np.nan,
            "mean_train_time_s": float(group["train_time_s_mean"].mean())
            if group["train_time_s_mean"].notna().any() else np.nan,
            "bs1_inference_us": float(group["infer_bs1_s_mean"].mean() * 1e6)
            if group["infer_bs1_s_mean"].notna().any() else np.nan,
            "verilog_a_ready": meta.get("verilog_a_ready", ""),
            "uncertainty": meta.get("uncertainty", ""),
            "interpretability": meta.get("interpretability", ""),
            "notes": meta.get("notes", ""),
        })

    aggregate = pd.DataFrame(agg_rows).sort_values(["mean_rmse_rank", "mean_r2"], ascending=[True, False])

    by_task.to_csv(output_dir / "compact_model_value_by_task.csv", index=False)
    aggregate.to_csv(output_dir / "compact_model_value_table.csv", index=False)

    write_value_report(by_task, aggregate, output_dir)
    save_value_plot(aggregate, output_dir)
    return by_task, aggregate


def write_value_report(by_task: pd.DataFrame, aggregate: pd.DataFrame, output_dir: Path) -> None:
    show_cols = [
        "model",
        "label",
        "family",
        "tasks_covered",
        "mean_r2",
        "mean_rmse_rank",
        "mean_param_or_storage",
        "bs1_inference_us",
        "verilog_a_ready",
        "uncertainty",
        "interpretability",
    ]
    lines = [
        "# Error-Cost Value Table",
        "",
        "## Purpose",
        "",
        "Summarize why the proposed KAN/B-KAN compact-model workflow is useful in a "
        "TCAD-to-compact-model setting, rather than claiming to replace a mature "
        "foundry sign-off model.",
        "",
        "Lower `mean_rmse_rank` is better. It ranks models by RMSE within each task "
        "before averaging, avoiding direct comparison of differently scaled targets.",
        "",
        "## Aggregate Table",
        "",
        df_to_markdown(aggregate[[c for c in show_cols if c in aggregate.columns]]),
        "",
        "## Per-Task Accuracy Table",
        "",
        df_to_markdown(by_task[[
            "task",
            "model",
            "rmse_target_mean",
            "r2_target_mean",
            "rmse_rank_in_task",
            "family",
            "verilog_a_ready",
            "uncertainty",
        ]].sort_values(["task", "rmse_rank_in_task"])),
        "",
        "## Output Files",
        "",
        "- `compact_model_value_table.csv`",
        "- `compact_model_value_by_task.csv`",
        "- `accuracy_cost_tradeoff.png`",
    ]
    (output_dir / "compact_model_value_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_value_plot(aggregate: pd.DataFrame, output_dir: Path) -> None:
    plot_df = aggregate.copy()
    plot_df = plot_df[pd.notna(plot_df["bs1_inference_us"])]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    colors = [MODEL_COLORS.get(m, "#777777") for m in plot_df["model"]]
    ax.scatter(plot_df["bs1_inference_us"], plot_df["mean_rmse_rank"], s=80, c=colors, alpha=0.88)
    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row["model"]),
            (row["bs1_inference_us"], row["mean_rmse_rank"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Single-sample inference time (us, log scale)")
    ax.set_ylabel("Mean RMSE rank across tasks (lower is better)")
    ax.set_title("Accuracy-Cost Tradeoff vs TCAD Reference", fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_cost_tradeoff.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P0-5 TCAD-referenced engineering baselines, structured splits, and value table."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--capacitance-data",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "results"
            / "apparent_capacitance_correction"
            / "primary_ge_si_apparent_capacitance.csv"
        ),
    )
    parser.add_argument("--sections", nargs="+", default=["engineering", "structured", "value"],
                        choices=["engineering", "structured", "value"])
    parser.add_argument("--tasks", nargs="+", default=["I_dark", "I_photo", "AC_Response"],
                        choices=list(TASKS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="auto")

    parser.add_argument("--engineering-output-dir", type=Path, default=DEFAULT_ENGINEERING_OUT)
    parser.add_argument("--structured-output-dir", type=Path, default=DEFAULT_STRUCTURED_OUT)
    parser.add_argument("--value-output-dir", type=Path, default=DEFAULT_VALUE_OUT)

    parser.add_argument("--engineering-models", nargs="+", default=ENGINEERING_MODELS,
                        choices=ENGINEERING_MODELS)
    parser.add_argument("--structured-models", nargs="+",
                        default=["pdk_lut_knn", "poly3_ridge", "spline_ridge", "rbf_nystroem", "mlp_l", "dkan", "bkan"],
                        choices=ENGINEERING_MODELS + ["mlp_l", "dkan", "bkan"])
    parser.add_argument("--structured-holdout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--structured-scenarios",
        nargs="+",
        default=None,
        help="Optional scenario-name subset, useful for resumable long runs.",
    )
    parser.add_argument("--structured-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--structured-seed", type=int, default=None,
                        help="Deprecated single-seed compatibility option.")

    parser.add_argument("--knn-neighbors", type=int, default=16)
    parser.add_argument("--rbf-components", type=int, default=384)
    parser.add_argument("--timing-repeats", type=int, default=30)

    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-lr", type=float, default=0.002)
    parser.add_argument("--mlp-lamb", type=float, default=0.001)
    parser.add_argument("--mlp-lamb-l1", type=float, default=1.0)
    parser.add_argument("--mlp-lamb-entropy", type=float, default=2.0)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument("--bkan-epochs", type=int, default=300)
    parser.add_argument("--bkan-mc-samples", type=int, default=50)
    parser.add_argument("--bkan-batch-size", type=int, default=32)

    args = parser.parse_args()
    if args.structured_seed is not None:
        args.structured_seeds = [args.structured_seed]
    if len(set(args.structured_seeds)) < 3:
        print(
            "[warning] Structured stochastic models have fewer than three unique seeds; "
            "paper-facing claims must identify this as a limited run."
        )
    args.structured_run_tag = ""
    return args


def main() -> None:
    args = parse_args()
    if "engineering" in args.sections:
        run_engineering_baselines(args)
    if "structured" in args.sections:
        run_structured_generalization(args)
    if "value" in args.sections:
        make_value_tables(args)


if __name__ == "__main__":
    main()
