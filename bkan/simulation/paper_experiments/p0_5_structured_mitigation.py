"""
Structured generalization mitigation probes.

This script keeps the model class and KAN architecture unchanged, then tests
whether safer preprocessing and boundary-focused training can reduce failures
in structured holdout scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = PROJECT_ROOT.parent
ARTIFACT_RESULTS = REPO_ROOT / "artifacts" / "results"
DEFAULT_DATA = ARTIFACT_RESULTS / "device_modeling" / "cleaned_data.csv"
DEFAULT_OUT = ARTIFACT_RESULTS / "structured_generalization_mitigation"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .p0_2_generalization_sweep import (  # type: ignore
        TASKS,
        compute_metrics,
        load_dataframe,
        make_scaled_arrays,
        prepare_task_dataframe,
        resolve_device,
        train_dkan,
        transformed_target,
    )
    from .p0_5_tcad_reference_experiments import (
        df_to_markdown,
        structured_split,
        summarize_metrics,
    )
except ImportError:
    from p0_2_generalization_sweep import (
        TASKS,
        compute_metrics,
        load_dataframe,
        make_scaled_arrays,
        prepare_task_dataframe,
        resolve_device,
        train_dkan,
        transformed_target,
    )
    from p0_5_tcad_reference_experiments import (
        df_to_markdown,
        structured_split,
        summarize_metrics,
    )


TARGET_SCENARIOS: Dict[str, List[Tuple[str, str, str, float]]] = {
    "I_photo": [("voltage_extreme_reverse", "light_voltage", "low", 0.20)],
    "AC_Response": [("frequency_high", "frequency_ghz", "high", 0.20)],
}

METHODS = [
    "raw_dkan",
    "feature_transform_dkan",
    "boundary_oversample_dkan",
    "transform_oversample_dkan",
    "anchor_5pct_transform_oversample_dkan",
    "anchor_10pct_transform_oversample_dkan",
]


def apply_feature_transform(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """Apply monotone input transforms without changing input dimensionality."""
    out = df.copy()
    if task_name == "I_dark" and "dark_voltage" in out.columns:
        v = pd.to_numeric(out["dark_voltage"], errors="coerce").to_numpy(float)
        out["dark_voltage"] = np.sign(v) * np.log1p(np.abs(v))
    if task_name == "I_photo" and "light_voltage" in out.columns:
        v = pd.to_numeric(out["light_voltage"], errors="coerce").to_numpy(float)
        out["light_voltage"] = np.sign(v) * np.log1p(np.abs(v))
    if task_name == "AC_Response" and "frequency_ghz" in out.columns:
        f = pd.to_numeric(out["frequency_ghz"], errors="coerce").to_numpy(float)
        out["frequency_ghz"] = np.log10(np.maximum(f, 1e-9))
    return out


def oversample_boundary(
    df_train: pd.DataFrame,
    column: str,
    side: str,
    tail_fraction: float,
    repeat: int,
) -> pd.DataFrame:
    if repeat <= 0:
        return df_train.copy()

    values = pd.to_numeric(df_train[column], errors="coerce")
    if side == "low":
        threshold = float(values.quantile(tail_fraction))
        mask = values <= threshold
    elif side == "high":
        threshold = float(values.quantile(1.0 - tail_fraction))
        mask = values >= threshold
    else:
        raise ValueError(f"Unknown holdout side: {side}")

    tail = df_train.loc[mask].copy()
    if tail.empty:
        return df_train.copy()
    return pd.concat([df_train] + [tail] * repeat, ignore_index=True)


def anchor_fraction_for_method(method: str) -> float:
    if method.startswith("anchor_5pct"):
        return 0.05
    if method.startswith("anchor_10pct"):
        return 0.10
    return 0.0


def add_anchor_samples(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    anchor_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if anchor_fraction <= 0.0:
        return df_train.copy(), df_test.copy(), 0
    n_anchor = max(1, int(round(len(df_test) * anchor_fraction)))
    n_anchor = min(n_anchor, len(df_test) - 1)
    anchor = df_test.sample(n=n_anchor, random_state=seed)
    remaining = df_test.drop(index=anchor.index)
    train_with_anchor = pd.concat([df_train, anchor], ignore_index=True)
    return train_with_anchor.reset_index(drop=True), remaining.reset_index(drop=True), n_anchor


def train_variant(
    method: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    task_name: str,
    column: str,
    side: str,
    seed: int,
    device,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, float]]:
    spec = TASKS[task_name]
    train_used = df_train.copy()
    test_used = df_test.copy()

    if "oversample" in method:
        train_used = oversample_boundary(
            train_used,
            column=column,
            side=side,
            tail_fraction=args.boundary_fraction,
            repeat=args.boundary_repeat,
        )

    if "transform" in method:
        train_used = apply_feature_transform(train_used, task_name)
        test_used = apply_feature_transform(test_used, task_name)

    arrays = make_scaled_arrays(train_used, test_used, spec)
    y_pred, info = train_dkan(
        arrays,
        seed=seed,
        device=device,
        steps=args.kan_steps,
        grid=8,
        k=3,
        width=8,
    )
    info.update({
        "n_train_effective": len(train_used),
        "feature_transform": bool("transform" in method),
        "boundary_oversample": bool("oversample" in method),
    })
    return y_pred, info


def save_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return
    scenarios = summary_df[["task", "scenario"]].drop_duplicates()
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(9.2, 3.8 * len(scenarios)), squeeze=False)
    for ax, (_, row) in zip(axes.flatten(), scenarios.iterrows()):
        task = row["task"]
        scenario = row["scenario"]
        sub = summary_df[(summary_df["task"] == task) & (summary_df["scenario"] == scenario)].copy()
        sub = sub.sort_values("rmse_target_mean")
        colors = [
            "#2196F3" if m == "raw_dkan"
            else "#4CAF50" if m.startswith("anchor")
            else "#009688" if "transform" in m
            else "#FF9800"
            for m in sub["method"]
        ]
        ax.bar(np.arange(len(sub)), sub["rmse_target_mean"], color=colors, alpha=0.88)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["method"], rotation=20, ha="right")
        ax.set_ylabel("RMSE (target space)")
        ax.set_title(f"{task} / {scenario}", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "structured_mitigation_rmse.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_report(summary_df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    display_cols = [
        "task",
        "scenario",
        "method",
        "rmse_target_mean",
        "r2_target_mean",
        "n_train_mean",
        "n_train_effective_mean",
        "n_anchor_mean",
        "anchor_fraction_mean",
    ]
    show = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
    lines = [
        "# Structured Generalization Mitigation Probes",
        "",
        "## Scope",
        "",
        "Model class and KAN architecture are unchanged. The probes only change input "
        "preprocessing and/or repeat near-boundary training samples.",
        "",
        "Anchor rows are sampled from the existing structured holdout subset in "
        "`cleaned_data.csv` and then removed from the evaluation subset. They are not "
        "synthetic points and are not newly run TCAD simulations; this is a "
        "retrospective active-sampling simulation.",
        "",
        "## Configuration",
        "",
        f"- Data: `{args.data}`",
        f"- Tasks: `{', '.join(args.tasks)}`",
        f"- Methods: `{', '.join(args.methods)}`",
        f"- KAN steps: `{args.kan_steps}`",
        f"- Boundary fraction/repeat: `{args.boundary_fraction}` / `{args.boundary_repeat}`",
        "",
        "## Summary",
        "",
        df_to_markdown(show),
        "",
        "## Output Files",
        "",
        "- `metrics_by_run.csv`",
        "- `metrics_summary.csv`",
        "- `structured_mitigation_rmse.png`",
    ]
    (output_dir / "structured_mitigation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    df_raw = load_dataframe(Path(args.data))
    rows = []

    config = vars(args).copy()
    config["data"] = str(config["data"])
    config["output_dir"] = str(config["output_dir"])
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for task_name in args.tasks:
        spec = TASKS[task_name]
        df_task = prepare_task_dataframe(df_raw, spec)
        for scenario, column, side, default_fraction in TARGET_SCENARIOS[task_name]:
            fraction = args.structured_holdout_fraction or default_fraction
            df_train, df_test, threshold = structured_split(df_task, column, side, fraction)
            y_true = transformed_target(df_test, spec)
            y_raw = df_test[spec.target_col].to_numpy(dtype=np.float64)

            for method in args.methods:
                anchor_fraction = anchor_fraction_for_method(method)
                df_train_method, df_test_method, n_anchor = add_anchor_samples(
                    df_train,
                    df_test,
                    anchor_fraction=anchor_fraction,
                    seed=args.seed,
                )
                y_true_method = transformed_target(df_test_method, spec)
                y_raw_method = df_test_method[spec.target_col].to_numpy(dtype=np.float64)
                print(
                    f"[mitigation] task={task_name} scenario={scenario} method={method} "
                    f"train={len(df_train_method)} test={len(df_test_method)} anchors={n_anchor}"
                )
                y_pred, info = train_variant(
                    method=method,
                    df_train=df_train_method,
                    df_test=df_test_method,
                    task_name=task_name,
                    column=column,
                    side=side,
                    seed=args.seed,
                    device=device,
                    args=args,
                )
                metrics = compute_metrics(y_true_method, y_raw_method, y_pred, spec)
                rows.append({
                    "task": task_name,
                    "scenario": scenario,
                    "holdout_column": column,
                    "holdout_side": side,
                    "threshold": threshold,
                    "method": method,
                    **metrics,
                    **info,
                    "n_anchor": n_anchor,
                    "anchor_fraction": anchor_fraction,
                    "n_train": len(df_train),
                    "n_test": len(df_test_method),
                })

    metrics_df = pd.DataFrame(rows)
    group_cols = ["task", "scenario", "holdout_column", "holdout_side", "method"]
    summary_df = summarize_metrics(metrics_df, group_cols)
    extra_cols = ["n_train_effective", "n_anchor", "anchor_fraction"]
    extra = metrics_df.groupby(group_cols, sort=False)[extra_cols].mean().reset_index()
    summary_df = summary_df.merge(extra, on=group_cols, how="left", suffixes=("", "_mean"))
    summary_df = summary_df.rename(columns={
        "n_train_effective": "n_train_effective_mean",
        "n_anchor": "n_anchor_mean",
        "anchor_fraction": "anchor_fraction_mean",
    })
    metrics_df.to_csv(output_dir / "metrics_by_run.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    save_plot(summary_df, output_dir)
    write_report(summary_df, args, output_dir)
    return metrics_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured holdout mitigation probes.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tasks", nargs="+", default=["I_photo", "AC_Response"], choices=list(TARGET_SCENARIOS))
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument("--structured-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--boundary-fraction", type=float, default=0.25)
    parser.add_argument("--boundary-repeat", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
