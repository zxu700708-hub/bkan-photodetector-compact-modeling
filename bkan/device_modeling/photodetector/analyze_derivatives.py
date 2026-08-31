"""Derivative analysis for the unified KAN device-model workflow."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # Plotting is optional; CSV and markdown reports still work.
    plt = None

try:
    import sympy as sp
except ImportError:  # Formula-only analysis is optional.
    sp = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler
from device_modeling.photodetector.physical_library import finite_difference_derivative
from device_modeling.photodetector.task_config import (
    DEFAULT_DATA,
    DEFAULT_RESULTS,
    TASKS,
    load_research_data,
    prepare_task_dataframe,
    split_task_dataframe,
)


DERIVATIVE_COLUMNS = {
    "reference": "reference_derivative_model_space",
    "prediction": "prediction_derivative_model_space",
    "error": "abs_derivative_error_model_space",
}

FORMULA_SECTION_HEADERS = (
    "Formula in model space:",
    "Formula in Transformed space:",
    "Formula in transformed space:",
    "Formula in physical space:",
    "Formula:",
)

SYMPY_LOCAL_NAMES = {
    "Abs": "Abs",
    "Max": "Max",
    "Min": "Min",
    "cos": "cos",
    "erf": "erf",
    "exp": "exp",
    "log": "log",
    "log10": "log",
    "pi": "pi",
    "sin": "sin",
    "sqrt": "sqrt",
    "tanh": "tanh",
}

METHOD_FAMILY = {
    "deterministic_kan": "B-spline/autograd",
    "symbolic_gated_kan": "B-spline/autograd",
    "bayesian_kan_vi": "Bayesian derivative",
    "bayesian_kan_dropout": "Bayesian derivative",
    "bayesian_kan_hmc": "Bayesian derivative",
}


def setup_style() -> None:
    if plt is None:
        return
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
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def regression_error_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mask = np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(mask):
        return {
            "count": 0.0,
            "derivative_mae": float("nan"),
            "derivative_rmse": float("nan"),
            "derivative_median_abs_error": float("nan"),
            "derivative_medape": float("nan"),
            "derivative_medape_nonzero": float("nan"),
        }

    ref = reference[mask]
    pred = prediction[mask]
    abs_error = np.abs(pred - ref)
    denominator = np.maximum(np.abs(ref), 1e-12)
    nonzero = np.abs(ref) > 1e-9
    return {
        "count": float(len(ref)),
        "derivative_mae": float(np.mean(abs_error)),
        "derivative_rmse": float(np.sqrt(np.mean((pred - ref) ** 2))),
        "derivative_median_abs_error": float(np.median(abs_error)),
        "derivative_medape": float(np.median(abs_error / denominator)),
        "derivative_medape_nonzero": (
            float(np.median(abs_error[nonzero] / np.abs(ref[nonzero])))
            if np.any(nonzero)
            else float("nan")
        ),
    }


def task_result_dirs(results_root: Path, task_key: str) -> dict[str, list[Path]]:
    return {
        "deterministic_kan": [
            results_root / f"{task_key}_deterministic_kan",
            results_root / "deterministic" / f"{task_key}_deterministic_kan",
        ],
        "symbolic_gated_kan": [
            results_root / f"{task_key}_symbolic_gated_kan",
            results_root / "symbolic_gated" / f"{task_key}_symbolic_gated_kan",
            results_root / "bayesian_symbolic_hmc" / f"{task_key}_symbolic_gated_kan",
            results_root / "bayesian_symbolic" / f"{task_key}_symbolic_gated_kan",
        ],
    }


def bayes_checkpoint_dirs(results_root: Path, result_subdir: str) -> dict[str, list[Path]]:
    return {
        "bayesian_kan_vi": [
            results_root / result_subdir,
            results_root / "bayesian" / result_subdir,
            results_root / "bayesian_compare" / result_subdir,
        ],
        "bayesian_kan_dropout": [
            results_root / f"{result_subdir}_dropout",
            results_root / "bayesian_compare" / f"{result_subdir}_dropout",
        ],
        "bayesian_kan_hmc": [
            results_root / f"{result_subdir}_hmc",
            results_root / "bayesian_compare" / f"{result_subdir}_hmc",
        ],
    }


def existing_first(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def formula_path_candidates(results_root: Path, task_key: str) -> list[Path]:
    spec = TASKS[task_key]
    return [
        results_root / spec.result_subdir / "symbolic_formula.txt",
        results_root / f"{spec.key}_symbolic_gated_kan" / "symbolic_formula.txt",
        results_root / "symbolic_gated" / f"{spec.key}_symbolic_gated_kan" / "symbolic_formula.txt",
        results_root / "bayesian_symbolic" / f"{spec.key}_symbolic_gated_kan" / "symbolic_formula.txt",
        results_root / "bayesian_symbolic_hmc" / f"{spec.key}_symbolic_gated_kan" / "symbolic_formula.txt",
    ]


def parse_variable_map(lines: list[str]) -> dict[str, str]:
    variable_map: dict[str, str] = {}
    in_map = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Variable map:":
            in_map = True
            continue
        if not in_map:
            continue
        if not stripped or stripped.startswith("Free variables"):
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            variable_map[key] = value
    return variable_map


def _formula_block_after_header(lines: list[str], header: str) -> str | None:
    for index, line in enumerate(lines):
        if header not in line:
            continue
        block = []
        remainder = line.split(header, 1)[1].strip()
        if remainder:
            block.append(remainder)
        for follow in lines[index + 1:]:
            stripped = follow.strip()
            if not stripped:
                if block:
                    break
                continue
            if any(stripped.startswith(candidate) for candidate in FORMULA_SECTION_HEADERS):
                break
            if stripped.endswith(":") and block:
                break
            block.append(stripped)
        formula_text = " ".join(block).strip()
        return formula_text or None
    return None


def extract_formula_text(text: str) -> str:
    lines = text.splitlines()
    for header in FORMULA_SECTION_HEADERS:
        formula_text = _formula_block_after_header(lines, header)
        if formula_text:
            break
    else:
        formula_text = None
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if any(token in stripped for token in ("=", "+", "-", "*", "/", "exp", "sqrt", "log")):
                formula_text = stripped
                break

    if not formula_text:
        raise ValueError("Could not locate a symbolic formula expression")

    if "=" in formula_text and re.match(r"^[A-Za-z_]\w*\s*=", formula_text):
        formula_text = formula_text.split("=", 1)[1].strip()
    return formula_text.replace("^", "**")


def parse_formula_file(formula_path: Path, input_columns: tuple[str, ...]) -> tuple[object, dict[str, str], str]:
    if sp is None:
        raise ImportError("sympy is required for formula-only derivative analysis")

    text = formula_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    variable_map = parse_variable_map(lines)
    formula_text = extract_formula_text(text)

    local_dict = {name: getattr(sp, sympy_name) for name, sympy_name in SYMPY_LOCAL_NAMES.items()}
    local_dict["log10"] = lambda value: sp.log(value, 10)
    symbol_names = set(input_columns)
    symbol_names.update(variable_map.keys())
    symbol_names.update(variable_map.values())
    symbol_names.update(re.findall(r"\b[A-Za-z_]\w*\b", formula_text))
    symbol_names.difference_update(SYMPY_LOCAL_NAMES)
    for name in sorted(symbol_names):
        local_dict.setdefault(name, sp.Symbol(name))

    formula_expr = sp.sympify(formula_text, locals=local_dict)
    return formula_expr, variable_map, formula_text


def axis_symbol_for_formula(
    axis_col: str,
    input_columns: tuple[str, ...],
    variable_map: dict[str, str],
    formula_expr: object,
) -> object:
    if sp is None:
        raise ImportError("sympy is required for formula-only derivative analysis")

    candidates = [axis_col]
    candidates.extend(symbol for symbol, column in variable_map.items() if column == axis_col)
    if axis_col in input_columns:
        candidates.append(f"x{input_columns.index(axis_col)}")
    free_symbols = {symbol.name: symbol for symbol in formula_expr.free_symbols}
    for candidate in candidates:
        if candidate in free_symbols:
            return free_symbols[candidate]
    raise ValueError(f"Formula does not contain axis variable {axis_col}; tried {candidates}")


def run_formula_only_analysis(args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    formula_dir = output_dir / "formula_only"
    formula_dir.mkdir(parents=True, exist_ok=True)

    for task_key in args.task:
        spec = TASKS[task_key]
        input_columns = tuple(spec.input_candidates)
        formula_path = args.formula_path if len(args.task) == 1 and args.formula_path else None
        if formula_path is None:
            formula_path = existing_first(formula_path_candidates(args.results_root, task_key))
        if formula_path is None or not formula_path.exists():
            missing.append(f"{spec.name}: no symbolic_formula.txt found")
            continue

        try:
            formula_expr, variable_map, formula_text = parse_formula_file(formula_path, input_columns)
            axis_symbol = axis_symbol_for_formula(spec.axis_col, input_columns, variable_map, formula_expr)
            derivative_expr = sp.diff(formula_expr, axis_symbol)
            try:
                derivative_expr = sp.simplify(derivative_expr)
            except Exception:
                pass
        except Exception as exc:
            missing.append(f"{spec.name}: failed formula-only derivative analysis for {formula_path}: {exc}")
            continue

        derivative_text = str(derivative_expr)
        detail_path = formula_dir / f"{spec.key}_formula_derivative.txt"
        detail_path.write_text(
            "\n".join(
                [
                    f"task: {spec.name}",
                    f"source: {formula_path}",
                    f"axis: {axis_symbol}",
                    "",
                    "formula:",
                    formula_text,
                    "",
                    "d_formula_d_axis:",
                    derivative_text,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "task": spec.name,
                "axis_column": spec.axis_col,
                "axis_symbol": str(axis_symbol),
                "formula_path": str(formula_path),
                "formula_characters": len(formula_text),
                "derivative_characters": len(derivative_text),
                "formula_ops": int(sp.count_ops(formula_expr)),
                "derivative_ops": int(sp.count_ops(derivative_expr)),
                "derivative_file": str(detail_path),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(formula_dir / "formula_derivative_summary.csv", index=False)
    write_formula_report(summary, missing, formula_dir)
    return summary, missing


def write_formula_report(summary: pd.DataFrame, missing: list[str], formula_dir: Path) -> None:
    lines = ["# Formula-Only Derivative Analysis", ""]
    if summary.empty:
        lines.append("No formula-only derivative results were generated.")
    else:
        report = summary.drop(columns=["formula_path", "derivative_file"]).copy()
        lines.append("| " + " | ".join(report.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(report.columns)) + " |")
        for _, row in report.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in report.columns) + " |")
        lines.extend(["", "Full derivative expressions are saved in `*_formula_derivative.txt`."])
    if missing:
        lines.extend(["", "## Missing Or Skipped Inputs", ""])
        lines.extend(f"- {item}" for item in missing)
    (formula_dir / "formula_derivative_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def derivative_points_from_predictions(
    predictions_path: Path,
    task_name: str,
    method: str,
    axis_col: str,
    split: str,
) -> pd.DataFrame | None:
    frame = pd.read_csv(predictions_path)
    needed = [DERIVATIVE_COLUMNS["reference"], DERIVATIVE_COLUMNS["prediction"], axis_col]
    if not all(column in frame.columns for column in needed):
        return None
    if split != "all" and "split" in frame.columns:
        frame = frame[frame["split"].eq(split)].copy()
    if frame.empty:
        return None

    out = pd.DataFrame(
        {
            "task": task_name,
            "method": method,
            "split": frame["split"] if "split" in frame.columns else split,
            "axis": frame[axis_col].to_numpy(dtype=np.float64),
            "reference_derivative": frame[DERIVATIVE_COLUMNS["reference"]].to_numpy(dtype=np.float64),
            "prediction_derivative": frame[DERIVATIVE_COLUMNS["prediction"]].to_numpy(dtype=np.float64),
            "derivative_std": np.nan,
            "source": str(predictions_path),
        }
    )
    out["abs_derivative_error"] = np.abs(out["prediction_derivative"] - out["reference_derivative"])
    return out


def load_task_frame(data_path: Path, task_key: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    spec = TASKS[task_key]
    raw = load_research_data(data_path)
    return prepare_task_dataframe(raw, spec)


def split_frame(frame: pd.DataFrame, task_key: str, seed: int, split: str) -> pd.DataFrame:
    spec = TASKS[task_key]
    train, validation, calibration, test = split_task_dataframe(frame, spec, seed)
    if split == "train":
        return train
    if split == "validation":
        return validation
    if split == "calibration":
        return calibration
    if split == "test":
        return test
    return pd.concat([train, validation, calibration, test], ignore_index=True)


def derivative_points_from_bayes_checkpoint(
    checkpoint_path: Path,
    task_key: str,
    method: str,
    data_path: Path,
    split: str,
    seed: int,
    n_mc: int,
    device: str,
) -> pd.DataFrame:
    spec = TASKS[task_key]
    data, _ = load_task_frame(data_path, task_key)
    eval_frame = split_frame(data, task_key, seed, split)
    modeler = BayesKANDeviceModeler.load_model(str(checkpoint_path), device=device)
    inputs = tuple(modeler.input_cols)
    reference = finite_difference_derivative(eval_frame, spec, inputs)
    x_values = eval_frame[list(inputs)].to_numpy(dtype=np.float32)
    derivative = modeler.predict_derivative_with_uncertainty(x_values, spec.axis_col, n_mc=n_mc)
    mean_key = "dlogI_dX_mean" if spec.use_log_transform else "dI_dX_mean"
    std_key = "dlogI_dX_std" if spec.use_log_transform else "dI_dX_std"

    out = pd.DataFrame(
        {
            "task": spec.name,
            "method": method,
            "split": split,
            "axis": eval_frame[spec.axis_col].to_numpy(dtype=np.float64),
            "reference_derivative": reference,
            "prediction_derivative": derivative[mean_key],
            "derivative_std": derivative[std_key],
            "source": str(checkpoint_path),
        }
    )
    out["abs_derivative_error"] = np.abs(out["prediction_derivative"] - out["reference_derivative"])
    return out


def summarize_points(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if points.empty:
        return pd.DataFrame()
    for (task, method, split), group in points.groupby(["task", "method", "split"], sort=True):
        metrics = regression_error_metrics(group["reference_derivative"], group["prediction_derivative"])
        metrics.update(
            {
                "task": task,
                "method": method,
                "split": split,
                "mean_derivative_std": float(np.nanmean(group["derivative_std"]))
                if "derivative_std" in group
                else float("nan"),
                "source": "; ".join(sorted(set(group["source"].dropna().astype(str)))),
            }
        )
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    columns = [
        "task",
        "method",
        "split",
        "count",
        "derivative_mae",
        "derivative_rmse",
        "derivative_median_abs_error",
        "derivative_medape",
        "derivative_medape_nonzero",
        "mean_derivative_std",
        "source",
    ]
    return pd.DataFrame(rows)[columns].sort_values(["task", "method", "split"])


def plot_task_derivatives(points: pd.DataFrame, output_dir: Path) -> None:
    if points.empty or plt is None:
        return
    for task, task_points in points.groupby("task", sort=True):
        fig, ax = plt.subplots(figsize=(8, 5))
        ordered = task_points.sort_values("axis")
        ax.scatter(
            ordered["axis"],
            ordered["reference_derivative"],
            s=12,
            alpha=0.35,
            color="#222222",
            label="finite-difference reference",
        )
        for method, group in ordered.groupby("method", sort=True):
            ax.scatter(group["axis"], group["prediction_derivative"], s=12, alpha=0.62, label=method)
        ax.set_title(f"{task} derivative vs axis")
        ax.set_xlabel("axis value")
        ax.set_ylabel("d target / d axis")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{task}_derivative_vs_axis.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, group in task_points.groupby("method", sort=True):
            ax.hist(
                group["abs_derivative_error"].dropna(),
                bins=40,
                alpha=0.45,
                label=method,
                log=True,
            )
        ax.set_title(f"{task} absolute derivative error")
        ax.set_xlabel("absolute derivative error")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{task}_derivative_error_hist.png", bbox_inches="tight")
        plt.close(fig)


def build_bspline_diagnostic_summary(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    rows = []
    for (task, split), group in points.groupby(["task", "split"], sort=True):
        reference = group["reference_derivative"].to_numpy(dtype=np.float64)
        reference = reference[np.isfinite(reference)]
        if len(reference):
            rows.append(
                {
                    "task": task,
                    "method_family": "finite-difference reference",
                    "method": "finite_difference_reference",
                    "split": split,
                    "count": float(len(reference)),
                    "derivative_mae": 0.0,
                    "derivative_rmse": 0.0,
                    "derivative_median_abs_error": 0.0,
                    "derivative_medape": 0.0,
                    "derivative_medape_nonzero": 0.0,
                    "mean_derivative_std": float("nan"),
                    "reference_derivative_mean": float(np.mean(reference)),
                    "reference_derivative_std": float(np.std(reference)),
                    "source": "finite_difference_derivative",
                }
            )

    model_summary = summarize_points(points)
    if not model_summary.empty:
        for _, row in model_summary.iterrows():
            row_dict = row.to_dict()
            row_dict["method_family"] = METHOD_FAMILY.get(row_dict["method"], "model derivative")
            task_points = points[
                points["task"].eq(row_dict["task"])
                & points["method"].eq(row_dict["method"])
                & points["split"].eq(row_dict["split"])
            ]
            reference = task_points["reference_derivative"].to_numpy(dtype=np.float64)
            reference = reference[np.isfinite(reference)]
            row_dict["reference_derivative_mean"] = float(np.mean(reference)) if len(reference) else float("nan")
            row_dict["reference_derivative_std"] = float(np.std(reference)) if len(reference) else float("nan")
            rows.append(row_dict)

    if not rows:
        return pd.DataFrame()
    columns = [
        "task",
        "method_family",
        "method",
        "split",
        "count",
        "derivative_mae",
        "derivative_rmse",
        "derivative_median_abs_error",
        "derivative_medape",
        "derivative_medape_nonzero",
        "mean_derivative_std",
        "reference_derivative_mean",
        "reference_derivative_std",
        "source",
    ]
    return pd.DataFrame(rows)[columns].sort_values(["task", "method_family", "method", "split"])


def plot_bspline_diagnostics(points: pd.DataFrame, output_dir: Path) -> None:
    if points.empty or plt is None:
        return
    for task, task_points in points.groupby("task", sort=True):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ordered = task_points.sort_values("axis")
        ax.scatter(
            ordered["axis"],
            ordered["reference_derivative"],
            s=14,
            alpha=0.32,
            color="#222222",
            label="finite-difference reference",
        )
        for method, group in ordered.groupby("method", sort=True):
            family = METHOD_FAMILY.get(method, "model derivative")
            group = group.sort_values("axis")
            label = f"{method} ({family})"
            ax.plot(group["axis"], group["prediction_derivative"], marker="o", markersize=2.4, linewidth=0.9, label=label)
            if "derivative_std" in group and np.isfinite(group["derivative_std"]).any():
                std = group["derivative_std"].to_numpy(dtype=np.float64)
                pred = group["prediction_derivative"].to_numpy(dtype=np.float64)
                axis = group["axis"].to_numpy(dtype=np.float64)
                finite = np.isfinite(std) & np.isfinite(pred) & np.isfinite(axis)
                ax.fill_between(axis[finite], pred[finite] - std[finite], pred[finite] + std[finite], alpha=0.12)
        ax.set_title(f"{task}: B-spline, finite-difference, Bayesian derivative comparison")
        ax.set_xlabel("axis value")
        ax.set_ylabel("d target / d axis")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{task}_bspline_bayes_derivative_comparison.png", bbox_inches="tight")
        plt.close(fig)


def write_bspline_report(summary: pd.DataFrame, missing: list[str], output_dir: Path) -> None:
    lines = ["# B-Spline / Finite-Difference / Bayesian Derivative Diagnostics", ""]
    if summary.empty:
        lines.append("No B-spline derivative diagnostics were generated.")
    else:
        view = summary.copy()
        for column in [
            "derivative_mae",
            "derivative_rmse",
            "derivative_median_abs_error",
            "derivative_medape",
            "derivative_medape_nonzero",
            "mean_derivative_std",
            "reference_derivative_mean",
            "reference_derivative_std",
        ]:
            view[column] = view[column].map(lambda value: f"{value:.6g}" if np.isfinite(value) else "nan")
        report_columns = [column for column in view.columns if column != "source"]
        lines.append("| " + " | ".join(report_columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(report_columns)) + " |")
        for _, row in view[report_columns].iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in report_columns) + " |")
        lines.extend(
            [
                "",
                "Method family mapping:",
                "- deterministic_kan and symbolic_gated_kan: B-spline/autograd derivative columns saved in predictions.csv.",
                "- bayesian_kan_vi, bayesian_kan_dropout, bayesian_kan_hmc: Bayesian derivative recomputed from model checkpoints.",
                "- finite_difference_reference: numerical reference from observed task curves.",
            ]
        )
    if missing:
        lines.extend(["", "## Missing Or Skipped Inputs", ""])
        lines.extend(f"- {item}" for item in missing)
    (output_dir / "bspline_derivative_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bspline_diagnostics(points: pd.DataFrame, missing: list[str], output_dir: Path) -> pd.DataFrame:
    bspline_dir = output_dir / "bspline_diagnostics"
    bspline_dir.mkdir(parents=True, exist_ok=True)
    summary = build_bspline_diagnostic_summary(points)
    if not summary.empty:
        summary.to_csv(bspline_dir / "bspline_derivative_comparison.csv", index=False)
    plot_bspline_diagnostics(points, bspline_dir)
    write_bspline_report(summary, missing, bspline_dir)
    return summary


def write_report(summary: pd.DataFrame, missing: list[str], output_dir: Path) -> None:
    lines = ["# Derivative Analysis", ""]
    if summary.empty:
        lines.append("No derivative results were found.")
    else:
        view = summary.copy()
        for column in [
            "derivative_mae",
            "derivative_rmse",
            "derivative_median_abs_error",
            "derivative_medape",
            "derivative_medape_nonzero",
            "mean_derivative_std",
        ]:
            view[column] = view[column].map(lambda value: f"{value:.6g}" if np.isfinite(value) else "nan")
        report_columns = [column for column in view.columns if column != "source"]
        lines.append("| " + " | ".join(report_columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(report_columns)) + " |")
        for _, row in view[report_columns].iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in report_columns) + " |")
    if missing:
        lines.extend(["", "## Missing Or Skipped Inputs", ""])
        lines.extend(f"- {item}" for item in missing)
    (output_dir / "derivative_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_derivative_points(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    points = []
    missing = []
    for task_key in args.task:
        spec = TASKS[task_key]

        if "saved_predictions" in args.source:
            for method, candidates in task_result_dirs(args.results_root, spec.key).items():
                result_dir = existing_first(candidates)
                if result_dir is None:
                    missing.append(f"{spec.name}: no saved prediction directory for {method}")
                    continue
                predictions_path = result_dir / "predictions.csv"
                if not predictions_path.exists():
                    missing.append(f"{spec.name}: missing {predictions_path}")
                    continue
                prediction_points = derivative_points_from_predictions(
                    predictions_path,
                    spec.name,
                    method,
                    spec.axis_col,
                    args.split,
                )
                if prediction_points is None:
                    missing.append(f"{spec.name}: {result_dir / 'predictions.csv'} has no derivative columns")
                    continue
                points.append(prediction_points)

        if "bayes_checkpoint" in args.source:
            for method, candidates in bayes_checkpoint_dirs(args.results_root, spec.result_subdir).items():
                result_dir = existing_first(candidates)
                checkpoint = result_dir / "model_checkpoint.pt" if result_dir is not None else None
                if checkpoint is None or not checkpoint.exists():
                    missing.append(f"{spec.name}: no checkpoint for {method}")
                    continue
                try:
                    points.append(
                        derivative_points_from_bayes_checkpoint(
                            checkpoint,
                            task_key,
                            method,
                            args.data,
                            args.split,
                            args.seed,
                            args.n_mc,
                            args.device,
                        )
                    )
                except Exception as exc:
                    missing.append(f"{spec.name}: failed to analyze {checkpoint}: {exc}")

    if not points:
        return pd.DataFrame(), missing
    return pd.concat(points, ignore_index=True), missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze derivative accuracy for unified KAN results.")
    parser.add_argument("--task", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "validation", "calibration", "test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mc", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--formula-only",
        action="store_true",
        help="Parse symbolic_formula.txt and write analytical derivative expressions for each task.",
    )
    parser.add_argument(
        "--formula-path",
        type=Path,
        default=None,
        help="Optional symbolic_formula.txt path. Intended for single-task formula-only analysis.",
    )
    parser.add_argument(
        "--bspline-diagnostics",
        "--bspline",
        action="store_true",
        dest="bspline_diagnostics",
        help="Write a B-spline/autograd, finite-difference, and Bayesian derivative comparison report.",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        choices=["saved_predictions", "bayes_checkpoint"],
        default=["saved_predictions", "bayes_checkpoint"],
        help="Use derivative columns from predictions.csv and/or recompute Bayesian checkpoint derivatives.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_root = args.results_root.resolve()
    args.data = args.data.resolve()
    output_dir = (args.output or (args.results_root / "derivative_analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    points, missing = collect_derivative_points(args)
    if not points.empty:
        points.to_csv(output_dir / "derivative_points.csv", index=False)
    summary = summarize_points(points)
    if not summary.empty:
        summary.to_csv(output_dir / "derivative_summary.csv", index=False)
    plot_task_derivatives(points, output_dir)
    write_report(summary, missing, output_dir)
    if args.formula_only:
        formula_summary, formula_missing = run_formula_only_analysis(args, output_dir)
        if formula_missing:
            missing.extend(formula_missing)
        if not formula_summary.empty:
            print(f"Saved formula-only derivatives for {len(formula_summary)} task(s).")
    if args.bspline_diagnostics:
        bspline_summary = run_bspline_diagnostics(points, missing, output_dir)
        if not bspline_summary.empty:
            print(f"Saved B-spline derivative diagnostics for {len(bspline_summary)} row(s).")

    print(f"Saved derivative analysis to: {output_dir}")
    if summary.empty:
        print("No derivative results were found. See derivative_analysis_report.md for missing inputs.")
    else:
        print(summary.drop(columns=["source"]).to_string(index=False))


if __name__ == "__main__":
    main()
