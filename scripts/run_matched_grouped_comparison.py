"""Run the unified 10-split grouped accuracy comparison.

The Bayesian KAN predictions are loaded from the BKAN training stage of the
current workflow. All comparison models are fitted on the exact same training
groups. Validation, calibration, and test groups remain quarantined; BKAN uses
validation for checkpoint selection and calibration for interval scaling.
Point-accuracy statistics use test data only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
PAPER_EXPERIMENTS = BKAN_ROOT / "simulation" / "paper_experiments"
for path in (BKAN_ROOT, PAPER_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from device_modeling.photodetector import task_config as shared  # noqa: E402
from p0_3_traditional_comparison import (  # noqa: E402
    ENGINEERING_MODELS,
    MLP_ARCHES,
    TASKS,
    build_engineering_model,
    compute_metrics,
    count_engineering_params,
    make_scaled_arrays,
    prepare_task_dataframe,
    resolve_device,
    train_dkan,
    train_mlp,
    transformed_target,
)
from compact_framework_baselines import (  # noqa: E402
    fit_predict_curve_lut,
    fit_predict_gmls,
    fit_predict_semiempirical_compact,
    train_autopinn_adapted,
)


DEFAULT_DATA = ROOT / "artifacts" / "results" / "device_modeling" / "cleaned_data.csv"
DEFAULT_BKAN = ROOT / "artifacts" / "results" / "uq_repeated_grouped"
DEFAULT_CAP_BKAN = ROOT / "artifacts" / "results" / "uq_repeated_grouped_capacitance"
DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "matched_grouped_comparison"
DEFAULT_MODELS = ("bkan", "dkan", "mlp_l", "poly3_ridge", "spline_ridge")
EXPANDED_MODELS = (
    "bkan",
    "dkan",
    "mlp_pm",
    "mlp_l",
    "poly3_ridge",
    "spline_ridge",
    "rbf_nystroem",
    "random_forest",
    "xgboost",
    "gmls",
    "autopinn",
    "curve_lut",
    "semiempirical",
)
CORE_FRAMEWORK_MODELS = (
    "bkan", "dkan", "mlp_l", "spline_ridge", "gmls", "autopinn"
)
COMPACT_EIGHT_MODELS = CORE_FRAMEWORK_MODELS + ("curve_lut", "semiempirical")
PRIMARY_BASELINE = "spline_ridge"
TASK_KEY = {
    "I_dark": "dark_current",
    "I_photo": "photo_current",
    "AC_Response": "ac_response",
    "Capacitance": "capacitance",
}
MODEL_LABEL = {
    "bkan": "BKAN",
    "dkan": "DKAN",
    "mlp_pm": "MLP-PM",
    "mlp_l": "MLP-L",
    "poly3_ridge": "Poly3-Ridge",
    "spline_ridge": "Spline-Ridge",
    "rbf_nystroem": "RBF-Nystroem",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gmls": "GMLS-adapted",
    "autopinn": "AutoPINN-adapted",
    "curve_lut": "Curve-LUT-PCHIP",
    "semiempirical": "PD SemiEmpirical-CM",
}

AUTOPINN_CONSTRAINTS = {
    "I_dark": ("dark_voltage", -1),
    "I_photo": ("light_voltage", -1),
    "AC_Response": ("frequency_ghz", -1),
    "Capacitance": ("bias_v", None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--capacitance-data", type=Path, default=shared.DEFAULT_CAPACITANCE_DATA)
    parser.add_argument("--bkan-root", type=Path, default=DEFAULT_BKAN)
    parser.add_argument("--capacitance-bkan-root", type=Path, default=DEFAULT_CAP_BKAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-root",
        type=Path,
        default=None,
        help="Optional compatible prediction root used before refitting a model.",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=list(TASK_KEY), default=list(TASK_KEY)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(EXPANDED_MODELS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--split-fractions", nargs=4, type=float, default=[0.65, 0.10, 0.15, 0.10]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-lr", type=float, default=0.002)
    parser.add_argument("--mlp-lamb", type=float, default=0.001)
    parser.add_argument("--mlp-lamb-l1", type=float, default=1.0)
    parser.add_argument("--mlp-lamb-entropy", type=float, default=2.0)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument("--autopinn-epochs", type=int, default=300)
    parser.add_argument("--bootstrap-replicates", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.data = args.data.resolve()
    args.capacitance_data = args.capacitance_data.resolve()
    args.bkan_root = args.bkan_root.resolve()
    args.capacitance_bkan_root = args.capacitance_bkan_root.resolve()
    args.output = args.output.resolve()
    if args.reuse_root is not None:
        args.reuse_root = args.reuse_root.resolve()
    shared.validate_split_fractions(args.split_fractions)
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be unique")
    if len(args.seeds) != 10:
        raise ValueError("The confirmatory protocol requires exactly 10 seeds")
    if args.bootstrap_replicates < 1000:
        raise ValueError("At least 1000 bootstrap replicates are required")
    if "bkan" not in args.models or PRIMARY_BASELINE not in args.models:
        raise ValueError("Models must include bkan and spline_ridge")
    if args.autopinn_epochs < 10:
        raise ValueError("AutoPINN requires at least 10 epochs")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if "Capacitance" in args.tasks and not args.capacitance_data.is_file():
        raise FileNotFoundError(args.capacitance_data)


def _load_task(task_name: str, args: argparse.Namespace):
    if task_name == "Capacitance":
        raw = shared.load_capacitance_data(args.capacitance_data)
    else:
        raw = shared.load_research_data(args.data)
    shared_spec = shared.TASKS[TASK_KEY[task_name]]
    frame, inputs = shared.prepare_task_dataframe(raw, shared_spec)
    # Build every baseline with exactly the feature list used by the frozen
    # BKAN run.  This matters when a candidate input is constant (the current
    # AC table has a constant bandwidth_bias), because the shared pipeline
    # intentionally removes constant inputs before both splitting and fitting.
    baseline_spec = replace(
        TASKS[task_name],
        target_col=shared_spec.target_col,
        input_cols=tuple(inputs),
        ylabel=(
            "log10(net_photocurrent)"
            if task_name == "I_photo"
            else TASKS[task_name].ylabel
        ),
    )
    return frame, baseline_spec


def _group_manifest(
    partitions: tuple[pd.DataFrame, ...], shared_spec: shared.TaskSpec, seed: int
) -> pd.DataFrame:
    rows = []
    for split_name, frame in zip(
        ("train", "validation", "calibration", "test"), partitions
    ):
        labels = shared.task_group_labels(frame, shared_spec)
        counts = labels.value_counts(sort=False)
        for group_id, n_points in counts.items():
            rows.append(
                {
                    "task": shared_spec.name,
                    "seed": seed,
                    "split": split_name,
                    "group_id": str(group_id),
                    "n_points": int(n_points),
                }
            )
    manifest = pd.DataFrame(rows)
    overlap = manifest.groupby("group_id")["split"].nunique()
    if (overlap > 1).any():
        raise RuntimeError(f"Grouped split leakage for {shared_spec.name}, seed={seed}")
    return manifest


def _source_root(task_name: str, args: argparse.Namespace) -> Path:
    return args.capacitance_bkan_root if task_name == "Capacitance" else args.bkan_root


def _bkan_predictions_path(task_name: str, seed: int, args: argparse.Namespace) -> Path:
    spec = shared.TASKS[TASK_KEY[task_name]]
    return _source_root(task_name, args) / f"seed_{seed}" / spec.result_subdir / "uq_test_predictions.csv"


def _bkan_manifest_path(task_name: str, seed: int, args: argparse.Namespace) -> Path:
    return _source_root(task_name, args) / f"seed_{seed}" / f"{TASK_KEY[task_name]}_split_manifest.csv"


def _verify_source_manifest(expected: pd.DataFrame, source_path: Path) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing frozen BKAN manifest: {source_path}")
    source = pd.read_csv(source_path, dtype={"group_id": str})
    expected_sets = {
        name: set(group["group_id"].astype(str))
        for name, group in expected.groupby("split")
    }
    source_sets = {
        name: set(group["group_id"].astype(str))
        for name, group in source.groupby("split")
    }
    if expected_sets != source_sets:
        raise RuntimeError(f"Frozen BKAN split does not match regenerated split: {source_path}")


def _prediction_output(args: argparse.Namespace, seed: int, task: str, model: str) -> Path:
    return args.output / "predictions" / f"seed_{seed}" / task / model / "test_predictions.csv"


def _write_prediction(
    path: Path,
    test: pd.DataFrame,
    spec,
    seed: int,
    model: str,
    prediction: np.ndarray,
) -> None:
    frame = test.copy()
    frame.insert(0, "model", model)
    frame.insert(0, "seed", seed)
    frame.insert(0, "task", spec.name)
    frame["actual_model_space"] = transformed_target(test, spec)
    frame["prediction_model_space"] = np.asarray(prediction, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _load_existing_prediction(path: Path, test: pd.DataFrame, spec) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    for column in dict.fromkeys(spec.input_cols):
        if column not in frame:
            raise RuntimeError(f"Prediction file lacks test input {column}: {path}")
        observed = frame[column].to_numpy()
        expected_column = test[column].to_numpy()
        if np.issubdtype(expected_column.dtype, np.number):
            matches = observed.shape == expected_column.shape and np.allclose(
                observed, expected_column, rtol=1e-10, atol=1e-12
            )
        else:
            matches = np.array_equal(observed, expected_column)
        if not matches:
            raise RuntimeError(f"Prediction file does not match test input {column}: {path}")
    actual = frame["actual_model_space"].to_numpy(dtype=np.float64)
    expected = transformed_target(test, spec)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-7, atol=1e-10):
        raise RuntimeError(f"Prediction file does not match regenerated test split: {path}")
    return frame["prediction_model_space"].to_numpy(dtype=np.float64)


def _load_frozen_bkan(test: pd.DataFrame, spec, task_name: str, seed: int, args) -> np.ndarray:
    path = _bkan_predictions_path(task_name, seed, args)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen BKAN predictions: {path}")
    frame = pd.read_csv(path)
    actual = frame["actual_model_space"].to_numpy(dtype=np.float64)
    expected = transformed_target(test, spec)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-7, atol=1e-10):
        raise RuntimeError(f"Frozen BKAN predictions do not match test targets: {path}")
    return frame["prediction_mean_model_space"].to_numpy(dtype=np.float64)


def paired_bootstrap_interval(
    differences: np.ndarray,
    replicates: int = 50000,
    seed: int = 20260804,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 10000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def corrected_repeated_split_statistics(
    differences: np.ndarray,
    train_fraction: float = 0.65,
    test_fraction: float = 0.10,
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:
    """Nadeau--Bengio-style corrected repeated-split inference.

    The repeated test splits reuse groups, so the split-wise differences are
    not treated as independent observations.  The variance correction is
    ``(1 / r + n_test / n_train) * s^2``.
    """

    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    if not 0.0 < train_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("train_fraction and test_fraction must lie in (0, 1)")
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    corrected_se = math.sqrt((1.0 / values.size + test_fraction / train_fraction) * variance)
    if corrected_se == 0.0:
        return mean, mean, 0.0, 0.0 if mean != 0.0 else 1.0
    critical = float(student_t.ppf(0.5 + confidence / 2.0, df=values.size - 1))
    statistic = mean / corrected_se
    p_value = float(2.0 * student_t.sf(abs(statistic), df=values.size - 1))
    return mean - critical * corrected_se, mean + critical * corrected_se, corrected_se, p_value


def _holm_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy(dtype=np.float64))
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    total = len(values)
    raw = values.to_numpy(dtype=np.float64)
    for position, index in enumerate(order):
        candidate = min(1.0, raw[index] * (total - position))
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def paired_statistics(
    metrics: pd.DataFrame,
    replicates: int = 50000,
    seed: int = 20260804,
    train_fraction: float = 0.65,
    test_fraction: float = 0.10,
    models: tuple[str, ...] = DEFAULT_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    baselines = [model for model in models if model != "bkan"]
    for task, task_frame in metrics.groupby("task", sort=False):
        pivot = task_frame.pivot(index="seed", columns="model", values="rmse_target")
        if "bkan" not in pivot or any(model not in pivot for model in baselines):
            raise RuntimeError(f"Incomplete matched model set for {task}")
        if pivot[list(models)].isna().any().any():
            raise RuntimeError(f"Incomplete seed pairing for {task}")
        for offset, baseline in enumerate(baselines):
            delta = (pivot["bkan"] - pivot[baseline]).to_numpy(dtype=np.float64)
            naive_low, naive_high = paired_bootstrap_interval(
                delta, replicates, seed + offset
            )
            low, high, corrected_se, raw_p = corrected_repeated_split_statistics(
                delta, train_fraction=train_fraction, test_fraction=test_fraction
            )
            scale = np.maximum(np.abs(pivot["bkan"]), np.abs(pivot[baseline]))
            tolerance = (1e-6 * np.maximum(scale, np.finfo(np.float64).tiny)).to_numpy(
                dtype=np.float64
            )
            wins = int(np.sum(delta < -tolerance))
            losses = int(np.sum(delta > tolerance))
            ties = int(len(delta) - wins - losses)
            std = float(np.std(delta, ddof=1))
            rows.append(
                {
                    "task": task,
                    "baseline": baseline,
                    "n_pairs": len(delta),
                    "bkan_rmse_mean": float(pivot["bkan"].mean()),
                    "baseline_rmse_mean": float(pivot[baseline].mean()),
                    "paired_delta_mean": float(np.mean(delta)),
                    "paired_delta_median": float(np.median(delta)),
                    "naive_paired_bootstrap_ci95_low": naive_low,
                    "naive_paired_bootstrap_ci95_high": naive_high,
                    "corrected_repeated_split_ci95_low": low,
                    "corrected_repeated_split_ci95_high": high,
                    "corrected_standard_error": corrected_se,
                    "corrected_p_value_raw": raw_p,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "standardized_effect_dz": float(np.mean(delta) / std) if std > 0 else np.nan,
                }
            )
    pairwise = pd.DataFrame(rows)
    pairwise["corrected_p_value_holm"] = pairwise.groupby(
        "task", sort=False, group_keys=False
    )["corrected_p_value_raw"].apply(_holm_adjust)
    pairwise["comparison_family"] = f"{len(baselines)} BKAN-vs-baseline contrasts"
    primary = pairwise[pairwise["baseline"].eq(PRIMARY_BASELINE)].copy()
    primary["selection_rule"] = (
        "architecture-independent Spline-Ridge reference used consistently across tasks"
    )
    return pairwise, primary.reset_index(drop=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, model), group in metrics.groupby(["task", "model"], sort=False):
        values = group["rmse_target"].astype(float)
        rows.append(
            {
                "task": task,
                "model": model,
                "n_seeds": int(group["seed"].nunique()),
                "rmse_mean": float(values.mean()),
                "rmse_sample_std": float(values.std(ddof=1)),
                "rmse_median": float(values.median()),
                "r2_mean": float(group["r2_target"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    """Plot grouped-test RMSE for every task in the active model set."""

    tasks = list(dict.fromkeys(summary["task"]))
    cols = 2
    rows = math.ceil(len(tasks) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.2 * cols, 4.2 * rows), squeeze=False
    )
    color_map = plt.get_cmap("tab20")
    for ax, task_name in zip(axes.flat, tasks):
        part = summary.loc[summary["task"].eq(task_name)].copy()
        labels = [MODEL_LABEL.get(model, model) for model in part["model"]]
        colors = [
            color_map(EXPANDED_MODELS.index(model) % color_map.N)
            for model in part["model"]
        ]
        x = np.arange(len(part))
        ax.bar(
            x,
            part["rmse_mean"],
            yerr=part["rmse_sample_std"].fillna(0.0),
            color=colors,
            capsize=3,
        )
        ax.set_xticks(x, labels, rotation=38, ha="right", fontsize=8)
        ax.set_ylabel("Grouped-test RMSE")
        ax.set_title(task_name.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.flat[len(tasks) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "matched_grouped_rmse.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "matched_grouped_rmse.pdf", bbox_inches="tight")
    plt.close(fig)


def _report(args, summary: pd.DataFrame, primary: pd.DataFrame) -> None:
    exploratory = [
        MODEL_LABEL[model]
        for model in ("gmls", "autopinn", "curve_lut", "semiempirical")
        if model in args.models
    ]
    lines = [
        "# Ten-Split Matched Grouped Comparison",
        "",
        "All models use the same grouped splits; "
        "validation, calibration, and test condition groups are disjoint. Negative paired "
        "delta means lower RMSE for BKAN.",
        "",
        "| Task | BKAN RMSE | Prespecified reference | Reference RMSE | Paired delta | Corrected 95% CI | Holm p | Median delta | W/T/L |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary.to_dict(orient="records"):
        lines.append(
            f"| {row['task']} | {row['bkan_rmse_mean']:.6g} | {MODEL_LABEL[row['baseline']]} | "
            f"{row['baseline_rmse_mean']:.6g} | {row['paired_delta_mean']:.6g} | "
            f"[{row['corrected_repeated_split_ci95_low']:.6g}, {row['corrected_repeated_split_ci95_high']:.6g}] | "
            f"{row['corrected_p_value_holm']:.4g} | {row['paired_delta_median']:.6g} | "
            f"{int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Seeds: `{', '.join(map(str, args.seeds))}`",
            "- Models: `" + ", ".join(MODEL_LABEL[model] for model in args.models) + "`",
            "- Split fractions (train/validation/calibration/test): "
            f"`{'/'.join(f'{value:.0%}' for value in args.split_fractions)}`",
            "- Hyperparameters are frozen in `config.json`; no test-set tuning is performed.",
            "- " + ", ".join(exploratory)
            + " are post hoc adaptations or implementations and are reported as exploratory "
            "comparisons, not source-code reproductions or new confirmatory hypotheses.",
            "- BKAN point predictions come from the current workflow's BKAN training stage "
            "after manifest and target-order verification.",
            "- Win/tie/loss uses paired RMSE with relative tolerance `1e-6 * scale`.",
            "- The main contrast uses the same Spline-Ridge reference for every task.",
            "- Corrected intervals use the repeated-split variance factor "
            "`1/r + test_fraction/train_fraction`; Holm correction covers all "
            "BKAN-vs-baseline contrasts within each task.",
            "- Naive split bootstrap intervals are retained only as descriptive audit columns "
            "in `paired_statistics.csv` and are not used for confirmatory claims.",
            "",
            "Full model-wise summaries are in `metrics_summary.csv`; every seed/model test "
            "prediction is under `predictions/`.",
            "",
        ]
    )
    (args.output / "matched_grouped_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    audit_frames = []
    for task_name in args.tasks:
        shared_spec = shared.TASKS[TASK_KEY[task_name]]
        task_frame, spec = _load_task(task_name, args)
        for seed in args.seeds:
            partitions = shared.split_task_dataframe(
                task_frame, shared_spec, seed, tuple(args.split_fractions)
            )
            train, validation, calibration, test = partitions
            manifest = _group_manifest(partitions, shared_spec, seed)
            _verify_source_manifest(manifest, _bkan_manifest_path(task_name, seed, args))
            audit_frames.append(manifest)
            arrays = make_scaled_arrays(train, test, spec)
            for model in args.models:
                output = _prediction_output(args, seed, task_name, model)
                start = time.perf_counter()
                info = {}
                if args.resume and output.is_file():
                    prediction = _load_existing_prediction(output, test, spec)
                    info["reused"] = True
                elif args.reuse_root is not None and (
                    reused_output := (
                        args.reuse_root
                        / "predictions"
                        / f"seed_{seed}"
                        / task_name
                        / model
                        / "test_predictions.csv"
                    )
                ).is_file():
                    prediction = _load_existing_prediction(reused_output, test, spec)
                    info["reused"] = True
                    info["reuse_source"] = str(reused_output)
                    _write_prediction(output, test, spec, seed, model, prediction)
                elif model == "bkan":
                    prediction = _load_frozen_bkan(test, spec, task_name, seed, args)
                    info["source_stage"] = "current_workflow_bkan_training"
                    _write_prediction(output, test, spec, seed, model, prediction)
                elif model == "dkan":
                    prediction, info = train_dkan(
                        arrays, seed, device, args.kan_steps, grid=8, k=3, width=8
                    )
                    _write_prediction(output, test, spec, seed, model, prediction)
                elif model in MLP_ARCHES:
                    prediction, info = train_mlp(
                        arrays,
                        MLP_ARCHES[model],
                        seed,
                        device,
                        args.mlp_epochs,
                        args.mlp_lr,
                        args.mlp_lamb,
                        args.mlp_lamb_l1,
                        args.mlp_lamb_entropy,
                    )
                    _write_prediction(output, test, spec, seed, model, prediction)
                elif model in ENGINEERING_MODELS:
                    x_train = train[list(spec.input_cols)].to_numpy(dtype=np.float64)
                    x_test = test[list(spec.input_cols)].to_numpy(dtype=np.float64)
                    y_train = transformed_target(train, spec)
                    fitted = build_engineering_model(model, len(train), len(spec.input_cols), seed)
                    fitted.fit(x_train, y_train)
                    prediction = fitted.predict(x_test).ravel()
                    info["param_count"] = count_engineering_params(
                        model, fitted, len(train), len(spec.input_cols)
                    )
                    _write_prediction(output, test, spec, seed, model, prediction)
                elif model == "gmls":
                    prediction, info = fit_predict_gmls(train, validation, test, spec)
                    _write_prediction(output, test, spec, seed, model, prediction)
                elif model == "curve_lut":
                    prediction, info = fit_predict_curve_lut(
                        train,
                        validation,
                        test,
                        spec,
                        axis_col=AUTOPINN_CONSTRAINTS[task_name][0],
                    )
                    _write_prediction(output, test, spec, seed, model, prediction)
                elif model == "semiempirical":
                    prediction, info = fit_predict_semiempirical_compact(
                        train,
                        validation,
                        test,
                        spec,
                        axis_col=AUTOPINN_CONSTRAINTS[task_name][0],
                    )
                    _write_prediction(output, test, spec, seed, model, prediction)
                elif model == "autopinn":
                    axis_col, monotonic_sign = AUTOPINN_CONSTRAINTS[task_name]
                    prediction, info = train_autopinn_adapted(
                        train,
                        validation,
                        test,
                        spec,
                        axis_col=axis_col,
                        monotonic_sign=monotonic_sign,
                        seed=seed,
                        device=device,
                        epochs=args.autopinn_epochs,
                    )
                    _write_prediction(output, test, spec, seed, model, prediction)
                else:
                    raise ValueError(model)
                values = compute_metrics(
                    arrays["y_test_transformed"], arrays["y_test_raw"], prediction, spec
                )
                metric_rows.append(
                    {
                        "task": spec.name,
                        "seed": seed,
                        "model": model,
                        **values,
                        **info,
                        "wall_time_s": time.perf_counter() - start,
                        "train_groups": manifest.loc[manifest["split"] == "train", "group_id"].nunique(),
                        "validation_groups": manifest.loc[manifest["split"] == "validation", "group_id"].nunique(),
                        "calibration_groups": manifest.loc[manifest["split"] == "calibration", "group_id"].nunique(),
                        "test_groups": manifest.loc[manifest["split"] == "test", "group_id"].nunique(),
                    }
                )
                print(f"[{task_name} seed={seed} {model}] RMSE={values['rmse_target']:.6g}")

    metrics = pd.DataFrame(metric_rows)
    expected = len(args.tasks) * len(args.seeds) * len(args.models)
    if len(metrics) != expected:
        raise RuntimeError(f"Expected {expected} metric rows, found {len(metrics)}")
    summary = summarize_metrics(metrics)
    pairwise, primary = paired_statistics(
        metrics,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        train_fraction=float(args.split_fractions[0]),
        test_fraction=float(args.split_fractions[3]),
        models=tuple(args.models),
    )
    metrics.to_csv(args.output / "metrics_by_seed.csv", index=False)
    summary.to_csv(args.output / "metrics_summary.csv", index=False)
    pairwise.to_csv(args.output / "paired_statistics.csv", index=False)
    primary.to_csv(args.output / "prespecified_primary_comparisons.csv", index=False)
    pd.concat(audit_frames, ignore_index=True).to_csv(
        args.output / "split_manifest.csv", index=False
    )
    plot_summary(summary, args.output)
    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol": (
            "compact_framework_eight_model_exploratory_extension_v1"
            if tuple(args.models) == COMPACT_EIGHT_MODELS
            else "compact_framework_seven_model_curve_lut_exploratory_v1"
            if "curve_lut" in args.models and "semiempirical" not in args.models
            else "compact_framework_compact_baselines_exploratory_v1"
            if {"curve_lut", "semiempirical"} & set(args.models)
            else "compact_framework_core_exploratory_adaptations_v1"
            if {"gmls", "autopinn"} & set(args.models)
            else "unified_fresh_10_seed_matched_grouped"
        ),
        "data": str(args.data),
        "capacitance_data": str(args.capacitance_data),
        "bkan_root": str(args.bkan_root),
        "capacitance_bkan_root": str(args.capacitance_bkan_root),
        "reuse_root": str(args.reuse_root) if args.reuse_root is not None else None,
        "tasks": args.tasks,
        "models": list(args.models),
        "seeds": args.seeds,
        "split_fractions": args.split_fractions,
        "frozen_hyperparameters": {
            "dkan": {"steps": args.kan_steps, "grid": 8, "k": 3, "width": 8, "lr": 0.002},
            "mlp_l": {
                "hidden": list(MLP_ARCHES["mlp_l"]),
                "epochs": args.mlp_epochs,
                "lr": args.mlp_lr,
                "lamb": args.mlp_lamb,
                "lamb_l1": args.mlp_lamb_l1,
                "lamb_entropy": args.mlp_lamb_entropy,
            },
            "mlp_pm": {
                "hidden": list(MLP_ARCHES["mlp_pm"]),
                "epochs": args.mlp_epochs,
                "lr": args.mlp_lr,
                "lamb": args.mlp_lamb,
                "lamb_l1": args.mlp_lamb_l1,
                "lamb_entropy": args.mlp_lamb_entropy,
            },
            "poly3_ridge": {"degree": 3, "alpha": 1e-6},
            "spline_ridge": {"n_knots": 6, "degree": 3, "alpha": 1e-5},
            "rbf_nystroem": {
                "components": 256,
                "gamma": "1 / n_features",
                "alpha": 1e-4,
            },
            "random_forest": {
                "n_estimators": 500,
                "max_features": 1.0,
                "min_samples_leaf": 1,
                "target_standardization": "training only",
            },
            "xgboost": {
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "target_standardization": "training only",
            },
            "gmls": {
                "variant": "validation-selected adapted Gaussian-weighted local polynomial",
                "candidate_degree_neighbor_factor": [[1, 4], [2, 4], [2, 8]],
                "neighbors": "min(n_train, max(48, factor * n_basis_terms))",
                "ridge": 1e-4,
                "prediction_clip": "training target range plus 10% margin",
                "input_and_target_standardization": "training only",
                "selection": "lowest validation-group MSE; test not used",
            },
            "autopinn": {
                "variant": "adapted smooth/monotone architecture-search network",
                "candidates": [[16, 16], [32, 16], [32, 32]],
                "epochs": args.autopinn_epochs,
                "activation": "tanh",
                "smoothness_weight": 1e-5,
                "monotonic_weight": 1e-2,
                "constraints": AUTOPINN_CONSTRAINTS,
                "selection": "lowest validation-group MSE; test not used",
            },
            "curve_lut": {
                "variant": "curve-object PCHIP lookup table with condition-space IDW",
                "candidate_neighbor_curves_and_power": [[1, 1], [2, 1], [4, 1], [4, 2], [8, 2]],
                "axis_extrapolation": "nearest characterized boundary hold",
                "condition_standardization": "training only",
                "selection": "lowest validation-group MSE; test not used",
            },
            "semiempirical": {
                "variant": "ridge-calibrated task-specific compact feature library",
                "candidate_alpha": [1e-6, 1e-4, 1e-2, 1, 100],
                "features": "bias/axis shape, avalanche hinges, relaxation terms, and condition interactions",
                "input_feature_and_target_standardization": "training only",
                "selection": "lowest validation-group MSE; test not used",
            },
            "bkan": "loaded from the current workflow's freshly trained BKAN stage",
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _report(args, summary, primary)


def summarize_existing(args: argparse.Namespace) -> None:
    metrics_path = args.output / "metrics_by_seed.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    summary = summarize_metrics(metrics)
    pairwise, primary = paired_statistics(
        metrics,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        train_fraction=float(args.split_fractions[0]),
        test_fraction=float(args.split_fractions[3]),
        models=tuple(args.models),
    )
    summary.to_csv(args.output / "metrics_summary.csv", index=False)
    pairwise.to_csv(args.output / "paired_statistics.csv", index=False)
    primary.to_csv(args.output / "prespecified_primary_comparisons.csv", index=False)
    plot_summary(summary, args.output)
    _report(args, summary, primary)


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.summary_only:
        summarize_existing(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
