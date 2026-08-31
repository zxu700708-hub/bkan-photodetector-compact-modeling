"""Train the matched second-device APD comparison.

The experiment uses complete physical structures as the independent split
unit.  It evaluates only paired DC-current targets, multiplication behavior,
and two scalar voltage figures of merit.  It does not consume AC, C--V,
terminal-charge, spatial-charge, or noise data.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
PAPER_EXPERIMENTS = BKAN_ROOT / "simulation" / "paper_experiments"
for path in (BKAN_ROOT, PAPER_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from device_modeling.apd.data import (  # noqa: E402
    APD_TASKS,
    DEFAULT_APD_ROOT,
    APDTask,
    load_apd_task,
    training_tables_dir,
)
from device_modeling.photodetector import task_config as shared  # noqa: E402
from device_modeling.photodetector.bayesian_modeler import (  # noqa: E402
    BayesKANDeviceModeler,
)
from device_modeling.photodetector.common import set_seed  # noqa: E402
from p0_3_traditional_comparison import (  # noqa: E402
    MLP_ARCHES,
    TaskSpec as BaselineTaskSpec,
    build_engineering_model,
    compute_metrics,
    count_engineering_params,
    make_scaled_arrays,
    train_dkan,
    train_mlp,
    transformed_target,
)


DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "apd_second_device"
DEFAULT_TASKS = tuple(APD_TASKS)
DEFAULT_MODELS = ("bkan", "dkan", "mlp_l", "poly3_ridge", "spline_ridge")
MODEL_LABELS = {
    "bkan": "BKAN-VI",
    "dkan": "DKAN",
    "mlp_l": "MLP-L",
    "poly3_ridge": "Poly3-Ridge",
    "spline_ridge": "Spline-Ridge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_APD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=list(APD_TASKS), default=list(DEFAULT_TASKS))
    parser.add_argument("--models", nargs="+", choices=list(DEFAULT_MODELS), default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument(
        "--split-fractions",
        nargs=4,
        type=float,
        default=[0.65, 0.10, 0.15, 0.10],
        metavar=("TRAIN", "VALIDATION", "CALIBRATION", "TEST"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--train-mc-samples", type=int, default=3)
    parser.add_argument("--validation-mc-samples", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--bootstrap-replicates", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    training_tables_dir(args.data_root)
    shared.validate_split_fractions(args.split_fractions)
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be unique")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if args.epochs < 1 or args.kan_steps < 1 or args.mlp_epochs < 1:
        raise ValueError("Training iteration counts must be positive")
    if args.bootstrap_replicates < 1000:
        raise ValueError("At least 1000 bootstrap replicates are required")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def shared_spec(task: APDTask) -> shared.TaskSpec:
    return shared.TaskSpec(
        key=task.key,
        name=task.name,
        result_subdir=task.key,
        target_col=task.target_col,
        axis_col=task.axis_col,
        input_candidates=task.input_cols,
        use_log_transform=task.use_log_transform,
        y_bounds=task.y_bounds,
    )


def baseline_spec(task: APDTask) -> BaselineTaskSpec:
    return BaselineTaskSpec(
        name=task.name,
        target_col=task.target_col,
        input_cols=task.input_cols,
        use_log_transform=task.use_log_transform,
        y_bounds=task.y_bounds,
        ylabel=task.ylabel,
        group_cols=("structure_group_id",),
    )


def split_task(frame: pd.DataFrame, task: APDTask, seed: int, fractions):
    spec = shared_spec(task)
    partitions = shared.split_task_dataframe(frame, spec, seed, fractions)
    rows = []
    for split_name, part in zip(
        ("train", "validation", "calibration", "test"), partitions
    ):
        for group_id, count in part.groupby("structure_group_id", sort=True).size().items():
            rows.append(
                {
                    "task": task.name,
                    "seed": seed,
                    "split": split_name,
                    "group_id": str(group_id),
                    "n_points": int(count),
                }
            )
    manifest = pd.DataFrame(rows)
    overlap = manifest.groupby("group_id")["split"].nunique()
    if (overlap != 1).any():
        raise RuntimeError(f"Structure-level split leakage for {task.name}, seed={seed}")
    if manifest["group_id"].nunique() != 78:
        raise RuntimeError(f"Incomplete structure split for {task.name}, seed={seed}")
    return partitions, manifest


def prediction_path(output: Path, task: APDTask, seed: int, model: str) -> Path:
    return output / "predictions" / f"seed_{seed}" / task.key / model / "test_predictions.csv"


def metadata_path(output: Path, task: APDTask, seed: int, model: str) -> Path:
    return prediction_path(output, task, seed, model).with_name("run_metadata.json")


def write_prediction(
    path: Path,
    test: pd.DataFrame,
    task: APDTask,
    seed: int,
    model: str,
    prediction: np.ndarray,
    prediction_std: np.ndarray | None = None,
) -> None:
    spec = baseline_spec(task)
    result = test.copy()
    result.insert(0, "model", model)
    result.insert(0, "seed", seed)
    result.insert(0, "task", task.name)
    result["actual_model_space"] = transformed_target(test, spec)
    result["prediction_model_space"] = np.asarray(prediction, dtype=np.float64)
    if prediction_std is not None:
        result["prediction_std_model_space"] = np.asarray(
            prediction_std, dtype=np.float64
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)


def load_prediction(path: Path, test: pd.DataFrame, task: APDTask):
    frame = pd.read_csv(path)
    expected = transformed_target(test, baseline_spec(task))
    actual = frame["actual_model_space"].to_numpy(dtype=np.float64)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-7, atol=1e-10):
        raise RuntimeError(f"Stale prediction does not match regenerated split: {path}")
    pred = frame["prediction_model_space"].to_numpy(dtype=np.float64)
    return pred


def train_bkan(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    task: APDTask,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    model_dir: Path,
):
    set_seed(seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    modeler = BayesKANDeviceModeler(
        task_name=task.name,
        results_dir=str(model_dir),
        device=str(device),
    )
    modeler.load_data(
        train,
        input_cols=list(task.input_cols),
        output_col=task.target_col,
        use_log_transform=task.use_log_transform,
        y_bounds=task.y_bounds,
    )
    modeler.build_model(
        {
            "width": [None, 8, 2],
            "seed": seed,
            "grid": 8,
            "k": 3,
            "grid_range": [-3, 3],
            "kl_weight": args.kl_weight,
            "num_mc_samples": args.mc_samples,
            "prior_mu": 0.0,
            "prior_log_sigma": 0.0,
            "posterior_init_sigma": 0.1,
            "likelihood": "gaussian",
            "inference_method": "vi",
            "prediction_seed": 1729 + seed,
        }
    )
    log_path = model_dir / "training.log"
    start = time.perf_counter()
    with (
        log_path.open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        modeler.train(
            {
                "num_epochs": args.epochs,
                "batch_size": min(args.batch_size, len(train)),
                "lr": args.learning_rate,
                "weight_decay": 1.0e-5,
                "val_freq": 10,
                "train_mc_samples": args.train_mc_samples,
                "validation_mc_samples": args.validation_mc_samples,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "validation_seed": 1729 + seed,
            },
            df_val=validation,
        )
    elapsed = time.perf_counter() - start
    prediction = modeler.predict_with_uncertainty(
        test[list(task.input_cols)].to_numpy(dtype=np.float32)
    )
    checkpoint = model_dir / "model_checkpoint.pt"
    modeler.save_model(str(checkpoint))
    history = modeler.experiment.history
    info = {
        "param_count": int(sum(p.numel() for p in modeler.model.parameters() if p.requires_grad)),
        "train_time_s": elapsed,
        "epochs_completed": len(history.get("train_loss", [])),
        "best_val_loss": float(history.get("best_val_loss", np.nan)),
        "checkpoint": str(checkpoint),
    }
    return (
        np.asarray(prediction["model_mean"], dtype=np.float64),
        np.asarray(prediction["model_std"], dtype=np.float64),
        info,
    )


def model_metrics(
    test: pd.DataFrame,
    task: APDTask,
    prediction: np.ndarray,
) -> dict[str, float]:
    spec = baseline_spec(task)
    y_transformed = transformed_target(test, spec)
    metrics = compute_metrics(
        y_transformed,
        test[task.target_col].to_numpy(dtype=np.float64),
        prediction,
        spec,
    )
    work = pd.DataFrame(
        {
            "structure_group_id": test["structure_group_id"].to_numpy(),
            "squared_error": (np.asarray(prediction) - y_transformed) ** 2,
            "absolute_error": np.abs(np.asarray(prediction) - y_transformed),
        }
    )
    group_rmse = np.sqrt(work.groupby("structure_group_id")["squared_error"].mean())
    group_mae = work.groupby("structure_group_id")["absolute_error"].mean()
    metrics["group_macro_rmse_target"] = float(group_rmse.mean())
    metrics["group_macro_mae_target"] = float(group_mae.mean())
    return metrics


def _holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (len(values) - position))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_statistics(metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(args.bootstrap_seed)
    train_fraction = float(args.split_fractions[0])
    test_fraction = float(args.split_fractions[3])
    for task_name, task_metrics in metrics.groupby("task", sort=False):
        pivot = task_metrics.pivot(index="seed", columns="model", values="rmse_target")
        if "bkan" not in pivot:
            continue
        for baseline in [m for m in args.models if m != "bkan" and m in pivot]:
            values = (pivot["bkan"] - pivot[baseline]).dropna().to_numpy(dtype=np.float64)
            if len(values) < 2:
                continue
            sample_indices = rng.integers(
                0, len(values), size=(args.bootstrap_replicates, len(values))
            )
            bootstrap = values[sample_indices].mean(axis=1)
            variance = float(values.var(ddof=1))
            corrected_se = math.sqrt((1.0 / len(values) + test_fraction / train_fraction) * variance)
            if corrected_se > 0.0:
                critical = float(student_t.ppf(0.975, df=len(values) - 1))
                low = float(values.mean() - critical * corrected_se)
                high = float(values.mean() + critical * corrected_se)
                raw_p = float(2.0 * student_t.sf(abs(values.mean() / corrected_se), df=len(values) - 1))
            else:
                low = high = float(values.mean())
                raw_p = 0.0 if values.mean() != 0.0 else 1.0
            rows.append(
                {
                    "task": task_name,
                    "baseline": baseline,
                    "n_pairs": len(values),
                    "bkan_rmse_mean": float(pivot["bkan"].mean()),
                    "baseline_rmse_mean": float(pivot[baseline].mean()),
                    "paired_delta_mean": float(values.mean()),
                    "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "corrected_ci95_low": low,
                    "corrected_ci95_high": high,
                    "corrected_p_value_raw": raw_p,
                    "wins": int(np.sum(values < 0.0)),
                    "losses": int(np.sum(values > 0.0)),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["corrected_p_value_holm"] = np.nan
    for _, indices in result.groupby("task", sort=False).groups.items():
        result.loc[indices, "corrected_p_value_holm"] = _holm(
            result.loc[indices, "corrected_p_value_raw"].to_numpy(dtype=np.float64)
        )
    return result


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "rmse_target",
        "mae_target",
        "r2_target",
        "rmse_raw",
        "mae_raw",
        "medape_percent",
        "p95ape_percent",
        "group_macro_rmse_target",
        "group_macro_mae_target",
        "wall_time_s",
        "param_count",
    ]
    summary = metrics.groupby(["task", "model"], sort=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    return summary.reset_index()


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    tasks = list(dict.fromkeys(summary["task"]))
    cols = 3
    rows = math.ceil(len(tasks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.7 * rows), squeeze=False)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    for ax, task_name in zip(axes.flat, tasks):
        part = summary.loc[summary["task"].eq(task_name)].copy()
        labels = [MODEL_LABELS.get(model, model) for model in part["model"]]
        x = np.arange(len(part))
        ax.bar(
            x,
            part["rmse_target_mean"],
            yerr=part["rmse_target_std"].fillna(0.0),
            color=colors[: len(part)],
            capsize=3,
        )
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("Grouped-test RMSE (model space)")
        ax.set_title(task_name.replace("APD_", "APD "))
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.flat[len(tasks) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "apd_second_device_rmse.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "apd_second_device_rmse.pdf", bbox_inches="tight")
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(summary: pd.DataFrame, paired: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# APD Second-Device Matched Grouped Comparison",
        "",
        "The independent split unit is `structure_group_id`. This is a device-specific retraining experiment, not zero-shot transfer.",
        "",
        "## Mean Test Metrics",
        "",
        "| Task | Model | RMSE | MAE | R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['task']} | {MODEL_LABELS.get(row['model'], row['model'])} | "
            f"{row['rmse_target_mean']:.6g} ± {row['rmse_target_std']:.3g} | "
            f"{row['mae_target_mean']:.6g} | {row['r2_target_mean']:.6g} |"
        )
    if not paired.empty:
        lines.extend(
            [
                "",
                "## Prespecified BKAN-VI vs Spline-Ridge Contrast",
                "",
                "| Task | BKAN RMSE | Spline RMSE | Delta | Corrected 95% CI | Wins/Losses |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in paired.loc[paired["baseline"].eq("spline_ridge")].iterrows():
            lines.append(
                f"| {row['task']} | {row['bkan_rmse_mean']:.6g} | "
                f"{row['baseline_rmse_mean']:.6g} | {row['paired_delta_mean']:.6g} | "
                f"[{row['corrected_ci95_low']:.6g}, {row['corrected_ci95_high']:.6g}] | "
                f"{int(row['wins'])}/{int(row['losses'])} |"
            )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Included: dark current, illuminated current, net photocurrent, multiplication gain, gain-threshold voltage, and breakdown voltage.",
            "- Excluded: AC response, C--V, terminal charge, spatial charge, and noise.",
            "- The evidence supports second-device-class retraining within the same TCAD source, not foundry, measurement, or zero-shot transfer.",
        ]
    )
    (args.output / "apd_second_device_report.md").write_text("\n".join(lines), encoding="utf-8")


def collect_existing(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    rows = []
    manifests = []
    for task_key in args.tasks:
        frame, task = load_apd_task(task_key, args.data_root)
        for seed in args.seeds:
            partitions, manifest = split_task(
                frame, task, seed, tuple(args.split_fractions)
            )
            test = partitions[-1]
            manifests.append(manifest)
            for model in args.models:
                out_path = prediction_path(args.output, task, seed, model)
                meta_path = metadata_path(args.output, task, seed, model)
                if not out_path.is_file() or not meta_path.is_file():
                    raise FileNotFoundError(
                        out_path if not out_path.is_file() else meta_path
                    )
                load_prediction(out_path, test, task)
                row = json.loads(meta_path.read_text(encoding="utf-8"))
                expected = {
                    "task": task.name,
                    "task_key": task.key,
                    "seed": seed,
                    "model": model,
                    "test_rows": len(test),
                }
                observed = {key: row.get(key) for key in expected}
                if observed != expected:
                    raise ValueError(
                        f"Metadata mismatch in {meta_path}: expected {expected}, "
                        f"observed {observed}"
                    )
                rows.append(row)
    return pd.DataFrame(rows), manifests


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    runtime_device = resolve_device(args.device)
    if args.summary_only:
        metrics, manifests = collect_existing(args)
    else:
        device = runtime_device
        print(f"Training on {device}")
        metric_rows = []
        manifests = []
        for task_key in args.tasks:
            frame, task = load_apd_task(task_key, args.data_root)
            spec = baseline_spec(task)
            for seed in args.seeds:
                partitions, manifest = split_task(
                    frame, task, seed, tuple(args.split_fractions)
                )
                train, validation, calibration, test = partitions
                manifests.append(manifest)
                arrays = make_scaled_arrays(train, test, spec)
                for model in args.models:
                    out_path = prediction_path(args.output, task, seed, model)
                    meta_path = metadata_path(args.output, task, seed, model)
                    if args.resume and out_path.is_file() and meta_path.is_file():
                        prediction = load_prediction(out_path, test, task)
                        info = json.loads(meta_path.read_text(encoding="utf-8"))
                        std = None
                        reused = True
                    else:
                        started = time.perf_counter()
                        std = None
                        if model == "bkan":
                            prediction, std, info = train_bkan(
                                train,
                                validation,
                                test,
                                task,
                                seed,
                                device,
                                args,
                                out_path.parent,
                            )
                        elif model == "dkan":
                            log_path = out_path.parent / "training.log"
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            with (
                                log_path.open("w", encoding="utf-8") as log,
                                contextlib.redirect_stdout(log),
                                contextlib.redirect_stderr(log),
                            ):
                                prediction, info = train_dkan(
                                    arrays,
                                    seed,
                                    device,
                                    args.kan_steps,
                                    grid=8,
                                    k=3,
                                    width=8,
                                )
                        elif model == "mlp_l":
                            prediction, info = train_mlp(
                                arrays,
                                MLP_ARCHES["mlp_l"],
                                seed,
                                device,
                                args.mlp_epochs,
                                0.002,
                                0.001,
                                1.0,
                                2.0,
                            )
                        elif model in {"poly3_ridge", "spline_ridge"}:
                            fitted = build_engineering_model(
                                model, len(train), len(task.input_cols), seed
                            )
                            x_train = train[list(task.input_cols)].to_numpy(dtype=np.float64)
                            x_test = test[list(task.input_cols)].to_numpy(dtype=np.float64)
                            fitted.fit(x_train, transformed_target(train, spec))
                            prediction = fitted.predict(x_test).ravel()
                            info = {
                                "param_count": count_engineering_params(
                                    model, fitted, len(train), len(task.input_cols)
                                )
                            }
                        else:
                            raise ValueError(model)
                        info["wall_time_s"] = time.perf_counter() - started
                        write_prediction(out_path, test, task, seed, model, prediction, std)
                        reused = False

                    values = model_metrics(test, task, prediction)
                    row = {
                        "task": task.name,
                        "task_key": task.key,
                        "primary_task": task.primary,
                        "seed": seed,
                        "model": model,
                        **values,
                        **{
                            key: value
                            for key, value in info.items()
                            if key not in {"task", "task_key", "seed", "model"}
                        },
                        "wall_time_s": float(info.get("wall_time_s", info.get("train_time_s", 0.0))),
                        "reused": reused,
                        "train_groups": int(manifest.loc[manifest.split.eq("train"), "group_id"].nunique()),
                        "validation_groups": int(manifest.loc[manifest.split.eq("validation"), "group_id"].nunique()),
                        "calibration_groups": int(manifest.loc[manifest.split.eq("calibration"), "group_id"].nunique()),
                        "test_groups": int(manifest.loc[manifest.split.eq("test"), "group_id"].nunique()),
                        "train_rows": len(train),
                        "test_rows": len(test),
                    }
                    meta_path.write_text(
                        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    metric_rows.append(row)
                    print(
                        f"[{task.key} seed={seed} {model}] "
                        f"RMSE={values['rmse_target']:.6g} R2={values['r2_target']:.6g}"
                    )
        metrics = pd.DataFrame(metric_rows)

    expected_rows = len(args.tasks) * len(args.seeds) * len(args.models)
    if len(metrics) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} matched metric rows, observed {len(metrics)}"
        )

    summary = summarize(metrics)
    paired = paired_statistics(metrics, args)
    metrics.to_csv(args.output / "metrics_by_seed.csv", index=False)
    summary.to_csv(args.output / "metrics_summary.csv", index=False)
    paired.to_csv(args.output / "paired_statistics.csv", index=False)
    if manifests:
        pd.concat(manifests, ignore_index=True).to_csv(
            args.output / "split_manifest.csv", index=False
        )
    plot_summary(summary, args.output)
    write_report(summary, paired, args)

    table_dir = training_tables_dir(args.data_root)
    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol": "apd_second_device_structure_grouped_retraining_v1",
        "data_root": str(args.data_root),
        "output": str(args.output),
        "tasks": args.tasks,
        "models": args.models,
        "seeds": args.seeds,
        "split_fractions": args.split_fractions,
        "independent_unit": "structure_group_id",
        "device": str(runtime_device),
        "gpu_name": (
            torch.cuda.get_device_name(runtime_device)
            if runtime_device.type == "cuda"
            else None
        ),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "task_definitions": {key: asdict(APD_TASKS[key]) for key in args.tasks},
        "training": {
            "bkan": {
                "width": 8,
                "grid": 8,
                "spline_order": 3,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "kl_weight": args.kl_weight,
                "train_mc_samples": args.train_mc_samples,
                "validation_mc_samples": args.validation_mc_samples,
                "prediction_mc_samples": args.mc_samples,
            },
            "dkan": {"width": 8, "grid": 8, "spline_order": 3, "steps": args.kan_steps},
            "mlp_l": {"hidden": list(MLP_ARCHES["mlp_l"]), "epochs": args.mlp_epochs},
            "poly3_ridge": {"degree": 3, "alpha": 1.0e-6},
            "spline_ridge": {"knots": 6, "degree": 3, "alpha": 1.0e-5},
        },
        "data_sha256": {
            name: file_sha256(table_dir / name)
            for name in ("apd_dc.csv", "apd_gain.csv", "apd_fom.csv")
        },
        "excluded_tasks": [
            "ac_response",
            "capacitance",
            "terminal_charge",
            "spatial_charge",
            "noise",
        ],
        "claim_boundary": {
            "device_specific_retraining": True,
            "zero_shot_transfer": False,
            "foundry_transfer": False,
            "measurement_validation": False,
        },
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
