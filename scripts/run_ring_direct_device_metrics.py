"""Direct device-metric model comparison for the active-microring dataset.

This is an isolated follow-up experiment.  It does not modify or overwrite the
full-spectrum experiment.  All methods reuse its frozen 80-structure repeated
split manifest, while the supervised rows are the 400 audited curve-level
metrics extracted from the high-resolution reference spectra.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
PAPER_EXPERIMENTS = BKAN_ROOT / "simulation" / "paper_experiments"
for path in (ROOT / "scripts", BKAN_ROOT, PAPER_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_framework_baselines import fit_predict_gmls, train_autopinn_adapted  # noqa: E402
from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler  # noqa: E402
from device_modeling.photodetector.common import set_seed  # noqa: E402
from experiment_ring_resonance_aware_dkan import fit_kan  # noqa: E402
from p0_3_traditional_comparison import (  # noqa: E402
    MLP_ARCHES,
    TaskSpec as BaselineTaskSpec,
    build_engineering_model,
    compute_metrics,
    count_engineering_params,
    make_scaled_arrays,
    train_mlp,
    transformed_target,
    transformed_to_raw,
)
from run_apd_second_device import paired_statistics  # noqa: E402
from run_ring_third_device import DEFAULT_DATA_ROOT, file_sha256, resolve_device  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "ring_direct_device_metrics"
INPUT_COLUMNS = ("core_width_nm", "coupling_gap_nm", "ring_radius_um", "bias_v")
MODELS = ("bkan", "dkan", "mlp_l", "spline_ridge", "gmls", "autopinn")
MODEL_LABELS = {
    "bkan": "BKAN-VI",
    "dkan": "DKAN",
    "mlp_l": "MLP-L",
    "spline_ridge": "Spline-Ridge",
    "gmls": "GMLS-adapted",
    "autopinn": "AutoPINN-adapted",
}


@dataclass(frozen=True)
class MetricTask:
    key: str
    name: str
    target_col: str
    ylabel: str
    use_log_transform: bool
    y_bounds: tuple[float, float]
    primary_raw_metric: str
    monotonic_sign: int | None = None
    input_cols: tuple[str, ...] = INPUT_COLUMNS
    axis_col: str = "bias_v"


TASKS = {
    "resonance_shift": MetricTask(
        "resonance_shift",
        "Ring_resonance_shift",
        "resonance_shift_pm",
        "resonance shift (pm)",
        False,
        (-1.0, 100.0),
        "nonzero_bias_rmse_raw",
        monotonic_sign=-1,
    ),
    "quality_factor": MetricTask(
        "quality_factor",
        "Ring_loaded_Q",
        "quality_factor",
        "loaded quality factor",
        True,
        (3.0, 6.0),
        "median_ape_percent",
    ),
    "drop_extinction": MetricTask(
        "drop_extinction",
        "Ring_drop_extinction_ratio",
        "drop_extinction_ratio_db",
        "drop-port extinction ratio (dB)",
        False,
        (0.0, 80.0),
        "mae_raw",
    ),
    "through_insertion": MetricTask(
        "through_insertion",
        "Ring_through_insertion_loss",
        "through_insertion_loss_db",
        "through-port insertion loss (dB)",
        True,
        (-8.0, 1.0),
        "median_ape_percent",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--autopinn-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--train-mc-samples", type=int, default=3)
    parser.add_argument("--validation-mc-samples", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    if args.smoke:
        args.output = args.output.with_name(args.output.name + "_smoke")
        args.seeds = [42]
        args.epochs = min(args.epochs, 20)
        args.kan_steps = min(args.kan_steps, 20)
        args.mlp_epochs = min(args.mlp_epochs, 20)
        args.autopinn_epochs = min(args.autopinn_epochs, 20)
        args.mc_samples = min(args.mc_samples, 10)
        args.validation_mc_samples = min(args.validation_mc_samples, 5)
        args.early_stopping_patience = min(args.early_stopping_patience, 2)
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    if len(args.models) != len(set(args.models)) or len(args.tasks) != len(set(args.tasks)):
        raise ValueError("Models and tasks must be unique")
    if min(args.epochs, args.kan_steps, args.mlp_epochs, args.autopinn_epochs) < 1:
        raise ValueError("Training budgets must be positive")
    if "autopinn" in args.models and args.autopinn_epochs < 10:
        raise ValueError("AutoPINN-adapted requires at least ten epochs")
    return args


def frame_fingerprint(frame: pd.DataFrame, task: MetricTask) -> str:
    columns = ["structure_id", *task.input_cols, task.target_col]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def baseline_spec(task: MetricTask) -> BaselineTaskSpec:
    return BaselineTaskSpec(
        name=task.name,
        target_col=task.target_col,
        input_cols=task.input_cols,
        use_log_transform=task.use_log_transform,
        y_bounds=task.y_bounds,
        ylabel=task.ylabel,
        group_cols=("structure_id",),
    )


def load_dataset(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metric_path = data_root / "training_tables" / "hifi_ring_derived_metrics.csv"
    ready = data_root / "training_ready"
    split_path = ready / "ring_repeated_group_splits.csv"
    manifest_path = ready / "ring_training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "ring_resonance_aware_grouped_training_v1":
        raise RuntimeError("Unexpected source training protocol")
    if file_sha256(split_path) != manifest["files"]["ring_repeated_group_splits.csv"]:
        raise RuntimeError("Frozen split manifest hash mismatch")
    frame = pd.read_csv(metric_path)
    required = {
        "structure_id", "bias_v", "metric_extraction_version", *INPUT_COLUMNS,
        "resonance_shift_nm", "quality_factor", "drop_extinction_ratio_db",
        "through_insertion_loss_db",
    }
    if not required.issubset(frame.columns):
        raise KeyError(f"Missing metric columns: {sorted(required - set(frame.columns))}")
    if len(frame) != 400 or frame["structure_id"].nunique() != 80:
        raise RuntimeError("Expected 400 curve rows from 80 independent structures")
    if frame.duplicated(["structure_id", "bias_v"]).any():
        raise RuntimeError("Duplicate structure-bias metric row")
    if not frame["metric_extraction_version"].eq("local_resonance_v1").all():
        raise RuntimeError("Unexpected source metric extractor")
    frame = frame.copy()
    frame["resonance_shift_pm"] = 1.0e3 * frame["resonance_shift_nm"].to_numpy(dtype=np.float64)
    numeric = [*INPUT_COLUMNS, *(task.target_col for task in TASKS.values())]
    if not np.isfinite(frame[list(dict.fromkeys(numeric))].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Non-finite direct metric input or target")
    splits = pd.read_csv(split_path)
    if len(splits) != 800 or splits.groupby(["seed", "structure_id"]).size().ne(1).any():
        raise RuntimeError("Incomplete repeated structure split manifest")
    provenance = {
        "metric_table": str(metric_path),
        "metric_table_sha256": file_sha256(metric_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": file_sha256(split_path),
        "source_manifest_sha256": file_sha256(manifest_path),
    }
    return frame, splits, provenance


def split_frame(
    frame: pd.DataFrame,
    splits: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment = splits.loc[
        splits["seed"].eq(seed), ["seed", "structure_id", "split"]
    ].copy()
    if len(assignment) != 80 or assignment["structure_id"].nunique() != 80:
        raise RuntimeError(f"Incomplete split for seed {seed}")
    # The source metric table carries a legacy one-off split label.  It is not
    # part of this experiment; only the frozen repeated split manifest is used.
    work = frame.drop(columns="split", errors="ignore").merge(
        assignment, on="structure_id", how="left", validate="many_to_one"
    )
    if work["split"].isna().any():
        raise RuntimeError(f"Unassigned structure for seed {seed}")
    parts = tuple(
        work.loc[work["split"].eq(name)].sort_values(["structure_id", "bias_v"]).reset_index(drop=True)
        for name in ("train", "validation", "calibration", "test")
    )
    groups = [set(part["structure_id"]) for part in parts]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
        raise RuntimeError(f"Structure leakage for seed {seed}")
    if tuple(len(group) for group in groups) != (52, 8, 12, 8):
        raise RuntimeError(f"Unexpected group counts for seed {seed}")
    return (*parts, assignment)


def prediction_path(output: Path, seed: int, task: MetricTask, model: str) -> Path:
    return output / "predictions" / f"seed_{seed}" / task.key / model / "test_predictions.csv"


def write_prediction(
    path: Path,
    test: pd.DataFrame,
    task: MetricTask,
    seed: int,
    model: str,
    prediction: np.ndarray,
    prediction_std: np.ndarray | None,
) -> None:
    spec = baseline_spec(task)
    result = test[["structure_id", *task.input_cols, task.target_col]].copy()
    result.insert(0, "model", model)
    result.insert(0, "seed", seed)
    result.insert(0, "task", task.name)
    result["actual_model_space"] = transformed_target(test, spec)
    result["prediction_model_space"] = np.asarray(prediction, dtype=np.float64)
    result["actual_raw"] = test[task.target_col].to_numpy(dtype=np.float64)
    result["prediction_raw"] = transformed_to_raw(np.asarray(prediction, dtype=np.float64), spec)
    if prediction_std is not None:
        result["prediction_std_model_space"] = np.asarray(prediction_std, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)


def load_prediction(path: Path, test: pd.DataFrame, task: MetricTask) -> tuple[np.ndarray, np.ndarray | None]:
    result = pd.read_csv(path)
    if len(result) != len(test) or not np.array_equal(
        result["structure_id"].astype(str).to_numpy(), test["structure_id"].astype(str).to_numpy()
    ):
        raise RuntimeError(f"Stale prediction structure order: {path}")
    if not np.allclose(
        result[list(task.input_cols)].to_numpy(dtype=np.float64),
        test[list(task.input_cols)].to_numpy(dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise RuntimeError(f"Stale prediction inputs: {path}")
    prediction = result["prediction_model_space"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Non-finite prediction: {path}")
    std = (
        result["prediction_std_model_space"].to_numpy(dtype=np.float64)
        if "prediction_std_model_space" in result
        else None
    )
    return prediction, std


def train_bkan(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    task: MetricTask,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    set_seed(seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    modeler = BayesKANDeviceModeler(task_name=task.name, results_dir=str(model_dir), device=str(device))
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
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        modeler.train(
            {
                "num_epochs": args.epochs,
                "batch_size": min(args.batch_size, len(train)),
                "lr": args.learning_rate,
                "weight_decay": 1.0e-5,
                "val_freq": min(10, args.epochs),
                "train_mc_samples": args.train_mc_samples,
                "validation_mc_samples": args.validation_mc_samples,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": 1.0e-4,
                "validation_seed": 1729 + seed,
            },
            df_val=validation,
        )
    result = modeler.predict_with_uncertainty(
        test[list(task.input_cols)].to_numpy(dtype=np.float32)
    )
    modeler.save_model(str(model_dir / "model_checkpoint.pt"))
    return (
        np.asarray(result["model_mean"], dtype=np.float64),
        np.asarray(result["model_std"], dtype=np.float64),
        {
            "param_count": int(sum(p.numel() for p in modeler.model.parameters() if p.requires_grad)),
            "train_time_s": float(time.perf_counter() - started),
            "architecture": (
                f"Bayesian scalar KAN [{len(task.input_cols)},8,2]; grid=8; k=3"
            ),
        },
    )


def train_one(
    model: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    task: MetricTask,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    spec = baseline_spec(task)
    if model == "bkan":
        return train_bkan(train, validation, test, task, seed, device, args, model_dir)
    if model == "dkan":
        y_train = transformed_target(train, spec).reshape(-1, 1)
        y_validation = transformed_target(validation, spec).reshape(-1, 1)
        fitted, info = fit_kan(
            train[list(task.input_cols)].to_numpy(dtype=np.float32),
            y_train,
            validation[list(task.input_cols)].to_numpy(dtype=np.float32),
            y_validation,
            seed=seed,
            device=device,
            steps=args.kan_steps,
            width=8,
            grid=8,
            log_path=model_dir / "training.log",
        )
        prediction = fitted.predict(test[list(task.input_cols)].to_numpy(dtype=np.float32))[:, 0]
        info["architecture"] = (
            f"deterministic scalar KAN [{len(task.input_cols)},8,1]; grid=8; k=3"
        )
        return prediction, None, info
    if model == "mlp_l":
        arrays = make_scaled_arrays(train, test, spec)
        prediction, info = train_mlp(
            arrays, MLP_ARCHES["mlp_l"], seed, device, args.mlp_epochs,
            0.002, 0.001, 1.0, 2.0,
        )
        return prediction, None, info
    if model == "spline_ridge":
        started = time.perf_counter()
        fitted = build_engineering_model(model, len(train), len(task.input_cols), seed)
        fitted.fit(
            train[list(task.input_cols)].to_numpy(dtype=np.float64),
            transformed_target(train, spec),
        )
        prediction = fitted.predict(test[list(task.input_cols)].to_numpy(dtype=np.float64)).reshape(-1)
        return prediction, None, {
            "param_count": count_engineering_params(model, fitted, len(train), len(task.input_cols)),
            "train_time_s": float(time.perf_counter() - started),
        }
    if model == "gmls":
        prediction, info = fit_predict_gmls(train, validation, test, spec)
        return prediction, None, info
    if model == "autopinn":
        prediction, info = train_autopinn_adapted(
            train,
            validation,
            test,
            spec,
            axis_col=task.axis_col,
            monotonic_sign=task.monotonic_sign,
            seed=seed,
            device=device,
            epochs=args.autopinn_epochs,
        )
        return prediction, None, info
    raise ValueError(model)


def task_metrics(test: pd.DataFrame, task: MetricTask, prediction: np.ndarray) -> dict[str, float]:
    spec = baseline_spec(task)
    actual_model = transformed_target(test, spec)
    prediction = np.asarray(prediction, dtype=np.float64)
    actual_raw = test[task.target_col].to_numpy(dtype=np.float64)
    prediction_raw = transformed_to_raw(prediction, spec)
    result = compute_metrics(actual_model, actual_raw, prediction, spec)
    result.update(
        {
            "rmse_db_or_raw": float(math.sqrt(mean_squared_error(actual_raw, prediction_raw))),
            "mae_raw": float(mean_absolute_error(actual_raw, prediction_raw)),
            "r2_raw": float(r2_score(actual_raw, prediction_raw)),
        }
    )
    nonzero = np.abs(actual_raw) > 1.0e-14
    result["nonzero_bias_rmse_raw"] = (
        float(math.sqrt(mean_squared_error(actual_raw[nonzero], prediction_raw[nonzero])))
        if nonzero.any()
        else float("nan")
    )
    ape = 100.0 * np.abs(prediction_raw[nonzero] - actual_raw[nonzero]) / np.abs(actual_raw[nonzero])
    result["median_ape_percent"] = float(np.median(ape)) if len(ape) else float("nan")
    group_rmse = []
    for _, indices in test.groupby("structure_id", sort=False).groups.items():
        idx = np.asarray(indices, dtype=int)
        group_rmse.append(math.sqrt(float(np.mean((prediction[idx] - actual_model[idx]) ** 2))))
    result["structure_macro_rmse_target"] = float(np.mean(group_rmse))
    return result


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "rmse_target", "mae_target", "r2_target", "rmse_raw", "mae_raw",
        "r2_raw", "medape_percent", "p95ape_percent", "median_ape_percent",
        "nonzero_bias_rmse_raw", "structure_macro_rmse_target", "wall_time_s", "param_count",
    ]
    result = metrics.groupby(["task_key", "task", "model"], sort=False)[numeric].agg(["mean", "std"])
    result.columns = [f"{name}_{stat}" for name, stat in result.columns]
    return result.reset_index()


def write_report(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Direct active-microring device-metric comparison",
        "",
        "All results use the frozen 80-structure repeated splits. BKAN-VI and DKAN share width 8, grid 8, and cubic splines; their output heads differ only as required for probabilistic versus deterministic prediction.",
        "",
        "| Task | Model | Model-space RMSE | Raw RMSE | Raw MAE | Median APE (%) | R2 raw |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.task_key} | {MODEL_LABELS[row.model]} | {row.rmse_target_mean:.6g} ± {row.rmse_target_std:.6g} | "
            f"{row.rmse_raw_mean:.6g} | {row.mae_raw_mean:.6g} | {row.median_ape_percent_mean:.6g} | {row.r2_raw_mean:.6g} |"
        )
    (output / "direct_metric_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def anchor_statistics(metrics: pd.DataFrame, args: argparse.Namespace, anchor: str) -> pd.DataFrame:
    """Reuse the matched-split routine with an arbitrary named anchor model."""
    if anchor not in set(metrics["model"]):
        raise ValueError(f"Missing anchor model {anchor}")
    work = metrics.copy()
    mapping = {anchor: "bkan"}
    if anchor != "bkan" and "bkan" in set(work["model"]):
        mapping["bkan"] = "bkan_method"
    work["model"] = work["model"].replace(mapping)
    models = [mapping.get(model, model) for model in args.models]
    paired_args = SimpleNamespace(
        models=models,
        bootstrap_seed=20260911,
        bootstrap_replicates=50000,
        split_fractions=[0.65, 0.10, 0.15, 0.10],
    )
    result = paired_statistics(work, paired_args)
    if result.empty:
        return result
    inverse = {value: key for key, value in mapping.items()}
    result["baseline"] = result["baseline"].replace(inverse)
    result.insert(1, "anchor", anchor)
    return result.rename(
        columns={
            "bkan_rmse_mean": "anchor_rmse_mean",
            "paired_delta_mean": "anchor_minus_baseline_mean",
        }
    )


def run(args: argparse.Namespace) -> None:
    frame, splits, provenance = load_dataset(args.data_root)
    if not set(args.seeds).issubset(set(splits["seed"].unique())):
        raise ValueError("Requested seed is absent from the frozen manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    metric_rows: list[dict[str, object]] = []
    manifests: list[pd.DataFrame] = []
    print(f"device={device}, tasks={args.tasks}, models={args.models}, seeds={args.seeds}")
    for seed in args.seeds:
        train, validation, calibration, test, assignment = split_frame(frame, splits, seed)
        manifests.append(assignment.assign(experiment="ring_direct_device_metrics_v1"))
        print(
            f"[seed={seed}] structures train/val/cal/test="
            f"{train.structure_id.nunique()}/{validation.structure_id.nunique()}/"
            f"{calibration.structure_id.nunique()}/{test.structure_id.nunique()}"
        )
        for task_key in args.tasks:
            task = TASKS[task_key]
            train_hash = frame_fingerprint(train, task)
            test_hash = frame_fingerprint(test, task)
            for model in args.models:
                out_path = prediction_path(args.output, seed, task, model)
                metadata_path = out_path.with_name("run_metadata.json")
                reused = False
                if args.resume and out_path.is_file() and metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("train_fingerprint") != train_hash or metadata.get("test_fingerprint") != test_hash:
                        raise RuntimeError(f"Refusing stale prediction: {out_path}")
                    prediction, prediction_std = load_prediction(out_path, test, task)
                    info = metadata
                    reused = True
                else:
                    started = time.perf_counter()
                    prediction, prediction_std, info = train_one(
                        model,
                        train,
                        validation,
                        test,
                        task,
                        seed,
                        device,
                        args,
                        out_path.parent,
                    )
                    info["wall_time_s"] = float(time.perf_counter() - started)
                    write_prediction(out_path, test, task, seed, model, prediction, prediction_std)
                    metadata_path.write_text(
                        json.dumps(
                            {
                                **info,
                                "seed": seed,
                                "task": task.key,
                                "model": model,
                                "train_fingerprint": train_hash,
                                "test_fingerprint": test_hash,
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                values = task_metrics(test, task, prediction)
                metric_rows.append(
                    {
                        "task_key": task.key,
                        "task": task.name,
                        "seed": seed,
                        "model": model,
                        **values,
                        "wall_time_s": float(info.get("wall_time_s", info.get("train_time_s", 0.0))),
                        "param_count": int(info.get("param_count", 0)),
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "test_rows": len(test),
                        "reused": reused,
                    }
                )
                primary = values[task.primary_raw_metric]
                print(f"[{seed} {task.key} {model}] primary={primary:.6g}, model-RMSE={values['rmse_target']:.6g}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    if not np.isfinite(metrics["rmse_target"]).all():
        raise RuntimeError("Non-finite response metric")
    metrics.to_csv(args.output / "metrics_by_seed.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(args.output / "metrics_summary.csv", index=False)
    split_output = pd.concat(manifests, ignore_index=True)
    split_output.to_csv(args.output / "split_manifest.csv", index=False)
    if split_output.groupby(["seed", "structure_id"])["split"].nunique().max() != 1:
        raise RuntimeError("Split audit failed")
    anchor_statistics(metrics, args, "bkan").to_csv(
        args.output / "paired_bkan_statistics.csv", index=False
    )
    anchor_statistics(metrics, args, "dkan").to_csv(
        args.output / "paired_dkan_statistics.csv", index=False
    )
    write_report(summary, args.output)
    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol": "ring_direct_device_metrics_v1",
        "data_root": str(args.data_root),
        "output": str(args.output),
        "tasks": {key: asdict(TASKS[key]) for key in args.tasks},
        "models": args.models,
        "seeds": args.seeds,
        "independent_unit": "structure_id",
        "split_counts": {"train": 52, "validation": 8, "calibration": 12, "test": 8},
        "bkan_dkan_matched_backbone": {"width": 8, "grid": 8, "k": 3},
        "device": str(device),
        "training": {
            "epochs": args.epochs,
            "kan_steps": args.kan_steps,
            "mlp_epochs": args.mlp_epochs,
            "autopinn_epochs": args.autopinn_epochs,
        },
        "provenance": provenance,
        "original_spectrum_code_modified": False,
        "paper_modified": False,
        "claim_boundary": "Direct scalar metric prediction; not full-spectrum reconstruction or measurement validation.",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = {
        "status": "PASS",
        "metric_rows": int(len(metrics)),
        "expected_metric_rows": len(args.seeds) * len(args.tasks) * len(args.models),
        "prediction_files": len(list((args.output / "predictions").rglob("test_predictions.csv"))),
        "split_rows": int(len(split_output)),
        "structure_split_leakage": False,
        "all_primary_rmse_finite": True,
        "metrics_sha256": file_sha256(args.output / "metrics_by_seed.csv"),
        "summary_sha256": file_sha256(args.output / "metrics_summary.csv"),
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


def main() -> int:
    args = resolve_args(parse_args())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
