"""Run matched heteroscedastic MLP UQ baselines on frozen grouped splits.

This experiment closes the likelihood mismatch in the historical comparison:
BKAN-VI, MC-dropout MLP, and deep-ensemble MLP all use a two-channel Gaussian
mean/variance head with the same softplus variance transformation.  The MLP
fits reuse the exact BKAN train/validation/calibration/test group manifests.
MC dropout matches the frozen BKAN epoch/update horizon.  The primary deep
ensemble distributes that same *total* optimizer-update budget over its
members; a separate optional higher-compute ensemble matches the budget per
member and is reported explicitly as an ``ensemble_size``-times reference.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import re
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.stats import t as student_t
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))

from device_modeling.photodetector import task_config as shared  # noqa: E402
from device_modeling.photodetector.bayesian_modeler import (  # noqa: E402
    BayesKANDeviceModeler,
)
from device_modeling.photodetector.heteroscedastic_mlp import (  # noqa: E402
    MLPTrainingConfig,
    predictive_moments,
    train_heteroscedastic_mlp,
)
from device_modeling.photodetector.run_research import (  # noqa: E402
    _normalize_metric_values,
)


DEFAULT_DATA = ROOT / "artifacts" / "results" / "device_modeling" / "cleaned_data.csv"
DEFAULT_BKAN = ROOT / "artifacts" / "results" / "uq_repeated_grouped"
DEFAULT_PHOTO_BKAN = (
    ROOT
    / "artifacts"
    / "results"
    / "net_photocurrent_retrain"
    / "uq_repeated_grouped"
)
DEFAULT_CAP_BKAN = (
    ROOT / "artifacts" / "results" / "uq_repeated_grouped_capacitance_device_cluster"
)
DEFAULT_CAP_MANIFEST = ROOT / "artifacts" / "results" / "uq_repeated_grouped_capacitance"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "results"
    / "matched_heteroscedastic_uq_equal_budget_current"
)
MODELS = (
    "mc_dropout_mlp",
    "equal_budget_deep_ensemble_mlp",
    "deep_ensemble_mlp",
)
DEFAULT_MODELS = ("mc_dropout_mlp", "equal_budget_deep_ensemble_mlp")
ENSEMBLE_MODELS = {"equal_budget_deep_ensemble_mlp", "deep_ensemble_mlp"}
TASKS = ("dark_current", "photo_current", "ac_response", "capacitance")
TASK_TOKEN = {
    "dark_current": "I-dark",
    "photo_current": "I-photo",
    "ac_response": "AC-response",
    "capacitance": "Capacitance",
}
SUMMARY_METRICS = (
    "rmse_model_space",
    "r2_model_space",
    "raw_gaussian_nll_task_space",
    "raw_gaussian_crps_task_space",
    "raw_picp_95",
    "raw_group_coverage_95",
    "raw_group_calibration_error_pp",
    "raw_mpiw_log",
    "raw_interval_score_95",
    "no_shrink_conformal_picp_95",
    "no_shrink_conformal_group_coverage_95",
    "no_shrink_group_calibration_error_pp",
    "no_shrink_conformal_mpiw_log",
    "no_shrink_conformal_interval_score_95",
    "calibration_raw_quantile",
)
PAIRED_METRICS = (
    "rmse_model_space",
    "raw_gaussian_nll_task_space",
    "raw_gaussian_crps_task_space",
    "raw_group_calibration_error_pp",
    "raw_mpiw_log",
    "raw_interval_score_95",
    "no_shrink_group_calibration_error_pp",
    "no_shrink_conformal_mpiw_log",
    "no_shrink_conformal_interval_score_95",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--capacitance-data", type=Path, default=shared.DEFAULT_CAPACITANCE_DATA
    )
    parser.add_argument("--bkan-root", type=Path, default=DEFAULT_BKAN)
    parser.add_argument(
        "--photo-current-bkan-root", type=Path, default=DEFAULT_PHOTO_BKAN
    )
    parser.add_argument("--capacitance-bkan-root", type=Path, default=DEFAULT_CAP_BKAN)
    parser.add_argument(
        "--capacitance-manifest-root", type=Path, default=DEFAULT_CAP_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument(
        "--split-fractions", nargs=4, type=float, default=[0.65, 0.10, 0.15, 0.10]
    )
    parser.add_argument("--hidden-widths", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--dropout-rate", type=float, default=0.05)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--prediction-seed", type=int, default=1729)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "data",
        "capacitance_data",
        "bkan_root",
        "photo_current_bkan_root",
        "capacitance_bkan_root",
        "capacitance_manifest_root",
        "output",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve())
    shared.validate_split_fractions(args.split_fractions)
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be unique")
    if not args.hidden_widths or any(width < 1 for width in args.hidden_widths):
        raise ValueError("hidden widths must be positive")
    if not 0.0 < args.dropout_rate < 1.0:
        raise ValueError("dropout rate must lie in (0, 1)")
    if args.ensemble_size < 2:
        raise ValueError("deep ensemble requires at least two members")
    if args.mc_samples < 2:
        raise ValueError("MC dropout requires at least two samples")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if "capacitance" in args.tasks and not args.capacitance_data.is_file():
        raise FileNotFoundError(args.capacitance_data)


def _literal_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = {}
    for key, value in payload.items():
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
        normalized[key] = value
    return normalized


def _manifest_root(task: str, args: argparse.Namespace) -> Path:
    if task == "capacitance":
        return args.capacitance_manifest_root
    if task == "photo_current":
        return args.photo_current_bkan_root
    return args.bkan_root


def _bkan_root(task: str, args: argparse.Namespace) -> Path:
    if task == "capacitance":
        return args.capacitance_bkan_root
    if task == "photo_current":
        return args.photo_current_bkan_root
    return args.bkan_root


def _load_task(task: str, args: argparse.Namespace):
    raw = (
        shared.load_capacitance_data(args.capacitance_data)
        if task == "capacitance"
        else shared.load_research_data(args.data)
    )
    spec = shared.TASKS[task]
    frame, inputs = shared.prepare_task_dataframe(raw, spec)
    return frame, spec, tuple(inputs)


def _regenerated_manifest(partitions, spec, seed: int) -> pd.DataFrame:
    rows = []
    for split_name, frame in zip(
        ("train", "validation", "calibration", "test"), partitions
    ):
        labels = shared.task_group_labels(frame, spec)
        for group_id, count in labels.value_counts(sort=False).items():
            rows.append(
                {
                    "task": spec.name,
                    "seed": int(seed),
                    "split": split_name,
                    "group_id": str(group_id),
                    "n_points": int(count),
                }
            )
    manifest = pd.DataFrame(rows)
    if (manifest.groupby("group_id")["split"].nunique() > 1).any():
        raise RuntimeError(f"Grouped leakage for {spec.name}, seed={seed}")
    return manifest


def _verify_frozen_manifest(
    regenerated: pd.DataFrame, task: str, seed: int, args: argparse.Namespace
) -> None:
    source = (
        _manifest_root(task, args)
        / f"seed_{seed}"
        / f"{task}_split_manifest.csv"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    frozen = pd.read_csv(source, dtype={"group_id": str})
    expected = {
        split: set(group["group_id"].astype(str))
        for split, group in regenerated.groupby("split")
    }
    observed = {
        split: set(group["group_id"].astype(str))
        for split, group in frozen.groupby("split")
    }
    if expected != observed:
        raise RuntimeError(f"Frozen split mismatch: {source}")


def _source_realized_epochs(
    task: str, seed: int, args: argparse.Namespace
) -> int:
    """Recover the realized frozen BKAN horizon from its immutable run log."""

    path = _manifest_root(task, args) / f"seed_{seed}" / "run.log"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = f"[{shared.TASKS[task].name}] samples="
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Task marker not found in {path}: {marker}")
    following = [
        position
        for key in TASKS
        if key != task
        for position in [text.find(f"[{shared.TASKS[key].name}] samples=", start + 1)]
        if position >= 0
    ]
    stop = min(following) if following else len(text)
    epochs = [int(value) for value in re.findall(r"Epoch (\d+)", text[start:stop])]
    if not epochs:
        raise RuntimeError(f"No completed epochs found for {task} in {path}")
    return max(epochs)


def _source_budget(
    task: str, seed: int, args: argparse.Namespace
) -> tuple[MLPTrainingConfig, dict]:
    path = _manifest_root(task, args) / f"seed_{seed}" / "run_config.json"
    source = _literal_config(path)
    if source.get("bayes_method", "vi") != "vi":
        raise ValueError(f"Expected VI source budget: {path}")
    maximum_epochs = int(source["epochs"])
    realized_epochs = _source_realized_epochs(task, seed, args)
    if realized_epochs > maximum_epochs:
        raise RuntimeError(
            f"Realized epochs exceed configured maximum in {path}: "
            f"{realized_epochs}>{maximum_epochs}"
        )
    budget = MLPTrainingConfig(
        # Use the already-frozen BKAN realized horizon and disable a second,
        # architecture-dependent stopping decision.  This gives MC dropout
        # exactly the same epoch and optimizer-update count while retaining
        # validation-NLL checkpoint selection within that horizon.
        epochs=realized_epochs,
        batch_size=int(source["batch_size"]),
        learning_rate=float(source["learning_rate"]),
        weight_decay=1.0e-5,
        train_mc_samples=int(source["train_mc_samples"]),
        validation_mc_samples=int(source["validation_mc_samples"]),
        validation_frequency=max(1, min(10, realized_epochs)),
        early_stopping_patience=0,
        early_stopping_min_delta=float(source["early_stopping_min_delta"]),
    )
    return budget, {
        "source_maximum_epochs": maximum_epochs,
        "source_realized_epochs": realized_epochs,
        "source_batch_size": int(source["batch_size"]),
        "source_learning_rate": float(source["learning_rate"]),
        "source_train_mc_samples": int(source["train_mc_samples"]),
        "source_validation_mc_samples": int(source["validation_mc_samples"]),
        "source_early_stopping_patience": int(source["early_stopping_patience"]),
        "matching_rule": "frozen_bkan_realized_epoch_and_update_horizon",
    }


def _fit_scalers(train: pd.DataFrame, spec, inputs: tuple[str, ...]):
    scaler_x = StandardScaler().fit(train[list(inputs)].to_numpy(dtype=np.float64))
    scaler_y = StandardScaler().fit(shared.target_values(train, spec).reshape(-1, 1))
    return scaler_x, scaler_y


def _conformal_group_columns(
    frame: pd.DataFrame, spec, inputs: tuple[str, ...]
) -> list[str]:
    """Use the exact independent unit represented by the frozen split."""

    if "_curve_id" in frame.columns:
        return ["_curve_id"]
    columns = [column for column in inputs if column != spec.axis_col]
    if not columns or any(column not in frame.columns for column in columns):
        raise ValueError(f"No grouped conformal columns available for {spec.name}")
    return columns


def _prediction_function(
    modeler: BayesKANDeviceModeler,
    models,
    *,
    method: str,
    mc_samples: int,
    prediction_seed: int,
    device: torch.device,
):
    def predict(inputs: np.ndarray) -> dict:
        scaled = modeler.scaler_x.transform(
            np.asarray(inputs, dtype=np.float64)
        ).astype(np.float32)
        moments = predictive_moments(
            models,
            scaled,
            method=method,
            mc_samples=mc_samples,
            prediction_seed=prediction_seed,
            device=device,
        )
        mean_samples = moments["mean_samples"]
        model_mean = modeler.scaler_y.inverse_transform(moments["mean"]).reshape(-1)
        target_scale = float(modeler.scaler_y.scale_[0])
        model_std = (
            np.sqrt(moments["total_variance"]) * target_scale
        ).reshape(-1)
        epistemic = (
            np.sqrt(moments["epistemic_variance"]) * target_scale
        ).reshape(-1)
        aleatoric = (
            np.sqrt(moments["aleatoric_variance"]) * target_scale
        ).reshape(-1)
        model_samples = modeler.scaler_y.inverse_transform(
            mean_samples.reshape(-1, 1)
        ).reshape(mean_samples.shape)
        if modeler.use_log_transform:
            linear_samples = np.power(10.0, np.clip(model_samples, -30.0, 30.0))
        else:
            linear_samples = model_samples
        linear_mean = np.mean(linear_samples, axis=0).reshape(-1)
        linear_std = np.std(linear_samples, axis=0).reshape(-1)

        calibrated_z = float(getattr(modeler, "_calib_z", 1.96))
        standard_z = float(
            getattr(modeler, "_calib_raw_quantile", calibrated_z)
        )
        if not np.isfinite(standard_z):
            standard_z = calibrated_z
        conformal_floor = (
            0.0
            if getattr(modeler, "_calib_unit", "uncalibrated") == "uncalibrated"
            else float(getattr(modeler, "_calib_sigma_floor", 0.0))
        )
        conformal_model_std = np.maximum(model_std, conformal_floor)
        raw_lower = model_mean - 1.96 * model_std
        raw_upper = model_mean + 1.96 * model_std
        standard_lower = model_mean - standard_z * conformal_model_std
        standard_upper = model_mean + standard_z * conformal_model_std
        calibrated_lower = model_mean - calibrated_z * conformal_model_std
        calibrated_upper = model_mean + calibrated_z * conformal_model_std

        def physical(values):
            if modeler.use_log_transform:
                return np.power(10.0, np.clip(values, -30.0, 30.0))
            return values

        return {
            "mean": linear_mean,
            "std": linear_std,
            "lower": physical(calibrated_lower),
            "upper": physical(calibrated_upper),
            "model_mean": model_mean,
            "model_std": model_std,
            "model_lower": calibrated_lower,
            "model_upper": calibrated_upper,
            "log_mean": model_mean,
            "log_std": model_std,
            "log_lower": calibrated_lower,
            "log_upper": calibrated_upper,
            "raw_lower": physical(raw_lower),
            "raw_upper": physical(raw_upper),
            "raw_model_lower": raw_lower,
            "raw_model_upper": raw_upper,
            "raw_log_lower": raw_lower,
            "raw_log_upper": raw_upper,
            "standard_conformal_model_lower": standard_lower,
            "standard_conformal_model_upper": standard_upper,
            "standard_conformal_log_lower": standard_lower,
            "standard_conformal_log_upper": standard_upper,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "samples": mean_samples,
            "log_samples": model_samples,
            "linear_samples": linear_samples,
        }

    return predict


def gaussian_crps(actual: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.maximum(np.asarray(std, dtype=np.float64), 1.0e-30)
    z = (np.asarray(actual) - np.asarray(mean)) / std
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return std * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * phi
        - 1.0 / math.sqrt(math.pi)
    )


def interval_score(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return (
        upper
        - lower
        + (2.0 / alpha) * (lower - actual) * (actual < lower)
        + (2.0 / alpha) * (actual - upper) * (actual > upper)
    )


def _proper_scores(
    prediction_path: Path, target_scale: float = 1.0
) -> dict[str, float]:
    """Score the moment-matched Gaussian, not the finite predictive mixture."""
    frame = pd.read_csv(prediction_path)
    target_scale = float(target_scale)
    if not np.isfinite(target_scale) or target_scale <= 0.0:
        raise ValueError(f"target scale must be positive and finite: {target_scale}")
    actual = (
        frame["actual_model_space"].to_numpy(dtype=np.float64) * target_scale
    )
    mean = (
        frame["prediction_mean_model_space"].to_numpy(dtype=np.float64)
        * target_scale
    )
    std = np.maximum(
        frame["prediction_std_model_space"].to_numpy(dtype=np.float64)
        * target_scale,
        1.0e-30,
    )
    raw_lower = mean - 1.96 * std
    raw_upper = mean + 1.96 * std
    calibrated_lower = (
        frame["calibrated_lower_model_space"].to_numpy(dtype=np.float64)
        * target_scale
    )
    calibrated_upper = (
        frame["calibrated_upper_model_space"].to_numpy(dtype=np.float64)
        * target_scale
    )
    nll = 0.5 * (
        np.log(2.0 * math.pi * std * std) + ((actual - mean) / std) ** 2
    )
    return {
        "raw_gaussian_nll_task_space": float(np.mean(nll)),
        "raw_gaussian_crps_task_space": float(
            np.mean(gaussian_crps(actual, mean, std))
        ),
        "raw_interval_score_95": float(
            np.mean(interval_score(actual, raw_lower, raw_upper))
        ),
        "no_shrink_conformal_interval_score_95": float(
            np.mean(interval_score(actual, calibrated_lower, calibrated_upper))
        ),
    }


def _prediction_path(root: Path, seed: int, task: str) -> Path:
    token = TASK_TOKEN[task].lower()
    candidates = [
        path
        for path in (root / f"seed_{seed}").rglob("uq_test_predictions.csv")
        if token in str(path).lower()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one prediction file for {task}, seed={seed}; found {candidates}"
        )
    return candidates[0]


def _bkan_metrics(
    task: str,
    seed: int,
    args: argparse.Namespace,
    source_budget: dict,
    train_samples: int,
    current_test,
) -> dict:
    root = _bkan_root(task, args)
    metadata = pd.read_csv(root / "uq_metrics_by_seed.csv")
    spec = shared.TASKS[task]
    row = metadata.loc[
        metadata["seed"].eq(seed) & metadata["task"].eq(spec.name)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Missing unique BKAN row for {spec.name}, seed={seed}")
    values = row.iloc[0].to_dict()
    legacy_aliases = {
        "raw_group_coverage_95": "raw_curve_coverage_95",
        "no_shrink_conformal_group_coverage_95": "calibrated_curve_coverage_95",
        "no_shrink_conformal_picp_95": "calibrated_picp_95",
        "no_shrink_conformal_mpiw_log": "calibrated_mpiw_log",
        "no_shrink_conformal_interval_score_95": "calibrated_interval_score_95",
    }
    for destination, source in legacy_aliases.items():
        if destination not in values and source in values:
            values[destination] = values[source]
    prediction_path = _prediction_path(root, seed, task)
    source_predictions = pd.read_csv(prediction_path)
    source_actual = np.sort(
        source_predictions["actual_model_space"].to_numpy(dtype=np.float64)
    )
    current_actual = np.sort(
        shared.target_values(current_test, spec).astype(np.float64)
    )
    if len(source_actual) != len(current_actual):
        raise RuntimeError(
            f"Frozen/current test size mismatch for {spec.name}, seed={seed}: "
            f"{len(source_actual)} != {len(current_actual)}"
        )
    nonzero = np.abs(source_actual) > 1.0e-30
    if not np.any(nonzero):
        target_scale = 1.0
    else:
        target_scale = float(
            np.median(np.abs(current_actual[nonzero] / source_actual[nonzero]))
        )
    if not np.allclose(
        current_actual,
        source_actual * target_scale,
        rtol=1.0e-8,
        atol=1.0e-30,
    ):
        raise RuntimeError(
            f"Frozen/current target mismatch is not a constant scale for "
            f"{spec.name}, seed={seed}; inferred scale={target_scale}"
        )
    for metric in (
        "rmse_model_space",
        "raw_mpiw_log",
        "no_shrink_conformal_mpiw_log",
        "raw_interval_score_95",
        "no_shrink_conformal_interval_score_95",
    ):
        if metric in values:
            values[metric] = float(values[metric]) * target_scale
    values.update(_proper_scores(prediction_path, target_scale=target_scale))
    values.update(
        {
            "task": spec.name,
            "task_key": task,
            "seed": int(seed),
            "model": "bkan_vi_frozen",
            "likelihood": "heteroscedastic_gaussian_softplus_variance",
            "architecture": "bkan_width8_grid8_order3",
            "budget_multiplier": 1.0,
            "epochs_completed_mean": float(source_budget["source_realized_epochs"]),
            "optimizer_updates_total": int(
                source_budget["source_realized_epochs"]
                * math.ceil(train_samples / source_budget["source_batch_size"])
            ),
            "train_samples": int(train_samples),
            "source_target_scale_to_current": target_scale,
            "source_prediction_file": str(prediction_path.relative_to(ROOT)),
            **source_budget,
            "frozen_source": str(root.relative_to(ROOT)),
        }
    )
    return _add_calibration_errors(values)


def _add_calibration_errors(metrics: dict) -> dict:
    metrics = dict(metrics)
    raw_group = float(metrics.get("raw_group_coverage_95", np.nan))
    calibrated_group = float(
        metrics.get("no_shrink_conformal_group_coverage_95", np.nan)
    )
    metrics["raw_group_calibration_error_pp"] = abs(raw_group - 95.0)
    metrics["no_shrink_group_calibration_error_pp"] = abs(
        calibrated_group - 95.0
    )
    return metrics


def _output_directory(args: argparse.Namespace, seed: int, task: str, model: str) -> Path:
    return args.output / f"seed_{seed}" / task / f"{model}_results"


def _scale_member_schedule(
    budget: MLPTrainingConfig, member_epochs: int
) -> MLPTrainingConfig:
    """Scale phase boundaries while preserving one member's short horizon.

    Equal-total-budget ensembles receive only a fraction of the source epochs.
    Leaving the original warm-up unchanged could keep every member in the MSE
    phase, so phase endpoints and the LR schedule are mapped by epoch fraction.
    """

    if member_epochs < 1:
        raise ValueError("member epochs must be positive")
    source_epochs = int(budget.epochs)
    warmup_end = math.floor(
        member_epochs * min(source_epochs, int(budget.warmup_epochs)) / source_epochs
    )
    transition_end = math.floor(
        member_epochs
        * min(
            source_epochs,
            int(budget.warmup_epochs) + int(budget.transition_epochs),
        )
        / source_epochs
    )
    return replace(
        budget,
        epochs=int(member_epochs),
        warmup_epochs=int(warmup_end),
        transition_epochs=int(max(0, transition_end - warmup_end)),
        scheduler_step_size=max(
            1,
            int(round(budget.scheduler_step_size * member_epochs / source_epochs)),
        ),
        validation_frequency=max(
            1, min(int(budget.validation_frequency), int(member_epochs))
        ),
    )


def _member_training_budgets(
    model_name: str,
    budget: MLPTrainingConfig,
    ensemble_size: int,
) -> list[MLPTrainingConfig]:
    """Return auditable per-member budgets for one baseline.

    Epoch allocation is sufficient for exact update matching because every
    member sees the same training rows and batch size, hence the same number of
    optimizer updates per epoch.
    """

    if model_name == "mc_dropout_mlp":
        return [budget]
    if model_name == "deep_ensemble_mlp":
        return [budget] * int(ensemble_size)
    if model_name != "equal_budget_deep_ensemble_mlp":
        raise ValueError(f"Unknown model budget policy: {model_name}")
    quotient, remainder = divmod(int(budget.epochs), int(ensemble_size))
    if quotient < 1:
        raise ValueError(
            "The frozen BKAN horizon must provide at least one epoch per "
            f"ensemble member: epochs={budget.epochs}, members={ensemble_size}"
        )
    allocation = [
        quotient + (1 if member < remainder else 0)
        for member in range(int(ensemble_size))
    ]
    if sum(allocation) != int(budget.epochs):
        raise RuntimeError("Equal-budget epoch allocation is not exact")
    return [_scale_member_schedule(budget, epochs) for epochs in allocation]


def _fit_model_set(
    model_name: str,
    arrays: dict,
    inputs: tuple[str, ...],
    seed: int,
    args: argparse.Namespace,
    budget: MLPTrainingConfig,
    device: torch.device,
):
    member_count = args.ensemble_size if model_name in ENSEMBLE_MODELS else 1
    dropout = args.dropout_rate if model_name == "mc_dropout_mlp" else 0.0
    member_budgets = _member_training_budgets(
        model_name, budget, member_count
    )
    models = []
    training_info = []
    for member, member_budget in enumerate(member_budgets):
        member_seed = int(seed * 1000 + member)
        model, info = train_heteroscedastic_mlp(
            arrays["x_train"],
            arrays["y_train"],
            arrays["x_validation"],
            arrays["y_validation"],
            hidden_widths=args.hidden_widths,
            dropout_rate=dropout,
            seed=member_seed,
            validation_seed=args.prediction_seed + member,
            device=device,
            config=member_budget,
        )
        models.append(model)
        training_info.append(info)
    return models, training_info, member_budgets


def _arrays(train, validation, spec, inputs, scaler_x, scaler_y) -> dict:
    return {
        "x_train": scaler_x.transform(
            train[list(inputs)].to_numpy(dtype=np.float64)
        ).astype(np.float32),
        "y_train": scaler_y.transform(
            shared.target_values(train, spec).reshape(-1, 1)
        ).astype(np.float32),
        "x_validation": scaler_x.transform(
            validation[list(inputs)].to_numpy(dtype=np.float64)
        ).astype(np.float32),
        "y_validation": scaler_y.transform(
            shared.target_values(validation, spec).reshape(-1, 1)
        ).astype(np.float32),
    }


def _run_baseline(
    model_name: str,
    partitions,
    spec,
    inputs,
    seed: int,
    args: argparse.Namespace,
    budget: MLPTrainingConfig,
    source_budget: dict,
    device: torch.device,
) -> dict:
    train, validation, calibration, test = partitions
    output = _output_directory(args, seed, spec.key, model_name)
    metrics_path = output / "metrics.json"
    if args.resume and metrics_path.is_file():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        member_count = (
            args.ensemble_size if model_name in ENSEMBLE_MODELS else 1
        )
        expected_budget_policy = {
            "mc_dropout_mlp": "frozen_bkan_total_updates",
            "equal_budget_deep_ensemble_mlp": (
                "frozen_bkan_total_updates_distributed_across_members"
            ),
            "deep_ensemble_mlp": "frozen_bkan_updates_per_member",
        }[model_name]
        expected = {
            "task": spec.name,
            "seed": int(seed),
            "model": model_name,
            "likelihood": "heteroscedastic_gaussian_softplus_variance",
            "architecture": f"mlp_{'-'.join(map(str, args.hidden_widths))}",
            "dropout_rate": (
                args.dropout_rate if model_name == "mc_dropout_mlp" else 0.0
            ),
            "ensemble_size": member_count,
            "mc_samples": (
                args.mc_samples
                if model_name == "mc_dropout_mlp"
                else member_count
            ),
            "source_realized_epochs": source_budget["source_realized_epochs"],
            "source_batch_size": source_budget["source_batch_size"],
            "budget_policy": expected_budget_policy,
        }
        mismatches = {
            key: {"expected": value, "observed": existing.get(key)}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Resume configuration mismatch for {metrics_path}: {mismatches}"
            )
        return existing
    if output.exists() and metrics_path.is_file():
        raise FileExistsError(f"Use --resume or a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    scaler_x, scaler_y = _fit_scalers(train, spec, inputs)
    arrays = _arrays(train, validation, spec, inputs, scaler_x, scaler_y)
    start = time.perf_counter()
    models, training_info, member_budgets = _fit_model_set(
        model_name, arrays, inputs, seed, args, budget, device
    )
    training_time = time.perf_counter() - start

    modeler = BayesKANDeviceModeler(
        task_name=spec.name, results_dir=str(output), device=str(device)
    )
    modeler.model = object()
    modeler.input_cols = list(inputs)
    modeler.output_col = spec.target_col
    modeler.use_log_transform = spec.use_log_transform
    modeler.y_bounds = spec.y_bounds
    modeler.scaler_x = scaler_x
    modeler.scaler_y = scaler_y
    modeler.data_stats = {
        "X_mean": scaler_x.mean_,
        "X_std": scaler_x.scale_,
        "y_mean": float(scaler_y.mean_[0]),
        "y_std": float(scaler_y.scale_[0]),
    }
    method = "mc_dropout" if model_name == "mc_dropout_mlp" else "deep_ensemble"
    modeler.predict_with_uncertainty = _prediction_function(
        modeler,
        models,
        method=method,
        mc_samples=args.mc_samples,
        prediction_seed=args.prediction_seed,
        device=device,
    )
    group_columns = _conformal_group_columns(calibration, spec, inputs)
    modeler.calibrate_uncertainty(
        df_calib=calibration,
        target_picp=95.0,
        min_z=1.96,
        calibration_unit="curve",
        group_cols=group_columns,
    )
    metrics = _normalize_metric_values(modeler.evaluate_on_test_set(test))
    metrics.update(_proper_scores(output / "uq_test_predictions.csv"))
    member_count = len(models)
    source_updates = int(
        source_budget["source_realized_epochs"]
        * math.ceil(len(train) / source_budget["source_batch_size"])
    )
    observed_updates = [int(info["optimizer_updates"]) for info in training_info]
    expected_member_updates = [
        int(member_budget.epochs)
        * math.ceil(len(train) / source_budget["source_batch_size"])
        for member_budget in member_budgets
    ]
    if observed_updates != expected_member_updates:
        raise RuntimeError(
            f"Optimizer-update budget mismatch for {spec.name}, seed={seed}, "
            f"model={model_name}: expected {expected_member_updates}, "
            f"observed {observed_updates}"
        )
    expected_total_updates = {
        "mc_dropout_mlp": source_updates,
        "equal_budget_deep_ensemble_mlp": source_updates,
        "deep_ensemble_mlp": source_updates * member_count,
    }[model_name]
    observed_total_updates = int(sum(observed_updates))
    if observed_total_updates != expected_total_updates:
        raise RuntimeError(
            f"Total optimizer-update mismatch for {spec.name}, seed={seed}, "
            f"model={model_name}: expected {expected_total_updates}, "
            f"observed {observed_total_updates}"
        )
    budget_policy = {
        "mc_dropout_mlp": "frozen_bkan_total_updates",
        "equal_budget_deep_ensemble_mlp": (
            "frozen_bkan_total_updates_distributed_across_members"
        ),
        "deep_ensemble_mlp": "frozen_bkan_updates_per_member",
    }[model_name]
    train_forward_multiplier = (
        int(budget.train_mc_samples) if method == "mc_dropout" else 1
    )
    metrics.update(
        {
            "task": spec.name,
            "task_key": spec.key,
            "seed": int(seed),
            "model": model_name,
            "likelihood": "heteroscedastic_gaussian_softplus_variance",
            "architecture": f"mlp_{'-'.join(map(str, args.hidden_widths))}",
            "dropout_rate": args.dropout_rate if method == "mc_dropout" else 0.0,
            "ensemble_size": member_count,
            "mc_samples": args.mc_samples if method == "mc_dropout" else member_count,
            "per_member_parameter_count": int(training_info[0]["parameter_count"]),
            "total_parameter_count": int(
                sum(info["parameter_count"] for info in training_info)
            ),
            "budget_policy": budget_policy,
            "budget_multiplier": float(observed_total_updates / source_updates),
            "training_time_s": float(training_time),
            "epochs_completed_mean": float(
                np.mean([info["epochs_completed"] for info in training_info])
            ),
            "optimizer_updates_total": observed_total_updates,
            "member_epoch_allocation": json.dumps(
                [int(member_budget.epochs) for member_budget in member_budgets]
            ),
            "member_optimizer_update_allocation": json.dumps(observed_updates),
            "estimated_training_forward_passes": int(
                observed_total_updates * train_forward_multiplier
            ),
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation)),
            "calibration_samples": int(len(calibration)),
            "test_samples": int(len(test)),
            **source_budget,
        }
    )
    metrics["per_member_optimizer_updates"] = float(np.mean(observed_updates))
    metrics["source_bkan_optimizer_updates"] = source_updates
    metrics["total_optimizer_update_ratio_vs_bkan"] = float(
        observed_total_updates / source_updates
    )
    metrics["budget_audit_passed"] = True
    metrics = _add_calibration_errors(metrics)

    checkpoint = {
        "model_states": [
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for model in models
        ],
        "model": model_name,
        "input_columns": list(inputs),
        "hidden_widths": list(args.hidden_widths),
        "dropout_rate": metrics["dropout_rate"],
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "training_info": training_info,
        "training_budget": asdict(budget),
        "member_training_budgets": [
            asdict(member_budget) for member_budget in member_budgets
        ],
        "source_budget": source_budget,
        "prediction_seed": args.prediction_seed,
    }
    torch.save(checkpoint, output / "model_checkpoint.pt")
    (output / "training_info.json").write_text(
        json.dumps(training_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, model), group in metrics.groupby(["task", "model"], sort=False):
        row = {"task": task, "model": model, "n_seeds": int(group["seed"].nunique())}
        for metric in SUMMARY_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            n = len(values)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else np.nan
            radius = (
                float(student_t.ppf(0.975, n - 1)) * std / math.sqrt(n)
                if n > 1
                else np.nan
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - radius
            row[f"{metric}_ci95_high"] = mean + radius
        rows.append(row)
    return pd.DataFrame(rows)


def corrected_repeated_split_statistics(
    differences: np.ndarray,
    train_fraction: float,
    test_fraction: float,
) -> tuple[float, float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan, np.nan, np.nan
    variance = float(values.var(ddof=1))
    corrected_se = math.sqrt(
        (1.0 / len(values) + test_fraction / train_fraction) * variance
    )
    mean = float(values.mean())
    if corrected_se == 0.0:
        return mean, mean, 0.0, 0.0 if mean != 0.0 else 1.0
    critical = float(student_t.ppf(0.975, len(values) - 1))
    p_value = float(
        2.0 * student_t.sf(abs(mean / corrected_se), len(values) - 1)
    )
    return mean - critical * corrected_se, mean + critical * corrected_se, corrected_se, p_value


def _holm_adjust(values: pd.Series) -> pd.Series:
    raw = values.to_numpy(dtype=np.float64)
    order = np.argsort(raw)
    adjusted = np.empty(len(raw), dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        candidate = min(1.0, raw[index] * (len(raw) - position))
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def paired_statistics(metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for task, task_frame in metrics.groupby("task", sort=False):
        for metric in PAIRED_METRICS:
            pivot = task_frame.pivot(index="seed", columns="model", values=metric)
            for baseline in args.models:
                if baseline not in pivot or "bkan_vi_frozen" not in pivot:
                    raise RuntimeError(f"Incomplete pairing: {task}, {metric}, {baseline}")
                # Positive delta means the baseline is worse for every endpoint
                # in PAIRED_METRICS (coverage is represented by absolute error).
                delta = (
                    pivot[baseline] - pivot["bkan_vi_frozen"]
                ).to_numpy(dtype=np.float64)
                low, high, corrected_se, p_value = corrected_repeated_split_statistics(
                    delta,
                    train_fraction=float(args.split_fractions[0]),
                    test_fraction=float(args.split_fractions[3]),
                )
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "baseline": baseline,
                        "n_pairs": int(len(delta)),
                        "baseline_minus_bkan_mean": float(np.mean(delta)),
                        "corrected_ci95_low": low,
                        "corrected_ci95_high": high,
                        "corrected_standard_error": corrected_se,
                        "corrected_p_value": p_value,
                        "baseline_better_runs": int(np.sum(delta < 0.0)),
                        "ties": int(np.sum(delta == 0.0)),
                        "bkan_better_runs": int(np.sum(delta > 0.0)),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["corrected_p_value_holm_global"] = _holm_adjust(
        frame["corrected_p_value"]
    )
    frame["multiplicity_family"] = (
        "all prespecified task x baseline x endpoint contrasts in this run"
    )
    return frame


def _write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    model_lines = []
    if "mc_dropout_mlp" in args.models:
        model_lines.append(
            f"- MC dropout uses {args.mc_samples} stochastic predictions and "
            f"dropout rate {args.dropout_rate:g}; its total epoch/update budget "
            "matches the frozen BKAN split exactly."
        )
    if "equal_budget_deep_ensemble_mlp" in args.models:
        model_lines.append(
            f"- Equal-budget deep ensemble uses {args.ensemble_size} independently "
            "initialized members and distributes the frozen BKAN epoch/update "
            "horizon across them; phase boundaries and the LR schedule are scaled "
            "by each member's epoch fraction, and total optimizer updates match "
            "BKAN exactly."
        )
    if "deep_ensemble_mlp" in args.models:
        model_lines.append(
            f"- Deep ensemble uses {args.ensemble_size} independently initialized "
            f"members; it is likelihood- and per-member-budget matched, but costs "
            f"{args.ensemble_size}x the optimizer-update budget."
        )
    else:
        model_lines.append(
            f"- Deep ensemble was not run in this invocation. The optional runner "
            f"uses {args.ensemble_size} per-member-matched fits and therefore costs "
            f"{args.ensemble_size}x the BKAN optimizer-update budget."
        )
    lines = [
        "# Matched Heteroscedastic UQ Baselines",
        "",
        "## Protocol",
        "",
        "- Exact frozen BKAN grouped manifests are replayed for every task and seed.",
        "- All methods use a Gaussian mean/variance head with `softplus(logit)+1e-6` variance.",
        "- MLP fits copy the frozen batch size and learning rate. MC dropout and the optional higher-compute ensemble retain the source schedule; the equal-budget ensemble rescales phase boundaries and the LR schedule to each member's allocated horizon. A second architecture-dependent early stop is disabled so optimizer-update counts are exact.",
        *model_lines,
        "- Calibration uses the untouched calibration groups and the maximum normalized residual within each condition curve or complete TCAD.",
        "- One training initialization is paired with each split, using `split_seed*1000+member_index`; as in the frozen BKAN run, split and initialization variability are not separately crossed.",
        "- Gaussian NLL and CRPS score the unchanged raw predictive Gaussian; conformal scaling changes only interval diagnostics.",
        "",
        "## Key means",
        "",
        "| Task | Model | RMSE | Raw NLL | Raw CRPS | Raw group cov. | Cal. group cov. | Cal. MPIW |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['task']} | {row['model']} | "
            f"{row.get('rmse_model_space_mean', np.nan):.6g} | "
            f"{row.get('raw_gaussian_nll_task_space_mean', np.nan):.6g} | "
            f"{row.get('raw_gaussian_crps_task_space_mean', np.nan):.6g} | "
            f"{row.get('raw_group_coverage_95_mean', np.nan):.2f}% | "
            f"{row.get('no_shrink_conformal_group_coverage_95_mean', np.nan):.2f}% | "
            f"{row.get('no_shrink_conformal_mpiw_log_mean', np.nan):.6g} |"
        )
    lines.extend(
        [
            "",
            "Corrected paired results are in `paired_statistics.csv`. Positive `baseline_minus_bkan_mean` means the baseline is worse; repeated test splits reuse groups, so the Nadeau--Bengio-style correction is used instead of an independent-split t interval. `corrected_p_value_holm_global` adjusts the full prespecified task x baseline x endpoint family from this invocation; conclusions are based on estimates and corrected intervals, not p values alone.",
            "",
        ]
    )
    (args.output / "matched_heteroscedastic_uq_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_acceptance_audit(
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    manifests: pd.DataFrame,
) -> None:
    expected_baselines = len(args.tasks) * len(args.seeds) * len(args.models)
    expected_total = len(args.tasks) * len(args.seeds) * (len(args.models) + 1)
    expected_summary = len(args.tasks) * (len(args.models) + 1)
    expected_paired = len(args.tasks) * len(args.models) * len(PAIRED_METRICS)
    baseline = metrics.loc[metrics["model"].isin(args.models)].copy()
    required_metrics = [
        "rmse_model_space",
        "raw_gaussian_nll_task_space",
        "raw_gaussian_crps_task_space",
        "raw_group_coverage_95",
        "no_shrink_conformal_group_coverage_95",
        "raw_mpiw_log",
        "no_shrink_conformal_mpiw_log",
        "raw_interval_score_95",
        "no_shrink_conformal_interval_score_95",
    ]
    overlap_count = int(
        (
            manifests.groupby(["task", "seed", "group_id"])["split"].nunique()
            > 1
        ).sum()
    )
    holm_applicable = len(args.seeds) >= 2
    holm_not_below_raw = True
    if holm_applicable:
        finite_holm = paired[
            ["corrected_p_value", "corrected_p_value_holm_global"]
        ].dropna()
        holm_not_below_raw = bool(
            (
                finite_holm["corrected_p_value_holm_global"] + 1.0e-15
                >= finite_holm["corrected_p_value"]
            ).all()
        ) and len(finite_holm) == len(paired)
    update_ratio_ok = True
    if "mc_dropout_mlp" in args.models:
        dropout = baseline.loc[baseline["model"].eq("mc_dropout_mlp")]
        update_ratio_ok &= bool(
            np.allclose(
                dropout["total_optimizer_update_ratio_vs_bkan"],
                1.0,
                rtol=0.0,
                atol=0.0,
            )
        )
    if "deep_ensemble_mlp" in args.models:
        ensemble = baseline.loc[baseline["model"].eq("deep_ensemble_mlp")]
        update_ratio_ok &= bool(
            np.allclose(
                ensemble["total_optimizer_update_ratio_vs_bkan"],
                float(args.ensemble_size),
                rtol=0.0,
                atol=0.0,
            )
        )
    if "equal_budget_deep_ensemble_mlp" in args.models:
        ensemble = baseline.loc[
            baseline["model"].eq("equal_budget_deep_ensemble_mlp")
        ]
        update_ratio_ok &= bool(
            np.allclose(
                ensemble["total_optimizer_update_ratio_vs_bkan"],
                1.0,
                rtol=0.0,
                atol=0.0,
            )
        )
    checks = {
        "metrics_rows": {"observed": len(metrics), "expected": expected_total},
        "baseline_rows": {
            "observed": len(baseline),
            "expected": expected_baselines,
        },
        "summary_rows": {"observed": len(summary), "expected": expected_summary},
        "paired_rows": {"observed": len(paired), "expected": expected_paired},
        "unique_baseline_task_seed_model": {
            "observed": int(
                baseline[["task", "seed", "model"]].drop_duplicates().shape[0]
            ),
            "expected": expected_baselines,
        },
        "required_metric_nan_count": int(
            metrics[required_metrics].isna().sum().sum()
        ),
        "baseline_budget_audits_pass": bool(baseline["budget_audit_passed"].all()),
        "total_update_ratio_matches_declared_budget": update_ratio_ok,
        "split_overlap_count": overlap_count,
        "split_task_seed_count": int(
            manifests[["task", "seed"]].drop_duplicates().shape[0]
        ),
        "holm_nan_count": int(
            paired["corrected_p_value_holm_global"].isna().sum()
        ),
        "holm_applicable": holm_applicable,
        "holm_not_below_raw": holm_not_below_raw,
        "checkpoint_count": len(list(args.output.rglob("model_checkpoint.pt"))),
        "prediction_count": len(list(args.output.rglob("uq_test_predictions.csv"))),
        "expected_local_artifact_count": expected_baselines,
    }
    failures = []
    for name in (
        "metrics_rows",
        "baseline_rows",
        "summary_rows",
        "paired_rows",
        "unique_baseline_task_seed_model",
    ):
        if checks[name]["observed"] != checks[name]["expected"]:
            failures.append(name)
    if checks["required_metric_nan_count"] != 0:
        failures.append("required_metric_nan_count")
    if not checks["baseline_budget_audits_pass"]:
        failures.append("baseline_budget_audits_pass")
    if not checks["total_update_ratio_matches_declared_budget"]:
        failures.append("total_update_ratio_matches_declared_budget")
    if checks["split_overlap_count"] != 0:
        failures.append("split_overlap_count")
    if checks["split_task_seed_count"] != len(args.tasks) * len(args.seeds):
        failures.append("split_task_seed_count")
    if holm_applicable and (
        checks["holm_nan_count"] != 0 or not checks["holm_not_below_raw"]
    ):
        failures.append("holm_adjustment")
    for name in ("checkpoint_count", "prediction_count"):
        if checks[name] != expected_baselines:
            failures.append(name)
    payload = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "checks": checks,
    }
    (args.output / "acceptance_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Acceptance audit failed: {failures}")


def write_outputs(
    args: argparse.Namespace, metrics: pd.DataFrame, manifests: pd.DataFrame
) -> None:
    expected_models = {"bkan_vi_frozen", *args.models}
    for task in (shared.TASKS[key].name for key in args.tasks):
        task_frame = metrics.loc[metrics["task"].eq(task)]
        if set(task_frame["model"]) != expected_models:
            raise RuntimeError(f"Incomplete model set for {task}")
        counts = task_frame.groupby("model")["seed"].nunique()
        if not (counts == len(args.seeds)).all():
            raise RuntimeError(f"Incomplete seed pairing for {task}: {counts.to_dict()}")
    summary = summarize_metrics(metrics)
    paired = paired_statistics(metrics, args)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics_by_seed.csv", index=False)
    summary.to_csv(args.output / "metrics_summary.csv", index=False)
    paired.to_csv(args.output / "paired_statistics.csv", index=False)
    manifests.to_csv(args.output / "split_manifest_audit.csv", index=False)
    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol": "frozen_grouped_heteroscedastic_likelihood_matched",
        "predictive_distribution_before_reduction": "finite equally weighted heteroscedastic Gaussian mixture over stochastic forward passes",
        "predictive_scoring_distribution": "Gaussian with the mixture's finite-sample mean and total variance",
        "mixture_nll_or_crps_reported": False,
        "data": str(args.data),
        "capacitance_data": str(args.capacitance_data),
        "bkan_root": str(args.bkan_root),
        "photo_current_bkan_root": str(args.photo_current_bkan_root),
        "capacitance_bkan_root": str(args.capacitance_bkan_root),
        "capacitance_manifest_root": str(args.capacitance_manifest_root),
        "tasks": args.tasks,
        "models": args.models,
        "seeds": args.seeds,
        "split_fractions": args.split_fractions,
        "hidden_widths": args.hidden_widths,
        "dropout_rate": args.dropout_rate,
        "ensemble_size": args.ensemble_size,
        "mc_samples": args.mc_samples,
        "prediction_seed": args.prediction_seed,
        "training_seed_rule": "split_seed*1000+member_index",
        "device": args.device,
        "multiplicity_family": (
            "all prespecified task x baseline x endpoint contrasts in this run"
        ),
        "fairness_scope": {
            "same_split": True,
            "same_likelihood": True,
            "mc_dropout_same_total_epoch_and_optimizer_update_budget": (
                "mc_dropout_mlp" in args.models
            ),
            "deep_ensemble_status": (
                "run_as_higher_compute_reference"
                if "deep_ensemble_mlp" in args.models
                else "not_run_in_this_invocation"
            ),
            "deep_ensemble_same_per_member_epoch_and_optimizer_update_budget": (
                "deep_ensemble_mlp" in args.models
            ),
            "deep_ensemble_total_budget_multiplier": (
                args.ensemble_size
                if "deep_ensemble_mlp" in args.models
                else None
            ),
            "equal_budget_deep_ensemble_status": (
                "total_optimizer_updates_match_frozen_bkan"
                if "equal_budget_deep_ensemble_mlp" in args.models
                else "not_run_in_this_invocation"
            ),
            "equal_budget_deep_ensemble_total_budget_multiplier": (
                1.0
                if "equal_budget_deep_ensemble_mlp" in args.models
                else None
            ),
            "equal_budget_scope": (
                "optimizer updates and realized epoch horizon; FLOPs and wall-clock "
                "are reported, not asserted equal"
            ),
            "same_architecture": False,
        },
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(args, summary, paired)
    _write_acceptance_audit(args, metrics, summary, paired, manifests)
    print(summary.to_string(index=False))


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    rows = []
    manifests = []
    for task in args.tasks:
        frame, spec, inputs = _load_task(task, args)
        for seed in args.seeds:
            partitions = shared.split_task_dataframe(
                frame, spec, seed, tuple(args.split_fractions)
            )
            manifest = _regenerated_manifest(partitions, spec, seed)
            _verify_frozen_manifest(manifest, task, seed, args)
            manifests.append(manifest)
            budget, source_budget = _source_budget(task, seed, args)
            rows.append(
                _bkan_metrics(
                    task,
                    seed,
                    args,
                    source_budget,
                    train_samples=len(partitions[0]),
                    current_test=partitions[3],
                )
            )
            for model_name in args.models:
                result = _run_baseline(
                    model_name,
                    partitions,
                    spec,
                    inputs,
                    seed,
                    args,
                    budget,
                    source_budget,
                    device,
                )
                rows.append(result)
                print(
                    f"[{spec.name} seed={seed} {model_name}] "
                    f"RMSE={result['rmse_model_space']:.6g}, "
                    f"NLL={result['raw_gaussian_nll_task_space']:.6g}, "
                    f"group_cov={result['no_shrink_conformal_group_coverage_95']:.2f}%"
                )
    write_outputs(
        args,
        pd.DataFrame(rows),
        pd.concat(manifests, ignore_index=True),
    )


def summarize_existing(args: argparse.Namespace) -> None:
    metrics_path = args.output / "metrics_by_seed.csv"
    manifest_path = args.output / "split_manifest_audit.csv"
    if not metrics_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Existing metrics and split audit are required")
    write_outputs(args, pd.read_csv(metrics_path), pd.read_csv(manifest_path))


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
