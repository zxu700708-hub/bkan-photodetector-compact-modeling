"""Run a data-budget-matched multi-teacher symbolic export Pareto audit.

The audit uses one frozen grouped split per task.  BKAN is converted to its
deterministic variational-parameter-mean KAN; DKAN and MLP-L are retrained with
the frozen confirmatory hyperparameters and checked against the previously
saved test predictions.  Every export student sees *training rows only*.
Validation rows choose the ridge coefficient independently at each fixed
complexity budget.  Calibration rows are never consumed, and test rows are
evaluated once after the export is frozen.

Two KAN-free export families are compared at three fixed complexity budgets:
total-degree polynomial Ridge and additive cubic B-spline Ridge.  The same
families and budgets are also fit directly to TCAD targets, so teacher choice
and direct fitting share identical rows, model classes, validation policy, and
test set.  Each fitted expression is independently replayed from JSON and is
also emitted as standalone arithmetic Verilog-A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
PAPER_EXPERIMENTS = BKAN_ROOT / "simulation" / "paper_experiments"
for path in (BKAN_ROOT, PAPER_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from device_modeling.photodetector import task_config as shared  # noqa: E402
from device_modeling.photodetector.bayesian_modeler import (  # noqa: E402
    BayesKANDeviceModeler,
)
from p0_3_traditional_comparison import (  # noqa: E402
    MLP_ARCHES,
    TASKS as BASELINE_TASKS,
    TorchMLP,
    set_seed,
    transformed_target,
)
from run_symbolic_export_audit import (  # noqa: E402
    _eval_poly,
    _eval_spline,
    _poly_payload,
    _spline_payload,
    _write_poly_va,
    _write_spline_va,
)


DEFAULT_DATA = ROOT / "artifacts/results/device_modeling/cleaned_data.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/results/multi_teacher_symbolic_pareto"
DEFAULT_MATCHED = ROOT / "artifacts/results/matched_grouped_comparison"
DEFAULT_BKAN = ROOT / "artifacts/results/uq_repeated_grouped"
DEFAULT_CAP_BKAN = ROOT / "artifacts/results/uq_repeated_grouped_capacitance"
TASK_KEY = {
    "I_dark": "dark_current",
    "I_photo": "photo_current",
    "AC_Response": "ac_response",
    "Capacitance": "capacitance",
}
SOURCES = ("bkan_parameter_mean", "dkan", "mlp_l", "tcad_direct")
CAPACITANCE_LEGACY_SCALE = 1000.0


def _artifact_path(path: Path) -> str:
    """Prefer repository-relative paths while supporting external test outputs."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--capacitance-data", type=Path, default=shared.DEFAULT_CAPACITANCE_DATA
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matched-root", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--bkan-root", type=Path, default=DEFAULT_BKAN)
    parser.add_argument("--capacitance-bkan-root", type=Path, default=DEFAULT_CAP_BKAN)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_KEY), default=list(TASK_KEY))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-fractions", nargs=4, type=float, default=(0.65, 0.10, 0.15, 0.10)
    )
    parser.add_argument("--poly-degrees", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--spline-knots", nargs="+", type=int, default=(4, 6, 8))
    parser.add_argument(
        "--ridge-alphas", nargs="+", type=float, default=(1e-8, 1e-6, 1e-4, 1e-2)
    )
    parser.add_argument("--dense-points", type=int, default=5000)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument(
        "--allow-teacher-replay-mismatch",
        action="store_true",
        help="Record rather than fail on a DKAN/MLP mismatch against frozen test CSVs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.data = args.data.resolve()
    args.capacitance_data = args.capacitance_data.resolve()
    args.output = args.output.resolve()
    args.matched_root = args.matched_root.resolve()
    args.bkan_root = args.bkan_root.resolve()
    args.capacitance_bkan_root = args.capacitance_bkan_root.resolve()
    shared.validate_split_fractions(args.split_fractions)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if "Capacitance" in args.tasks and not args.capacitance_data.is_file():
        raise FileNotFoundError(args.capacitance_data)
    if any(value < 1 for value in args.poly_degrees):
        raise ValueError("Polynomial degrees must be positive")
    if any(value < 3 for value in args.spline_knots):
        raise ValueError("Spline knot counts must be at least 3")
    if any(value <= 0.0 for value in args.ridge_alphas):
        raise ValueError("Ridge coefficients must be positive")
    if args.dense_points < 100:
        raise ValueError("--dense-points must be at least 100")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_task(task: str, args: argparse.Namespace):
    key = TASK_KEY[task]
    shared_spec = shared.TASKS[key]
    raw = (
        shared.load_capacitance_data(args.capacitance_data)
        if task == "Capacitance"
        else shared.load_research_data(args.data)
    )
    frame, inputs = shared.prepare_task_dataframe(raw, shared_spec)
    baseline_spec = replace(
        BASELINE_TASKS[task],
        target_col=shared_spec.target_col,
        input_cols=tuple(inputs),
    )
    return frame, tuple(inputs), shared_spec, baseline_spec


def _split_manifest(
    task: str,
    seed: int,
    spec: shared.TaskSpec,
    partitions: tuple[pd.DataFrame, ...],
) -> pd.DataFrame:
    rows: list[dict] = []
    seen: dict[str, set[str]] = {}
    for split, frame in zip(
        ("train", "validation", "calibration", "test"), partitions
    ):
        labels = shared.task_group_labels(frame, spec).astype(str)
        seen[split] = set(labels)
        for group_id, count in labels.value_counts(sort=False).items():
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "split": split,
                    "group_id": group_id,
                    "n_points": int(count),
                }
            )
    for left_index, left in enumerate(seen):
        for right in list(seen)[left_index + 1 :]:
            overlap = seen[left] & seen[right]
            if overlap:
                raise RuntimeError(
                    f"Grouped leakage for {task}: {left}/{right} share {len(overlap)} groups"
                )
    return pd.DataFrame(rows)


def _target(frame: pd.DataFrame, spec) -> np.ndarray:
    return transformed_target(frame, spec).astype(np.float64)


def _scaled_training_arrays(train: pd.DataFrame, inputs: tuple[str, ...], spec):
    x_scaler = StandardScaler().fit(train[list(inputs)].to_numpy(dtype=np.float32))
    y_scaler = StandardScaler().fit(_target(train, spec).reshape(-1, 1))
    x = x_scaler.transform(train[list(inputs)].to_numpy(dtype=np.float32)).astype(np.float32)
    y = y_scaler.transform(_target(train, spec).reshape(-1, 1)).astype(np.float32)
    return x_scaler, y_scaler, x, y


def _fit_mlp_teacher(
    train: pd.DataFrame,
    inputs: tuple[str, ...],
    spec,
    seed: int,
    epochs: int,
) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    x_scaler, y_scaler, x_train, y_train = _scaled_training_arrays(train, inputs, spec)
    set_seed(seed)
    device = torch.device("cpu")
    model = TorchMLP(len(inputs), MLP_ARCHES["mlp_l"], seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = torch.nn.MSELoss()
    x_tensor = torch.from_numpy(x_train).float().to(device)
    y_tensor = torch.from_numpy(y_train).float().to(device)

    def regularization() -> torch.Tensor:
        total = torch.tensor(0.0, device=device)
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.ndim >= 2:
                absolute = torch.abs(parameter)
                total += torch.sum(absolute)
                row_probability = absolute / (torch.sum(absolute, dim=1, keepdim=True) + 1e-4)
                entropy = -torch.mean(
                    torch.sum(row_probability * torch.log2(row_probability + 1e-4), dim=1)
                )
                total += 2.0 * entropy
        return total

    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_tensor), y_tensor) + 0.001 * regularization()
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    model.eval()

    def predict(frame: pd.DataFrame) -> np.ndarray:
        raw = frame[list(inputs)].to_numpy(dtype=np.float32)
        scaled = x_scaler.transform(raw).astype(np.float32)
        with torch.no_grad():
            output = model(torch.from_numpy(scaled).float()).cpu().numpy()
        values = y_scaler.inverse_transform(output).reshape(-1).astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("MLP teacher produced non-finite predictions")
        return values

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return predict, {"train_time_s": elapsed, "parameter_count": parameter_count}


def _fit_dkan_teacher(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    inputs: tuple[str, ...],
    spec,
    seed: int,
    steps: int,
) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    from kan import KAN

    x_scaler, y_scaler, x_train, y_train = _scaled_training_arrays(train, inputs, spec)
    x_validation = x_scaler.transform(
        validation[list(inputs)].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    y_validation = y_scaler.transform(_target(validation, spec).reshape(-1, 1)).astype(
        np.float32
    )
    set_seed(seed)
    device = torch.device("cpu")
    model = KAN(
        width=[len(inputs), 8, 1], grid=8, k=3, seed=seed, device=device,
        auto_save=False,
    )
    dataset = {
        "train_input": torch.from_numpy(x_train).float(),
        "train_label": torch.from_numpy(y_train).float(),
        # PyKAN requires a test field for diagnostic loss.  Validation, never
        # final test, is supplied here; it does not select a checkpoint.
        "test_input": torch.from_numpy(x_validation).float(),
        "test_label": torch.from_numpy(y_validation).float(),
    }
    batch = min(256, len(x_train), len(x_validation))
    start = time.perf_counter()
    model.fit(
        dataset,
        opt="Adam",
        steps=steps,
        lr=0.002,
        lamb=0.001,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.1,
        lamb_coefdiff=0.1,
        batch=batch,
        log=max(steps + 1, 1),
    )
    elapsed = time.perf_counter() - start
    model.eval()

    def predict(frame: pd.DataFrame) -> np.ndarray:
        raw = frame[list(inputs)].to_numpy(dtype=np.float32)
        scaled = x_scaler.transform(raw).astype(np.float32)
        with torch.no_grad():
            output = model(torch.from_numpy(scaled).float())
            if isinstance(output, tuple):
                output = output[0]
        values = y_scaler.inverse_transform(
            output.detach().cpu().numpy().reshape(-1, 1)
        ).reshape(-1)
        values = values.astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("DKAN teacher produced non-finite predictions")
        return values

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return predict, {"train_time_s": elapsed, "parameter_count": parameter_count}


def _bkan_checkpoint(task: str, seed: int, spec: shared.TaskSpec, args) -> Path:
    root = args.capacitance_bkan_root if task == "Capacitance" else args.bkan_root
    return root / f"seed_{seed}" / spec.result_subdir / "model_checkpoint.pt"


def _load_bkan_parameter_mean(
    task: str,
    train: pd.DataFrame,
    inputs: tuple[str, ...],
    spec: shared.TaskSpec,
    args: argparse.Namespace,
) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    checkpoint = _bkan_checkpoint(task, args.seed, spec, args)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    modeler = BayesKANDeviceModeler.load_model(str(checkpoint), device="cpu")
    if list(modeler.input_cols) != list(inputs):
        raise RuntimeError(f"BKAN input mismatch for {task}")
    observed_mean = train[list(inputs)].to_numpy(dtype=np.float64).mean(axis=0)
    expected_mean = np.asarray(modeler.scaler_x.mean_, dtype=np.float64)
    relative_delta = float(
        np.max(np.abs(expected_mean - observed_mean) / np.maximum(np.abs(expected_mean), 1.0))
    )
    if relative_delta > 1e-7:
        raise RuntimeError(
            f"BKAN checkpoint/split mismatch for {task}: scaler delta={relative_delta:.3g}"
        )
    output_scale = 1.0
    if task == "Capacitance":
        observed_y_mean = float(np.mean(_target(train, spec)))
        checkpoint_y_mean = float(np.asarray(modeler.scaler_y.mean_, dtype=np.float64).reshape(-1)[0])
        if checkpoint_y_mean == 0.0:
            raise RuntimeError("Cannot infer capacitance checkpoint output scale from zero mean")
        ratio = observed_y_mean / checkpoint_y_mean
        if np.isclose(ratio, CAPACITANCE_LEGACY_SCALE, rtol=1e-7, atol=0.0):
            output_scale = CAPACITANCE_LEGACY_SCALE
        elif not np.isclose(ratio, 1.0, rtol=1e-7, atol=0.0):
            raise RuntimeError(
                "Unregistered capacitance checkpoint/target scale mismatch: "
                f"mean ratio={ratio:.12g}"
            )
    deterministic = modeler.model.to_deterministic_kan(
        device="cpu", symbolic_enabled=False, auto_save=False
    ).eval()

    def predict(frame: pd.DataFrame) -> np.ndarray:
        raw = frame[list(inputs)].to_numpy(dtype=np.float32)
        scaled = modeler.scaler_x.transform(raw).astype(np.float32)
        with torch.no_grad():
            output = deterministic(torch.from_numpy(scaled).float())
            if isinstance(output, tuple):
                output = output[0]
        values = modeler.scaler_y.inverse_transform(
            output.detach().cpu().numpy().reshape(-1, 1)
        ).reshape(-1) * output_scale
        values = values.astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("BKAN parameter-mean teacher produced non-finite predictions")
        return values

    parameter_count = int(sum(parameter.numel() for parameter in deterministic.parameters()))
    return predict, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "scaler_mean_max_relative_delta": relative_delta,
        "parameter_count": parameter_count,
        "checkpoint_output_scale_applied": output_scale,
        "train_time_s": np.nan,
    }


def _frozen_prediction_path(task: str, seed: int, teacher: str, args) -> Path:
    model = {"dkan": "dkan", "mlp_l": "mlp_l", "bkan_parameter_mean": "bkan"}[teacher]
    return (
        args.matched_root
        / "predictions"
        / f"seed_{seed}"
        / task
        / model
        / "test_predictions.csv"
    )


def _replay_audit(
    task: str,
    teacher: str,
    test: pd.DataFrame,
    actual: np.ndarray,
    prediction: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    path = _frozen_prediction_path(task, args.seed, teacher, args)
    if not path.is_file():
        raise FileNotFoundError(path)
    frozen = pd.read_csv(path)
    frozen_actual = frozen["actual_model_space"].to_numpy(dtype=np.float64)
    frozen_prediction = frozen["prediction_model_space"].to_numpy(dtype=np.float64)
    frozen_scale = 1.0
    if task == "Capacitance":
        usable = np.isfinite(actual) & np.isfinite(frozen_actual) & (np.abs(frozen_actual) > 0.0)
        if not np.any(usable):
            raise RuntimeError(f"Cannot infer frozen capacitance scale: {path}")
        ratio = float(np.median(actual[usable] / frozen_actual[usable]))
        if np.isclose(ratio, CAPACITANCE_LEGACY_SCALE, rtol=1e-7, atol=0.0):
            frozen_scale = CAPACITANCE_LEGACY_SCALE
        elif not np.isclose(ratio, 1.0, rtol=1e-7, atol=0.0):
            raise RuntimeError(
                f"Unregistered frozen capacitance scale mismatch: ratio={ratio:.12g}; {path}"
            )
        frozen_actual = frozen_actual * frozen_scale
        frozen_prediction = frozen_prediction * frozen_scale
    actual_scale = max(float(np.max(np.abs(actual))), np.finfo(np.float64).tiny)
    actual_atol = 100.0 * np.finfo(np.float64).eps * actual_scale
    if frozen_actual.shape != actual.shape or not np.allclose(
        frozen_actual, actual, rtol=1e-7, atol=actual_atol
    ):
        raise RuntimeError(f"Frozen test rows do not match regenerated split: {path}")
    difference = prediction - frozen_prediction
    max_abs = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    prediction_scale = max(
        float(np.max(np.abs(frozen_prediction))), np.finfo(np.float64).tiny
    )
    tolerance = max(
        100.0 * np.finfo(np.float64).eps * prediction_scale,
        1e-5 * prediction_scale,
    )
    # The BKAN comparison file contains an MC predictive mean, while this audit
    # deliberately uses variational parameter means; their difference is
    # measured but is not a replay failure.
    status = "reference_difference_only" if teacher == "bkan_parameter_mean" else "pass"
    if teacher != "bkan_parameter_mean" and max_abs > tolerance:
        status = "mismatch_allowed" if args.allow_teacher_replay_mismatch else "fail"
        if status == "fail":
            raise RuntimeError(
                f"{teacher} replay mismatch for {task}: max_abs={max_abs:.6g}, "
                f"tolerance={tolerance:.6g}"
            )
    return {
        "frozen_test_prediction": str(path),
        "replay_status": status,
        "replay_max_abs_error": max_abs,
        "replay_rmse": rmse,
        "replay_tolerance": tolerance,
        "frozen_scale_applied": frozen_scale,
        "test_rows": len(test),
    }


def _pipeline(family: str, budget: int, alpha: float) -> Pipeline:
    if family == "polynomial":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=budget, include_bias=False)),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
    if family == "additive_spline":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "spline",
                    SplineTransformer(n_knots=budget, degree=3, include_bias=False),
                ),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
    raise ValueError(family)


def _metrics(reference: np.ndarray, prediction: np.ndarray, prefix: str) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(valid):
        return {
            f"{prefix}_rmse": np.nan,
            f"{prefix}_r2": np.nan,
            f"{prefix}_p95_abs_error": np.nan,
            f"{prefix}_max_abs_error": np.nan,
        }
    error = prediction[valid] - reference[valid]
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(error * error))),
        f"{prefix}_r2": float(r2_score(reference[valid], prediction[valid])),
        f"{prefix}_p95_abs_error": float(np.percentile(np.abs(error), 95)),
        f"{prefix}_max_abs_error": float(np.max(np.abs(error))),
    }


def _dense_design(train: pd.DataFrame, inputs: tuple[str, ...], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dense = np.empty((n, len(inputs)), dtype=np.float64)
    for index, name in enumerate(inputs):
        low = float(train[name].min())
        high = float(train[name].max())
        dense[:, index] = rng.uniform(low, high, n) if high > low else low
    return dense


def _derivative_finite_rate(
    evaluator: Callable[[dict, np.ndarray], np.ndarray],
    payload: dict,
    train: pd.DataFrame,
    inputs: tuple[str, ...],
    seed: int,
) -> float:
    sample = train.sample(n=min(128, len(train)), random_state=seed)
    raw = sample[list(inputs)].to_numpy(dtype=np.float64)
    axis = 0
    low = float(train[inputs[axis]].min())
    high = float(train[inputs[axis]].max())
    step = max((high - low) * 1e-5, 1e-12)
    left = raw.copy()
    right = raw.copy()
    left[:, axis] = np.maximum(low, left[:, axis] - step)
    right[:, axis] = np.minimum(high, right[:, axis] + step)
    denominator = right[:, axis] - left[:, axis]
    derivative = (evaluator(payload, right) - evaluator(payload, left)) / denominator
    return float(np.isfinite(derivative).mean())


def _select_and_export(
    family: str,
    budget: int,
    alphas: tuple[float, ...],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    inputs: tuple[str, ...],
    export_dir: Path,
) -> tuple[Pipeline, dict, Callable, dict]:
    export_dir = export_dir.resolve()
    candidates = []
    for alpha in alphas:
        model = _pipeline(family, budget, alpha)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_time = time.perf_counter() - start
        prediction = model.predict(x_validation).reshape(-1)
        rmse = float(np.sqrt(np.mean((prediction - y_validation) ** 2)))
        candidates.append((rmse, alpha, fit_time, model))
    validation_rmse, selected_alpha, fit_time, model = min(
        candidates, key=lambda row: (row[0], row[1])
    )
    if family == "polynomial":
        payload = _poly_payload(model, inputs)
        payload["family"] = f"poly_degree_{budget}_ridge"
        evaluator = _eval_poly
        writer = _write_poly_va
        terms = int(len(payload["coefficients"]) + 1)
    else:
        payload = _spline_payload(model, inputs)
        payload["family"] = f"additive_cubic_spline_knots_{budget}_ridge"
        evaluator = _eval_spline
        writer = _write_spline_va
        terms = int(1 + np.asarray(model.named_steps["ridge"].coef_).size)
    payload.update(
        {
            "selected_alpha": float(selected_alpha),
            "selection": "minimum validation RMSE; tie broken by smaller alpha",
            "fit_rows": int(len(x_train)),
            "validation_rows": int(len(x_validation)),
        }
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "formula.json"
    va_path = export_dir / "formula.va"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    writer(payload, va_path)
    core = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    metadata = {
        "selected_alpha": float(selected_alpha),
        "validation_source_rmse": validation_rmse,
        "fit_time_selected_candidate_s": fit_time,
        "candidate_fit_time_total_s": float(sum(row[2] for row in candidates)),
        "terms": terms,
        "formula_chars": len(core),
        "veriloga_lines": len(va_path.read_text(encoding="utf-8").splitlines()),
        "json_path": _artifact_path(json_path),
        "veriloga_path": _artifact_path(va_path),
    }
    return model, payload, evaluator, metadata


def mark_pareto(
    frame: pd.DataFrame,
    error_column: str = "formula_to_tcad_rmse",
    complexity_column: str = "terms",
) -> pd.Series:
    """Return a minimization Pareto flag, grouped independently by task."""
    flags = pd.Series(False, index=frame.index, dtype=bool)
    for _, group in frame.groupby("task", sort=False):
        error = group[error_column].to_numpy(dtype=np.float64)
        complexity = group[complexity_column].to_numpy(dtype=np.float64)
        for local_index, frame_index in enumerate(group.index):
            dominated = (
                (error <= error[local_index])
                & (complexity <= complexity[local_index])
                & ((error < error[local_index]) | (complexity < complexity[local_index]))
            )
            flags.loc[frame_index] = not bool(np.any(dominated))
    return flags


def _write_report(output: Path, rows: pd.DataFrame, teachers: pd.DataFrame) -> None:
    lines = [
        "# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto",
        "",
        "All export students and TCAD-direct controls use identical training rows. "
        "Validation selects ridge alpha at each fixed complexity budget; calibration is "
        "untouched and test is evaluated only after freezing.",
        "",
        "Scope: these common polynomial/additive-spline KAN-free response-surface exports "
        "isolate teacher choice under identical student families. They do not rerun or "
        "replace the task-specific mechanism-gated symbolic formulas in the main audit.",
        "",
        "Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 "
        "and emitted an estimator-version warning for scaler objects serialized under "
        "1.9.0. Reproduction should use the recorded environment or regenerate teachers.",
        "",
        "## Teacher audit",
        "",
        "| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for row in teachers.to_dict(orient="records"):
        lines.append(
            f"| {row['task']} | {row['teacher']} | {row['teacher_to_tcad_rmse']:.6g} | "
            f"{row['replay_status']} | {row['replay_max_abs_error']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## RMSE/term Pareto front",
            "",
            "| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    front = rows.loc[rows["pareto_rmse_terms"]].sort_values(
        ["task", "terms", "formula_to_tcad_rmse"]
    )
    for row in front.to_dict(orient="records"):
        source_rmse = row["formula_to_network_rmse"]
        source_text = "NA" if not np.isfinite(source_rmse) else f"{source_rmse:.6g}"
        lines.append(
            f"| {row['task']} | {row['source']} | {row['family']} | {row['budget']} | "
            f"{row['terms']} | {row['formula_to_tcad_rmse']:.6g} | "
            f"{row['formula_to_tcad_r2']:.6g} | {source_text} |"
        )
    lines.extend(
        [
            "",
            "The BKAN replay row compares a variational-parameter-mean deterministic KAN "
            "against the historical MC predictive mean and is therefore a reference "
            "difference, not a deterministic replay test. DKAN and MLP replay rows must pass.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    teacher_rows: list[dict] = []
    manifest_frames: list[pd.DataFrame] = []
    checkpoint_records: dict[str, dict] = {}

    for task in args.tasks:
        frame, inputs, shared_spec, baseline_spec = _load_task(task, args)
        partitions = shared.split_task_dataframe(
            frame, shared_spec, args.seed, fractions=tuple(args.split_fractions)
        )
        train, validation, calibration, test = partitions
        manifest_frames.append(
            _split_manifest(task, args.seed, shared_spec, partitions)
        )
        actual = {
            "train": _target(train, baseline_spec),
            "validation": _target(validation, baseline_spec),
            "test": _target(test, baseline_spec),
        }
        x = {
            "train": train[list(inputs)].to_numpy(dtype=np.float64),
            "validation": validation[list(inputs)].to_numpy(dtype=np.float64),
            "test": test[list(inputs)].to_numpy(dtype=np.float64),
        }

        bkan_predict, bkan_info = _load_bkan_parameter_mean(
            task, train, inputs, shared_spec, args
        )
        dkan_predict, dkan_info = _fit_dkan_teacher(
            train, validation, inputs, baseline_spec, args.seed, args.kan_steps
        )
        mlp_predict, mlp_info = _fit_mlp_teacher(
            train, inputs, baseline_spec, args.seed, args.mlp_epochs
        )
        predictors = {
            "bkan_parameter_mean": bkan_predict,
            "dkan": dkan_predict,
            "mlp_l": mlp_predict,
        }
        info_by_teacher = {
            "bkan_parameter_mean": bkan_info,
            "dkan": dkan_info,
            "mlp_l": mlp_info,
        }
        teacher_targets: dict[str, dict[str, np.ndarray]] = {}
        for teacher, predict in predictors.items():
            values = {
                "train": predict(train),
                "validation": predict(validation),
                "test": predict(test),
            }
            teacher_targets[teacher] = values
            replay = _replay_audit(
                task, teacher, test, actual["test"], values["test"], args
            )
            row = {
                "task": task,
                "seed": args.seed,
                "teacher": teacher,
                **_metrics(actual["test"], values["test"], "teacher_to_tcad"),
                **info_by_teacher[teacher],
                **replay,
            }
            teacher_rows.append(row)
            if teacher == "bkan_parameter_mean":
                checkpoint_records[task] = {
                    "path": bkan_info["checkpoint"],
                    "sha256": bkan_info["checkpoint_sha256"],
                }
            print(
                f"[{task} {teacher}] teacher-to-TCAD RMSE="
                f"{row['teacher_to_tcad_rmse']:.6g}; replay={row['replay_status']}"
            )

        source_targets = {
            **teacher_targets,
            "tcad_direct": actual,
        }
        dense = _dense_design(train, inputs, args.dense_points, args.seed)
        for source in SOURCES:
            for family, budgets in (
                ("polynomial", args.poly_degrees),
                ("additive_spline", args.spline_knots),
            ):
                for budget in budgets:
                    export_dir = args.output / "exports" / task / source / f"{family}_{budget}"
                    model, payload, evaluator, export_info = _select_and_export(
                        family,
                        int(budget),
                        tuple(args.ridge_alphas),
                        x["train"],
                        source_targets[source]["train"],
                        x["validation"],
                        source_targets[source]["validation"],
                        inputs,
                        export_dir,
                    )
                    prediction = evaluator(payload, x["test"])
                    sklearn_prediction = model.predict(x["test"]).reshape(-1)
                    replay_error = float(np.max(np.abs(prediction - sklearn_prediction)))
                    dense_prediction = evaluator(payload, dense)
                    latency_batch = dense[: min(1000, len(dense))]
                    start = time.perf_counter()
                    for _ in range(args.latency_repeats):
                        evaluator(payload, latency_batch)
                    latency = (time.perf_counter() - start) / (
                        args.latency_repeats * len(latency_batch)
                    )
                    network_metrics = (
                        {
                            "formula_to_network_rmse": np.nan,
                            "formula_to_network_r2": np.nan,
                            "formula_to_network_p95_abs_error": np.nan,
                            "formula_to_network_max_abs_error": np.nan,
                        }
                        if source == "tcad_direct"
                        else _metrics(
                            source_targets[source]["test"], prediction,
                            "formula_to_network",
                        )
                    )
                    metric_rows.append(
                        {
                            "task": task,
                            "seed": args.seed,
                            "source": source,
                            "source_kind": "direct_tcad" if source == "tcad_direct" else "teacher_distillation",
                            "family": family,
                            "budget": int(budget),
                            "fit_rows": len(train),
                            "validation_rows": len(validation),
                            "calibration_rows_untouched": len(calibration),
                            "test_rows": len(test),
                            **export_info,
                            **network_metrics,
                            **_metrics(actual["test"], prediction, "formula_to_tcad"),
                            "test_finite_rate": float(np.isfinite(prediction).mean()),
                            "dense_train_envelope_points": len(dense),
                            "dense_finite_rate": float(np.isfinite(dense_prediction).mean()),
                            "swept_axis_derivative_finite_rate": _derivative_finite_rate(
                                evaluator, payload, train, inputs, args.seed
                            ),
                            "independent_json_replay_max_abs_error": replay_error,
                            "latency_per_point_s": latency,
                            "calibration_used": False,
                            "test_used_for_selection": False,
                        }
                    )
                    print(
                        f"[{task} {source} {family}={budget}] "
                        f"TCAD RMSE={metric_rows[-1]['formula_to_tcad_rmse']:.6g}; "
                        f"terms={metric_rows[-1]['terms']}"
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics["pareto_rmse_terms"] = mark_pareto(metrics)
    metrics["pareto_rmse_chars"] = mark_pareto(
        metrics, complexity_column="formula_chars"
    )
    teachers = pd.DataFrame(teacher_rows)
    manifest = pd.concat(manifest_frames, ignore_index=True)
    metrics.to_csv(args.output / "metrics_by_export.csv", index=False)
    summary_columns = [
        "task", "source", "source_kind", "family", "budget", "selected_alpha",
        "terms", "formula_chars", "latency_per_point_s", "test_finite_rate",
        "dense_finite_rate", "swept_axis_derivative_finite_rate",
        "formula_to_network_rmse", "formula_to_network_r2",
        "formula_to_tcad_rmse", "formula_to_tcad_r2",
        "pareto_rmse_terms", "pareto_rmse_chars", "json_path", "veriloga_path",
    ]
    metrics[summary_columns].sort_values(
        ["task", "source", "family", "budget"]
    ).to_csv(args.output / "summary.csv", index=False)
    metrics.loc[metrics["pareto_rmse_terms"]].sort_values(
        ["task", "terms", "formula_to_tcad_rmse"]
    ).to_csv(args.output / "pareto_front_rmse_terms.csv", index=False)
    teachers.to_csv(args.output / "teacher_audit.csv", index=False)
    manifest.to_csv(args.output / "split_manifest.csv", index=False)
    protocol = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "data-budget-matched multi-teacher symbolic/direct-export Pareto",
        "scope_limitation": (
            "Common polynomial and additive-spline KAN-free response-surface students "
            "isolate teacher source. This audit does not rerun the task-specific "
            "mechanism-gated symbolic student."
        ),
        "tasks": args.tasks,
        "seed": args.seed,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "checkpoint_load_warning": (
                "BKAN checkpoint scaler objects report serialization under "
                "scikit-learn 1.9.0; this run uses the version recorded above."
            ),
        },
        "split_fractions": list(args.split_fractions),
        "partition_policy": {
            "teacher_fit": "train groups only",
            "student_and_direct_fit": "train groups only",
            "ridge_selection": "validation groups only, separately at each fixed complexity budget",
            "calibration": "untouched",
            "test": "evaluated once after teacher/student/direct model is frozen",
        },
        "teachers": {
            "bkan_parameter_mean": {
                "construction": "copy VI variational means to deterministic KAN; retain mean output channel",
                "checkpoints": checkpoint_records,
            },
            "dkan": {
                "width": ["n_inputs", 8, 1], "grid": 8, "spline_order": 3,
                "steps": args.kan_steps, "optimizer": "Adam", "learning_rate": 0.002,
                "replay_reference": "matched_grouped_comparison frozen test predictions",
            },
            "mlp_l": {
                "hidden": list(MLP_ARCHES["mlp_l"]), "epochs": args.mlp_epochs,
                "optimizer": "Adam", "learning_rate": 0.002,
                "regularization": {"lambda": 0.001, "l1": 1.0, "entropy": 2.0},
                "replay_reference": "matched_grouped_comparison frozen test predictions",
            },
        },
        "export_families": {
            "polynomial": {"degrees": list(args.poly_degrees)},
            "additive_spline": {
                "degree": 3, "n_knots": list(args.spline_knots), "interaction_terms": False,
            },
            "ridge_alphas": list(args.ridge_alphas),
        },
        "source_data": {
            "main": str(args.data),
            "capacitance": str(args.capacitance_data),
        },
        "outputs": {
            "per_export_metrics": "metrics_by_export.csv",
            "summary": "summary.csv",
            "pareto": "pareto_front_rmse_terms.csv",
            "teacher_audit": "teacher_audit.csv",
            "split_manifest": "split_manifest.csv",
            "exports": "exports/<task>/<source>/<family_budget>/formula.{json,va}",
        },
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(args.output, metrics, teachers)


def main() -> int:
    args = parse_args()
    validate_args(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
