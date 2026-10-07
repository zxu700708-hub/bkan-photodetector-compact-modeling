"""Physics-anchored sparse exports for dark current, net photocurrent, and Q.

This is the frozen post hoc development implementation used by the manuscript.
It reuses the registered grouped splits and source data, while deployment
validation remains a separate, hash-bound stage:

* log-current formulas separate the response at a reference voltage from a
  C2 voltage correction that is exactly zero at that reference;
* every declared device-condition input retains an explicit main effect;
* terminal charge preserves Q(0, p) = 0 and analytic dQ/dV;
* compensated deletion refits all remaining coefficients after every removal;
* validation checks values, complete-response tails, and physical derivatives.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = ROOT / "artifacts/experiments/balanced_symbolic_20260917"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "bkan"), str(BASE_DIR)]

import run_balanced_extraction as base  # noqa: E402
import run_symbolic_structure_guidance_ablation as feature_api  # noqa: E402
from device_modeling.photodetector.task_config import TASKS  # noqa: E402


CURRENT_BUDGETS = (8, 10, 12, 16, 20, 24, 30, 36, 42, 54)
CHARGE_BUDGETS = (8, 10, 12, 16, 20, 24, 30, 42, 60, 84)
ALPHAS = (1e-10, 1e-8, 1e-6, 1e-4)
TEACHER_WEIGHTS = (0.0, 0.05, 0.25)
CURRENT_DERIVATIVE_WEIGHTS = (0.1, 0.3, 1.0)
CHARGE_DERIVATIVE_WEIGHTS = (3.0, 10.0, 30.0)
VALIDATION_SLACK = 1.05


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_spec(task: str):
    return TASKS["dark_current" if task == "I_dark" else "photo_current"]


def group_labels(task: str, frame: pd.DataFrame) -> np.ndarray:
    if task == "Q_terminal":
        return frame["production_source_row"].astype(str).to_numpy()
    return (
        feature_api.single.shared.task_group_labels(frame, task_spec(task))
        .astype(str)
        .to_numpy()
    )


def grouped_axis_derivative(
    values: np.ndarray,
    axis: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    result = np.empty_like(values)
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        order = rows[np.argsort(axis[rows])]
        if len(order) < 3:
            raise ValueError("At least three axis points are required per response group")
        result[order] = np.gradient(values[order], axis[order], edge_order=2)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Non-finite grouped derivative target")
    return result


def anchored_current_matrix(
    frame: pd.DataFrame,
    inputs: list[str],
    library: dict,
    reference_voltage: float,
) -> np.ndarray:
    raw = frame[inputs].to_numpy(dtype=np.float64)
    matrix = feature_api.candidate_matrix(raw, library)
    reference = frame[inputs].copy()
    reference[inputs[0]] = reference_voltage
    reference_matrix = feature_api.candidate_matrix(
        reference.to_numpy(dtype=np.float64), library
    )
    axis = inputs[0]
    for index, feature in enumerate(library["features"]):
        if any(atom["input"] == axis for atom in feature["atoms"]):
            matrix[:, index] -= reference_matrix[:, index]
    return np.column_stack([np.ones(len(frame)), matrix])


def current_derivative_matrix(
    frame: pd.DataFrame,
    inputs: list[str],
    library: dict,
    reference_voltage: float,
    step: float = 1e-5,
) -> np.ndarray:
    high = frame.copy()
    low = frame.copy()
    high[inputs[0]] += step
    low[inputs[0]] -= step
    return (
        anchored_current_matrix(high, inputs, library, reference_voltage)
        - anchored_current_matrix(low, inputs, library, reference_voltage)
    ) / (2.0 * step)


def current_mandatory_columns(library: dict, inputs: list[str]) -> tuple[int, ...]:
    mandatory = [0]
    features = library["features"]
    axis = inputs[0]
    for name in inputs[1:]:
        matches = [
            index + 1
            for index, feature in enumerate(features)
            if len(feature["atoms"]) == 1
            and feature["atoms"][0]["kind"] == "power"
            and feature["atoms"][0]["input"] == name
            and int(feature["atoms"][0]["value"]) == 1
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one main effect for {name}, found {matches}")
        mandatory.extend(matches)
    linear_axis = [
        index + 1
        for index, feature in enumerate(features)
        if len(feature["atoms"]) == 1
        and feature["atoms"][0]["kind"] == "power"
        and feature["atoms"][0]["input"] == axis
        and int(feature["atoms"][0]["value"]) == 1
    ]
    if len(linear_axis) != 1:
        raise RuntimeError("Expected one linear voltage feature")
    mandatory.extend(linear_axis)
    return tuple(sorted(set(mandatory)))


def charge_mandatory_columns(model) -> tuple[int, ...]:
    powers = np.asarray(model.parameter_powers, dtype=int)
    n_parameter_terms = len(powers)
    mandatory = [0, n_parameter_terms, 2 * n_parameter_terms]
    for parameter_index in range(5):
        matches = np.flatnonzero(powers[:, parameter_index] == 1)
        matches = [int(index) for index in matches if int(powers[index].sum()) == 1]
        if len(matches) != 1:
            raise RuntimeError("Charge basis lacks a unique linear parameter effect")
        mandatory.append(matches[0])
    return tuple(sorted(set(mandatory)))


def prepare_current(task: str, seed: int, output: Path) -> dict:
    data = base.prepare_current(task, seed, output)
    reference_voltage = float(data["frames"]["train"][data["inputs"][0]].min())
    data["reference_voltage"] = reference_voltage
    data["X"] = {
        name: anchored_current_matrix(
            frame, data["inputs"], data["library"], reference_voltage
        )
        for name, frame in data["frames"].items()
    }
    data["D"] = {
        name: current_derivative_matrix(
            frame, data["inputs"], data["library"], reference_voltage
        )
        for name, frame in data["frames"].items()
    }
    data["labels"] = {
        name: group_labels(task, frame) for name, frame in data["frames"].items()
    }
    data["dy"] = {
        name: grouped_axis_derivative(
            data["ys"][name],
            frame[data["inputs"][0]].to_numpy(dtype=np.float64),
            data["labels"][name],
        )
        for name, frame in data["frames"].items()
    }
    data["mandatory"] = current_mandatory_columns(data["library"], data["inputs"])
    return data


def prepare_charge(seed: int, output: Path) -> dict:
    data = base.prepare_charge(seed, output)
    data["labels"] = {
        name: group_labels("Q_terminal", frame)
        for name, frame in data["frames"].items()
    }
    data["mandatory"] = charge_mandatory_columns(data["model"])
    return data


def current_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    derivative_reference: np.ndarray,
    derivative_prediction: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = prediction - reference
    relative = np.abs(np.power(10.0, np.clip(residual, -100.0, 100.0)) - 1.0)
    group_rmse = [
        float(np.sqrt(np.mean(np.square(residual[labels == label]))))
        for label in np.unique(labels)
    ]
    current = np.power(10.0, reference)
    predicted_current = np.power(10.0, prediction)
    conductance = math.log(10.0) * current * derivative_reference
    predicted_conductance = (
        math.log(10.0) * predicted_current * derivative_prediction
    )
    conductance_scale = max(
        float(np.sqrt(np.mean(np.square(conductance)))), 1e-30
    )
    floor = max(float(np.quantile(np.abs(conductance), 0.95)) * 0.01, 1e-30)
    conductance_relative = np.abs(predicted_conductance - conductance) / np.maximum(
        np.abs(conductance), floor
    )
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "relative_p95": float(np.quantile(relative, 0.95)),
        "group_rmse_p95": float(np.quantile(group_rmse, 0.95)),
        "conductance_nrmse": float(
            np.sqrt(np.mean(np.square(predicted_conductance - conductance)))
            / conductance_scale
        ),
        "conductance_relative_p95": float(np.quantile(conductance_relative, 0.95)),
    }


def charge_metrics(data: dict, split: str, prediction, derivative) -> dict[str, float]:
    metrics = base.physical_metrics(
        "Q_terminal",
        data["ys"][split],
        prediction,
        data["dy"][split],
        derivative,
        data["qfloor"],
        data["cfloor"],
    )
    residual = np.asarray(prediction) - data["ys"][split]
    labels = data["labels"][split]
    group_rmse = [
        float(np.sqrt(np.mean(np.square(residual[labels == label]))))
        for label in np.unique(labels)
    ]
    metrics["group_rmse_p95"] = float(np.quantile(group_rmse, 0.95))
    return metrics


def current_formula_eval(payload: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    formula = payload["current_formula"]
    inputs = payload["input_order"]
    library = {
        key: formula[key]
        for key in (
            "input_order",
            "input_scalers",
            "family",
            "axis_names",
            "condition_names",
            "interior_knots",
            "features",
        )
    }
    matrix = anchored_current_matrix(
        frame, inputs, library, float(formula["reference_voltage"])
    )
    derivative_matrix = current_derivative_matrix(
        frame, inputs, library, float(formula["reference_voltage"])
    )
    coefficient = np.asarray(formula["coefficients"], dtype=np.float64)
    return matrix @ coefficient, derivative_matrix @ coefficient


def make_payload(data: dict, coefficient: np.ndarray, config: dict) -> dict:
    common = {
        "schema": "physics_balanced_sparse_v1",
        "task": data["task"],
        "seed": data["seed"],
        "input_order": data["inputs"],
        "coefficient_count": int(np.count_nonzero(coefficient)),
        "config": config,
        "selection_data": "validation_only",
        "test_used_for_selection": False,
        "uses_kan_at_inference": False,
    }
    if data["task"] == "Q_terminal":
        common["physical_contract"] = {
            "reference": "Q(0,p)=0 exactly",
            "derivative": "analytic dQ/dV",
            "input_coverage": "one explicit linear sensitivity per device parameter",
        }
        model = data["model"].to_dict()
        common["charge_basis"] = {key: value for key, value in model.items() if key != "coefficients"}
        common["terms"] = [
            {"index": int(index), "coefficient": float(coefficient[index])}
            for index in np.flatnonzero(coefficient)
        ]
    else:
        library = data["library"]
        active_features = np.flatnonzero(coefficient[1:])
        compact_library = {
            **{key: value for key, value in library.items() if key != "features"},
            "features": [library["features"][int(index)] for index in active_features],
        }
        common["output_space"] = "log10(I/A)"
        common["physical_contract"] = {
            "decomposition": "reference-condition amplitude plus C2 bias correction",
            "reference_voltage_V": float(data["reference_voltage"]),
            "bias_correction_at_reference": 0.0,
            "input_coverage": "explicit main effect for every declared device parameter",
        }
        common["current_formula"] = {
            **compact_library,
            "reference_voltage": float(data["reference_voltage"]),
            "coefficients": [
                float(coefficient[0]),
                *[float(coefficient[int(index) + 1]) for index in active_features],
            ],
            "active_columns": [0, *[int(index) + 1 for index in active_features]],
            "original_feature_indices": [int(index) for index in active_features],
        }
    return common


def baseline_predictions(data: dict, split: str):
    if data["task"] == "Q_terminal":
        return data["baseline"][split], data["baseline_derivative"][split]
    frame = data["frames"][split]
    prediction = data["baseline"][split]
    axis = data["inputs"][0]
    high = frame.copy()
    low = frame.copy()
    high[axis] += 1e-5
    low[axis] -= 1e-5
    derivative = (
        base.old_eval(data["baseline_payload"], high[data["inputs"]].to_numpy())
        - base.old_eval(data["baseline_payload"], low[data["inputs"]].to_numpy())
    ) / 2e-5
    return prediction, derivative


def metrics(data: dict, split: str, prediction, derivative) -> dict[str, float]:
    if data["task"] == "Q_terminal":
        return charge_metrics(data, split, prediction, derivative)
    return current_metrics(
        data["ys"][split],
        prediction,
        data["dy"][split],
        derivative,
        data["labels"][split],
    )


def numerical_audit(data: dict, payload: dict) -> dict:
    dense = data["dense"]
    if data["task"] == "Q_terminal":
        prediction, derivative = base.charge_sparse_eval(payload, dense)
        zero = dense.iloc[:1000].copy()
        zero["bias_v"] = 0.0
        zero_error = float(np.max(np.abs(base.charge_sparse_eval(payload, zero)[0])))
        high = dense.copy()
        low = dense.copy()
        high["bias_v"] += 1e-5
        low["bias_v"] -= 1e-5
        finite_difference = (
            base.charge_sparse_eval(payload, high)[0]
            - base.charge_sparse_eval(payload, low)[0]
        ) / 2e-5
        derivative_error = float(
            np.max(
                np.abs(derivative - finite_difference)
                / np.maximum(np.abs(derivative), 1e-30)
            )
        )
        passed = bool(
            np.isfinite(prediction).all()
            and np.isfinite(derivative).all()
            and float(np.min(derivative)) > 0.0
            and zero_error < 1e-26
            and derivative_error < 1e-3
        )
        return {
            "passed": passed,
            "dense_points": len(dense),
            "minimum_dqdv_F": float(np.min(derivative)),
            "zero_reference_max_abs_C": zero_error,
            "derivative_fd_max_relative": derivative_error,
        }
    prediction, derivative = current_formula_eval(payload, dense)
    train_direction = float(np.median(data["dy"]["train"]))
    tolerance = 1e-8
    violation = derivative > tolerance if train_direction < 0 else derivative < -tolerance
    return {
        "passed": bool(
            np.isfinite(prediction).all()
            and np.isfinite(derivative).all()
            and np.isfinite(np.power(10.0, prediction)).all()
            and float(np.mean(violation)) <= 0.005
        ),
        "dense_points": len(dense),
        "expected_derivative_sign": "nonpositive" if train_direction < 0 else "nonnegative",
        "derivative_sign_violation_fraction": float(np.mean(violation)),
    }


def fit_fixed_config(data: dict, config: dict) -> np.ndarray:
    """Reproduce one registered candidate without consulting validation/test rows."""
    is_charge = data["task"] == "Q_terminal"
    value_scale = (
        data["model"].q_scale_C
        if is_charge
        else max(float(np.std(data["ys"]["train"])), 1e-12)
    )
    value_offset = 0.0 if is_charge else float(np.mean(data["ys"]["train"]))
    normalized_value = (data["ys"]["train"] - value_offset) / value_scale
    normalized_teacher = (data["teacher"] - value_offset) / value_scale
    if is_charge:
        derivative_scale = data["model"].c_scale_F
        derivative_target = data["dy"]["train"] / derivative_scale
        derivative_design = data["D"]["train"]
    else:
        derivative_scale = max(float(np.std(data["dy"]["train"])), 1e-8)
        derivative_target = data["dy"]["train"] / derivative_scale
        derivative_design = data["D"]["train"] * value_scale / derivative_scale
    blocks = [data["X"]["train"]]
    targets = [normalized_value]
    teacher_weight = float(config["teacher_weight"])
    derivative_weight = float(config["derivative_weight"])
    if teacher_weight:
        blocks.append(math.sqrt(teacher_weight) * data["X"]["train"])
        targets.append(math.sqrt(teacher_weight) * normalized_teacher)
    blocks.append(math.sqrt(derivative_weight) * derivative_design)
    targets.append(math.sqrt(derivative_weight) * derivative_target)
    path = base.elimination_path(
        np.vstack(blocks),
        np.concatenate(targets),
        float(config["ridge_alpha"]),
        [int(config["budget"])],
        data["mandatory"],
    )
    coefficient = path[int(config["budget"])].copy()
    if not is_charge:
        coefficient *= value_scale
        coefficient[0] += value_offset
    return coefficient


def run_one(task: str, seed: int, output_root: Path) -> dict:
    destination = output_root / task / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    data = prepare_charge(seed, destination) if task == "Q_terminal" else prepare_current(task, seed, destination)
    is_charge = task == "Q_terminal"
    baseline_prediction, baseline_derivative = baseline_predictions(data, "validation")
    baseline_validation = metrics(
        data, "validation", baseline_prediction, baseline_derivative
    )
    value_scale = (
        data["model"].q_scale_C
        if is_charge
        else max(float(np.std(data["ys"]["train"])), 1e-12)
    )
    value_offset = 0.0 if is_charge else float(np.mean(data["ys"]["train"]))
    normalized_value = (data["ys"]["train"] - value_offset) / value_scale
    normalized_teacher = (data["teacher"] - value_offset) / value_scale
    if is_charge:
        derivative_scale = data["model"].c_scale_F
        derivative_target = data["dy"]["train"] / derivative_scale
        derivative_design = data["D"]["train"]
        budgets = CHARGE_BUDGETS
        derivative_weights = CHARGE_DERIVATIVE_WEIGHTS
    else:
        derivative_scale = max(float(np.std(data["dy"]["train"])), 1e-8)
        derivative_target = data["dy"]["train"] / derivative_scale
        derivative_design = data["D"]["train"] * value_scale / derivative_scale
        budgets = CURRENT_BUDGETS
        derivative_weights = CURRENT_DERIVATIVE_WEIGHTS

    rows: list[dict] = []
    coefficients: dict[int, np.ndarray] = {}
    for teacher_weight in TEACHER_WEIGHTS:
        for derivative_weight in derivative_weights:
            blocks = [data["X"]["train"]]
            targets = [normalized_value]
            if teacher_weight:
                blocks.append(math.sqrt(teacher_weight) * data["X"]["train"])
                targets.append(math.sqrt(teacher_weight) * normalized_teacher)
            blocks.append(math.sqrt(derivative_weight) * derivative_design)
            targets.append(math.sqrt(derivative_weight) * derivative_target)
            design = np.vstack(blocks)
            target = np.concatenate(targets)
            allowed_budgets = [value for value in budgets if value <= design.shape[1]]
            for alpha in ALPHAS:
                path = base.elimination_path(
                    design,
                    target,
                    alpha,
                    allowed_budgets,
                    data["mandatory"],
                )
                for budget, coefficient in path.items():
                    coefficient = coefficient.copy()
                    if not is_charge:
                        coefficient *= value_scale
                        coefficient[0] += value_offset
                    prediction = data["X"]["validation"] @ coefficient
                    if is_charge:
                        prediction *= value_scale
                        derivative_prediction = (
                            data["D"]["validation"]
                            @ coefficient
                            * data["model"].c_scale_F
                        )
                    else:
                        derivative_prediction = data["D"]["validation"] @ coefficient
                    candidate_metrics = metrics(
                        data, "validation", prediction, derivative_prediction
                    )
                    ratios = {
                        key: candidate_metrics[key] / max(value, 1e-30)
                        for key, value in baseline_validation.items()
                    }
                    candidate_id = len(rows)
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "teacher_weight": teacher_weight,
                            "derivative_weight": derivative_weight,
                            "ridge_alpha": alpha,
                            "budget": budget,
                            **{f"validation_{key}": value for key, value in candidate_metrics.items()},
                            **{f"validation_{key}_ratio": value for key, value in ratios.items()},
                            "worst_validation_ratio": max(ratios.values()),
                            "eligible": bool(max(ratios.values()) <= VALIDATION_SLACK),
                        }
                    )
                    coefficients[candidate_id] = coefficient
    candidates = pd.DataFrame(rows)
    candidates.to_csv(destination / "validation_candidates.csv", index=False)
    ranked = candidates[candidates["eligible"]].sort_values(
        ["budget", "worst_validation_ratio", "teacher_weight", "derivative_weight", "ridge_alpha"]
    )
    selected = None
    audits = []
    for row in ranked.to_dict(orient="records"):
        coefficient = coefficients[int(row["candidate_id"])]
        config = {
            key: row[key]
            for key in ("teacher_weight", "derivative_weight", "ridge_alpha", "budget")
        }
        payload = make_payload(data, coefficient, config)
        audit = numerical_audit(data, payload)
        audits.append({"candidate_id": int(row["candidate_id"]), **audit})
        if audit["passed"]:
            selected = (row, coefficient, payload, audit)
            break
    if selected is None:
        payload = {
            "schema": "unchanged_baseline",
            "task": task,
            "seed": seed,
            "coefficient_count": data["baseline_count"],
            "baseline_payload": data["baseline_payload"],
        }
        selected_kind = "fallback"
        selected_count = int(data["baseline_count"])
    else:
        row, coefficient, payload, audit = selected
        selected_kind = "physics_sparse_refit"
        selected_count = int(payload["coefficient_count"])
        payload["selection"] = {
            "rule": "minimum coefficient count with every validation metric <= 1.05 times the registered baseline",
            "baseline_validation": baseline_validation,
            "selected_candidate_id": int(row["candidate_id"]),
            "numerical_audit": audit,
            "mandatory_columns": list(data["mandatory"]),
            "test_used_for_selection": False,
        }
    dump(destination / "selected_formula.json", payload)
    frozen = json.loads((destination / "selected_formula.json").read_text(encoding="utf-8"))
    if selected is None:
        test_prediction, test_derivative = baseline_predictions(data, "test")
    elif is_charge:
        test_prediction, _ = base.charge_sparse_eval(frozen, data["frames"]["test"])
        _, test_derivative = base.charge_sparse_eval(frozen, data["cframes"]["test"])
    else:
        test_prediction, test_derivative = current_formula_eval(
            frozen, data["frames"]["test"]
        )
    baseline_test_prediction, baseline_test_derivative = baseline_predictions(data, "test")
    selected_test = metrics(data, "test", test_prediction, test_derivative)
    baseline_test = metrics(
        data, "test", baseline_test_prediction, baseline_test_derivative
    )
    result = {
        "task": task,
        "seed": seed,
        "selected_kind": selected_kind,
        "baseline_count": int(data["baseline_count"]),
        "selected_count": selected_count,
        **{f"baseline_test_{key}": value for key, value in baseline_test.items()},
        **{f"selected_test_{key}": value for key, value in selected_test.items()},
        "formula": str((destination / "selected_formula.json").relative_to(ROOT)),
    }
    if selected is not None:
        result.update(payload["config"])
        result.update(payload["selection"]["numerical_audit"])
    dump(destination / "test_metrics.json", result)
    prediction_table = data["frames"]["test"][data["inputs"]].copy()
    prediction_table["group"] = data["labels"]["test"]
    prediction_table["target"] = data["ys"]["test"]
    prediction_table["baseline"] = baseline_test_prediction
    prediction_table["selected"] = test_prediction
    prediction_table.to_csv(destination / "test_predictions.csv", index=False)
    if is_charge:
        derivative_table = data["cframes"]["test"][data["inputs"]].copy()
        derivative_table["group"] = group_labels(
            "Q_terminal", data["cframes"]["test"]
        )
        derivative_table["target_derivative"] = data["dy"]["test"]
        derivative_table["baseline_derivative"] = baseline_test_derivative
        derivative_table["selected_derivative"] = test_derivative
        derivative_table.to_csv(
            destination / "derivative_test_predictions.csv", index=False
        )
    else:
        prediction_table["target_derivative"] = data["dy"]["test"]
        prediction_table["baseline_derivative"] = baseline_test_derivative
        prediction_table["selected_derivative"] = test_derivative
        prediction_table.to_csv(destination / "test_predictions.csv", index=False)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("I_dark", "I_photo", "Q_terminal"),
        default=["I_dark", "I_photo", "Q_terminal"],
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = {
        "status": "running",
        "script_sha256": sha256(Path(__file__)),
        "seeds": args.seeds,
        "tasks": args.tasks,
        "current_budgets": CURRENT_BUDGETS,
        "charge_budgets": CHARGE_BUDGETS,
        "teacher_weights": TEACHER_WEIGHTS,
        "current_derivative_weights": CURRENT_DERIVATIVE_WEIGHTS,
        "charge_derivative_weights": CHARGE_DERIVATIVE_WEIGHTS,
        "validation_slack": VALIDATION_SLACK,
        "selection": "minimum size subject to all value, grouped-tail, and derivative metrics <= registered baseline * 1.05; dense physical audit; fallback",
        "scope": "post hoc grouped development evidence; simulator validation is recorded separately",
    }
    dump(output / "protocol.json", protocol)
    results = []
    with threadpool_limits(limits=1):
        for task in args.tasks:
            for seed in args.seeds:
                results.append(run_one(task, seed, output))
                pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
    protocol["status"] = "complete"
    dump(output / "protocol.json", protocol)


if __name__ == "__main__":
    main()
