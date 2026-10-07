"""Compare direct and BKAN-guided sparse symbolic structures on matched splits.

This development experiment keeps the paper and frozen exports untouched.  It
uses an explicit truncated-power spline tensor library that can be evaluated
without a KAN.  BKAN predictions may alter only the greedy feature ordering;
all reported formula coefficients are refit to TCAD training targets and ridge
strength is selected with TCAD validation targets.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_multi_teacher_symbolic_pareto as single  # noqa: E402
import run_multi_teacher_symbolic_pareto_10split as repeated  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts/results/symbolic_structure_guidance_ablation"
DEFAULT_NET_ROOT = ROOT / "artifacts/results/net_photocurrent_retrain"
DEFAULT_SEEDS = tuple(range(42, 52))
DEFAULT_TASKS = tuple(single.TASK_KEY)
DEFAULT_BUDGETS = (8, 12, 18, 24, 36, 54, 69)
DEFAULT_GUIDANCE = (0.0, 0.2, 1.0)
DEFAULT_ALPHAS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0)
REFERENCE_BUDGET = {
    "I_dark": 36,
    "I_photo": 69,
    "AC_Response": 12,
    "Capacitance": 24,
}
ACCEPTED_SEED42 = {
    "I_dark": {"terms": 36, "rmse": 0.01639},
    "I_photo": {"terms": 69, "rmse": 0.00348},
    "AC_Response": {"terms": 11, "rmse": 0.11271},
    "Capacitance": {"terms": 24, "rmse": 5.03197e-16},
}


@dataclass(frozen=True)
class FormulaFit:
    alpha: float
    intercept: float
    coefficients: np.ndarray
    validation_rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=single.DEFAULT_DATA)
    parser.add_argument(
        "--net-photocurrent-data",
        type=Path,
        default=DEFAULT_NET_ROOT / "device_modeling/cleaned_data.csv",
    )
    parser.add_argument(
        "--capacitance-data", type=Path, default=single.shared.DEFAULT_CAPACITANCE_DATA
    )
    parser.add_argument("--bkan-root", type=Path, default=single.DEFAULT_BKAN)
    parser.add_argument(
        "--capacitance-bkan-root", type=Path, default=single.DEFAULT_CAP_BKAN
    )
    parser.add_argument("--net-root", type=Path, default=DEFAULT_NET_ROOT)
    parser.add_argument("--legacy-audit", type=Path, default=repeated.DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--guidance-weights", nargs="+", type=float, default=list(DEFAULT_GUIDANCE)
    )
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--interior-knots", type=int, default=6)
    parser.add_argument("--dense-points", type=int, default=2000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "output",
        "data",
        "net_photocurrent_data",
        "capacitance_data",
        "bkan_root",
        "capacitance_bkan_root",
        "net_root",
        "legacy_audit",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if any(value < 2 for value in args.budgets):
        raise ValueError("Budgets include the intercept and must be at least two")
    if any(value < 0.0 or value > 1.0 for value in args.guidance_weights):
        raise ValueError("Guidance weights must lie in [0, 1]")
    if 0.0 not in args.guidance_weights:
        raise ValueError("The direct weight 0 must be included")
    if any(value < 0.0 for value in args.ridge_alphas):
        raise ValueError("Ridge alphas must be nonnegative")
    if args.interior_knots < 1:
        raise ValueError("At least one interior knot is required")
    if args.dense_points < 100:
        raise ValueError("At least 100 dense points are required")
    for path in (args.data, args.net_photocurrent_data, args.capacitance_data):
        if not path.is_file():
            raise FileNotFoundError(path)


def _task_args(args: argparse.Namespace, task: str, seed: int) -> SimpleNamespace:
    is_photo = task == "I_photo"
    return SimpleNamespace(
        data=args.net_photocurrent_data if is_photo else args.data,
        capacitance_data=args.capacitance_data,
        bkan_root=(args.net_root / "uq_repeated_grouped" if is_photo else args.bkan_root),
        capacitance_bkan_root=args.capacitance_bkan_root,
        seed=seed,
    )


def _atom_key(atom: dict) -> tuple:
    return atom["kind"], atom["input"], float(atom["value"])


def _input_atom(name: str, power: int) -> dict:
    return {"kind": "power", "input": name, "value": int(power)}


def _hinge_atom(name: str, knot: float) -> dict:
    return {"kind": "hinge3", "input": name, "value": float(knot)}


def _feature(name: str, *atoms: dict) -> dict:
    return {"name": name, "atoms": list(atoms)}


def _standardized_inputs(raw: np.ndarray, payload: dict) -> dict[str, np.ndarray]:
    names = payload["input_order"]
    return {
        name: (raw[:, index] - payload["input_scalers"][name]["mean"])
        / payload["input_scalers"][name]["scale"]
        for index, name in enumerate(names)
    }


def _atom_values(atom: dict, standardized: dict[str, np.ndarray]) -> np.ndarray:
    values = standardized[atom["input"]]
    if atom["kind"] == "power":
        return np.power(values, int(atom["value"]))
    if atom["kind"] == "hinge3":
        return np.power(np.maximum(values - float(atom["value"]), 0.0), 3)
    raise ValueError(f"Unknown atom kind: {atom['kind']}")


def _raw_feature_values(feature: dict, standardized: dict[str, np.ndarray]) -> np.ndarray:
    values = np.ones(len(next(iter(standardized.values()))), dtype=np.float64)
    for atom in feature["atoms"]:
        values *= _atom_values(atom, standardized)
    return values


def candidate_matrix(raw: np.ndarray, payload: dict) -> np.ndarray:
    standardized = _standardized_inputs(raw, payload)
    columns = []
    for feature in payload["features"]:
        values = _raw_feature_values(feature, standardized)
        columns.append((values - feature["center"]) / feature["scale"])
    matrix = np.column_stack(columns)
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("Candidate matrix contains non-finite values")
    return matrix


def build_library(
    train: pd.DataFrame,
    inputs: tuple[str, ...],
    task: str,
    interior_knots: int,
) -> dict:
    raw_train = train[list(inputs)].to_numpy(dtype=np.float64)
    means = raw_train.mean(axis=0)
    scales = raw_train.std(axis=0)
    scales[scales < 1e-12] = 1.0
    payload = {
        "input_order": list(inputs),
        "input_scalers": {
            name: {"mean": float(means[index]), "scale": float(scales[index])}
            for index, name in enumerate(inputs)
        },
    }
    standardized = _standardized_inputs(raw_train, payload)
    axis_names = [inputs[0]]
    if task == "Capacitance":
        axis_names.append("log_frequency_ghz")
    condition_names = [
        name
        for name in inputs
        if name not in axis_names and np.std(standardized[name]) >= 1e-12
    ]

    axis_atoms: dict[str, list[dict]] = {}
    knot_records: dict[str, list[float]] = {}
    quantiles = np.linspace(0.0, 1.0, interior_knots + 2)[1:-1]
    for axis in axis_names:
        knots = np.unique(np.quantile(standardized[axis], quantiles)).astype(np.float64)
        knot_records[axis] = [float(value) for value in knots]
        axis_atoms[axis] = [
            _input_atom(axis, 1),
            _input_atom(axis, 2),
            _input_atom(axis, 3),
            *[_hinge_atom(axis, value) for value in knots],
        ]

    features: list[dict] = []
    for condition in condition_names:
        features.append(_feature(condition, _input_atom(condition, 1)))
        features.append(_feature(f"{condition}^2", _input_atom(condition, 2)))
    for left_index, left in enumerate(condition_names):
        for right in condition_names[left_index + 1 :]:
            features.append(
                _feature(
                    f"{left}*{right}",
                    _input_atom(left, 1),
                    _input_atom(right, 1),
                )
            )
    for axis in axis_names:
        for atom in axis_atoms[axis]:
            suffix = (
                f"^{int(atom['value'])}"
                if atom["kind"] == "power"
                else f"_hinge3_{float(atom['value']):.8g}"
            )
            features.append(_feature(f"{axis}{suffix}", atom))
            for condition in condition_names:
                features.append(
                    _feature(
                        f"{condition}*{axis}{suffix}",
                        _input_atom(condition, 1),
                        atom,
                    )
                )
    if len(axis_names) == 2:
        left, right = axis_names
        for left_atom in axis_atoms[left]:
            for right_atom in axis_atoms[right]:
                features.append(
                    _feature(
                        f"{left}:{_atom_key(left_atom)}*{right}:{_atom_key(right_atom)}",
                        left_atom,
                        right_atom,
                    )
                )

    raw_columns = [
        _raw_feature_values(feature, standardized) for feature in features
    ]
    matrix = np.column_stack(raw_columns)
    centers = matrix.mean(axis=0)
    feature_scales = matrix.std(axis=0)
    retained = feature_scales >= 1e-12
    final_features = []
    for feature, center, scale, keep in zip(features, centers, feature_scales, retained):
        if keep:
            final_features.append(
                {
                    **feature,
                    "center": float(center),
                    "scale": float(scale),
                }
            )
    payload.update(
        {
            "family": "truncated_power_spline_tensor",
            "axis_names": axis_names,
            "condition_names": condition_names,
            "interior_knots": knot_records,
            "features": final_features,
        }
    )
    return payload


def greedy_order(
    matrix: np.ndarray,
    guide_target: np.ndarray,
    max_features: int,
    ridge: float = 1e-10,
) -> list[int]:
    centered_target = guide_target - float(np.mean(guide_target))
    gram = matrix.T @ matrix / len(matrix)
    cross = matrix.T @ centered_target / len(matrix)
    selected: list[int] = []
    remaining = np.ones(matrix.shape[1], dtype=bool)
    for _ in range(min(max_features, matrix.shape[1])):
        if selected:
            sub = np.ix_(selected, selected)
            coefficient = np.linalg.solve(
                gram[sub] + ridge * np.eye(len(selected)), cross[selected]
            )
            correlation = cross - gram[:, selected] @ coefficient
        else:
            correlation = cross.copy()
        correlation[~remaining] = 0.0
        chosen = int(np.argmax(np.abs(correlation)))
        if not remaining[chosen] or not np.isfinite(correlation[chosen]):
            raise RuntimeError("Greedy structure selection stalled")
        selected.append(chosen)
        remaining[chosen] = False
    return selected


def fit_tcad_coefficients(
    train_matrix: np.ndarray,
    validation_matrix: np.ndarray,
    train_target: np.ndarray,
    validation_target: np.ndarray,
    selected: list[int],
    alphas: tuple[float, ...],
) -> FormulaFit:
    x_train = train_matrix[:, selected]
    x_validation = validation_matrix[:, selected]
    target_mean = float(np.mean(train_target))
    target_scale = float(np.std(train_target))
    if target_scale < 1e-30:
        target_scale = 1.0
    normalized_target = (train_target - target_mean) / target_scale
    gram = x_train.T @ x_train / len(x_train)
    cross = x_train.T @ normalized_target / len(x_train)
    candidates: list[FormulaFit] = []
    for alpha in alphas:
        lhs = gram + float(alpha) * np.eye(len(selected))
        try:
            normalized_coefficient = np.linalg.solve(lhs, cross)
        except np.linalg.LinAlgError:
            normalized_coefficient = np.linalg.lstsq(lhs, cross, rcond=None)[0]
        coefficient = target_scale * normalized_coefficient
        prediction = target_mean + x_validation @ coefficient
        score = float(np.sqrt(np.mean(np.square(validation_target - prediction))))
        candidates.append(
            FormulaFit(float(alpha), target_mean, coefficient, score)
        )
    return min(candidates, key=lambda value: (value.validation_rmse, value.alpha))


def formula_payload(
    library: dict,
    task: str,
    seed: int,
    guidance_weight: float,
    selected: list[int],
    fit: FormulaFit,
) -> dict:
    terms = []
    for index, coefficient in zip(selected, fit.coefficients):
        feature = library["features"][index]
        terms.append({**feature, "coefficient": float(coefficient)})
    return {
        "schema_version": 1,
        "task": task,
        "seed": seed,
        "family": library["family"],
        "guidance_weight": guidance_weight,
        "guidance_role": "feature_ordering_only",
        "coefficient_target": "TCAD training rows",
        "ridge_selection_target": "TCAD validation rows",
        "selected_alpha": fit.alpha,
        "intercept": fit.intercept,
        "input_order": library["input_order"],
        "input_scalers": library["input_scalers"],
        "axis_names": library["axis_names"],
        "condition_names": library["condition_names"],
        "interior_knots": library["interior_knots"],
        "selected_terms": terms,
    }


def evaluate_formula(payload: dict, raw: np.ndarray) -> np.ndarray:
    standardized = _standardized_inputs(raw, payload)
    prediction = np.full(len(raw), float(payload["intercept"]), dtype=np.float64)
    for term in payload["selected_terms"]:
        values = _raw_feature_values(term, standardized)
        values = (values - float(term["center"])) / float(term["scale"])
        prediction += float(term["coefficient"]) * values
    return prediction


def _metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r2": float(r2_score(reference, prediction)),
        "p95_abs_error": float(np.percentile(np.abs(error), 95)),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def _max_group_rmse(
    reference: np.ndarray, prediction: np.ndarray, labels: np.ndarray
) -> float:
    values = []
    for label in np.unique(labels):
        mask = labels == label
        values.append(float(np.sqrt(np.mean(np.square(prediction[mask] - reference[mask])))))
    return float(max(values))


def _dense_design(
    train: pd.DataFrame, inputs: tuple[str, ...], points: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dense = np.empty((points, len(inputs)), dtype=np.float64)
    for index, name in enumerate(inputs):
        low = float(train[name].min())
        high = float(train[name].max())
        dense[:, index] = rng.uniform(low, high, points) if high > low else low
    return dense


def _shape_audit(
    task: str,
    payload: dict,
    dense: np.ndarray,
    train: pd.DataFrame,
    inputs: tuple[str, ...],
    target_scale: float,
) -> dict[str, float]:
    prediction = evaluate_formula(payload, dense)
    axis_low = float(train[inputs[0]].min())
    axis_high = float(train[inputs[0]].max())
    step = max((axis_high - axis_low) * 1e-5, 1e-12)
    left = dense.copy()
    right = dense.copy()
    left[:, 0] = np.maximum(axis_low, left[:, 0] - step)
    right[:, 0] = np.minimum(axis_high, right[:, 0] + step)
    denominator = right[:, 0] - left[:, 0]
    derivative = (evaluate_formula(payload, right) - evaluate_formula(payload, left)) / denominator
    tolerance = 1e-8 * max(target_scale, 1e-30) / max(axis_high - axis_low, 1e-12)
    constrained = task in {"I_dark", "I_photo", "AC_Response"}
    return {
        "dense_finite_rate": float(np.isfinite(prediction).mean()),
        "derivative_finite_rate": float(np.isfinite(derivative).mean()),
        "nonincreasing_violation_rate": (
            float(np.mean(derivative > tolerance)) if constrained else np.nan
        ),
        "ac_positive_rate": (
            float(np.mean(prediction > 1e-9)) if task == "AC_Response" else np.nan
        ),
        "ac_max_prediction": (
            float(np.max(prediction)) if task == "AC_Response" else np.nan
        ),
    }


def _complexity(payload: dict) -> dict[str, int]:
    primitive_keys = {
        _atom_key(atom)
        for term in payload["selected_terms"]
        for atom in term["atoms"]
    }
    operations = 1
    used_inputs = {key[1] for key in primitive_keys}
    operations += 2 * len(used_inputs)
    for term in payload["selected_terms"]:
        operations += 4 + max(0, len(term["atoms"]) - 1)
        for atom in term["atoms"]:
            if atom["kind"] == "hinge3":
                operations += 3
            elif int(atom["value"]) > 1:
                operations += 1
    return {
        "coefficient_count": len(payload["selected_terms"]) + 1,
        "unique_basis_primitives": len(primitive_keys),
        "operation_proxy": operations,
    }


def _route_name(weight: float) -> str:
    if weight == 0.0:
        return "direct_ordering"
    if weight == 1.0:
        return "bkan_teacher_ordering"
    return f"bkan_guided_{weight:.3g}"


def run_one(
    args: argparse.Namespace, task: str, seed: int
) -> tuple[list[dict], dict, pd.DataFrame]:
    task_args = _task_args(args, task, seed)
    frame, inputs, shared_spec, baseline_spec = single._load_task(task, task_args)
    partitions = single.shared.split_task_dataframe(
        frame, shared_spec, seed, fractions=(0.65, 0.10, 0.15, 0.10)
    )
    train, validation, calibration, test = partitions
    manifest = single._split_manifest(task, seed, shared_spec, partitions)
    actual = {
        "train": single._target(train, baseline_spec),
        "validation": single._target(validation, baseline_spec),
        "test": single._target(test, baseline_spec),
    }
    bkan_predict, bkan_info = single._load_bkan_parameter_mean(
        task, train, inputs, shared_spec, task_args
    )
    teacher = {
        "train": bkan_predict(train),
        "validation": bkan_predict(validation),
        "test": bkan_predict(test),
    }
    library = build_library(train, inputs, task, args.interior_knots)
    raw = {
        name: part[list(inputs)].to_numpy(dtype=np.float64)
        for name, part in zip(
            ("train", "validation", "calibration", "test"), partitions
        )
    }
    matrix = {
        name: candidate_matrix(values, library) for name, values in raw.items()
    }
    valid_budgets = [
        int(value) for value in args.budgets if int(value) <= matrix["train"].shape[1] + 1
    ]
    if not valid_budgets:
        raise RuntimeError(f"No budget fits the {task} candidate library")
    y_mean = float(np.mean(actual["train"]))
    y_scale = float(np.std(actual["train"]))
    y_scale = y_scale if y_scale >= 1e-30 else 1.0
    y_standard = (actual["train"] - y_mean) / y_scale
    teacher_standard = (teacher["train"] - y_mean) / y_scale
    dense = _dense_design(train, inputs, args.dense_points, seed + 1000)
    test_labels = single.shared.task_group_labels(test, shared_spec).astype(str).to_numpy()
    rows: list[dict] = []

    for guidance_weight in args.guidance_weights:
        route = _route_name(float(guidance_weight))
        guide = (1.0 - guidance_weight) * y_standard + guidance_weight * teacher_standard
        ordering = greedy_order(
            matrix["train"], guide, max(valid_budgets) - 1
        )
        for budget in valid_budgets:
            selected = ordering[: budget - 1]
            fit = fit_tcad_coefficients(
                matrix["train"],
                matrix["validation"],
                actual["train"],
                actual["validation"],
                selected,
                tuple(float(value) for value in args.ridge_alphas),
            )
            payload = formula_payload(
                library, task, seed, float(guidance_weight), selected, fit
            )
            export_dir = args.output / "exports" / f"seed_{seed}" / task / route
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / f"budget_{budget}.json"
            serialized = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
            export_path.write_text(serialized, encoding="utf-8")
            replay_payload = json.loads(export_path.read_text(encoding="utf-8"))
            prediction = evaluate_formula(payload, raw["test"])
            replay = evaluate_formula(replay_payload, raw["test"])
            metric = _metrics(actual["test"], prediction)
            complexity = _complexity(payload)
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "route": route,
                    "guidance_weight": float(guidance_weight),
                    "budget": budget,
                    "candidate_count": len(library["features"]) + 1,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "calibration_rows_untouched": len(calibration),
                    "test_rows": len(test),
                    "selected_alpha": fit.alpha,
                    "validation_tcad_rmse": fit.validation_rmse,
                    "formula_to_tcad_rmse": metric["rmse"],
                    "formula_to_tcad_r2": metric["r2"],
                    "formula_to_tcad_p95_abs_error": metric["p95_abs_error"],
                    "formula_to_tcad_max_abs_error": metric["max_abs_error"],
                    "max_group_rmse": _max_group_rmse(
                        actual["test"], prediction, test_labels
                    ),
                    "formula_to_bkan_rmse": float(
                        np.sqrt(np.mean(np.square(teacher["test"] - prediction)))
                    ),
                    "bkan_to_tcad_rmse": float(
                        np.sqrt(np.mean(np.square(actual["test"] - teacher["test"])))
                    ),
                    "test_finite_rate": float(np.isfinite(prediction).mean()),
                    "json_replay_max_abs_error": float(np.max(np.abs(replay - prediction))),
                    "formula_chars": len(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
                    "export_path": single._artifact_path(export_path),
                    "selected_features": ";".join(
                        library["features"][index]["name"] for index in selected
                    ),
                    **complexity,
                    **_shape_audit(task, payload, dense, train, inputs, y_scale),
                }
            )
    metadata = {
        "task": task,
        "seed": seed,
        "inputs": list(inputs),
        "candidate_count": len(library["features"]) + 1,
        "train_groups": int(single.shared.task_group_labels(train, shared_spec).nunique()),
        "validation_groups": int(
            single.shared.task_group_labels(validation, shared_spec).nunique()
        ),
        "calibration_groups": int(
            single.shared.task_group_labels(calibration, shared_spec).nunique()
        ),
        "test_groups": int(single.shared.task_group_labels(test, shared_spec).nunique()),
        "checkpoint": bkan_info["checkpoint"],
        "checkpoint_sha256": bkan_info["checkpoint_sha256"],
    }
    return rows, metadata, manifest


def _jaccard_stability(values: pd.Series) -> tuple[float, float]:
    selections = [set(str(value).split(";")) for value in values]
    scores = []
    for left_index, left in enumerate(selections):
        for right in selections[left_index + 1 :]:
            union = left | right
            scores.append(len(left & right) / len(union) if union else 1.0)
    if not scores:
        return np.nan, np.nan
    return float(np.mean(scores)), float(np.min(scores))


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["task", "route", "guidance_weight", "budget"], as_index=False)
        .agg(
            splits=("seed", "nunique"),
            coefficient_count=("coefficient_count", "first"),
            mean_rmse=("formula_to_tcad_rmse", "mean"),
            sd_rmse=("formula_to_tcad_rmse", "std"),
            median_rmse=("formula_to_tcad_rmse", "median"),
            mean_r2=("formula_to_tcad_r2", "mean"),
            min_r2=("formula_to_tcad_r2", "min"),
            mean_p95_abs_error=("formula_to_tcad_p95_abs_error", "mean"),
            mean_max_abs_error=("formula_to_tcad_max_abs_error", "mean"),
            mean_max_group_rmse=("max_group_rmse", "mean"),
            mean_formula_chars=("formula_chars", "mean"),
            mean_operation_proxy=("operation_proxy", "mean"),
            min_test_finite_rate=("test_finite_rate", "min"),
            min_dense_finite_rate=("dense_finite_rate", "min"),
            min_derivative_finite_rate=("derivative_finite_rate", "min"),
            mean_nonincreasing_violation_rate=("nonincreasing_violation_rate", "mean"),
            mean_ac_positive_rate=("ac_positive_rate", "mean"),
            max_json_replay_error=("json_replay_max_abs_error", "max"),
        )
        .sort_values(["task", "budget", "guidance_weight"])
    )
    stability_rows = []
    for keys, group in metrics.groupby(
        ["task", "route", "guidance_weight", "budget"], sort=True
    ):
        mean_jaccard, min_jaccard = _jaccard_stability(group["selected_features"])
        stability_rows.append(
            {
                "task": keys[0],
                "route": keys[1],
                "guidance_weight": keys[2],
                "budget": keys[3],
                "structure_jaccard_mean": mean_jaccard,
                "structure_jaccard_min": min_jaccard,
            }
        )
    return summary.merge(
        pd.DataFrame(stability_rows),
        on=["task", "route", "guidance_weight", "budget"],
        validate="one_to_one",
    )


def paired_guidance(metrics: pd.DataFrame) -> pd.DataFrame:
    direct = metrics.loc[
        metrics["guidance_weight"].eq(0.0),
        ["task", "seed", "budget", "formula_to_tcad_rmse"],
    ].rename(columns={"formula_to_tcad_rmse": "direct_rmse"})
    rows = []
    for weight in sorted(set(metrics["guidance_weight"]) - {0.0}):
        guided = metrics.loc[
            metrics["guidance_weight"].eq(weight),
            ["task", "seed", "budget", "formula_to_tcad_rmse"],
        ].rename(columns={"formula_to_tcad_rmse": "guided_rmse"})
        paired = guided.merge(direct, on=["task", "seed", "budget"], validate="one_to_one")
        paired["delta"] = paired["guided_rmse"] - paired["direct_rmse"]
        for (task, budget), group in paired.groupby(["task", "budget"], sort=True):
            stats = _candidate_paired_statistics(group["delta"].to_numpy(dtype=np.float64))
            rows.append(
                {
                    "task": task,
                    "budget": int(budget),
                    "guidance_weight": float(weight),
                    "route": _route_name(float(weight)),
                    "mean_guided_rmse": float(group["guided_rmse"].mean()),
                    "mean_direct_rmse": float(group["direct_rmse"].mean()),
                    "mean_relative_change": float(
                        np.mean(
                            (group["guided_rmse"] - group["direct_rmse"])
                            / group["direct_rmse"]
                        )
                    ),
                    **stats,
                }
            )
    result = pd.DataFrame(rows)
    result["holm_p"] = repeated.holm_adjust(result["corrected_two_sided_p"])
    result["corrected_favors_guided"] = result["corrected_ci_high"] < 0.0
    result["corrected_favors_direct"] = result["corrected_ci_low"] > 0.0
    return result.sort_values(["task", "budget", "guidance_weight"])


def accepted_seed42_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, reference in ACCEPTED_SEED42.items():
        available = metrics.loc[
            metrics["task"].eq(task)
            & metrics["seed"].eq(42)
            & metrics["guidance_weight"].eq(0.0)
        ].copy()
        if available.empty:
            continue
        available["budget_distance"] = np.abs(available["budget"] - reference["terms"])
        selected = available.sort_values(["budget_distance", "budget"]).iloc[0]
        rows.append(
            {
                "task": task,
                "accepted_terms": reference["terms"],
                "accepted_seed42_rmse": reference["rmse"],
                "new_direct_terms": int(selected["coefficient_count"]),
                "new_direct_seed42_rmse": float(selected["formula_to_tcad_rmse"]),
                "relative_rmse_reduction": float(
                    (reference["rmse"] - selected["formula_to_tcad_rmse"])
                    / reference["rmse"]
                ),
                "comparison_limit": (
                    "same frozen seed-42 test groups; formula families and fitting-row "
                    "budgets differ because the accepted gated fit also used calibration groups"
                ),
            }
        )
    return pd.DataFrame(rows)


def _candidate_paired_statistics(difference: np.ndarray) -> dict[str, float | int]:
    stats = repeated.corrected_paired_statistics(difference, 0.10, 0.65)
    return {
        "mean_delta_candidate_minus_reference": stats[
            "mean_delta_teacher_minus_direct"
        ],
        "median_delta_candidate_minus_reference": stats[
            "median_delta_teacher_minus_direct"
        ],
        "sample_sd_delta": stats["sample_sd_delta"],
        "corrected_standard_error": stats["corrected_standard_error"],
        "corrected_ci_low": stats["corrected_ci_low"],
        "corrected_ci_high": stats["corrected_ci_high"],
        "corrected_two_sided_p": stats["corrected_two_sided_p"],
        "candidate_wins": stats["teacher_wins"],
        "ties": stats["ties"],
        "candidate_losses": stats["teacher_losses"],
    }


def nearest_legacy_comparison(
    summary: pd.DataFrame, legacy_path: Path
) -> pd.DataFrame:
    if not legacy_path.is_file():
        return pd.DataFrame()
    legacy = pd.read_csv(legacy_path)
    legacy = legacy.loc[
        legacy["source"].eq("tcad_direct") & legacy["family"].eq("additive_spline")
    ]
    rows = []
    for task, target_budget in REFERENCE_BUDGET.items():
        current = summary.loc[
            summary["task"].eq(task)
            & summary["guidance_weight"].eq(0.0)
            & summary["budget"].eq(target_budget)
        ]
        old = legacy.loc[legacy["task"].eq(task)].copy()
        if current.empty or old.empty:
            continue
        old["distance"] = np.abs(old["terms"] - target_budget)
        prior = old.sort_values(["distance", "terms"]).iloc[0]
        new = current.iloc[0]
        rows.append(
            {
                "task": task,
                "new_terms": int(new["coefficient_count"]),
                "new_mean_rmse": float(new["mean_rmse"]),
                "legacy_terms": int(prior["terms"]),
                "legacy_mean_rmse": float(prior["mean_rmse"]),
                "relative_rmse_reduction": float(
                    (prior["mean_rmse"] - new["mean_rmse"]) / prior["mean_rmse"]
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_family_comparison(
    metrics: pd.DataFrame, legacy_path: Path
) -> pd.DataFrame:
    if not legacy_path.is_file():
        return pd.DataFrame()
    legacy = pd.read_csv(legacy_path)
    legacy_budget = {
        "I_dark": 4,
        "I_photo": 8,
        "AC_Response": 4,
        "Capacitance": 4,
    }
    rows = []
    for task, new_budget in REFERENCE_BUDGET.items():
        new = metrics.loc[
            metrics["task"].eq(task)
            & metrics["route"].eq("direct_ordering")
            & metrics["budget"].eq(new_budget),
            ["seed", "formula_to_tcad_rmse"],
        ].rename(columns={"formula_to_tcad_rmse": "new_rmse"})
        old = legacy.loc[
            legacy["task"].eq(task)
            & legacy["source"].eq("tcad_direct")
            & legacy["family"].eq("additive_spline")
            & legacy["budget"].eq(legacy_budget[task]),
            ["seed", "terms", "formula_to_tcad_rmse"],
        ].rename(columns={"formula_to_tcad_rmse": "legacy_rmse"})
        paired = new.merge(old, on="seed", validate="one_to_one")
        difference = paired["new_rmse"] - paired["legacy_rmse"]
        rows.append(
            {
                "task": task,
                "new_terms": new_budget,
                "legacy_terms": int(paired["terms"].iloc[0]),
                "mean_new_rmse": float(paired["new_rmse"].mean()),
                "mean_legacy_rmse": float(paired["legacy_rmse"].mean()),
                **_candidate_paired_statistics(
                    difference.to_numpy(dtype=np.float64)
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_p"] = repeated.holm_adjust(result["corrected_two_sided_p"])
    return result


def write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    accepted: pd.DataFrame,
    legacy: pd.DataFrame,
    family_paired: pd.DataFrame,
) -> None:
    lines = [
        "# Symbolic structure guidance ablation",
        "",
        "This is an exploratory development experiment. It does not modify or replace the frozen paper results.",
        "All structures use the same explicit truncated-power spline tensor library. BKAN affects only feature ordering; every final coefficient is refit to TCAD training rows and ridge strength is selected on TCAD validation rows. Calibration and test rows remain unused during fitting and selection.",
        "",
        "## Reference-budget results",
        "",
        "| Task | Route | Coefficients | Mean test RMSE | SD | Mean R2 | Shape violation | Jaccard mean/min |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, budget in REFERENCE_BUDGET.items():
        selected = summary.loc[
            summary["task"].eq(task) & summary["budget"].eq(budget)
        ]
        for row in selected.to_dict(orient="records"):
            violation = row["mean_nonincreasing_violation_rate"]
            violation_text = "n/a" if not np.isfinite(violation) else f"{violation:.4f}"
            lines.append(
                f"| {task} | {row['route']} | {row['coefficient_count']} | "
                f"{row['mean_rmse']:.6g} | {row['sd_rmse']:.3g} | "
                f"{row['mean_r2']:.6f} | {violation_text} | "
                f"{row['structure_jaccard_mean']:.3f}/{row['structure_jaccard_min']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## BKAN guidance at reference budgets",
            "",
            "Positive deltas mean BKAN-guided ordering is worse than direct ordering.",
            "",
            "| Task | Guidance | Mean delta | Corrected 95% CI | Holm p | W/T/L |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for task, budget in REFERENCE_BUDGET.items():
        selected = paired.loc[
            paired["task"].eq(task) & paired["budget"].eq(budget)
        ]
        for row in selected.to_dict(orient="records"):
            lines.append(
                f"| {task} | {row['guidance_weight']:.3g} | "
                f"{row['mean_delta_candidate_minus_reference']:.6g} | "
                f"[{row['corrected_ci_low']:.6g}, {row['corrected_ci_high']:.6g}] | "
                f"{row['holm_p']:.4g} | {row['candidate_wins']}/{row['ties']}/{row['candidate_losses']} |"
            )
    if not legacy.empty:
        lines.extend(
            [
                "",
                "## Candidate-family comparison",
                "",
                "The legacy control is the nearest-complexity direct additive-spline result from the existing ten-split audit.",
                "",
                "| Task | New/legacy terms | New mean RMSE | Legacy mean RMSE | Relative reduction |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in legacy.to_dict(orient="records"):
            lines.append(
                f"| {row['task']} | {row['new_terms']}/{row['legacy_terms']} | "
                f"{row['new_mean_rmse']:.6g} | {row['legacy_mean_rmse']:.6g} | "
                f"{100.0 * row['relative_rmse_reduction']:.2f}% |"
            )
    if not accepted.empty:
        lines.extend(
            [
                "",
                "## Frozen seed-42 gated reference",
                "",
                "This comparison is diagnostic because the accepted gated students also used calibration groups for fitting.",
                "",
                "| Task | Accepted/new terms | Accepted RMSE | New direct RMSE | Relative reduction |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in accepted.to_dict(orient="records"):
            lines.append(
                f"| {row['task']} | {row['accepted_terms']}/{row['new_direct_terms']} | "
                f"{row['accepted_seed42_rmse']:.6g} | {row['new_direct_seed42_rmse']:.6g} | "
                f"{100.0 * row['relative_rmse_reduction']:.2f}% |"
            )
    if not family_paired.empty:
        lines.extend(
            [
                "",
                "## Paired candidate-family comparison",
                "",
                "The difference is new task-specific direct RMSE minus the existing direct additive-spline RMSE. Negative values favor the new family.",
                "",
                "| Task | New/legacy terms | Mean delta | Corrected 95% CI | Holm p | W/T/L |",
                "| --- | ---: | ---: | --- | ---: | --- |",
            ]
        )
        for row in family_paired.to_dict(orient="records"):
            lines.append(
                f"| {row['task']} | {row['new_terms']}/{row['legacy_terms']} | "
                f"{row['mean_delta_candidate_minus_reference']:.6g} | "
                f"[{row['corrected_ci_low']:.6g}, {row['corrected_ci_high']:.6g}] | "
                f"{row['holm_p']:.4g} | {row['candidate_wins']}/{row['ties']}/{row['candidate_losses']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A large improvement from the new direct library indicates that the response family needs redesign. A reproducible improvement from BKAN-guided ordering over direct ordering indicates that BKAN contributes useful structural information. Optimizer or gate tuning alone is supported only when the family comparison is small and BKAN guidance remains useful at matched complexity.",
        ]
    )
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    metadata: list[dict] = []
    manifests: list[pd.DataFrame] = []
    for seed in args.seeds:
        for task in args.tasks:
            print(f"[run] seed={seed} task={task}", flush=True)
            task_rows, task_metadata, task_manifest = run_one(args, task, seed)
            rows.extend(task_rows)
            metadata.append(task_metadata)
            manifests.append(task_manifest)
            print(
                f"[done] seed={seed} task={task} candidates={task_metadata['candidate_count']}",
                flush=True,
            )
    metrics = pd.DataFrame(rows)
    if not metrics["test_finite_rate"].eq(1.0).all():
        raise RuntimeError("At least one test formula is non-finite")
    if not metrics["dense_finite_rate"].eq(1.0).all():
        raise RuntimeError("At least one dense formula is non-finite")
    if not metrics["derivative_finite_rate"].eq(1.0).all():
        raise RuntimeError("At least one derivative audit is non-finite")
    if not metrics["json_replay_max_abs_error"].eq(0.0).all():
        raise RuntimeError("At least one JSON replay differs from the in-memory formula")
    summary = aggregate(metrics)
    paired = paired_guidance(metrics)
    accepted = accepted_seed42_comparison(metrics)
    legacy = nearest_legacy_comparison(
        summary, args.legacy_audit / "aggregate_by_export.csv"
    )
    family_paired = paired_family_comparison(
        metrics, args.legacy_audit / "metrics_by_export.csv"
    )
    metrics.to_csv(args.output / "metrics.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    paired.to_csv(args.output / "paired_guidance_vs_direct.csv", index=False)
    accepted.to_csv(args.output / "accepted_seed42_comparison.csv", index=False)
    legacy.to_csv(args.output / "legacy_family_comparison.csv", index=False)
    family_paired.to_csv(args.output / "paired_family_comparison.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output / "task_seed_audit.csv", index=False)
    manifest = pd.concat(manifests, ignore_index=True)
    manifest.to_csv(args.output / "split_manifest.csv", index=False)
    manifest_match = None
    legacy_manifest_path = args.legacy_audit / "split_manifest.csv"
    if legacy_manifest_path.is_file():
        legacy_manifest = pd.read_csv(legacy_manifest_path)
        columns = ["task", "seed", "split", "group_id", "n_points"]
        expected = legacy_manifest.loc[
            legacy_manifest["task"].isin(args.tasks)
            & legacy_manifest["seed"].isin(args.seeds),
            columns,
        ].sort_values(columns).reset_index(drop=True)
        observed = manifest[columns].sort_values(columns).reset_index(drop=True)
        manifest_match = bool(expected.equals(observed))
        if not manifest_match:
            raise RuntimeError("New split manifest differs from the frozen ten-split audit")
    protocol = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "matched-split direct versus BKAN-guided symbolic structure ablation",
        "paper_files_modified": False,
        "tasks": list(args.tasks),
        "seeds": list(args.seeds),
        "split_fractions": [0.65, 0.10, 0.15, 0.10],
        "budgets_include_intercept": list(args.budgets),
        "guidance_weights": list(args.guidance_weights),
        "guidance_definition": "(1-w)*standardized_TCAD + w*standardized_BKAN_parameter_mean, used only for greedy feature ordering",
        "coefficient_fit": "TCAD training rows only",
        "ridge_selection": "minimum TCAD validation RMSE",
        "calibration_policy": "untouched",
        "test_policy": "evaluation only after formula freeze",
        "split_manifest_matches_existing_ten_split_audit": manifest_match,
        "candidate_family": "standardized linear/quadratic condition terms, condition interactions, truncated cubic swept-axis bases, condition-axis products, and for capacitance bias-frequency tensor products",
        "interior_knots": args.interior_knots,
        "ridge_alphas": list(args.ridge_alphas),
        "complexity": "coefficient count plus unique primitive, compact JSON character, and arithmetic-operation proxies",
        "statistics": "paired corrected-resampled-t intervals with 1/R+n_test/n_train variance and Holm adjustment across all guidance contrasts",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    write_report(args, summary, paired, accepted, legacy, family_paired)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
