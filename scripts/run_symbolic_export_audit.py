"""Build the frozen four-task symbolic/export audit and direct baselines.

This script does not retrain BKANs or symbolic students.  It independently
replays the selected KAN-free JSON formulas, evaluates dense in-envelope
points and axis derivatives, summarizes three distillation seeds, and exports
fresh Poly3-Ridge and Spline-Ridge response surfaces to machine-readable JSON
and Verilog-A source.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, PPoly
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
PAPER_EXPERIMENTS = BKAN_ROOT / "simulation" / "paper_experiments"
for path in (BKAN_ROOT, PAPER_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from device_modeling.photodetector import bayesian_symbolic  # noqa: E402
from device_modeling.photodetector import task_config as shared  # noqa: E402
from device_modeling.photodetector.physical_library import PhysicalFeatureLibrary  # noqa: E402
from device_modeling.photodetector.symbolic_gated_kan import (  # noqa: E402
    SymbolicGatedKAN,
    evaluate_exported_pure_symbolic_formula,
)
from p0_3_traditional_comparison import (  # noqa: E402
    TASKS as BASELINE_TASKS,
    build_engineering_model,
    transformed_target,
)


DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "symbolic_export_audit"
DEFAULT_DATA = ROOT / "artifacts" / "results" / "device_modeling" / "cleaned_data.csv"
DEFAULT_NET_PHOTO_DATA = ROOT / "artifacts" / "results" / "net_photocurrent_retrain" / "device_modeling" / "cleaned_data.csv"
TASK_INFO = {
    "I_dark": ("dark_current", "dark_voltage"),
    "I_photo": ("photo_current", "light_voltage"),
    "AC_Response": ("ac_response", "frequency_ghz"),
    "Capacitance": ("capacitance", "bias_v"),
}
PRIMARY = {
    "I_dark": ROOT / "artifacts/results/symbolic_gate_replacement/dark_teacher_budget36/dark_current_symbolic_gated_kan",
    "I_photo": ROOT / "artifacts/results/net_photocurrent_retrain/symbolic_fixed_teacher/seed_42/photo_current_symbolic_gated_kan",
    "AC_Response": ROOT / "artifacts/results/symbolic_gate_replacement/ac_lowpass_gated_studentseed42_audited/ac_response_lowpass_gated",
    "Capacitance": ROOT / "artifacts/results/symbolic_gate_replacement/cap_teacher_curvature_budget24_seed42/capacitance_symbolic_gated_kan",
}
REPLICATES = {
    "I_dark": [
        PRIMARY["I_dark"],
        ROOT / "artifacts/results/symbolic_gate_replacement/dark_teacher_budget36_studentseed43/dark_current_symbolic_gated_kan",
        ROOT / "artifacts/results/symbolic_gate_replacement/dark_teacher_budget36_studentseed44/dark_current_symbolic_gated_kan",
    ],
    "I_photo": [
        PRIMARY["I_photo"],
        ROOT / "artifacts/results/net_photocurrent_retrain/symbolic_fixed_teacher/seed_43/photo_current_symbolic_gated_kan",
        ROOT / "artifacts/results/net_photocurrent_retrain/symbolic_fixed_teacher/seed_44/photo_current_symbolic_gated_kan",
    ],
    "AC_Response": [
        PRIMARY["AC_Response"],
        ROOT / "artifacts/results/symbolic_gate_replacement/ac_lowpass_gated_studentseed43/ac_response_lowpass_gated",
        ROOT / "artifacts/results/symbolic_gate_replacement/ac_lowpass_gated_studentseed44/ac_response_lowpass_gated",
    ],
    "Capacitance": [
        PRIMARY["Capacitance"],
        ROOT / "artifacts/results/symbolic_gate_replacement/cap_teacher_curvature_budget24_studentseed43/capacitance_symbolic_gated_kan",
        ROOT / "artifacts/results/symbolic_gate_replacement/cap_teacher_curvature_budget24_studentseed44/capacitance_symbolic_gated_kan",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--net-photocurrent-data", type=Path, default=DEFAULT_NET_PHOTO_DATA)
    parser.add_argument("--capacitance-data", type=Path, default=shared.DEFAULT_CAPACITANCE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-points", type=int, default=20000)
    parser.add_argument("--axis-points", type=int, default=1001)
    parser.add_argument("--axis-anchors", type=int, default=24)
    return parser.parse_args()


def _formula_path(directory: Path) -> Path:
    return directory / "pure_symbolic_formula.json"


def _prediction_path(directory: Path) -> Path:
    return directory / "predictions.csv"


def _load_task(task: str, args) -> tuple[pd.DataFrame, tuple[str, ...], shared.TaskSpec]:
    key = TASK_INFO[task][0]
    spec = shared.TASKS[key]
    raw = (
        shared.load_capacitance_data(args.capacitance_data)
        if task == "Capacitance"
        else shared.load_research_data(args.net_photocurrent_data if task == "I_photo" else args.data)
    )
    frame, inputs = shared.prepare_task_dataframe(raw, spec)
    return frame, tuple(inputs), spec


def _eval_ac(payload: dict, x_raw: np.ndarray) -> np.ndarray:
    inputs = tuple(payload["inputs"])
    raw = np.asarray(x_raw, dtype=np.float64)
    values = {name: raw[:, inputs.index(name)] for name in inputs}
    columns = []
    for feature in payload["feature_definitions"]:
        kind = feature["kind"]
        if kind == "one":
            columns.append(np.ones(len(raw)))
        elif kind == "z":
            columns.append((values[feature["input"]] - feature["mean"]) / feature["scale"])
        elif kind == "z2":
            z = (values[feature["input"]] - feature["mean"]) / feature["scale"]
            columns.append(z * z)
        elif kind == "z_product":
            left = (values[feature["left"]] - feature["left_mean"]) / feature["left_scale"]
            right = (values[feature["right"]] - feature["right_mean"]) / feature["right_scale"]
            columns.append(left * right)
        else:
            raise ValueError(kind)
    phi = np.column_stack(columns)
    y0, floor, logfc = (
        phi @ np.asarray(payload["coefficients"][name], dtype=np.float64)
        for name in ("y0", "floor", "logfc")
    )
    fc = np.exp(np.clip(logfc, np.log(0.02), np.log(500.0)))
    ratio = np.power(
        np.maximum(values["frequency_ghz"] / fc, 1e-12), float(payload["p"])
    )
    return floor + (y0 - floor) / (1.0 + ratio)


def evaluate_formula(payload: dict, x_raw: np.ndarray) -> np.ndarray:
    if payload.get("family") == "gated_lowpass":
        return _eval_ac(payload, x_raw)
    if payload.get("uses_kan_at_inference", False):
        raise RuntimeError("Hybrid KAN result cannot be audited as a pure formula")
    return evaluate_exported_pure_symbolic_formula(payload, x_raw)


def _formula_inputs(payload: dict) -> tuple[str, ...]:
    if payload.get("family") == "gated_lowpass":
        return tuple(payload["inputs"])
    return tuple(payload["input_scalers"].keys())


def _formula_terms(payload: dict) -> set[str]:
    if payload.get("family") == "gated_lowpass":
        return {
            f"{row['submodel']}:{row['feature']}" for row in payload["active_terms"]
        }
    terms = set()
    for branch in payload.get("branches", []):
        for row in branch.get("terms", []):
            terms.add(f"mechanism:{branch.get('name', 'branch')}:{row['term']}")
    for label in ("residual_branch", "generic_branch"):
        for row in payload.get(label, {}).get("terms", []):
            terms.add(f"{label}:{row['term']}")
    return terms


def _formula_chars(payload: dict) -> int:
    return len(str(payload.get("model_space_formula", payload.get("formula", ""))))


def _prediction_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    teacher = "teacher" if "teacher" in frame else "bayesian_teacher_mean"
    return "actual", teacher, "prediction"


def _error_metrics(reference: np.ndarray, prediction: np.ndarray, prefix: str) -> dict:
    valid = np.isfinite(reference) & np.isfinite(prediction)
    error = prediction[valid] - reference[valid]
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(error * error))),
        f"{prefix}_p95_abs_error": float(np.percentile(np.abs(error), 95)),
        f"{prefix}_max_abs_error": float(np.max(np.abs(error))),
        f"{prefix}_r2": float(r2_score(reference[valid], prediction[valid])),
    }


def _dense_design(frame: pd.DataFrame, inputs: tuple[str, ...], task: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty((n, len(inputs)), dtype=np.float64)
    for index, name in enumerate(inputs):
        low, high = frame[name].min(), frame[name].max()
        if task == "AC_Response" and name == "frequency_ghz":
            values[:, index] = np.exp(rng.uniform(np.log(low), np.log(high), n))
        else:
            values[:, index] = rng.uniform(low, high, n)
    return values


def _derivative_audit(
    payload: dict,
    frame: pd.DataFrame,
    task: str,
    axis: str,
    anchors: int,
    points: int,
    seed: int,
) -> dict:
    inputs = _formula_inputs(payload)
    rng = np.random.default_rng(seed)
    sample = frame.iloc[rng.choice(len(frame), size=min(anchors, len(frame)), replace=False)]
    low, high = float(frame[axis].min()), float(frame[axis].max())
    if task == "AC_Response":
        grid = np.exp(np.linspace(np.log(low), np.log(high), points))
    else:
        grid = np.linspace(low, high, points)
    all_derivatives, all_jumps = [], []
    for _, anchor in sample.iterrows():
        raw = np.tile(anchor[list(inputs)].to_numpy(dtype=np.float64), (points, 1))
        raw[:, inputs.index(axis)] = grid
        prediction = evaluate_formula(payload, raw)
        derivative = np.gradient(prediction, grid, edge_order=2)
        all_derivatives.append(derivative)
        all_jumps.append(np.abs(np.diff(derivative)))
    derivative = np.concatenate(all_derivatives)
    jumps = np.concatenate(all_jumps)
    scale = max(float(np.percentile(np.abs(derivative[np.isfinite(derivative)]), 95)), 1e-30)
    return {
        "derivative_finite_rate": float(np.isfinite(derivative).mean()),
        "derivative_jump_p95": float(np.percentile(jumps[np.isfinite(jumps)], 95)),
        "derivative_jump_max": float(np.max(jumps[np.isfinite(jumps)])),
        "derivative_jump_p95_normalized": float(np.percentile(jumps[np.isfinite(jumps)], 95) / scale),
        "derivative_jump_max_normalized": float(np.max(jumps[np.isfinite(jumps)]) / scale),
        "derivative_grid_points": points,
        "derivative_anchor_count": len(sample),
    }


def audit_formulas(args) -> pd.DataFrame:
    rows = []
    selected_rows = []
    for task, directory in PRIMARY.items():
        payload = json.loads(_formula_path(directory).read_text(encoding="utf-8"))
        frame, _, spec = _load_task(task, args)
        inputs = _formula_inputs(payload)
        predictions = pd.read_csv(_prediction_path(directory))
        test = predictions.loc[predictions["split"].eq("test")].reset_index(drop=True)
        actual_col, teacher_col, prediction_col = _prediction_columns(test)
        actual = test[actual_col].to_numpy(dtype=np.float64)
        teacher = test[teacher_col].to_numpy(dtype=np.float64)
        prediction = test[prediction_col].to_numpy(dtype=np.float64)
        if all(name in test for name in inputs):
            replay_inputs = test[list(inputs)].to_numpy(dtype=np.float64)
        else:
            regenerated_test = shared.split_task_dataframe(
                frame, spec, args.seed, fractions=(0.70, 0.10, 0.10, 0.10)
            )[3]
            regenerated_actual = shared.target_values(regenerated_test, spec)
            if regenerated_actual.shape != actual.shape or not np.allclose(
                regenerated_actual, actual, rtol=1e-7, atol=1e-10
            ):
                raise RuntimeError(
                    f"Cannot reconstruct the frozen test order for {task}"
                )
            replay_inputs = regenerated_test[list(inputs)].to_numpy(dtype=np.float64)
        replay = evaluate_formula(payload, replay_inputs)
        replay_error = float(np.max(np.abs(replay - prediction)))
        if replay_error > max(1e-12, 1e-6 * float(np.max(np.abs(prediction)))):
            raise RuntimeError(f"Independent replay mismatch for {task}: {replay_error}")
        dense = _dense_design(frame, inputs, task, args.dense_points, args.seed)
        dense_prediction = evaluate_formula(payload, dense)
        start = time.perf_counter()
        for _ in range(20):
            evaluate_formula(payload, dense[:1000])
        latency = (time.perf_counter() - start) / 20000.0
        row = {
            "task": task,
            "method": "pure_symbolic_distillation",
            "uses_kan_at_inference": False,
            "test_points": len(test),
            "test_finite_rate": float(np.isfinite(prediction).mean()),
            "dense_points": len(dense_prediction),
            "dense_finite_rate": float(np.isfinite(dense_prediction).mean()),
            "formula_chars": _formula_chars(payload),
            "active_terms": len(_formula_terms(payload)),
            "independent_replay_max_abs_error": replay_error,
            "latency_per_point_s": latency,
            **_error_metrics(teacher, prediction, "formula_to_network"),
            **_error_metrics(actual, prediction, "formula_to_tcad"),
            **_derivative_audit(
                payload, frame, task, TASK_INFO[task][1], args.axis_anchors,
                args.axis_points, args.seed,
            ),
        }
        rows.append(row)
        for term in sorted(_formula_terms(payload)):
            selected_rows.append({"task": task, "term": term})
    pd.DataFrame(selected_rows).to_csv(args.output / "pure_formula_selected_terms.csv", index=False)
    return pd.DataFrame(rows)


def audit_stability(args) -> pd.DataFrame:
    rows = []
    for task, directories in REPLICATES.items():
        payloads = [json.loads(_formula_path(path).read_text(encoding="utf-8")) for path in directories]
        term_sets = [_formula_terms(payload) for payload in payloads]
        jaccard = [
            len(left & right) / len(left | right) if left | right else 1.0
            for left, right in itertools.combinations(term_sets, 2)
        ]
        prediction_arrays = []
        for directory in directories:
            frame = pd.read_csv(_prediction_path(directory))
            prediction_arrays.append(
                frame.loc[frame["split"].eq("test"), "prediction"].to_numpy(dtype=np.float64)
            )
        disagreement = [
            float(np.sqrt(np.mean((left - right) ** 2)))
            for left, right in itertools.combinations(prediction_arrays, 2)
        ]
        rows.append(
            {
                "task": task,
                "distillation_seeds": "42,43,44",
                "teacher_scope": (
                    "same frozen 500-pass q_beta predictive-mean teacher"
                    if task == "AC_Response"
                    else "same frozen q_beta parameter-mean deterministic KAN teacher"
                ),
                "pairwise_term_jaccard_mean": float(np.mean(jaccard)),
                "pairwise_term_jaccard_min": float(np.min(jaccard)),
                "pairwise_prediction_rmse_mean": float(np.mean(disagreement)),
                "pairwise_prediction_rmse_max": float(np.max(disagreement)),
                "term_count_min": min(map(len, term_sets)),
                "term_count_max": max(map(len, term_sets)),
            }
        )
    return pd.DataFrame(rows)


def _poly_payload(model, inputs: tuple[str, ...]) -> dict:
    scaler, poly, ridge = (model.named_steps[name] for name in ("scale", "poly", "ridge"))
    return {
        "family": "poly3_ridge",
        "inputs": list(inputs),
        "x_mean": scaler.mean_.tolist(),
        "x_scale": scaler.scale_.tolist(),
        "powers": poly.powers_.astype(int).tolist(),
        "coefficients": np.asarray(ridge.coef_).reshape(-1).tolist(),
        "intercept": float(np.asarray(ridge.intercept_).reshape(-1)[0]),
    }


def _eval_poly(payload: dict, raw: np.ndarray) -> np.ndarray:
    z = (raw - np.asarray(payload["x_mean"])) / np.asarray(payload["x_scale"])
    powers = np.asarray(payload["powers"], dtype=int)
    design = np.prod(np.power(z[:, None, :], powers[None, :, :]), axis=2)
    return float(payload["intercept"]) + design @ np.asarray(payload["coefficients"])


def _spline_payload(model, inputs: tuple[str, ...]) -> dict:
    scaler, transformer, ridge = (
        model.named_steps[name] for name in ("scale", "spline", "ridge")
    )
    coefficient = np.asarray(ridge.coef_).reshape(-1)
    features, offset = [], 0
    for name, basis in zip(inputs, transformer.bsplines_):
        emitted = basis.c.shape[0] - 1  # include_bias=False drops the last basis
        combined = np.zeros(basis.c.shape[0], dtype=np.float64)
        combined[:emitted] = coefficient[offset:offset + emitted]
        offset += emitted
        pp = PPoly.from_spline(BSpline(basis.t, combined, basis.k, extrapolate=False))
        features.append(
            {
                "input": name,
                "degree": int(basis.k),
                "domain": [float(basis.t[basis.k]), float(basis.t[-basis.k - 1])],
                "breaks": pp.x.tolist(),
                "coefficients": pp.c.tolist(),
            }
        )
    return {
        "family": "spline_ridge",
        "inputs": list(inputs),
        "x_mean": scaler.mean_.tolist(),
        "x_scale": scaler.scale_.tolist(),
        "intercept": float(np.asarray(ridge.intercept_).reshape(-1)[0]),
        "features": features,
    }


def _eval_pp(feature: dict, value: np.ndarray) -> np.ndarray:
    breaks = np.asarray(feature["breaks"], dtype=np.float64)
    coefficients = np.asarray(feature["coefficients"], dtype=np.float64)
    low, high = feature["domain"]
    clipped = np.clip(value, low, high)
    index = np.searchsorted(breaks, clipped, side="right") - 1
    index = np.clip(index, 0, len(breaks) - 2)
    delta = clipped - breaks[index]
    out = np.zeros(len(value), dtype=np.float64)
    for row in coefficients:
        out = out * delta + row[index]
    return out


def _eval_spline(payload: dict, raw: np.ndarray) -> np.ndarray:
    z = (raw - np.asarray(payload["x_mean"])) / np.asarray(payload["x_scale"])
    result = np.full(len(raw), float(payload["intercept"]))
    for index, feature in enumerate(payload["features"]):
        result += _eval_pp(feature, z[:, index])
    return result


def _va_number(value: float) -> str:
    return f"{float(value):.17g}"


def _write_poly_va(payload: dict, path: Path) -> None:
    lines = ["`include \"constants.vams\"", "`include \"disciplines.vams\"", "module poly3_export(out);", "output out; electrical out;"]
    for index, name in enumerate(payload["inputs"]):
        lines.append(f"parameter real x{index}=0.0; // {name}")
    lines.extend(["real y;", "analog begin", f"  y = {_va_number(payload['intercept'])}"])
    for coefficient, powers in zip(payload["coefficients"], payload["powers"]):
        factors = []
        for index, power in enumerate(powers):
            if power:
                z = f"((x{index}-{_va_number(payload['x_mean'][index])})/{_va_number(payload['x_scale'][index])})"
                factors.extend([z] * int(power))
        lines.append(f"    + ({_va_number(coefficient)})*({'*'.join(factors) if factors else '1.0'})")
    lines.extend(["  ;", "  V(out) <+ y;", "end", "endmodule"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_spline_va(payload: dict, path: Path) -> None:
    lines = ["`include \"constants.vams\"", "`include \"disciplines.vams\"", "module spline_ridge_export(out);", "output out; electrical out;"]
    for index, name in enumerate(payload["inputs"]):
        lines.append(f"parameter real x{index}=0.0; // {name}")
    lines.extend(["real y,z,g;", "analog begin", f"  y = {_va_number(payload['intercept'])};"])
    for feature_index, feature in enumerate(payload["features"]):
        lines.append(f"  z = (x{feature_index}-{_va_number(payload['x_mean'][feature_index])})/{_va_number(payload['x_scale'][feature_index])};")
        lines.append(f"  z = min(max(z,{_va_number(feature['domain'][0])}),{_va_number(feature['domain'][1])});")
        breaks = feature["breaks"]
        coeff = feature["coefficients"]
        valid = [i for i in range(len(breaks) - 1) if breaks[i + 1] > breaks[i]]
        for order, interval in enumerate(valid):
            if order == len(valid) - 1:
                branch = "begin" if order == 0 else "else begin"
            else:
                condition = "if" if order == 0 else "else if"
                branch = f"{condition} (z < {_va_number(breaks[interval + 1])}) begin"
            lines.append(f"  {branch}")
            expression = _va_number(coeff[0][interval])
            for row in coeff[1:]:
                expression = f"(({expression})*(z-{_va_number(breaks[interval])})+{_va_number(row[interval])})"
            lines.extend([f"    g = {expression};", "  end"])
        lines.append("  y = y + g;")
    lines.extend(["  V(out) <+ y;", "end", "endmodule"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_baselines(args) -> pd.DataFrame:
    rows = []
    export_root = args.output / "direct_baselines"
    export_root.mkdir(parents=True, exist_ok=True)
    for task in TASK_INFO:
        frame, inputs, spec = _load_task(task, args)
        train, validation, calibration, test = shared.split_task_dataframe(
            frame, spec, args.seed, fractions=(0.70, 0.10, 0.10, 0.10)
        )
        fit_parts = []
        for partition in (train, calibration):
            clean = partition.copy()
            clean.attrs = {}
            fit_parts.append(clean)
        fit = pd.concat(fit_parts, ignore_index=True)
        x_fit = fit[list(inputs)].to_numpy(dtype=np.float64)
        y_fit = shared.target_values(fit, spec)
        x_test = test[list(inputs)].to_numpy(dtype=np.float64)
        y_test = shared.target_values(test, spec)
        dense = _dense_design(frame, inputs, task, args.dense_points, args.seed)
        task_root = export_root / task
        task_root.mkdir(parents=True, exist_ok=True)
        for model_name in ("poly3_ridge", "spline_ridge"):
            model = build_engineering_model(model_name, len(fit), len(inputs), args.seed)
            model.fit(x_fit, y_fit)
            payload = _poly_payload(model, inputs) if model_name == "poly3_ridge" else _spline_payload(model, inputs)
            evaluator = _eval_poly if model_name == "poly3_ridge" else _eval_spline
            prediction = evaluator(payload, x_test)
            sklearn_prediction = model.predict(x_test).reshape(-1)
            replay = float(np.max(np.abs(prediction - sklearn_prediction)))
            dense_prediction = evaluator(payload, dense)
            json_path = task_root / f"{model_name}.json"
            va_path = task_root / f"{model_name}.va"
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            (_write_poly_va if model_name == "poly3_ridge" else _write_spline_va)(payload, va_path)
            start = time.perf_counter()
            for _ in range(50):
                evaluator(payload, dense[:1000])
            latency = (time.perf_counter() - start) / 50000.0
            error = prediction - y_test
            rows.append(
                {
                    "task": task,
                    "model": model_name,
                    "test_rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                    "test_r2": float(r2_score(y_test, prediction)),
                    "test_p95_abs_error": float(np.percentile(np.abs(error), 95)),
                    "test_max_abs_error": float(np.max(np.abs(error))),
                    "test_finite_rate": float(np.isfinite(prediction).mean()),
                    "dense_finite_rate": float(np.isfinite(dense_prediction).mean()),
                    "independent_replay_max_abs_error": replay,
                    "formula_chars": len(json_path.read_text(encoding="utf-8")),
                    "veriloga_lines": len(va_path.read_text(encoding="utf-8").splitlines()),
                    "latency_per_point_s": latency,
                    "fitted_terms": len(payload.get("coefficients", [])) if model_name == "poly3_ridge" else sum(len(feature["breaks"]) - 1 for feature in payload["features"]),
                    "validation_used_for_selection": False,
                    "test_isolation": True,
                    "json_path": str(json_path.relative_to(ROOT)),
                    "veriloga_path": str(va_path.relative_to(ROOT)),
                }
            )
    return pd.DataFrame(rows)


def write_protocol(args) -> None:
    edge_rows = []
    for stage_index, stage in enumerate(bayesian_symbolic.PHYSICS_INFORMED_SYMBOLIC_STAGES, 1):
        for function in stage["lib"]:
            edge_rows.append({"task_family": "DC/capacitance", "stage": stage_index, "function": function, "r2_threshold": stage["r2_threshold"]})
    for stage_index, stage in enumerate(bayesian_symbolic.AC_RESPONSE_SYMBOLIC_STAGES, 1):
        for function in stage["lib"]:
            edge_rows.append({"task_family": "AC", "stage": stage_index, "function": function, "r2_threshold": stage["r2_threshold"]})
    pd.DataFrame(edge_rows).to_csv(args.output / "edgewise_candidate_library.csv", index=False)

    candidate_rows: list[dict] = []
    mask_rows: list[dict] = []
    for task in ("I_dark", "I_photo", "Capacitance"):
        payload = json.loads(_formula_path(PRIMARY[task]).read_text(encoding="utf-8"))
        task_key = str(payload["task_key"])
        spec = shared.TASKS[task_key]
        inputs = tuple(payload["input_scalers"].keys())
        library = object.__new__(PhysicalFeatureLibrary)
        library.spec = spec
        library.inputs = inputs
        library.generic_symbolic_terms = bool(payload.get("generic_symbolic_terms", False))
        terms = library._build_terms()
        centers = np.asarray(payload["feature_centers"], dtype=np.float64)
        scales = np.asarray(payload["feature_scales"], dtype=np.float64)
        if len(terms) != len(centers) or len(terms) != len(scales):
            raise RuntimeError(f"Candidate/normalizer mismatch for {task}")
        for index, term in enumerate(terms):
            candidate_rows.append(
                {
                    "task": task,
                    "candidate_index": index,
                    "candidate_name": term.name,
                    "computational_expression": term.expression,
                    "dependencies": ",".join(term.dependencies),
                    "feature_center": float(centers[index]),
                    "feature_scale": float(scales[index]),
                    "normalization": "(raw_term-feature_center)/feature_scale",
                    "submodel": "mechanism_gated",
                }
            )
        fake_model = object.__new__(SymbolicGatedKAN)
        fake_model.spec = spec
        for branch in payload.get("branches", []):
            for index, term in enumerate(terms):
                mask_rows.append(
                    {
                        "task": task,
                        "branch": branch["name"],
                        "role": branch.get("role", "mechanism"),
                        "drive": branch.get("drive", "all"),
                        "candidate_index": index,
                        "candidate_name": term.name,
                        "allowed": bool(fake_model._term_allowed(str(branch["name"]), term.name)),
                    }
                )
        for index, term in enumerate(terms):
            mask_rows.append(
                {
                    "task": task,
                    "branch": "higher_order_symbolic_residual",
                    "role": "additive_residual",
                    "drive": "all",
                    "candidate_index": index,
                    "candidate_name": term.name,
                    "allowed": True,
                }
            )

    ac_payload = json.loads(_formula_path(PRIMARY["AC_Response"]).read_text(encoding="utf-8"))
    for feature_index, feature in enumerate(ac_payload["feature_definitions"]):
        for head in ("y0", "floor", "logfc"):
            candidate_rows.append(
                {
                    "task": "AC_Response",
                    "candidate_index": feature_index,
                    "candidate_name": feature["name"],
                    "computational_expression": json.dumps(feature, sort_keys=True),
                    "dependencies": ",".join(
                        value for key, value in feature.items()
                        if key in {"input", "left", "right"}
                    ),
                    "feature_center": "embedded_in_definition",
                    "feature_scale": "embedded_in_definition",
                    "normalization": "as serialized in feature_definitions",
                    "submodel": head,
                }
            )
            mask_rows.append(
                {
                    "task": "AC_Response",
                    "branch": head,
                    "role": "lowpass_condition_head",
                    "drive": "all",
                    "candidate_index": feature_index,
                    "candidate_name": feature["name"],
                    "allowed": True,
                }
            )
    candidate_rows.append(
        {
            "task": "AC_Response",
            "candidate_index": -1,
            "candidate_name": "log_p",
            "computational_expression": "p=0.5+exp(clip(log_p,log(0.05),log(20)))",
            "dependencies": "",
            "feature_center": "not_applicable",
            "feature_scale": "not_applicable",
            "normalization": "not_applicable",
            "submodel": "global_ungated_shape_parameter",
        }
    )
    pd.DataFrame(candidate_rows).to_csv(args.output / "gated_candidate_library.csv", index=False)
    pd.DataFrame(mask_rows).to_csv(args.output / "gated_mechanism_masks.csv", index=False)

    primary_runs = {
        "shared_general_student": {
            "tasks": ["I_dark", "I_photo", "Capacitance"],
            "fit_partitions": "gradient-fit plus conformal-calibration groups (80% total)",
            "validation_partition": "10%; strict lower validation objective retains checkpoint; exact ties retain earlier checkpoint",
            "test_partition": "10%; opened once after stage-3 freeze",
            "teacher": "f(x; E_q_beta[theta]) deterministic parameter-mean KAN",
            "sample_weight": "clip((sigma_q^2+floor^2)^(-1)/mean_precision,0.2,5); floor=max(0.1*median_positive_sigma,1e-12)",
            "objective": "w_y*Huber_0.05(student,TCAD)+0.25*Huber_0.05(student,teacher)+w_d*Huber_0.05(axis_derivative,finite_difference)+w_c*curve_shape+lambda_r*KAN_residual_MSE+lambda_k*mean_gate+lambda_b*mean(g*(1-g))",
            "observed_weight": 1.0,
            "optimizer": "fresh Adam (PyTorch defaults) per stage; gradient norm clipped to 10",
            "initialization": "seeded 0.02*N(0,1) coefficients; zero biases except photo loss-branch biases=-3; gate prior clip(0.45+0.10*sensitivity,0.05,0.95)",
            "validation_frequency_steps": 25,
            "stages": {
                "1": "lr=0.001; temperature=2; no regularizers",
                "2": "lr=0.001; temperature geometric 2->0.35; residual/complexity/binary weights each 1e-5",
                "3": "lr=0.0005; hard mask; temperature=0.2; residual weight=5e-5; complexity=binary=0",
            },
            "pruning_probability_threshold": 0.02,
            "pruning_temperature": 0.2,
            "pruning_score": "sigmoid(gate_logit/0.2)*abs(coefficient)",
        },
        "I_dark": {"steps": [300, 1000, 300], "budget": 36, "derivative_weight": 0.05, "curve_shape_weight": 0.02, "pruning": "equal mechanism quota floor(B/(M+1)); residual receives remaining slots"},
        "I_photo": {"steps": [300, 1500, 3000], "budget": 72, "realized_terms": 69, "derivative_weight": 0.0, "curve_shape_weight": 0.0, "pruning": "one anchor per mechanism, then global score order across mechanism/residual candidates; no generic terms and no reserved residual slots"},
        "Capacitance": {"steps": [800, 1800, 3200], "budget": 24, "derivative_weight": 0.0, "curve_shape_weight": 0.0, "pruning": "global top-B eligible gate-score entries"},
        "AC_Response": {
            "family": "floor(z)+(y0(z)-floor(z))/(1+(f/exp(logfc(z)))^p)",
            "teacher": "fixed 500-pass finite-MC q_beta predictive mean (not parameter-mean KAN)",
            "fit_partitions": "gradient-fit plus conformal-calibration groups",
            "validation_partition": "evaluation only; not used for structure/checkpoint selection",
            "test_partition": "opened once after hard pruning and refit",
            "features_per_head": 6,
            "heads": ["y0", "floor", "logfc"],
            "initialization": "scipy least_squares soft_l1, f_scale=0.08, max_nfev=10000",
            "objective": "MSE(student,teacher)+1.0*MSE(student,TCAD)+gate_penalty*sum(sigmoid(logit/temperature)) during competition",
            "optimizer": "one Adam optimizer, lr=0.02, PyTorch defaults; gradient norm clipped to 20",
            "stages": ["600 steps T=1 no gate penalty", "1800 steps T geometric 2->0.15, gate penalty 2e-4", "hard prune", "2200 steps T=0.1 no gate penalty"],
            "budget": 10,
            "mandatory_terms": ["y0 intercept", "floor intercept", "logfc intercept"],
            "pruning_score": "abs(coefficient)*sigmoid(gate_logit); remaining slots in descending score",
            "guards": "fc in [0.02,500] GHz; f/fc floored at 1e-12; p=0.5+exp(clip(log_p,log(0.05),log(20)))",
        },
        "source_configs": {task: str((directory / "run_config.json").relative_to(ROOT)) if (directory / "run_config.json").is_file() else "historical run predates run_config serialization; effective values registered here" for task, directory in PRIMARY.items()},
    }
    (args.output / "gated_primary_run_configs.json").write_text(
        json.dumps(primary_runs, indent=2), encoding="utf-8"
    )
    protocol = {
        "schema_version": 2,
        "variational_mean_conversion": "copy means of q_beta; retain mean output channel only for dark current, net photocurrent, and capacitance; AC uses the separately registered finite-MC q_beta predictive mean",
        "edgewise": {
            "sampling_domain": "empirical train-set activation range at each edge after train-only input standardization",
            "sampling_count": "all training rows (task-specific; no validation or test rows)",
            "fit_family": "c*f(a*x+b)+d",
            "fit_loss": "maximize squared Pearson correlation R2; solve c,d by ordinary least squares",
            "a_range": [-10, 10], "b_range": [-10, 10],
            "parameter_grid_points_per_axis": 101, "grid_refinement_iterations": 3,
            "selection_loss": "0.8*library_complexity + 0.2*log2(1+1e-5-R2)",
            "node_prune_threshold": 0.01, "edge_prune_threshold": 0.03,
            "candidate_library_csv": "edgewise_candidate_library.csv",
        },
        "pure_gated": {
            "feature_source": "physical_library.py and the AC gated-lowpass feature builder; complete definitions and frozen normalizers are in gated_candidate_library.csv",
            "normalization": "train-set mean/std for inputs and candidate terms; scales below 1e-12 replaced by 1",
            "fitting_loss": "fully expanded in gated_primary_run_configs.json",
            "gate_probability_threshold": 0.02,
            "hard_term_budgets": {"I_dark": 36, "I_photo": 72, "AC_Response": 10, "Capacitance": 24},
            "selection": "general student restores the strict-best validation checkpoint in each stage; AC is the registered exception whose validation split is evaluation-only; neither path uses test for fitting or pruning",
            "candidate_library_csv": "gated_candidate_library.csv",
            "mechanism_masks_csv": "gated_mechanism_masks.csv",
            "primary_run_configs_json": "gated_primary_run_configs.json",
        },
        "numerical_guards": ["softplus arguments clipped to [-40,40]", "positive denominators floored", "AC frequency floored at 1e-12", "AC fc clipped to [0.02,500] GHz", "photo loss exponents clipped to [0,8]"],
        "formula_assembly": "serialize train normalizers, selected terms, branch coefficients, mechanism drives, and output scaling to JSON; independent evaluator reads no KAN weights",
        "veriloga_translation": "replace serialized arithmetic with real-valued analog expressions; exp/log/tanh map directly; softplus uses guarded log1p-exp form; spline baseline is emitted as clamped piecewise cubic Horner form",
        "test_isolation": "test groups never participate in fitting, pruning, thresholding, or formula selection",
    }
    (args.output / "symbolic_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def copy_structured_summary(args) -> None:
    frames = []
    for source in (
        ROOT / "artifacts/results/structured_generalization/metrics_summary.csv",
        ROOT / "artifacts/results/structured_generalization_capacitance/metrics_summary.csv",
    ):
        frame = pd.read_csv(source)
        frame = frame[frame["model"].isin(["poly3_ridge", "spline_ridge"])].copy()
        frames.append(frame[["task", "scenario", "model", "n_runs", "rmse_target_mean", "r2_target_mean"]])
    pd.concat(frames, ignore_index=True).to_csv(args.output / "direct_baseline_structured_extrapolation.csv", index=False)


def write_report(args, formula: pd.DataFrame, stability: pd.DataFrame, baseline: pd.DataFrame) -> None:
    lines = [
        "# Four-Task Symbolic and Direct-Export Audit", "",
        "Only JSON formulas with no KAN inference path are treated as symbolic exports. "
        "All reported test sets are isolated from fitting and structural selection.", "",
        "## Pure formula fidelity", "",
        "| Task | Formula-network RMSE | Formula-TCAD RMSE | p95 / max TCAD error | Test / dense finite | Derivative finite | Terms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in formula.to_dict(orient="records"):
        lines.append(
            f"| {row['task']} | {row['formula_to_network_rmse']:.6g} | {row['formula_to_tcad_rmse']:.6g} | "
            f"{row['formula_to_tcad_p95_abs_error']:.6g} / {row['formula_to_tcad_max_abs_error']:.6g} | "
            f"{row['test_finite_rate']:.3f} / {row['dense_finite_rate']:.3f} | "
            f"{row['derivative_finite_rate']:.3f} | {int(row['active_terms'])} |"
        )
    lines.extend(["", "## Distillation-seed stability", "", "| Task | Term Jaccard mean (min) | Prediction disagreement RMSE mean (max) |", "| --- | ---: | ---: |"]) 
    for row in stability.to_dict(orient="records"):
        lines.append(f"| {row['task']} | {row['pairwise_term_jaccard_mean']:.3f} ({row['pairwise_term_jaccard_min']:.3f}) | {row['pairwise_prediction_rmse_mean']:.6g} ({row['pairwise_prediction_rmse_max']:.6g}) |")
    lines.extend(["", "## Direct response-surface exports", "", "Poly3-Ridge and Spline-Ridge were refit on the frozen seed-42 grouped split and independently replayed from JSON. Exact per-task metrics, latency, formula length, Verilog-A line count, and structured holdouts are provided in the accompanying CSV files.", "", "Important scope: the three stability seeds vary distillation optimization while keeping the registered teacher fixed. Dark current, net photocurrent, and capacitance use the q_beta parameter-mean deterministic KAN; AC uses the frozen 500-pass q_beta predictive mean. These runs measure conditional extraction stability, not q_beta-draw or model-seed structural stability.", ""])
    (args.output / "symbolic_export_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    write_protocol(args)
    formula = audit_formulas(args)
    stability = audit_stability(args)
    baseline = audit_baselines(args)
    copy_structured_summary(args)
    formula.to_csv(args.output / "four_task_formula_audit.csv", index=False)
    stability.to_csv(args.output / "formula_stability.csv", index=False)
    baseline.to_csv(args.output / "direct_symbolic_baselines.csv", index=False)
    write_report(args, formula, stability, baseline)
    print(formula.to_string(index=False))
    print(stability.to_string(index=False))
    print(baseline.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
