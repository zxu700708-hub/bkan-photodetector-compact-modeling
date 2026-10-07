"""NumPy/pandas-only replay for physics_balanced_sparse_v1 exports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _atom(atom: dict, standardized: dict[str, np.ndarray], scales: dict[str, float]):
    name = atom["input"]
    values = standardized[name]
    if atom["kind"] == "power":
        power = int(atom["value"])
        value = values**power
        derivative = (
            power * values ** (power - 1) / scales[name]
            if power
            else np.zeros_like(values)
        )
    elif atom["kind"] == "hinge3":
        positive = np.maximum(values - float(atom["value"]), 0.0)
        value = positive**3
        derivative = 3.0 * positive**2 / scales[name]
    else:
        raise ValueError(f"Unsupported atom kind: {atom['kind']}")
    return value, derivative


def _feature(
    feature: dict,
    standardized: dict[str, np.ndarray],
    scales: dict[str, float],
    axis: str,
):
    values = []
    derivatives = []
    for atom in feature["atoms"]:
        value, derivative = _atom(atom, standardized, scales)
        values.append(value)
        derivatives.append(derivative if atom["input"] == axis else np.zeros_like(value))
    product = np.ones_like(values[0])
    for value in values:
        product *= value
    derivative = np.zeros_like(product)
    for index, local_derivative in enumerate(derivatives):
        if not np.any(local_derivative):
            continue
        term = local_derivative.copy()
        for other, value in enumerate(values):
            if other != index:
                term *= value
        derivative += term
    return (
        (product - float(feature["center"])) / float(feature["scale"]),
        derivative / float(feature["scale"]),
    )


def replay_current(payload: dict, frame: pd.DataFrame):
    formula = payload["current_formula"]
    inputs = payload["input_order"]
    scalers = formula["input_scalers"]
    means = {name: float(scalers[name]["mean"]) for name in inputs}
    scales = {name: float(scalers[name]["scale"]) for name in inputs}
    standardized = {
        name: (frame[name].to_numpy(dtype=np.float64) - means[name]) / scales[name]
        for name in inputs
    }
    reference = dict(standardized)
    axis = inputs[0]
    reference[axis] = np.full(
        len(frame),
        (float(formula["reference_voltage"]) - means[axis]) / scales[axis],
    )
    columns = [np.ones(len(frame), dtype=np.float64)]
    derivative_columns = [np.zeros(len(frame), dtype=np.float64)]
    for feature in formula["features"]:
        value, derivative = _feature(feature, standardized, scales, axis)
        if any(atom["input"] == axis for atom in feature["atoms"]):
            reference_value, _ = _feature(feature, reference, scales, axis)
            value -= reference_value
        columns.append(value)
        derivative_columns.append(derivative)
    coefficient = np.asarray(formula["coefficients"], dtype=np.float64)
    return np.column_stack(columns) @ coefficient, np.column_stack(derivative_columns) @ coefficient


def replay_charge(payload: dict, frame: pd.DataFrame):
    model = payload["charge_basis"]
    names = model["parameter_names"]
    standardized = (
        frame[names].to_numpy(dtype=np.float64)
        - np.asarray(model["parameter_mean"], dtype=np.float64)
    ) / np.asarray(model["parameter_scale"], dtype=np.float64)
    voltage = frame["bias_v"].to_numpy(dtype=np.float64)
    powers = np.asarray(model["parameter_powers"], dtype=int)
    n_parameter_terms = len(powers)
    charge = np.zeros(len(frame), dtype=np.float64)
    derivative = np.zeros(len(frame), dtype=np.float64)
    for term in payload["terms"]:
        voltage_index, parameter_index = divmod(int(term["index"]), n_parameter_terms)
        parameter = np.prod(standardized ** powers[parameter_index], axis=1)
        if voltage_index < 3:
            degree = voltage_index + 1
            basis = voltage**degree
            basis_derivative = degree * voltage ** (degree - 1)
        else:
            knot = float(model["voltage_knots_V"][voltage_index - 3])
            positive = np.maximum(voltage - knot, 0.0)
            basis = positive**3 - max(-knot, 0.0) ** 3
            basis_derivative = 3.0 * positive**2
        coefficient = float(term["coefficient"])
        charge += coefficient * parameter * basis
        derivative += coefficient * parameter * basis_derivative
    scale = float(model["q_scale_C"])
    return charge * scale, derivative * scale


def replay(payload: dict, frame: pd.DataFrame):
    if payload["task"] == "Q_terminal":
        return replay_charge(payload, frame)
    return replay_current(payload, frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.formula.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.input)
    prediction, derivative = replay(payload, frame)
    result = frame.copy()
    result["formula_prediction"] = prediction
    result["formula_axis_derivative"] = derivative
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
