"""Portable independent replay of physics_balanced_sparse_v1 JSON formulas."""

import numpy as np


def _atom(atom, standardized, scales):
    name = atom["input"]
    values = standardized[name]
    if atom["kind"] == "power":
        power = int(atom["value"])
        value = values ** power
        derivative = (power * values ** (power - 1) / scales[name]
                      if power else np.zeros_like(values))
    elif atom["kind"] == "hinge3":
        positive = np.maximum(values - float(atom["value"]), 0.0)
        value = positive ** 3
        derivative = 3.0 * positive ** 2 / scales[name]
    else:
        raise ValueError("Unsupported atom kind: {}".format(atom["kind"]))
    return value, derivative


def _feature(feature, standardized, scales, axis):
    atoms = [_atom(atom, standardized, scales) for atom in feature["atoms"]]
    product = np.ones_like(atoms[0][0])
    for value, _ in atoms:
        product *= value
    derivative = np.zeros_like(product)
    for index, (value, local_derivative) in enumerate(atoms):
        if feature["atoms"][index]["input"] != axis:
            continue
        term = local_derivative.copy()
        for other, (other_value, _) in enumerate(atoms):
            if other != index:
                term *= other_value
        derivative += term
    return ((product - float(feature["center"])) / float(feature["scale"]),
            derivative / float(feature["scale"]))


def replay_current(payload, frame):
    formula = payload["current_formula"]
    inputs = payload["input_order"]
    scalers = formula["input_scalers"]
    means = {name: float(scalers[name]["mean"]) for name in inputs}
    scales = {name: float(scalers[name]["scale"]) for name in inputs}
    standardized = {
        name: (np.asarray(frame[name], dtype=np.float64) - means[name]) / scales[name]
        for name in inputs
    }
    reference = dict(standardized)
    axis = inputs[0]
    reference[axis] = np.full(len(frame),
                              (float(formula["reference_voltage"]) - means[axis]) / scales[axis])
    columns = [np.ones(len(frame), dtype=np.float64)]
    derivative_columns = [np.zeros(len(frame), dtype=np.float64)]
    for feature in formula["features"]:
        value, derivative = _feature(feature, standardized, scales, axis)
        if any(atom["input"] == axis for atom in feature["atoms"]):
            reference_value, _ = _feature(feature, reference, scales, axis)
            value -= reference_value
        columns.append(value)
        derivative_columns.append(derivative)
    coefficients = np.asarray(formula["coefficients"], dtype=np.float64)
    return (np.column_stack(columns) @ coefficients,
            np.column_stack(derivative_columns) @ coefficients)


def replay_charge(payload, frame):
    model = payload["charge_basis"]
    names = model["parameter_names"]
    standardized = ((np.asarray(frame[names], dtype=np.float64)
                     - np.asarray(model["parameter_mean"], dtype=np.float64))
                    / np.asarray(model["parameter_scale"], dtype=np.float64))
    voltage = np.asarray(frame["bias_v"], dtype=np.float64)
    powers = np.asarray(model["parameter_powers"], dtype=int)
    parameter_count = len(powers)
    charge = np.zeros(len(frame), dtype=np.float64)
    derivative = np.zeros(len(frame), dtype=np.float64)
    for term in payload["terms"]:
        voltage_index, parameter_index = divmod(int(term["index"]), parameter_count)
        parameter = np.prod(standardized ** powers[parameter_index], axis=1)
        if voltage_index < 3:
            degree = voltage_index + 1
            basis = voltage ** degree
            basis_derivative = degree * voltage ** (degree - 1)
        else:
            knot = float(model["voltage_knots_V"][voltage_index - 3])
            positive = np.maximum(voltage - knot, 0.0)
            basis = positive ** 3 - max(-knot, 0.0) ** 3
            basis_derivative = 3.0 * positive ** 2
        coefficient = float(term["coefficient"])
        charge += coefficient * parameter * basis
        derivative += coefficient * parameter * basis_derivative
    scale = float(model["q_scale_C"])
    return charge * scale, derivative * scale


def replay(payload, frame):
    return replay_charge(payload, frame) if payload["task"] == "Q_terminal" else replay_current(payload, frame)
