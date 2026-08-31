"""Independent NumPy evaluator for exported terminal-charge model JSON."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _parameter_basis(z: np.ndarray, powers: List[List[int]]) -> np.ndarray:
    result = np.ones((len(z), len(powers)), dtype=np.float64)
    for column, power in enumerate(powers):
        for feature, exponent in enumerate(power):
            if exponent:
                result[:, column] *= z[:, feature] ** exponent
    return result


def _voltage_basis(
    voltage: np.ndarray, knots: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    values = [voltage, voltage * voltage, voltage * voltage * voltage]
    slopes = [np.ones_like(voltage), 2.0 * voltage, 3.0 * voltage * voltage]
    for knot in knots:
        positive = np.maximum(voltage - knot, 0.0)
        values.append(positive**3 - max(-float(knot), 0.0) ** 3)
        slopes.append(3.0 * positive**2)
    return np.column_stack(values), np.column_stack(slopes)


class TerminalChargeReference:
    def __init__(self, model_path: Path):
        document = json.loads(model_path.read_text(encoding="utf-8"))
        model = document["exported_model"]
        self.parameter_names = model["parameter_names"]
        self.parameter_mean = np.asarray(model["parameter_mean"], dtype=np.float64)
        self.parameter_scale = np.asarray(model["parameter_scale"], dtype=np.float64)
        self.parameter_powers = model["parameter_powers"]
        self.knots = np.asarray(model["voltage_knots_V"], dtype=np.float64)
        self.coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        self.q_scale = float(model["q_scale_C"])

    def evaluate(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        parameters = frame[self.parameter_names].values.astype(np.float64, copy=False)
        z = (parameters - self.parameter_mean) / self.parameter_scale
        parameter = _parameter_basis(z, self.parameter_powers)
        voltage = frame["bias_v"].values.astype(np.float64, copy=False)
        value, slope = _voltage_basis(voltage, self.knots)
        value_design = np.einsum("ni,nj->nij", value, parameter).reshape(len(frame), -1)
        slope_design = np.einsum("ni,nj->nij", slope, parameter).reshape(len(frame), -1)
        return (
            self.q_scale * (value_design @ self.coefficients),
            self.q_scale * (slope_design @ self.coefficients),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    reference = TerminalChargeReference(args.model)
    charge, derivative = reference.evaluate(frame)
    result = frame.copy()
    result["terminal_charge_reference_C"] = charge
    result["dQdV_reference_F"] = derivative
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
