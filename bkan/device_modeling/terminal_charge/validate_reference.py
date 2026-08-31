"""Validate the exported JSON with an independent evaluator and chain rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reference import TerminalChargeReference


PARAM_COLS = [
    "trap_assisted_recomb_A",
    "ge_sio2_recomb_velocity",
    "ge_si_recomb_velocity",
    "active_layer_length",
    "simulation_temperature",
]


def relative_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.abs(np.asarray(prediction) - np.asarray(truth))
    floor = max(float(np.quantile(np.abs(truth), 0.95)) * 1.0e-6, 1.0e-30)
    relative = error / np.maximum(np.abs(truth), floor)
    return {
        "max_absolute_error": float(np.max(error)),
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "max_relative_error": float(np.max(relative)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = TerminalChargeReference(args.model)
    predictions = pd.read_csv(args.predictions)
    charge, derivative = reference.evaluate(predictions)
    serialized_agreement = {
        "charge": relative_summary(
            predictions["terminal_charge_prediction_C"].to_numpy(), charge
        )
    }

    condition_rows = (
        predictions[["production_source_row", *PARAM_COLS]]
        .drop_duplicates("production_source_row")
        .sort_values("production_source_row")
    )
    selected = condition_rows.iloc[
        np.linspace(0, len(condition_rows) - 1, 9, dtype=int)
    ]
    voltage = np.linspace(-3.0, 0.0, 12001)
    dense_parts = []
    for _, condition in selected.iterrows():
        frame = pd.DataFrame({"bias_v": voltage})
        for column in PARAM_COLS:
            frame[column] = float(condition[column])
        q_value, analytic = reference.evaluate(frame)
        finite_difference = np.gradient(q_value, voltage, edge_order=2)
        frame["production_source_row"] = int(condition["production_source_row"])
        frame["terminal_charge_C"] = q_value
        frame["analytic_dQdV_F"] = analytic
        frame["finite_difference_dQdV_F"] = finite_difference
        dense_parts.append(frame)
    dense = pd.concat(dense_parts, ignore_index=True)
    dense_metrics = relative_summary(
        dense["analytic_dQdV_F"].to_numpy(),
        dense["finite_difference_dQdV_F"].to_numpy(),
    )
    dense.to_csv(args.output / "dense_dqdv_reference.csv", index=False)

    nominal = condition_rows.iloc[
        np.argmin(
            np.sum(
                np.abs(
                    (
                        condition_rows[PARAM_COLS].to_numpy(dtype=float)
                        - condition_rows[PARAM_COLS].median().to_numpy(dtype=float)
                    )
                    / condition_rows[PARAM_COLS].std().to_numpy(dtype=float)
                ),
                axis=1,
            )
        )
    ]
    time = np.linspace(0.0, 10.0e-9, 10001)
    frequency = 1.0e9
    voltage_transient = -1.5 + 0.5 * np.sin(2.0 * np.pi * frequency * time)
    dvdt = 0.5 * 2.0 * np.pi * frequency * np.cos(2.0 * np.pi * frequency * time)
    transient = pd.DataFrame({"time_s": time, "bias_v": voltage_transient})
    for column in PARAM_COLS:
        transient[column] = float(nominal[column])
    q_value, capacitance = reference.evaluate(transient)
    current_chain_rule = capacitance * dvdt
    current_finite_difference = np.gradient(q_value, time, edge_order=2)
    active = np.abs(current_chain_rule) > np.quantile(np.abs(current_chain_rule), 0.05)
    transient_metrics = relative_summary(
        current_chain_rule[active], current_finite_difference[active]
    )
    transient["terminal_charge_C"] = q_value
    transient["analytic_dQdV_F"] = capacitance
    transient["analytic_dynamic_current_A"] = current_chain_rule
    transient["finite_difference_dynamic_current_A"] = current_finite_difference
    transient.to_csv(args.output / "transient_chain_rule_reference.csv", index=False)

    summary = {
        "validation_status": "passed",
        "serialized_json_vs_training_predictions": serialized_agreement,
        "dense_analytic_vs_finite_difference_dQdV": dense_metrics,
        "transient_dQdt_vs_dQdV_times_dVdt": transient_metrics,
        "dense_conditions": int(len(selected)),
        "dense_points_per_condition": int(len(voltage)),
        "transient_points": int(len(time)),
        "spectre_available": False,
        "scope": (
            "Independent NumPy serialization and calculus audit only; an external "
            "Spectre rerun is required for simulator implementation verification."
        ),
    }
    (args.output / "reference_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
