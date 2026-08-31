"""Fit and export a locally audited terminal-charge compact model.

The fit uses the production Q-V table for value coverage and independently
validated two-contact SSAC capacitance for dQ/dV regularization.  The voltage
basis is a reference-constrained cubic regression spline, so Q(0 V) is exactly
zero for every parameter combination.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .audit import (
        BIAS_COL,
        C_COL,
        PARAM_COLS,
        Q_COL,
        audit_terminal_charge,
        sha256_file,
    )
except ImportError:
    from audit import (  # type: ignore
        BIAS_COL,
        C_COL,
        PARAM_COLS,
        Q_COL,
        audit_terminal_charge,
        sha256_file,
    )


DEFAULT_PRODUCTION = Path(
    r"<FROZEN_TCAD_ROOT>/terminal_charge_timing\production\terminal_qv_dynamic_training.csv"
)
DEFAULT_CERTIFICATE = Path(
    r"<FROZEN_TCAD_ROOT>/terminal_charge_timing\production\terminal_qv_validation_certificate.csv"
)
DEFAULT_OUTPUT = Path("artifacts/results/terminal_charge_model")
DEFAULT_TEMPLATE = Path(
    "bkan/device_modeling/veriloga/ge_si_photodetector_terminal_charge.va"
)
SIMULATOR_EXECUTION_STATUS = {
    "spectre_compilation": "pending",
    "spectre_dc": "pending",
    "spectre_ac_admittance": "pending",
    "spectre_transient_stability": "pending",
    "legacy_proxy_evidence_transfer": False,
}
DEFAULT_VERILOGA = Path(
    "bkan/device_modeling/veriloga/ge_si_photodetector_terminal_charge.va"
)
DEFAULT_KNOTS = [
    -2.75,
    -2.5,
    -2.25,
    -2.0,
    -1.75,
    -1.5,
    -1.25,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
]


def polynomial_powers(n_features: int, degree: int) -> list[tuple[int, ...]]:
    powers = [tuple([0] * n_features)]
    for current_degree in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), current_degree):
            power = [0] * n_features
            for index in combo:
                power[index] += 1
            powers.append(tuple(power))
    return powers


def parameter_design(z: np.ndarray, powers: list[tuple[int, ...]]) -> np.ndarray:
    design = np.ones((len(z), len(powers)), dtype=np.float64)
    for column, power in enumerate(powers):
        for feature, exponent in enumerate(power):
            if exponent:
                design[:, column] *= z[:, feature] ** exponent
    return design


def voltage_design(
    voltage: np.ndarray, knots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    voltage = np.asarray(voltage, dtype=np.float64)
    basis = [voltage, voltage**2, voltage**3]
    derivative = [np.ones_like(voltage), 2.0 * voltage, 3.0 * voltage**2]
    for knot in knots:
        positive = np.maximum(voltage - knot, 0.0)
        reference = max(-float(knot), 0.0)
        basis.append(positive**3 - reference**3)
        derivative.append(3.0 * positive**2)
    return np.column_stack(basis), np.column_stack(derivative)


@dataclass
class TerminalChargeSplineModel:
    parameter_mean: np.ndarray
    parameter_scale: np.ndarray
    parameter_powers: list[tuple[int, ...]]
    voltage_knots: np.ndarray
    coefficients: np.ndarray
    q_scale_C: float
    c_scale_F: float
    alpha: float
    derivative_weight: float

    def _designs(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        parameters = frame[PARAM_COLS].to_numpy(dtype=np.float64)
        z = (parameters - self.parameter_mean) / self.parameter_scale
        p = parameter_design(z, self.parameter_powers)
        voltage = frame["bias_v"].to_numpy(dtype=np.float64)
        v, dv = voltage_design(voltage, self.voltage_knots)
        return (
            np.einsum("ni,nj->nij", v, p).reshape(len(frame), -1),
            np.einsum("ni,nj->nij", dv, p).reshape(len(frame), -1),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design, _ = self._designs(frame)
        return self.q_scale_C * (design @ self.coefficients)

    def predict_derivative(self, frame: pd.DataFrame) -> np.ndarray:
        _, derivative = self._designs(frame)
        return self.q_scale_C * (derivative @ self.coefficients)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "reference_constrained_tensor_cubic_spline_ridge",
            "parameter_names": PARAM_COLS,
            "parameter_mean": self.parameter_mean.tolist(),
            "parameter_scale": self.parameter_scale.tolist(),
            "parameter_powers": [list(power) for power in self.parameter_powers],
            "voltage_basis": "V,V^2,V^3,(max(V-k,0)^3-max(-k,0)^3)",
            "voltage_knots_V": self.voltage_knots.tolist(),
            "reference_bias_V": 0.0,
            "q_scale_C": self.q_scale_C,
            "c_scale_F": self.c_scale_F,
            "alpha": self.alpha,
            "derivative_weight": self.derivative_weight,
            "coefficients": self.coefficients.tolist(),
        }


def fit_model(
    q_frame: pd.DataFrame,
    derivative_frame: pd.DataFrame,
    alpha: float,
    derivative_weight: float,
    parameter_degree: int = 2,
    voltage_knots: list[float] | np.ndarray = DEFAULT_KNOTS,
) -> TerminalChargeSplineModel:
    parameter_values = q_frame[PARAM_COLS].to_numpy(dtype=np.float64)
    parameter_mean = parameter_values.mean(axis=0)
    parameter_scale = parameter_values.std(axis=0)
    parameter_scale = np.where(parameter_scale > 0.0, parameter_scale, 1.0)
    powers = polynomial_powers(len(PARAM_COLS), parameter_degree)
    knots = np.asarray(voltage_knots, dtype=np.float64)
    q_scale = max(float(np.quantile(np.abs(q_frame[Q_COL]), 0.95)), 1.0e-30)
    c_scale = max(float(np.quantile(np.abs(derivative_frame[C_COL]), 0.95)), 1.0e-30)
    seed_model = TerminalChargeSplineModel(
        parameter_mean,
        parameter_scale,
        powers,
        knots,
        np.zeros((3 + len(knots)) * len(powers)),
        q_scale,
        c_scale,
        alpha,
        derivative_weight,
    )
    q_design, _ = seed_model._designs(q_frame)
    _, derivative_design = seed_model._designs(derivative_frame)
    matrices = [q_design]
    targets = [q_frame[Q_COL].to_numpy(dtype=np.float64) / q_scale]
    if derivative_weight > 0.0:
        factor = math.sqrt(derivative_weight)
        matrices.append(factor * derivative_design * (q_scale / c_scale))
        targets.append(
            factor * derivative_frame[C_COL].to_numpy(dtype=np.float64) / c_scale
        )
    design = np.vstack(matrices)
    target = np.concatenate(targets)
    regularizer = math.sqrt(max(alpha, 0.0)) * np.eye(design.shape[1])
    coefficients = np.linalg.lstsq(
        np.vstack([design, regularizer]),
        np.concatenate([target, np.zeros(design.shape[1])]),
        rcond=None,
    )[0]
    seed_model.coefficients = coefficients
    return seed_model


def regression_metrics(
    truth: np.ndarray, prediction: np.ndarray, relative_floor: float
) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - truth
    absolute = np.abs(error)
    denominator = np.maximum(np.abs(truth), relative_floor)
    active = np.abs(truth) > 5.0 * relative_floor
    ss_total = float(np.sum((truth - truth.mean()) ** 2))
    ss_residual = float(np.sum(error**2))
    return {
        "n": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "r2": float(1.0 - ss_residual / ss_total) if ss_total else float("nan"),
        "median_relative_error": float(np.median(absolute / denominator)),
        "p95_relative_error": float(np.quantile(absolute / denominator, 0.95)),
        "max_relative_error": float(np.max(absolute / denominator)),
        "active_n": int(active.sum()),
        "active_p95_relative_error": (
            float(np.quantile(absolute[active] / np.abs(truth[active]), 0.95))
            if active.any()
            else float("nan")
        ),
        "max_absolute_error": float(np.max(absolute)),
        "relative_floor": float(relative_floor),
    }


def make_condition_split(
    source_rows: list[int], validated_rows: list[int], seed: int
) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    validated = np.asarray(sorted(set(validated_rows)), dtype=int)
    rng.shuffle(validated)
    n_validated = len(validated)
    n_val_validated = max(1, int(round(0.2 * n_validated)))
    n_test_validated = max(1, int(round(0.2 * n_validated)))
    split: dict[int, str] = {}
    for row in validated[:n_test_validated]:
        split[int(row)] = "test"
    for row in validated[n_test_validated : n_test_validated + n_val_validated]:
        split[int(row)] = "val"
    for row in validated[n_test_validated + n_val_validated :]:
        split[int(row)] = "train"

    remaining = np.asarray(sorted(set(source_rows).difference(split)), dtype=int)
    rng.shuffle(remaining)
    targets = {
        "test": int(round(0.15 * len(source_rows))),
        "val": int(round(0.15 * len(source_rows))),
    }
    for label in ("test", "val"):
        needed = max(0, targets[label] - sum(value == label for value in split.values()))
        for row in remaining[:needed]:
            split[int(row)] = label
        remaining = remaining[needed:]
    for row in remaining:
        split[int(row)] = "train"
    return split


def select_model(
    production: pd.DataFrame,
    evidence: pd.DataFrame,
    split_map: dict[int, str],
    alphas: list[float],
    derivative_weights: list[float],
    parameter_degrees: list[int],
    q_active_p95_limit: float,
) -> tuple[TerminalChargeSplineModel, pd.DataFrame, dict[str, dict[str, object]]]:
    q_train = production[production["split"].eq("train")]
    q_val = production[production["split"].eq("val")]
    c_train = evidence[evidence["split"].eq("train")]
    c_val = evidence[evidence["split"].eq("val")]
    if c_train.empty or c_val.empty:
        raise ValueError("Strict derivative evidence must populate train and validation splits.")
    q_floor = max(float(np.quantile(np.abs(q_train[Q_COL]), 0.95)) * 0.01, 1.0e-30)
    c_floor = max(float(np.quantile(np.abs(c_train[C_COL]), 0.95)) * 0.01, 1.0e-30)
    rows: list[dict[str, float | bool]] = []
    models: dict[tuple[int, float, float], TerminalChargeSplineModel] = {}
    for parameter_degree in parameter_degrees:
        for alpha in alphas:
            for weight in derivative_weights:
                model = fit_model(
                    q_train,
                    c_train,
                    alpha,
                    weight,
                    parameter_degree=parameter_degree,
                )
                models[(parameter_degree, alpha, weight)] = model
                q_metrics = regression_metrics(q_val[Q_COL], model.predict(q_val), q_floor)
                c_metrics = regression_metrics(
                    c_val[C_COL], model.predict_derivative(c_val), c_floor
                )
                rows.append(
                    {
                        "parameter_degree": parameter_degree,
                        "alpha": alpha,
                        "derivative_weight": weight,
                        "q_validation_r2": float(q_metrics["r2"]),
                        "q_validation_active_p95_relative_error": float(
                            q_metrics["active_p95_relative_error"]
                        ),
                        "q_validation_rmse_C": float(q_metrics["rmse"]),
                        "dqdv_validation_median_relative_error": float(
                            c_metrics["median_relative_error"]
                        ),
                        "dqdv_validation_p95_relative_error": float(
                            c_metrics["p95_relative_error"]
                        ),
                        "dqdv_validation_max_relative_error": float(
                            c_metrics["max_relative_error"]
                        ),
                    }
                )
    selection = pd.DataFrame(rows)
    selection["q_accuracy_eligible"] = (
        selection["q_validation_active_p95_relative_error"] <= q_active_p95_limit
    )
    eligible = selection[selection["q_accuracy_eligible"]]
    if eligible.empty:
        ranked = selection.sort_values(
            ["q_validation_active_p95_relative_error", "dqdv_validation_p95_relative_error"]
        )
        criterion = "fallback_minimum_q_error"
    else:
        ranked = eligible.sort_values(
            [
                "dqdv_validation_p95_relative_error",
                "dqdv_validation_median_relative_error",
                "q_validation_active_p95_relative_error",
                "parameter_degree",
                "alpha",
                "derivative_weight",
            ]
        )
        criterion = "minimum_dqdv_p95_subject_to_q_active_p95_limit"
    best = ranked.iloc[0]
    key = (
        int(best["parameter_degree"]),
        float(best["alpha"]),
        float(best["derivative_weight"]),
    )
    selection["selected"] = (
        selection["parameter_degree"].eq(key[0])
        & selection["alpha"].eq(key[1])
        & selection["derivative_weight"].eq(key[2])
    )
    selection["selection_criterion"] = criterion
    model = models[key]

    metrics: dict[str, dict[str, object]] = {}
    for label in ("train", "val", "test"):
        q_part = production[production["split"].eq(label)]
        c_part = evidence[evidence["split"].eq(label)]
        metrics[f"q_{label}"] = regression_metrics(
            q_part[Q_COL], model.predict(q_part), q_floor
        )
        metrics[f"dqdv_{label}"] = regression_metrics(
            c_part[C_COL], model.predict_derivative(c_part), c_floor
        )
    return model, selection, metrics


def _monomial_text(power: tuple[int, ...]) -> str:
    factors: list[str] = []
    for index, exponent in enumerate(power):
        factors.extend([f"z{index}"] * exponent)
    return " * ".join(factors) if factors else "1.0"


def verilog_q_function(model: TerminalChargeSplineModel) -> str:
    n_v = 3 + len(model.voltage_knots)
    n_p = len(model.parameter_powers)
    lines = [
        "  analog function real terminal_charge_model;",
        "    input vd_raw, trap_assisted_recomb_A, ge_sio2_recomb_velocity, ge_si_recomb_velocity, active_layer_length, sim_temp;",
        "    real vd_raw, trap_assisted_recomb_A, ge_sio2_recomb_velocity, ge_si_recomb_velocity, active_layer_length, sim_temp;",
        "    real " + ", ".join(f"z{i}" for i in range(len(PARAM_COLS))) + ";",
        "    real " + ", ".join(f"p{i}" for i in range(n_p)) + ";",
        "    real " + ", ".join(f"b{i}" for i in range(n_v)) + ";",
        "    real " + ", ".join(f"t{i}" for i in range(len(model.voltage_knots))) + ";",
        "    real acc;",
        "    begin",
    ]
    for index, (mean, scale) in enumerate(
        zip(model.parameter_mean, model.parameter_scale)
    ):
        variable = PARAM_COLS[index]
        if variable == "simulation_temperature":
            variable = "sim_temp"
        lines.append(f"      z{index} = ({variable} - {mean:.17e}) / {scale:.17e};")
    for index, power in enumerate(model.parameter_powers):
        lines.append(f"      p{index} = {_monomial_text(power)};")
    lines.extend(
        [
            "      b0 = vd_raw;",
            "      b1 = vd_raw * vd_raw;",
            "      b2 = b1 * vd_raw;",
        ]
    )
    for index, knot in enumerate(model.voltage_knots):
        lines.extend(
            [
                f"      if (vd_raw > {knot:.17e})",
                f"        t{index} = vd_raw - {knot:.17e};",
                "      else",
                f"        t{index} = 0.0;",
                f"      b{index + 3} = t{index} * t{index} * t{index} - {(-knot) ** 3:.17e};",
            ]
        )
    lines.append("      acc = 0.0;")
    coefficient_index = 0
    for v_index in range(n_v):
        for p_index in range(n_p):
            coefficient = model.coefficients[coefficient_index]
            lines.append(
                f"      acc = acc + {coefficient:.17e} * b{v_index} * p{p_index};"
            )
            coefficient_index += 1
    lines.extend(
        [
            f"      terminal_charge_model = {model.q_scale_C:.17e} * acc;",
            "    end",
            "  endfunction",
            "",
        ]
    )
    return "\n".join(lines)


def generate_veriloga(model: TerminalChargeSplineModel, template_path: Path) -> str:
    text = template_path.read_text(encoding="utf-8")
    function_pattern = re.compile(
        r"  analog function real terminal_charge_model;.*?  endfunction\s+",
        re.DOTALL,
    )
    text, count = function_pattern.subn(verilog_q_function(model), text, count=1)
    if count != 1:
        raise ValueError("Could not replace terminal-Q function in Verilog-A template.")
    text = re.sub(
        r"\A// .*?\n// Generated by bkan/device_modeling/terminal_charge/train.py",
        "// Locally audited terminal-charge dynamic model for Ge/Si photodetector\n"
        "// Generated by bkan/device_modeling/terminal_charge/train.py",
        text,
        count=1,
    )
    return text


def flatten_metrics(metrics: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for section, values in metrics.items():
        for metric, value in values.items():
            rows.append({"section": section, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def write_plots(
    q_predictions: pd.DataFrame,
    derivative_predictions: pd.DataFrame,
    selection: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(figsize=(5.2, 4.3))
    truth = q_predictions[Q_COL].to_numpy(dtype=float)
    prediction = q_predictions["terminal_charge_prediction_C"].to_numpy(dtype=float)
    axis.scatter(truth, prediction, s=9, alpha=0.45, color="#2864b7")
    limits = [min(truth.min(), prediction.min()), max(truth.max(), prediction.max())]
    axis.plot(limits, limits, color="black", linewidth=1.0)
    axis.set_xlabel("Production terminal charge (C)")
    axis.set_ylabel("Model terminal charge (C)")
    axis.set_title("Terminal-charge value fit")
    figure.tight_layout()
    figure.savefig(output / "01_terminal_charge_actual_vs_pred.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.2, 4.3))
    truth = derivative_predictions[C_COL].to_numpy(dtype=float)
    prediction = derivative_predictions["dQdV_model_F"].to_numpy(dtype=float)
    colors = derivative_predictions["split"].map(
        {"train": "#2864b7", "val": "#d17b0f", "test": "#b32d36"}
    )
    axis.scatter(truth, prediction, s=24, alpha=0.8, c=colors)
    limits = [min(truth.min(), prediction.min()), max(truth.max(), prediction.max())]
    axis.plot(limits, limits, color="black", linewidth=1.0)
    axis.set_xlabel("Strict two-contact SSAC capacitance (F)")
    axis.set_ylabel("Analytic model dQ/dV (F)")
    axis.set_title("Independent derivative validation")
    figure.tight_layout()
    figure.savefig(output / "02_terminal_charge_dqdv_validation.png", bbox_inches="tight")
    plt.close(figure)

    representative_row = int(
        q_predictions.groupby("production_source_row")[PARAM_COLS]
        .first()
        .sub(q_predictions[PARAM_COLS].median())
        .abs()
        .sum(axis=1)
        .idxmin()
    )
    curve = q_predictions[
        q_predictions["production_source_row"].eq(representative_row)
    ].sort_values("bias_v")
    figure, axis = plt.subplots(figsize=(5.4, 4.2))
    axis.plot(curve["bias_v"], curve[Q_COL], "o-", label="production Q-V")
    axis.plot(
        curve["bias_v"],
        curve["terminal_charge_prediction_C"],
        "s--",
        label="terminal-charge model",
    )
    axis.set_xlabel("Bias (V)")
    axis.set_ylabel("Terminal charge (C)")
    axis.set_title(f"Representative Q-V curve (source row {representative_row})")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "03_terminal_charge_curve_example.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.7, 4.3))
    for (degree, alpha), group in selection.groupby(["parameter_degree", "alpha"]):
        group = group.sort_values("derivative_weight")
        axis.semilogx(
            group["derivative_weight"].replace(0.0, 1.0e-4),
            100.0 * group["dqdv_validation_p95_relative_error"],
            marker="o",
            markersize=3,
            label=f"deg={degree}, alpha={alpha:g}",
        )
    chosen = selection[selection["selected"]].iloc[0]
    axis.scatter(
        max(float(chosen["derivative_weight"]), 1.0e-4),
        100.0 * float(chosen["dqdv_validation_p95_relative_error"]),
        s=75,
        facecolors="none",
        edgecolors="black",
        linewidth=1.4,
        label="selected",
    )
    axis.set_xlabel("Derivative loss weight (0 shown at 1e-4)")
    axis.set_ylabel("Validation dQ/dV p95 relative error (%)")
    axis.set_title("Terminal-charge model selection")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "04_terminal_charge_model_selection.png", bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--evidence", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template-veriloga", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--veriloga-output", type=Path, default=DEFAULT_VERILOGA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    parser.add_argument(
        "--derivative-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    )
    parser.add_argument("--parameter-degrees", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--q-active-p95-limit", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_terminal_charge(
        args.production,
        args.evidence,
        args.certificate,
    )
    production = pd.read_csv(args.production).copy()
    evidence = audit.evidence.copy()
    source_rows = sorted(production["production_source_row"].astype(int).unique())
    validated_rows = sorted(evidence["production_source_row"].astype(int).unique())
    split_map = make_condition_split(source_rows, validated_rows, args.seed)
    production["split"] = production["production_source_row"].astype(int).map(split_map)
    evidence["split"] = evidence["production_source_row"].astype(int).map(split_map)

    evaluation_model, selection, evaluation_metrics = select_model(
        production,
        evidence,
        split_map,
        args.alphas,
        args.derivative_weights,
        args.parameter_degrees,
        args.q_active_p95_limit,
    )
    selected = selection[selection["selected"]].iloc[0]
    final_model = fit_model(
        production,
        evidence,
        float(selected["alpha"]),
        float(selected["derivative_weight"]),
        parameter_degree=int(selected["parameter_degree"]),
    )
    q_floor = max(float(np.quantile(np.abs(production[Q_COL]), 0.95)) * 0.01, 1e-30)
    c_floor = max(float(np.quantile(np.abs(evidence[C_COL]), 0.95)) * 0.01, 1e-30)
    final_metrics = {
        "q_all_refit": regression_metrics(
            production[Q_COL], final_model.predict(production), q_floor
        ),
        "dqdv_all_refit": regression_metrics(
            evidence[C_COL], final_model.predict_derivative(evidence), c_floor
        ),
    }
    grid_derivative = final_model.predict_derivative(production)
    grid_charge = final_model.predict(production)
    physical_sanity = {
        "q_at_reference_max_abs_C": float(
            np.max(np.abs(final_model.predict(production.assign(bias_v=0.0))))
        ),
        "minimum_grid_dQdV_F": float(np.min(grid_derivative)),
        "maximum_grid_dQdV_F": float(np.max(grid_derivative)),
        "nonpositive_grid_dQdV_count": int(np.sum(grid_derivative <= 0.0)),
        "positive_reverse_bias_charge_count": int(
            np.sum((production["bias_v"].to_numpy() < 0.0) & (grid_charge > 0.0))
        ),
    }
    if physical_sanity["q_at_reference_max_abs_C"] != 0.0:
        raise RuntimeError("Exported charge model does not satisfy exact Q(0 V)=0.")
    if physical_sanity["nonpositive_grid_dQdV_count"] != 0:
        raise RuntimeError("Exported charge model has non-positive dQ/dV on the production grid.")
    if physical_sanity["positive_reverse_bias_charge_count"] != 0:
        raise RuntimeError("Exported charge model has positive Q at reverse bias.")
    metrics = {**evaluation_metrics, **final_metrics}

    args.output.mkdir(parents=True, exist_ok=True)
    args.veriloga_output.parent.mkdir(parents=True, exist_ok=True)
    q_predictions = production.copy()
    q_predictions["terminal_charge_prediction_C"] = final_model.predict(production)
    q_predictions["terminal_charge_error_C"] = (
        q_predictions["terminal_charge_prediction_C"] - q_predictions[Q_COL]
    )
    derivative_predictions = evidence.copy()
    derivative_predictions["dQdV_model_F"] = final_model.predict_derivative(evidence)
    derivative_predictions["dQdV_model_error_F"] = (
        derivative_predictions["dQdV_model_F"] - derivative_predictions[C_COL]
    )
    q_predictions.to_csv(args.output / "terminal_charge_predictions.csv", index=False)
    derivative_predictions.to_csv(
        args.output / "terminal_charge_derivative_predictions.csv", index=False
    )
    selection.to_csv(args.output / "terminal_charge_model_selection.csv", index=False)
    flatten_metrics(metrics).to_csv(args.output / "metrics_summary.csv", index=False)
    audit.condition_table.to_csv(
        args.output / "terminal_charge_audit_conditions.csv", index=False
    )
    (args.output / "terminal_charge_audit.json").write_text(
        json.dumps(audit.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    split_table = pd.DataFrame(
        [{"production_source_row": key, "split": value} for key, value in split_map.items()]
    ).sort_values("production_source_row")
    split_table.to_csv(args.output / "condition_split.csv", index=False)
    write_plots(q_predictions, derivative_predictions, selection, args.output)

    model_document = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_quantity": "full_grid_fixed_mesh_terminal_charge_with_representative_strict_evidence",
        "charge_conservation_statement": "Qp = Q_terminal; Qn = -Q_terminal",
        "validation_scope": (
            "Full-grid fixed-mesh Q-V extraction with representative strict "
            "two-contact dynamic validation coverage"
        ),
        "simulator_execution_status": dict(SIMULATOR_EXECUTION_STATUS),
        "audit": audit.summary,
        "selection": selection[selection["selected"]].iloc[0].to_dict(),
        "evaluation_metrics": evaluation_metrics,
        "refit_metrics": final_metrics,
        "physical_sanity": physical_sanity,
        "evaluation_model": evaluation_model.to_dict(),
        "exported_model": final_model.to_dict(),
        "split_seed": args.seed,
    }
    models_path = args.output / "terminal_charge_models.json"
    models_path.write_text(
        json.dumps(model_document, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    verilog = generate_veriloga(final_model, args.template_veriloga)
    local_verilog = args.output / "ge_si_photodetector_terminal_charge.va"
    local_verilog.write_text(verilog, encoding="utf-8")
    args.veriloga_output.write_text(verilog, encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "production": {"path": str(args.production), "sha256": sha256_file(args.production)},
            "certificate": {"path": str(args.certificate), "sha256": sha256_file(args.certificate)},
            "template_veriloga": {
                "path": str(args.template_veriloga),
                "sha256": sha256_file(args.template_veriloga),
            },
            "evidence": [
                {"path": path, "sha256": digest}
                for path, digest in zip(
                    audit.summary["evidence_paths"], audit.summary["evidence_sha256"]
                )
            ],
        },
        "training_script_sha256": sha256_file(Path(__file__)),
        "simulator_execution_status": dict(SIMULATOR_EXECUTION_STATUS),
        "outputs": {
            "models_sha256": sha256_file(models_path),
            "veriloga_sha256": sha256_file(local_verilog),
            "exported_veriloga": str(args.veriloga_output),
        },
    }
    (args.output / "terminal_charge_model_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "# Locally audited terminal-charge compact model",
        "",
        f"- Production rows: {len(production)} ({len(source_rows)} conditions)",
        f"- Strict validation conditions: {len(validated_rows)}",
        f"- Selected alpha: {float(selected['alpha']):.6g}",
        f"- Selected parameter degree: {int(selected['parameter_degree'])}",
        f"- Selected derivative weight: {float(selected['derivative_weight']):.6g}",
        f"- Held-out Q test R2: {float(evaluation_metrics['q_test']['r2']):.8f}",
        f"- Held-out Q active p95 relative error: {float(evaluation_metrics['q_test']['active_p95_relative_error']):.4%}",
        f"- Held-out dQ/dV test p95 relative error: {float(evaluation_metrics['dqdv_test']['p95_relative_error']):.4%}",
        f"- Full refit Q active p95 relative error: {float(final_metrics['q_all_refit']['active_p95_relative_error']):.4%}",
        f"- Full refit dQ/dV p95 relative error: {float(final_metrics['dqdv_all_refit']['p95_relative_error']):.4%}",
        "- Q(0 V)=0 is enforced analytically by the voltage basis.",
        "- The exported two-terminal branch uses Qn=-Qp and I_dynamic=ddt(Qp).",
        "- External Spectre compilation and DC/AC/transient execution remain required before any simulator-level verification or sign-off claim.",
    ]
    (args.output / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
