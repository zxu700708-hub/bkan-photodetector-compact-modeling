#!/usr/bin/env python3
"""Prepare and verify Spectre evidence for the replacement terminal-Q source.

This module deliberately rejects legacy charge-proxy evidence.  ``preflight``
freezes machine-readable DC/AC references and source hashes without claiming a
simulator pass.  ``verify`` accepts only PSF-ASCII output produced from the
exact replacement source and evaluates compilation, DC, AC-admittance,
transient chain-rule, and time-step-convergence checks.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MODEL_SOURCE = HERE / "ge_si_photodetector_terminal_charge.va"
ARTIFACT_SOURCE = (
    REPO_ROOT
    / "artifacts"
    / "results"
    / "terminal_charge_model"
    / "ge_si_photodetector_terminal_charge.va"
)
MODEL_JSON = (
    REPO_ROOT
    / "artifacts"
    / "results"
    / "terminal_charge_model"
    / "terminal_charge_models.json"
)
DARK_FORMULA = (
    REPO_ROOT
    / "artifacts"
    / "results"
    / "device_modeling"
    / "I-dark-bayes-results"
    / "symbolic_formula.txt"
)
NET_PHOTO_FORMULA = (
    REPO_ROOT
    / "artifacts"
    / "results"
    / "net_photocurrent_retrain"
    / "symbolic_fixed_teacher"
    / "seed_42"
    / "photo_current_symbolic_gated_kan"
    / "pure_symbolic_formula.json"
)
DECK_NAMES = (
    "testbench_dc_terminal_charge_ic618.scs",
    "testbench_ac_bias_temp_terminal_charge_ic618.scs",
    "testbench_transient_terminal_charge_ic618.scs",
    "testbench_transient_terminal_charge_fine_ic618.scs",
)

BIAS_VALUES = np.asarray([-3.0, -2.0, -1.0, -0.5, 0.0], dtype=np.float64)
TEMPERATURE_VALUES = np.asarray([290.0, 300.0, 310.0], dtype=np.float64)
VAC_MAG = 1.0e-3
V_MIN = -3.0
V_MAX = 0.0
SMOOTH_DELTA = 1.0e-18

NOMINAL_PARAMETERS = {
    "trap_assisted_recomb_A": 1.0e-9,
    "ge_sio2_recomb_velocity": 2500.0,
    "ge_si_recomb_velocity": 500.0,
    "active_layer_length": 40.0,
}

TOLERANCES = {
    "dc_relative": 2.0e-3,
    "dc_absolute_A": 1.0e-11,
    "ac_conductance_relative": 2.0e-2,
    "ac_conductance_absolute_S": 1.0e-10,
    "ac_dqdv_relative": 2.0e-2,
    "ac_dqdv_absolute_F": 1.0e-18,
    "transient_normalized_rmse": 8.0e-2,
    "transient_charge_relative": 5.0e-2,
    "transient_charge_absolute_C": 5.0e-18,
    "transient_step_convergence": 5.0e-2,
}

LINE_RE = re.compile(r'^"([^"]+)"\s+(.+)$')
LOG2 = float(np.log(2.0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_linear_formula(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    markers = (
        "Formula in Linear space:",
        "Formula in physical space:",
        "Formula in Physical space:",
    )
    marker = next((candidate for candidate in markers if candidate in text), None)
    if marker is None:
        raise ValueError(f"No linear-space formula found in {path}")
    lines: List[str] = []
    for raw_line in text.split(marker, 1)[1].splitlines():
        line = raw_line.strip()
        if not line:
            if lines:
                break
            continue
        lines.append(line)
    formula = " ".join(lines)
    if not formula:
        raise ValueError(f"Empty formula in {path}")
    return formula.split("=", 1)[1].strip() if "=" in formula else formula


def safe_sqrt_np(value, eps: float = 1.0e-30, delta: float = SMOOTH_DELTA):
    array = np.asarray(value, dtype=np.float64)
    positive = 0.5 * (array + np.sqrt(array * array + 4.0 * delta * delta))
    return np.sqrt(positive + eps)


def safe_exp_np(value):
    array = np.asarray(value, dtype=np.float64)
    return np.where(
        array < -80.0,
        np.exp(-80.0),
        np.where(array > 80.0, np.exp(80.0) * (1.0 + array - 80.0), np.exp(array)),
    )


def compile_formula(path: Path, variable_names: Tuple[str, ...]):
    expression = extract_linear_formula(path).replace("^", "**")
    code = compile(expression, str(path), "eval")

    def evaluate(*values):
        namespace = dict(zip(variable_names, values))
        namespace.update(
            {
                "sqrt": safe_sqrt_np,
                "exp": safe_exp_np,
                "abs": np.abs,
                "log": np.log,
                "sin": np.sin,
                "cos": np.cos,
                "tanh": np.tanh,
            }
        )
        return eval(code, {"__builtins__": {}}, namespace)

    return evaluate


def smax0(value):
    array = np.asarray(value, dtype=np.float64)
    return 0.5 * (array + np.sqrt(array * array + 4.0 * SMOOTH_DELTA**2))


def dark_boundary_gate(voltage):
    voltage = np.asarray(voltage, dtype=np.float64)
    u = np.clip((voltage + 0.2) / 0.2, 0.0, 1.0)
    return 1.0 - u * u * (3.0 - 2.0 * u)


def _unique_net_terms(payload: dict) -> Dict[int, dict]:
    terms: Dict[int, dict] = {}
    branches = [
        *payload["branches"],
        payload["residual_branch"],
        payload["generic_branch"],
    ]
    for branch in branches:
        for row in branch.get("terms", []):
            index = int(row["term_index"])
            if index in terms and terms[index]["term"] != row["term"]:
                raise ValueError(f"Conflicting net-photocurrent feature index {index}")
            terms[index] = row
    return terms


def _net_raw_feature(name: str, raw: Dict[str, np.ndarray], scalers: dict) -> np.ndarray:
    voltage = raw["light_voltage"]
    temperature = raw["simulation_temperature"]
    length = raw["active_layer_length"]
    trap = raw["trap_assisted_recomb_A"]
    sio2 = raw["ge_sio2_recomb_velocity"]
    si = raw["ge_si_recomb_velocity"]
    reverse_soft = np.logaddexp(0.0, -voltage / 0.2)
    reverse_zero = np.maximum(reverse_soft - LOG2, 0.0)

    def z(column: str) -> np.ndarray:
        item = scalers[column]
        return (raw[column] - float(item["mean"])) / float(item["scale"])

    features = {
        "one": np.ones_like(voltage),
        "light_voltage_z": z("light_voltage"),
        "light_voltage_z2": z("light_voltage") ** 2,
        "soft_reverse_light_voltage": reverse_soft,
        "tanh_reverse_light_voltage": np.tanh(-voltage),
        "reverse_saturation_light_voltage": 1.0 - np.exp(-reverse_zero),
        "forward_gate_light_voltage": 1.0 / (1.0 + np.exp(-voltage / 0.2)),
        "reverse_gate_light_voltage": 1.0 / (1.0 + np.exp(voltage / 0.2)),
        "temperature_z": z("simulation_temperature"),
        "inverse_temperature": 1.0 / temperature,
        "light_voltage_over_temperature": voltage / temperature,
        "length_z": z("active_layer_length"),
        "inverse_length": 1.0 / length,
        "light_voltage_over_length": voltage / length,
        "trap_A_z": z("trap_assisted_recomb_A"),
        "ge_sio2_velocity_z": z("ge_sio2_recomb_velocity"),
        "ge_si_velocity_z": z("ge_si_recomb_velocity"),
        "ge_sio2_velocity_over_length": sio2 / length,
        "ge_si_velocity_over_length": si / length,
        "reverse_drive_over_length": reverse_zero / length,
        "surface_recomb_sum_z": sio2 + si,
        "surface_recomb_over_length": (sio2 + si) / length,
        "reverse_drive_surface_recomb": reverse_zero * (sio2 + si) / length,
        "reverse_drive_trap_A": reverse_zero * trap,
        "collection_saturation_length": (1.0 - np.exp(-reverse_zero)) * length,
        "length_collection_saturation": 1.0 - np.exp(-np.maximum(length, 0.0) / 40.0),
        "thermal_surface_recomb_over_length": (sio2 + si) / (length * temperature),
        "reverse_thermal_surface_recomb": reverse_zero * (sio2 + si) / (length * temperature),
        "trap_A_times_length": trap * length,
        "trap_A_over_temperature": trap / temperature,
    }
    if name not in features:
        raise ValueError(f"Unsupported net-photocurrent feature: {name}")
    return features[name]


def evaluate_net_photo_graph(payload: dict, raw_values: np.ndarray) -> np.ndarray:
    """Independent NumPy replay of the KAN-free net-photocurrent graph."""
    inputs = tuple(payload["input_scalers"])
    raw_array = np.asarray(raw_values, dtype=np.float64)
    raw = {name: raw_array[:, index] for index, name in enumerate(inputs)}
    terms = _unique_net_terms(payload)
    centers = np.asarray(payload["feature_centers"], dtype=np.float64)
    scales = np.asarray(payload["feature_scales"], dtype=np.float64)
    phi = {
        index: (
            _net_raw_feature(row["term"], raw, payload["input_scalers"])
            - centers[index]
        )
        / scales[index]
        for index, row in terms.items()
    }

    def linear(branch: dict) -> np.ndarray:
        result = np.full(len(raw_array), float(branch["bias"]), dtype=np.float64)
        for row in branch.get("terms", []):
            result += float(row["coefficient_normalized_output"]) * phi[int(row["term_index"])]
        return result

    voltage = raw["light_voltage"]
    scale = float(payload["reverse_scale"])
    norm = float(payload["reverse_norm"])
    reverse_zero = np.maximum(np.logaddexp(0.0, -voltage / scale) - LOG2, 0.0) / norm
    forward_zero = np.maximum(np.logaddexp(0.0, voltage / scale) - LOG2, 0.0) / norm

    def drive(name: str) -> np.ndarray:
        if name == "reverse_zero":
            return reverse_zero
        if name == "forward_zero":
            return forward_zero
        if name == "reverse":
            return 1.0 / (1.0 + np.exp(-4.0 * reverse_zero))
        if name == "forward":
            return 1.0 / (1.0 + np.exp(-4.0 * forward_zero))
        return np.ones(len(raw_array), dtype=np.float64)

    values = []
    for branch in payload["branches"]:
        amplitude = np.logaddexp(0.0, linear(branch)) * drive(str(branch["drive"]))
        values.append((branch, amplitude))
    main = sum(
        float(branch["sign"]) * abs(float(branch["scale"])) * value
        for branch, value in values
        if branch["role"] == "main"
    )
    surface = sum(
        abs(float(branch["scale"])) * value
        for branch, value in values
        if branch["role"] == "surface_loss"
    )
    trap_loss = sum(
        abs(float(branch["scale"])) * value
        for branch, value in values
        if branch["role"] == "trap_loss"
    )
    additive = sum(
        float(branch["sign"]) * abs(float(branch["scale"])) * value
        for branch, value in values
        if branch["role"] == "additive"
    )
    normalized = (
        main * np.exp(-np.clip(surface, 0.0, 8.0)) * np.exp(-np.clip(trap_loss, 0.0, 8.0))
        + additive
        + linear(payload["residual_branch"])
        + linear(payload["generic_branch"])
    )
    return np.power(10.0, normalized * float(payload["output_scale"]))


class SymbolicCurrentReference:
    """Independent evaluator of dark plus paired net photocurrent."""

    def __init__(self) -> None:
        self.dark = compile_formula(
            DARK_FORMULA,
            (
                "dark_voltage",
                "active_layer_length",
                "trap_assisted_recomb_A",
                "ge_sio2_recomb_velocity",
                "ge_si_recomb_velocity",
                "simulation_temperature",
            ),
        )
        self.net_photo_payload = json.loads(NET_PHOTO_FORMULA.read_text(encoding="utf-8"))
        if (
            self.net_photo_payload.get("task_key") != "photo_current"
            or self.net_photo_payload.get("output_space") != "log10"
            or self.net_photo_payload.get("uses_kan_at_inference")
        ):
            raise ValueError("Invalid KAN-free net-photocurrent deployment payload")

    def _evaluate_unclipped(self, voltage, temperature=300.0) -> np.ndarray:
        voltage = np.atleast_1d(np.asarray(voltage, dtype=np.float64))
        temperature = np.broadcast_to(np.asarray(temperature, dtype=np.float64), voltage.shape)
        args = (
            voltage,
            NOMINAL_PARAMETERS["active_layer_length"],
            NOMINAL_PARAMETERS["trap_assisted_recomb_A"],
            NOMINAL_PARAMETERS["ge_sio2_recomb_velocity"],
            NOMINAL_PARAMETERS["ge_si_recomb_velocity"],
            temperature,
        )
        dark = np.asarray(self.dark(*args), dtype=np.float64)
        raw = {
            "light_voltage": voltage,
            "trap_assisted_recomb_A": np.full_like(
                voltage, NOMINAL_PARAMETERS["trap_assisted_recomb_A"]
            ),
            "ge_sio2_recomb_velocity": np.full_like(
                voltage, NOMINAL_PARAMETERS["ge_sio2_recomb_velocity"]
            ),
            "ge_si_recomb_velocity": np.full_like(
                voltage, NOMINAL_PARAMETERS["ge_si_recomb_velocity"]
            ),
            "active_layer_length": np.full_like(
                voltage, NOMINAL_PARAMETERS["active_layer_length"]
            ),
            "simulation_temperature": temperature,
        }
        order = tuple(self.net_photo_payload["input_scalers"])
        net_photo = evaluate_net_photo_graph(
            self.net_photo_payload,
            np.column_stack([raw[name] for name in order]),
        )
        return dark_boundary_gate(voltage) * smax0(dark) + smax0(net_photo)

    def evaluate(self, voltage, temperature=300.0) -> np.ndarray:
        voltage = np.atleast_1d(np.asarray(voltage, dtype=np.float64))
        return self._evaluate_unclipped(np.clip(voltage, V_MIN, V_MAX), temperature)

    def derivative(self, voltage, temperature=300.0, step: float = 1.0e-5) -> np.ndarray:
        """Return the Spectre active-branch small-signal tangent.

        At either registered voltage boundary, Spectre linearizes the branch
        selected at equality.  At ``V_MAX`` this is not the in-envelope
        backward derivative because ``net_pos(0)`` selects its zero branch.
        A forward raw-formula perturbation reproduces that implementation
        tangent without changing the bounded DC evaluator.
        """
        voltage = np.atleast_1d(np.asarray(voltage, dtype=np.float64))
        temperature = np.broadcast_to(np.asarray(temperature, dtype=np.float64), voltage.shape)
        result = np.zeros_like(voltage)
        interior = (voltage > V_MIN) & (voltage < V_MAX)
        if np.any(interior):
            result[interior] = (
                self._evaluate_unclipped(voltage[interior] + step, temperature[interior])
                - self._evaluate_unclipped(voltage[interior] - step, temperature[interior])
            ) / (2.0 * step)
        boundary = (voltage == V_MIN) | (voltage == V_MAX)
        if np.any(boundary):
            result[boundary] = (
                self._evaluate_unclipped(voltage[boundary] + step, temperature[boundary])
                - self._evaluate_unclipped(voltage[boundary], temperature[boundary])
            ) / step
        return result


def terminal_reference():
    import sys

    module_dir = REPO_ROOT / "bkan" / "device_modeling" / "terminal_charge"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from reference import TerminalChargeReference

    return TerminalChargeReference(MODEL_JSON)


def condition_frame(voltage, temperature=300.0) -> pd.DataFrame:
    voltage = np.atleast_1d(np.asarray(voltage, dtype=np.float64))
    temperature = np.broadcast_to(np.asarray(temperature, dtype=np.float64), voltage.shape)
    return pd.DataFrame(
        {
            "bias_v": voltage,
            **{name: value for name, value in NOMINAL_PARAMETERS.items()},
            "simulation_temperature": temperature,
        }
    )


def dec_frequencies(start: float = 1.0e6, stop: float = 1.0e12, points: int = 20):
    count = int(round(math.log10(stop / start) * points)) + 1
    return np.logspace(math.log10(start), math.log10(stop), count)


def write_references(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    current = SymbolicCurrentReference()
    charge = terminal_reference()

    dc_rows = []
    for temperature in TEMPERATURE_VALUES:
        voltage = np.linspace(V_MIN, V_MAX, 61)
        prediction = current.evaluate(voltage, temperature)
        dc_rows.extend(
            {
                "temperature_K": float(temperature),
                "bias_v": float(v),
                "device_current_A": float(i),
            }
            for v, i in zip(voltage, prediction)
        )
    pd.DataFrame(dc_rows).to_csv(directory / "dc_reference.csv", index=False)

    ac_rows = []
    for bias in BIAS_VALUES:
        for temperature in TEMPERATURE_VALUES:
            frame = condition_frame([bias], temperature)
            q_value, dqdv = charge.evaluate(frame)
            didv = current.derivative([bias], temperature)[0]
            for frequency in dec_frequencies():
                omega = 2.0 * math.pi * frequency
                ac_rows.append(
                    {
                        "bias_v": float(bias),
                        "temperature_K": float(temperature),
                        "frequency_Hz": float(frequency),
                        "terminal_charge_C": float(q_value[0]),
                        "dIdV_S": float(didv),
                        "dQdV_F": float(dqdv[0]),
                        "source_current_real_A": float(-VAC_MAG * didv),
                        "source_current_imag_A": float(-VAC_MAG * omega * dqdv[0]),
                    }
                )
    pd.DataFrame(ac_rows).to_csv(directory / "ac_reference.csv", index=False)
    return {"dc_rows": len(dc_rows), "ac_rows": len(ac_rows)}


def parse_number(text: str) -> Optional[complex]:
    cleaned = text.strip().replace("(", " ").replace(")", " ").replace(",", " ")
    values = []
    for token in cleaned.split()[:2]:
        try:
            values.append(float(token))
        except ValueError:
            pass
    if not values:
        return None
    return complex(values[0], values[1] if len(values) > 1 else 0.0)


def parse_psf_ascii(path: Path) -> Dict[str, np.ndarray]:
    columns: Dict[str, List[complex]] = {}
    in_values = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "VALUE":
                in_values = True
                continue
            if not in_values:
                continue
            if line == "END":
                break
            match = LINE_RE.match(line)
            if not match:
                continue
            value = parse_number(match.group(2))
            if value is not None:
                columns.setdefault(match.group(1), []).append(value)
    if not columns:
        raise RuntimeError(f"No PSF-ASCII VALUE records in {path}")
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"Misaligned PSF-ASCII columns in {path}: {lengths}")
    return {name: np.asarray(values, dtype=np.complex128) for name, values in columns.items()}


def within_tolerance(actual, expected, relative: float, absolute: float) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return np.abs(actual - expected) <= absolute + relative * np.abs(expected)


def indexed_files(root: Path, suffix: str) -> List[Path]:
    return sorted(path for path in root.rglob(f"*.{suffix}") if path.is_file())


def validate_logs(results: Path) -> dict:
    required = (
        "dc.log",
        "ac.log",
        "transient.log",
        "transient_fine.log",
        "spectre_version.txt",
    )
    failures = []
    for name in required:
        path = results / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing_or_empty:{name}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"fatal|segmentation|failed\s+to\s+converge|convergence\s+failure", text, re.I):
            failures.append(f"fatal_or_convergence:{name}")
        if name.endswith(".log"):
            if not re.search(r"spectre completes with 0 errors", text, re.I):
                failures.append(f"no_zero_error_completion:{name}")
    dc_log = results / "dc.log"
    compilation_via_dc = bool(
        dc_log.is_file()
        and re.search(
            r"spectre completes with 0 errors",
            dc_log.read_text(encoding="utf-8", errors="replace"),
            re.I,
        )
    )
    if not compilation_via_dc:
        failures.append("replacement_ahdl_not_executed_via_dc")
    return {
        "passed": not failures,
        "failures": failures,
        "compilation_method": "implicit_AHDL_compile_or_load_during_exact_DC_deck",
        "compilation_via_dc_passed": compilation_via_dc,
    }


def verify_dc(root: Path) -> dict:
    files = indexed_files(root / "dc.raw", "dc")
    current = SymbolicCurrentReference()
    rows = []
    failures = []
    for path in files:
        match = re.search(r"temperature_sweep-(\d+)", path.name)
        if not match:
            failures.append(f"unmapped_file:{path.name}")
            continue
        index = int(match.group(1))
        if index >= len(TEMPERATURE_VALUES):
            failures.append(f"temperature_index:{path.name}")
            continue
        data = parse_psf_ascii(path)
        bias = data["Vb_dc"].real
        source_current = data["Vdc:p"].real
        expected = current.evaluate(bias, TEMPERATURE_VALUES[index])
        actual = -source_current
        passed = within_tolerance(
            actual, expected, TOLERANCES["dc_relative"], TOLERANCES["dc_absolute_A"]
        )
        for v, ref, value, ok in zip(bias, expected, actual, passed):
            rows.append((float(v), float(TEMPERATURE_VALUES[index]), float(ref), float(value), bool(ok)))
    if len(files) != len(TEMPERATURE_VALUES):
        failures.append(f"dc_file_count:{len(files)}")
    if rows and not all(row[-1] for row in rows):
        failures.append("dc_numeric_mismatch")
    return {
        "passed": bool(rows) and not failures,
        "failures": failures,
        "files": len(files),
        "points": len(rows),
        "max_absolute_error_A": max((abs(row[3] - row[2]) for row in rows), default=None),
    }


def verify_ac(root: Path) -> dict:
    files = indexed_files(root / "ac.raw", "ac")
    current = SymbolicCurrentReference()
    charge = terminal_reference()
    failures = []
    points = 0
    conductance_errors = []
    dqdv_errors = []
    for path in files:
        match = re.search(r"bias_sweep-(\d+)_temperature_sweep-(\d+)", path.name)
        if not match:
            failures.append(f"unmapped_file:{path.name}")
            continue
        bias_index, temperature_index = map(int, match.groups())
        if bias_index >= len(BIAS_VALUES) or temperature_index >= len(TEMPERATURE_VALUES):
            failures.append(f"sweep_index:{path.name}")
            continue
        bias = BIAS_VALUES[bias_index]
        temperature = TEMPERATURE_VALUES[temperature_index]
        data = parse_psf_ascii(path)
        frequency = data.get("freq", data.get("frequency"))
        if frequency is None or "Vac:p" not in data:
            failures.append(f"missing_trace:{path.name}")
            continue
        frequency = frequency.real
        source_current = data["Vac:p"]
        actual_didv = -source_current.real / VAC_MAG
        actual_dqdv = -source_current.imag / (VAC_MAG * 2.0 * math.pi * frequency)
        expected_didv = np.full_like(frequency, current.derivative([bias], temperature)[0])
        _, derivative = charge.evaluate(condition_frame([bias], temperature))
        expected_dqdv = np.full_like(frequency, derivative[0])
        conductance_ok = within_tolerance(
            actual_didv,
            expected_didv,
            TOLERANCES["ac_conductance_relative"],
            TOLERANCES["ac_conductance_absolute_S"],
        )
        dqdv_ok = within_tolerance(
            actual_dqdv,
            expected_dqdv,
            TOLERANCES["ac_dqdv_relative"],
            TOLERANCES["ac_dqdv_absolute_F"],
        )
        if not np.all(conductance_ok):
            failures.append(f"conductance_mismatch:{path.name}")
        if not np.all(dqdv_ok):
            failures.append(f"dqdv_mismatch:{path.name}")
        conductance_errors.extend(np.abs(actual_didv - expected_didv).tolist())
        dqdv_errors.extend(np.abs(actual_dqdv - expected_dqdv).tolist())
        points += len(frequency)
    expected_files = len(BIAS_VALUES) * len(TEMPERATURE_VALUES)
    if len(files) != expected_files:
        failures.append(f"ac_file_count:{len(files)}")
    return {
        "passed": points > 0 and not failures,
        "failures": failures,
        "files": len(files),
        "points": points,
        "max_conductance_absolute_error_S": max(conductance_errors, default=None),
        "max_dqdv_absolute_error_F": max(dqdv_errors, default=None),
    }


TRANSIENT_TRACES = {
    "fast": ("Vfast:p", "a_fast"),
    "slow": ("Vslow:p", "a_slow"),
    "sine": ("Vsine:p", "a_sine"),
}


def transient_metrics(path: Path) -> Dict[str, dict]:
    data = parse_psf_ascii(path)
    time = data["time"].real
    if len(time) < 5 or not np.all(np.diff(time) > 0.0):
        raise RuntimeError(f"Invalid transient time vector in {path}")
    current_ref = SymbolicCurrentReference()
    charge_ref = terminal_reference()
    result = {}
    for name, (branch, node) in TRANSIENT_TRACES.items():
        voltage = data[node].real
        source_current = data[branch].real
        dc_current = current_ref.evaluate(voltage, 300.0)
        observed_dynamic = -source_current - dc_current
        _, dqdv = charge_ref.evaluate(condition_frame(voltage, 300.0))
        dvdt = np.gradient(voltage, time, edge_order=2)
        expected_dynamic = dqdv * dvdt
        transition_active = np.abs(dvdt) > 0.01 * max(
            float(np.max(np.abs(dvdt))), 1.0
        )
        transition_active[:2] = False
        transition_active[-2:] = False
        if np.count_nonzero(transition_active) < 5:
            raise RuntimeError(f"Insufficient active transient samples for {name}")

        transitions = []
        indices = np.flatnonzero(transition_active)
        starts = np.r_[indices[0], indices[1:][np.diff(indices) > 1]]
        ends = np.r_[indices[:-1][np.diff(indices) > 1], indices[-1]]
        # A pulse ramp is continuous but not differentiable at its two corners.
        # A centered finite difference assigns a half-slope at those samples,
        # whereas Spectre's ddt contribution follows the one-sided source slope.
        # Exclude exactly the first and last active sample of each segment from
        # the pointwise chain-rule metric.  The complete segment, including both
        # corners, remains in the independent integrated-charge check below.
        chain_rule_active = np.zeros_like(transition_active)
        for start, end in zip(starts, ends):
            interior_start = int(start) + 1
            interior_end = int(end) - 1
            if interior_end >= interior_start:
                chain_rule_active[interior_start : interior_end + 1] = True
        if np.count_nonzero(chain_rule_active) < 5:
            raise RuntimeError(
                f"Insufficient differentiable transient samples for {name}"
            )
        error = (
            observed_dynamic[chain_rule_active]
            - expected_dynamic[chain_rule_active]
        )
        denominator = max(
            float(np.sqrt(np.mean(expected_dynamic[chain_rule_active] ** 2))),
            1.0e-30,
        )
        normalized_rmse = float(np.sqrt(np.mean(error**2)) / denominator)

        for start, end in zip(starts, ends):
            start = max(int(start) - 1, 0)
            end = min(int(end) + 1, len(time) - 1)
            actual_delta = float(
                np.trapz(
                    observed_dynamic[start : end + 1], time[start : end + 1]
                )
            )
            q_value, _ = charge_ref.evaluate(condition_frame([voltage[start], voltage[end]], 300.0))
            expected_delta = float(q_value[1] - q_value[0])
            allowed = TOLERANCES["transient_charge_absolute_C"] + TOLERANCES[
                "transient_charge_relative"
            ] * abs(expected_delta)
            transitions.append(
                {
                    "actual_delta_Q_C": actual_delta,
                    "expected_delta_Q_C": expected_delta,
                    "absolute_error_C": abs(actual_delta - expected_delta),
                    "passed": abs(actual_delta - expected_delta) <= allowed,
                }
            )
        result[name] = {
            "normalized_rmse": normalized_rmse,
            "chain_rule_passed": normalized_rmse <= TOLERANCES["transient_normalized_rmse"],
            "transitions": transitions,
            "charge_integral_passed": bool(transitions) and all(item["passed"] for item in transitions),
            "finite": bool(
                np.all(np.isfinite(voltage))
                and np.all(np.isfinite(source_current))
                and np.all(np.isfinite(observed_dynamic))
            ),
            "time": time,
            "observed_dynamic": observed_dynamic,
            "active": chain_rule_active,
            "excluded_nondifferentiable_corner_samples": int(
                np.count_nonzero(transition_active)
                - np.count_nonzero(chain_rule_active)
            ),
        }
    return result


def only_analysis_file(root: Path) -> Path:
    files = indexed_files(root, "tran")
    if len(files) != 1:
        raise RuntimeError(f"Expected one transient analysis under {root}, found {len(files)}")
    return files[0]


def verify_transient(root: Path) -> dict:
    failures = []
    try:
        coarse = transient_metrics(only_analysis_file(root / "transient.raw"))
        fine = transient_metrics(only_analysis_file(root / "transient_fine.raw"))
    except Exception as exc:
        return {"passed": False, "failures": [f"parse_or_metric:{exc}"]}

    summary = {}
    for name in TRANSIENT_TRACES:
        coarse_item = coarse[name]
        fine_item = fine[name]
        time = coarse_item["time"]
        fine_interp = np.interp(time, fine_item["time"], fine_item["observed_dynamic"])
        active = coarse_item["active"]
        denominator = max(
            float(np.sqrt(np.mean(fine_interp[active] ** 2))), 1.0e-30
        )
        step_difference = float(
            np.sqrt(
                np.mean(
                    (coarse_item["observed_dynamic"][active] - fine_interp[active]) ** 2
                )
            )
            / denominator
        )
        passed = bool(
            coarse_item["finite"]
            and fine_item["finite"]
            and coarse_item["chain_rule_passed"]
            and fine_item["chain_rule_passed"]
            and coarse_item["charge_integral_passed"]
            and fine_item["charge_integral_passed"]
            and step_difference <= TOLERANCES["transient_step_convergence"]
        )
        if not passed:
            failures.append(f"transient_mismatch:{name}")
        summary[name] = {
            "passed": passed,
            "coarse_normalized_rmse": coarse_item["normalized_rmse"],
            "fine_normalized_rmse": fine_item["normalized_rmse"],
            "step_convergence_normalized_rmse": step_difference,
            "coarse_transitions": coarse_item["transitions"],
            "fine_transitions": fine_item["transitions"],
        }
    return {"passed": not failures, "failures": failures, "traces": summary}


def strip_arrays(value):
    if isinstance(value, dict):
        return {key: strip_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_arrays(item) for item in value]
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _relative_or_absolute(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def bind_returned_evidence(results: Path) -> dict:
    bundle_source = results.parent / "bundle" / MODEL_SOURCE.name
    current_hash = sha256(MODEL_SOURCE)
    bundle_hash = sha256(bundle_source) if bundle_source.is_file() else None
    evidence_files = []
    for path in sorted(item for item in results.rglob("*") if item.is_file()):
        evidence_files.append(
            {
                "path": path.relative_to(results.parent).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "passed": bool(bundle_hash and bundle_hash == current_hash),
        "evidence_root": _relative_or_absolute(results.parent, REPO_ROOT),
        "current_source_sha256": current_hash,
        "returned_bundle_source_sha256": bundle_hash,
        "returned_bundle_source": (
            bundle_source.relative_to(results.parent).as_posix()
            if bundle_source.is_file()
            else None
        ),
        "evidence_files": evidence_files,
    }


def preflight(output: Path) -> int:
    required = [
        MODEL_SOURCE,
        ARTIFACT_SOURCE,
        MODEL_JSON,
        DARK_FORMULA,
        NET_PHOTO_FORMULA,
    ]
    required.extend(HERE / name for name in DECK_NAMES)
    missing = [str(path) for path in required if not path.is_file()]
    checks = {
        "required_files_present": not missing,
        "replacement_source_matches_artifact": False,
        "replacement_module_identity": False,
        "portable_replacement_only_decks": False,
    }
    failures = [f"missing:{path}" for path in missing]
    if not missing:
        source = MODEL_SOURCE.read_text(encoding="utf-8")
        checks["replacement_source_matches_artifact"] = sha256(MODEL_SOURCE) == sha256(ARTIFACT_SOURCE)
        checks["replacement_module_identity"] = bool(
            "module ge_si_photodetector_terminal_charge" in source
            and "ddt(q_terminal)" in source
            and "net_photocurrent_model" in source
            and "dark_current + light_power * photo_current" in source
            and "q_proxy" not in source
        )
        deck_text = "\n".join((HERE / name).read_text(encoding="utf-8") for name in DECK_NAMES)
        checks["portable_replacement_only_decks"] = bool(
            "/home/" not in deck_text
            and "charge_proxy" not in deck_text
            and "ge_si_pdet_fixed" not in deck_text
            and deck_text.count('ahdl_include "ge_si_photodetector_terminal_charge.va"')
            == len(DECK_NAMES)
        )
        failures.extend(name for name, passed in checks.items() if not passed)

    references = write_references(output.parent / "reference") if not missing else {}
    hashes = {str(path.relative_to(REPO_ROOT)): sha256(path) for path in required if path.is_file()}
    payload = {
        "schema_version": 1,
        "evidence_scope": "CORRECTED_NET_PHOTOCURRENT_TERMINAL_Q",
        "status": "ready_to_run" if not failures else "failed",
        "simulator_execution_status": "pending",
        "legacy_proxy_evidence_transfer": False,
        "checks": checks,
        "failures": failures,
        "tolerances": TOLERANCES,
        "references": references,
        "sha256": hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


def verify(results: Path, output: Path) -> int:
    checks = {
        "source_binding": bind_returned_evidence(results),
        "logs_and_compilation": validate_logs(results),
        "dc": verify_dc(results),
        "ac_admittance": verify_ac(results),
        "transient_stability": verify_transient(results),
    }
    failures = [name for name, result in checks.items() if not result.get("passed")]
    payload = {
        "schema_version": 1,
        "evidence_scope": "CORRECTED_NET_PHOTOCURRENT_TERMINAL_Q",
        "status": "passed" if not failures else "failed",
        "legacy_proxy_evidence_transfer": False,
        "simulator_execution_status": {
            "spectre_compilation": (
                "passed"
                if checks["source_binding"].get("passed")
                and checks["logs_and_compilation"].get("compilation_via_dc_passed")
                else "failed"
            ),
            "spectre_dc": "passed" if checks["dc"].get("passed") else "failed",
            "spectre_ac_admittance": (
                "passed" if checks["ac_admittance"].get("passed") else "failed"
            ),
            "spectre_transient_stability": (
                "passed" if checks["transient_stability"].get("passed") else "failed"
            ),
        },
        "model_sha256": sha256(MODEL_SOURCE),
        "tolerances": TOLERANCES,
        "failed_checks": failures,
        "checks": strip_arrays(checks),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--results", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required")
    if args.command == "preflight":
        return preflight(args.output)
    return verify(args.results, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
