"""Version-specific export and Spectre checks for the 10/8/20 sparse students.

The frozen JSON files, not the historical 54-term Verilog-A source, define this
candidate. This module is also copied into a portable validation bundle.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOCAL_FORMULAS = HERE / "formulas"
FORMULA_ROOT = (HERE if LOCAL_FORMULAS.is_dir() else
                ROOT / "artifacts/experiments/physics_balanced_export_20260917")
FORMULA_DIR = FORMULA_ROOT / "consensus_3split_compact"
if LOCAL_FORMULAS.is_dir():
    FORMULA_DIR = LOCAL_FORMULAS
REPLAY_PATH = HERE / "physics_sparse_replay.py"
if not REPLAY_PATH.is_file():
    REPLAY_PATH = HERE / "replay.py"
TASKS = ("I_dark", "I_photo", "Q_terminal")
MODEL_NAME = "ge_si_photodetector_terminal_charge.va"
MODEL_NAME_IN_BUNDLE = MODEL_NAME
MODULE_NAME = "ge_si_photodetector_terminal_charge"
V_MIN, V_MAX = -3.0, 0.0
BIAS_VALUES = np.array([-3.0, -2.0, -1.0, -0.5, 0.0])
TEMPERATURE_VALUES = np.array([290.0, 300.0, 310.0])
FREQUENCIES = np.logspace(6.0, 12.0, 121)
VAC_MAG = 1.0e-3
CONDITIONS = {
    "nominal": (1.0e-9, 2500.0, 500.0, 40.0),
    "corner": (1.15e-9, 2650.0, 530.0, 43.0),
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formula_paths(formula_dir=FORMULA_DIR):
    return {task: formula_dir / task / "seed_42/selected_formula.json" for task in TASKS}


def load_formulas(formula_dir=FORMULA_DIR):
    formulas = {}
    for task, path in formula_paths(formula_dir).items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {"I_dark": 10, "I_photo": 8, "Q_terminal": 20}[task]
        if (payload.get("schema"), payload.get("task"), payload.get("seed"),
                payload.get("coefficient_count")) != (
                    "physics_balanced_sparse_v1", task, 42, expected
                ) or payload.get("uses_kan_at_inference", False):
            raise ValueError(f"Unexpected frozen formula: {path}")
        formulas[task] = payload
    return formulas


def load_replay(path=REPLAY_PATH):
    spec = importlib.util.spec_from_file_location("physics_sparse_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load independent replay: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def number(value):
    return f"{float(value):.17g}"


def cube(expression):
    return f"(({expression})*({expression})*({expression}))"


def atom_expression(atom, scalers):
    name = atom["input"]
    scaler = scalers[name]
    z = f"(({name}-({number(scaler['mean'])}))/{number(scaler['scale'])})"
    if atom["kind"] == "power":
        degree = int(atom["value"])
        if degree < 0 or degree > 3:
            raise ValueError(f"Unsupported power: {degree}")
        return "1.0" if degree == 0 else "(" + "*".join([z] * degree) + ")"
    if atom["kind"] == "hinge3":
        return cube(f"sparse_pos({z}-({number(atom['value'])}))")
    raise ValueError(f"Unsupported atom: {atom['kind']}")


def feature_expression(feature, scalers, axis, reference):
    atoms = feature["atoms"]
    product = "*".join(atom_expression(atom, scalers) for atom in atoms)
    if any(atom["input"] == axis for atom in atoms):
        # Substitute only the swept voltage. Condition factors remain live.
        reference_product = "*".join(
            _reference_atom(atom, scalers, axis, reference)
            for atom in atoms
        )
        return f"(({product})-({reference_product}))/{number(feature['scale'])}"
    return f"(({product})-({number(feature['center'])}))/{number(feature['scale'])}"


def _reference_atom(atom, scalers, axis, reference):
    if atom["input"] != axis:
        return atom_expression(atom, scalers)
    z = (reference - float(scalers[axis]["mean"])) / float(scalers[axis]["scale"])
    if atom["kind"] == "power":
        return number(z ** int(atom["value"]))
    if atom["kind"] == "hinge3":
        return number(max(z - float(atom["value"]), 0.0) ** 3)
    raise ValueError(f"Unsupported atom: {atom['kind']}")


def current_function(payload, name):
    formula = payload["current_formula"]
    inputs = formula["input_order"]
    axis = inputs[0]
    if len(formula["features"]) + 1 != len(formula["coefficients"]):
        raise ValueError(f"Coefficient/feature mismatch in {name}")
    terms = [number(formula["coefficients"][0])]
    for coefficient, feature in zip(formula["coefficients"][1:], formula["features"]):
        terms.append(f"({number(coefficient)})*({feature_expression(feature, formula['input_scalers'], axis, formula['reference_voltage'])})")
    if len(terms) != payload["coefficient_count"]:
        raise ValueError(f"Coefficient/feature mismatch in {name}")
    body = "\n        + ".join(terms)
    args = ", ".join(inputs)
    return (f"  analog function real {name};\n"
            f"    input {args};\n    real {args};\n"
            f"    begin\n      {name} = pow(10.0,\n        {body});\n"
            "    end\n  endfunction\n")


def charge_function(payload):
    model = payload["charge_basis"]
    names = model["parameter_names"]
    powers = model["parameter_powers"]
    basis_count = len(powers)
    terms = []
    for term in payload["terms"]:
        voltage_index, parameter_index = divmod(int(term["index"]), basis_count)
        if voltage_index < 3:
            voltage = "*".join(["bias_v"] * (voltage_index + 1))
        else:
            knot = float(model["voltage_knots_V"][voltage_index - 3])
            voltage = f"({cube(f'sparse_pos(bias_v-({number(knot)}))')}-{number(max(-knot, 0.0) ** 3)})"
        factors = [voltage]
        for index, power in enumerate(powers[parameter_index]):
            if power:
                if power != 1:
                    raise ValueError("Sparse charge export expects linear condition effects")
                factors.append(f"(({names[index]}-({number(model['parameter_mean'][index])}))/"
                               f"{number(model['parameter_scale'][index])})")
        terms.append(f"({number(term['coefficient'])})*({'*'.join(factors)})")
    if len(terms) != payload["coefficient_count"]:
        raise ValueError("Charge term count mismatch")
    args = ", ".join(["bias_v", *names])
    return ("  analog function real sparse_charge;\n"
            f"    input {args};\n    real {args};\n"
            "    begin\n      sparse_charge = " + number(model["q_scale_C"])
            + " * (\n        " + "\n        + ".join(terms) + ");\n"
            "    end\n  endfunction\n")


def render_model(formulas):
    dark = formulas["I_dark"]
    photo = formulas["I_photo"]
    charge = formulas["Q_terminal"]
    args = ("vd_safe, trap_assisted_recomb_A, ge_sio2_recomb_velocity, "
            "ge_si_recomb_velocity, active_layer_length, sim_temp")
    return ("// Frozen physics_balanced_sparse_v1 seed-42 10/8/20 candidate.\n"
            "// Only the characterized illumination endpoints and area=1 are validated.\n"
            '`include "disciplines.vams"\n`include "constants.vams"\n\n'
            f"module {MODULE_NAME}(anode, cathode);\n"
            "  inout anode, cathode;\n  electrical anode, cathode;\n"
            "  parameter real active_layer_length=40.0 from [34.83:44.83];\n"
            "  parameter real trap_assisted_recomb_A=1.0e-9 from [7.52e-10:1.26e-9];\n"
            "  parameter real ge_sio2_recomb_velocity=2500.0 from [2276.0:2767.0];\n"
            "  parameter real ge_si_recomb_velocity=500.0 from [454.0:547.0];\n"
            "  parameter real sim_temp=300.0 from [285.18:311.33];\n"
            "  parameter real area=1.0 from (0:1.0e6];\n"
            "  parameter real light_power=0.0 from [0:1];\n"
            "  real vd, vd_safe, q_terminal;\n\n"
            "  analog function real sparse_pos;\n    input x;\n    real x;\n"
            "    begin\n      if (x > 0.0) sparse_pos = x;\n"
            "      else sparse_pos = 0.0;\n    end\n  endfunction\n\n"
            + current_function(dark, "sparse_dark") + "\n"
            + current_function(photo, "sparse_photo") + "\n"
            + charge_function(charge) + "\n"
            "  analog begin\n    vd = V(anode, cathode);\n"
            "    if (vd < -3.0) vd_safe = -3.0;\n"
            "    else if (vd > 0.0) vd_safe = 0.0;\n"
            "    else vd_safe = vd;\n"
            f"    q_terminal = area * sparse_charge({args});\n"
            f"    I(anode, cathode) <+ area * (sparse_dark({args})"
            f" + light_power * sparse_photo({args}));\n"
            "    I(anode, cathode) <+ ddt(q_terminal);\n"
            "  end\nendmodule\n")


def condition_frame(voltage, temperature=300.0, condition="nominal"):
    voltage = np.atleast_1d(np.asarray(voltage, dtype=float))
    temperature = np.broadcast_to(np.asarray(temperature, dtype=float), voltage.shape)
    trap, sio2, si, length = CONDITIONS[condition]
    return pd.DataFrame({
        "bias_v": voltage, "dark_voltage": voltage, "light_voltage": voltage,
        "trap_assisted_recomb_A": trap, "ge_sio2_recomb_velocity": sio2,
        "ge_si_recomb_velocity": si, "active_layer_length": length,
        "simulation_temperature": temperature,
    })


class SymbolicCurrentReference:
    def __init__(self, formula_dir=FORMULA_DIR, replay_path=REPLAY_PATH):
        formulas = load_formulas(formula_dir)
        self.dark, self.photo = formulas["I_dark"], formulas["I_photo"]
        self.replay = load_replay(replay_path)

    def evaluate(self, voltage, temperature=300.0, light=1.0, condition="nominal"):
        frame = condition_frame(np.clip(voltage, V_MIN, V_MAX), temperature, condition)
        dark, _ = self.replay.replay_current(self.dark, frame)
        photo, _ = self.replay.replay_current(self.photo, frame)
        return np.power(10.0, dark) + light * np.power(10.0, photo)

    def derivative(self, voltage, temperature=300.0, light=1.0, condition="nominal"):
        voltage = np.atleast_1d(np.asarray(voltage, dtype=float))
        frame = condition_frame(voltage, temperature, condition)
        dark, dark_d = self.replay.replay_current(self.dark, frame)
        photo, photo_d = self.replay.replay_current(self.photo, frame)
        result = math.log(10.0) * (np.power(10.0, dark) * dark_d
                                   + light * np.power(10.0, photo) * photo_d)
        # Hard voltage holding has a zero outside tangent. At equality, the
        # in-envelope branch is active, with the sparse formulas C2 in voltage.
        return np.where((voltage < V_MIN) | (voltage > V_MAX), 0.0, result)


class ChargeReference:
    def __init__(self, formula_dir=FORMULA_DIR, replay_path=REPLAY_PATH):
        self.formula = load_formulas(formula_dir)["Q_terminal"]
        self.replay = load_replay(replay_path)

    def evaluate(self, frame):
        bounded = frame.copy()
        raw_voltage = np.asarray(frame["bias_v"], dtype=float)
        bounded["bias_v"] = np.clip(raw_voltage, V_MIN, V_MAX)
        charge, derivative = self.replay.replay_charge(self.formula, bounded)
        derivative = np.where((raw_voltage < V_MIN) | (raw_voltage > V_MAX),
                              0.0, derivative)
        return charge, derivative


def terminal_reference():
    return ChargeReference()


def parse_psf(path):
    columns = {}
    in_values = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "VALUE":
            in_values = True
            continue
        if not in_values:
            continue
        if line == "END":
            break
        match = re.match(r'^"([^"]+)"\s+(.+)$', line)
        if match:
            tokens = match.group(2).strip("() ").replace(",", " ").split()
            try:
                parts = [float(token) for token in tokens[:2]]
            except ValueError:
                continue
            columns.setdefault(match.group(1), []).append(
                complex(parts[0], parts[1] if len(parts) > 1 else 0.0))
    if not columns or len({len(values) for values in columns.values()}) != 1:
        raise ValueError(f"Incomplete PSF-ASCII columns: {path}")
    return {key: np.asarray(value, dtype=complex) for key, value in columns.items()}


def within(actual, expected, relative, absolute):
    return bool(np.all(np.isfinite(actual)) and np.all(np.abs(actual - expected)
                <= absolute + relative * np.abs(expected)))


def validate_dc(files, current):
    expected_names = {f"temperature_sweep-{index:03d}_dc_sparse.dc"
                      for index in range(len(TEMPERATURE_VALUES))}
    failures = []
    if {path.name for path in files} != expected_names:
        failures.append("dc_file_set")
    for path in files:
        match = re.fullmatch(r"temperature_sweep-(\d{3})_dc_sparse.dc", path.name)
        if not match or int(match.group(1)) >= len(TEMPERATURE_VALUES):
            failures.append(f"dc_filename:{path.name}")
            continue
        temperature = TEMPERATURE_VALUES[int(match.group(1))]
        data = parse_psf(path)
        grid = np.linspace(V_MIN, V_MAX, 61)
        if "Vb_dc" not in data or len(data["Vb_dc"]) != len(grid) or not np.allclose(
                data["Vb_dc"].real, grid, rtol=0, atol=1e-9):
            failures.append(f"dc_grid:{path.name}")
            continue
        for trace, light, condition in (("Vdark:p", 0, "nominal"),
                                        ("Vlight:p", 1, "nominal"),
                                        ("Vcorner:p", 1, "corner")):
            expected = current.evaluate(grid, temperature, light, condition)
            if trace not in data or not within(-data[trace].real, expected,
                                              TOLERANCES["dc_relative"], TOLERANCES["dc_absolute_A"]):
                failures.append(f"dc_mismatch:{path.name}:{trace}")
    return {"passed": not failures, "failures": failures, "points_per_trace": 183 if not failures else None}


def validate_ac(files, current, charge):
    expected_names = {f"bias_sweep-{b:03d}_temperature_sweep-{t:03d}_ac_sparse.ac"
                      for b in range(len(BIAS_VALUES)) for t in range(len(TEMPERATURE_VALUES))}
    failures = []
    if {path.name for path in files} != expected_names:
        failures.append("ac_file_set")
    for path in files:
        match = re.fullmatch(r"bias_sweep-(\d{3})_temperature_sweep-(\d{3})_ac_sparse.ac", path.name)
        if not match or int(match.group(1)) >= len(BIAS_VALUES) or int(match.group(2)) >= len(TEMPERATURE_VALUES):
            failures.append(f"ac_filename:{path.name}")
            continue
        bias = BIAS_VALUES[int(match.group(1))]
        temperature = TEMPERATURE_VALUES[int(match.group(2))]
        data = parse_psf(path)
        frequency = data.get("freq", data.get("frequency"))
        if frequency is None or len(frequency) != 121 or not np.allclose(
                frequency.real, FREQUENCIES, rtol=1e-5, atol=0):
            failures.append(f"ac_grid:{path.name}")
            continue
        for trace, light, condition in (("Vac_dark:p", 0, "nominal"),
                                        ("Vac_light:p", 1, "nominal"),
                                        ("Vac_corner:p", 1, "corner")):
            if trace not in data:
                failures.append(f"ac_trace:{path.name}:{trace}")
                continue
            didv = current.derivative([bias], temperature, light, condition)[0]
            _, dqdv = charge.evaluate(condition_frame([bias], temperature, condition))
            actual_g = -data[trace].real / VAC_MAG
            actual_c = -data[trace].imag / (VAC_MAG * 2.0 * math.pi * frequency.real)
            if not within(actual_g, didv, TOLERANCES["ac_conductance_relative"],
                          TOLERANCES["ac_conductance_absolute_S"]):
                failures.append(f"ac_conductance:{path.name}:{trace}")
            if not within(actual_c, dqdv[0], TOLERANCES["ac_dqdv_relative"],
                          TOLERANCES["ac_dqdv_absolute_F"]):
                failures.append(f"ac_charge:{path.name}:{trace}")
    return {"passed": not failures, "failures": failures, "points_per_trace": 1815 if not failures else None}


def validate_transient(path, current, charge):
    data = parse_psf(path)
    time = data["time"].real
    if len(time) < 100 or not np.all(np.diff(time) > 0) or not np.isclose(time[-1], 10e-9, rtol=0, atol=1e-13):
        raise ValueError(f"Incomplete transient timeline: {path}")
    results = {}
    for name, light, condition in (("fast", 1, "nominal"), ("slow", 0, "nominal"),
                                   ("sine", 1, "corner")):
        voltage = data[f"a_{name}"].real
        observed = -data[f"V{name}:p"].real - current.evaluate(voltage, 300.0, light, condition)
        _, dqdv = charge.evaluate(condition_frame(voltage, 300.0, condition))
        dvdt = np.gradient(voltage, time, edge_order=2)
        expected = dqdv * dvdt
        active = np.abs(dvdt) > 0.01 * max(float(np.max(np.abs(dvdt))), 1.0)
        indices = np.flatnonzero(active)
        if len(indices) < 10:
            raise ValueError(f"No active transient: {name}")
        starts = np.r_[indices[0], indices[1:][np.diff(indices) > 1]]
        ends = np.r_[indices[:-1][np.diff(indices) > 1], indices[-1]]
        interior = np.zeros(len(time), dtype=bool)
        integrated = []
        for start, end in zip(starts, ends):
            interior[int(start) + 1:int(end)] = True
            lo, hi = max(int(start) - 1, 0), min(int(end) + 1, len(time) - 1)
            q, _ = charge.evaluate(condition_frame([voltage[lo], voltage[hi]], 300.0, condition))
            actual = float(np.trapz(observed[lo:hi + 1], time[lo:hi + 1]))
            delta = float(q[1] - q[0])
            integrated.append(abs(actual - delta) <= TOLERANCES["transient_charge_absolute_C"]
                              + TOLERANCES["transient_charge_relative"] * abs(delta))
        if np.count_nonzero(interior) < 10:
            raise ValueError(f"Too few smooth transient points: {name}")
        scale = max(float(np.sqrt(np.mean(expected[interior] ** 2))), 1e-30)
        error = float(np.sqrt(np.mean((observed[interior] - expected[interior]) ** 2)) / scale)
        results[name] = {"passed": bool(np.isfinite(observed).all() and integrated and all(integrated)
                                        and error <= TOLERANCES["transient_normalized_rmse"]),
                         "normalized_rmse": error, "integral_checks": len(integrated),
                         "time": time, "observed": observed, "active": interior}
    return results


def verify(results, output, formula_dir=FORMULA_DIR,
           replay_path=REPLAY_PATH, model=None):
    model = model or HERE / MODEL_NAME
    manifest_path = results.parent / "preflight_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = results.parent / "bundle"
    expected_hashes = manifest["sha256"]
    checks = {"preflight_status": manifest.get("status") == "ready_to_run",
              "source_binding": bool(model.is_file() and sha256(model) == expected_hashes["model"]
                                      and sha256(bundle / MODEL_NAME) == expected_hashes["model"])}
    for task, path in formula_paths(formula_dir).items():
        checks[f"formula_{task}"] = sha256(path) == expected_hashes[task]
    checks["replay"] = sha256(replay_path) == expected_hashes["replay"]
    for name in ("testbench_dc_sparse.scs", "testbench_ac_sparse.scs",
                 "testbench_transient_sparse.scs", "testbench_transient_sparse_fine.scs"):
        checks[f"deck_{name}"] = (name in expected_hashes and
                                  sha256(model.parent / name) == expected_hashes[name] and
                                  sha256(bundle / name) == expected_hashes[name])
    required_logs = ("dc.log", "ac.log", "transient.log", "transient_fine.log")
    checks["logs"] = all((results / name).is_file() and
                         re.search(r"spectre completes with 0 errors",
                                   (results / name).read_text(encoding="utf-8", errors="replace"), re.I)
                         and not re.search(r"fatal|failed\s+to\s+converge|convergence\s+failure",
                                           (results / name).read_text(encoding="utf-8", errors="replace"), re.I)
                         for name in required_logs)
    checks["version"] = (results / "spectre_version.txt").is_file()
    current = SymbolicCurrentReference(formula_dir, replay_path)
    charge = ChargeReference(formula_dir, replay_path)
    dc = validate_dc(sorted((results / "dc.raw").rglob("*.dc")), current)
    ac = validate_ac(sorted((results / "ac.raw").rglob("*.ac")), current, charge)
    checks["dc"] = dc["passed"]
    checks["ac"] = ac["passed"]
    try:
        coarse_files = list((results / "transient.raw").rglob("*.tran"))
        fine_files = list((results / "transient_fine.raw").rglob("*.tran"))
        if len(coarse_files) != 1 or len(fine_files) != 1:
            raise ValueError("Expected one coarse and one fine transient")
        coarse = validate_transient(coarse_files[0], current, charge)
        fine = validate_transient(fine_files[0], current, charge)
        step = {}
        for name in coarse:
            t = coarse[name]["time"]
            interpolated = np.interp(t, fine[name]["time"], fine[name]["observed"])
            mask = coarse[name]["active"]
            scale = max(float(np.sqrt(np.mean(interpolated[mask] ** 2))), 1e-30)
            step[name] = float(np.sqrt(np.mean((coarse[name]["observed"][mask]
                                             - interpolated[mask]) ** 2)) / scale)
        checks["transient"] = all(coarse[name]["passed"] and fine[name]["passed"]
                                  and step[name] <= TOLERANCES["transient_step_convergence"]
                                  for name in coarse)
        transient = {"coarse_nrmse": {name: value["normalized_rmse"] for name, value in coarse.items()},
                     "fine_nrmse": {name: value["normalized_rmse"] for name, value in fine.items()},
                     "step_nrmse": step}
    except (KeyError, ValueError, OSError) as exc:
        checks["transient"] = False
        transient = {"error": str(exc)}
    report = {"status": "passed" if all(checks.values()) else "failed",
              "candidate": "physics_balanced_sparse_v1_seed42_10_8_20",
              "simulator_execution_status": "passed" if all(checks.values()) else "failed",
              "checks": checks, "dc": dc, "ac": ac, "transient": transient,
              "model_sha256": sha256(model), "tolerances": TOLERANCES}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def preflight(output, formula_dir=FORMULA_DIR,
              replay_path=REPLAY_PATH, model=None):
    model = model or HERE / MODEL_NAME
    formulas = load_formulas(formula_dir)
    expected = render_model(formulas)
    if not model.is_file() or model.read_text(encoding="utf-8") != expected:
        raise ValueError("Verilog-A source differs from the frozen 10/8/20 export")
    replay = load_replay(replay_path)
    for task in TASKS:
        payload = formulas[task]
        table = pd.read_csv(formula_dir / task / "seed_42/test_predictions.csv") if (formula_dir / task / "seed_42/test_predictions.csv").is_file() else None
        if table is not None:
            prediction, derivative = replay.replay(payload, table)
            if not np.allclose(prediction, table["selected"], rtol=0, atol=2e-14):
                raise ValueError(f"Frozen {task} replay mismatch")
            if task != "Q_terminal" and not np.allclose(derivative, table["selected_derivative"], rtol=0, atol=2e-9):
                raise ValueError(f"Frozen {task} derivative mismatch")
    current = SymbolicCurrentReference(formula_dir, replay_path)
    charge = ChargeReference(formula_dir, replay_path)
    grid = np.linspace(-3.0, 0.0, 1001)
    for condition in CONDITIONS:
        for light in (0, 1):
            values = current.evaluate(grid, 300.0, light, condition)
            derivatives = current.derivative(grid, 300.0, light, condition)
            q, dqdv = charge.evaluate(condition_frame(grid, 300.0, condition))
            if not all(np.isfinite(a).all() for a in (values, derivatives, q, dqdv)):
                raise ValueError("Nonfinite dense local reference")
            if np.any(values <= 0) or np.any(dqdv <= 0) or abs(q[-1]) > 1e-28:
                raise ValueError("Sparse physical-contract check failed")
    hashes = {task: sha256(path) for task, path in formula_paths(formula_dir).items()}
    hashes.update(model=sha256(model), replay=sha256(replay_path))
    for name in ("testbench_dc_sparse.scs", "testbench_ac_sparse.scs",
                 "testbench_transient_sparse.scs", "testbench_transient_sparse_fine.scs"):
        path = model.parent / name
        if path.is_file():
            hashes[name] = sha256(path)
    report = {"status": "ready_to_run", "simulator_execution_status": "pending",
              "candidate": "physics_balanced_sparse_v1_seed42_10_8_20",
              "sha256": hashes, "tolerances": TOLERANCES,
              "expected_points_per_trace": {"dc": 183, "ac": 1815},
              "cases": ["nominal_dark", "nominal_light", "corner_light"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for command in ("preflight", "verify"):
        entry = commands.add_parser(command)
        entry.add_argument("--output", type=Path, required=True)
        entry.add_argument("--formula-dir", type=Path, default=FORMULA_DIR)
        entry.add_argument("--replay", type=Path, default=REPLAY_PATH)
        entry.add_argument("--model", type=Path, default=HERE / MODEL_NAME)
        if command == "verify":
            entry.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if args.command is None:
        parser.error("preflight or verify is required")
    report = (preflight(args.output, args.formula_dir, args.replay, args.model)
              if args.command == "preflight" else
              verify(args.results, args.output, args.formula_dir, args.replay, args.model))
    print(json.dumps({"status": report["status"], "candidate": report["candidate"]}, indent=2))
    if report["status"] not in ("passed", "ready_to_run"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
