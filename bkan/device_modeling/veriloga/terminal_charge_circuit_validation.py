#!/usr/bin/env python3
"""Prepare and verify bounded circuit-level Spectre tests for terminal-Q.

The reference path is intentionally independent of Verilog-A execution.  It
evaluates the frozen current formulas and terminal-charge JSON, solves the
external circuit equations in Python, and writes immutable CSV references.
The ``verify`` command needs only the returned PSF-ASCII data and the frozen
bundle; it does not regenerate expected values after Spectre has run.

Scope: electrical bias/load, finite-bandwidth behavioral TIA AC/transient, and
1/10/100-instance scaling.  The injected electrical current is not an optical
port and these tests do not establish measurement agreement, PDK integration,
or device qualification.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MODEL = HERE / "ge_si_photodetector_terminal_charge.va"
RUNNER = HERE / "run_spectre_terminal_charge_circuits.sh"
MODEL_SHA256 = "2dbe37208a520f2abbf6244b5e88c520db7793c1d4ebe51044fe2886651a062b"

RF_OHM = 2.0e3
CF_F = 5.0e-14
GM_S = 1.0e-1
ROUT_OHM = 1.0e6
CDOM_F = 1.0e-12
CLOAD_F = 1.0e-13
VBIAS_V = 1.0
IAC_A = 1.0e-6
VAC_V = 1.0e-3
RBIAS_OHM = 1.0e3
FREQ = np.logspace(6.0, 11.0, 101)
MULTI_COUNTS = (1, 10, 100)

TOLERANCES = {
    "bias_node_absolute_V": 2.0e-4,
    "bias_current_relative": 5.0e-3,
    "bias_current_absolute_A": 2.0e-8,
    "tia_ac_complex_relative": 3.0e-2,
    "tia_ac_complex_absolute_V": 2.0e-6,
    "tia_transient_input_nrmse": 5.0e-2,
    "tia_transient_output_nrmse": 5.0e-2,
    "tia_transient_peak_relative": 5.0e-2,
    "multi_complex_relative": 3.0e-2,
    "multi_complex_absolute_A": 2.0e-9,
}

LINE_RE = re.compile(r'^"([^"]+)"\s+(.+)$')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_models():
    # Deferred import keeps remote verification independent of repository data.
    import sys

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import verify_terminal_charge_spectre as device

    return device, device.SymbolicCurrentReference(), device.terminal_reference()


def _condition(device, voltage: Union[np.ndarray, List[float]]):
    return device.condition_frame(voltage, 300.0)


def _device_values(device, current, charge, voltage):
    voltage = np.atleast_1d(np.asarray(voltage, dtype=float))
    i_dc = current.evaluate(voltage, 300.0)
    conductance = current.derivative(voltage, 300.0)
    q_value, capacitance = charge.evaluate(_condition(device, voltage))
    return i_dc, conductance, np.asarray(q_value), np.asarray(capacitance)


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty reference: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_text_lf(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _read_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


def _bias_reference(device, current, charge) -> List[dict]:
    from scipy.optimize import brentq

    rows = []
    # This source range keeps the solved device terminal voltage inside the
    # characterized [-3, 0] V interval despite the 1-kohm load drop.
    for source_v in np.linspace(-2.4, 0.4, 57):
        def residual(node_v: float) -> float:
            i_dc = float(current.evaluate([node_v], 300.0)[0])
            return i_dc + (node_v - source_v) / RBIAS_OHM

        node_v = brentq(residual, -3.0, 0.0, xtol=1.0e-13)
        i_dc = float(current.evaluate([node_v], 300.0)[0])
        rows.append(
            {
                "source_V": source_v,
                "pd_anode_V": node_v,
                "pd_voltage_V": node_v,
                "source_branch_current_A": -i_dc,
                "device_current_A": i_dc,
            }
        )
    return rows


def _tia_dc(device, current, charge) -> Tuple[float, float]:
    from scipy.optimize import root

    def residual(state):
        vin, vout = state
        i_dc = float(current.evaluate([vin - VBIAS_V], 300.0)[0])
        return [
            i_dc + (vin - vout) / RF_OHM,
            GM_S * vin + vout / ROUT_OHM + (vout - vin) / RF_OHM,
        ]

    solved = root(residual, [-5.0e-3, 1.0])
    if not solved.success or np.max(np.abs(residual(solved.x))) > 1.0e-9:
        raise RuntimeError(f"TIA DC solve failed: {solved.message}")
    return float(solved.x[0]), float(solved.x[1])


def _tia_ac_reference(device, current, charge) -> Tuple[List[dict], dict]:
    vin_dc, vout_dc = _tia_dc(device, current, charge)
    _, conductance, _, capacitance = _device_values(
        device, current, charge, [vin_dc - VBIAS_V]
    )
    gpd = float(conductance[0])
    cpd = float(capacitance[0])
    rows = []
    for frequency in FREQ:
        omega = 2.0 * math.pi * frequency
        ypd = gpd + 1j * omega * cpd
        yf = 1.0 / RF_OHM + 1j * omega * CF_F
        yout = 1.0 / ROUT_OHM + 1j * omega * (CDOM_F + CLOAD_F)
        matrix = np.asarray(
            [[ypd + yf, -yf], [GM_S - yf, yf + yout]], dtype=complex
        )
        vin, vout = np.linalg.solve(matrix, np.asarray([-IAC_A, 0.0], dtype=complex))
        rows.append(
            {
                "frequency_Hz": frequency,
                "input_real_V": vin.real,
                "input_imag_V": vin.imag,
                "output_real_V": vout.real,
                "output_imag_V": vout.imag,
                "transimpedance_magnitude_ohm": abs(vout / IAC_A),
                "transimpedance_phase_deg": math.degrees(np.angle(vout / IAC_A)),
            }
        )
    operating = {
        "input_dc_V": vin_dc,
        "output_dc_V": vout_dc,
        "pd_voltage_dc_V": vin_dc - VBIAS_V,
        "pd_conductance_S": gpd,
        "pd_dQdV_F": cpd,
    }
    return rows, operating


def _pulse_current(time: float) -> float:
    phase = time % 5.0e-9
    if phase < 1.0e-9:
        return 0.0
    if phase < 1.1e-9:
        return 2.0e-5 * (phase - 1.0e-9) / 1.0e-10
    if phase < 3.1e-9:
        return 2.0e-5
    if phase < 3.2e-9:
        return 2.0e-5 * (3.2e-9 - phase) / 1.0e-10
    return 0.0


def _tia_transient_reference(device, current, charge) -> List[dict]:
    from scipy.integrate import solve_ivp

    initial = np.asarray(_tia_dc(device, current, charge), dtype=float)
    c_out = CDOM_F + CLOAD_F

    def rhs(time, state):
        vin, vout = state
        i_dc, _, _, cpd = _device_values(
            device, current, charge, [vin - VBIAS_V]
        )
        mass = np.asarray([[cpd[0] + CF_F, -CF_F], [-CF_F, c_out + CF_F]])
        static = np.asarray(
            [
                i_dc[0] + (vin - vout) / RF_OHM + _pulse_current(time),
                GM_S * vin + vout / ROUT_OHM + (vout - vin) / RF_OHM,
            ]
        )
        return np.linalg.solve(mass, -static)

    sample_time = np.linspace(0.0, 10.0e-9, 5001)
    solved = solve_ivp(
        rhs,
        (sample_time[0], sample_time[-1]),
        initial,
        method="BDF",
        t_eval=sample_time,
        rtol=2.0e-8,
        atol=[1.0e-10, 1.0e-9],
        max_step=2.0e-12,
    )
    if not solved.success or not np.isfinite(solved.y).all():
        raise RuntimeError(f"TIA transient solve failed: {solved.message}")
    return [
        {
            "time_s": time,
            "injected_current_A": _pulse_current(time),
            "input_V": vin,
            "output_V": vout,
        }
        for time, vin, vout in zip(sample_time, solved.y[0], solved.y[1])
    ]


def _multi_reference(device, current, charge) -> List[dict]:
    _, conductance, _, capacitance = _device_values(
        device, current, charge, [-1.0]
    )
    rows = []
    for count in MULTI_COUNTS:
        for frequency in FREQ:
            admittance = conductance[0] + 1j * 2.0 * math.pi * frequency * capacitance[0]
            branch = -count * VAC_V * admittance
            rows.append(
                {
                    "instances": count,
                    "frequency_Hz": frequency,
                    "source_current_real_A": branch.real,
                    "source_current_imag_A": branch.imag,
                }
            )
    return rows


def _instance_deck(count: int) -> str:
    instances = "\n".join(
        f"Xpd{index:03d} (pd 0) ge_si_photodetector_terminal_charge "
        "active_layer_length=40.0 sim_temp=300.0 trap_assisted_recomb_A=1.0e-9 "
        "ge_sio2_recomb_velocity=2500.0 ge_si_recomb_velocity=500.0 "
        "area=1.0 light_power=1.0"
        for index in range(1, count + 1)
    )
    return f"""simulator lang=spectre
ahdl_include \"ge_si_photodetector_terminal_charge.va\"

Vstim (pd 0) vsource dc=-1.0 mag=1m type=sine
{instances}

save Vstim:p pd
multi_{count:03d} ac dec=20 start=1M stop=100G
"""


DECKS = {
    "testbench_bias_load_terminal_charge_ic618.scs": """simulator lang=spectre
ahdl_include \"ge_si_photodetector_terminal_charge.va\"

parameters Vsweep=-2.4
Vdrive (supply 0) vsource dc=Vsweep
Rbias (supply pd) resistor r=1k
Xpd (pd 0) ge_si_photodetector_terminal_charge active_layer_length=40.0 sim_temp=300.0 trap_assisted_recomb_A=1.0e-9 ge_sio2_recomb_velocity=2500.0 ge_si_recomb_velocity=500.0 area=1.0 light_power=1.0

save Vdrive:p supply pd
bias_load dc param=Vsweep start=-2.4 stop=0.4 step=0.05
""",
    "testbench_tia_ac_terminal_charge_ic618.scs": """simulator lang=spectre
ahdl_include \"ge_si_photodetector_terminal_charge.va\"

Vbias (bias 0) vsource dc=1.0
Iinj (in 0) isource dc=0 mag=1u type=sine
Xpd (in bias) ge_si_photodetector_terminal_charge active_layer_length=40.0 sim_temp=300.0 trap_assisted_recomb_A=1.0e-9 ge_sio2_recomb_velocity=2500.0 ge_si_recomb_velocity=500.0 area=1.0 light_power=1.0
Gm (out 0 in 0) vccs gm=0.1
Rout (out 0) resistor r=1Meg
Cdom (out 0) capacitor c=1p
Rf (out in) resistor r=2k
Cf (out in) capacitor c=50f
Cload (out 0) capacitor c=100f

save in out Vbias:p
tia_ac ac dec=20 start=1M stop=100G
""",
    "testbench_tia_transient_terminal_charge_ic618.scs": """simulator lang=spectre
ahdl_include \"ge_si_photodetector_terminal_charge.va\"

Vbias (bias 0) vsource dc=1.0
Iinj (in 0) isource type=pulse val0=0 val1=20u delay=1n rise=100p fall=100p width=2n period=5n
Xpd (in bias) ge_si_photodetector_terminal_charge active_layer_length=40.0 sim_temp=300.0 trap_assisted_recomb_A=1.0e-9 ge_sio2_recomb_velocity=2500.0 ge_si_recomb_velocity=500.0 area=1.0 light_power=1.0
Gm (out 0 in 0) vccs gm=0.1
Rout (out 0) resistor r=1Meg
Cdom (out 0) capacitor c=1p
Rf (out in) resistor r=2k
Cf (out in) capacitor c=50f
Cload (out 0) capacitor c=100f

save in out Vbias:p
tia_transient tran stop=10n maxstep=2p
""",
}


def prepare(bundle: Path, expected_model_sha256: str = MODEL_SHA256) -> int:
    expected_model_sha256 = expected_model_sha256.lower()
    if sha256(MODEL) != expected_model_sha256:
        raise RuntimeError(
            "Canonical Verilog-A bytes do not match the explicitly requested bundle hash"
        )
    if bundle.exists() and any(bundle.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty bundle: {bundle}")
    bundle.mkdir(parents=True, exist_ok=True)
    reference_dir = bundle / "reference"
    device, current, charge = _reference_models()
    ac_rows, operating = _tia_ac_reference(device, current, charge)
    references = {
        "bias_load_reference.csv": _bias_reference(device, current, charge),
        "tia_ac_reference.csv": ac_rows,
        "tia_transient_reference.csv": _tia_transient_reference(device, current, charge),
        "multi_instance_reference.csv": _multi_reference(device, current, charge),
    }
    for name, rows in references.items():
        _write_csv(reference_dir / name, rows)
    (reference_dir / "tia_operating_point.json").write_text(
        json.dumps(operating, indent=2), encoding="utf-8"
    )
    shutil.copy2(MODEL, bundle / MODEL.name)
    shutil.copy2(Path(__file__), bundle / Path(__file__).name)
    shutil.copy2(RUNNER, bundle / RUNNER.name)
    for name, content in DECKS.items():
        _write_text_lf(bundle / name, content)
    for count in MULTI_COUNTS:
        name = f"testbench_multi_instance_{count:03d}_terminal_charge_ic618.scs"
        _write_text_lf(bundle / name, _instance_deck(count))
    rows = []
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(bundle).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "ready_for_remote_spectre",
        "scope": "bounded_electrical_circuit_level_validation_no_measurements",
        "model_sha256": expected_model_sha256,
        "reference_generation": "independent_frozen_formula_and_terminal_Q_JSON",
        "optical_port_validation": False,
        "device_qualification": False,
        "tolerances": TOLERANCES,
        "files": rows,
    }
    (bundle / "preflight_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"bundle": str(bundle.resolve()), "files": len(rows) + 1}, indent=2))
    return 0


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
        for raw in handle:
            line = raw.strip()
            if line == "VALUE":
                in_values = True
                continue
            if not in_values:
                continue
            if line == "END":
                break
            match = LINE_RE.match(line)
            if match:
                value = parse_number(match.group(2))
                if value is not None:
                    columns.setdefault(match.group(1), []).append(value)
    if not columns or len({len(value) for value in columns.values()}) != 1:
        raise RuntimeError(f"Invalid or misaligned PSF-ASCII data: {path}")
    return {key: np.asarray(value, dtype=complex) for key, value in columns.items()}


def _only(root: Path, suffix: str) -> Path:
    files = sorted(root.rglob(f"*.{suffix}"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one .{suffix} below {root}, found {len(files)}")
    return files[0]


def _complex_error(actual, expected, relative, absolute) -> Tuple[bool, float]:
    error = np.abs(actual - expected)
    allowed = absolute + relative * np.abs(expected)
    return bool(np.all(error <= allowed)), float(np.max(error))


def _verify_bias(results: Path, reference: Path) -> dict:
    data = parse_psf_ascii(_only(results / "bias_load.raw", "dc"))
    ref = _read_csv(reference / "bias_load_reference.csv")
    source = data.get("Vsweep")
    if source is None:
        source = data.get("sweep")
    if source is None or "pd" not in data or "Vdrive:p" not in data:
        return {"passed": False, "failures": ["missing_bias_trace"]}
    x = source.real
    expected_node = np.interp(x, ref["source_V"], ref["pd_anode_V"])
    expected_current = np.interp(x, ref["source_V"], ref["source_branch_current_A"])
    node_error = np.abs(data["pd"].real - expected_node)
    current_error = np.abs(data["Vdrive:p"].real - expected_current)
    current_allowed = TOLERANCES["bias_current_absolute_A"] + TOLERANCES["bias_current_relative"] * np.abs(expected_current)
    passed = bool(
        len(x) == len(ref["source_V"])
        and np.all(node_error <= TOLERANCES["bias_node_absolute_V"])
        and np.all(current_error <= current_allowed)
    )
    return {
        "passed": passed,
        "points": len(x),
        "max_node_error_V": float(np.max(node_error)),
        "max_current_error_A": float(np.max(current_error)),
    }


def _verify_tia_ac(results: Path, reference: Path) -> dict:
    data = parse_psf_ascii(_only(results / "tia_ac.raw", "ac"))
    ref = _read_csv(reference / "tia_ac_reference.csv")
    frequency = data.get("freq", data.get("frequency"))
    if frequency is None or "in" not in data or "out" not in data:
        return {"passed": False, "failures": ["missing_tia_ac_trace"]}
    expected_in = ref["input_real_V"] + 1j * ref["input_imag_V"]
    expected_out = ref["output_real_V"] + 1j * ref["output_imag_V"]
    actual_in = data["in"]
    actual_out = data["out"]
    if len(frequency) != len(expected_in):
        return {"passed": False, "failures": [f"tia_ac_point_count:{len(frequency)}"]}
    in_ok, in_error = _complex_error(actual_in, expected_in, TOLERANCES["tia_ac_complex_relative"], TOLERANCES["tia_ac_complex_absolute_V"])
    out_ok, out_error = _complex_error(actual_out, expected_out, TOLERANCES["tia_ac_complex_relative"], TOLERANCES["tia_ac_complex_absolute_V"])
    return {
        "passed": in_ok and out_ok,
        "points": len(frequency),
        "max_input_complex_error_V": in_error,
        "max_output_complex_error_V": out_error,
    }


def _verify_tia_transient(results: Path, reference: Path) -> dict:
    data = parse_psf_ascii(_only(results / "tia_transient.raw", "tran"))
    ref = _read_csv(reference / "tia_transient_reference.csv")
    if "time" not in data or "in" not in data or "out" not in data:
        return {"passed": False, "failures": ["missing_tia_transient_trace"]}
    time = data["time"].real
    expected_in = np.interp(time, ref["time_s"], ref["input_V"])
    expected_out = np.interp(time, ref["time_s"], ref["output_V"])
    actual_in = data["in"].real
    actual_out = data["out"].real
    input_scale = max(float(np.ptp(expected_in)), 1.0e-9)
    output_scale = max(float(np.ptp(expected_out)), 1.0e-6)
    input_nrmse = float(np.sqrt(np.mean((actual_in - expected_in) ** 2)) / input_scale)
    output_nrmse = float(np.sqrt(np.mean((actual_out - expected_out) ** 2)) / output_scale)
    peak_error = float(abs(np.ptp(actual_out) - np.ptp(expected_out)) / output_scale)
    passed = bool(
        np.isfinite(actual_in).all()
        and np.isfinite(actual_out).all()
        and input_nrmse <= TOLERANCES["tia_transient_input_nrmse"]
        and output_nrmse <= TOLERANCES["tia_transient_output_nrmse"]
        and peak_error <= TOLERANCES["tia_transient_peak_relative"]
    )
    return {
        "passed": passed,
        "points": len(time),
        "input_nrmse": input_nrmse,
        "output_nrmse": output_nrmse,
        "output_peak_relative_error": peak_error,
    }


def _verify_multi(results: Path, reference: Path) -> dict:
    ref = _read_csv(reference / "multi_instance_reference.csv")
    details = {}
    passed = True
    for count in MULTI_COUNTS:
        data = parse_psf_ascii(_only(results / f"multi_{count:03d}.raw", "ac"))
        subset = ref["instances"] == count
        expected = ref["source_current_real_A"][subset] + 1j * ref["source_current_imag_A"][subset]
        actual = data.get("Vstim:p")
        if actual is None or len(actual) != len(expected):
            details[str(count)] = {"passed": False, "failure": "missing_or_count"}
            passed = False
            continue
        ok, error = _complex_error(actual, expected, TOLERANCES["multi_complex_relative"], TOLERANCES["multi_complex_absolute_A"])
        runtime_path = results / f"multi_{count:03d}.runtime.txt"
        runtime_text = runtime_path.read_text(encoding="utf-8", errors="replace") if runtime_path.is_file() else ""
        runtime_match = re.search(r"real\s+([0-9.]+)", runtime_text)
        details[str(count)] = {
            "passed": ok and bool(runtime_match),
            "points": len(actual),
            "max_complex_error_A": error,
            "wall_time_s": float(runtime_match.group(1)) if runtime_match else None,
        }
        passed = passed and details[str(count)]["passed"]
    return {"passed": passed, "instances": details}


def _verify_logs(results: Path) -> dict:
    names = ["bias_load", "tia_ac", "tia_transient"] + [f"multi_{n:03d}" for n in MULTI_COUNTS]
    failures = []
    for name in names:
        path = results / f"{name}.log"
        if not path.is_file():
            failures.append(f"missing:{name}.log")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"spectre completes with 0 errors", text, re.I):
            failures.append(f"no_zero_error_completion:{name}")
        if re.search(r"failed\s+to\s+converge|convergence\s+failure|fatal", text, re.I):
            failures.append(f"fatal_or_convergence:{name}")
    return {"passed": not failures, "failures": failures}


def verify(run_root: Path, output: Path) -> int:
    bundle = run_root / "bundle"
    results = run_root / "results"
    reference = bundle / "reference"
    manifest_path = bundle / "preflight_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hash = sha256(bundle / MODEL.name)
    expected_hash = str(manifest.get("model_sha256", "")).lower()
    source_binding = {
        "passed": bool(expected_hash and source_hash == expected_hash),
        "expected_sha256": expected_hash,
        "returned_sha256": source_hash,
        "historical_default_sha256": MODEL_SHA256,
    }
    checks = {
        "source_binding": source_binding,
        "logs_and_convergence": _verify_logs(results),
        "bias_load": _verify_bias(results, reference),
        "tia_ac": _verify_tia_ac(results, reference),
        "tia_transient": _verify_tia_transient(results, reference),
        "multi_instance": _verify_multi(results, reference),
    }
    failed = [name for name, item in checks.items() if not item.get("passed")]
    evidence = []
    for path in sorted(item for item in run_root.rglob("*") if item.is_file() and item != output):
        evidence.append({"path": path.relative_to(run_root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    resolved_root = run_root.resolve()
    try:
        evidence_root = resolved_root.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        evidence_root = str(resolved_root)
    report = {
        "schema_version": 1,
        "status": "passed" if not failed else "failed",
        "scope": "bounded_electrical_circuit_level_validation_no_measurements",
        "evidence_root": evidence_root,
        "claims_supported": [
            "bias_load_self_consistency",
            "behavioral_tia_ac_embedding",
            "behavioral_tia_electrical_transient_embedding",
            "1_10_100_instance_execution_and_scaling",
        ],
        "claims_not_supported": [
            "measured_device_agreement",
            "optical_dynamic_port_response",
            "foundry_PDK_integration",
            "device_qualification",
            "deployment_ready_operation",
        ],
        "tolerances": TOLERANCES,
        "failed_checks": failed,
        "checks": checks,
        "evidence_files": evidence,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "failed_checks": failed, "report": str(output)}, indent=2))
    return 0 if not failed else 1


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--bundle", type=Path, required=True)
    prepare_parser.add_argument(
        "--expected-model-sha256",
        default=MODEL_SHA256,
        help="Explicit source hash to freeze; default preserves the historical bundle.",
    )
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--run-root", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required")
    if args.command == "prepare":
        return prepare(args.bundle.resolve(), args.expected_model_sha256)
    output = args.output or args.run_root / "circuit_acceptance_report.json"
    return verify(args.run_root.resolve(), output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
