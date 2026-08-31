"""Export the audited KAN-free net-photocurrent student to Verilog-A.

The primary TCAD tables contain a dark sweep and an illuminated-total-current
sweep.  The model exported here represents their paired difference
``I_photo,net = I_illum - I_dark``.  It must therefore be added to the dark
branch exactly once.

The JSON formula deliberately stores a structured graph rather than executable
Python.  This exporter reconstructs the public photocurrent feature catalogue,
emits each selected normalized feature once, and independently replays the
same graph before touching a Verilog-A template.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BKAN_ROOT = ROOT / "bkan"
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))

from device_modeling.photodetector.symbolic_gated_kan import (  # noqa: E402
    evaluate_exported_pure_symbolic_formula,
)
from device_modeling.photodetector.task_config import (  # noqa: E402
    TASKS,
    apply_dark_current_floor,
    prepare_task_dataframe,
)


LOG2 = float(np.log(2.0))
FUNCTION_BEGIN = "  // BEGIN NET_PHOTOCURRENT_FUNCTION"
FUNCTION_END = "  // END NET_PHOTOCURRENT_FUNCTION"


def _number(value: float) -> str:
    return f"{float(value):.17g}"


def _unique_terms(payload: dict) -> dict[int, dict]:
    terms: dict[int, dict] = {}
    for branch in [*payload["branches"], payload["residual_branch"], payload["generic_branch"]]:
        for row in branch.get("terms", []):
            index = int(row["term_index"])
            if index in terms and terms[index]["term"] != row["term"]:
                raise ValueError(f"Conflicting names for feature index {index}")
            terms[index] = row
    return terms


def _raw_feature_numpy(name: str, raw: dict[str, np.ndarray], scalers: dict) -> np.ndarray:
    v = raw["light_voltage"]
    temp = raw["simulation_temperature"]
    length = raw["active_layer_length"]
    trap = raw["trap_assisted_recomb_A"]
    sio2 = raw["ge_sio2_recomb_velocity"]
    si = raw["ge_si_recomb_velocity"]
    reverse_soft = np.logaddexp(0.0, -v / 0.2)
    reverse_zero = np.maximum(reverse_soft - LOG2, 0.0)

    def z(column: str) -> np.ndarray:
        return (raw[column] - float(scalers[column]["mean"])) / float(scalers[column]["scale"])

    features = {
        "one": np.ones_like(v),
        "light_voltage_z": z("light_voltage"),
        "light_voltage_z2": z("light_voltage") ** 2,
        "soft_reverse_light_voltage": reverse_soft,
        "tanh_reverse_light_voltage": np.tanh(-v),
        "reverse_saturation_light_voltage": 1.0 - np.exp(-reverse_zero),
        "forward_gate_light_voltage": 1.0 / (1.0 + np.exp(-v / 0.2)),
        "reverse_gate_light_voltage": 1.0 / (1.0 + np.exp(v / 0.2)),
        "temperature_z": z("simulation_temperature"),
        "inverse_temperature": 1.0 / temp,
        "light_voltage_over_temperature": v / temp,
        "length_z": z("active_layer_length"),
        "inverse_length": 1.0 / length,
        "light_voltage_over_length": v / length,
        "trap_A_z": z("trap_assisted_recomb_A"),
        "ge_sio2_velocity_z": z("ge_sio2_recomb_velocity"),
        "ge_si_velocity_z": z("ge_si_recomb_velocity"),
        "ge_sio2_velocity_over_length": sio2 / length,
        "ge_si_velocity_over_length": si / length,
        "reverse_drive_over_length": reverse_zero / length,
        # The catalogue name says "scaled", but the authoritative feature is
        # the raw velocity sum followed by the serialized feature normalizer.
        "surface_recomb_sum_z": sio2 + si,
        "surface_recomb_over_length": (sio2 + si) / length,
        "reverse_drive_surface_recomb": reverse_zero * (sio2 + si) / length,
        "reverse_drive_trap_A": reverse_zero * trap,
        "collection_saturation_length": (1.0 - np.exp(-reverse_zero)) * length,
        "length_collection_saturation": 1.0 - np.exp(-np.maximum(length, 0.0) / 40.0),
        "thermal_surface_recomb_over_length": (sio2 + si) / (length * temp),
        "reverse_thermal_surface_recomb": reverse_zero * (sio2 + si) / (length * temp),
        "trap_A_times_length": trap * length,
        "trap_A_over_temperature": trap / temp,
    }
    if name not in features:
        raise ValueError(f"Unsupported net-photocurrent feature: {name}")
    return features[name]


def _raw_feature_va(name: str, scalers: dict) -> str:
    v = "light_voltage"
    temp = "simulation_temperature"
    length = "active_layer_length"
    trap = "trap_assisted_recomb_A"
    sio2 = "ge_sio2_recomb_velocity"
    si = "ge_si_recomb_velocity"
    reverse_soft = f"net_softplus(-{v}/0.2)"
    reverse_zero = f"net_pos(({reverse_soft})-{_number(LOG2)})"

    def z(column: str) -> str:
        item = scalers[column]
        return f"(({column})-({_number(item['mean'])}))/({_number(item['scale'])})"

    features = {
        "one": "1.0",
        "light_voltage_z": z("light_voltage"),
        "light_voltage_z2": f"net_sqr({z('light_voltage')})",
        "soft_reverse_light_voltage": reverse_soft,
        "tanh_reverse_light_voltage": f"tanh(-{v})",
        "reverse_saturation_light_voltage": f"1.0-exp(-({reverse_zero}))",
        "forward_gate_light_voltage": f"net_sigmoid({v}/0.2)",
        "reverse_gate_light_voltage": f"net_sigmoid(-{v}/0.2)",
        "temperature_z": z("simulation_temperature"),
        "inverse_temperature": f"1.0/{temp}",
        "light_voltage_over_temperature": f"{v}/{temp}",
        "length_z": z("active_layer_length"),
        "inverse_length": f"1.0/{length}",
        "light_voltage_over_length": f"{v}/{length}",
        "trap_A_z": z("trap_assisted_recomb_A"),
        "ge_sio2_velocity_z": z("ge_sio2_recomb_velocity"),
        "ge_si_velocity_z": z("ge_si_recomb_velocity"),
        "ge_sio2_velocity_over_length": f"{sio2}/{length}",
        "ge_si_velocity_over_length": f"{si}/{length}",
        "reverse_drive_over_length": f"({reverse_zero})/{length}",
        "surface_recomb_sum_z": f"({sio2}+{si})",
        "surface_recomb_over_length": f"({sio2}+{si})/{length}",
        "reverse_drive_surface_recomb": f"({reverse_zero})*({sio2}+{si})/{length}",
        "reverse_drive_trap_A": f"({reverse_zero})*{trap}",
        "collection_saturation_length": f"(1.0-exp(-({reverse_zero})))*{length}",
        "length_collection_saturation": f"1.0-exp(-net_pos({length})/40.0)",
        "thermal_surface_recomb_over_length": f"({sio2}+{si})/({length}*{temp})",
        "reverse_thermal_surface_recomb": f"({reverse_zero})*({sio2}+{si})/({length}*{temp})",
        "trap_A_times_length": f"{trap}*{length}",
        "trap_A_over_temperature": f"{trap}/{temp}",
    }
    if name not in features:
        raise ValueError(f"Unsupported net-photocurrent feature: {name}")
    return features[name]


def evaluate_deployment_graph(payload: dict, x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy replay of the exact graph emitted to Verilog-A."""
    inputs = tuple(payload["input_scalers"])
    raw_array = np.asarray(x_raw, dtype=np.float64)
    raw = {name: raw_array[:, index] for index, name in enumerate(inputs)}
    terms = _unique_terms(payload)
    phi: dict[int, np.ndarray] = {}
    centers = np.asarray(payload["feature_centers"], dtype=np.float64)
    scales = np.asarray(payload["feature_scales"], dtype=np.float64)
    for index, row in terms.items():
        phi[index] = (_raw_feature_numpy(row["term"], raw, payload["input_scalers"]) - centers[index]) / scales[index]

    def linear(branch: dict) -> np.ndarray:
        out = np.full(len(raw_array), float(branch["bias"]), dtype=np.float64)
        for row in branch.get("terms", []):
            out += float(row["coefficient_normalized_output"]) * phi[int(row["term_index"])]
        return out

    v = raw["light_voltage"]
    reverse_zero = np.maximum(np.logaddexp(0.0, -v / float(payload["reverse_scale"])) - LOG2, 0.0)
    forward_zero = np.maximum(np.logaddexp(0.0, v / float(payload["reverse_scale"])) - LOG2, 0.0)
    reverse_zero /= float(payload["reverse_norm"])
    forward_zero /= float(payload["reverse_norm"])

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
    main = sum(float(b["sign"]) * abs(float(b["scale"])) * value for b, value in values if b["role"] == "main")
    surface = sum(abs(float(b["scale"])) * value for b, value in values if b["role"] == "surface_loss")
    trap = sum(abs(float(b["scale"])) * value for b, value in values if b["role"] == "trap_loss")
    additive = sum(float(b["sign"]) * abs(float(b["scale"])) * value for b, value in values if b["role"] == "additive")
    normalized = (
        main * np.exp(-np.clip(surface, 0.0, 8.0)) * np.exp(-np.clip(trap, 0.0, 8.0))
        + additive
        + linear(payload["residual_branch"])
        + linear(payload["generic_branch"])
    )
    model_space = normalized * float(payload["output_scale"])
    physical = np.power(10.0, model_space)
    return model_space, physical


def generate_function(payload: dict) -> str:
    if payload.get("task_key") != "photo_current" or payload.get("output_space") != "log10":
        raise ValueError("Expected a log10 photo_current pure-symbolic payload")
    if payload.get("flat_gates"):
        raise ValueError("This exporter requires the audited hierarchical photocurrent graph")
    if payload.get("uses_kan_at_inference"):
        raise ValueError("The deployment payload must be KAN-free")

    terms = _unique_terms(payload)
    centers = payload["feature_centers"]
    scales = payload["feature_scales"]
    branch_names = [str(branch["name"]) for branch in payload["branches"]]
    declarations = [f"phi_{index}" for index in sorted(terms)] + branch_names + [
        "residual", "generic", "reverse_zero", "forward_zero", "normalized_output", "model_space_output"
    ]
    lines = [
        FUNCTION_BEGIN,
        "  // KAN-free student for I_photo,net = I_illum - I_dark.",
        "  analog function real net_pos;",
        "    input x; real x;",
        "    begin if (x > 0.0) net_pos = x; else net_pos = 0.0; end",
        "  endfunction",
        "",
        "  analog function real net_clip08;",
        "    input x; real x;",
        "    begin if (x < 0.0) net_clip08 = 0.0; else if (x > 8.0) net_clip08 = 8.0; else net_clip08 = x; end",
        "  endfunction",
        "",
        "  analog function real net_sqr;",
        "    input x; real x; begin net_sqr = x*x; end",
        "  endfunction",
        "",
        "  analog function real net_softplus;",
        "    input x; real x;",
        "    begin",
        "      if (x > 40.0) net_softplus = x;",
        "      else if (x < -40.0) net_softplus = exp(x);",
        "      else net_softplus = ln(1.0 + exp(x));",
        "    end",
        "  endfunction",
        "",
        "  analog function real net_sigmoid;",
        "    input x; real x;",
        "    begin",
        "      if (x >= 0.0) net_sigmoid = 1.0/(1.0 + exp(-x));",
        "      else net_sigmoid = exp(x)/(1.0 + exp(x));",
        "    end",
        "  endfunction",
        "",
        "  analog function real net_photocurrent_model;",
        "    input light_voltage, trap_assisted_recomb_A, ge_sio2_recomb_velocity, ge_si_recomb_velocity, active_layer_length, simulation_temperature;",
        "    real light_voltage, trap_assisted_recomb_A, ge_sio2_recomb_velocity, ge_si_recomb_velocity, active_layer_length, simulation_temperature;",
        "    real " + ", ".join(declarations) + ";",
        "    begin",
    ]
    for index, row in sorted(terms.items()):
        raw_expr = _raw_feature_va(row["term"], payload["input_scalers"])
        lines.append(
            f"      phi_{index} = (({raw_expr})-({_number(centers[index])}))/({_number(scales[index])}); // {row['term']}"
        )

    reverse_scale = _number(payload["reverse_scale"])
    reverse_norm = _number(payload["reverse_norm"])
    lines.extend(
        [
            f"      reverse_zero = net_pos(net_softplus(-light_voltage/{reverse_scale})-{_number(LOG2)})/{reverse_norm};",
            f"      forward_zero = net_pos(net_softplus(light_voltage/{reverse_scale})-{_number(LOG2)})/{reverse_norm};",
        ]
    )

    def linear(branch: dict) -> str:
        pieces = [_number(branch["bias"])]
        pieces.extend(
            f"({_number(row['coefficient_normalized_output'])})*phi_{int(row['term_index'])}"
            for row in branch.get("terms", [])
        )
        return " + ".join(pieces)

    def drive(name: str) -> str:
        return {
            "reverse_zero": "reverse_zero",
            "forward_zero": "forward_zero",
            "reverse": "net_sigmoid(4.0*reverse_zero)",
            "forward": "net_sigmoid(4.0*forward_zero)",
            "all": "1.0",
        }[name]

    for branch in payload["branches"]:
        lines.append(
            f"      {branch['name']} = ({_number(abs(float(branch['scale'])))})*net_softplus({linear(branch)})*({drive(str(branch['drive']))});"
        )
    main = " + ".join(
        f"({_number(branch['sign'])})*{branch['name']}" for branch in payload["branches"] if branch["role"] == "main"
    ) or "0.0"
    surface = " + ".join(branch["name"] for branch in payload["branches"] if branch["role"] == "surface_loss") or "0.0"
    trap = " + ".join(branch["name"] for branch in payload["branches"] if branch["role"] == "trap_loss") or "0.0"
    additive = " + ".join(
        f"({_number(branch['sign'])})*{branch['name']}" for branch in payload["branches"] if branch["role"] == "additive"
    ) or "0.0"
    lines.extend(
        [
            f"      residual = {linear(payload['residual_branch'])};",
            f"      generic = {linear(payload['generic_branch'])};",
            f"      normalized_output = ({main})*exp(-net_clip08({surface}))*exp(-net_clip08({trap})) + ({additive}) + residual + generic;",
            f"      model_space_output = ({_number(payload['output_scale'])})*normalized_output;",
            "      net_photocurrent_model = pow(10.0, model_space_output);",
            "    end",
            "  endfunction",
            FUNCTION_END,
        ]
    )
    return "\n".join(lines)


def update_veriloga(template: str, function_text: str) -> str:
    if FUNCTION_BEGIN in template:
        pattern = re.compile(re.escape(FUNCTION_BEGIN) + r".*?" + re.escape(FUNCTION_END), re.DOTALL)
        template, count = pattern.subn(function_text, template, count=1)
        if count != 1:
            raise RuntimeError("Could not replace the existing net-photocurrent function")
    else:
        marker = "\nanalog begin\n"
        if marker not in template:
            raise RuntimeError("Could not locate the Verilog-A analog block")
        template = template.replace(marker, "\n" + function_text + marker, 1)
    assignment = re.compile(r"^\s*photo_current\s*=.*?;\s*$", re.MULTILINE)
    replacement = (
        "    photo_current = net_photocurrent_model(vd_safe, trap_assisted_recomb_A, "
        "ge_sio2_recomb_velocity, ge_si_recomb_velocity, active_layer_length, T);"
    )
    template, count = assignment.subn(replacement, template, count=1)
    if count != 1:
        raise RuntimeError("Could not replace the old photo_current assignment")
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.formula.read_text(encoding="utf-8"))
    spec = TASKS["photo_current"]
    prepared, _ = prepare_task_dataframe(apply_dark_current_floor(pd.read_csv(args.data)), spec)
    frame = prepared
    inputs = tuple(payload["input_scalers"])
    x_raw = frame[list(inputs)].to_numpy(dtype=np.float64)
    authoritative_model_space = evaluate_exported_pure_symbolic_formula(payload, x_raw)
    deployment_model_space, deployment_physical = evaluate_deployment_graph(payload, x_raw)
    model_error = np.abs(deployment_model_space - authoritative_model_space)
    physical_reference = np.power(10.0, authoritative_model_space)
    physical_error = np.abs(deployment_physical - physical_reference)
    # The public replay intentionally reconstructs the feature graph in
    # float32, matching the exported-student audit.  The deployment evaluator
    # uses float64, so use the same 1e-6 relative acceptance convention as the
    # original independent JSON replay.
    tolerance = max(1e-10, 1e-6 * float(np.max(np.abs(authoritative_model_space))))
    if not np.all(np.isfinite(deployment_physical)) or float(model_error.max()) > tolerance:
        raise RuntimeError(
            f"Independent deployment replay failed: max model-space error={model_error.max():.6g}, tolerance={tolerance:.6g}"
        )

    function_text = generate_function(payload)
    updated = update_veriloga(args.template.read_text(encoding="utf-8"), function_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(updated, encoding="utf-8")
    audit = {
        "target_definition": "net_photocurrent = illuminated_total_current - dark_current at paired bias/condition rows",
        "formula": str(args.formula),
        "data": str(args.data),
        "samples": int(len(frame)),
        "selected_branch_terms": int(payload["selected_term_count"]),
        "unique_features": int(len(_unique_terms(payload))),
        "independent_replay_model_space_max_abs_error": float(model_error.max()),
        "independent_replay_model_space_tolerance": float(tolerance),
        "independent_replay_physical_max_abs_error_A": float(physical_error.max()),
        "finite_physical_rate": float(np.isfinite(deployment_physical).mean()),
        "veriloga_function_markers_present": FUNCTION_BEGIN in updated and FUNCTION_END in updated,
        "old_inline_photo_assignment_removed": "photo_current = smax0(pow(10" not in updated,
        "spectre_status": "not rerun for this corrected net-photocurrent source",
        "passed_local_replay": True,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
