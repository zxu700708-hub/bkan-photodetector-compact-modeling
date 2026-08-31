#!/usr/bin/env python3
"""
Generate Verilog-A validation artifacts from the current symbolic formulas.

The generated files overwrite the standard Verilog-A validation entry points
so the CentOS/Spectre flow always runs the latest extracted model.
"""

from __future__ import annotations

import math
from pathlib import Path

import sympy as sp
from sympy.printing.c import C99CodePrinter


VERILOGA_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "artifacts" / "results" / "device_modeling"
OUT_DIR = VERILOGA_DIR

DARK_FORMULA = RESULT_ROOT / "I-dark-bayes-results" / "symbolic_formula.txt"
PHOTO_FORMULA = RESULT_ROOT / "I-photo-bayes-results" / "symbolic_formula.txt"
AC_FORMULA = RESULT_ROOT / "AC-response-bayes-results" / "symbolic_formula.txt"
DARK_GATE_START_V = -0.2
DARK_GATE_END_V = 0.0


def dark_boundary_gate(voltage: float) -> float:
    u = min(
        max(
            (voltage - DARK_GATE_START_V)
            / (DARK_GATE_END_V - DARK_GATE_START_V),
            0.0,
        ),
        1.0,
    )
    return 1.0 - u * u * (3.0 - 2.0 * u)


class VerilogAPrinter(C99CodePrinter):
    """Small C-like printer tuned for Spectre Verilog-A expressions."""

    def _print_Pow(self, expr):
        base, exponent = expr.as_base_exp()
        base_text = self._print(base)
        if exponent == 2:
            return f"sqr({base_text})"
        if exponent == sp.Rational(1, 2):
            return f"safe_sqrt_kan({base_text}, sqrt_eps, smooth_delta)"
        return f"pow({base_text}, {self._print(exponent)})"

    def _print_Function(self, expr):
        if expr.func == sp.exp:
            return f"safe_exp_kan({self._print(expr.args[0])})"
        return super()._print_Function(expr)

    def _print_exp(self, expr):
        return f"safe_exp_kan({self._print(expr.args[0])})"


def extract_model_formula(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    markers = (
        "Formula in model space:",
        "Formula in Transformed space:",
        "Formula in transformed space:",
        "Formula in Transformed/Linear space:",
    )
    marker = next((candidate for candidate in markers if candidate in text), None)
    if marker is None:
        raise ValueError(f"Could not find model-space formula in {path}")
    tail = text.split(marker, 1)[1]
    lines = []
    for raw_line in tail.splitlines():
        line = raw_line.strip()
        if not line:
            if lines:
                break
            continue
        if line.startswith("Formula in physical space:") or line.startswith("Formula in Linear space:"):
            break
        lines.append(line)
    if not lines:
        raise ValueError(f"Empty model-space formula in {path}")
    return " ".join(lines)


def extract_linear_formula(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    markers = (
        "Formula in Linear space:",
        "Formula in physical space:",
        "Formula in Physical space:",
    )
    marker = next((candidate for candidate in markers if candidate in text), None)
    if marker is None:
        raise ValueError(
            f"Could not find linear/physical formula in {path}; "
            "DC Verilog-A export must use physical current, not log/model space."
        )
    tail = text.split(marker, 1)[1]
    lines = []
    for raw_line in tail.splitlines():
        line = raw_line.strip()
        if not line:
            if lines:
                break
            continue
        lines.append(line)
    if not lines:
        raise ValueError(f"Empty linear/physical formula in {path}")
    return " ".join(lines)


def parse_formula(formula: str) -> sp.Expr:
    local_dict = {"exp": sp.exp, "sqrt": sp.sqrt}
    if "=" in formula:
        formula = formula.split("=", 1)[1].strip()
    return sp.sympify(formula.replace("^", "**"), locals=local_dict)


def replace_symbols(expr: sp.Expr, mapping: dict[str, str]) -> sp.Expr:
    return expr.xreplace({sp.Symbol(src): sp.Symbol(dst) for src, dst in mapping.items()})


def va_expr(expr: sp.Expr) -> str:
    return VerilogAPrinter(settings={"strict": False}).doprint(expr)


def generate_dc_model(dark_expr: sp.Expr, photo_expr: sp.Expr) -> str:
    dark_va = va_expr(
        replace_symbols(
            dark_expr,
            {
                "dark_voltage": "vd_safe",
                "simulation_temperature": "T",
            },
        )
    )
    photo_va = va_expr(
        replace_symbols(
            photo_expr,
            {
                "light_voltage": "vd_safe",
                "simulation_temperature": "T",
            },
        )
    )

    return f"""// Verilog-A Compact Model for Ge/Si Photodetector
// Generated from current symbolic KAN formulas.
// Source: artifacts/results/device_modeling/*/symbolic_formula.txt

`include "disciplines.vams"
`include "constants.vams"

module ge_si_photodetector_ic618(anode, cathode);

  inout anode, cathode;
  electrical anode, cathode;

  parameter real active_layer_length = 40.0 from (0.1:100.0);
  parameter real trap_assisted_recomb_A = 1.0e-9 from [0:1.0e-5];
  parameter real ge_sio2_recomb_velocity = 2500.0 from [1.0:1.0e6];
  parameter real ge_si_recomb_velocity = 500.0 from [1.0:1.0e6];
  parameter real sim_temp = 300.0;
  parameter real area = 1.0;
  // Dimensionless selector/scale for the fixed-illumination net-photocurrent branch.
  // Only light_power=0 (dark) and 1 (the characterized illumination) are validated.
  parameter real light_power = 0.0 from [0.0:1.0];

  parameter real v_min = -3.0;
  parameter real v_max = 0.0;
  parameter real sqrt_eps = 1.0e-18;
  parameter real smooth_delta = 1.0e-18;

  real vd, vd_safe, T;
  real dark_raw, photo_raw;
  real dark_current, net_photo_current, total_current;

  analog function real sqr;
    input x;
    real x;
    begin
      sqr = x * x;
    end
  endfunction

  analog function real smax0;
    input x, delta;
    real x, delta;
    begin
      smax0 = 0.5 * (x + sqrt(x*x + 4.0*delta*delta));
    end
  endfunction

  analog function real safe_sqrt_kan;
    input arg, eps, delta;
    real arg, eps, delta;
    begin
      safe_sqrt_kan = sqrt(eps + smax0(arg, delta));
    end
  endfunction

  analog function real safe_exp_kan;
    input x;
    real x;
    begin
      if (x < -80.0)
        safe_exp_kan = exp(-80.0);
      else if (x <= 80.0)
        safe_exp_kan = exp(x);
      else
        safe_exp_kan = exp(80.0) * (1.0 + x - 80.0);
    end
  endfunction

  analog function real dark_boundary_gate;
    input v;
    real v, u;
    begin
      if (v <= {DARK_GATE_START_V})
        dark_boundary_gate = 1.0;
      else if (v >= {DARK_GATE_END_V})
        dark_boundary_gate = 0.0;
      else begin
        u = (v - ({DARK_GATE_START_V})) /
            (({DARK_GATE_END_V}) - ({DARK_GATE_START_V}));
        dark_boundary_gate = 1.0 - u*u*(3.0 - 2.0*u);
      end
    end
  endfunction

  analog begin
    vd = V(anode, cathode);
    T = sim_temp;

    if (vd < v_min)
      vd_safe = v_min;
    else if (vd > v_max)
      vd_safe = v_max;
    else
      vd_safe = vd;

    dark_raw = {dark_va};

    photo_raw = {photo_va};

    dark_current = area * dark_boundary_gate(vd_safe) * smax0(dark_raw, smooth_delta);
    net_photo_current = area * light_power * smax0(photo_raw, smooth_delta);
    total_current = dark_current + net_photo_current;

    I(anode, cathode) <+ total_current;
  end

endmodule
"""


def generate_dc_testbench() -> str:
    return """simulator lang=spectre

ahdl_include "<CADENCE_RUN_ROOT>/ge_si_pdet_fixed.va"

parameters Vb_dark=0 Vb_total=0

// Dark current sweep for the current formula model.
Vdark (a_dark 0) vsource dc=Vb_dark
Xdark (a_dark 0) ge_si_photodetector_ic618 active_layer_length=40.0 sim_temp=300.0 trap_assisted_recomb_A=1.0e-9 ge_sio2_recomb_velocity=2500.0 ge_si_recomb_velocity=500.0 area=1.0 light_power=0.0
dc_dark dc param=Vb_dark start=-3 stop=0 step=0.05

// Dark + photocurrent sweep for the current formula model.
Vtotal (a_total 0) vsource dc=Vb_total
Xtotal (a_total 0) ge_si_photodetector_ic618 active_layer_length=40.0 sim_temp=300.0 trap_assisted_recomb_A=1.0e-9 ge_sio2_recomb_velocity=2500.0 ge_si_recomb_velocity=500.0 area=1.0 light_power=1.0
dc_total dc param=Vb_total start=-3 stop=0 step=0.05
"""


def generate_ac_model(ac_expr: sp.Expr) -> str:
    ac_va = va_expr(
        replace_symbols(
            ac_expr,
            {
                "simulation_temperature": "T",
            },
        )
    )
    return f"""// Verilog-A helper for the 2026-06-15 AC compact formula.
// This file is not a two-terminal small-signal device model; it exposes
// the fitted bandwidth expression for downstream integration.

`include "disciplines.vams"
`include "constants.vams"

module pdet_ac_compact_20260615(out);
  output out;
  electrical out;

  parameter real active_layer_length = 40.0 from (0.1:100.0);
  parameter real sim_temp = 300.0;
  parameter real frequency_ghz = 1.0 from [0:1.0e6];
  parameter real sqrt_eps = 1.0e-18;
  parameter real smooth_delta = 1.0e-18;

  real T;
  real bandwidth;

  analog function real sqr;
    input x;
    real x;
    begin
      sqr = x * x;
    end
  endfunction

  analog function real smax0;
    input x, delta;
    real x, delta;
    begin
      smax0 = 0.5 * (x + sqrt(x*x + 4.0*delta*delta));
    end
  endfunction

  analog function real safe_sqrt_kan;
    input arg, eps, delta;
    real arg, eps, delta;
    begin
      safe_sqrt_kan = sqrt(eps + smax0(arg, delta));
    end
  endfunction

  analog function real safe_exp_kan;
    input x;
    real x;
    begin
      if (x < -80.0)
        safe_exp_kan = exp(-80.0);
      else if (x <= 80.0)
        safe_exp_kan = exp(x);
      else
        safe_exp_kan = exp(80.0) * (1.0 + x - 80.0);
    end
  endfunction

  analog begin
    T = sim_temp;
    bandwidth = {ac_va};
    V(out) <+ bandwidth;
  end
endmodule
"""


def generate_reference(dark_expr: sp.Expr, photo_expr: sp.Expr) -> str:
    symbols = {
        name: sp.Symbol(name)
        for name in (
            "dark_voltage",
            "light_voltage",
            "active_layer_length",
            "trap_assisted_recomb_A",
            "ge_sio2_recomb_velocity",
            "ge_si_recomb_velocity",
            "simulation_temperature",
        )
    }
    dark_fn = sp.lambdify(tuple(symbols.values()), dark_expr, "math")
    photo_fn = sp.lambdify(tuple(symbols.values()), photo_expr, "math")

    fixed = {
        "active_layer_length": 40.0,
        "trap_assisted_recomb_A": 1.0e-9,
        "ge_sio2_recomb_velocity": 2500.0,
        "ge_si_recomb_velocity": 500.0,
        "simulation_temperature": 300.0,
    }

    lines = [
        "# Current formula reference generated from artifacts/results/device_modeling",
        "# Vbias I_dark I_photo I_total",
    ]
    for idx in range(61):
        v = -3.0 + 0.05 * idx
        args = (
            v,
            v,
            fixed["active_layer_length"],
            fixed["trap_assisted_recomb_A"],
            fixed["ge_sio2_recomb_velocity"],
            fixed["ge_si_recomb_velocity"],
            fixed["simulation_temperature"],
        )
        dark = dark_boundary_gate(v) * max(float(dark_fn(*args)), 0.0)
        photo = max(float(photo_fn(*args)), 0.0)
        total = dark + photo
        if not all(math.isfinite(x) for x in (dark, photo, total)):
            raise FloatingPointError(f"Non-finite reference at V={v}")
        lines.append(f"{v:.3f} {dark:.12e} {photo:.12e} {total:.12e}")
    return "\n".join(lines) + "\n"


def main() -> int:
    dark_expr = parse_formula(extract_linear_formula(DARK_FORMULA))
    photo_expr = parse_formula(extract_linear_formula(PHOTO_FORMULA))
    ac_expr = parse_formula(extract_model_formula(AC_FORMULA))

    outputs = {
        OUT_DIR / "ge_si_pdet_fixed.va": generate_dc_model(dark_expr, photo_expr),
        OUT_DIR / "testbench_dc_ic618.scs": generate_dc_testbench(),
        OUT_DIR / "spectre_reference.txt": generate_reference(dark_expr, photo_expr),
    }
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
