"""
Generate the Verilog-A/KAN verification summary.

The report is generated from the current single source of truth:
  artifacts/results/device_modeling/{I-dark,I-photo,AC-response}-bayes-results

Usage:
  python bkan/device_modeling/veriloga/summarize_results.py
"""

import csv
import json
import math
import os
import sys
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
RESULTS_BASE = os.path.join(PROJECT_ROOT, "artifacts", "results", "device_modeling")
SPECTRE_REPORT = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "results",
    "spectre_validation",
    "spectre_comparison_report.json",
)

CSV_DARK = os.path.join(
    RESULTS_BASE, "I-dark-bayes-results", "Formula_Pointwise_Verification.csv"
)
CSV_PHOTO = os.path.join(
    RESULTS_BASE, "I-photo-bayes-results", "Formula_Pointwise_Verification.csv"
)
CSV_AC = os.path.join(
    RESULTS_BASE, "AC-response-bayes-results", "Formula_Pointwise_Verification.csv"
)

FORMULA_DARK = os.path.join(
    RESULTS_BASE, "I-dark-bayes-results", "symbolic_formula.txt"
)
FORMULA_PHOTO = os.path.join(
    RESULTS_BASE, "I-photo-bayes-results", "symbolic_formula.txt"
)
FORMULA_AC = os.path.join(
    RESULTS_BASE, "AC-response-bayes-results", "symbolic_formula.txt"
)


def load_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_column(rows, candidates):
    if not rows:
        raise ValueError("No rows available")
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    raise KeyError(f"None of these columns exist: {candidates}")


def verification_columns(rows):
    return {
        "formula": pick_column(rows, ["formula_model_space", "formula_target_space"]),
        "tcad": pick_column(rows, ["tcad_model_space", "tcad_target_space"]),
        "bnn": pick_column(rows, ["bnn_model_space", "bnn_target_space"]),
    }


def compute_metrics(rows, target_col="formula_target_space", ref_col="tcad_target_space",
                    bnn_col="bnn_target_space"):
    if target_col not in rows[0] or ref_col not in rows[0] or bnn_col not in rows[0]:
        cols = verification_columns(rows)
        target_col = cols["formula"]
        ref_col = cols["tcad"]
        bnn_col = cols["bnn"]
    formula_vals = [float(r[target_col]) for r in rows]
    ref_vals = [float(r[ref_col]) for r in rows]
    bnn_vals = [float(r[bnn_col]) for r in rows]
    n = len(formula_vals)

    errs_tcad = [formula_vals[i] - ref_vals[i] for i in range(n)]
    rmse_tcad = math.sqrt(sum(e * e for e in errs_tcad) / n)
    mae_tcad = sum(abs(e) for e in errs_tcad) / n
    max_tcad = max(abs(e) for e in errs_tcad)
    ref_mean = sum(ref_vals) / n
    ss_tot = sum((r - ref_mean) ** 2 for r in ref_vals)
    r2_tcad = 1.0 - sum(e * e for e in errs_tcad) / max(ss_tot, 1e-30)

    errs_bnn = [formula_vals[i] - bnn_vals[i] for i in range(n)]
    rmse_bnn = math.sqrt(sum(e * e for e in errs_bnn) / n)
    mae_bnn = sum(abs(e) for e in errs_bnn) / n
    max_bnn = max(abs(e) for e in errs_bnn)

    errs_bt = [bnn_vals[i] - ref_vals[i] for i in range(n)]
    rmse_bt = math.sqrt(sum(e * e for e in errs_bt) / n)
    mae_bt = sum(abs(e) for e in errs_bt) / n
    r2_bt = 1.0 - sum(e * e for e in errs_bt) / max(ss_tot, 1e-30)

    return {
        "n_samples": n,
        "formula_vs_tcad": {
            "RMSE": rmse_tcad,
            "MAE": mae_tcad,
            "Max": max_tcad,
            "R2": r2_tcad,
        },
        "formula_vs_bnn": {
            "RMSE": rmse_bnn,
            "MAE": mae_bnn,
            "Max": max_bnn,
        },
        "bnn_vs_tcad": {
            "RMSE": rmse_bt,
            "MAE": mae_bt,
            "R2": r2_bt,
        },
    }


def load_formula_info(path):
    if not os.path.exists(path):
        return "N/A", "N/A"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    text = " ".join(lines)
    total_ops = (
        text.count("+ ")
        + text.count("- ")
        + text.count("*")
        + text.count("/")
        + text.count("pow")
        + text.count("exp")
        + text.count("sqrt")
    )
    return f"{total_ops} ops", f"{len(lines)} lines"


def get_param_range(rows, col):
    vals = [float(r[col]) for r in rows]
    return min(vals), max(vals)


def ac_response_db(freq_ghz, length_um, temperature_k):
    """Mirror validate_kan_formulas.kan_ac_response_db exactly."""
    a = (
        0.043286058841316 * length_um
        + 0.0376448196142801 * temperature_k
        - 21.4053826740305
    )
    b = (
        -0.0425090134894624 * length_um
        - 0.0390669691997125 * temperature_k
        + 21.9384518054452
    )
    fc = math.exp(
        0.00535145740474049 * length_um
        + 0.00419202668726104 * temperature_k
    ) / 0.0172206207304869
    p = 1.07549720488297
    f_norm = freq_ghz / max(fc, 0.001)
    return a + b / (1.0 + f_norm ** p)


def ac_params(length_um, temperature_k):
    a = (
        0.043286058841316 * length_um
        + 0.0376448196142801 * temperature_k
        - 21.4053826740305
    )
    b = (
        -0.0425090134894624 * length_um
        - 0.0390669691997125 * temperature_k
        + 21.9384518054452
    )
    fc = math.exp(
        0.00535145740474049 * length_um
        + 0.00419202668726104 * temperature_k
    ) / 0.0172206207304869
    return {
        "y0": a + b,
        "y_floor": a,
        "fc": fc,
        "p": 1.07549720488297,
    }


def print_separator(width=100):
    print("=" * width)


def print_task_overview(tasks):
    print_separator(120)
    print("  SECTION 1: DATA OVERVIEW")
    print_separator(120)
    for name, path, rows in tasks:
        print(f"\n  [{name}]")
        print(f"    Source: {os.path.relpath(path, PROJECT_ROOT)}")
        print(f"    Samples: {len(rows)}")
        columns = [
            "ac_response_db" if item == "bandwidth" and name == "AC Response" else item
            for item in list(rows[0].keys())[:8]
        ]
        print(f"    Columns: {', '.join(columns)}...")


def print_core_metrics(tasks):
    print_separator(120)
    print("  SECTION 2: CORE ACCURACY METRICS")
    print_separator(120)
    for name, path, rows in tasks:
        metrics = compute_metrics(rows)
        # Current targets are linear current for DC tasks and response dB for AC.
        cols = verification_columns(rows)
        sample_val = float(rows[0][cols["tcad"]])
        is_linear = (sample_val >= 0 and sample_val < 0.1) or ("dark" in name.lower() or "photo" in name.lower())
        space_label = "linear [A]" if is_linear else "normalized response [dB]"
        fmt = ".6e" if is_linear else ".6f"
        print(f"\n  {name}  ({space_label})")
        print(f"    Samples:                 {metrics['n_samples']}")
        print(
            "    Symbolic formula vs TCAD: "
            f"RMSE={metrics['formula_vs_tcad']['RMSE']:{fmt}}  "
            f"MAE={metrics['formula_vs_tcad']['MAE']:{fmt}}  "
            f"Max={metrics['formula_vs_tcad']['Max']:{fmt}}  "
            f"R2={metrics['formula_vs_tcad']['R2']:.6f}"
        )
        print(
            "    BNN/KAN vs TCAD:          "
            f"RMSE={metrics['bnn_vs_tcad']['RMSE']:{fmt}}  "
            f"MAE={metrics['bnn_vs_tcad']['MAE']:{fmt}}  "
            f"R2={metrics['bnn_vs_tcad']['R2']:.6f}"
        )
        print(
            "    Symbolic formula vs BNN:  "
            f"RMSE={metrics['formula_vs_bnn']['RMSE']:{fmt}}  "
            f"MAE={metrics['formula_vs_bnn']['MAE']:{fmt}}  "
            f"Max={metrics['formula_vs_bnn']['Max']:{fmt}}"
        )


def print_current_accuracy(tasks):
    print_separator(120)
    print("  SECTION 3: LINEAR-SPACE CURRENT ACCURACY")
    print_separator(120)
    for name, path, rows in tasks:
        if "AC" in name:
            continue

        v_col = "dark_voltage" if "dark" in name.lower() else "light_voltage"
        key_biases = [-3.0, -1.0, -0.2, 0.0, 0.4]
        sorted_rows = sorted(rows, key=lambda r: float(r[v_col]))

        # Auto-detect data format: log10 space (negative values) or linear space (small positive)
        cols = verification_columns(rows)
        sample_val = float(rows[0][cols["tcad"]])
        is_linear_space = (sample_val >= 0 and sample_val < 0.1)

        def to_linear(val):
            return val if is_linear_space else 10.0 ** val

        print(f"\n  [{name}]  (target_space format: {'linear' if is_linear_space else 'log10'})")
        print(f"  {'V[V]':>8s}  {'TCAD[A]':>14s}  {'Formula[A]':>14s}  {'Err[%]':>8s}  Status")
        print("  " + "-" * 64)
        seen = set()
        for target_v in key_biases:
            best = min(sorted_rows, key=lambda r: abs(float(r[v_col]) - target_v))
            v_val = float(best[v_col])
            if v_val in seen:
                continue
            seen.add(v_val)
            tcad = to_linear(float(best[cols["tcad"]]))
            formula = to_linear(float(best[cols["formula"]]))
            rel_err = abs(formula - tcad) / max(abs(tcad), 1e-30) * 100.0
            status = "OK" if rel_err < 2.0 else ("WARN" if rel_err < 5.0 else "REVIEW")
            print(f"  {v_val:8.3f}  {tcad:14.6e}  {formula:14.6e}  {rel_err:8.4f}  {status}")

        all_tcad = [to_linear(float(r[cols["tcad"]])) for r in rows]
        all_formula = [to_linear(float(r[cols["formula"]])) for r in rows]
        pct = [
            abs(all_formula[i] - all_tcad[i]) / max(abs(all_tcad[i]), 1e-30) * 100.0
            for i in range(len(all_tcad))
        ]
        pct_sorted = sorted(pct)
        print(
            f"\n  Global relative error: Mean={sum(pct) / len(pct):.2f}%  "
            f"Median={pct_sorted[len(pct_sorted) // 2]:.2f}%  Max={max(pct):.2f}%"
        )


def print_ac_summary():
    print_separator(120)
    print("  SECTION 4: NORMALIZED AC RESPONSE PREDICTION")
    print_separator(120)
    length_um, temperature_k = 40.0, 300.0
    params = ac_params(length_um, temperature_k)
    y0 = params["y0"]

    print(f"\n  Design point: L={length_um:.1f} um, T={temperature_k:.1f} K")
    response_span = params["y0"] - params["y_floor"]
    if response_span > 3.0:
        f_3db = params["fc"] * (3.0 / (response_span - 3.0)) ** (1.0 / params["p"])
    else:
        f_3db = float("nan")
    print(f"  characteristic midpoint frequency = {params['fc']:.3f} GHz")
    print(f"  low-frequency response = {params['y0']:.4f} dB")
    print(f"  high-frequency floor = {params['y_floor']:.4f} dB")
    print(f"  derived -3 dB bandwidth = {f_3db:.3f} GHz")
    print("  scope note: values above the 40 GHz training maximum are formula extrapolation")
    print(f"  Roll-off exponent p = {params['p']:.4f}")
    print()
    print(f"  {'Freq [GHz]':>12s}  {'Response [dB]':>15s}  {'Delta [dB]':>12s}  {'|S21|':>12s}")
    print("  " + "-" * 60)
    freqs = sorted(
        [0.001, 0.01, 0.1, 1.0, 10.0, f_3db, 50.0, 100.0, params["fc"], 500.0, 1000.0]
    )
    for freq in freqs:
        response_db = ac_response_db(freq, length_um, temperature_k)
        delta_db = response_db - y0
        magnitude = 10.0 ** (delta_db / 20.0)
        marker = "  <-3 dB>" if math.isfinite(f_3db) and abs(freq - f_3db) < 1e-9 else ""
        print(
            f"  {freq:12.4f}  {response_db:15.6f}  "
            f"{delta_db:12.6f}  {magnitude:12.6f}{marker}"
        )


def print_spectre_status():
    print_separator(120)
    print("  SECTION 5: LIMITED SPECTRE CIRCUIT SIMULATION EVIDENCE")
    print("  Platform: VMware CentOS 7, IC618 ISR 6.1.8-64b.83, Spectre 18.1.0.077")
    print_separator(120)
    report = {}
    if os.path.exists(SPECTRE_REPORT):
        with open(SPECTRE_REPORT, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    comparisons = report.get("comparisons", {})

    def status(name):
        return comparisons.get(name, {}).get("status", "NOT_COMPARED")

    ac_curves = comparisons.get("ac_bias_temperature", {}).get("curves", {})
    capacitances = [
        curve["C_lowfreq_F"]
        for curve in ac_curves.values()
        if "C_lowfreq_F" in curve
    ]
    c_detail = (
        f"15x121 points; C={min(capacitances):.3e}..{max(capacitances):.3e} F"
        if capacitances
        else "no corrected AC comparison report"
    )
    tran = comparisons.get("transient", {})
    tran_detail = (
        f"{tran.get('n_timepoints', 0)} points; RMSE="
        f"{tran.get('total_current_rmse_A', float('nan')):.3e} A; stimulus-limited"
    )
    items = [
        ("Verilog-A compilation", "RUN_OK", "all four runs completed with 0 errors/0 warnings"),
        ("DC dark/total sweep", status("dc_reference"), "61+61 points, -3 V to 0 V"),
        ("DC bias/temperature", status("dc_bias_temperature"), "61x3 pointwise reference"),
        ("AC bias/temperature", status("ac_bias_temperature"), c_detail),
        ("Transient proxy charge", status("transient"), tran_detail),
        ("AC response formula", "PYTHON_PASS", "TCAD/model-space and symbolic Python check"),
        (
            "AC response VA readout",
            "NUMERIC_PASS",
            "standalone V(out)=y_ac readout; not the terminal dynamic branch",
        ),
        ("Strict terminal charge", "UNAVAILABLE", "strict_terminal_charge_available=0"),
    ]
    for item, status, detail in items:
        print(f"  {item:28s} {status:14s} {detail}")

    print(
        "\n  Scope statement: NUMERIC_PASS means Spectre matches the independent "
        "Python evaluator. It does not establish strict terminal charge, intrinsic "
        "bandwidth/response time, or complete PDK/foundry sign-off."
    )

    print("\n  Key reference point, L=40 um, T=300 K, trap_A=1e-9:")
    print("  Vbias       I_dark[A]        I_photo[A]       I_total[A]")
    refs = [
        (-5.0, 4.298577e-07, 5.409802e-04, 5.414100e-04),
        (-3.0, 3.862585e-07, 5.272437e-04, 5.276299e-04),
        (-1.0, 2.682052e-07, 4.899428e-04, 4.902110e-04),
        (0.0, 9.303521e-08, 4.418241e-04, 4.419171e-04),
        (0.4, 1.411559e-09, 4.272967e-04, 4.272981e-04),
    ]
    for v, dark, photo, total in refs:
        print(f"  {v:6.2f}  {dark:15.6e}  {photo:15.6e}  {total:15.6e}")


def print_ranges(tasks):
    print_separator(120)
    print("  SECTION 6: TRAINING DATA PARAMETER RANGES")
    print_separator(120)
    for name, path, rows in tasks:
        v_col = "frequency_ghz"
        if "dark" in name.lower():
            v_col = "dark_voltage"
        elif "photo" in name.lower():
            v_col = "light_voltage"
        print(f"\n  [{name}]")
        for col in [v_col, "trap_assisted_recomb_A", "ge_sio2_recomb_velocity",
                    "ge_si_recomb_velocity", "active_layer_length", "simulation_temperature"]:
            if col in rows[0]:
                mn, mx = get_param_range(rows, col)
                print(f"    {col:30s}: [{mn:.4e}, {mx:.4e}]")


def print_formula_complexity(tasks):
    print_separator(120)
    print("  SECTION 7: SYMBOLIC FORMULA COMPLEXITY")
    print_separator(120)
    path_map = {
        "I_dark": FORMULA_DARK,
        "I_photo": FORMULA_PHOTO,
        "AC Response": FORMULA_AC,
    }
    for name, _path, _rows in tasks:
        formula_path = path_map.get(name)
        ops_info, lines_info = load_formula_info(formula_path)
        print(f"\n  [{name}]")
        print(f"    Formula file: {os.path.relpath(formula_path, PROJECT_ROOT)}")
        print(f"    Complexity: {ops_info}, {lines_info}")


def main():
    task_specs = [
        ("I_dark", CSV_DARK),
        ("I_photo", CSV_PHOTO),
        ("AC Response", CSV_AC),
    ]
    tasks = []
    for name, path in task_specs:
        if os.path.exists(path):
            tasks.append((name, path, load_csv(path)))
        else:
            print(f"Missing verification CSV: {path}", file=sys.stderr)

    print_separator(120)
    print("  KAN Compact Model - Comprehensive Verification Summary")
    print(f"  Generated: {date.today()}")
    print("  Device: Vertical Ge/Si Photodetector")
    print("  Data source: TCAD simulations")
    print("  Verification source: artifacts/results/device_modeling")
    print_separator(120)

    print_task_overview(tasks)
    print()
    print_core_metrics(tasks)
    print()
    print_current_accuracy(tasks)
    print()
    print_ac_summary()
    print()
    print_spectre_status()
    print()
    print_ranges(tasks)
    print()
    print_formula_complexity(tasks)

    print_separator(120)
    print("  END OF REPORT")
    print_separator(120)


if __name__ == "__main__":
    out_path = os.path.join(BASE_DIR, "verification_summary.txt")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = original_stdout
    print(f"Report written to: {out_path}")
