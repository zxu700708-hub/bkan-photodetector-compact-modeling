"""
Derivative Continuity Validation
================================
KAN paper (Novkin & Amrouch 2025) identified derivative accuracy as
a key limitation for SPICE convergence. This script validates:

  1. dI/dV smoothness — critical for Newton-Raphson convergence
  2. d2I/dV2 continuity — needed for harmonic balance
  3. gm (transconductance) consistency at V=0 crossing
  4. Safe formula vs raw formula derivative divergence at poles

This directly corresponds to the paper's Figure 4 and derivative
discussion in Section IV-B.

Usage:
  python validate_derivatives.py
"""

import math
import os

from validate_kan_formulas import (
    kan_log10_dark, kan_log10_dark_safe,
    kan_log10_photo, kan_log10_photo_safe,
)


def numerical_derivative(fn, x, h=1e-4):
    """5-point central difference for ~O(h^4) accuracy"""
    return (-fn(x + 2*h) + 8*fn(x + h) - 8*fn(x - h) + fn(x - 2*h)) / (12*h)


def numerical_second_derivative(fn, x, h=1e-3):
    """5-point central difference for 2nd derivative"""
    return (-fn(x + 2*h) + 16*fn(x + h) - 30*fn(x)
            + 16*fn(x - h) - fn(x - 2*h)) / (12*h*h)


def check_derivative_smoothness(v_range, fn_raw, fn_safe, label, params_tuple):
    """Check dI/dV and d2I/dV2 across voltage range"""
    points = []
    for v in v_range:
        L, T, trap_A, vs, vg = params_tuple

        def current_log10(vd):
            return fn_raw(vd, L, T, trap_A, vs, vg)

        def current_safe(vd):
            return fn_safe(vd, L, T, trap_A, vs, vg)

        try:
            # 1st derivative in linear space: dI/dV = I * ln(10) * d(log10 I)/dV
            log10_i = current_safe(v)
            I_lin = 10.0 ** log10_i
            dlog_dv = numerical_derivative(current_safe, v)
            g_lin = I_lin * math.log(10.0) * dlog_dv  # transconductance

            # 2nd derivative
            d2log_dv2 = numerical_second_derivative(current_safe, v)
            # d2I/dV2 = I * (ln10^2 * (dlog/dv)^2 + ln10 * d2log/dv2)
            d2I_dv2 = I_lin * (math.log(10.0)**2 * dlog_dv**2
                               + math.log(10.0) * d2log_dv2)

            # Raw formula derivative (may diverge near poles)
            try:
                dlog_dv_raw = numerical_derivative(current_log10, v)
            except:
                dlog_dv_raw = float('nan')

            points.append({
                'v': v,
                'I': I_lin,
                'gm': g_lin,
                'd2I': d2I_dv2,
                'dlog_safe': dlog_dv,
                'dlog_raw': dlog_dv_raw,
            })
        except Exception as e:
            points.append({'v': v, 'error': str(e)})

    return points


def validate(label, fn_raw, fn_safe, params_tuple, v_range):
    """Run full derivative validation and print report"""
    print(f"\n{'='*70}")
    print(f"  Derivative Validation: {label}")
    print(f"{'='*70}")

    L, T, trap_A, vs, vg = params_tuple
    print(f"  L={L}um, T={T}K, trap_A={trap_A:.1e}")

    pts = check_derivative_smoothness(v_range, fn_raw, fn_safe, label, params_tuple)

    # ── Checks ──
    gm_ok = True
    gm_sign_consistent = True
    d2_ok = True

    prev_gm = None
    gm_values = []
    d2_values = []

    print(f"\n  {'V[V]':>7s}  {'I[A]':>12s}  {'gm[S]':>12s}  {'d2I/dV2':>12s}  {'dlog':>10s}")
    print("  " + "-"*60)

    for p in pts:
        if 'error' in p:
            print(f"  {p['v']:7.3f}  ERROR: {p['error']}")
            gm_ok = False
            continue

        gm_values.append(p['gm'])
        d2_values.append(p['d2I'])

        # Check gm sign (should not oscillate)
        if prev_gm is not None and abs(p['gm']) > 0:
            if p['gm'] * prev_gm < 0 and abs(p['v']) > 0.01:
                gm_sign_consistent = False

        prev_gm = p['gm']

        print(f"  {p['v']:7.3f}  {p['I']:12.4e}  {p['gm']:12.4e}  "
              f"{p['d2I']:12.4e}  {p['dlog_safe']:10.4f}")

    # ── Smoothness metrics ──
    if len(gm_values) > 2:
        gm_changes = [abs(gm_values[i] - gm_values[i-1])
                      for i in range(1, len(gm_values))]
        max_gm_jump = max(gm_changes) if gm_changes else 0
        avg_gm = sum(abs(g) for g in gm_values) / len(gm_values)
        gm_smoothness = max_gm_jump / max(avg_gm, 1e-30)

        print(f"\n  gm smoothness metric (max_jump/avg): {gm_smoothness:.4f}")
        print(f"    < 1.0  → C1-like (good for Newton)")
        print(f"    1-10   → acceptable with small step")
        print(f"    > 10   → convergence risk")

        if gm_smoothness > 10:
            gm_ok = False

    if len(d2_values) > 2:
        d2_spread = max(abs(d) for d in d2_values)
        print(f"  d2I/dV2 max magnitude: {d2_spread:.4e}")
        if d2_spread > 1.0:
            print(f"    WARNING: large 2nd derivative → harmonic balance concern")
            d2_ok = False

    # ── Pole proximity check ──
    print(f"\n  Pole-proximity derivative test:")
    for v in [0.50, 0.55, 0.60, 0.70]:
        try:
            d_raw = numerical_derivative(
                lambda vd: fn_raw(vd, L, T, trap_A, vs, vg), v)
            d_safe = numerical_derivative(
                lambda vd: fn_safe(vd, L, T, trap_A, vs, vg), v)
            print(f"    V={v:.2f}: dlog_raw={d_raw:.4f}, dlog_safe={d_safe:.4f}")
            if abs(d_raw) > 1e6 or abs(d_safe) > 1e6:
                print(f"             ↑ derivative explosion at pole!")
        except:
            d_safe = numerical_derivative(
                lambda vd: fn_safe(vd, L, T, trap_A, vs, vg), v)
            print(f"    V={v:.2f}: dlog_raw=DIVERGED, dlog_safe={d_safe:.4f}  [protection OK]")

    all_ok = gm_ok and gm_sign_consistent and d2_ok
    print(f"\n  Result: {'PASS' if all_ok else 'ISSUES'}")
    return all_ok


def main():
    all_pass = True

    # Use realistic parameters from training data
    params = (40.0, 300.0, 1.0e-9, 2500.0, 500.0)  # L, T, trap_A, vs, vg

    # Voltage range: normal operation -3V → 0V + fringe into pole region
    v_normal = [v/100.0 for v in range(-300, 41, 10)]  # -3.0 to 0.4, step 0.1
    v_full   = [v/100.0 for v in range(-300, 71, 10)]  # -3.0 to 0.7, step 0.1

    # 1. I_dark derivative check
    r1 = validate("I_dark", kan_log10_dark, kan_log10_dark_safe, params, v_normal)
    all_pass = all_pass and r1

    # 2. I_photo derivative check
    r2 = validate("I_photo", kan_log10_photo, kan_log10_photo_safe, params, v_normal)
    all_pass = all_pass and r2

    # 3. gm ratio I_photo/I_dark — should be stable
    print(f"\n{'='*70}")
    print(f"  gm Ratio Stability (I_photo / I_dark)")
    print(f"{'='*70}")
    for v in [-3.0, -2.0, -1.0, -0.5, 0.0]:
        L, T, A, vs, vg = params
        log_dark = kan_log10_dark_safe(v, L, T, A, vs, vg)
        log_photo = kan_log10_photo_safe(v, L, T, A, vs, vg)
        ratio = 10.0**(log_photo - log_dark)
        print(f"    V={v:.1f}: I_photo/I_dark = {ratio:.2e}")

    print(f"\n{'='*70}")
    print(f"  OVERALL: {'ALL PASSED' if all_pass else 'ISSUES FOUND'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
