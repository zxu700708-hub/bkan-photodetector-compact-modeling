"""
KAN Compact Formula Correctness Validation
===========================================
1. Read CSV verification data (golden reference from KAN extraction)
2. Recompute formulas and compare against CSV reference values
3. Validate convergence-safe formulas match raw formulas in valid range
4. Generate Spectre testbench reference data

Usage:
  python validate_kan_formulas.py              # full validation
  python validate_kan_formulas.py --csv-only   # CSV comparison only
  python validate_kan_formulas.py --spectre    # output Spectre ref data
"""

import math
import csv
import os
import sys
from dataclasses import dataclass

# ── Project paths ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
RESULTS_BASE = os.path.join(PROJECT_ROOT, "artifacts", "results", "device_modeling")
CSV_DARK = os.path.join(
    RESULTS_BASE, "I-dark-bayes-results", "Formula_Pointwise_Verification.csv"
)

# ═══════════════════════════════════════════════════════════
# Convergence protection (matching Verilog-A exactly)
# ═══════════════════════════════════════════════════════════

def smooth_max(x, eps=0.01, delta=0.001):
    d = x - eps
    return eps + 0.5 * (d + math.sqrt(d*d + 4.0*delta*delta))

def safe_reciprocal(denom, eps=0.01, delta=0.001):
    return 1.0 / smooth_max(denom, eps, delta)

def safe_reciprocal_neg(denom, eps=0.01, delta=0.001):
    """Protect denominators whose valid operating branch is negative."""
    return -1.0 / smooth_max(-denom, eps, delta)

def safe_sqrt(arg, eps=1.0e-6):
    safe_arg = 0.5 * (arg + math.sqrt(arg*arg + 4.0*eps*eps))
    return math.sqrt(safe_arg)

def safe_exp(x, clip=80.0):
    if x <= clip:
        return math.exp(x)
    return math.exp(clip) * (1.0 + x - clip)

def clamp_voltage(v, vmin=-10.0, vmax=0.5):
    if v > vmax:
        return vmax + 0.1 * math.log(1.0 + (v - vmax) / 0.1)
    elif v < vmin:
        return vmin - 0.1 * math.log(1.0 + (vmin - v) / 0.1)
    return v


def smax0(x, delta=0.001):
    return 0.5 * (x + math.sqrt(x*x + 4.0*delta*delta))


def protect_denom_pos(d, eps=0.01, delta=0.001):
    """Mirror ge_si_pdet_fixed.va protect_denom()."""
    return eps + smax0(d - eps, delta)

# ═══════════════════════════════════════════════════════════
# KAN original formulas (raw transcription)
# ═══════════════════════════════════════════════════════════

def kan_log10_dark(vd, L, T, trap_A, vs, vg):
    """KAN original: log10 of dark current (2026-05-19 symbolic formula)."""
    return (
        0.000801006327369681 * (
            7.4631199836731
            + 3.5395200252533 * (
                (1.46747388818052 * vd - 1.58986152442267)
                * (
                    0.00144951930269599 * (0.0449672995724005 * vs - 102.903615442195) ** 2
                    + 0.0356194972991943 * (0.277608279573617 * T - 78.1184990296283) ** 2
                    + 0.346398610621691
                )
                + 3.48451709747314
            ) / (1.46747388818052 * vd - 1.58986152442267)
        ) ** 2
        - 0.000518337697397635 * (
            9.77896022796631
            - 4.68792009353638 * (
                (1.48937218593957 * vd - 0.996492927620805)
                * (
                    0.0594017107796874 * T
                    + 0.00102126121055335 * (3.02441544539414 * L - 111.968572911812) ** 2
                    + 0.00299568567425013 * (0.029174036816848 * vs - 66.821989319932) ** 2
                    - 0.00880279032031265 * math.exp(3525157866.23469 * trap_A)
                    - 16.9966354485062
                )
                + 2.73896384239197
            ) / (1.48937218593957 * vd - 0.996492927620805)
        ) ** 2
        + 0.000328799206727079 * (
            -0.294942247675786 * T
            + 7.12113580098572 * math.exp(0.945108042401125 * vd)
            + 75.201421469862
        ) ** 2
        - 0.000513332363397298 * (
            0.448329320733052 * T
            + 0.0208178273731641 * (1.96466183972586 * L - 72.6242085403211) ** 2
            - 141.285109565234
        ) ** 2
        - 0.00617401361499954 * (
            0.113607193389579 * T
            - 33.1213938418065
            - 3.06942637535681 * math.exp(-(1491751245.70734 * trap_A - 1.32033629941622) ** 2)
            + 6.56595868377855 / (4.07535579916028 * vd - 2.64640420408292)
        ) ** 2
        - 0.0019216669006353 * (
            0.139052944768876 * T
            - 46.1861669722303
            - 1.86145999557806 * math.exp(-(3927457357.39122 * trap_A - 5.35345751836133) ** 2)
            + 9.57815781084173 / (4.41182041399576 * vd - 2.35551329923972)
        ) ** 2
        - 6.41014538446538
        + 0.0423057882837075 * (
            (3.3805284117781 * vd - 2.36291428682919)
            * (0.0760638274973239 * T - 22.5115045962676)
            + 3.33391904830933
        ) / (3.3805284117781 * vd - 2.36291428682919)
        + 0.0432708170027137 * (
            (0.741725552013947 * vd - 0.693446508474703)
            * (0.0643528219131527 * T - 18.8048719680835)
            + 1.63028883934021
        ) / (0.741725552013947 * vd - 0.693446508474703)
    )


def kan_log10_dark_safe(vd, L, T, trap_A, vs, vg):
    """Convergence-safe dark current log10 (2026-05-19 symbolic formula)."""
    vc = clamp_voltage(vd)

    d1 = 1.46747388818052 * vc - 1.58986152442267
    d2 = 1.48937218593957 * vc - 0.996492927620805
    d3 = 4.07535579916028 * vc - 2.64640420408292
    d4 = 4.41182041399576 * vc - 2.35551329923972
    d5 = 3.3805284117781 * vc - 2.36291428682919
    d6 = 0.741725552013947 * vc - 0.693446508474703

    r1 = safe_reciprocal_neg(d1)
    r2 = safe_reciprocal_neg(d2)
    r3 = safe_reciprocal_neg(d3)
    r4 = safe_reciprocal_neg(d4)
    r5 = safe_reciprocal_neg(d5)
    r6 = safe_reciprocal_neg(d6)

    t1 = 0.000801006327369681 * (
        7.4631199836731
        + 3.5395200252533 * (
            d1 * (
                0.00144951930269599 * (0.0449672995724005 * vs - 102.903615442195) ** 2
                + 0.0356194972991943 * (0.277608279573617 * T - 78.1184990296283) ** 2
                + 0.346398610621691
            )
            + 3.48451709747314
        ) * r1
    ) ** 2

    t2 = -0.000518337697397635 * (
        9.77896022796631
        - 4.68792009353638 * (
            d2 * (
                0.0594017107796874 * T
                + 0.00102126121055335 * (3.02441544539414 * L - 111.968572911812) ** 2
                + 0.00299568567425013 * (0.029174036816848 * vs - 66.821989319932) ** 2
                - 0.00880279032031265 * safe_exp(3525157866.23469 * trap_A)
                - 16.9966354485062
            )
            + 2.73896384239197
        ) * r2
    ) ** 2

    t3 = 0.000328799206727079 * (
        -0.294942247675786 * T
        + 7.12113580098572 * safe_exp(0.945108042401125 * vc)
        + 75.201421469862
    ) ** 2

    t4 = -0.000513332363397298 * (
        0.448329320733052 * T
        + 0.0208178273731641 * (1.96466183972586 * L - 72.6242085403211) ** 2
        - 141.285109565234
    ) ** 2

    t5 = -0.00617401361499954 * (
        0.113607193389579 * T
        - 33.1213938418065
        - 3.06942637535681 * math.exp(-(1491751245.70734 * trap_A - 1.32033629941622) ** 2)
        + 6.56595868377855 * r3
    ) ** 2

    t6 = -0.0019216669006353 * (
        0.139052944768876 * T
        - 46.1861669722303
        - 1.86145999557806 * math.exp(-(3927457357.39122 * trap_A - 5.35345751836133) ** 2)
        + 9.57815781084173 * r4
    ) ** 2

    t7 = 0.0423057882837075 * (
        d5 * (0.0760638274973239 * T - 22.5115045962676)
        + 3.33391904830933
    ) * r5

    t8 = 0.0432708170027137 * (
        d6 * (0.0643528219131527 * T - 18.8048719680835)
        + 1.63028883934021
    ) * r6

    return t1 + t2 + t3 + t4 + t5 + t6 - 6.41014538446538 + t7 + t8


def kan_log10_dark_veriloga_fixed(vd, L, T, trap_A, vs, vg):
    """Final IC618 Verilog-A dark-current formula mirrored from ge_si_pdet_fixed.va."""
    vc = clamp_voltage(vd)

    D1 = 3.2210296055173  - 4.24396465414667 * vc
    D2 = 1.76341470289568 - 2.02411896987616 * vc
    D3 = 2.48429431415029 - 3.83703257379696 * vc
    D4 = 0.91586290271048 - 1.62388973382964 * vc
    D5 = 2.17226953775529 - 3.81699547997708 * vc
    D6 = 2.96237190253045 - 3.88500658612811 * vc
    D7 = 0.159143600514749 - 0.192511630945259 * vc

    pD1 = protect_denom_pos(D1)
    pD2 = protect_denom_pos(D2)
    pD3 = protect_denom_pos(D3)
    pD4 = protect_denom_pos(D4)
    pD5 = protect_denom_pos(D5)
    pD6 = protect_denom_pos(D6)
    pD7 = protect_denom_pos(D7)

    exp_trap  = safe_exp(3572176770.10085 * trap_A)
    exp_trap2 = safe_exp(2823820880.58729 * trap_A)
    exp_trap3 = safe_exp(3407145794.23451 * trap_A)

    t1 = 0.00373665400560995 * (52.9792253715194 - 0.18692286600282 * T) ** 2
    t2 = 0.000241534443835963 * (61.4862469165146 - 0.0267332178476484 * vs) ** 2
    t3 = 0.000820413402540749 * (
        -0.323910369871754 * T + 84.9304981432571 + 29.0988763573955 / pD1
    ) ** 2
    t4 = -0.00204617298100098 * (
        0.117817887951727 * T - 44.2413153079335 - 5.41518353357085 / pD2
    ) ** 2
    t5 = -0.00202967658315031 * (
        0.199900801883862 * T - 66.9827831075827 - 7.97712494026598 / pD3
    ) ** 2
    t6 = 0.000942678330093845 * (
        0.250187012138281 * T
        - 0.00627692894450482 * (104.289641448622 - 2.80586356404214 * L) ** 2
        - 81.6871349581236
    ) ** 2
    t7 = -0.000526231075528485 * (
        0.218429566503778 * T + 0.0428388370115219 * exp_trap
        - 69.8412235343275 - 4.5957062110171 / pD4
    ) ** 2
    t8 = -0.00092731616197464 * (
        0.477448155803932 * T - 0.0613986841527609 * exp_trap2
        - 149.008278315552 - 18.9695630402334 / pD5
    ) ** 2
    t9 = -0.000332935374323827 * (
        0.0451639631722497 * T
        + 0.0216288742266748 * (94.7285530269762 - 0.0411497869276456 * vs) ** 2
        + 0.0242065098347615 * (103.42176686322 - 2.79663526413733 * L) ** 2
        - 0.035670279642838 * exp_trap3
        - 18.8399660667752
        - 31.172217222497 / pD6
    ) ** 2

    log10_dark = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9
    log10_dark = log10_dark - 6.18138211059667 - 0.0197022777401781 / pD7
    return max(min(log10_dark, 10.0), -40.0)


def kan_log10_photo(vd, L, T, trap_A, vs, vg):
    """KAN original: log10 of photo current (transformed space)"""
    return (
        0.000676450972350242 * (
            -4.83323942444576 + 3.73404095279437 / (1.47656620656122 - 0.925648684508075 * vd)
        ) ** 2
        + 0.00736217338561201 * math.sqrt(max(0.103720927016698 - 3.45420985978465 * vd, 0.0))
        + 0.00314095423344744 * math.sqrt(max(0.256218565766632 - 5.9854380265094 * vd, 0.0))
        - 4.45183967043515e-5 * (-4.75168591896887 * vd - 15.0286317875679) ** 2
        + 1.66692853978889e-5 * (
            -2.63566253461131 * vd - 14.8021370333584
            + 16.1937844317283 * math.exp(-(4.23898495035396 - 2677042402.04251 * trap_A) ** 2)
        ) ** 2
        - 0.000107826115766932 * (
            3.79548075198568 * math.sqrt(max(0.176998821562885 - 4.15664832159642 * vd, 0.0))
            - 22.9442807143863
            - 2.70598500918067 * math.exp(-(13.9485781468885 - 0.00495665792266289 * vs) ** 2)
            + 4.69331198587702 * math.exp(-(2.52921586298211 - 2805231472.10032 * trap_A) ** 2)
        ) ** 2
        - 3.3283066910418
        - 0.00731371825772092 * math.exp(-(28.6638489333504 - 0.0905533335097269 * T) ** 2)
        - 0.00305793260316419 * math.exp(-(13.918100792178 - 0.025011730298817 * vg) ** 2)
        + 0.00344642308129117 * math.exp(-(8.00465237521269 - 0.177936892991658 * L) ** 2)
        + 0.00662813487188099 * math.exp(-(7.95233182927684 - 0.176675763902635 * L) ** 2)
        + 0.00674898769181466 * math.exp(-(7.74444603893553 - 5761479138.43643 * trap_A) ** 2)
        + 0.00965677853131678 * math.exp(-(7.0194203459318 - 0.141812561977296 * L) ** 2)
    )


def kan_log10_photo_safe(vd, L, T, trap_A, vs, vg):
    """Convergence-safe photo current log10"""
    vc = clamp_voltage(vd)

    return (
        0.000676450972350242 * (
            -4.83323942444576
            + 3.73404095279437 * safe_reciprocal(1.47656620656122 - 0.925648684508075 * vc)
        ) ** 2
        + 0.00736217338561201 * safe_sqrt(0.103720927016698 - 3.45420985978465 * vc)
        + 0.00314095423344744 * safe_sqrt(0.256218565766632 - 5.9854380265094 * vc)
        - 4.45183967043515e-5 * (-4.75168591896887 * vc - 15.0286317875679) ** 2
        + 1.66692853978889e-5 * (
            -2.63566253461131 * vc - 14.8021370333584
            + 16.1937844317283 * math.exp(-(4.23898495035396 - 2677042402.04251 * trap_A) ** 2)
        ) ** 2
        - 0.000107826115766932 * (
            3.79548075198568 * safe_sqrt(0.176998821562885 - 4.15664832159642 * vc)
            - 22.9442807143863
            - 2.70598500918067 * math.exp(-(13.9485781468885 - 0.00495665792266289 * vs) ** 2)
            + 4.69331198587702 * math.exp(-(2.52921586298211 - 2805231472.10032 * trap_A) ** 2)
        ) ** 2
        - 3.3283066910418
        - 0.00731371825772092 * math.exp(-(28.6638489333504 - 0.0905533335097269 * T) ** 2)
        - 0.00305793260316419 * math.exp(-(13.918100792178 - 0.025011730298817 * vg) ** 2)
        + 0.00344642308129117 * math.exp(-(8.00465237521269 - 0.177936892991658 * L) ** 2)
        + 0.00662813487188099 * math.exp(-(7.95233182927684 - 0.176675763902635 * L) ** 2)
        + 0.00674898769181466 * math.exp(-(7.74444603893553 - 5761479138.43643 * trap_A) ** 2)
        + 0.00965677853131678 * math.exp(-(7.0194203459318 - 0.141812561977296 * L) ** 2)
    )


def kan_log10_photo_veriloga_fixed(vd, L, T, trap_A, vs, vg):
    """Final IC618 Verilog-A photo-current formula mirrored from ge_si_pdet_fixed.va."""
    vc = clamp_voltage(vd)

    D8 = 1.47656620656122 - 0.925648684508075 * vc
    S1 = 0.103720927016698 - 3.45420985978465 * vc
    S2 = 0.256218565766632 - 5.9854380265094 * vc
    S3 = 0.176998821562885 - 4.15664832159642 * vc

    pD8 = protect_denom_pos(D8)
    sS1 = math.sqrt(smax0(S1, 0.001))
    sS2 = math.sqrt(smax0(S2, 0.001))
    sS3 = math.sqrt(smax0(S3, 0.001))

    log10_photo = (
        0.000676450972350242 * (
            -4.83323942444576 + 3.73404095279437 / pD8
        ) ** 2
        + 0.00736217338561201 * sS1
        + 0.00314095423344744 * sS2
        - 4.45183967043515e-5 * (-4.75168591896887 * vc - 15.0286317875679) ** 2
        + 1.66692853978889e-5 * (
            -2.63566253461131 * vc - 14.8021370333584
            + 16.1937844317283 * math.exp(-(4.23898495035396 - 2677042402.04251 * trap_A) ** 2)
        ) ** 2
        - 0.000107826115766932 * (
            3.79548075198568 * sS3
            - 22.9442807143863
            - 2.70598500918067 * math.exp(-(13.9485781468885 - 0.00495665792266289 * vs) ** 2)
            + 4.69331198587702 * math.exp(-(2.52921586298211 - 2805231472.10032 * trap_A) ** 2)
        ) ** 2
        - 3.3283066910418
        - 0.00731371825772092 * math.exp(-(28.6638489333504 - 0.0905533335097269 * T) ** 2)
        - 0.00305793260316419 * math.exp(-(13.918100792178 - 0.025011730298817 * vg) ** 2)
        + 0.00344642308129117 * math.exp(-(8.00465237521269 - 0.177936892991658 * L) ** 2)
        + 0.00662813487188099 * math.exp(-(7.95233182927684 - 0.176675763902635 * L) ** 2)
        + 0.00674898769181466 * math.exp(-(7.74444603893553 - 5761479138.43643 * trap_A) ** 2)
        + 0.00965677853131678 * math.exp(-(7.0194203459318 - 0.141812561977296 * L) ** 2)
    )
    return max(min(log10_photo, 10.0), -40.0)


def kan_ac_response_db(freq_ghz, L, T):
    """KAN AC compact: normalized small-signal magnitude response in dB."""
    A = 0.043286058841316 * L + 0.0376448196142801 * T - 21.4053826740305
    B = -0.0425090134894624 * L - 0.0390669691997125 * T + 21.9384518054452
    fc = math.exp(0.00535145740474049 * L + 0.00419202668726104 * T) / 0.0172206207304869
    p  = 1.07549720488297
    f_norm = freq_ghz / max(fc, 0.001)
    return A + B / (1.0 + f_norm ** p)

# ═══════════════════════════════════════════════════════════
# CSV golden-reference validation
# ═══════════════════════════════════════════════════════════

def validate_from_csv(csv_path):
    """Check the retained legacy raw transcription against its CSV reference.

    This is an archival drift detector.  The maintained Verilog-A mirror is
    validated separately by ``validate_veriloga_fixed_from_csv``.
    """
    print("=" * 80)
    print("Legacy Raw-Transcription Drift Check: I_dark (advisory)")
    print("=" * 80)

    if not os.path.exists(csv_path):
        print(f"  ERROR: CSV not found at {csv_path}")
        return False

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    errors = []
    for i, row in enumerate(rows):
        vd    = float(row["dark_voltage"])
        trap_A= float(row["trap_assisted_recomb_A"])
        vs    = float(row["ge_sio2_recomb_velocity"])
        vg    = float(row["ge_si_recomb_velocity"])
        L     = float(row["active_layer_length"])
        T     = float(row["simulation_temperature"])
        csv_target = float(row["formula_target_space"])  # log10 space
        csv_dark   = float(row["dark_current"])

        # Our recomputation
        our_log10  = kan_log10_dark(vd, L, T, trap_A, vs, vg)
        our_safe   = kan_log10_dark_safe(vd, L, T, trap_A, vs, vg)
        our_dark   = 10.0 ** our_log10

        err_log10  = our_log10 - csv_target
        err_dark   = our_dark - csv_dark

        errors.append({
            "row": i+1, "vd": vd,
            "csv_target": csv_target, "our_log10": our_log10,
            "err_log10": err_log10, "err_dark": err_dark,
            "raw_vs_safe": abs(our_log10 - our_safe),
        })

    # Statistics
    abs_err = [abs(e["err_log10"]) for e in errors]
    max_err = max(abs_err)
    mean_err = sum(abs_err) / len(abs_err)
    rms_err = math.sqrt(sum(e**2 for e in abs_err) / len(abs_err))

    # Formula raw vs safe consistency
    raw_vs_safe_errs = [e["raw_vs_safe"] for e in errors]
    max_rvs = max(raw_vs_safe_errs)

    print(f"  Points tested: {len(errors)}")
    print(f"  Log10 space:")
    print(f"    Max  |error| : {max_err:.6f}")
    print(f"    Mean |error| : {mean_err:.6f}")
    print(f"    RMS  error   : {rms_err:.6f}")
    print(f"  Raw vs Safe divergence: max={max_rvs:.2e}")
    print()

    # Show first 5 points
    print(f"  {'Row':>4s} {'Vd':>7s}  {'CSV log10':>10s}  {'Our log10':>10s}  {'Error':>10s}  {'RvsSafe':>10s}")
    print("  " + "-" * 62)
    for e in errors[:5]:
        print(f"  {e['row']:4d} {e['vd']:7.3f}  {e['csv_target']:10.6f}  {e['our_log10']:10.6f}  "
              f"{e['err_log10']:10.6f}  {e['raw_vs_safe']:10.2e}")

    threshold = 0.02  # 2% in log10 space is very tight
    pass_count = sum(1 for e in abs_err if e < threshold)
    all_ok = pass_count == len(errors)

    print(f"\n  Points within {threshold} tolerance: {pass_count}/{len(errors)}")
    print(f"  Result: {'PASS' if all_ok else 'FAIL — formula transcription differs from CSV'}")

    return all_ok


def validate_veriloga_fixed_from_csv(csv_path):
    """Compare final ge_si_pdet_fixed.va mirror against CSV symbolic/TCAD data."""
    print("\n" + "=" * 80)
    print("Final Verilog-A Formula Validation: I_dark")
    print("=" * 80)

    if not os.path.exists(csv_path):
        print(f"  ERROR: CSV not found at {csv_path}")
        return False

    with open(csv_path, 'r') as f:
        rows = list(csv.DictReader(f))

    formula_diffs = []
    tcad_diffs = []
    for row in rows:
        vd     = float(row["dark_voltage"])
        trap_A = float(row["trap_assisted_recomb_A"])
        vs     = float(row["ge_sio2_recomb_velocity"])
        vg     = float(row["ge_si_recomb_velocity"])
        L      = float(row["active_layer_length"])
        T      = float(row["simulation_temperature"])

        va_log10 = kan_log10_dark_veriloga_fixed(vd, L, T, trap_A, vs, vg)
        formula_diffs.append(va_log10 - float(row["formula_target_space"]))
        tcad_diffs.append(va_log10 - float(row["tcad_target_space"]))

    def stats(vals):
        abs_vals = [abs(v) for v in vals]
        rms = math.sqrt(sum(v*v for v in vals) / len(vals))
        mae = sum(abs_vals) / len(abs_vals)
        max_err = max(abs_vals)
        linear_pct = 100.0 * (10.0 ** rms - 1.0)
        return rms, mae, max_err, linear_pct

    f_rms, f_mae, f_max, f_pct = stats(formula_diffs)
    t_rms, t_mae, t_max, t_pct = stats(tcad_diffs)

    print(f"  Points tested: {len(rows)}")
    print("  Against extracted symbolic formula:")
    print(f"    RMS={f_rms:.6f} log10  (~{f_pct:.2f}% linear), MAE={f_mae:.6f}, Max={f_max:.6f}")
    print("  Against TCAD target:")
    print(f"    RMS={t_rms:.6f} log10  (~{t_pct:.2f}% linear), MAE={t_mae:.6f}, Max={t_max:.6f}")

    all_ok = t_rms < 0.02 and t_max < 0.05
    print(f"  Result: {'PASS' if all_ok else 'REVIEW'}")
    return all_ok


# ═══════════════════════════════════════════════════════════
# Self-consistency: raw vs safe formula
# ═══════════════════════════════════════════════════════════

def validate_raw_vs_safe():
    """Verify safe formula matches raw formula in valid operating range"""
    print("\n" + "=" * 80)
    print("Raw vs Safe Formula Consistency (I_dark)")
    print("=" * 80)

    # Realistic parameter ranges (from CSV data)
    test_points = [
        # (vd, L, T, trap_A, vs, vg)
        (-3.0, 41.5, 301.5, 1.35e-9, 2570.9, 484.7),
        (-2.0, 40.8, 307.1, 9.72e-10, 2440.7, 516.5),
        (-1.0, 39.3, 298.9, 1.03e-9, 2484.0, 496.5),
        (-0.5, 44.4, 292.9, 1.00e-9, 2641.1, 471.0),
        ( 0.0, 40.7, 306.0, 7.93e-10, 2427.3, 501.0),
    ]

    all_ok = True
    print(f"  {'Vd':>7s}  {'Raw log10':>12s}  {'Safe log10':>12s}  {'Diff':>12s}  Status")
    print("  " + "-" * 58)

    for vd, L, T, trap_A, vs, vg in test_points:
        raw  = kan_log10_dark(vd, L, T, trap_A, vs, vg)
        safe = kan_log10_dark_safe(vd, L, T, trap_A, vs, vg)
        diff = abs(raw - safe)
        ok = diff < 1e-4  # 0.01% in log10 space is negligible
        if not ok:
            all_ok = False
        print(f"  {vd:7.3f}  {raw:12.6f}  {safe:12.6f}  {diff:12.2e}  {'OK' if ok else 'FAIL'}")

    # Also test the pole-proximity area where protection should activate
    print(f"\n  Pole-proximity test (V=0.5~0.8, where denominators → 0):")
    L, T, trap_A, vs, vg = 40.0, 300.0, 1.0e-9, 2500.0, 500.0
    for vd in [0.5, 0.55, 0.6, 0.7, 0.8]:
        try:
            raw = kan_log10_dark(vd, L, T, trap_A, vs, vg)
            safe = kan_log10_dark_safe(vd, L, T, trap_A, vs, vg)
            print(f"    V={vd:.2f}: raw={raw:.4f}, safe={safe:.4f}")
        except Exception as ex:
            safe = kan_log10_dark_safe(vd, L, T, trap_A, vs, vg)
            print(f"    V={vd:.2f}: raw DIVERGES, safe={safe:.4f}  [protection active]")

    print(f"\n  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def validate_ac():
    """AC response self-consistency"""
    print("\n" + "=" * 80)
    print("AC Response Validation")
    print("=" * 80)

    L, T = 40.0, 300.0
    fc = math.exp(0.00535145740474049 * L + 0.00419202668726104 * T) / 0.0172206207304869
    A = 0.043286058841316 * L + 0.0376448196142801 * T - 21.4053826740305
    B = -0.0425090134894624 * L - 0.0390669691997125 * T + 21.9384518054452
    y0, y_floor, p = A + B, A, 1.07549720488297

    print(f"  L={L}um, T={T}K, fc={fc:.3f} GHz")
    response_span = y0 - y_floor
    f_3db = fc * (3.0 / (response_span - 3.0)) ** (1.0 / p)
    print(f"  Low-frequency response={y0:.4f} dB")
    print(f"  High-frequency floor={y_floor:.4f} dB")
    print(f"  Derived -3 dB bandwidth={f_3db:.4f} GHz")

    all_ok = True
    # Test: at f=fc, gain should be (y0+y_floor)/2
    gain_fc = kan_ac_response_db(fc, L, T)
    mid = (y0 + y_floor) / 2
    err = abs(gain_fc - mid)
    print(f"  At f=fc: gain={gain_fc:.6f}, expected_mid={mid:.6f}, err={err:.2e}  "
          f"{'OK' if err < 1e-6 else 'FAIL'}")

    if err >= 1e-6:
        all_ok = False

    # Test: monotonic low-pass
    gains = []
    for f in [0.001, 0.1, 1.0, 10, 100, 500, 1000]:
        g = kan_ac_response_db(f, L, T)
        gains.append((f, g, 10 ** ((g - y0) / 20.0)))

    print(f"  {'Freq[GHz]':>10s}  {'Response[dB]':>12s}  {'|S21|':>12s}")
    for f, g, magnitude in gains:
        print(f"  {f:10.4f}  {g:12.4f}  {magnitude:12.4e}")

    # Check monotonic: each gain <= previous
    for i in range(1, len(gains)):
        if gains[i][1] > gains[i-1][1] + 1e-10:
            print(f"  WARN: non-monotonic at f={gains[i][0]}")
            all_ok = False

    print(f"\n  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def print_spectre_reference():
    """Output Spectre-compatible reference data"""
    print("\n" + "=" * 80)
    print("Spectre Reference Data - final ge_si_pdet_fixed.va mirror")
    print("=" * 80)

    # Use realistic parameters from CSV range
    L, T, trap_A, vs, vg = 40.0, 300.0, 1.0e-9, 2500.0, 500.0

    print(f"# Parameters: L={L}um, T={T}K, trap_A={trap_A:.1e}")
    print(f"#              vs={vs:.1f}, vg={vg:.1f}")
    print(f"#")
    print(f"# Vbias    I_dark[A]        I_photo[A]       I_total[A]")
    for v in [v/10.0 for v in range(-50, 6, 2)]:
        idark_log10 = kan_log10_dark_veriloga_fixed(v, L, T, trap_A, vs, vg)
        iphoto_log10 = kan_log10_photo_veriloga_fixed(v, L, T, trap_A, vs, vg)
        idark  = 10.0 ** idark_log10
        iphoto = 10.0 ** iphoto_log10
        itotal = idark + iphoto
        print(f"  {v:7.3f}  {idark:15.6e}  {iphoto:15.6e}  {itotal:15.6e}")

    # Frequency response
    print(f"\n# Frequency response (AC sweep)")
    print(f"# Freq[GHz]  response[dB]  normalized_|S21|")
    for f in [0.001, 0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000]:
        response_db = kan_ac_response_db(f, L, T)
        magnitude = 10.0 ** ((response_db - kan_ac_response_db(0.001, L, T)) / 20.0)
        print(f"  {f:10.4f}  {response_db:12.4f}  {magnitude:16.4e}")

    # Parameter validity range (from CSV data)
    print(f"\n# Parameter validity range (outside this, extrapolation may be poor):")
    print(f"#   trap_A   : [7.9e-10, 1.4e-09]")
    print(f"#   vs       : [2237, 2641]")
    print(f"#   vg       : [470, 532]")
    print(f"#   L        : [37, 45] um")
    print(f"#   T        : [290, 309] K")
    print(f"#   Vbias    : [-3.0, -0.2] V")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    maintained_results = {}

    # Archival check: reported, but not part of maintained-model pass/fail.
    legacy_csv_ok = validate_from_csv(CSV_DARK)

    # Maintained Verilog-A formula quality
    maintained_results["VerilogA_fixed"] = validate_veriloga_fixed_from_csv(CSV_DARK)

    maintained_results["Raw_vs_Safe"] = validate_raw_vs_safe()

    maintained_results["AC"] = validate_ac()

    # 4. Spectre reference
    print_spectre_reference()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Legacy_CSV':16s}: {'MATCH' if legacy_csv_ok else 'DRIFT (advisory)'}")
    for name, ok in maintained_results.items():
        print(f"  {name:16s}: {'PASS' if ok else 'FAIL'}")
    all_pass = all(maintained_results.values())
    print(
        f"\n  Maintained checks: {'ALL PASSED' if all_pass else 'ISSUES FOUND'}"
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
