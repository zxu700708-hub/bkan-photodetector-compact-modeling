#!/usr/bin/env python
from __future__ import print_function

import glob
import math
import os
import re
import sys


LINE_RE = re.compile(r'^"([^"]+)"\s+([-+0-9.eE]+)')


def load_reference(path):
    rows = []
    if not path or not os.path.exists(path):
        raise RuntimeError("reference file not found: %s" % path)

    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                v = float(parts[0])
                dark = float(parts[1])
                total = float(parts[3])
            except ValueError:
                continue
            rows.append((v, dark, total))

    if not rows:
        raise RuntimeError("no numeric rows found in reference: %s" % path)
    return rows


def find_dc_file(raw_root, name):
    candidates = [
        os.path.join(raw_root, name),
        os.path.join(raw_root, "testbench_dc_ic618.raw", name),
    ]
    for raw_dir in glob.glob(os.path.join(raw_root, "*.raw")):
        candidates.append(os.path.join(raw_dir, name))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def parse_psfascii_dc(path, sweep_name, branch_name):
    values = {}
    in_value = False
    current_v = None

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "VALUE":
                in_value = True
                continue
            if not in_value:
                continue
            if line == "END":
                break

            match = LINE_RE.match(line)
            if not match:
                continue

            key = match.group(1)
            val = float(match.group(2))
            if key == sweep_name:
                current_v = val
            elif key == branch_name and current_v is not None:
                values["%.3f" % current_v] = abs(val)

    return values


def load_simulation(raw_root):
    if not os.path.isdir(raw_root):
        raise RuntimeError("raw output directory not found: %s" % raw_root)

    dark_file = find_dc_file(raw_root, "dc_dark.dc")
    total_file = find_dc_file(raw_root, "dc_total.dc")

    if not dark_file:
        raise RuntimeError("dc_dark.dc not found under: %s" % raw_root)
    if not total_file:
        raise RuntimeError("dc_total.dc not found under: %s" % raw_root)

    return {
        "dark_file": dark_file,
        "total_file": total_file,
        "dark": parse_psfascii_dc(dark_file, "Vb_dark", "Vdark:p"),
        "total": parse_psfascii_dc(total_file, "Vb_total", "Vtotal:p"),
    }


def nearest_value(values, voltage):
    key = "%.3f" % voltage
    if key in values:
        return values[key]

    best_key = None
    best_dist = None
    for item in values:
        try:
            dist = abs(float(item) - voltage)
        except ValueError:
            continue
        if best_dist is None or dist < best_dist:
            best_key = item
            best_dist = dist

    if best_key is not None and best_dist is not None and best_dist <= 0.026:
        return values[best_key]
    return None


def compare_one(label, rows, column_index, sim_values, rel_tol_percent):
    print("")
    print("[%s]" % label)
    print("  %-9s %14s %14s %12s %8s" % ("V[V]", "Ref[A]", "Sim[A]", "RelErr[%]", "Status"))
    print("  " + "-" * 67)

    matched = 0
    failed = 0
    max_rel = 0.0
    max_abs = 0.0
    sum_sq = 0.0

    for row in rows:
        voltage = row[0]
        ref = row[column_index]
        sim = nearest_value(sim_values, voltage)

        if sim is None:
            print("  %9.3f %14.6e %14s %12s %8s" % (voltage, ref, "N/A", "--", "MISS"))
            failed += 1
            continue

        abs_err = abs(sim - ref)
        if abs(ref) > 1.0e-30:
            rel = 100.0 * abs_err / abs(ref)
        else:
            rel = 0.0 if abs_err < 1.0e-30 else 100.0

        status = "OK" if rel <= rel_tol_percent else "FAIL"
        if status != "OK":
            failed += 1

        matched += 1
        max_rel = max(max_rel, rel)
        max_abs = max(max_abs, abs_err)
        sum_sq += abs_err * abs_err

        print("  %9.3f %14.6e %14.6e %12.6g %8s" % (voltage, ref, sim, rel, status))

    rmse = math.sqrt(sum_sq / matched) if matched else float("nan")
    print("")
    print("  matched: %d / %d" % (matched, len(rows)))
    print("  max_abs_error: %.6e A" % max_abs)
    print("  rmse: %.6e A" % rmse)
    print("  max_rel_error: %.6g %%" % max_rel)

    return failed == 0 and matched == len(rows)


def usage():
    print("Usage:")
    print("  python compare_results.py dc_result_ascii spectre_reference.txt")
    print("")
    print("Expected files:")
    print("  dc_result_ascii/testbench_dc_ic618.raw/dc_dark.dc")
    print("  dc_result_ascii/testbench_dc_ic618.raw/dc_total.dc")


def main(argv):
    if len(argv) < 3:
        usage()
        return 2

    raw_root = argv[1]
    ref_path = argv[2]
    rel_tol_percent = 0.5

    print("[INFO] raw root: %s" % raw_root)
    print("[INFO] reference: %s" % ref_path)

    rows = load_reference(ref_path)
    sim = load_simulation(raw_root)

    print("[INFO] dark data: %s (%d points)" % (sim["dark_file"], len(sim["dark"])))
    print("[INFO] total data: %s (%d points)" % (sim["total_file"], len(sim["total"])))
    print("[INFO] relative tolerance: %.3g %%" % rel_tol_percent)

    dark_ok = compare_one("Dark current: Vdark:p vs reference I_dark", rows, 1, sim["dark"], rel_tol_percent)
    total_ok = compare_one("Total current: Vtotal:p vs reference I_total", rows, 2, sim["total"], rel_tol_percent)

    print("")
    if dark_ok and total_ok:
        print("Overall: PASS")
        return 0

    print("Overall: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
