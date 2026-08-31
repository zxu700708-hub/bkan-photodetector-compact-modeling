#!/usr/bin/env python3
"""Generate the AC reference for the certified terminal-charge Verilog-A."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "bkan" / "device_modeling" / "terminal_charge"))

from reference import TerminalChargeReference  # noqa: E402
from verify_terminal_charge_spectre import (  # noqa: E402
    BIAS_VALUES,
    MODEL_JSON,
    SymbolicCurrentReference,
    TEMPERATURE_VALUES,
    VAC_MAG,
    condition_frame,
    dec_frequencies,
)

OUT_CSV = HERE / "ac_terminal_charge_reference.csv"


def main() -> int:
    dc_reference = SymbolicCurrentReference()
    terminal_reference = TerminalChargeReference(MODEL_JSON)
    bias = float(BIAS_VALUES[2])
    temperature = float(TEMPERATURE_VALUES[1])
    frame = condition_frame([bias], temperature)
    charge, dqdv = terminal_reference.evaluate(frame)
    d_idv = float(dc_reference.derivative([bias], temperature)[0])
    dc_current = float(dc_reference.evaluate([bias], temperature)[0])
    d_qdv = float(dqdv[0])

    rows = []
    for frequency_hz in dec_frequencies(1.0e6, 1.0e12, 20):
        omega = 2.0 * math.pi * frequency_hz
        admittance = complex(d_idv, omega * d_qdv)
        current = VAC_MAG * admittance
        rows.append(
            {
                "frequency_hz": frequency_hz,
                "bias_v": bias,
                "vac_mag_v": VAC_MAG,
                "dc_current_a": dc_current,
                "terminal_charge_C": float(charge[0]),
                "dIdV_S": d_idv,
                "dQdV_F": d_qdv,
                "dynamic_current_mag_a": abs(VAC_MAG * omega * d_qdv),
                "ac_current_real_a": current.real,
                "ac_current_imag_a": current.imag,
                "ac_current_mag_a": abs(current),
                "admittance_mag_s": abs(admittance),
                "phase_deg": math.degrees(math.atan2(current.imag, current.real)),
            }
        )
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} points)")
    print(f"dIdV={d_idv:.6e} S, dQdV={d_qdv:.6e} F")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
