#!/usr/bin/env python
"""Compare standalone AC-response Verilog-A output against Python reference.

This script supports two Cadence handoff styles:

1. Compare a Spectre PSF-ASCII result file or directory:

   python compare_ac_response_symbolic.py --psf result/ac_response_symbolic.raw

2. Compare a CSV exported from ADE/ViVA.  The CSV should contain a frequency
   column named one of ``frequency_ghz``, ``f_ghz``, ``freq_ghz``, or ``freq``;
   output columns may be named either ``y_L40_T300`` or ``V(y_L40_T300)``.

   python compare_ac_response_symbolic.py --csv ac_response_symbolic.csv

The Spectre module intentionally outputs the dB formula as a voltage-like probe.
It is a standalone implementation-consistency check, not the terminal dynamic
branch of the photodetector compact model.
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_FREQ_GHZ = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
DESIGN_POINTS = [
    ("y_L36_T290", 36.0, 290.0),
    ("y_L36_T310", 36.0, 310.0),
    ("y_L40_T300", 40.0, 300.0),
    ("y_L44_T290", 44.0, 290.0),
    ("y_L44_T310", 44.0, 310.0),
]

FREQ_COLUMNS = ("frequency_ghz", "f_ghz", "freq_ghz", "freq")
VALUE_RE = re.compile(r'^\s*"([^"]+)"\s+(.+)$')
COMPLEX_RE = re.compile(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")


def ac_response_db(frequency_ghz: float, active_layer_length: float, simulation_temperature: float) -> float:
    """Distilled AC-response formula, matching paper Eq. app_ac."""
    f_safe = max(float(frequency_ghz), 1.0e-12)
    scaled_frequency = f_safe * math.exp(
        -0.000690444884035588 * active_layer_length
        + 0.00545076806767964 * simulation_temperature
    )
    numerator = (
        -0.0117266759934525 * active_layer_length
        - 0.03488283072414 * simulation_temperature
        + 19.4018156517685
    )
    denominator = 1.0 + 0.0142779533594861 * scaled_frequency ** 1.09293379917491
    return (
        0.0127235709487393 * active_layer_length
        + 0.0351387573208426 * simulation_temperature
        - 19.3768505375964
        + numerator / denominator
    )


def reference_frame(frequencies: Iterable[float] = DEFAULT_FREQ_GHZ) -> pd.DataFrame:
    rows = []
    for freq in frequencies:
        row = {"frequency_ghz": float(freq)}
        for name, length, temperature in DESIGN_POINTS:
            row[name] = ac_response_db(float(freq), length, temperature)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_psf_ascii(path: Path) -> pd.DataFrame:
    """Parse a simple Spectre PSF-ASCII sweep file into a dataframe."""
    rows: List[Dict[str, float]] = []
    current: Dict[str, float] = {}
    in_value = False
    sweep_seen = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.rstrip("\n\r")
            if stripped.strip() == "VALUE":
                in_value = True
                current = {}
                continue
            if not in_value:
                continue
            if stripped.strip() == "END":
                break

            match = VALUE_RE.match(stripped)
            if not match:
                continue
            key = match.group(1)
            raw = match.group(2).strip()

            if key in FREQ_COLUMNS:
                if current and sweep_seen:
                    rows.append(current)
                    current = {}
                sweep_seen = True

            complex_match = COMPLEX_RE.match(raw)
            if complex_match:
                real = float(complex_match.group(1))
                imag = float(complex_match.group(2))
                current[key] = math.hypot(real, imag)
            else:
                try:
                    current[key] = float(raw)
                except ValueError:
                    current[key] = float("nan")

    if current:
        rows.append(current)
    if not rows:
        raise RuntimeError(f"No PSF-ASCII rows parsed from {path}")
    return pd.DataFrame(rows)


def load_psf(path: Path) -> pd.DataFrame:
    if path.is_file():
        return parse_psf_ascii(path)
    candidates = sorted(path.rglob("*.dc"))
    if not candidates:
        candidates = sorted(path.rglob("*.tran")) + sorted(path.rglob("*.ac"))
    if not candidates:
        raise FileNotFoundError(f"No PSF-ASCII result file found under {path}")
    return parse_psf_ascii(candidates[0])


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    lower_map = {col.lower(): col for col in frame.columns}

    for freq_col in FREQ_COLUMNS:
        if freq_col in lower_map:
            renamed[lower_map[freq_col]] = "frequency_ghz"
            break
    if "frequency_ghz" not in renamed.values():
        raise KeyError(
            "Could not find frequency column. Available columns: "
            + ", ".join(str(c) for c in frame.columns)
        )

    for name, _, _ in DESIGN_POINTS:
        aliases = [
            name,
            name.lower(),
            f"v({name})",
            f"V({name})",
            f"{name}:v",
            f"/{name}",
        ]
        found = None
        for alias in aliases:
            if alias in frame.columns:
                found = alias
                break
            if alias.lower() in lower_map:
                found = lower_map[alias.lower()]
                break
        if found is not None:
            renamed[found] = name

    normalized = frame.rename(columns=renamed)
    missing = [name for name, _, _ in DESIGN_POINTS if name not in normalized.columns]
    if missing:
        raise KeyError(
            "Missing output columns "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(str(c) for c in frame.columns)
        )
    return normalized


def compare(frame: pd.DataFrame, tol_db: float) -> Tuple[pd.DataFrame, Dict[str, object]]:
    observed = normalize_columns(frame).copy()
    observed["frequency_ghz"] = observed["frequency_ghz"].astype(float)
    observed = observed.sort_values("frequency_ghz").reset_index(drop=True)
    reference = reference_frame(observed["frequency_ghz"].values)

    rows = []
    for idx, obs_row in observed.iterrows():
        freq = float(obs_row["frequency_ghz"])
        for name, _, _ in DESIGN_POINTS:
            obs = float(obs_row[name])
            ref = float(reference.loc[idx, name])
            rows.append(
                {
                    "frequency_ghz": freq,
                    "node": name,
                    "spectre_db": obs,
                    "python_db": ref,
                    "abs_err_db": abs(obs - ref),
                }
            )

    detail = pd.DataFrame(rows)
    max_abs = float(detail["abs_err_db"].max())
    rmse = float(np.sqrt(np.mean(np.square(detail["spectre_db"] - detail["python_db"]))))
    status = "NUMERIC_PASS" if max_abs <= tol_db else "FAIL"
    summary = {
        "status": status,
        "points": int(len(detail)),
        "max_abs_err_db": max_abs,
        "rmse_db": rmse,
        "tol_db": float(tol_db),
    }
    return detail, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV exported from ADE/ViVA.")
    parser.add_argument("--psf", type=Path, help="Spectre PSF-ASCII file or raw directory.")
    parser.add_argument("--write-reference", type=Path, help="Write the Python reference CSV and exit.")
    parser.add_argument("--output-detail", type=Path, help="Optional CSV path for pointwise comparison details.")
    parser.add_argument("--tol-db", type=float, default=1.0e-6, help="Absolute dB tolerance for NUMERIC_PASS.")
    args = parser.parse_args()

    if args.write_reference:
        reference_frame().to_csv(args.write_reference, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Wrote reference CSV: {args.write_reference}")
        return 0

    if bool(args.csv) == bool(args.psf):
        parser.error("Provide exactly one of --csv or --psf, or use --write-reference.")

    frame = pd.read_csv(args.csv) if args.csv else load_psf(args.psf)
    detail, summary = compare(frame, args.tol_db)

    if args.output_detail:
        args.output_detail.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.output_detail, index=False)

    print("[Standalone AC-response Verilog-A formula check]")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0 if summary["status"] == "NUMERIC_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
