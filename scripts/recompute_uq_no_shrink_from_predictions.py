"""Refresh repeated-UQ conformal metrics from saved prediction files.

This is a maintenance helper for completed grouped-split UQ runs.  It applies
the current no-shrink conformal rule to existing test predictions without
retraining models, then lets ``run_repeated_grouped_uq.py --summary-only``
rebuild the aggregate CSVs and report.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "uq_repeated_grouped"
TASK_DIRS = {
    "I_dark": "I-dark-bayes-results",
    "I_photo": "I-photo-bayes-results",
    "AC_Response": "AC-response-bayes-results",
}
GROUP_COLS = {
    "I_dark": [
        "trap_assisted_recomb_A",
        "ge_sio2_recomb_velocity",
        "ge_si_recomb_velocity",
        "active_layer_length",
        "simulation_temperature",
    ],
    "I_photo": [
        "trap_assisted_recomb_A",
        "ge_sio2_recomb_velocity",
        "ge_si_recomb_velocity",
        "active_layer_length",
        "simulation_temperature",
    ],
    "AC_Response": ["active_layer_length", "simulation_temperature"],
}
SUMMARY_TO_PREDICTION = {
    "picp_95": "calibrated_picp_95",
    "mpiw_log": "calibrated_mpiw_log",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--floor-z", type=float, default=1.96)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_DIRS),
        default=sorted(TASK_DIRS),
    )
    parser.add_argument(
        "--skip-summary-only",
        action="store_true",
        help="Only refresh per-seed files; do not rebuild aggregate outputs.",
    )
    return parser.parse_args()


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    alpha = 0.05
    width = upper - lower
    below = np.maximum(lower - y, 0.0)
    above = np.maximum(y - upper, 0.0)
    return float(np.mean(width + 2.0 * (below + above) / alpha))


def gaussian_nll(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    std = np.maximum(std, 1.0e-12)
    residual = y - mean
    return float(
        np.mean(0.5 * np.log(2.0 * np.pi * std**2) + 0.5 * residual**2 / std**2)
    )


def refresh_prediction_file(
    path: Path,
    task: str,
    calibration_z: float,
) -> dict[str, float]:
    frame = pd.read_csv(path)
    y = frame["actual_model_space"].to_numpy(dtype=np.float64)
    mean = frame["prediction_mean_model_space"].to_numpy(dtype=np.float64)
    raw_std = np.maximum(
        frame["prediction_std_model_space"].to_numpy(dtype=np.float64),
        1.0e-12,
    )
    lower = mean - calibration_z * raw_std
    upper = mean + calibration_z * raw_std
    covered = (y >= lower) & (y <= upper)

    frame["calibrated_lower_model_space"] = lower
    frame["calibrated_upper_model_space"] = upper
    frame["calibrated_point_covered"] = covered

    group_cols = [col for col in GROUP_COLS[task] if col in frame.columns]
    curve_coverage = np.nan
    if group_cols:
        coverage_frame = frame[group_cols].copy()
        coverage_frame["_calibrated_covered"] = covered
        grouped = coverage_frame.groupby(group_cols, dropna=False).agg(
            calibrated_covered=("_calibrated_covered", "all")
        )
        curve_coverage = float(grouped["calibrated_covered"].mean() * 100.0)
        frame["calibrated_curve_covered"] = (
            coverage_frame.groupby(group_cols, dropna=False)["_calibrated_covered"]
            .transform("all")
            .to_numpy(dtype=bool)
        )

    calibrated_std = raw_std * calibration_z / 1.96
    metrics = {
        "picp_95": float(covered.mean() * 100.0),
        "calibrated_picp_95": float(covered.mean() * 100.0),
        "mpiw_log": float(np.mean(upper - lower)),
        "calibrated_mpiw_log": float(np.mean(upper - lower)),
        "mpiw_model_space": float(np.mean(upper - lower)),
        "calibrated_gaussian_nll_model_space": gaussian_nll(
            y,
            mean,
            calibrated_std,
        ),
        "calibrated_interval_score_95": interval_score(y, lower, upper),
        "calibrated_curve_coverage_95": curve_coverage,
        "calibration_z": calibration_z,
        "calibration_std_scale_vs_normal": calibration_z / 1.96,
    }
    frame.to_csv(path, index=False)
    return metrics


def refresh_seed(seed_dir: Path, tasks: set[str], floor_z: float) -> int:
    summary_path = seed_dir / "summary.csv"
    if not summary_path.is_file():
        return 0
    summary = pd.read_csv(summary_path)
    changed = 0
    for idx, row in summary.iterrows():
        task = str(row["task"])
        if task not in tasks:
            continue
        task_dir = seed_dir / TASK_DIRS[task]
        prediction_path = task_dir / "uq_test_predictions.csv"
        if not prediction_path.is_file():
            continue
        raw_quantile = pd.to_numeric(
            pd.Series([row.get("calibration_raw_quantile")]),
            errors="coerce",
        ).iloc[0]
        old_z = pd.to_numeric(
            pd.Series([row.get("calibration_z")]),
            errors="coerce",
        ).iloc[0]
        candidates = [
            float(value)
            for value in (raw_quantile, old_z, floor_z)
            if np.isfinite(value)
        ]
        calibration_z = max(candidates) if candidates else floor_z
        metrics = refresh_prediction_file(prediction_path, task, calibration_z)
        for name, value in metrics.items():
            if name in summary.columns and (np.isfinite(value) or math.isnan(value)):
                summary.loc[idx, name] = value
        task_dir.joinpath("calibration_z.txt").write_text(
            f"{calibration_z:.12g}\n",
            encoding="utf-8",
        )
        changed += 1
    if changed:
        summary.to_csv(summary_path, index=False)
    return changed


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    tasks = set(args.tasks)
    changed = 0
    for seed_dir in sorted(output.glob("seed_*")):
        changed += refresh_seed(seed_dir, tasks, args.floor_z)
    print(f"Refreshed {changed} seed-task prediction summaries under {output}")

    if not args.skip_summary_only:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_repeated_grouped_uq.py"),
                "--summary-only",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
