"""Recalibrate saved capacitance BKAN checkpoints at the DEVICE-cluster level.

The training checkpoints and the original grouped split manifests are reused;
only post-training conformal calibration and test-set UQ metrics are recomputed.
Old result directories are never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))

from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler
from device_modeling.photodetector.task_config import (
    DEFAULT_CAPACITANCE_DATA,
    TASKS,
    load_capacitance_data,
    prepare_task_dataframe,
)


DEFAULT_SOURCE = ROOT / "artifacts" / "results" / "uq_repeated_grouped_capacitance"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "results"
    / "uq_repeated_grouped_capacitance_device_cluster"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--capacitance-data",
        type=Path,
        default=DEFAULT_CAPACITANCE_DATA,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--target-picp", type=float, default=95.0)
    parser.add_argument("--min-z", type=float, default=1.96)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def split_ids(manifest: pd.DataFrame, split: str) -> set[str]:
    return set(
        manifest.loc[manifest["split"].eq(split), "group_id"].astype(str)
    )


def select_groups(frame: pd.DataFrame, group_ids: set[str]) -> pd.DataFrame:
    selected = frame.loc[frame["_curve_id"].astype(str).isin(group_ids)].copy()
    selected.reset_index(drop=True, inplace=True)
    return selected


def coverage_ci(percent: float, n: int) -> tuple[float, float]:
    if not np.isfinite(percent) or n <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    successes = int(round(float(percent) * n / 100.0))
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = (
        z
        * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return 100.0 * (center - radius), 100.0 * (center + radius)


def snake_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {str(key).lower(): value for key, value in metrics.items()}


def audit_partitions(manifest: pd.DataFrame) -> dict[str, object]:
    ids = {
        split: split_ids(manifest, split)
        for split in ("train", "validation", "calibration", "test")
    }
    overlap = 0
    names = list(ids)
    for left, left_name in enumerate(names):
        for right_name in names[:left]:
            overlap += len(ids[left_name] & ids[right_name])
    return {
        "train_ids": ids["train"],
        "validation_ids": ids["validation"],
        "calibration_ids": ids["calibration"],
        "test_ids": ids["test"],
        "overlap_groups": overlap,
    }


def recompute_seed(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    source_seed = args.source / f"seed_{seed}"
    manifest_path = source_seed / "capacitance_split_manifest.csv"
    checkpoint_candidates = sorted(
        path
        for path in source_seed.rglob("model_checkpoint.pt")
        if "Capacitance-bayes-results" in str(path.relative_to(source_seed))
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if len(checkpoint_candidates) != 1:
        raise RuntimeError(
            f"Seed {seed}: expected exactly one capacitance checkpoint, "
            f"found {len(checkpoint_candidates)}"
        )
    checkpoint_path = checkpoint_candidates[0]

    manifest = pd.read_csv(manifest_path, dtype={"group_id": str})
    audit = audit_partitions(manifest)
    if audit["overlap_groups"]:
        raise RuntimeError(f"Seed {seed}: DEVICE overlap in split manifest")

    calibration = select_groups(frame, audit["calibration_ids"])
    test = select_groups(frame, audit["test_ids"])
    calibration_count = calibration["_curve_id"].nunique()
    test_count = test["_curve_id"].nunique()
    if calibration_count != len(audit["calibration_ids"]):
        raise RuntimeError(f"Seed {seed}: missing calibration DEVICE rows")
    if test_count != len(audit["test_ids"]):
        raise RuntimeError(f"Seed {seed}: missing test DEVICE rows")

    points_per_calibration_device = calibration.groupby("_curve_id").size()
    points_per_test_device = test.groupby("_curve_id").size()
    if points_per_calibration_device.nunique() != 1:
        raise RuntimeError(f"Seed {seed}: unequal calibration DEVICE sizes")
    if points_per_test_device.nunique() != 1:
        raise RuntimeError(f"Seed {seed}: unequal test DEVICE sizes")

    target_result = (
        args.output
        / f"seed_{seed}"
        / checkpoint_path.parent.relative_to(source_seed)
    )
    target_result.mkdir(parents=True, exist_ok=True)

    modeler = BayesKANDeviceModeler.load_model(
        checkpoint_path,
        device=args.device,
    )
    inference_method = str(modeler.inference_method)
    model_label = (
        "mc_dropout" if inference_method == "dropout" else "bayesian_kan_vi"
    )
    modeler.save_dir = str(target_result)
    modeler.calibrate_uncertainty(
        df_calib=calibration,
        target_picp=args.target_picp,
        min_z=args.min_z,
        calibration_unit="curve",
        group_cols=["_curve_id"],
    )
    if modeler._calib_score_count != calibration_count:
        raise RuntimeError(
            f"Seed {seed}: expected {calibration_count} DEVICE scores, "
            f"got {modeler._calib_score_count}"
        )

    metrics = snake_metrics(modeler.evaluate_on_test_set(test))
    metrics.update(
        {
            "seed": seed,
            "task": "Capacitance",
            "model": model_label,
            "inference_method": inference_method,
            "calibration_device_count": calibration_count,
            "calibration_subcurve_count": int(
                calibration.groupby(["_curve_id", "log_frequency_ghz"]).ngroups
            ),
            "calibration_point_count": int(len(calibration)),
            "test_device_count": test_count,
            "test_subcurve_count": int(
                test.groupby(["_curve_id", "log_frequency_ghz"]).ngroups
            ),
            "test_point_count": int(len(test)),
            "points_per_calibration_device": int(
                points_per_calibration_device.iloc[0]
            ),
            "points_per_test_device": int(points_per_test_device.iloc[0]),
            "split_seed": seed,
            "run_label": "device_cluster_checkpoint_recalibration",
            "source_checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "source_manifest": str(manifest_path.relative_to(ROOT)),
        }
    )
    for prefix in (
        "raw",
        "standard_conformal",
        "no_shrink_conformal",
        "calibrated",
    ):
        low, high = coverage_ci(
            float(metrics[f"{prefix}_device_coverage_95"]),
            test_count,
        )
        metrics[f"{prefix}_device_coverage_95_wilson_low"] = low
        metrics[f"{prefix}_device_coverage_95_wilson_high"] = high

    modeler.save_model(target_result / "model_checkpoint.pt")
    pd.DataFrame([metrics]).to_csv(
        args.output / f"seed_{seed}" / "summary.csv",
        index=False,
    )
    audit_row = {
        "seed": seed,
        "task": "Capacitance",
        "overlap_groups": audit["overlap_groups"],
        "train_devices": len(audit["train_ids"]),
        "validation_devices": len(audit["validation_ids"]),
        "calibration_devices": calibration_count,
        "test_devices": test_count,
        "calibration_scores": modeler._calib_score_count,
        "calibration_points": len(calibration),
        "test_points": len(test),
    }
    return metrics, audit_row


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregate = {
        "task": "Capacitance",
        "model": str(metrics["model"].iloc[0]),
        "inference_method": str(metrics["inference_method"].iloc[0]),
        "n_seeds": int(metrics["seed"].nunique()),
    }
    metric_names = [
        "rmse_model_space",
        "r2_model_space",
        "raw_picp_95",
        "standard_conformal_picp_95",
        "no_shrink_conformal_picp_95",
        "calibrated_picp_95",
        "raw_subcurve_coverage_95",
        "standard_conformal_subcurve_coverage_95",
        "no_shrink_conformal_subcurve_coverage_95",
        "calibrated_subcurve_coverage_95",
        "raw_device_coverage_95",
        "standard_conformal_device_coverage_95",
        "no_shrink_conformal_device_coverage_95",
        "calibrated_device_coverage_95",
        "raw_mpiw_log",
        "standard_conformal_mpiw_log",
        "no_shrink_conformal_mpiw_log",
        "calibrated_mpiw_log",
        "raw_gaussian_nll_model_space",
        "standard_conformal_gaussian_nll_model_space",
        "no_shrink_conformal_gaussian_nll_model_space",
        "calibrated_gaussian_nll_model_space",
        "raw_interval_score_95",
        "standard_conformal_interval_score_95",
        "no_shrink_conformal_interval_score_95",
        "calibrated_interval_score_95",
        "raw_z",
        "standard_conformal_z",
        "no_shrink_conformal_z",
        "calibration_raw_quantile",
        "calibration_z",
        "calibration_score_count",
    ]
    for name in metric_names:
        values = pd.to_numeric(metrics[name], errors="coerce").dropna()
        aggregate[f"{name}_mean"] = float(values.mean())
        aggregate[f"{name}_std"] = float(values.std(ddof=1))
        aggregate[f"{name}_median"] = float(values.median())
    return pd.DataFrame([aggregate])


def write_report(
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> None:
    row = aggregate.iloc[0]
    lines = [
        "# DEVICE-Cluster Conformal Recalibration",
        "",
        f"Generated: {datetime.now(timezone.utc).astimezone().isoformat()}",
        "",
        "- Training checkpoints were reused; no model retraining was performed.",
        "- Calibration unit: one complete DEVICE response surface.",
        "- Each calibration DEVICE contributes one maximum normalized residual.",
        f"- Seeds: {', '.join(map(str, metrics['seed'].tolist()))}",
        f"- Calibration scores per seed: {int(metrics['calibration_score_count'].iloc[0])}",
        f"- Calibration points per seed: {int(metrics['calibration_point_count'].iloc[0])}",
        f"- Test DEVICEs per seed: {int(metrics['test_device_count'].iloc[0])}",
        "",
        "## Aggregate",
        "",
        f"- Model: {row['model']} ({row['inference_method']})",
        f"- Raw point coverage: {row['raw_picp_95_mean']:.2f}% ± {row['raw_picp_95_std']:.2f}%",
        f"- No-shrink point coverage: {row['calibrated_picp_95_mean']:.2f}% ± {row['calibrated_picp_95_std']:.2f}%",
        f"- Raw DEVICE coverage: {row['raw_device_coverage_95_mean']:.2f}% ± {row['raw_device_coverage_95_std']:.2f}%",
        f"- No-shrink DEVICE coverage: {row['calibrated_device_coverage_95_mean']:.2f}% ± {row['calibrated_device_coverage_95_std']:.2f}%",
        f"- Conformal q_raw: {row['calibration_raw_quantile_mean']:.4f} ± {row['calibration_raw_quantile_std']:.4f}",
        f"- No-shrink q: {row['calibration_z_mean']:.4f} ± {row['calibration_z_std']:.4f}",
        "",
        "## Raw / Standard / No-Shrink Comparison",
        "",
        "| Variant | q | Point coverage | DEVICE coverage | Subcurve coverage | MPIW |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, prefix in (
        ("Raw Gaussian", "raw"),
        ("Standard conformal", "standard_conformal"),
        ("No-shrink conformal", "no_shrink_conformal"),
    ):
        lines.append(
            f"| {label} | {row[f'{prefix}_z_mean']:.4f} | "
            f"{row[f'{prefix}_picp_95_mean']:.2f}% | "
            f"{row[f'{prefix}_device_coverage_95_mean']:.2f}% | "
            f"{row[f'{prefix}_subcurve_coverage_95_mean']:.2f}% | "
            f"{row[f'{prefix}_mpiw_log_mean']:.6g} |"
        )
    lines.extend([
        "",
        "DEVICE coverage has 6.25 percentage-point resolution per seed because "
        "the test split contains 16 DEVICEs.",
        "",
    ])
    (args.output / "device_cluster_recalibration_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.capacitance_data = args.capacitance_data.resolve()
    if args.source == args.output:
        raise ValueError("--output must differ from --source")

    raw = load_capacitance_data(args.capacitance_data)
    frame, _ = prepare_task_dataframe(raw, TASKS["capacitance"])
    args.output.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    audit_rows = []
    for seed in args.seeds:
        metrics, audit = recompute_seed(args, frame, seed)
        metric_rows.append(metrics)
        audit_rows.append(audit)
        print(
            f"seed={seed}: q_raw={metrics['calibration_raw_quantile']:.4f}, "
            f"q={metrics['calibration_z']:.4f}, "
            f"device_cov={metrics['calibrated_device_coverage_95']:.2f}%"
        )

    metrics = pd.DataFrame(metric_rows).sort_values("seed").reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values("seed").reset_index(drop=True)
    aggregate = aggregate_metrics(metrics)
    metrics.to_csv(args.output / "uq_metrics_by_seed.csv", index=False)
    aggregate.to_csv(args.output / "uq_metrics_summary.csv", index=False)
    audit.to_csv(args.output / "split_audit.csv", index=False)
    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": str(args.source),
        "output": str(args.output),
        "capacitance_data": str(args.capacitance_data),
        "seeds": args.seeds,
        "target_picp": args.target_picp,
        "min_z": args.min_z,
        "device": args.device,
        "calibration_group_columns": ["_curve_id"],
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(args, metrics, aggregate)
    print(f"Saved DEVICE-cluster recalibration to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
