"""Run and summarize repeated independent-group Bayesian KAN UQ experiments.

Each seed receives an isolated output directory.  Validation is used only for
model selection, calibration only for conformal scaling, and test groups only
for final metrics.  The default 65/10/15/10 split gives 24 independent
calibration groups for the current 160-group dataset.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_RESEARCH = (
    ROOT
    / "bkan"
    / "device_modeling"
    / "photodetector"
    / "run_research.py"
)
DEFAULT_DATA = ROOT / "artifacts" / "results" / "device_modeling" / "cleaned_data.csv"
DEFAULT_CAPACITANCE_DATA = (
    ROOT
    / "artifacts"
    / "results"
    / "apparent_capacitance_correction"
    / "primary_ge_si_apparent_capacitance.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "uq_repeated_grouped"

CORE_METRICS = (
    "rmse_model_space",
    "mae_model_space",
    "r2_model_space",
    "raw_picp_95",
    "standard_conformal_picp_95",
    "no_shrink_conformal_picp_95",
    "calibrated_picp_95",
    "raw_curve_coverage_95",
    "standard_conformal_curve_coverage_95",
    "no_shrink_conformal_curve_coverage_95",
    "calibrated_curve_coverage_95",
    "raw_group_coverage_95",
    "standard_conformal_group_coverage_95",
    "no_shrink_conformal_group_coverage_95",
    "calibrated_group_coverage_95",
    "raw_device_coverage_95",
    "standard_conformal_device_coverage_95",
    "no_shrink_conformal_device_coverage_95",
    "calibrated_device_coverage_95",
    "raw_subcurve_coverage_95",
    "standard_conformal_subcurve_coverage_95",
    "no_shrink_conformal_subcurve_coverage_95",
    "calibrated_subcurve_coverage_95",
    "raw_mpiw_log",
    "standard_conformal_mpiw_log",
    "no_shrink_conformal_mpiw_log",
    "calibrated_mpiw_log",
    "mpiw_model_space",
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
    "calibration_z",
    "calibration_score_count",
    "ood_ratio",
    "derivative_mae_model_space",
    "derivative_rmse_model_space",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--task",
        nargs="+",
        choices=["dark_current", "photo_current", "ac_response", "capacitance"],
        default=["dark_current", "photo_current", "ac_response"],
    )
    parser.add_argument(
        "--capacitance-data",
        type=Path,
        default=DEFAULT_CAPACITANCE_DATA,
        help="Vac-corrected total-admittance apparent-capacitance CSV.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(42, 52)),
    )
    parser.add_argument(
        "--split-fractions",
        nargs=4,
        type=float,
        default=[0.65, 0.10, 0.15, 0.10],
        metavar=("TRAIN", "VALIDATION", "CALIBRATION", "TEST"),
    )
    parser.add_argument(
        "--min-calibration-groups",
        "--min-calibration-curves",
        dest="min_calibration_groups",
        type=int,
        default=19,
        help=(
            "Minimum independent calibration groups per seed. The legacy "
            "--min-calibration-curves spelling is retained as an alias."
        ),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument(
        "--bayes-method",
        choices=["vi", "dropout", "hmc"],
        default="vi",
        help="Bayesian inference workflow passed to run_research.py.",
    )
    parser.add_argument("--dropout-rate", type=float, default=0.05)
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--train-mc-samples", type=int, default=3)
    parser.add_argument("--validation-mc-samples", type=int, default=20)
    parser.add_argument("--prediction-seed", type=int, default=1729)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--label",
        default="confirmatory",
        help="Run label recorded in the aggregate report, e.g. smoke or confirmatory.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Generate per-seed diagnostic plots; disabled by default for throughput.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse seed directories whose summary.csv already exists.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Aggregate existing seed directories without launching training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining seeds after a failed subprocess.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training or writing summaries.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    fractions = np.asarray(args.split_fractions, dtype=np.float64)
    if len(fractions) != 4 or np.any(fractions <= 0.0):
        raise ValueError("Four positive split fractions are required")
    if not np.isclose(float(fractions.sum()), 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"Split fractions must sum to 1.0, received {fractions.sum():.12g}"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if not args.data.is_file():
        raise FileNotFoundError(f"Input data not found: {args.data}")
    if "capacitance" in args.task and not args.capacitance_data.exists():
        raise FileNotFoundError(
            f"Capacitance data not found: {args.capacitance_data}"
        )
    if args.min_calibration_groups < 1:
        raise ValueError("--min-calibration-groups must be positive")


def seed_output(args: argparse.Namespace, seed: int) -> Path:
    return args.output / f"seed_{seed}"


def command_for_seed(args: argparse.Namespace, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(RUN_RESEARCH),
        "--task",
        *args.task,
        "--model",
        "bayesian",
        "--bayes-method",
        args.bayes_method,
        "--data",
        str(args.data),
        "--capacitance-data",
        str(args.capacitance_data),
        "--output",
        str(seed_output(args, seed)),
        "--seed",
        str(seed),
        "--split-fractions",
        *(f"{value:.12g}" for value in args.split_fractions),
        "--epochs",
        str(args.epochs),
        "--width",
        str(args.width),
        "--grid",
        str(args.grid),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--kl-weight",
        str(args.kl_weight),
        "--dropout-rate",
        str(args.dropout_rate),
        "--mc-samples",
        str(args.mc_samples),
        "--train-mc-samples",
        str(args.train_mc_samples),
        "--validation-mc-samples",
        str(args.validation_mc_samples),
        "--prediction-seed",
        str(args.prediction_seed),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--conformal-unit",
        "curve",
        "--charge-mode",
        "off",
        "--device",
        args.device,
    ]
    if args.diagnostics:
        command.append("--diagnostics")
    return command


def seed_config_mismatches(
    args: argparse.Namespace,
    seed: int,
) -> list[str]:
    path = seed_output(args, seed) / "run_config.json"
    if not path.is_file():
        return [f"missing {path.name}"]
    config = json.loads(path.read_text(encoding="utf-8"))

    def literal(name: str):
        value = config.get(name)
        if not isinstance(value, str):
            return value
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    expected = {
        "task": list(args.task),
        "model": "bayesian",
        "bayes_method": args.bayes_method,
        "seed": seed,
        "split_fractions": list(args.split_fractions),
        "epochs": args.epochs,
        "width": args.width,
        "grid": args.grid,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "dropout_rate": args.dropout_rate,
        "mc_samples": args.mc_samples,
        "train_mc_samples": args.train_mc_samples,
        "validation_mc_samples": args.validation_mc_samples,
        "prediction_seed": args.prediction_seed,
        "conformal_unit": "curve",
    }
    if "capacitance" in args.task:
        expected["capacitance_data"] = str(args.capacitance_data)
    mismatches = []
    for name, expected_value in expected.items():
        actual = literal(name)
        if isinstance(expected_value, float):
            matches = (
                isinstance(actual, (int, float))
                and np.isclose(
                    float(actual),
                    expected_value,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        elif (
            isinstance(expected_value, list)
            and expected_value
            and isinstance(expected_value[0], float)
        ):
            matches = bool(
                isinstance(actual, (list, tuple))
                and len(actual) == len(expected_value)
                and np.allclose(
                    np.asarray(actual, dtype=float),
                    np.asarray(expected_value, dtype=float),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        else:
            matches = actual == expected_value
        if not matches:
            mismatches.append(
                f"{name}: expected {expected_value!r}, found {actual!r}"
            )
    return mismatches


def run_seed(args: argparse.Namespace, seed: int) -> None:
    output = seed_output(args, seed)
    summary = output / "summary.csv"
    if summary.is_file():
        if args.resume:
            mismatches = seed_config_mismatches(args, seed)
            if mismatches:
                raise RuntimeError(
                    f"Seed {seed} configuration mismatch:\n"
                    + "\n".join(mismatches)
                )
            print(f"[seed={seed}] reuse {summary}")
            return
        raise FileExistsError(
            f"{summary} already exists; use --resume or a new --output"
        )

    command = command_for_seed(args, seed)
    print(f"[seed={seed}] {' '.join(command)}")
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Seed {seed} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )


def wilson_interval(percent: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not np.isfinite(percent) or n <= 0:
        return np.nan, np.nan
    successes = int(round(float(percent) * n / 100.0))
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return 100.0 * (center - radius), 100.0 * (center + radius)


def audit_split_manifests(
    args: argparse.Namespace,
    available_seeds: list[int],
) -> pd.DataFrame:
    rows = []
    for seed in available_seeds:
        output = seed_output(args, seed)
        for path in sorted(output.glob("*_split_manifest.csv")):
            frame = pd.read_csv(path, dtype={"group_id": str})
            if frame.empty:
                continue
            task = str(frame["task"].iloc[0])
            split_counts = frame.groupby("split")["group_id"].nunique()
            memberships = frame.groupby("group_id")["split"].nunique()
            overlap_groups = int((memberships > 1).sum())
            row = {
                "seed": seed,
                "task": task,
                "manifest": str(path.resolve().relative_to(ROOT)),
                "overlap_groups": overlap_groups,
                "train_groups": int(split_counts.get("train", 0)),
                "validation_groups": int(split_counts.get("validation", 0)),
                "calibration_groups": int(split_counts.get("calibration", 0)),
                "test_groups": int(split_counts.get("test", 0)),
            }
            row["calibration_minimum_ok"] = bool(
                row["calibration_groups"] >= args.min_calibration_groups
            )
            row["split_disjoint"] = overlap_groups == 0
            rows.append(row)
    return pd.DataFrame(rows)


def load_seed_metrics(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[int]]:
    frames = []
    available_seeds = []
    for seed in args.seeds:
        path = seed_output(args, seed) / "summary.csv"
        if not path.is_file():
            continue
        mismatches = seed_config_mismatches(args, seed)
        if mismatches:
            raise RuntimeError(
                f"Seed {seed} configuration mismatch:\n"
                + "\n".join(mismatches)
            )
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
        available_seeds.append(seed)
    if not frames:
        raise FileNotFoundError(
            f"No seed summary.csv files found under {args.output}"
        )
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    metrics["run_label"] = args.label
    for coverage_column in (
        "raw_curve_coverage_95",
        "standard_conformal_curve_coverage_95",
        "no_shrink_conformal_curve_coverage_95",
        "calibrated_curve_coverage_95",
        "raw_group_coverage_95",
        "standard_conformal_group_coverage_95",
        "no_shrink_conformal_group_coverage_95",
        "calibrated_group_coverage_95",
        "raw_device_coverage_95",
        "standard_conformal_device_coverage_95",
        "no_shrink_conformal_device_coverage_95",
        "calibrated_device_coverage_95",
    ):
        if coverage_column not in metrics:
            continue
        lows = []
        highs = []
        for row in metrics.itertuples(index=False):
            count = getattr(row, "test_group_count", None)
            if count is None or not np.isfinite(float(count)):
                count = getattr(row, "test_curve_count")
            low, high = wilson_interval(
                float(getattr(row, coverage_column)),
                int(count),
            )
            lows.append(low)
            highs.append(high)
        metrics[f"{coverage_column}_wilson_low"] = lows
        metrics[f"{coverage_column}_wilson_high"] = highs
    return metrics, available_seeds


def t_critical_95(n: int) -> float:
    if n <= 1:
        return np.nan
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=n - 1))
    except Exception:
        return 1.959963984540054


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["task", "model"]
    for keys, group in metrics.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in CORE_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            n = int(len(values))
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else np.nan
            median = float(values.median())
            q25 = float(values.quantile(0.25))
            q75 = float(values.quantile(0.75))
            radius = (
                t_critical_95(n) * std / math.sqrt(n)
                if n > 1 and np.isfinite(std)
                else np.nan
            )
            ci_low = mean - radius
            ci_high = mean + radius
            if "picp" in metric or "coverage" in metric:
                ci_low = max(0.0, ci_low) if np.isfinite(ci_low) else ci_low
                ci_high = min(100.0, ci_high) if np.isfinite(ci_high) else ci_high
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_median"] = median
            row[f"{metric}_q25"] = q25
            row[f"{metric}_q75"] = q75
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        coverage_metric = (
            "calibrated_group_coverage_95"
            if "calibrated_group_coverage_95" in group
            else "calibrated_curve_coverage_95"
        )
        if f"{coverage_metric}_mean" in row:
            row["calibrated_group_coverage_gap_pp"] = (
                row[f"{coverage_metric}_mean"] - 95.0
            )
            coverage = pd.to_numeric(
                group[coverage_metric],
                errors="coerce",
            ).dropna()
            row["group_coverage_below_95_runs"] = int((coverage < 95.0).sum())
            row["group_coverage_below_90_runs"] = int((coverage < 90.0).sum())
            row["group_coverage_below_95_fraction"] = float(
                (coverage < 95.0).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def format_number(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def write_report(
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    split_audit: pd.DataFrame,
) -> None:
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )
    lines = [
        "# Repeated Grouped-Split UQ Report",
        "",
        f"Generated: {generated_at}",
        "",
        "## Design",
        "",
        f"- Run label: `{args.label}`",
        f"- Bayesian inference method: `{args.bayes_method}`",
        f"- Seeds completed: `{', '.join(map(str, sorted(metrics['seed'].unique())))}`",
        "- Split unit: independent group (condition curve; capacitance DEVICE)",
        "- Fractions (train/validation/calibration/test): "
        f"`{'/'.join(f'{value:.0%}' for value in args.split_fractions)}`",
        f"- Minimum calibration groups required: `{args.min_calibration_groups}`",
        "- Validation selects checkpoints; calibration fits the conformal multiplier; "
        "test is used only for final metrics.",
        "",
        "## Split Audit",
        "",
    ]
    if split_audit.empty:
        lines.append("- No split manifests were found.")
    else:
        lines.extend(
            [
                f"- Manifests checked: `{len(split_audit)}`",
                f"- Overlapping condition groups: `{int(split_audit['overlap_groups'].sum())}`",
                "- Calibration minimum satisfied: "
                f"`{int(split_audit['calibration_minimum_ok'].sum())}/{len(split_audit)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Task | Seeds | RMSE mean | Cal. point PICP | Cal. group coverage | "
            "95% CI across seeds | Runs <95% | q mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate.to_dict(orient="records"):
        ci = (
            f"[{format_number(row.get('calibrated_group_coverage_95_ci95_low', row.get('calibrated_curve_coverage_95_ci95_low')), 2)}, "
            f"{format_number(row.get('calibrated_group_coverage_95_ci95_high', row.get('calibrated_curve_coverage_95_ci95_high')), 2)}]"
        )
        lines.append(
            f"| {row.get('task', '')} | {int(row.get('n_seeds', 0))} | "
            f"{format_number(row.get('rmse_model_space_mean'), 6)} | "
            f"{format_number(row.get('calibrated_picp_95_mean'), 2)}% | "
            f"{format_number(row.get('calibrated_group_coverage_95_mean', row.get('calibrated_curve_coverage_95_mean')), 2)}% | "
            f"{ci} | {int(row.get('group_coverage_below_95_runs', 0))}/"
            f"{int(row.get('n_seeds', 0))} | "
            f"{format_number(row.get('calibration_z_mean'), 4)} |"
        )
    lines.extend(
        [
            "",
            "The confidence interval above describes variation across repeated grouped "
            "splits. Per-seed Wilson intervals are available in "
            "`uq_metrics_by_seed.csv`; reused groups across repetitions mean they "
            "should not be interpreted as fully independent pooled observations.",
            "",
        ]
    )
    (args.output / "repeated_uq_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_outputs(args: argparse.Namespace) -> None:
    metrics, available_seeds = load_seed_metrics(args)
    split_audit = audit_split_manifests(args, available_seeds)
    if split_audit.empty:
        raise RuntimeError("No split manifests found; cannot audit group leakage")
    if not split_audit["split_disjoint"].all():
        raise RuntimeError("Independent-group leakage detected in split manifests")
    if not split_audit["calibration_minimum_ok"].all():
        failed = split_audit.loc[
            ~split_audit["calibration_minimum_ok"],
            ["seed", "task", "calibration_groups"],
        ]
        raise RuntimeError(
            "Calibration group minimum not met:\n"
            + failed.to_string(index=False)
        )

    aggregate = aggregate_metrics(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "uq_metrics_by_seed.csv", index=False)
    aggregate.to_csv(args.output / "uq_metrics_summary.csv", index=False)
    split_audit.to_csv(args.output / "split_audit.csv", index=False)
    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "data": str(args.data),
        "tasks": args.task,
        "seeds_requested": args.seeds,
        "seeds_completed": available_seeds,
        "split_fractions": args.split_fractions,
        "min_calibration_groups": args.min_calibration_groups,
        "epochs": args.epochs,
        "width": args.width,
        "grid": args.grid,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "bayes_method": args.bayes_method,
        "dropout_rate": args.dropout_rate,
        "mc_samples": args.mc_samples,
        "train_mc_samples": args.train_mc_samples,
        "validation_mc_samples": args.validation_mc_samples,
        "prediction_seed": args.prediction_seed,
        "device": args.device,
        "label": args.label,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(args, metrics, aggregate, split_audit)
    print(aggregate.to_string(index=False))
    print(f"\nRepeated-UQ summary saved to: {args.output}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.dry_run:
        for seed in args.seeds:
            run_seed(args, seed)
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        failures = []
        for seed in args.seeds:
            try:
                run_seed(args, seed)
            except Exception as exc:
                failures.append((seed, str(exc)))
                print(f"[seed={seed}] ERROR: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
        if failures:
            (args.output / "failures.json").write_text(
                json.dumps(failures, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    write_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
