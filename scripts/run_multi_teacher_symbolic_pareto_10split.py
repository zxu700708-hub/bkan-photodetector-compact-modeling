"""Extend the common-family symbolic/direct audit to ten matched grouped splits.

Each seed/task run delegates to :mod:`run_multi_teacher_symbolic_pareto`, so the
student families, ridge selection, teacher reconstruction, replay gates, and
export serialization remain identical to the frozen seed-42 audit.  The
corrected net-photocurrent task is explicitly routed to its retrained BKAN and
matched-prediction roots.  Results are then aggregated with paired repeated-
split statistics; the ten overlapping partitions are not treated as ten
independent datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_multi_teacher_symbolic_pareto as single  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts/results/multi_teacher_symbolic_pareto_10split"
DEFAULT_MATCHED = ROOT / "artifacts/results/matched_grouped_comparison"
DEFAULT_BKAN = ROOT / "artifacts/results/uq_repeated_grouped"
DEFAULT_CAP_BKAN = ROOT / "artifacts/results/uq_repeated_grouped_capacitance"
DEFAULT_NET_ROOT = ROOT / "artifacts/results/net_photocurrent_retrain"
DEFAULT_SEEDS = tuple(range(42, 52))
TEACHER_SOURCES = ("bkan_parameter_mean", "dkan", "mlp_l")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=single.DEFAULT_DATA)
    parser.add_argument(
        "--net-photocurrent-data",
        type=Path,
        default=DEFAULT_NET_ROOT / "device_modeling/cleaned_data.csv",
    )
    parser.add_argument(
        "--capacitance-data", type=Path, default=single.shared.DEFAULT_CAPACITANCE_DATA
    )
    parser.add_argument("--matched-root", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--bkan-root", type=Path, default=DEFAULT_BKAN)
    parser.add_argument("--capacitance-bkan-root", type=Path, default=DEFAULT_CAP_BKAN)
    parser.add_argument("--net-root", type=Path, default=DEFAULT_NET_ROOT)
    parser.add_argument("--tasks", nargs="+", choices=tuple(single.TASK_KEY), default=list(single.TASK_KEY))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--split-fractions", nargs=4, type=float, default=(0.65, 0.10, 0.15, 0.10)
    )
    parser.add_argument("--poly-degrees", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--spline-knots", nargs="+", type=int, default=(4, 6, 8))
    parser.add_argument(
        "--ridge-alphas", nargs="+", type=float, default=(1e-8, 1e-6, 1e-4, 1e-2)
    )
    parser.add_argument("--dense-points", type=int, default=5000)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun complete seed/task directories instead of resuming them.",
    )
    parser.add_argument(
        "--allow-teacher-replay-mismatch",
        action="store_true",
        help="Forward the diagnostic mismatch override to the single-split runner.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.output = args.output.resolve()
    args.data = args.data.resolve()
    args.net_photocurrent_data = args.net_photocurrent_data.resolve()
    args.capacitance_data = args.capacitance_data.resolve()
    args.matched_root = args.matched_root.resolve()
    args.bkan_root = args.bkan_root.resolve()
    args.capacitance_bkan_root = args.capacitance_bkan_root.resolve()
    args.net_root = args.net_root.resolve()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")
    if len(args.seeds) < 2:
        raise ValueError("At least two matched seeds are required for paired statistics")
    if not args.net_photocurrent_data.is_file():
        raise FileNotFoundError(args.net_photocurrent_data)
    for seed in args.seeds:
        expected = args.matched_root / "predictions" / f"seed_{seed}"
        if any(task != "I_photo" for task in args.tasks) and not expected.is_dir():
            raise FileNotFoundError(expected)
        if "I_photo" in args.tasks:
            net_expected = (
                args.net_root / "matched_grouped_comparison/predictions" / f"seed_{seed}"
            )
            if not net_expected.is_dir():
                raise FileNotFoundError(net_expected)


def _run_dir(output: Path, seed: int, task: str) -> Path:
    return output / f"seed_{seed}" / task


def _complete(path: Path, seed: int, task: str) -> bool:
    required = (
        path / "metrics_by_export.csv",
        path / "teacher_audit.csv",
        path / "split_manifest.csv",
        path / "protocol.json",
    )
    if not all(item.is_file() for item in required):
        return False
    try:
        metrics = pd.read_csv(required[0])
        teachers = pd.read_csv(required[1])
        protocol = json.loads(required[3].read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        len(metrics) == 24
        and len(teachers) == 3
        and set(metrics["seed"].astype(int)) == {seed}
        and set(metrics["task"].astype(str)) == {task}
        and protocol.get("seed") == seed
        and protocol.get("tasks") == [task]
        and int(metrics["test_finite_rate"].eq(1.0).sum()) == 24
        and int(metrics["dense_finite_rate"].eq(1.0).sum()) == 24
    )


def _single_args(args: argparse.Namespace, seed: int, task: str) -> SimpleNamespace:
    is_photo = task == "I_photo"
    return SimpleNamespace(
        data=args.net_photocurrent_data if is_photo else args.data,
        capacitance_data=args.capacitance_data,
        output=_run_dir(args.output, seed, task),
        matched_root=(
            args.net_root / "matched_grouped_comparison" if is_photo else args.matched_root
        ),
        bkan_root=(args.net_root / "uq_repeated_grouped" if is_photo else args.bkan_root),
        capacitance_bkan_root=args.capacitance_bkan_root,
        tasks=[task],
        seed=seed,
        split_fractions=tuple(args.split_fractions),
        poly_degrees=tuple(args.poly_degrees),
        spline_knots=tuple(args.spline_knots),
        ridge_alphas=tuple(args.ridge_alphas),
        dense_points=args.dense_points,
        latency_repeats=args.latency_repeats,
        mlp_epochs=args.mlp_epochs,
        kan_steps=args.kan_steps,
        device="cpu",
        allow_teacher_replay_mismatch=args.allow_teacher_replay_mismatch,
    )


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def corrected_paired_statistics(
    differences: np.ndarray,
    test_fraction: float,
    train_fraction: float,
) -> dict:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    count = len(differences)
    if count < 2:
        raise ValueError("At least two finite paired differences are required")
    mean = float(np.mean(differences))
    median = float(np.median(differences))
    sample_sd = float(np.std(differences, ddof=1))
    variance_factor = 1.0 / count + test_fraction / train_fraction
    corrected_se = math.sqrt(variance_factor) * sample_sd
    critical = float(stats.t.ppf(0.975, df=count - 1))
    if corrected_se == 0.0:
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        p_value = float(2.0 * stats.t.sf(abs(mean / corrected_se), df=count - 1))
    tolerance = np.maximum(
        np.finfo(np.float64).tiny,
        100.0 * np.finfo(np.float64).eps * np.abs(differences),
    )
    return {
        "pairs": count,
        "mean_delta_teacher_minus_direct": mean,
        "median_delta_teacher_minus_direct": median,
        "sample_sd_delta": sample_sd,
        "corrected_standard_error": corrected_se,
        "corrected_ci_low": mean - critical * corrected_se,
        "corrected_ci_high": mean + critical * corrected_se,
        "corrected_two_sided_p": p_value,
        "teacher_wins": int(np.sum(differences < -tolerance)),
        "ties": int(np.sum(np.abs(differences) <= tolerance)),
        "teacher_losses": int(np.sum(differences > tolerance)),
    }


def aggregate_results(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_frames = []
    teacher_frames = []
    manifest_frames = []
    for seed in args.seeds:
        for task in args.tasks:
            path = _run_dir(args.output, seed, task)
            if not _complete(path, seed, task):
                raise RuntimeError(f"Incomplete seed/task output: {path}")
            metrics_frames.append(pd.read_csv(path / "metrics_by_export.csv"))
            teacher_frames.append(pd.read_csv(path / "teacher_audit.csv"))
            manifest_frames.append(pd.read_csv(path / "split_manifest.csv"))
    metrics = pd.concat(metrics_frames, ignore_index=True)
    teachers = pd.concat(teacher_frames, ignore_index=True)
    manifests = pd.concat(manifest_frames, ignore_index=True)
    return metrics, teachers, manifests


def summarize_exports(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "source", "source_kind", "family", "budget", "terms"]
    summary = (
        metrics.groupby(keys, as_index=False, sort=True)
        .agg(
            splits=("seed", "nunique"),
            mean_rmse=("formula_to_tcad_rmse", "mean"),
            sd_rmse=("formula_to_tcad_rmse", "std"),
            median_rmse=("formula_to_tcad_rmse", "median"),
            mean_r2=("formula_to_tcad_r2", "mean"),
            min_r2=("formula_to_tcad_r2", "min"),
            mean_formula_chars=("formula_chars", "mean"),
            mean_latency_per_point_s=("latency_per_point_s", "mean"),
            min_test_finite_rate=("test_finite_rate", "min"),
            min_dense_finite_rate=("dense_finite_rate", "min"),
            min_derivative_finite_rate=("swept_axis_derivative_finite_rate", "min"),
            max_json_replay_error=("independent_json_replay_max_abs_error", "max"),
        )
    )
    summary["pareto_mean_rmse_terms"] = single.mark_pareto(
        summary, error_column="mean_rmse", complexity_column="terms"
    )
    return summary


def paired_teacher_direct(metrics: pd.DataFrame, fractions: tuple[float, ...]) -> pd.DataFrame:
    direct = metrics.loc[
        metrics["source"].eq("tcad_direct"),
        ["task", "seed", "family", "budget", "terms", "formula_to_tcad_rmse"],
    ].rename(columns={"formula_to_tcad_rmse": "direct_rmse"})
    rows = []
    for source in TEACHER_SOURCES:
        teacher = metrics.loc[
            metrics["source"].eq(source),
            ["task", "seed", "family", "budget", "terms", "formula_to_tcad_rmse"],
        ].rename(columns={"formula_to_tcad_rmse": "teacher_rmse"})
        paired = teacher.merge(
            direct, on=["task", "seed", "family", "budget", "terms"], validate="one_to_one"
        )
        paired["delta"] = paired["teacher_rmse"] - paired["direct_rmse"]
        for keys, group in paired.groupby(["task", "family", "budget", "terms"], sort=True):
            stats_row = corrected_paired_statistics(
                group["delta"].to_numpy(), fractions[3], fractions[0]
            )
            rows.append(
                {
                    "task": keys[0],
                    "teacher_source": source,
                    "family": keys[1],
                    "budget": int(keys[2]),
                    "terms": int(keys[3]),
                    "mean_teacher_rmse": float(group["teacher_rmse"].mean()),
                    "mean_direct_rmse": float(group["direct_rmse"].mean()),
                    "mean_relative_reduction_teacher": float(
                        np.mean((group["direct_rmse"] - group["teacher_rmse"]) / group["direct_rmse"])
                    ),
                    **stats_row,
                }
            )
    result = pd.DataFrame(rows)
    result["holm_p_global_72"] = holm_adjust(result["corrected_two_sided_p"])
    result["holm_p_within_task_18"] = result.groupby("task")[
        "corrected_two_sided_p"
    ].transform(lambda values: holm_adjust(values).to_numpy())
    result["corrected_ci_favors_teacher"] = result["corrected_ci_high"] < 0.0
    result["corrected_ci_favors_direct"] = result["corrected_ci_low"] > 0.0
    return result.sort_values(["task", "family", "budget", "teacher_source"])


def summarize_teachers(teachers: pd.DataFrame) -> pd.DataFrame:
    return (
        teachers.groupby(["task", "teacher"], as_index=False, sort=True)
        .agg(
            splits=("seed", "nunique"),
            mean_teacher_to_tcad_rmse=("teacher_to_tcad_rmse", "mean"),
            sd_teacher_to_tcad_rmse=("teacher_to_tcad_rmse", "std"),
            max_replay_abs_error=("replay_max_abs_error", "max"),
            replay_statuses=("replay_status", lambda values: ";".join(sorted(set(values)))),
        )
    )


def write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    teachers: pd.DataFrame,
) -> None:
    front = summary.loc[summary["pareto_mean_rmse_terms"]]
    source_counts = front["source"].value_counts().to_dict()
    significant_teacher = paired.loc[
        paired["holm_p_global_72"].lt(0.05) & paired["corrected_ci_favors_teacher"]
    ]
    significant_direct = paired.loc[
        paired["holm_p_global_72"].lt(0.05) & paired["corrected_ci_favors_direct"]
    ]
    lines = [
        "# Ten-Split Data-Budget-Matched Multi-Teacher Symbolic Audit",
        "",
        f"Seeds: {', '.join(map(str, args.seeds))}. Each seed uses the frozen matched "
        "65/10/15/10 grouped partition. The corrected net-photocurrent task uses the "
        "dark-subtracted retrained checkpoints and frozen predictions.",
        "",
        f"The audit contains {len(summary) * len(args.seeds)} fitted exports "
        f"({len(summary)} task/source/family/budget cells across {len(args.seeds)} splits). "
        "Every JSON evaluator and dense/derivative audit is required to remain finite.",
        "",
        "Paired confidence intervals use the repeated-split variance factor "
        "`1/r + n_test/n_train`; Holm columns are reported both globally across all 72 "
        "teacher/direct contrasts and within each task. The splits reuse the same group "
        "pool and are not interpreted as independent datasets.",
        "",
        "## Aggregate RMSE--term Pareto composition",
        "",
        f"Mean-split front source counts: `{json.dumps(source_counts, sort_keys=True)}`.",
        "",
        "## Multiplicity-corrected teacher/direct results",
        "",
        f"Global-Holm contrasts favoring a teacher: {len(significant_teacher)}; "
        f"favoring direct TCAD: {len(significant_direct)}.",
        "",
        "| Task | Teacher | Family | Budget | Terms | Mean teacher RMSE | Mean direct RMSE | Corrected CI for teacher-direct | Global Holm p | W/T/L |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    selected = paired.loc[
        paired["budget"].eq(4) & paired["family"].eq("additive_spline")
    ]
    for row in selected.to_dict(orient="records"):
        lines.append(
            f"| {row['task']} | {row['teacher_source']} | {row['family']} | {row['budget']} | "
            f"{row['terms']} | {row['mean_teacher_rmse']:.6g} | {row['mean_direct_rmse']:.6g} | "
            f"[{row['corrected_ci_low']:.6g}, {row['corrected_ci_high']:.6g}] | "
            f"{row['holm_p_global_72']:.4g} | {row['teacher_wins']}/{row['ties']}/{row['teacher_losses']} |"
        )
    lines.extend(
        [
            "",
            "The four-knot rows above are the exact-complexity comparison used in the main "
            "paper. Complete 72-contrast results, all 960 per-export rows, per-seed formulas, "
            "split manifests, and teacher replay records accompany this report.",
            "",
            "## Teacher replay summary",
            "",
            "| Task | Teacher | Splits | Mean teacher-TCAD RMSE | Max replay abs error | Statuses |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in teachers.to_dict(orient="records"):
        lines.append(
            f"| {row['task']} | {row['teacher']} | {row['splits']} | "
            f"{row['mean_teacher_to_tcad_rmse']:.6g} | {row['max_replay_abs_error']:.6g} | "
            f"{row['replay_statuses']} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol(args: argparse.Namespace, metrics: pd.DataFrame) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "ten-matched-split data-budget-matched multi-teacher symbolic/direct audit",
        "seeds": list(args.seeds),
        "tasks": list(args.tasks),
        "split_fractions": list(args.split_fractions),
        "exports_expected": len(args.seeds) * len(args.tasks) * 4 * 6,
        "exports_observed": len(metrics),
        "net_photocurrent_routing": {
            "target": "net_photocurrent = light_current - dark_current at matched condition and bias",
            "data": str(args.net_photocurrent_data),
            "matched_predictions": str(args.net_root / "matched_grouped_comparison"),
            "bkan_checkpoints": str(args.net_root / "uq_repeated_grouped"),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "statistics": {
            "paired_difference": "teacher-distilled RMSE minus TCAD-direct RMSE",
            "variance_factor": "1/r + n_test/n_train",
            "confidence_interval": "two-sided t interval, df=r-1",
            "multiplicity": "Holm globally over 72 contrasts and separately within each task over 18 contrasts",
            "dependence_warning": "ten splits reuse the same finite group pool and are not independent datasets",
        },
        "teacher_replay_policy": {
            "allow_mismatch": bool(args.allow_teacher_replay_mismatch),
            "interpretation": (
                "Any tolerated DKAN/MLP mismatch remains labeled mismatch_allowed in "
                "teacher_audit.csv and report.md; it is never relabeled pass."
            ),
        },
        "outputs": {
            "per_export": "metrics_by_export.csv",
            "aggregate": "aggregate_by_export.csv",
            "paired": "paired_teacher_vs_direct.csv",
            "pareto": "aggregate_pareto_front.csv",
            "teachers": "teacher_audit.csv and teacher_summary.csv",
            "manifests": "split_manifest.csv",
            "formulas": "seed_<seed>/<task>/exports/<task>/<source>/<family_budget>/formula.{json,va}",
        },
    }
    (args.output / "protocol.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for task in args.tasks:
            run_args = _single_args(args, seed, task)
            if _complete(run_args.output, seed, task) and not args.overwrite:
                print(f"[resume] seed={seed} task={task}")
                continue
            print(f"[run] seed={seed} task={task}")
            single.validate_args(run_args)
            single.run(run_args)

    metrics, teachers, manifests = aggregate_results(args)
    expected = len(args.seeds) * len(args.tasks) * 4 * 6
    if len(metrics) != expected:
        raise RuntimeError(f"Expected {expected} export rows, found {len(metrics)}")
    if not metrics["test_finite_rate"].eq(1.0).all():
        raise RuntimeError("At least one held-out export is non-finite")
    if not metrics["dense_finite_rate"].eq(1.0).all():
        raise RuntimeError("At least one dense-envelope export is non-finite")
    if not metrics["swept_axis_derivative_finite_rate"].eq(1.0).all():
        raise RuntimeError("At least one swept-axis derivative audit is non-finite")

    summary = summarize_exports(metrics)
    paired = paired_teacher_direct(metrics, tuple(args.split_fractions))
    teacher_summary = summarize_teachers(teachers)
    front = summary.loc[summary["pareto_mean_rmse_terms"]].sort_values(
        ["task", "terms", "mean_rmse"]
    )

    metrics.to_csv(args.output / "metrics_by_export.csv", index=False)
    summary.to_csv(args.output / "aggregate_by_export.csv", index=False)
    paired.to_csv(args.output / "paired_teacher_vs_direct.csv", index=False)
    front.to_csv(args.output / "aggregate_pareto_front.csv", index=False)
    teachers.to_csv(args.output / "teacher_audit.csv", index=False)
    teacher_summary.to_csv(args.output / "teacher_summary.csv", index=False)
    manifests.to_csv(args.output / "split_manifest.csv", index=False)
    write_protocol(args, metrics)
    write_report(args, summary, paired, teacher_summary)


def main() -> int:
    args = parse_args()
    validate_args(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
