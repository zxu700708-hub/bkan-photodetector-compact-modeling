"""Rerun structured OOD holdouts with frozen-protocol Bayesian KAN UQ.

The Bayesian KAN architecture is unchanged (width 8, grid 8, spline order 3,
two output channels).  Validation is used for checkpoint selection, an
independent group split is used for conformal calibration, and OOD rows are
never used for fitting, selection, or calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bkan"))

from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler  # noqa: E402
from device_modeling.photodetector.task_config import (  # noqa: E402
    DEFAULT_CAPACITANCE_DATA,
    TASKS,
    load_capacitance_data,
    load_research_data,
    prepare_task_dataframe,
    target_values,
    task_group_labels,
)


DEFAULT_DATA = ROOT / "artifacts" / "results" / "device_modeling" / "cleaned_data.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "structured_ood_uq"
SPLIT_FRACTIONS = (0.65, 0.10, 0.15, 0.10)
AXIS_COLUMNS = {"dark_voltage", "light_voltage", "frequency_ghz", "bias_v", "log_frequency_ghz"}


@dataclass(frozen=True)
class Scenario:
    task: str
    name: str
    column: str
    side: str
    fraction: float = 0.20


SCENARIOS = (
    Scenario("dark_current", "voltage_extreme_reverse", "dark_voltage", "low"),
    Scenario("dark_current", "length_high", "active_layer_length", "high"),
    Scenario("dark_current", "temperature_high", "simulation_temperature", "high"),
    Scenario("dark_current", "trap_high", "trap_assisted_recomb_A", "high"),
    Scenario("photo_current", "voltage_extreme_reverse", "light_voltage", "low"),
    Scenario("photo_current", "length_high", "active_layer_length", "high"),
    Scenario("photo_current", "temperature_high", "simulation_temperature", "high"),
    Scenario("photo_current", "trap_high", "trap_assisted_recomb_A", "high"),
    Scenario("ac_response", "frequency_high", "frequency_ghz", "high"),
    Scenario("ac_response", "length_high", "active_layer_length", "high"),
    Scenario("ac_response", "temperature_high", "simulation_temperature", "high"),
    Scenario("capacitance", "voltage_extreme_reverse", "bias_v", "low"),
    Scenario("capacitance", "frequency_high", "log_frequency_ghz", "high"),
    Scenario("capacitance", "length_high", "active_layer_length", "high"),
    Scenario("capacitance", "temperature_high", "simulation_temperature", "high"),
    Scenario("capacitance", "trap_high", "trap_assisted_recomb_A", "high"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--capacitance-data", type=Path, default=DEFAULT_CAPACITANCE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--train-mc-samples", type=int, default=3)
    parser.add_argument("--validation-mc-samples", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--cap-batch-size", type=int, default=2048)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--fit-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job-indices", nargs="+", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _group_count_allocation(n: int, fractions=SPLIT_FRACTIONS) -> tuple[int, ...]:
    exact = np.asarray(fractions) * n
    counts = np.floor(exact).astype(int)
    for idx in np.argsort(-(exact - counts), kind="stable")[: n - int(counts.sum())]:
        counts[idx] += 1
    if np.any(counts < 1):
        raise ValueError(f"Too few groups ({n}) for four nonempty partitions")
    return tuple(map(int, counts))


def _split_groups(groups: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    groups = np.asarray(groups, dtype=str).copy()
    np.random.default_rng(seed).shuffle(groups)
    counts = _group_count_allocation(len(groups))
    cuts = np.cumsum(counts)[:-1]
    return tuple(np.split(groups, cuts))


def structured_partitions(
    frame: pd.DataFrame,
    spec,
    scenario: Scenario,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Create leakage-free fit/validation/calibration/ID/OOD partitions."""
    work = frame.copy().reset_index(drop=True)
    work["_row_id"] = np.arange(len(work), dtype=int)
    work["_curve_id"] = task_group_labels(work, spec).astype(str).to_numpy()
    values = pd.to_numeric(work[scenario.column], errors="raise")
    quantile = scenario.fraction if scenario.side == "low" else 1.0 - scenario.fraction
    threshold = float(values.quantile(quantile))
    is_ood = values <= threshold if scenario.side == "low" else values >= threshold

    is_axis_holdout = scenario.column == spec.axis_col or scenario.column in AXIS_COLUMNS
    if is_axis_holdout:
        train_g, val_g, cal_g, test_g = _split_groups(work["_curve_id"].unique(), seed)
        group_sets = {"train": train_g, "validation": val_g, "calibration": cal_g, "test": test_g}
        partitions = {
            "train": work.loc[work["_curve_id"].isin(train_g) & ~is_ood].copy(),
            "validation": work.loc[work["_curve_id"].isin(val_g) & ~is_ood].copy(),
            "calibration": work.loc[work["_curve_id"].isin(cal_g) & ~is_ood].copy(),
            "id_test": work.loc[work["_curve_id"].isin(test_g) & ~is_ood].copy(),
            "ood_test": work.loc[work["_curve_id"].isin(test_g) & is_ood].copy(),
        }
        design = "axis_tail_on_independent_test_groups"
        discarded_ood_groups = int(sum(len(group_sets[key]) for key in ("train", "validation", "calibration")))
    else:
        group_value = work.groupby("_curve_id", sort=False)[scenario.column].median()
        group_ood = group_value <= threshold if scenario.side == "low" else group_value >= threshold
        ood_groups = group_value.index[group_ood].to_numpy(dtype=str)
        id_groups = group_value.index[~group_ood].to_numpy(dtype=str)
        train_g, val_g, cal_g, test_g = _split_groups(id_groups, seed)
        partitions = {
            "train": work.loc[work["_curve_id"].isin(train_g)].copy(),
            "validation": work.loc[work["_curve_id"].isin(val_g)].copy(),
            "calibration": work.loc[work["_curve_id"].isin(cal_g)].copy(),
            "id_test": work.loc[work["_curve_id"].isin(test_g)].copy(),
            "ood_test": work.loc[work["_curve_id"].isin(ood_groups)].copy(),
        }
        design = "condition_tail_group_holdout"
        discarded_ood_groups = 0

    if any(part.empty for part in partitions.values()):
        sizes = {key: len(value) for key, value in partitions.items()}
        raise ValueError(f"Empty structured partition for {scenario}: {sizes}")
    fit_names = ("train", "validation", "calibration")
    fit_rows = set().union(*(set(partitions[name]["_row_id"]) for name in fit_names))
    test_rows = set(partitions["id_test"]["_row_id"]) | set(partitions["ood_test"]["_row_id"])
    if fit_rows & test_rows:
        raise RuntimeError("Row leakage between fit/calibration and test partitions")
    fit_groups = set().union(*(set(partitions[name]["_curve_id"]) for name in fit_names))
    test_groups = set(partitions["id_test"]["_curve_id"])
    if fit_groups & test_groups:
        raise RuntimeError("Independent test-group leakage detected")
    if not is_axis_holdout and fit_groups & set(partitions["ood_test"]["_curve_id"]):
        raise RuntimeError("Condition-OOD group leakage detected")

    audit = {
        "design": design,
        "threshold": threshold,
        "holdout_fraction": scenario.fraction,
        "discarded_ood_group_fragments": discarded_ood_groups,
        "split_rows": {key: int(len(value)) for key, value in partitions.items()},
        "split_groups": {key: int(value["_curve_id"].nunique()) for key, value in partitions.items()},
        "row_overlap": 0,
        "independent_test_group_overlap": 0,
    }
    return {key: value.reset_index(drop=True) for key, value in partitions.items()}, audit


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_task(args: argparse.Namespace, task: str):
    spec = TASKS[task]
    raw = load_capacitance_data(args.capacitance_data) if task == "capacitance" else load_research_data(args.data)
    frame, inputs = prepare_task_dataframe(raw, spec)
    return spec, frame, inputs


def _fit(args: argparse.Namespace, scenario: Scenario, seed: int) -> None:
    spec, frame, inputs = _load_task(args, scenario.task)
    partitions, audit = structured_partitions(frame, spec, scenario, seed)
    run_dir = args.output / "runs" / scenario.task / scenario.name / f"seed_{seed}" / "bkan-results"
    prediction_path = run_dir / "predictions.csv"
    if prediction_path.is_file():
        if args.resume:
            print(f"[reuse] {scenario.task}/{scenario.name}/seed={seed}", flush=True)
            return
        raise FileExistsError(f"Existing result: {prediction_path}; use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    modeler = BayesKANDeviceModeler(spec.name, str(run_dir), device=args.device)
    modeler.load_data(
        partitions["train"], list(inputs), spec.target_col,
        use_log_transform=spec.use_log_transform, y_bounds=spec.y_bounds,
    )
    modeler.build_model({
        "width": [None, 8, 2], "seed": seed, "grid": 8, "k": 3,
        "grid_range": [-3, 3], "kl_weight": 0.1,
        "num_mc_samples": args.mc_samples, "prior_mu": 0.0,
        "prior_log_sigma": 0.0, "posterior_init_sigma": 0.1,
        "likelihood": "gaussian", "inference_method": "vi",
        "prediction_seed": 1729,
    })
    batch_size = args.cap_batch_size if scenario.task == "capacitance" else 32
    modeler.train({
        "num_epochs": args.epochs, "batch_size": batch_size, "lr": 1e-3,
        "weight_decay": 1e-5, "val_freq": max(1, min(10, args.epochs)),
        "train_mc_samples": args.train_mc_samples,
        "validation_mc_samples": args.validation_mc_samples,
        "early_stopping_patience": 7, "early_stopping_min_delta": 1e-4,
        "validation_seed": 1729,
    }, df_val=partitions["validation"])
    q = modeler.calibrate_uncertainty(
        df_calib=partitions["calibration"], target_picp=95.0, min_z=1.96,
        calibration_unit="curve", group_cols=["_curve_id"],
    )
    modeler.save_model(run_dir / "model_checkpoint.pt")

    outputs = []
    for domain in ("calibration", "id_test", "ood_test"):
        part = partitions[domain].copy()
        part.attrs = {}
        pred = modeler.predict_with_uncertainty(part[list(inputs)].to_numpy(dtype=np.float32))
        out = part[["_row_id", "_curve_id", *inputs, spec.target_col]].copy()
        out.attrs = {}
        out.insert(0, "domain", domain)
        out["y_true_model"] = target_values(part, spec)
        for key in ("model_mean", "model_std", "epistemic", "aleatoric", "raw_model_lower", "raw_model_upper", "model_lower", "model_upper"):
            out[key] = np.asarray(pred[key]).reshape(-1)
        outputs.append(out)
    pd.concat(outputs, ignore_index=True).to_csv(prediction_path, index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario": asdict(scenario), "seed": seed, "inputs": list(inputs),
        "architecture": {"width": [len(inputs), 8, 2], "grid": 8, "spline_order": 3},
        "training": {"epochs": args.epochs, "batch_size": batch_size, "learning_rate": 1e-3, "kl_weight": 0.1},
        "calibration": {"unit": "independent_curve_or_DEVICE", "min_z": 1.96, "q": float(q), "score_count": int(modeler._calib_score_count)},
        "audit": audit,
    }
    (run_dir / "protocol.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = []
    for name, part in partitions.items():
        manifest.extend({"split": name, "row_id": int(row), "group_id": str(group)} for row, group in zip(part["_row_id"], part["_curve_id"]))
    pd.DataFrame(manifest).to_csv(run_dir / "split_manifest.csv", index=False)
    print(f"[done] {scenario.task}/{scenario.name}/seed={seed}", flush=True)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _domain_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true_model"].to_numpy(float)
    mu = frame["model_mean"].to_numpy(float)
    sigma = frame["model_std"].to_numpy(float)
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    floor = max(float(np.median(positive)) * 0.1, np.finfo(float).tiny) if len(positive) else 1e-8
    sigma = np.maximum(sigma, floor)
    raw_lo, raw_hi = frame["raw_model_lower"].to_numpy(float), frame["raw_model_upper"].to_numpy(float)
    cal_lo, cal_hi = frame["model_lower"].to_numpy(float), frame["model_upper"].to_numpy(float)
    raw_hit = (y >= raw_lo) & (y <= raw_hi)
    cal_hit = (y >= cal_lo) & (y <= cal_hi)
    grouped = pd.DataFrame({"group": frame["_curve_id"], "raw": raw_hit, "cal": cal_hit}).groupby("group").all()
    residual = y - mu
    z = residual / sigma
    crps = sigma * (z * (2 * _normal_cdf(z) - 1) + 2 * np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))
    alpha = 0.05
    interval_score = (cal_hi - cal_lo) + (2 / alpha) * (cal_lo - y) * (y < cal_lo) + (2 / alpha) * (y - cal_hi) * (y > cal_hi)
    rho = spearmanr(np.abs(residual), sigma).statistic if len(y) > 2 else np.nan
    order = np.argsort(sigma)
    risk_curve = np.cumsum(np.abs(residual)[order]) / np.arange(1, len(y) + 1)
    return {
        "n_rows": len(frame), "n_groups": frame["_curve_id"].nunique(),
        "rmse": math.sqrt(mean_squared_error(y, mu)), "r2": r2_score(y, mu),
        "raw_point_coverage_95": 100 * raw_hit.mean(), "calibrated_point_coverage_95": 100 * cal_hit.mean(),
        "raw_group_coverage_95": 100 * grouped["raw"].mean(), "calibrated_group_coverage_95": 100 * grouped["cal"].mean(),
        "raw_mpiw": float(np.mean(raw_hi - raw_lo)), "calibrated_mpiw": float(np.mean(cal_hi - cal_lo)),
        "mean_predictive_std": float(np.mean(sigma)), "median_predictive_std": float(np.median(sigma)),
        "mean_epistemic_std": float(frame["epistemic"].mean()), "mean_aleatoric_std": float(frame["aleatoric"].mean()),
        "gaussian_nll": float(np.mean(0.5 * np.log(2 * math.pi * sigma**2) + 0.5 * z**2)),
        "gaussian_crps": float(np.mean(crps)), "interval_score_95": float(np.mean(interval_score)),
        "abs_error_std_spearman": float(rho), "aurc_abs_error": float(np.mean(risk_curve)),
    }


def aggregate(args: argparse.Namespace, jobs: list[tuple[Scenario, int]]) -> None:
    metric_rows, shift_rows = [], []
    for scenario, seed in jobs:
        run_dir = args.output / "runs" / scenario.task / scenario.name / f"seed_{seed}" / "bkan-results"
        path = run_dir / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed run: {path}")
        frame = pd.read_csv(path, dtype={"_curve_id": str})
        per_domain = {}
        for domain in ("id_test", "ood_test"):
            metrics = _domain_metrics(frame.loc[frame["domain"] == domain].copy())
            per_domain[domain] = metrics
            metric_rows.append({"task": scenario.task, "scenario": scenario.name, "seed": seed, "domain": domain, **metrics})
        labels = frame["domain"].isin(["ood_test"]).astype(int)
        subset = frame[frame["domain"].isin(["id_test", "ood_test"])]
        labels = (subset["domain"] == "ood_test").astype(int)
        auc = roc_auc_score(labels, subset["model_std"]) if labels.nunique() == 2 else np.nan
        shift_rows.append({
            "task": scenario.task, "scenario": scenario.name, "seed": seed,
            "ood_std_over_id": per_domain["ood_test"]["mean_predictive_std"] / max(per_domain["id_test"]["mean_predictive_std"], np.finfo(float).tiny),
            "ood_epistemic_over_id": per_domain["ood_test"]["mean_epistemic_std"] / max(per_domain["id_test"]["mean_epistemic_std"], np.finfo(float).tiny),
            "ood_detection_auroc_predictive_std": auc,
            "rmse_shift_ood_minus_id": per_domain["ood_test"]["rmse"] - per_domain["id_test"]["rmse"],
            "coverage_shift_ood_minus_id_pp": per_domain["ood_test"]["calibrated_point_coverage_95"] - per_domain["id_test"]["calibrated_point_coverage_95"],
        })
    metrics = pd.DataFrame(metric_rows)
    shifts = pd.DataFrame(shift_rows)
    metrics.to_csv(args.output / "metrics_by_run_domain.csv", index=False)
    shifts.to_csv(args.output / "uncertainty_shift_by_run.csv", index=False)
    metric_summary = metrics.groupby(["task", "scenario", "domain"], sort=True).agg(
        n_seeds=("seed", "nunique"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"), calibrated_point_coverage_95_mean=("calibrated_point_coverage_95", "mean"),
        calibrated_group_coverage_95_mean=("calibrated_group_coverage_95", "mean"),
        calibrated_mpiw_mean=("calibrated_mpiw", "mean"), mean_predictive_std_mean=("mean_predictive_std", "mean"),
        gaussian_nll_mean=("gaussian_nll", "mean"), gaussian_crps_mean=("gaussian_crps", "mean"),
        interval_score_95_mean=("interval_score_95", "mean"), abs_error_std_spearman_mean=("abs_error_std_spearman", "mean"),
    ).reset_index()
    shift_summary = shifts.groupby(["task", "scenario"], sort=True).agg(
        n_seeds=("seed", "nunique"), ood_std_over_id_mean=("ood_std_over_id", "mean"),
        ood_std_over_id_std=("ood_std_over_id", "std"), ood_epistemic_over_id_mean=("ood_epistemic_over_id", "mean"),
        ood_detection_auroc_mean=("ood_detection_auroc_predictive_std", "mean"),
        rmse_shift_ood_minus_id_mean=("rmse_shift_ood_minus_id", "mean"),
        coverage_shift_ood_minus_id_pp_mean=("coverage_shift_ood_minus_id_pp", "mean"),
    ).reset_index()
    metric_summary.to_csv(args.output / "metrics_summary.csv", index=False)
    shift_summary.to_csv(args.output / "uncertainty_shift_summary.csv", index=False)
    protocol = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "tasks": args.tasks,
        "seeds": args.seeds, "scenarios": [asdict(s) for s, _ in jobs[::len(args.seeds)]],
        "split_fractions": SPLIT_FRACTIONS, "architecture_changed": False,
        "architecture": {"hidden_width": 8, "grid": 8, "spline_order": 3, "output_channels": 2},
        "calibration": "independent-group normalized conformal, no-shrink q >= 1.96",
        "ood_use": "OOD rows/groups excluded from training, validation, and calibration",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    lines = [
        "# Structured OOD Holdout UQ Rerun", "", "## Protocol", "",
        "- The B-KAN architecture is unchanged: width 8, grid 8, spline order 3, two output channels.",
        "- Training, validation, conformal calibration, ID test, and OOD test are disjoint.",
        "- Axis-tail OOD is assessed on independent test groups; condition-tail OOD holds out complete condition groups.",
        "- Calibration is group-level normalized conformal with a no-shrink 1.96 floor.", "", "## OOD uncertainty trajectory", "",
        "| Task | Scenario | Seeds | OOD/ID std | OOD AUROC | OOD-ID RMSE | OOD-ID coverage (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in shift_summary.itertuples(index=False):
        lines.append(f"| {row.task} | {row.scenario} | {row.n_seeds} | {row.ood_std_over_id_mean:.3f} | {row.ood_detection_auroc_mean:.3f} | {row.rmse_shift_ood_minus_id_mean:.5g} | {row.coverage_shift_ood_minus_id_pp_mean:.2f} |")
    lines.extend(["", "Full per-domain calibration, proper scores, uncertainty decomposition, risk-ranking correlation, and saved predictions are in the accompanying CSV files."])
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    if args.epochs < 50 and not args.allow_smoke:
        raise ValueError("Confirmatory reruns require at least 50 epochs")
    selected = [s for s in SCENARIOS if s.task in args.tasks and (args.scenarios is None or s.name in args.scenarios)]
    jobs = [(scenario, seed) for scenario in selected for seed in args.seeds]
    if args.job_indices is not None:
        jobs_to_fit = [jobs[idx] for idx in args.job_indices]
    else:
        jobs_to_fit = jobs
    for scenario, seed in jobs_to_fit:
        _fit(args, scenario, seed)
    if not args.fit_only:
        args.output.mkdir(parents=True, exist_ok=True)
        aggregate(args, jobs)


if __name__ == "__main__":
    main()
