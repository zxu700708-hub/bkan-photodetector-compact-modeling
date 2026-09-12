"""Isolated DKAN architecture experiments for the active-microring dataset.

This file intentionally leaves ``run_ring_third_device.py`` unchanged.  It
reuses the frozen data, structure splits, full-grid loader, and metric
definitions, while writing every result to a separate output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
PAPER_EXPERIMENTS = BKAN_ROOT / "simulation" / "paper_experiments"
for path in (ROOT / "scripts", BKAN_ROOT, PAPER_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_ring_derived_metrics import (  # noqa: E402
    add_errors,
    aggregate_by_seed,
    aggregate_summary,
    extract_prediction_rows,
    load_pair,
    reference_lookup,
)
from device_modeling.photodetector.common import set_seed  # noqa: E402
from run_ring_third_device import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    INPUT_COLUMNS,
    TASKS,
    file_sha256,
    frame_fingerprint,
    load_full_test,
    load_inputs,
    load_prediction,
    prediction_metrics,
    prediction_path,
    resolve_device,
    save_prediction,
    split_partitions,
)


DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "ring_dkan_architecture_experiment"
DEFAULT_BASELINE = ROOT / "artifacts" / "results" / "ring_third_device"
VARIANTS = (
    "large_separate",
    "joint",
    "joint_weighted",
    "aligned_joint_weighted",
)
DISPLAY_NAMES = {
    "dkan_original": "DKAN-original",
    "gmls_original": "GMLS-original",
    "large_separate": "DKAN-large-separate",
    "joint": "DKAN-joint",
    "joint_weighted": "DKAN-joint-weighted",
    "aligned_joint_weighted": "DKAN-aligned-joint-weighted",
}
CURVE_INPUTS = ("core_width_nm", "coupling_gap_nm", "ring_radius_um", "bias_v")
ALIGNED_INPUTS = (*CURVE_INPUTS, "normalized_detuning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--metric-steps", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    args.baseline_results = args.baseline_results.resolve()
    if args.smoke:
        args.output = args.output.with_name(args.output.name + "_smoke")
        args.seeds = [42]
        args.steps = min(args.steps, 20)
        args.metric_steps = min(args.metric_steps, 30)
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    if args.steps < 1 or args.metric_steps < 1:
        raise ValueError("Optimization steps must be positive")
    return args


def resonance_mixture_weights(frame: pd.DataFrame, resonance_fraction: float = 0.65) -> np.ndarray:
    """Mix a restored-full-grid objective with a resonance-only objective."""
    if not 0.0 <= resonance_fraction <= 1.0:
        raise ValueError("resonance_fraction must lie in [0, 1]")
    represented = frame["represented_full_grid_rows"].to_numpy(dtype=np.float64)
    if np.any(represented <= 0.0):
        raise ValueError("represented_full_grid_rows must be positive")
    full = represented / represented.mean()
    dense = frame["sample_reason"].eq("resonance_dense").to_numpy(dtype=np.float64)
    if dense.mean() <= 0.0:
        raise ValueError("No resonance-dense rows are available")
    resonance = dense / dense.mean()
    mixed = (1.0 - resonance_fraction) * full + resonance_fraction * resonance
    return (mixed / mixed.mean()).astype(np.float32)


@dataclass
class FittedKAN:
    model: torch.nn.Module
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    device: torch.device
    output_count: int

    def predict(self, x: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
        x_scaled = self.x_scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
        pieces: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(x_scaled), chunk_size):
                value = self.model(torch.from_numpy(x_scaled[start : start + chunk_size]).to(self.device))
                if isinstance(value, tuple):
                    value = value[0]
                pieces.append(value.detach().cpu().numpy())
        scaled = np.concatenate(pieces, axis=0)
        return self.y_scaler.inverse_transform(scaled).reshape(-1, self.output_count)


def fit_kan(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    width: int,
    grid: int,
    train_weights: np.ndarray | None = None,
    validation_weights: np.ndarray | None = None,
    log_path: Path,
) -> tuple[FittedKAN, dict[str, float | int]]:
    """Fit a deterministic KAN without exposing held-out test labels."""
    from kan import KAN

    set_seed(seed)
    train_x = np.asarray(train_x, dtype=np.float32)
    validation_x = np.asarray(validation_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32).reshape(len(train_x), -1)
    validation_y = np.asarray(validation_y, dtype=np.float32).reshape(len(validation_x), -1)
    x_scaler = StandardScaler().fit(train_x)
    y_scaler = StandardScaler().fit(train_y)
    x_train = x_scaler.transform(train_x).astype(np.float32)
    x_validation = x_scaler.transform(validation_x).astype(np.float32)
    y_train = y_scaler.transform(train_y).astype(np.float32)
    y_validation = y_scaler.transform(validation_y).astype(np.float32)
    output_count = y_train.shape[1]

    weighted = train_weights is not None
    if weighted:
        if validation_weights is None:
            raise ValueError("Validation weights are required with train weights")
        train_weights = np.asarray(train_weights, dtype=np.float32).reshape(-1, 1)
        validation_weights = np.asarray(validation_weights, dtype=np.float32).reshape(-1, 1)
        if len(train_weights) != len(train_x) or len(validation_weights) != len(validation_x):
            raise ValueError("Weight and row counts differ")
        y_train_label = np.concatenate([y_train, train_weights], axis=1)
        y_validation_label = np.concatenate([y_validation, validation_weights], axis=1)

        def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            truth = target[:, :output_count]
            weight = target[:, output_count]
            row_mse = torch.mean((prediction - truth) ** 2, dim=1)
            return torch.sum(weight * row_mse) / torch.clamp(torch.sum(weight), min=1.0e-12)
    else:
        y_train_label = y_train
        y_validation_label = y_validation
        loss_fn = None

    model = KAN(
        width=[x_train.shape[1], width, output_count],
        grid=grid,
        k=3,
        seed=seed,
        device=device,
        auto_save=False,
    )
    dataset = {
        "train_input": torch.from_numpy(x_train).float().to(device),
        "train_label": torch.from_numpy(y_train_label).float().to(device),
        "test_input": torch.from_numpy(x_validation).float().to(device),
        "test_label": torch.from_numpy(y_validation_label).float().to(device),
    }
    batch = min(256, len(train_x), len(validation_x))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        model.fit(
            dataset,
            opt="Adam",
            steps=steps,
            lr=0.002,
            lamb=0.001,
            lamb_l1=1.0,
            lamb_entropy=2.0,
            lamb_coef=0.1,
            lamb_coefdiff=0.1,
            batch=batch,
            loss_fn=loss_fn,
            log=max(steps + 1, 1),
        )
    fitted = FittedKAN(model, x_scaler, y_scaler, device, output_count)
    return fitted, {
        "param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "train_time_s": float(time.perf_counter() - started),
        "width": width,
        "grid": grid,
        "weighted": weighted,
    }


def curve_table(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    curves = frame.drop_duplicates(["structure_id", "bias_v"])[
        ["structure_id", *CURVE_INPUTS]
    ].copy()
    metrics = reference.reset_index()[
        ["structure_id", "bias_v", "resonance_wavelength_nm", "fwhm_nm"]
    ]
    curves = curves.merge(metrics, on=["structure_id", "bias_v"], how="left", validate="one_to_one")
    if curves[["resonance_wavelength_nm", "fwhm_nm"]].isna().any().any():
        raise RuntimeError("Missing curve-level reference metric")
    curves["log_fwhm_nm"] = np.log(curves["fwhm_nm"].to_numpy(dtype=np.float64))
    return curves


def predicted_alignment_map(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    metric_model: FittedKAN,
) -> tuple[dict[tuple[str, float], tuple[float, float]], pd.DataFrame]:
    curves = curve_table(frame, reference)
    prediction = metric_model.predict(curves[list(CURVE_INPUTS)].to_numpy(dtype=np.float32))
    curves["predicted_resonance_wavelength_nm"] = prediction[:, 0]
    curves["predicted_fwhm_nm"] = np.exp(np.clip(prediction[:, 1], -10.0, 2.0))
    mapping = {
        (str(row.structure_id), float(row.bias_v)): (
            float(row.predicted_resonance_wavelength_nm),
            float(row.predicted_fwhm_nm),
        )
        for row in curves.itertuples(index=False)
    }
    return mapping, curves


def add_predicted_detuning(
    frame: pd.DataFrame,
    mapping: dict[tuple[str, float], tuple[float, float]],
) -> pd.DataFrame:
    result = frame.copy()
    keys = list(zip(result["structure_id"].astype(str), result["bias_v"].astype(float)))
    missing = sorted({key for key in keys if key not in mapping})
    if missing:
        raise RuntimeError(f"Missing predicted alignment for {missing[:3]}")
    center = np.asarray([mapping[key][0] for key in keys], dtype=np.float64)
    fwhm = np.asarray([mapping[key][1] for key in keys], dtype=np.float64)
    wavelength_nm = result["wavelength_m"].to_numpy(dtype=np.float64) * 1.0e9
    result["normalized_detuning"] = np.clip((wavelength_nm - center) / fwhm, -512.0, 512.0)
    return result


def train_variant(
    variant: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    target_columns = ["through_db", "drop_db"]
    if variant == "large_separate":
        predictions = []
        infos = []
        for target in target_columns:
            fitted, info = fit_kan(
                train[list(INPUT_COLUMNS)].to_numpy(dtype=np.float32),
                train[[target]].to_numpy(dtype=np.float32),
                validation[list(INPUT_COLUMNS)].to_numpy(dtype=np.float32),
                validation[[target]].to_numpy(dtype=np.float32),
                seed=seed,
                device=device,
                steps=args.steps,
                width=16,
                grid=12,
                log_path=run_dir / f"{target}_training.log",
            )
            predictions.append(fitted.predict(test[list(INPUT_COLUMNS)].to_numpy(dtype=np.float32))[:, 0])
            infos.append(info)
            del fitted
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return predictions[0], predictions[1], {
            "param_count": int(sum(int(info["param_count"]) for info in infos)),
            "train_time_s": float(sum(float(info["train_time_s"]) for info in infos)),
            "architecture": "two independent [5,16,1] KANs; grid=12",
            "extra_metric_supervision": False,
        }

    train_frame = train
    validation_frame = validation
    test_frame = test
    stage_info: dict[str, object] = {}
    if variant == "aligned_joint_weighted":
        train_curves = curve_table(train, reference)
        validation_curves = curve_table(validation, reference)
        metric_model, metric_info = fit_kan(
            train_curves[list(CURVE_INPUTS)].to_numpy(dtype=np.float32),
            train_curves[["resonance_wavelength_nm", "log_fwhm_nm"]].to_numpy(dtype=np.float32),
            validation_curves[list(CURVE_INPUTS)].to_numpy(dtype=np.float32),
            validation_curves[["resonance_wavelength_nm", "log_fwhm_nm"]].to_numpy(dtype=np.float32),
            seed=seed + 10000,
            device=device,
            steps=args.metric_steps,
            width=12,
            grid=8,
            log_path=run_dir / "metric_head_training.log",
        )
        train_map, train_metric_rows = predicted_alignment_map(train, reference, metric_model)
        validation_map, validation_metric_rows = predicted_alignment_map(validation, reference, metric_model)
        test_map, test_metric_rows = predicted_alignment_map(test, reference, metric_model)
        train_frame = add_predicted_detuning(train, train_map)
        validation_frame = add_predicted_detuning(validation, validation_map)
        test_frame = add_predicted_detuning(test, test_map)
        feature_columns = ALIGNED_INPUTS
        test_center_error = (
            test_metric_rows["predicted_resonance_wavelength_nm"]
            - test_metric_rows["resonance_wavelength_nm"]
        ).abs()
        test_fwhm_ape = 100.0 * (
            test_metric_rows["predicted_fwhm_nm"] - test_metric_rows["fwhm_nm"]
        ).abs() / test_metric_rows["fwhm_nm"]
        stage_info = {
            "metric_head_param_count": int(metric_info["param_count"]),
            "metric_head_train_time_s": float(metric_info["train_time_s"]),
            "test_center_mae_pm": float(1.0e3 * test_center_error.mean()),
            "test_fwhm_median_ape_percent": float(test_fwhm_ape.median()),
            "alignment_uses_predicted_train_coordinates": True,
        }
        del metric_model, train_metric_rows, validation_metric_rows, test_metric_rows
        if device.type == "cuda":
            torch.cuda.empty_cache()
        width, grid = 24, 16
        weighted = True
        architecture = "curve-metric [4,12,2] head + joint [5,24,2] aligned spectral KAN"
    else:
        feature_columns = INPUT_COLUMNS
        width, grid = 16, 12
        weighted = variant == "joint_weighted"
        architecture = "joint [5,16,2] KAN; grid=12"

    fitted, info = fit_kan(
        train_frame[list(feature_columns)].to_numpy(dtype=np.float32),
        train_frame[target_columns].to_numpy(dtype=np.float32),
        validation_frame[list(feature_columns)].to_numpy(dtype=np.float32),
        validation_frame[target_columns].to_numpy(dtype=np.float32),
        seed=seed,
        device=device,
        steps=args.steps,
        width=width,
        grid=grid,
        train_weights=resonance_mixture_weights(train_frame) if weighted else None,
        validation_weights=resonance_mixture_weights(validation_frame) if weighted else None,
        log_path=run_dir / "spectral_training.log",
    )
    prediction = fitted.predict(test_frame[list(feature_columns)].to_numpy(dtype=np.float32))
    result_info: dict[str, object] = {
        **info,
        **stage_info,
        "architecture": architecture,
        "extra_metric_supervision": variant == "aligned_joint_weighted",
    }
    result_info["param_count"] = int(info["param_count"]) + int(stage_info.get("metric_head_param_count", 0))
    result_info["train_time_s"] = float(info["train_time_s"]) + float(stage_info.get("metric_head_train_time_s", 0.0))
    return prediction[:, 0], prediction[:, 1], result_info


def summarize_response(metrics: pd.DataFrame, baseline_results: Path, output: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_results / "metrics_by_seed.csv")
    baseline = baseline.loc[baseline["model"].isin(["dkan", "gmls"])].copy()
    baseline["model"] = baseline["model"].map(
        {"dkan": "dkan_original", "gmls": "gmls_original"}
    )
    keep = [
        "task", "task_key", "seed", "model", "rmse_target", "rmse_db", "mae_db", "r2",
        "structure_macro_rmse_db", "curve_macro_rmse_db", "resonance_region_rmse_db",
        "wall_time_s", "param_count",
    ]
    combined = pd.concat([baseline[keep], metrics[keep]], ignore_index=True)
    combined.to_csv(output / "metrics_with_original_dkan.csv", index=False)
    numeric = [column for column in keep if column not in {"task", "task_key", "seed", "model"}]
    summary = combined.groupby(["task_key", "model"], sort=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "metrics_summary.csv", index=False)
    return combined


def _holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (len(values) - position))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_response_contrasts(response: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Compare each architecture with original DKAN using matched split errors."""
    rows: list[dict[str, object]] = []
    for task, part in response.groupby("task_key", sort=False):
        for metric in ("rmse_target", "resonance_region_rmse_db"):
            pivot = part.pivot(index="seed", columns="model", values=metric)
            if "dkan_original" not in pivot:
                raise RuntimeError("Original DKAN is missing from the paired comparison")
            for model in [column for column in pivot.columns if column != "dkan_original"]:
                values = (pivot[model] - pivot["dkan_original"]).dropna().to_numpy(dtype=np.float64)
                variance = float(values.var(ddof=1))
                corrected_se = math.sqrt((1.0 / len(values) + 0.10 / 0.65) * variance)
                if corrected_se > 0.0:
                    critical = float(student_t.ppf(0.975, df=len(values) - 1))
                    low = float(values.mean() - critical * corrected_se)
                    high = float(values.mean() + critical * corrected_se)
                    raw_p = float(2.0 * student_t.sf(abs(values.mean() / corrected_se), df=len(values) - 1))
                else:
                    low = high = float(values.mean())
                    raw_p = 0.0 if values.mean() != 0.0 else 1.0
                rows.append(
                    {
                        "task_key": task,
                        "metric": metric,
                        "model": model,
                        "baseline": "dkan_original",
                        "n_pairs": len(values),
                        "candidate_minus_original_mean": float(values.mean()),
                        "corrected_ci95_low": low,
                        "corrected_ci95_high": high,
                        "corrected_p_value_raw": raw_p,
                        "candidate_wins": int(np.sum(values < 0.0)),
                        "candidate_losses": int(np.sum(values > 0.0)),
                    }
                )
    result = pd.DataFrame(rows)
    result["corrected_p_value_holm"] = np.nan
    for _, indices in result.groupby(["task_key", "metric"], sort=False).groups.items():
        result.loc[indices, "corrected_p_value_holm"] = _holm(
            result.loc[indices, "corrected_p_value_raw"].to_numpy(dtype=np.float64)
        )
    result.to_csv(output / "paired_response_vs_original_dkan.csv", index=False)
    return result


def audit_derived_metrics(args: argparse.Namespace, reference: pd.DataFrame) -> pd.DataFrame:
    helper_root = args.data_root.parents[1]
    if str(helper_root) not in sys.path:
        sys.path.insert(0, str(helper_root))
    from hifi_common import spectrum_metrics

    records: list[dict[str, object]] = []
    for seed in args.seeds:
        for variant in args.variants:
            records.extend(
                extract_prediction_rows(
                    load_pair(args.output, seed, variant),
                    reference,
                    seed,
                    variant,
                    spectrum_metrics,
                )
            )
    curves = add_errors(pd.DataFrame(records))
    expected = len(args.seeds) * len(args.variants) * 40
    if len(curves) != expected:
        raise RuntimeError(f"Expected {expected} derived curves, found {len(curves)}")
    new_by_seed = aggregate_by_seed(curves)
    baseline_by_seed = pd.read_csv(args.baseline_results / "derived_metrics_by_seed.csv")
    baseline_by_seed = baseline_by_seed.loc[
        baseline_by_seed["model"].isin(["dkan", "gmls"])
        & baseline_by_seed["seed"].isin(args.seeds)
    ].copy()
    baseline_by_seed["model"] = baseline_by_seed["model"].map(
        {"dkan": "dkan_original", "gmls": "gmls_original"}
    )
    combined = pd.concat([baseline_by_seed, new_by_seed], ignore_index=True)
    curves.to_csv(args.output / "derived_metrics_by_curve.csv", index=False)
    combined.to_csv(args.output / "derived_metrics_by_seed.csv", index=False)
    aggregate_summary(combined).to_csv(args.output / "derived_metrics_summary.csv", index=False)
    return combined


def write_report(response: pd.DataFrame, derived: pd.DataFrame, args: argparse.Namespace) -> None:
    response_summary = response.groupby(["task_key", "model"], sort=False)[
        ["rmse_target", "resonance_region_rmse_db"]
    ].agg(["mean", "std"])
    derived_summary = derived.groupby("model", sort=False)[
        ["recovery_fraction", "resonance_mae_pm", "fwhm_median_ape_percent", "quality_factor_median_ape_percent"]
    ].agg(["mean", "std"])
    lines = [
        "# Isolated microring DKAN architecture experiment",
        "",
        "The frozen structure splits and full-grid evaluation are reused. The original training script and result bundle are unchanged.",
        "",
        "## Response metrics",
        "",
        "| Task | Model | Full RMSE (dB) | Resonance RMSE (dB) |",
        "|---|---|---:|---:|",
    ]
    for (task, model), row in response_summary.iterrows():
        lines.append(
            f"| {task} | {DISPLAY_NAMES.get(model, model)} | "
            f"{row[('rmse_target', 'mean')]:.6g} ± {row[('rmse_target', 'std')]:.6g} | "
            f"{row[('resonance_region_rmse_db', 'mean')]:.6g} ± {row[('resonance_region_rmse_db', 'std')]:.6g} |"
        )
    lines.extend([
        "",
        "## Derived metrics",
        "",
        "| Model | Recovery | Resonance MAE (pm) | FWHM median APE (%) | Q median APE (%) |",
        "|---|---:|---:|---:|---:|",
    ])
    for model, row in derived_summary.iterrows():
        lines.append(
            f"| {DISPLAY_NAMES.get(model, model)} | "
            f"{row[('recovery_fraction', 'mean')]:.4f} | "
            f"{row[('resonance_mae_pm', 'mean')]:.6g} | "
            f"{row[('fwhm_median_ape_percent', 'mean')]:.6g} | "
            f"{row[('quality_factor_median_ape_percent', 'mean')]:.6g} |"
        )
    lines.extend([
        "",
        "`aligned_joint_weighted` uses resonance wavelength and FWHM targets from training/validation structures only; it is an explicitly supervised two-stage experiment and is not directly equivalent to the original spectrum-only task.",
    ])
    (args.output / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    adaptive, splits, data_manifest = load_inputs(args.data_root)
    if not set(args.seeds).issubset(set(splits["seed"].unique())):
        raise ValueError("Requested seed is absent from the frozen split manifest")
    if not (args.baseline_results / "metrics_by_seed.csv").is_file():
        raise FileNotFoundError("Original DKAN baseline results are unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    reference = reference_lookup(args.data_root)
    metric_rows: list[dict[str, object]] = []
    manifests: list[pd.DataFrame] = []
    print(f"device={device}, seeds={args.seeds}, variants={args.variants}")

    for seed in args.seeds:
        train, validation, calibration, _, split_manifest = split_partitions(adaptive, splits, seed, 0)
        test_ids = set(split_manifest.loc[split_manifest["split"].eq("test"), "structure_id"])
        test = load_full_test(args.data_root, test_ids)
        manifests.append(split_manifest.assign(experiment="ring_dkan_architecture_v1"))
        if set(train["structure_id"]) & set(test["structure_id"]):
            raise RuntimeError("Structure leakage")
        print(
            f"[seed={seed}] structures train/val/cal/test="
            f"{train.structure_id.nunique()}/{validation.structure_id.nunique()}/"
            f"{calibration.structure_id.nunique()}/{test.structure_id.nunique()}"
        )
        test_hash = frame_fingerprint(test, ["structure_id", *INPUT_COLUMNS, "through_db", "drop_db"])
        train_hash = frame_fingerprint(train, ["structure_id", *INPUT_COLUMNS, "through_db", "drop_db"])
        for variant in args.variants:
            paths = {
                key: prediction_path(args.output, seed, TASKS[key], variant)
                for key in ("through", "drop")
            }
            metadata_path = args.output / "predictions" / f"seed_{seed}" / variant / "run_metadata.json"
            reused = False
            if args.resume and all(path.is_file() for path in paths.values()) and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("train_fingerprint") != train_hash or metadata.get("test_fingerprint") != test_hash:
                    raise RuntimeError(f"Refusing stale resume artifacts for seed={seed}, variant={variant}")
                through_prediction = load_prediction(paths["through"], len(test))
                drop_prediction = load_prediction(paths["drop"], len(test))
                info = metadata
                reused = True
            else:
                started = time.perf_counter()
                through_prediction, drop_prediction, info = train_variant(
                    variant,
                    train,
                    validation,
                    test,
                    reference,
                    seed,
                    device,
                    args,
                    args.output / "training_logs" / f"seed_{seed}" / variant,
                )
                info["wall_time_s"] = float(time.perf_counter() - started)
                save_prediction(paths["through"], test, through_prediction)
                save_prediction(paths["drop"], test, drop_prediction)
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(
                    json.dumps(
                        {
                            **info,
                            "seed": seed,
                            "variant": variant,
                            "train_fingerprint": train_hash,
                            "test_fingerprint": test_hash,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            for key, prediction in (("through", through_prediction), ("drop", drop_prediction)):
                values = prediction_metrics(test, TASKS[key], prediction)
                metric_rows.append(
                    {
                        "task": TASKS[key].name,
                        "task_key": key,
                        "seed": seed,
                        "model": variant,
                        **values,
                        "wall_time_s": float(info.get("wall_time_s", info.get("train_time_s", 0.0))),
                        "param_count": int(info.get("param_count", 0)),
                        "reused": reused,
                        "extra_metric_supervision": bool(info.get("extra_metric_supervision", False)),
                    }
                )
                print(
                    f"[{seed} {variant} {key}] full={values['rmse_target']:.5f}, "
                    f"resonance={values['resonance_region_rmse_db']:.5f} dB"
                )
            del through_prediction, drop_prediction
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "metrics_by_seed.csv", index=False)
    split_output = pd.concat(manifests, ignore_index=True)
    split_output.to_csv(args.output / "split_manifest.csv", index=False)
    if split_output.groupby(["seed", "structure_id"])["split"].nunique().max() != 1:
        raise RuntimeError("Split audit failed")
    combined_response = summarize_response(metrics, args.baseline_results, args.output)
    paired_response_contrasts(combined_response, args.output)
    combined_derived = audit_derived_metrics(args, reference)
    write_report(combined_response, combined_derived, args)
    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol": "ring_dkan_architecture_experiment_v1",
        "data_root": str(args.data_root),
        "source_training_manifest_sha256": file_sha256(
            args.data_root / "training_ready" / "ring_training_manifest.json"
        ),
        "baseline_results": str(args.baseline_results),
        "output": str(args.output),
        "seeds": args.seeds,
        "variants": args.variants,
        "steps": args.steps,
        "metric_steps": args.metric_steps,
        "device": str(device),
        "original_code_modified": False,
        "aligned_variant_extra_supervision": "training/validation resonance wavelength and FWHM only",
        "structure_split_leakage": False,
        "full_grid_test_rows_per_seed": 400040,
        "source_protocol": data_manifest["protocol"],
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = {
        "status": "PASS",
        "response_rows": int(len(metrics)),
        "derived_seed_rows_with_baseline": int(len(combined_derived)),
        "all_response_metrics_finite": bool(np.isfinite(metrics["rmse_target"]).all()),
        "structure_split_leakage": False,
        "metrics_sha256": file_sha256(args.output / "metrics_by_seed.csv"),
        "derived_summary_sha256": file_sha256(args.output / "derived_metrics_summary.csv"),
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


def main() -> int:
    args = resolve_args(parse_args())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
