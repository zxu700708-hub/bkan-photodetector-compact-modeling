"""Run structure-grouped microring through/drop model comparisons.

The adaptive table is used only for fitting and validation.  Evaluation can
use either the held-out adaptive rows (for smoke tests) or complete 10001-point
held-out spectra.  The independent split unit is always ``structure_id``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
PAPER_EXPERIMENTS = BKAN_ROOT / "simulation" / "paper_experiments"
for path in (ROOT / "scripts", BKAN_ROOT, PAPER_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_framework_baselines import fit_predict_gmls, train_autopinn_adapted  # noqa: E402
from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler  # noqa: E402
from device_modeling.photodetector.common import set_seed  # noqa: E402
from p0_3_traditional_comparison import (  # noqa: E402
    MLP_ARCHES,
    TaskSpec as BaselineTaskSpec,
    build_engineering_model,
    count_engineering_params,
    make_scaled_arrays,
    train_dkan,
    train_mlp,
)
from run_apd_second_device import paired_statistics  # noqa: E402


DEFAULT_DATA_ROOT = Path(r"<FROZEN_RING_ROOT>/hifi_ring_80\results\paper_80_hifi_v2")
DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "ring_third_device"
INPUT_COLUMNS = (
    "core_width_nm",
    "coupling_gap_nm",
    "ring_radius_um",
    "bias_v",
    "frequency_thz",
)
CORE_MODELS = ("bkan", "dkan", "mlp_l", "spline_ridge", "gmls", "autopinn")
ALL_MODELS = (*CORE_MODELS, "poly3_ridge")
MODEL_LABELS = {
    "bkan": "BKAN-VI",
    "dkan": "DKAN",
    "mlp_l": "MLP-L",
    "spline_ridge": "Spline-Ridge",
    "gmls": "GMLS-adapted",
    "autopinn": "AutoPINN-adapted",
    "poly3_ridge": "Poly3-Ridge",
}


@dataclass(frozen=True)
class RingTask:
    key: str
    name: str
    target_col: str
    ylabel: str
    input_cols: tuple[str, ...] = INPUT_COLUMNS
    axis_col: str = "frequency_thz"
    use_log_transform: bool = False
    y_bounds: tuple[float, float] = (-100.0, 5.0)
    primary: bool = True


TASKS = {
    "through": RingTask("through", "Ring_through_dB", "through_db", "Through response (dB)"),
    "drop": RingTask("drop", "Ring_drop_dB", "drop_db", "Drop response (dB)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--models", nargs="+", choices=list(ALL_MODELS), default=list(CORE_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--evaluation", choices=("adaptive", "full"), default="full")
    parser.add_argument("--max-fit-rows-per-curve", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--autopinn-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--train-mc-samples", type=int, default=3)
    parser.add_argument("--validation-mc-samples", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    if args.smoke:
        args.seeds = [42]
        args.evaluation = "adaptive"
        args.max_fit_rows_per_curve = 128
        args.epochs = min(args.epochs, 20)
        args.kan_steps = min(args.kan_steps, 20)
        args.mlp_epochs = min(args.mlp_epochs, 20)
        args.autopinn_epochs = min(args.autopinn_epochs, 20)
        args.mc_samples = min(args.mc_samples, 10)
        args.validation_mc_samples = min(args.validation_mc_samples, 5)
        args.early_stopping_patience = min(args.early_stopping_patience, 2)
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        raise ValueError("Seeds must be nonempty and unique")
    if len(args.models) != len(set(args.models)):
        raise ValueError("Models must be unique")
    if args.batch_size < 1 or min(args.epochs, args.kan_steps, args.mlp_epochs, args.autopinn_epochs) < 1:
        raise ValueError("Training sizes and iteration counts must be positive")
    if args.autopinn_epochs < 10 and "autopinn" in args.models:
        raise ValueError("AutoPINN-adapted requires at least 10 epochs")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_fingerprint(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> str:
    hashed = pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def baseline_spec(task: RingTask) -> BaselineTaskSpec:
    return BaselineTaskSpec(
        name=task.name,
        target_col=task.target_col,
        input_cols=task.input_cols,
        use_log_transform=False,
        y_bounds=task.y_bounds,
        ylabel=task.ylabel,
        group_cols=("structure_id",),
    )


def deterministic_limit(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if maximum <= 0:
        return frame.reset_index(drop=True)
    pieces = []
    for _, curve in frame.groupby("curve_id", sort=False):
        if len(curve) <= maximum:
            pieces.append(curve)
            continue
        dense = curve[curve["sample_reason"].eq("resonance_dense")]
        coarse = curve[curve["sample_reason"].eq("off_resonance_coarse")]
        dense_budget = min(len(dense), max(1, int(round(0.75 * maximum))))
        coarse_budget = min(len(coarse), maximum - dense_budget)
        if dense_budget + coarse_budget < maximum:
            dense_budget = min(len(dense), maximum - coarse_budget)

        def evenly(part: pd.DataFrame, count: int) -> pd.DataFrame:
            if count <= 0:
                return part.iloc[:0]
            positions = np.unique(np.linspace(0, len(part) - 1, count).round().astype(int))
            return part.iloc[positions]

        pieces.append(pd.concat([evenly(dense, dense_budget), evenly(coarse, coarse_budget)]))
    result = pd.concat(pieces).sort_values(
        ["structure_id", "bias_v", "curve_row_index"], kind="stable"
    )
    return result.reset_index(drop=True)


def load_inputs(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ready = data_root / "training_ready"
    manifest = json.loads((ready / "ring_training_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "ring_resonance_aware_grouped_training_v1":
        raise RuntimeError("Unexpected or obsolete microring training manifest")
    paths = {
        "ring_training_adaptive.csv": ready / "ring_training_adaptive.csv",
        "ring_repeated_group_splits.csv": ready / "ring_repeated_group_splits.csv",
        "ring_sampling_audit.csv": ready / "ring_sampling_audit.csv",
    }
    for name, path in paths.items():
        if file_sha256(path) != manifest["files"][name]:
            raise RuntimeError(f"Training-data hash mismatch: {path}")
    adaptive = pd.read_csv(paths["ring_training_adaptive.csv"])
    splits = pd.read_csv(paths["ring_repeated_group_splits.csv"])
    required = {*INPUT_COLUMNS, "structure_id", "curve_id", "sample_reason", "through_db", "drop_db"}
    if not required.issubset(adaptive.columns):
        raise KeyError(f"Adaptive training table is missing {sorted(required - set(adaptive.columns))}")
    return adaptive, splits, manifest


def split_partitions(
    adaptive: pd.DataFrame,
    splits: pd.DataFrame,
    seed: int,
    max_fit_rows_per_curve: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = splits[splits["seed"].eq(seed)]
    if manifest["structure_id"].nunique() != 80 or len(manifest) != 80:
        raise RuntimeError(f"Missing 80 unique structure assignments for seed {seed}")
    assignments = dict(zip(manifest["structure_id"], manifest["split"]))
    if not set(adaptive["structure_id"]).issubset(assignments):
        raise RuntimeError(f"Unassigned adaptive structure for seed {seed}")
    frame = adaptive.copy()
    frame["split"] = frame["structure_id"].map(assignments)
    raw = tuple(frame[frame["split"].eq(name)].copy() for name in ("train", "validation", "calibration", "test"))
    train = deterministic_limit(raw[0], max_fit_rows_per_curve)
    validation = deterministic_limit(raw[1], max_fit_rows_per_curve)
    calibration = deterministic_limit(raw[2], max_fit_rows_per_curve)
    return train, validation, calibration, raw[3].reset_index(drop=True), manifest


def load_full_test(data_root: Path, test_structures: set[str]) -> pd.DataFrame:
    script_root = data_root.parents[1]
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))
    from prepare_ring_training_data import C_LIGHT_M_PER_S, resonance_aware_indices

    frames = []
    for structure_id in sorted(test_structures):
        path = data_root / "physical_models" / structure_id / "interconnect" / "through_drop_spectra.csv"
        spectra = pd.read_csv(path)
        for bias, response in spectra.groupby("bias_v", sort=False):
            response = response.reset_index(drop=True)
            selected, _, reason, _ = resonance_aware_indices(response["drop_db"].to_numpy(dtype=float))
            dense = np.zeros(len(response), dtype=bool)
            dense[selected[np.asarray(reason) == "resonance_dense"]] = True
            response["frequency_thz"] = C_LIGHT_M_PER_S / response["wavelength_m"] / 1e12
            response["curve_row_index"] = np.arange(len(response))
            response["sample_reason"] = np.where(dense, "resonance_dense", "full_grid_background")
            response["represented_full_grid_rows"] = 1.0
            response["curve_id"] = f"{structure_id}|{float(bias):g}"
            frames.append(response)
    return pd.concat(frames, ignore_index=True)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def train_bkan(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    task: RingTask,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    modeler = BayesKANDeviceModeler(task_name=task.name, results_dir=str(model_dir), device=str(device))
    modeler.load_data(
        train,
        input_cols=list(task.input_cols),
        output_col=task.target_col,
        use_log_transform=False,
        y_bounds=task.y_bounds,
    )
    modeler.build_model(
        {
            "width": [None, 8, 2],
            "seed": seed,
            "grid": 8,
            "k": 3,
            "grid_range": [-3, 3],
            "kl_weight": args.kl_weight,
            "num_mc_samples": args.mc_samples,
            "prior_mu": 0.0,
            "prior_log_sigma": 0.0,
            "posterior_init_sigma": 0.1,
            "likelihood": "gaussian",
            "inference_method": "vi",
            "prediction_seed": 1729 + seed,
        }
    )
    log_path = model_dir / "training.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        modeler.train(
            {
                "num_epochs": args.epochs,
                "batch_size": min(args.batch_size, len(train)),
                "lr": args.learning_rate,
                "weight_decay": 1.0e-5,
                "val_freq": min(10, args.epochs),
                "train_mc_samples": args.train_mc_samples,
                "validation_mc_samples": args.validation_mc_samples,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": 1.0e-4,
                "validation_seed": 1729 + seed,
            },
            df_val=validation,
        )
    predictions = []
    for start in range(0, len(test), 8192):
        result = modeler.predict_with_uncertainty(
            test.iloc[start : start + 8192][list(task.input_cols)].to_numpy(dtype=np.float32)
        )
        predictions.append(np.asarray(result["model_mean"], dtype=np.float64))
    modeler.save_model(str(model_dir / "model_checkpoint.pt"))
    return np.concatenate(predictions), {
        "param_count": int(sum(p.numel() for p in modeler.model.parameters() if p.requires_grad)),
        "train_time_s": float(time.perf_counter() - started),
        "epochs_completed": len(modeler.experiment.history.get("train_loss", [])),
        "best_val_loss": float(modeler.experiment.history.get("best_val_loss", np.nan)),
    }


def prediction_path(output: Path, seed: int, task: RingTask, model: str) -> Path:
    return output / "predictions" / f"seed_{seed}" / task.key / model / "test_predictions.npz"


def save_prediction(path: Path, test: pd.DataFrame, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        prediction=np.asarray(prediction, dtype=np.float32),
        through_db=test["through_db"].to_numpy(dtype=np.float32),
        drop_db=test["drop_db"].to_numpy(dtype=np.float32),
        structure_id=test["structure_id"].astype(str).to_numpy(dtype="U"),
        bias_v=test["bias_v"].to_numpy(dtype=np.float32),
        wavelength_m=test["wavelength_m"].to_numpy(dtype=np.float64),
        sample_reason=test["sample_reason"].astype(str).to_numpy(dtype="U"),
        represented_full_grid_rows=test["represented_full_grid_rows"].to_numpy(dtype=np.float32),
    )
    temporary.replace(path)


def load_prediction(path: Path, expected_rows: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as bundle:
        prediction = np.asarray(bundle["prediction"], dtype=np.float64)
    if len(prediction) != expected_rows or not np.isfinite(prediction).all():
        raise RuntimeError(f"Stale or invalid prediction: {path}")
    return prediction


def prediction_metrics(test: pd.DataFrame, task: RingTask, prediction: np.ndarray) -> dict[str, float]:
    actual = test[task.target_col].to_numpy(dtype=np.float64)
    error = np.asarray(prediction, dtype=np.float64) - actual
    weights = test["represented_full_grid_rows"].to_numpy(dtype=np.float64)
    weighted_mse = float(np.average(error**2, weights=weights))
    group_rmse = []
    curve_rmse = []
    for _, indices in test.groupby("structure_id", sort=False).groups.items():
        idx = np.asarray(indices, dtype=int)
        group_rmse.append(math.sqrt(float(np.average(error[idx] ** 2, weights=weights[idx]))))
    for _, indices in test.groupby("curve_id", sort=False).groups.items():
        idx = np.asarray(indices, dtype=int)
        curve_rmse.append(math.sqrt(float(np.average(error[idx] ** 2, weights=weights[idx]))))
    dense = test["sample_reason"].eq("resonance_dense").to_numpy()
    return {
        "rmse_target": math.sqrt(weighted_mse),
        "rmse_db": math.sqrt(mean_squared_error(actual, prediction)),
        "mae_db": mean_absolute_error(actual, prediction),
        "r2": r2_score(actual, prediction),
        "structure_macro_rmse_db": float(np.mean(group_rmse)),
        "curve_macro_rmse_db": float(np.mean(curve_rmse)),
        "resonance_region_rmse_db": math.sqrt(float(np.mean(error[dense] ** 2))),
    }


def train_one(
    model: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    task: RingTask,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[np.ndarray, dict]:
    spec = baseline_spec(task)
    if model == "bkan":
        return train_bkan(train, validation, test, task, seed, device, args, model_dir)
    arrays = make_scaled_arrays(train, test, spec)
    if model == "dkan":
        return train_dkan(arrays, seed, device, args.kan_steps, grid=8, k=3, width=8)
    if model == "mlp_l":
        return train_mlp(arrays, MLP_ARCHES["mlp_l"], seed, device, args.mlp_epochs, 0.002, 0.001, 1.0, 2.0)
    if model in {"spline_ridge", "poly3_ridge"}:
        fitted = build_engineering_model(model, len(train), len(task.input_cols), seed)
        fitted.fit(train[list(task.input_cols)].to_numpy(dtype=np.float64), train[task.target_col].to_numpy(dtype=np.float64))
        prediction = fitted.predict(test[list(task.input_cols)].to_numpy(dtype=np.float64)).ravel()
        return prediction, {
            "param_count": count_engineering_params(model, fitted, len(train), len(task.input_cols)),
            "train_time_s": 0.0,
        }
    if model == "gmls":
        return fit_predict_gmls(train, validation, test, spec)
    if model == "autopinn":
        return train_autopinn_adapted(
            train,
            validation,
            test,
            spec,
            axis_col=task.axis_col,
            monotonic_sign=None,
            seed=seed,
            device=device,
            epochs=args.autopinn_epochs,
        )
    raise ValueError(model)


def write_outputs(metrics: pd.DataFrame, manifests: list[pd.DataFrame], args: argparse.Namespace, data_manifest: dict, device: torch.device) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics_by_seed.csv", index=False)
    numeric = [column for column in metrics.columns if column.endswith(("rmse_db", "mae_db")) or column in {"rmse_target", "r2", "wall_time_s", "param_count"}]
    summary = metrics.groupby(["task", "model"], sort=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "metrics_summary.csv", index=False)
    if len(args.seeds) >= 2:
        paired_args = SimpleNamespace(
            models=args.models,
            bootstrap_seed=20260911,
            bootstrap_replicates=50000,
            split_fractions=data_manifest["split_fractions"],
        )
        paired_statistics(metrics, paired_args).to_csv(args.output / "paired_statistics.csv", index=False)
    pd.concat(manifests, ignore_index=True).to_csv(args.output / "split_manifest.csv", index=False)

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D"]
    fig, axes = plt.subplots(1, len(args.tasks), figsize=(6.0 * len(args.tasks), 4.2), squeeze=False)
    for ax, task_key in zip(axes.flat, args.tasks):
        part = summary[summary["task"].eq(TASKS[task_key].name)]
        x = np.arange(len(part))
        ax.bar(x, part["rmse_target_mean"], yerr=part["rmse_target_std"].fillna(0.0), color=colors[: len(part)], capsize=3)
        ax.set_xticks(x, [MODEL_LABELS[value] for value in part["model"]], rotation=35, ha="right")
        ax.set_ylabel("Held-out weighted RMSE (dB)")
        ax.set_title(TASKS[task_key].name)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "ring_model_rmse.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol": "ring_structure_grouped_model_comparison_smoke_v1" if args.smoke else "ring_structure_grouped_model_comparison_v1",
        "data_root": str(args.data_root),
        "training_manifest_sha256": file_sha256(args.data_root / "training_ready" / "ring_training_manifest.json"),
        "output": str(args.output),
        "tasks": args.tasks,
        "models": args.models,
        "seeds": args.seeds,
        "evaluation": args.evaluation,
        "max_fit_rows_per_curve": args.max_fit_rows_per_curve,
        "independent_unit": "structure_id",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "smoke_test_only": bool(args.smoke),
        "training": {
            "epochs": args.epochs,
            "kan_steps": args.kan_steps,
            "mlp_epochs": args.mlp_epochs,
            "autopinn_epochs": args.autopinn_epochs,
            "batch_size": args.batch_size,
            "mc_samples": args.mc_samples,
        },
        "claim_boundary": {
            "paper_result": not args.smoke,
            "measurement_validation": False,
            "fixed_official_baseline_dispersion_ldf": True,
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# 微环 {'烟雾测试' if args.smoke else '结构分组模型比较'}",
        "",
        f"独立拆分单位为 `structure_id`；评估模式为 `{args.evaluation}`。",
        "本报告未写入论文。" if args.smoke else "结果仍需完成派生指标复算与统计审计后才能进入论文。",
        "",
        "| Task | Model | Weighted RMSE (dB) | Resonance RMSE (dB) | R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.task} | {MODEL_LABELS[row.model]} | {row.rmse_target_mean:.6g} | "
            f"{row.resonance_region_rmse_db_mean:.6g} | {row.r2_mean:.6g} |"
        )
    (args.output / "ring_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    adaptive, splits, data_manifest = load_inputs(args.data_root)
    if not set(args.seeds).issubset(set(splits["seed"].unique())):
        raise ValueError("Requested seed is absent from the frozen split manifest")
    device = resolve_device(args.device)
    print(f"Training on {device}: tasks={args.tasks}, models={args.models}, seeds={args.seeds}")
    metric_rows = []
    manifests = []
    for seed in args.seeds:
        train, validation, calibration, adaptive_test, split_manifest = split_partitions(
            adaptive, splits, seed, args.max_fit_rows_per_curve
        )
        test_ids = set(split_manifest.loc[split_manifest["split"].eq("test"), "structure_id"])
        test = adaptive_test if args.evaluation == "adaptive" else load_full_test(args.data_root, test_ids)
        manifests.append(split_manifest.assign(task="both"))
        print(
            f"[seed={seed}] fit/val/cal/test structures="
            f"{train.structure_id.nunique()}/{validation.structure_id.nunique()}/"
            f"{calibration.structure_id.nunique()}/{test.structure_id.nunique()}, "
            f"rows={len(train)}/{len(validation)}/{len(calibration)}/{len(test)}"
        )
        for task_key in args.tasks:
            task = TASKS[task_key]
            for model in args.models:
                out_path = prediction_path(args.output, seed, task, model)
                metadata_path = out_path.with_name("run_metadata.json")
                fingerprint_columns = ["structure_id", *task.input_cols, task.target_col]
                train_hash = frame_fingerprint(train, fingerprint_columns)
                test_hash = frame_fingerprint(test, fingerprint_columns)
                if args.resume and out_path.is_file() and metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("train_fingerprint") != train_hash or metadata.get("test_fingerprint") != test_hash:
                        raise RuntimeError(f"Refusing stale resume prediction: {out_path}")
                    prediction = load_prediction(out_path, len(test))
                    info = metadata
                    reused = True
                else:
                    started = time.perf_counter()
                    prediction, info = train_one(
                        model, train, validation, test, task, seed, device, args, out_path.parent
                    )
                    info["wall_time_s"] = float(time.perf_counter() - started)
                    save_prediction(out_path, test, prediction)
                    reused = False
                values = prediction_metrics(test, task, prediction)
                row = {
                    "task": task.name,
                    "task_key": task.key,
                    "seed": seed,
                    "model": model,
                    **values,
                    "wall_time_s": float(info.get("wall_time_s", info.get("train_time_s", 0.0))),
                    "param_count": int(info.get("param_count", 0)),
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "test_rows": int(len(test)),
                    "train_groups": int(train["structure_id"].nunique()),
                    "test_groups": int(test["structure_id"].nunique()),
                    "evaluation": args.evaluation,
                    "reused": reused,
                    "train_fingerprint": train_hash,
                    "test_fingerprint": test_hash,
                }
                metadata_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
                metric_rows.append(row)
                print(
                    f"[{task.key} seed={seed} {model}] weighted RMSE={values['rmse_target']:.5f} dB, "
                    f"resonance RMSE={values['resonance_region_rmse_db']:.5f} dB"
                )
    write_outputs(pd.DataFrame(metric_rows), manifests, args, data_manifest, device)


def main() -> int:
    args = resolve_args(parse_args())
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
