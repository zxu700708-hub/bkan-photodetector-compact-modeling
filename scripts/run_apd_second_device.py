"""Train the matched second-device APD comparison.

The experiment uses complete physical structures as the independent split
unit.  It evaluates only paired DC-current targets, multiplication behavior,
and two scalar voltage figures of merit.  It does not consume AC, C--V,
terminal-charge, spatial-charge, or noise data.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
PAPER_EXPERIMENTS = BKAN_ROOT / "simulation" / "paper_experiments"
for path in (BKAN_ROOT, PAPER_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from device_modeling.apd.data import (  # noqa: E402
    APD_TASKS,
    DEFAULT_APD_ROOT,
    APDTask,
    load_apd_task,
    training_tables_dir,
)
from device_modeling.photodetector import task_config as shared  # noqa: E402
from device_modeling.photodetector.bayesian_modeler import (  # noqa: E402
    BayesKANDeviceModeler,
)
from device_modeling.photodetector.common import set_seed  # noqa: E402
from p0_3_traditional_comparison import (  # noqa: E402
    ENGINEERING_MODELS,
    MLP_ARCHES,
    TaskSpec as BaselineTaskSpec,
    build_engineering_model,
    compute_metrics,
    count_engineering_params,
    make_scaled_arrays,
    train_dkan,
    train_mlp,
    transformed_target,
)
from compact_framework_baselines import (  # noqa: E402
    fit_predict_curve_lut,
    fit_predict_gmls,
    fit_predict_semiempirical_compact,
    train_autopinn_adapted,
)


DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "apd_second_device"
DEFAULT_TASKS = tuple(APD_TASKS)
DEFAULT_MODELS = ("bkan", "dkan", "mlp_l", "poly3_ridge", "spline_ridge")
EXPANDED_MODELS = (
    "bkan", "dkan", "mlp_pm", "mlp_l", "poly3_ridge", "spline_ridge",
    "rbf_nystroem", "random_forest", "xgboost",
    "gmls", "autopinn", "curve_lut", "semiempirical",
)
CORE_FRAMEWORK_MODELS = (
    "bkan", "dkan", "mlp_l", "spline_ridge", "gmls", "autopinn"
)
COMPACT_EIGHT_MODELS = CORE_FRAMEWORK_MODELS + ("curve_lut", "semiempirical")
MODEL_LABELS = {
    "bkan": "BKAN-VI",
    "dkan": "DKAN",
    "mlp_pm": "MLP-PM",
    "mlp_l": "MLP-L",
    "poly3_ridge": "Poly3-Ridge",
    "spline_ridge": "Spline-Ridge",
    "rbf_nystroem": "RBF-Nystroem",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gmls": "GMLS-adapted",
    "autopinn": "AutoPINN-adapted",
    "curve_lut": "Curve-LUT-PCHIP",
    "semiempirical": "APD SemiEmpirical-CM",
}

AUTOPINN_MONOTONIC_SIGN = {
    "dark_current": -1,
    "photo_current": -1,
    "net_photocurrent": -1,
    "multiplication_gain": -1,
    "gain_threshold_voltage": None,
    "breakdown_voltage": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_APD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-root", type=Path, default=None,
                        help="Reuse compatible frozen predictions after data and split checks.")
    parser.add_argument("--tasks", nargs="+", choices=list(APD_TASKS), default=list(DEFAULT_TASKS))
    parser.add_argument("--models", nargs="+", choices=list(EXPANDED_MODELS), default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument(
        "--split-fractions",
        nargs=4,
        type=float,
        default=[0.65, 0.10, 0.15, 0.10],
        metavar=("TRAIN", "VALIDATION", "CALIBRATION", "TEST"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--train-mc-samples", type=int, default=3)
    parser.add_argument("--validation-mc-samples", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--autopinn-epochs", type=int, default=300)
    parser.add_argument("--bootstrap-replicates", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    if args.reuse_root is not None:
        args.reuse_root = args.reuse_root.resolve()
        if args.reuse_root == args.output:
            raise ValueError("Use --resume when the reuse root equals the output")
    training_tables_dir(args.data_root)
    shared.validate_split_fractions(args.split_fractions)
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be unique")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if args.epochs < 1 or args.kan_steps < 1 or args.mlp_epochs < 1:
        raise ValueError("Training iteration counts must be positive")
    if args.autopinn_epochs < 10:
        raise ValueError("AutoPINN requires at least 10 epochs")
    if args.bootstrap_replicates < 1000:
        raise ValueError("At least 1000 bootstrap replicates are required")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def shared_spec(task: APDTask) -> shared.TaskSpec:
    return shared.TaskSpec(
        key=task.key,
        name=task.name,
        result_subdir=task.key,
        target_col=task.target_col,
        axis_col=task.axis_col,
        input_candidates=task.input_cols,
        use_log_transform=task.use_log_transform,
        y_bounds=task.y_bounds,
    )


def baseline_spec(task: APDTask) -> BaselineTaskSpec:
    return BaselineTaskSpec(
        name=task.name,
        target_col=task.target_col,
        input_cols=task.input_cols,
        use_log_transform=task.use_log_transform,
        y_bounds=task.y_bounds,
        ylabel=task.ylabel,
        group_cols=("structure_group_id",),
    )


def split_task(frame: pd.DataFrame, task: APDTask, seed: int, fractions):
    spec = shared_spec(task)
    partitions = shared.split_task_dataframe(frame, spec, seed, fractions)
    rows = []
    for split_name, part in zip(
        ("train", "validation", "calibration", "test"), partitions
    ):
        for group_id, count in part.groupby("structure_group_id", sort=True).size().items():
            rows.append(
                {
                    "task": task.name,
                    "seed": seed,
                    "split": split_name,
                    "group_id": str(group_id),
                    "n_points": int(count),
                }
            )
    manifest = pd.DataFrame(rows)
    overlap = manifest.groupby("group_id")["split"].nunique()
    if (overlap != 1).any():
        raise RuntimeError(f"Structure-level split leakage for {task.name}, seed={seed}")
    if manifest["group_id"].nunique() != 78:
        raise RuntimeError(f"Incomplete structure split for {task.name}, seed={seed}")
    return partitions, manifest


def prediction_path(output: Path, task: APDTask, seed: int, model: str) -> Path:
    return output / "predictions" / f"seed_{seed}" / task.key / model / "test_predictions.csv"


def metadata_path(output: Path, task: APDTask, seed: int, model: str) -> Path:
    return prediction_path(output, task, seed, model).with_name("run_metadata.json")


def write_prediction(
    path: Path,
    test: pd.DataFrame,
    task: APDTask,
    seed: int,
    model: str,
    prediction: np.ndarray,
    prediction_std: np.ndarray | None = None,
) -> None:
    spec = baseline_spec(task)
    result = test.copy()
    result.insert(0, "model", model)
    result.insert(0, "seed", seed)
    result.insert(0, "task", task.name)
    result["actual_model_space"] = transformed_target(test, spec)
    result["prediction_model_space"] = np.asarray(prediction, dtype=np.float64)
    if prediction_std is not None:
        result["prediction_std_model_space"] = np.asarray(
            prediction_std, dtype=np.float64
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)


def load_prediction(path: Path, test: pd.DataFrame, task: APDTask):
    frame = pd.read_csv(path)
    for column in dict.fromkeys(("structure_group_id", *task.input_cols)):
        observed = frame[column].to_numpy()
        expected_column = test[column].to_numpy()
        matches = (
            np.allclose(observed, expected_column, rtol=1e-10, atol=1e-12)
            if np.issubdtype(expected_column.dtype, np.number)
            else np.array_equal(observed, expected_column)
        ) if observed.shape == expected_column.shape else False
        if not matches:
            raise RuntimeError(f"Stale prediction input {column}: {path}")
    expected = transformed_target(test, baseline_spec(task))
    actual = frame["actual_model_space"].to_numpy(dtype=np.float64)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-7, atol=1e-10):
        raise RuntimeError(f"Stale prediction does not match regenerated split: {path}")
    pred = frame["prediction_model_space"].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise RuntimeError(f"Non-finite prediction: {path}")
    return pred


def validate_reuse(args: argparse.Namespace) -> pd.DataFrame | None:
    if args.reuse_root is None:
        return None
    config = json.loads((args.reuse_root / "config.json").read_text(encoding="utf-8"))
    if config["split_fractions"] != list(args.split_fractions):
        raise ValueError("Reuse split fractions differ")
    table_dir = training_tables_dir(args.data_root)
    for name, digest in config["data_sha256"].items():
        if file_sha256(table_dir / name) != digest:
            raise ValueError(f"Reuse data fingerprint differs: {name}")
    return pd.read_csv(args.reuse_root / "split_manifest.csv")


def train_bkan(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    task: APDTask,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    model_dir: Path,
):
    set_seed(seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    modeler = BayesKANDeviceModeler(
        task_name=task.name,
        results_dir=str(model_dir),
        device=str(device),
    )
    modeler.load_data(
        train,
        input_cols=list(task.input_cols),
        output_col=task.target_col,
        use_log_transform=task.use_log_transform,
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
    start = time.perf_counter()
    with (
        log_path.open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        modeler.train(
            {
                "num_epochs": args.epochs,
                "batch_size": min(args.batch_size, len(train)),
                "lr": args.learning_rate,
                "weight_decay": 1.0e-5,
                "val_freq": 10,
                "train_mc_samples": args.train_mc_samples,
                "validation_mc_samples": args.validation_mc_samples,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "validation_seed": 1729 + seed,
            },
            df_val=validation,
        )
    elapsed = time.perf_counter() - start
    prediction = modeler.predict_with_uncertainty(
        test[list(task.input_cols)].to_numpy(dtype=np.float32)
    )
    checkpoint = model_dir / "model_checkpoint.pt"
    modeler.save_model(str(checkpoint))
    history = modeler.experiment.history
    info = {
        "param_count": int(sum(p.numel() for p in modeler.model.parameters() if p.requires_grad)),
        "train_time_s": elapsed,
        "epochs_completed": len(history.get("train_loss", [])),
        "best_val_loss": float(history.get("best_val_loss", np.nan)),
        "checkpoint": str(checkpoint),
    }
    return (
        np.asarray(prediction["model_mean"], dtype=np.float64),
        np.asarray(prediction["model_std"], dtype=np.float64),
        info,
    )


def model_metrics(
    test: pd.DataFrame,
    task: APDTask,
    prediction: np.ndarray,
) -> dict[str, float]:
    spec = baseline_spec(task)
    y_transformed = transformed_target(test, spec)
    metrics = compute_metrics(
        y_transformed,
        test[task.target_col].to_numpy(dtype=np.float64),
        prediction,
        spec,
    )
    work = pd.DataFrame(
        {
            "structure_group_id": test["structure_group_id"].to_numpy(),
            "squared_error": (np.asarray(prediction) - y_transformed) ** 2,
            "absolute_error": np.abs(np.asarray(prediction) - y_transformed),
        }
    )
    group_rmse = np.sqrt(work.groupby("structure_group_id")["squared_error"].mean())
    group_mae = work.groupby("structure_group_id")["absolute_error"].mean()
    metrics["group_macro_rmse_target"] = float(group_rmse.mean())
    metrics["group_macro_mae_target"] = float(group_mae.mean())
    return metrics


def _holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (len(values) - position))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_statistics(metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(args.bootstrap_seed)
    train_fraction = float(args.split_fractions[0])
    test_fraction = float(args.split_fractions[3])
    for task_name, task_metrics in metrics.groupby("task", sort=False):
        pivot = task_metrics.pivot(index="seed", columns="model", values="rmse_target")
        if "bkan" not in pivot:
            continue
        for baseline in [m for m in args.models if m != "bkan" and m in pivot]:
            values = (pivot["bkan"] - pivot[baseline]).dropna().to_numpy(dtype=np.float64)
            if len(values) < 2:
                continue
            sample_indices = rng.integers(
                0, len(values), size=(args.bootstrap_replicates, len(values))
            )
            bootstrap = values[sample_indices].mean(axis=1)
            variance = float(values.var(ddof=1))
            corrected_se = math.sqrt((1.0 / len(values) + test_fraction / train_fraction) * variance)
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
                    "task": task_name,
                    "baseline": baseline,
                    "n_pairs": len(values),
                    "bkan_rmse_mean": float(pivot["bkan"].mean()),
                    "baseline_rmse_mean": float(pivot[baseline].mean()),
                    "paired_delta_mean": float(values.mean()),
                    "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "corrected_ci95_low": low,
                    "corrected_ci95_high": high,
                    "corrected_p_value_raw": raw_p,
                    "wins": int(np.sum(values < 0.0)),
                    "losses": int(np.sum(values > 0.0)),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["corrected_p_value_holm"] = np.nan
    for _, indices in result.groupby("task", sort=False).groups.items():
        result.loc[indices, "corrected_p_value_holm"] = _holm(
            result.loc[indices, "corrected_p_value_raw"].to_numpy(dtype=np.float64)
        )
    return result


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "rmse_target",
        "mae_target",
        "r2_target",
        "rmse_raw",
        "mae_raw",
        "medape_percent",
        "p95ape_percent",
        "group_macro_rmse_target",
        "group_macro_mae_target",
        "wall_time_s",
        "param_count",
    ]
    summary = metrics.groupby(["task", "model"], sort=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    return summary.reset_index()


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    tasks = list(dict.fromkeys(summary["task"]))
    cols = 3
    rows = math.ceil(len(tasks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.7 * rows), squeeze=False)
    colors = ["#4C78A8", "#F58518", "#B279A2", "#54A24B", "#E45756",
              "#72B7B2", "#FF9DA6", "#9D755D", "#BAB0AC", "#59A14F",
              "#EDC948", "#AF7AA1", "#76B7B2"]
    for ax, task_name in zip(axes.flat, tasks):
        part = summary.loc[summary["task"].eq(task_name)].copy()
        labels = [MODEL_LABELS.get(model, model) for model in part["model"]]
        x = np.arange(len(part))
        ax.bar(
            x,
            part["rmse_target_mean"],
            yerr=part["rmse_target_std"].fillna(0.0),
            color=[colors[EXPANDED_MODELS.index(model)] for model in part["model"]],
            capsize=3,
        )
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Grouped-test RMSE (model space)")
        ax.set_title(task_name.replace("APD_", "APD "))
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.flat[len(tasks) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "apd_second_device_rmse.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "apd_second_device_rmse.pdf", bbox_inches="tight")
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(summary: pd.DataFrame, paired: pd.DataFrame, args: argparse.Namespace) -> None:
    exploratory = [
        MODEL_LABELS[model]
        for model in ("gmls", "autopinn", "curve_lut", "semiempirical")
        if model in args.models
    ]
    lines = [
        f"# APD {len(args.models)} 模型结构分组比较",
        "",
        "以 `structure_group_id` 为独立划分单位，在 APD 数据上训练和测试各模型。",
        f"共 {len(args.tasks)} 个任务、{len(args.models)} 个模型、{len(args.seeds)} 个随机种子。",
        "电流和倍增增益的 RMSE 在 log10 空间计算；两个电压任务的 RMSE 单位为 V。均值 ± 标准差描述不同划分的结果。",
        "全部模型使用相同的训练与测试结构；标准化只拟合训练集。",
        "、".join(exploratory)
        + " 是针对本数据接口的事后方法适配或实现，不等同于原论文源码复现，"
        "其比较按探索性证据解释。",
        f"原有预测来源：`{args.reuse_root}`。" if args.reuse_root else "本次未指定外部预测复用目录。",
        "",
        "## 测试指标",
        "",
        "| Task | Model | RMSE | MAE | R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['task']} | {MODEL_LABELS.get(row['model'], row['model'])} | "
            f"{row['rmse_target_mean']:.6g} ± {row['rmse_target_std']:.3g} | "
            f"{row['mae_target_mean']:.6g} | {row['r2_target_mean']:.6g} |"
        )
    if not paired.empty:
        lines.extend(
            [
                "",
                "## 预先指定的 BKAN-VI 与 Spline-Ridge 比较",
                "",
                "| Task | BKAN RMSE | Spline RMSE | Delta | Corrected 95% CI | Wins/Losses |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in paired.loc[paired["baseline"].eq("spline_ridge")].iterrows():
            lines.append(
                f"| {row['task']} | {row['bkan_rmse_mean']:.6g} | "
                f"{row['baseline_rmse_mean']:.6g} | {row['paired_delta_mean']:.6g} | "
                f"[{row['corrected_ci95_low']:.6g}, {row['corrected_ci95_high']:.6g}] | "
                f"{int(row['wins'])}/{int(row['losses'])} |"
            )
    if not paired.empty:
        lines.extend([
            "", "## BKAN-VI 与全部基线的配对比较", "",
            "Delta = BKAN RMSE − 基线 RMSE；负值表示 BKAN 误差更低。校正区间考虑重复划分的相关性；Holm 校正在每个任务的全部基线比较内进行。", "",
            "| Task | Baseline | Delta | Corrected 95% CI | Holm p | Wins/Losses |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for _, row in paired.iterrows():
            lines.append(
                f"| {row['task']} | {MODEL_LABELS[row['baseline']]} | {row['paired_delta_mean']:.6g} | "
                f"[{row['corrected_ci95_low']:.6g}, {row['corrected_ci95_high']:.6g}] | "
                f"{row['corrected_p_value_holm']:.4g} | {int(row['wins'])}/{int(row['losses'])} |"
            )
    lines.extend(
        [
            "",
            "## 范围",
            "",
            "- 包含暗电流、受光电流、净光电流、倍增增益、增益阈值电压和击穿电压。",
            "- 数据范围不包含 AC、C–V、端电荷、空间电荷及噪声。",
            "- 结果用于评估同一 TCAD 来源下第二类器件的重新训练，不支持零样本迁移、跨工艺或测量验证结论。",
        ]
    )
    (args.output / "apd_second_device_report.md").write_text("\n".join(lines), encoding="utf-8")


def collect_existing(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    rows = []
    manifests = []
    for task_key in args.tasks:
        frame, task = load_apd_task(task_key, args.data_root)
        for seed in args.seeds:
            partitions, manifest = split_task(
                frame, task, seed, tuple(args.split_fractions)
            )
            test = partitions[-1]
            manifests.append(manifest)
            for model in args.models:
                out_path = prediction_path(args.output, task, seed, model)
                meta_path = metadata_path(args.output, task, seed, model)
                if not out_path.is_file() or not meta_path.is_file():
                    raise FileNotFoundError(
                        out_path if not out_path.is_file() else meta_path
                    )
                load_prediction(out_path, test, task)
                row = json.loads(meta_path.read_text(encoding="utf-8"))
                expected = {
                    "task": task.name,
                    "task_key": task.key,
                    "seed": seed,
                    "model": model,
                    "test_rows": len(test),
                }
                observed = {key: row.get(key) for key in expected}
                if observed != expected:
                    raise ValueError(
                        f"Metadata mismatch in {meta_path}: expected {expected}, "
                        f"observed {observed}"
                    )
                rows.append(row)
    return pd.DataFrame(rows), manifests


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    reuse_manifest = validate_reuse(args)
    runtime_device = resolve_device(args.device)
    if args.summary_only:
        metrics, manifests = collect_existing(args)
    else:
        device = runtime_device
        print(f"Training on {device}")
        metric_rows = []
        manifests = []
        for task_key in args.tasks:
            frame, task = load_apd_task(task_key, args.data_root)
            spec = baseline_spec(task)
            for seed in args.seeds:
                partitions, manifest = split_task(
                    frame, task, seed, tuple(args.split_fractions)
                )
                train, validation, calibration, test = partitions
                manifests.append(manifest)
                if reuse_manifest is not None:
                    source_manifest = reuse_manifest.loc[
                        reuse_manifest.task.eq(task.name) & reuse_manifest.seed.eq(seed)
                    ]
                    columns = ["split", "group_id", "n_points"]
                    pd.testing.assert_frame_equal(
                        manifest[columns].sort_values(columns).reset_index(drop=True),
                        source_manifest[columns].sort_values(columns).reset_index(drop=True),
                        check_dtype=False,
                    )
                arrays = make_scaled_arrays(train, test, spec)
                for model in args.models:
                    out_path = prediction_path(args.output, task, seed, model)
                    meta_path = metadata_path(args.output, task, seed, model)
                    external_reused = False
                    if args.reuse_root is not None:
                        source = prediction_path(args.reuse_root, task, seed, model)
                        source_meta = metadata_path(args.reuse_root, task, seed, model)
                        if source.is_file() and source_meta.is_file():
                            load_prediction(source, test, task)
                            source_info = json.loads(source_meta.read_text(encoding="utf-8"))
                            for key, value in {"task": task.name, "task_key": task.key,
                                               "seed": seed, "model": model,
                                               "test_rows": len(test)}.items():
                                if source_info.get(key) != value:
                                    raise ValueError(f"Reuse metadata mismatch: {source_meta}: {key}")
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, out_path)
                            source_info["source_prediction"] = str(source)
                            meta_path.write_text(json.dumps(source_info, indent=2), encoding="utf-8")
                            external_reused = True
                    if ((args.resume or external_reused)
                            and out_path.is_file() and meta_path.is_file()):
                        prediction = load_prediction(out_path, test, task)
                        info = json.loads(meta_path.read_text(encoding="utf-8"))
                        std = None
                        reused = True
                    else:
                        started = time.perf_counter()
                        std = None
                        if model == "bkan":
                            prediction, std, info = train_bkan(
                                train,
                                validation,
                                test,
                                task,
                                seed,
                                device,
                                args,
                                out_path.parent,
                            )
                        elif model == "dkan":
                            log_path = out_path.parent / "training.log"
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            with (
                                log_path.open("w", encoding="utf-8") as log,
                                contextlib.redirect_stdout(log),
                                contextlib.redirect_stderr(log),
                            ):
                                prediction, info = train_dkan(
                                    arrays,
                                    seed,
                                    device,
                                    args.kan_steps,
                                    grid=8,
                                    k=3,
                                    width=8,
                                )
                        elif model in MLP_ARCHES:
                            prediction, info = train_mlp(
                                arrays,
                                MLP_ARCHES[model],
                                seed,
                                device,
                                args.mlp_epochs,
                                0.002,
                                0.001,
                                1.0,
                                2.0,
                            )
                        elif model in ENGINEERING_MODELS:
                            fitted = build_engineering_model(
                                model, len(train), len(task.input_cols), seed
                            )
                            x_train = train[list(task.input_cols)].to_numpy(dtype=np.float64)
                            x_test = test[list(task.input_cols)].to_numpy(dtype=np.float64)
                            fitted.fit(x_train, transformed_target(train, spec))
                            prediction = fitted.predict(x_test).ravel()
                            info = {
                                "param_count": count_engineering_params(
                                    model, fitted, len(train), len(task.input_cols)
                                )
                            }
                        elif model == "gmls":
                            prediction, info = fit_predict_gmls(
                                train, validation, test, spec
                            )
                        elif model == "curve_lut":
                            prediction, info = fit_predict_curve_lut(
                                train,
                                validation,
                                test,
                                spec,
                                axis_col=task.axis_col,
                            )
                        elif model == "semiempirical":
                            prediction, info = fit_predict_semiempirical_compact(
                                train,
                                validation,
                                test,
                                spec,
                                axis_col=task.axis_col,
                            )
                        elif model == "autopinn":
                            prediction, info = train_autopinn_adapted(
                                train,
                                validation,
                                test,
                                spec,
                                axis_col=task.axis_col,
                                monotonic_sign=AUTOPINN_MONOTONIC_SIGN[task.key],
                                seed=seed,
                                device=device,
                                epochs=args.autopinn_epochs,
                            )
                        else:
                            raise ValueError(model)
                        info["wall_time_s"] = time.perf_counter() - started
                        write_prediction(out_path, test, task, seed, model, prediction, std)
                        reused = False

                    values = model_metrics(test, task, prediction)
                    row = {
                        "task": task.name,
                        "task_key": task.key,
                        "primary_task": task.primary,
                        "seed": seed,
                        "model": model,
                        **{
                            key: value
                            for key, value in info.items()
                            if key not in {"task", "task_key", "seed", "model"}
                        },
                        **values,
                        "wall_time_s": float(info.get("wall_time_s", info.get("train_time_s", 0.0))),
                        "reused": reused,
                        "train_groups": int(manifest.loc[manifest.split.eq("train"), "group_id"].nunique()),
                        "validation_groups": int(manifest.loc[manifest.split.eq("validation"), "group_id"].nunique()),
                        "calibration_groups": int(manifest.loc[manifest.split.eq("calibration"), "group_id"].nunique()),
                        "test_groups": int(manifest.loc[manifest.split.eq("test"), "group_id"].nunique()),
                        "train_rows": len(train),
                        "test_rows": len(test),
                    }
                    meta_path.write_text(
                        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    metric_rows.append(row)
                    print(
                        f"[{task.key} seed={seed} {model}] "
                        f"RMSE={values['rmse_target']:.6g} R2={values['r2_target']:.6g}"
                    )
        metrics = pd.DataFrame(metric_rows)

    expected_rows = len(args.tasks) * len(args.seeds) * len(args.models)
    if len(metrics) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} matched metric rows, observed {len(metrics)}"
        )

    summary = summarize(metrics)
    paired = paired_statistics(metrics, args)
    metrics.to_csv(args.output / "metrics_by_seed.csv", index=False)
    summary.to_csv(args.output / "metrics_summary.csv", index=False)
    paired.to_csv(args.output / "paired_statistics.csv", index=False)
    if manifests:
        pd.concat(manifests, ignore_index=True).to_csv(
            args.output / "split_manifest.csv", index=False
        )
    plot_summary(summary, args.output)
    write_report(summary, paired, args)

    table_dir = training_tables_dir(args.data_root)
    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol": (
            "apd_compact_framework_eight_model_exploratory_extension_v1"
            if tuple(args.models) == COMPACT_EIGHT_MODELS
            else "apd_compact_framework_seven_model_curve_lut_exploratory_v1"
            if "curve_lut" in args.models and "semiempirical" not in args.models
            else "apd_compact_framework_compact_baselines_exploratory_v1"
            if {"curve_lut", "semiempirical"} & set(args.models)
            else "apd_compact_framework_exploratory_adaptations_v1"
            if {"gmls", "autopinn"} & set(args.models)
            else "apd_second_device_structure_grouped_retraining_v1"
        ),
        "data_root": str(args.data_root),
        "output": str(args.output),
        "reuse_root": str(args.reuse_root) if args.reuse_root is not None else None,
        "reused_runs": int(metrics["reused"].sum()),
        "command": sys.argv,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "tasks": args.tasks,
        "models": args.models,
        "seeds": args.seeds,
        "split_fractions": args.split_fractions,
        "independent_unit": "structure_group_id",
        "device": str(runtime_device),
        "gpu_name": (
            torch.cuda.get_device_name(runtime_device)
            if runtime_device.type == "cuda"
            else None
        ),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "task_definitions": {key: asdict(APD_TASKS[key]) for key in args.tasks},
        "training": {
            "bkan": {
                "width": 8,
                "grid": 8,
                "spline_order": 3,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "kl_weight": args.kl_weight,
                "train_mc_samples": args.train_mc_samples,
                "validation_mc_samples": args.validation_mc_samples,
                "prediction_mc_samples": args.mc_samples,
            },
            "dkan": {"width": 8, "grid": 8, "spline_order": 3, "steps": args.kan_steps},
            "mlp_l": {"hidden": list(MLP_ARCHES["mlp_l"]), "epochs": args.mlp_epochs},
            "mlp_pm": {"hidden": list(MLP_ARCHES["mlp_pm"]), "epochs": args.mlp_epochs,
                       "lr": 0.002, "lamb": 0.001, "lamb_l1": 1.0, "lamb_entropy": 2.0},
            "poly3_ridge": {"degree": 3, "alpha": 1.0e-6},
            "spline_ridge": {"knots": 6, "degree": 3, "alpha": 1.0e-5},
            "rbf_nystroem": {"components": "min(256, n_train)", "gamma": "1 / n_features", "alpha": 1e-4},
            "random_forest": {"n_estimators": 500, "max_features": 1.0,
                              "min_samples_leaf": 1, "target_standardization": "training only"},
            "xgboost": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
                        "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 1.0,
                        "tree_method": "hist", "target_standardization": "training only"},
            "gmls": {"variant": "validation-selected adapted Gaussian-weighted local polynomial",
                     "candidate_degree_neighbor_factor": [[1, 4], [2, 4], [2, 8]],
                     "neighbors": "min(n_train, max(48, factor * n_basis_terms))",
                     "ridge": 1e-4,
                     "prediction_clip": "training target range plus 10% margin",
                     "input_and_target_standardization": "training only",
                     "selection": "lowest validation-group MSE; test not used"},
            "autopinn": {"variant": "adapted smooth/monotone architecture-search network",
                         "candidates": [[16, 16], [32, 16], [32, 32]],
                         "epochs": args.autopinn_epochs, "activation": "tanh",
                         "smoothness_weight": 1e-5, "monotonic_weight": 1e-2,
                         "monotonic_sign_by_task": AUTOPINN_MONOTONIC_SIGN,
                         "selection": "lowest validation-group MSE; test not used"},
            "curve_lut": {
                "variant": "curve-object PCHIP lookup table with condition-space IDW",
                "candidate_neighbor_curves_and_power": [[1, 1], [2, 1], [4, 1], [4, 2], [8, 2]],
                "axis_extrapolation": "nearest characterized boundary hold",
                "condition_standardization": "training only",
                "selection": "lowest validation-group MSE; test not used",
            },
            "semiempirical": {
                "variant": "ridge-calibrated task-specific compact feature library",
                "candidate_alpha": [1e-6, 1e-4, 1e-2, 1, 100],
                "features": "bias/axis shape, avalanche hinges, relaxation terms, and condition interactions",
                "input_feature_and_target_standardization": "training only",
                "selection": "lowest validation-group MSE; test not used",
            },
        },
        "data_sha256": {
            name: file_sha256(table_dir / name)
            for name in ("apd_dc.csv", "apd_gain.csv", "apd_fom.csv")
        },
        "excluded_tasks": [
            "ac_response",
            "capacitance",
            "terminal_charge",
            "spatial_charge",
            "noise",
        ],
        "claim_boundary": {
            "device_specific_retraining": True,
            "zero_shot_transfer": False,
            "foundry_transfer": False,
            "measurement_validation": False,
        },
    }
    config["training"] = {model: config["training"][model] for model in args.models}
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
