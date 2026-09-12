"""
P0-1 baseline comparison: MLP vs KAN for compact device modeling.

This script is intentionally self-contained and paper-oriented:
  - fixed train/validation/test split per seed,
  - identical preprocessing for all deterministic models,
  - multiple MLP capacities,
  - optional deterministic KAN and Bayesian KAN baselines,
  - CSV summaries and publication-ready sanity plots.

Default run:
  python bkan/simulation/paper_experiments/p0_1_mlp_baseline.py

Faster smoke test:
  python bkan/simulation/paper_experiments/p0_1_mlp_baseline.py --seeds 42 --mlp-epochs 50 --models mlp_s mlp_m

Fuller comparison:
  python bkan/simulation/paper_experiments/p0_1_mlp_baseline.py --models mlp_s mlp_m mlp_l dkan bkan
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = PROJECT_ROOT.parent
ARTIFACT_RESULTS = REPO_ROOT / "artifacts" / "results"
DEFAULT_DATA = ARTIFACT_RESULTS / "device_modeling" / "cleaned_data.csv"
DEFAULT_OUT = ARTIFACT_RESULTS / "mlp_baseline_newdata"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from device_modeling.photodetector.data_ingestion import (  # noqa: E402
    AC_RESPONSE_DB_COLUMN,
    canonicalize_model_columns,
)
from device_modeling.photodetector.task_config import (  # noqa: E402
    NET_PHOTOCURRENT_COLUMN,
    apply_dark_current_floor,
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_col: str
    input_cols: Tuple[str, ...]
    use_log_transform: bool
    y_bounds: Tuple[float, float]
    ylabel: str


TASKS: Dict[str, TaskSpec] = {
    "I_dark": TaskSpec(
        name="I_dark",
        target_col="dark_current",
        input_cols=(
            "dark_voltage",
            "trap_assisted_recomb_A",
            "ge_sio2_recomb_velocity",
            "ge_si_recomb_velocity",
            "active_layer_length",
            "simulation_temperature",
        ),
        use_log_transform=True,
        y_bounds=(-15.0, 5.0),
        ylabel="log10(dark_current)",
    ),
    "I_photo": TaskSpec(
        name="I_photo",
        target_col=NET_PHOTOCURRENT_COLUMN,
        input_cols=(
            "light_voltage",
            "trap_assisted_recomb_A",
            "ge_sio2_recomb_velocity",
            "ge_si_recomb_velocity",
            "active_layer_length",
            "simulation_temperature",
        ),
        use_log_transform=True,
        y_bounds=(-15.0, 5.0),
        ylabel="log10(net_photocurrent)",
    ),
    "AC_Response": TaskSpec(
        name="AC_Response",
        target_col=AC_RESPONSE_DB_COLUMN,
        input_cols=(
            "frequency_ghz",
            "bandwidth_bias",
            "active_layer_length",
            "simulation_temperature",
        ),
        use_log_transform=False,
        y_bounds=(-80.0, 10.0),
        ylabel="normalized optical-SSAC contact-current response (dB)",
    ),
}


MLP_ARCHES: Dict[str, Tuple[int, ...]] = {
    "mlp_s": (16, 8),           # ~257p (≈ KAN width=3)
    "mlp_m": (24, 12),          # ~481p (≈ KAN width=5)
    "mlp_n": (32, 16),          # ~817p (≈ KAN width=8, 952p — matched)
    "mlp_l": (64, 32),          # ~2,561p
}

KAN_ARCHES: Dict[str, int] = {
    "dkan": 8,                   # ~952p
    "dkan_w": 14,                # ~1,666p (matches BKAN's ~1,664p)
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return apply_dark_current_floor(canonicalize_model_columns(df))


def prepare_task_dataframe(df_raw: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    required = list(spec.input_cols) + [spec.target_col]
    missing = [col for col in required if col not in df_raw.columns]
    if missing:
        raise KeyError(f"[{spec.name}] Missing columns: {missing}")

    df = df_raw[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    if spec.use_log_transform:
        df = df[df[spec.target_col] > 0.0].copy()
        y_trans = np.log10(np.maximum(df[spec.target_col].to_numpy(float), 1e-30))
    else:
        y_trans = df[spec.target_col].to_numpy(float)

    valid = (y_trans > spec.y_bounds[0]) & (y_trans < spec.y_bounds[1])
    df = df.loc[valid].copy()
    if len(df) < 30:
        raise ValueError(f"[{spec.name}] Too few valid rows after cleaning: {len(df)}")
    return df.reset_index(drop=True)


def transformed_target(df: pd.DataFrame, spec: TaskSpec) -> np.ndarray:
    y_raw = df[spec.target_col].to_numpy(dtype=np.float64)
    if spec.use_log_transform:
        return np.log10(np.maximum(y_raw, 1e-30))
    return y_raw


def split_task(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train, df_rest = train_test_split(df, test_size=0.30, random_state=seed, shuffle=True)
    df_val, df_test = train_test_split(df_rest, test_size=0.50, random_state=seed, shuffle=True)
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def make_scaled_arrays(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    spec: TaskSpec,
) -> Dict[str, np.ndarray | StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = df_train[list(spec.input_cols)].to_numpy(dtype=np.float32)
    x_val = df_val[list(spec.input_cols)].to_numpy(dtype=np.float32)
    x_test = df_test[list(spec.input_cols)].to_numpy(dtype=np.float32)

    y_train = transformed_target(df_train, spec).reshape(-1, 1)
    y_val = transformed_target(df_val, spec).reshape(-1, 1)
    y_test = transformed_target(df_test, spec).reshape(-1, 1)

    x_train_s = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = x_scaler.transform(x_val).astype(np.float32)
    x_test_s = x_scaler.transform(x_test).astype(np.float32)

    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)
    y_test_s = y_scaler.transform(y_test).astype(np.float32)

    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "x_train": x_train_s,
        "x_val": x_val_s,
        "x_test": x_test_s,
        "y_train": y_train_s,
        "y_val": y_val_s,
        "y_test": y_test_s,
        "y_test_transformed": y_test.reshape(-1),
        "y_test_raw": df_test[spec.target_col].to_numpy(dtype=np.float64),
    }


class TorchMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: Iterable[int], seed: int):
        super().__init__()
        torch.manual_seed(seed)
        layers: List[torch.nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers.append(torch.nn.Linear(prev, width))
            layers.append(torch.nn.SiLU())
            prev = width
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_mlp(
    arrays: Dict[str, np.ndarray | StandardScaler],
    hidden: Tuple[int, ...],
    seed: int,
    device: torch.device,
    epochs: int,
    lr: float,
    lamb: float,
    lamb_l1: float,
    lamb_entropy: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Train MLP with same optimizer + regularization as KAN (Adam + L1/L2/entropy)."""
    set_seed(seed)
    model = TorchMLP(arrays["x_train"].shape[1], hidden, seed).to(device)  # type: ignore[index]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    x_train = torch.from_numpy(arrays["x_train"]).float().to(device)  # type: ignore[arg-type]
    y_train = torch.from_numpy(arrays["y_train"]).float().to(device)  # type: ignore[arg-type]
    x_val = torch.from_numpy(arrays["x_val"]).float().to(device)  # type: ignore[arg-type]
    y_val = torch.from_numpy(arrays["y_val"]).float().to(device)  # type: ignore[arg-type]
    x_test = torch.from_numpy(arrays["x_test"]).float().to(device)  # type: ignore[arg-type]

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    start = time.perf_counter()

    def _reg_loss() -> torch.Tensor:
        """L1 + entropy regularization on weights (same spirit as KAN)."""
        reg = torch.tensor(0.0, device=device)
        for p in model.parameters():
            if p.requires_grad and p.ndim >= 2:
                w_abs = torch.abs(p)
                reg += lamb_l1 * torch.sum(w_abs)
                # entropy along input dim
                p_row = w_abs / (torch.sum(w_abs, dim=1, keepdim=True) + 1e-4)
                entropy_row = -torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
                reg += lamb_entropy * entropy_row
        return reg

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train)
        mse = criterion(pred, y_train)
        loss = mse + lamb * _reg_loss()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()

    # 最后取 val_loss 最低的 epoch（记录最后一个 epoch 的验证损失）
    best_val = val_loss
    best_epoch = epoch + 1

    # 取最终模型（无 early stopping，与 KAN 一致）
    with torch.no_grad():
        y_pred_scaled = model(x_test).detach().cpu().numpy()

    y_scaler: StandardScaler = arrays["y_scaler"]  # type: ignore[assignment]
    y_pred = y_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    elapsed = time.perf_counter() - start
    info = {
        "param_count": count_parameters(model),
        "train_time_s": elapsed,
        "best_epoch": float(best_epoch),
        "best_val_mse_scaled": best_val,
    }
    return y_pred, info


def train_dkan(
    arrays: Dict[str, np.ndarray | StandardScaler],
    seed: int,
    device: torch.device,
    steps: int,
    grid: int,
    k: int,
    width: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    set_seed(seed)
    from kan import KAN

    model = KAN(
        width=[arrays["x_train"].shape[1], width, 1],  # type: ignore[index]
        grid=grid,
        k=k,
        seed=seed,
        device=device,
        auto_save=False,
    )

    dataset = {
        "train_input": torch.from_numpy(arrays["x_train"]).float().to(device),  # type: ignore[arg-type]
        "train_label": torch.from_numpy(arrays["y_train"]).float().to(device),  # type: ignore[arg-type]
        "test_input": torch.from_numpy(arrays["x_val"]).float().to(device),  # type: ignore[arg-type]
        "test_label": torch.from_numpy(arrays["y_val"]).float().to(device),  # type: ignore[arg-type]
    }
    batch_size = min(
        256,
        int(dataset["train_input"].shape[0]),
        int(dataset["test_input"].shape[0]),
    )
    start = time.perf_counter()
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
        batch=batch_size,
        log=max(steps + 1, 1),
    )
    elapsed = time.perf_counter() - start

    x_test = torch.from_numpy(arrays["x_test"]).float().to(device)  # type: ignore[arg-type]
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_test)
        if isinstance(y_pred_scaled, tuple):
            y_pred_scaled = y_pred_scaled[0]
        y_pred_scaled_np = y_pred_scaled.detach().cpu().numpy()
    y_scaler: StandardScaler = arrays["y_scaler"]  # type: ignore[assignment]
    y_pred = y_scaler.inverse_transform(y_pred_scaled_np).reshape(-1)
    info = {
        "param_count": count_parameters(model),
        "train_time_s": elapsed,
        "best_epoch": float(steps),
        "best_val_mse_scaled": float("nan"),
    }
    return y_pred, info


def train_bkan(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    spec: TaskSpec,
    seed: int,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    mc_samples: int,
    grid: int,
    k: int,
    width: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    set_seed(seed)
    from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler

    save_dir = output_dir / "_bkan_runs" / spec.name / f"seed_{seed}"
    modeler = BayesKANDeviceModeler(
        task_name=f"{spec.name}_B_KAN_seed_{seed}",
        results_dir=str(save_dir),
        device=str(device),
    )
    modeler.load_data(
        df_train,
        input_cols=list(spec.input_cols),
        output_col=spec.target_col,
        use_log_transform=spec.use_log_transform,
        y_bounds=spec.y_bounds,
    )
    bayes_config = {
        "width": [None, width, 2],
        "seed": seed,
        "grid": grid,
        "k": k,
        "grid_range": [-3, 3],
        "kl_weight": 0.1,
        "num_mc_samples": mc_samples,
        "prior_mu": 0.0,
        "prior_log_sigma": 0.0,
        "posterior_init_sigma": 0.1,
        "likelihood": "gaussian",
    }
    train_config = {
        "num_epochs": epochs,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "val_freq": max(10, epochs // 10),
    }
    start = time.perf_counter()
    modeler.build_model(bayes_config)
    modeler.train(train_config, df_val=df_val)
    elapsed = time.perf_counter() - start

    x_test = df_test[list(spec.input_cols)].to_numpy(dtype=np.float32)
    pred = modeler.predict_with_uncertainty(x_test)
    y_pred = pred["log_mean"] if spec.use_log_transform else pred["mean"]
    info = {
        "param_count": count_parameters(modeler.model),
        "train_time_s": elapsed,
        "best_epoch": float(epochs),
        "best_val_mse_scaled": float("nan"),
        "mean_std_target": float(np.mean(pred["log_std"] if spec.use_log_transform else pred["std"])),
    }
    return y_pred.reshape(-1), info


def transformed_to_raw(y_transformed: np.ndarray, spec: TaskSpec) -> np.ndarray:
    if spec.use_log_transform:
        return np.power(10.0, np.clip(y_transformed, -30.0, 30.0))
    return y_transformed


def compute_metrics(
    y_true_transformed: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_transformed: np.ndarray,
    spec: TaskSpec,
) -> Dict[str, float]:
    y_pred_raw = transformed_to_raw(y_pred_transformed, spec)
    rmse_target = math.sqrt(mean_squared_error(y_true_transformed, y_pred_transformed))
    mae_target = mean_absolute_error(y_true_transformed, y_pred_transformed)
    r2_target = r2_score(y_true_transformed, y_pred_transformed)
    rmse_raw = math.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mae_raw = mean_absolute_error(y_true_raw, y_pred_raw)

    denom = np.maximum(np.abs(y_true_raw), 1e-30)
    ape = np.abs(y_pred_raw - y_true_raw) / denom * 100.0
    finite = np.isfinite(ape)
    if finite.any():
        med_ape = float(np.median(ape[finite]))
        p95_ape = float(np.percentile(ape[finite], 95))
        max_ape = float(np.max(ape[finite]))
    else:
        med_ape = p95_ape = max_ape = float("nan")

    return {
        "rmse_target": float(rmse_target),
        "mae_target": float(mae_target),
        "r2_target": float(r2_target),
        "rmse_raw": float(rmse_raw),
        "mae_raw": float(mae_raw),
        "medape_percent": med_ape,
        "p95ape_percent": p95_ape,
        "maxape_percent": max_ape,
    }


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse_target",
        "mae_target",
        "r2_target",
        "rmse_raw",
        "mae_raw",
        "medape_percent",
        "p95ape_percent",
        "maxape_percent",
        "param_count",
        "train_time_s",
        "best_epoch",
    ]
    rows = []
    grouped = metrics_df.groupby(["task", "model"], sort=False)
    for (task, model), group in grouped:
        row = {"task": task, "model": model, "n_seeds": int(group["seed"].nunique())}
        for col in metric_cols:
            if col in group:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def save_actual_vs_pred_plot(pred_df: pd.DataFrame, spec: TaskSpec, output_dir: Path) -> None:
    if pred_df.empty:
        return
    first_seed = int(pred_df["seed"].min())
    df = pred_df[pred_df["seed"] == first_seed].copy()
    models = list(df["model"].drop_duplicates())
    cols = min(3, len(models))
    rows = int(math.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.0), squeeze=False)
    axes_flat = axes.flatten()
    for ax, model in zip(axes_flat, models):
        sub = df[df["model"] == model]
        ax.scatter(sub["y_true_target"], sub["y_pred_target"], s=14, alpha=0.65)
        lo = min(sub["y_true_target"].min(), sub["y_pred_target"].min())
        hi = max(sub["y_true_target"].max(), sub["y_pred_target"].max())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel(f"TCAD {spec.ylabel}")
        ax.set_ylabel(f"Predicted {spec.ylabel}")
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(models) :]:
        ax.axis("off")
    fig.suptitle(f"{spec.name}: actual vs predicted (seed={first_seed})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{spec.name}_actual_vs_pred.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_rmse_bar_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return
    tasks = list(summary_df["task"].drop_duplicates())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.0), squeeze=False)
    for ax, task in zip(axes.flatten(), tasks):
        sub = summary_df[summary_df["task"] == task].copy()
        x = np.arange(len(sub))
        ax.bar(x, sub["rmse_target_mean"], yerr=sub["rmse_target_std"], capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"], rotation=35, ha="right")
        ax.set_ylabel("RMSE in target space")
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_rmse_summary.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_report(summary_df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    def dataframe_to_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for col in headers:
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.6g}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# P0-1 MLP Baseline Comparison",
        "",
        "## Configuration",
        "",
        f"- Data: `{args.data}`",
        f"- Tasks: `{', '.join(args.tasks)}`",
        f"- Models: `{', '.join(args.models)}`",
        f"- Seeds: `{', '.join(map(str, args.seeds))}`",
        f"- MLP: `{args.mlp_epochs}` epochs, Adam lr=`{args.mlp_lr}`, "
        f"lamb=`{args.mlp_lamb}`, lamb_l1=`{args.mlp_lamb_l1}`, lamb_entropy=`{args.mlp_lamb_entropy}`",
        f"- D-KAN: `{args.kan_steps}` steps, grid=8, k=3, width=8",
        f"- B-KAN: `{args.bkan_epochs}` epochs, MC samples=`{args.bkan_mc_samples}`",
        "",
        "## Summary",
        "",
    ]
    display_cols = [
        "task",
        "model",
        "n_seeds",
        "rmse_target_mean",
        "rmse_target_std",
        "mae_target_mean",
        "r2_target_mean",
        "medape_percent_mean",
        "param_count_mean",
        "train_time_s_mean",
    ]
    existing = [col for col in display_cols if col in summary_df.columns]
    if not summary_df.empty:
        lines.append(dataframe_to_markdown(summary_df[existing]))
    else:
        lines.append("_No results generated._")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `rmse_target` and `mae_target` are computed in the trained target space.",
            "- For DC current tasks the target space is `log10(current)`.",
            "- For AC response the target is `20*log10(|dI(f)|/max_f|dI|)`; it is not a port-derived S parameter.",
            "- `medape_percent` is mainly meaningful for positive-current DC tasks.",
        ]
    )
    (output_dir / "p0_1_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0-1 MLP baseline comparison.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tasks", nargs="+", default=["I_dark", "I_photo", "AC_Response"], choices=list(TASKS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp_s", "mlp_m", "mlp_n", "mlp_l", "dkan", "dkan_w", "bkan"],
        choices=list(MLP_ARCHES) + list(KAN_ARCHES) + ["bkan"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="auto")
    # MLP 训练参数 (优化器 = Adam, 正则 = L1+L2+entropy, 与 KAN fit() 对齐)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-lr", type=float, default=0.002,
                        help="Adam LR (same as KAN fit())")
    parser.add_argument("--mlp-lamb", type=float, default=0.001,
                        help="L2 reg weight (same as KAN lamb)")
    parser.add_argument("--mlp-lamb-l1", type=float, default=1.0,
                        help="L1 reg weight (same as KAN lamb_l1)")
    parser.add_argument("--mlp-lamb-entropy", type=float, default=2.0,
                        help="Entropy reg weight (same as KAN lamb_entropy)")
    # KAN / BKAN 训练参数 (沿用项目默认值)
    parser.add_argument("--kan-steps", type=int, default=300)
    parser.add_argument("--bkan-epochs", type=int, default=300)
    parser.add_argument("--bkan-mc-samples", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TQDM_DISABLE", "1")
    device = resolve_device(args.device)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataframe(args.data)
    metrics_rows: List[Dict[str, float | int | str]] = []
    prediction_rows: List[pd.DataFrame] = []

    config_path = output_dir / "config.json"
    config_payload = {
        "data": str(args.data),
        "tasks": args.tasks,
        "models": args.models,
        "seeds": args.seeds,
        "device": str(device),
        "mlp_epochs": args.mlp_epochs,
        "mlp_lr": args.mlp_lr,
        "mlp_lamb": args.mlp_lamb,
        "mlp_lamb_l1": args.mlp_lamb_l1,
        "mlp_lamb_entropy": args.mlp_lamb_entropy,
        "kan_steps": args.kan_steps,
        "bkan_epochs": args.bkan_epochs,
        "bkan_mc_samples": args.bkan_mc_samples,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    for task_name in args.tasks:
        spec = TASKS[task_name]
        df_task = prepare_task_dataframe(df_raw, spec)
        print(f"\n[{spec.name}] valid rows: {len(df_task)}")

        for seed in args.seeds:
            print(f"\n[{spec.name}] seed={seed}")
            df_train, df_val, df_test = split_task(df_task, seed)
            arrays = make_scaled_arrays(df_train, df_val, df_test, spec)
            y_true_target = arrays["y_test_transformed"]  # type: ignore[assignment]
            y_true_raw = arrays["y_test_raw"]  # type: ignore[assignment]

            # KAN / BKAN 沿用项目已有默认配置: grid=8, k=3, width=8
            KAN_GRID, KAN_K, KAN_WIDTH = 8, 3, 8
            for model_name in args.models:
                print(f"  training {model_name} ...", flush=True)
                run_start = time.perf_counter()
                if model_name in MLP_ARCHES:
                    # MLP: Adam + L1/L2/entropy, 与 KAN 的 fit() 正则项对齐
                    y_pred, info = train_mlp(
                        arrays,
                        MLP_ARCHES[model_name],
                        seed=seed,
                        device=device,
                        epochs=args.mlp_epochs,
                        lr=args.mlp_lr,
                        lamb=args.mlp_lamb,
                        lamb_l1=args.mlp_lamb_l1,
                        lamb_entropy=args.mlp_lamb_entropy,
                    )
                elif model_name in KAN_ARCHES:
                    y_pred, info = train_dkan(
                        arrays,
                        seed=seed,
                        device=device,
                        steps=args.kan_steps,
                        grid=KAN_GRID,
                        k=KAN_K,
                        width=KAN_ARCHES[model_name],
                    )
                elif model_name == "bkan":
                    y_pred, info = train_bkan(
                        df_train,
                        df_val,
                        df_test,
                        spec,
                        seed=seed,
                        device=device,
                        output_dir=output_dir,
                        epochs=args.bkan_epochs,
                        mc_samples=args.bkan_mc_samples,
                        grid=KAN_GRID,
                        k=KAN_K,
                        width=KAN_WIDTH,
                    )
                else:
                    raise ValueError(model_name)

                metrics = compute_metrics(
                    y_true_target,  # type: ignore[arg-type]
                    y_true_raw,  # type: ignore[arg-type]
                    y_pred,
                    spec,
                )
                row: Dict[str, float | int | str] = {
                    "task": spec.name,
                    "seed": seed,
                    "model": model_name,
                    **metrics,
                    **info,
                    "wall_time_s": time.perf_counter() - run_start,
                    "n_train": len(df_train),
                    "n_val": len(df_val),
                    "n_test": len(df_test),
                }
                metrics_rows.append(row)
                print(
                    "    "
                    f"RMSE={metrics['rmse_target']:.5f}, "
                    f"MAE={metrics['mae_target']:.5f}, "
                    f"R2={metrics['r2_target']:.5f}, "
                    f"params={int(info['param_count'])}"
                )

                pred_part = pd.DataFrame(
                    {
                        "task": spec.name,
                        "seed": seed,
                        "model": model_name,
                        "y_true_target": y_true_target,  # type: ignore[arg-type]
                        "y_pred_target": y_pred,
                        "y_true_raw": y_true_raw,  # type: ignore[arg-type]
                        "y_pred_raw": transformed_to_raw(y_pred, spec),
                    }
                )
                prediction_rows.append(pred_part)

    metrics_df = pd.DataFrame(metrics_rows)
    pred_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    summary_df = summarize_metrics(metrics_df)

    metrics_df.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    for task_name in args.tasks:
        spec = TASKS[task_name]
        save_actual_vs_pred_plot(pred_df[pred_df["task"] == spec.name], spec, output_dir)
    save_rmse_bar_plot(summary_df, output_dir)
    write_report(summary_df, args, output_dir)

    print(f"\nSaved results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
