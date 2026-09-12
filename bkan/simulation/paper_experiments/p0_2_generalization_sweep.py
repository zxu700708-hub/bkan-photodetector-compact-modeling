"""
P0-2: Extreme generalization experiment — training data fraction sweep.

Quantifies "how much data is enough" by training on progressively smaller
fractions of the training set and evaluating on the full held-out test set.
Directly supports the "accelerating model development" narrative.

Default run (full):
  python bkan/simulation/paper_experiments/p0_2_generalization_sweep.py

Faster smoke test:
  python bkan/simulation/paper_experiments/p0_2_generalization_sweep.py --seeds 42 --fractions 0.10 0.50 1.00 --models dkan mlp_n --mlp-epochs 50 --kan-steps 50 --bkan-epochs 50
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
import matplotlib.ticker as mticker
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
DEFAULT_CAPACITANCE_DATA = (
    ARTIFACT_RESULTS
    / "apparent_capacitance_correction"
    / "primary_ge_si_apparent_capacitance.csv"
)
DEFAULT_OUT = ARTIFACT_RESULTS / "generalization_sweep"

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


# ── Task definitions (same as P0-1) ────────────────────────────────────────

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
            "dark_voltage", "trap_assisted_recomb_A",
            "ge_sio2_recomb_velocity", "ge_si_recomb_velocity",
            "active_layer_length", "simulation_temperature",
        ),
        use_log_transform=True,
        y_bounds=(-15.0, 5.0),
        ylabel="log10(dark_current)",
    ),
    "I_photo": TaskSpec(
        name="I_photo",
        target_col=NET_PHOTOCURRENT_COLUMN,
        input_cols=(
            "light_voltage", "trap_assisted_recomb_A",
            "ge_sio2_recomb_velocity", "ge_si_recomb_velocity",
            "active_layer_length", "simulation_temperature",
        ),
        use_log_transform=True,
        y_bounds=(-15.0, 5.0),
        ylabel="log10(net_photocurrent)",
    ),
    "AC_Response": TaskSpec(
        name="AC_Response",
        target_col=AC_RESPONSE_DB_COLUMN,
        input_cols=(
            "frequency_ghz", "bandwidth_bias",
            "active_layer_length", "simulation_temperature",
        ),
        use_log_transform=False,
        y_bounds=(-80.0, 10.0),
        ylabel="normalized optical-SSAC contact-current response (dB)",
    ),
    "Capacitance": TaskSpec(
        name="Capacitance",
        target_col="capacitance_F",
        input_cols=(
            "bias_v", "log_frequency_ghz",
            "trap_assisted_recomb_A", "ge_sio2_recomb_velocity",
            "ge_si_recomb_velocity", "active_layer_length",
            "simulation_temperature",
        ),
        use_log_transform=False,
        y_bounds=(-np.inf, np.inf),
        ylabel="capacitance (F)",
    ),
}

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.80, 1.00]

MLP_ARCHES: Dict[str, Tuple[int, ...]] = {
    "mlp_n": (32, 16),
    "mlp_l": (64, 32),
}


# ── Utilities ──────────────────────────────────────────────────────────────

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


def load_capacitance_dataframe(path: Path) -> pd.DataFrame:
    """Load the maintained AC-CV table with the task's log-frequency feature."""
    if not path.is_file():
        raise FileNotFoundError(f"Capacitance data file not found: {path}")
    df = pd.read_csv(path)
    if "frequency_ghz" not in df or "sample_id" not in df:
        raise KeyError("Capacitance data require frequency_ghz and sample_id")
    df = df[df["frequency_ghz"] > 0.0].copy()
    df["log_frequency_ghz"] = np.log10(df["frequency_ghz"].astype(float))
    return df


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
    else:
        med_ape = p95_ape = float("nan")

    return {
        "rmse_target": float(rmse_target),
        "mae_target": float(mae_target),
        "r2_target": float(r2_target),
        "rmse_raw": float(rmse_raw),
        "mae_raw": float(mae_raw),
        "medape_percent": med_ape,
        "p95ape_percent": p95_ape,
    }


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ── Fixed train/test split ─────────────────────────────────────────────────
#  We split 70/30 once per task, then subsample the train portion.

def fixed_split(
    df: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(
        df, test_size=0.30, random_state=seed, shuffle=True
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def subsample_train(
    df_train: pd.DataFrame, fraction: float, sample_seed: int
) -> pd.DataFrame:
    """Randomly subsample `fraction` of the training data."""
    n_total = len(df_train)
    n_sub = max(10, int(n_total * fraction))
    rng = np.random.RandomState(sample_seed)
    indices = rng.choice(n_total, size=n_sub, replace=False)
    return df_train.iloc[indices].reset_index(drop=True)


def make_scaled_arrays(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    spec: TaskSpec,
) -> Dict[str, np.ndarray | StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = df_train[list(spec.input_cols)].to_numpy(dtype=np.float32)
    x_test = df_test[list(spec.input_cols)].to_numpy(dtype=np.float32)

    y_train = transformed_target(df_train, spec).reshape(-1, 1)
    y_test = transformed_target(df_test, spec).reshape(-1, 1)

    x_train_s = x_scaler.fit_transform(x_train).astype(np.float32)
    x_test_s = x_scaler.transform(x_test).astype(np.float32)

    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_test_s = y_scaler.transform(y_test).astype(np.float32)

    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "x_train": x_train_s,
        "x_test": x_test_s,
        "y_train": y_train_s,
        "y_test": y_test_s,
        "y_test_transformed": y_test.reshape(-1),
        "y_test_raw": df_test[spec.target_col].to_numpy(dtype=np.float64),
    }


# ── MLP ────────────────────────────────────────────────────────────────────

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


def train_mlp(
    arrays: Dict,
    hidden: Tuple[int, ...],
    seed: int,
    device: torch.device,
    epochs: int,
    lr: float,
    lamb: float,
    lamb_l1: float,
    lamb_entropy: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    set_seed(seed)
    model = TorchMLP(arrays["x_train"].shape[1], hidden, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    x_train_t = torch.from_numpy(arrays["x_train"]).float().to(device)
    y_train_t = torch.from_numpy(arrays["y_train"]).float().to(device)
    x_test_t = torch.from_numpy(arrays["x_test"]).float().to(device)

    start = time.perf_counter()

    def _reg_loss() -> torch.Tensor:
        reg = torch.tensor(0.0, device=device)
        for p in model.parameters():
            if p.requires_grad and p.ndim >= 2:
                w_abs = torch.abs(p)
                reg += lamb_l1 * torch.sum(w_abs)
                p_row = w_abs / (torch.sum(w_abs, dim=1, keepdim=True) + 1e-4)
                entropy_row = -torch.mean(
                    torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1)
                )
                reg += lamb_entropy * entropy_row
        return reg

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t) + lamb * _reg_loss()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_test_t).detach().cpu().numpy()

    y_scaler: StandardScaler = arrays["y_scaler"]
    y_pred = y_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    elapsed = time.perf_counter() - start
    info = {
        "param_count": count_parameters(model),
        "train_time_s": elapsed,
    }
    return y_pred, info


# ── D-KAN ──────────────────────────────────────────────────────────────────

def train_dkan(
    arrays: Dict,
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
        width=[arrays["x_train"].shape[1], width, 1],
        grid=grid,
        k=k,
        seed=seed,
        device=device,
        auto_save=False,
    )

    dataset = {
        "train_input": torch.from_numpy(arrays["x_train"]).float().to(device),
        "train_label": torch.from_numpy(arrays["y_train"]).float().to(device),
        "test_input": torch.from_numpy(arrays["x_test"]).float().to(device),
        "test_label": torch.from_numpy(arrays["y_test"]).float().to(device),
    }
    batch_size = min(256, int(dataset["train_input"].shape[0]),
                     int(dataset["test_input"].shape[0]))

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

    x_test_t = torch.from_numpy(arrays["x_test"]).float().to(device)
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_test_t)
        if isinstance(y_pred_scaled, tuple):
            y_pred_scaled = y_pred_scaled[0]
        y_pred_scaled_np = y_pred_scaled.detach().cpu().numpy()
    y_scaler: StandardScaler = arrays["y_scaler"]
    y_pred = y_scaler.inverse_transform(y_pred_scaled_np).reshape(-1)
    info = {
        "param_count": count_parameters(model),
        "train_time_s": elapsed,
    }
    return y_pred, info


# ── B-KAN ──────────────────────────────────────────────────────────────────

def train_bkan(
    df_train: pd.DataFrame,
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
    run_tag: str,
    batch_size: int = 256,
    early_stopping_patience: int = 7,
) -> Tuple[np.ndarray, Dict[str, float]]:
    set_seed(seed)
    from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler

    save_dir = output_dir / "_bkan_runs" / spec.name / run_tag
    modeler = BayesKANDeviceModeler(
        task_name=f"{spec.name}_B_KAN_{run_tag}",
        results_dir=str(save_dir),
        device=str(device),
    )
    # Keep checkpoint selection data outside the gradient-fitting subset.
    df_tr, df_val = train_test_split(
        df_train, test_size=0.15, random_state=seed, shuffle=True
    )
    modeler.load_data(
        df_tr.reset_index(drop=True),
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
        "batch_size": batch_size,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "val_freq": max(10, epochs // 10),
        "early_stopping_patience": early_stopping_patience,
    }
    start = time.perf_counter()
    modeler.build_model(bayes_config)
    modeler.train(train_config, df_val=df_val.reset_index(drop=True))
    elapsed = time.perf_counter() - start

    x_test = df_test[list(spec.input_cols)].to_numpy(dtype=np.float32)
    pred = modeler.predict_with_uncertainty(x_test)
    y_pred = pred["log_mean"] if spec.use_log_transform else pred["mean"]
    info = {
        "param_count": count_parameters(modeler.model),
        "train_time_s": elapsed,
    }
    return y_pred.reshape(-1), info


# ── Plotting ───────────────────────────────────────────────────────────────

def save_generalization_curves(
    summary_df: pd.DataFrame, output_dir: Path
) -> None:
    """Accuracy vs training data fraction curves, one panel per task."""
    tasks = sorted(summary_df["task"].drop_duplicates())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.8 * len(tasks), 4.8),
                             squeeze=False)

    model_order = ["dkan", "bkan", "mlp_n", "mlp_l"]
    model_colors = {
        "dkan":   "#2196F3",  # blue
        "bkan":   "#4CAF50",  # green
        "mlp_n":  "#FF9800",  # orange
        "mlp_l":  "#9C27B0",  # purple
    }
    model_markers = {
        "dkan":   "o",
        "bkan":   "s",
        "mlp_n":  "D",
        "mlp_l":  "^",
    }

    for ax, task in zip(axes.flatten(), tasks):
        sub = summary_df[summary_df["task"] == task].copy()
        models_present = [m for m in model_order if m in sub["model"].values]

        for model in models_present:
            model_sub = sub[sub["model"] == model].sort_values("fraction")
            fractions = model_sub["fraction"].values
            rmse_mean = model_sub["rmse_target_mean"].values
            rmse_std = model_sub["rmse_target_std"].values

            color = model_colors.get(model, "#888888")
            marker = model_markers.get(model, "x")
            ax.errorbar(
                fractions, rmse_mean, yerr=rmse_std,
                color=color, marker=marker, markersize=7,
                linewidth=1.6, capsize=4, capthick=1.2,
                label=f"{model} ($\\pm$1 std)",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training data fraction", fontsize=11)
        ax.set_ylabel("RMSE (target space)", fontsize=11)
        ax.set_title(task, fontweight="bold", fontsize=12)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.1%}"
        ))

    fig.suptitle("P0-2: Generalization — RMSE vs Training Data Fraction",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        output_dir / "generalization_rmse_vs_fraction.png",
        dpi=250, bbox_inches="tight",
    )
    plt.close(fig)


def save_r2_curves(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """R² vs training data fraction curves."""
    tasks = sorted(summary_df["task"].drop_duplicates())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.8 * len(tasks), 4.8),
                             squeeze=False)

    model_order = ["dkan", "bkan", "mlp_n", "mlp_l"]
    model_colors = {
        "dkan":   "#2196F3",
        "bkan":   "#4CAF50",
        "mlp_n":  "#FF9800",
        "mlp_l":  "#9C27B0",
    }
    model_markers = {
        "dkan":   "o",
        "bkan":   "s",
        "mlp_n":  "D",
        "mlp_l":  "^",
    }

    for ax, task in zip(axes.flatten(), tasks):
        sub = summary_df[summary_df["task"] == task].copy()
        models_present = [m for m in model_order if m in sub["model"].values]

        for model in models_present:
            model_sub = sub[sub["model"] == model].sort_values("fraction")
            fractions = model_sub["fraction"].values
            r2_mean = model_sub["r2_target_mean"].values
            r2_std = model_sub["r2_target_std"].values

            color = model_colors.get(model, "#888888")
            marker = model_markers.get(model, "x")
            ax.errorbar(
                fractions, r2_mean, yerr=r2_std,
                color=color, marker=marker, markersize=7,
                linewidth=1.6, capsize=4, capthick=1.2,
                label=f"{model} ($\\pm$1 std)",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training data fraction", fontsize=11)
        ax.set_ylabel("$R^2$ (target space)", fontsize=11)
        ax.set_title(task, fontweight="bold", fontsize=12)
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.1%}"
        ))

    fig.suptitle("P0-2: Generalization — $R^2$ vs Training Data Fraction",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        output_dir / "generalization_r2_vs_fraction.png",
        dpi=250, bbox_inches="tight",
    )
    plt.close(fig)


def save_kane_vs_mlp_advantage_plot(
    summary_df: pd.DataFrame, output_dir: Path
) -> None:
    """KAN advantage (RMSE ratio MLP/KAN) vs data fraction."""
    tasks = sorted(summary_df["task"].drop_duplicates())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.8 * len(tasks), 4.2),
                             squeeze=False)

    for ax, task in zip(axes.flatten(), tasks):
        sub = summary_df[summary_df["task"] == task].copy()
        fractions = sorted(sub["fraction"].drop_duplicates())

        ratios_dkan = []
        ratios_bkan = []
        for f in fractions:
            f_sub = sub[sub["fraction"] == f]
            mlp_rmse = f_sub[f_sub["model"] == "mlp_n"]["rmse_target_mean"].values
            dkan_rmse = f_sub[f_sub["model"] == "dkan"]["rmse_target_mean"].values
            bkan_rmse = f_sub[f_sub["model"] == "bkan"]["rmse_target_mean"].values
            if len(mlp_rmse) and len(dkan_rmse):
                ratios_dkan.append(float(mlp_rmse[0] / max(dkan_rmse[0], 1e-10)))
            else:
                ratios_dkan.append(float("nan"))
            if len(mlp_rmse) and len(bkan_rmse):
                ratios_bkan.append(float(mlp_rmse[0] / max(bkan_rmse[0], 1e-10)))
            else:
                ratios_bkan.append(float("nan"))

        ax.plot(fractions, ratios_dkan, "o-", color="#2196F3", linewidth=1.6,
                markersize=7, label="MLP / D-KAN")
        ax.plot(fractions, ratios_bkan, "s-", color="#4CAF50", linewidth=1.6,
                markersize=7, label="MLP / B-KAN")
        ax.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Training data fraction", fontsize=11)
        ax.set_ylabel("RMSE ratio (MLP / KAN)", fontsize=11)
        ax.set_title(task, fontweight="bold", fontsize=12)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.1%}"
        ))

    fig.suptitle("P0-2: KAN Advantage — RMSE Ratio (MLP$_{n}$ / KAN) vs Data Fraction",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        output_dir / "generalization_kan_advantage.png",
        dpi=250, bbox_inches="tight",
    )
    plt.close(fig)


def write_report(
    summary_df: pd.DataFrame, metrics_df: pd.DataFrame,
    args: argparse.Namespace, output_dir: Path,
) -> None:
    def _md_table(df: pd.DataFrame) -> str:
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
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    values.append(f"{float(val):.6g}")
                else:
                    values.append(str(val))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# P0-2 Generalization Sweep — Training Data Fraction Experiment",
        "",
        "## Purpose",
        "",
        "Quantify how much training data is required for KAN vs MLP to reach "
        "a given accuracy level. Directly supports the \"accelerating model "
        "development\" narrative: KAN needs less data.",
        "",
        "## Configuration",
        "",
        f"- Data: `{args.data}`",
        f"- Tasks: `{', '.join(args.tasks)}`",
        f"- Models: `{', '.join(args.models)}`",
        f"- Fractions: `{', '.join(str(f) for f in args.fractions)}`",
        f"- Sampling seeds per fraction: `{', '.join(map(str, args.seeds))}`",
        f"- MLP epochs: `{args.mlp_epochs}`, lr=`{args.mlp_lr}`",
        f"- D-KAN steps: `{args.kan_steps}`, grid=8, k=3, width=8",
        f"- B-KAN epochs: `{args.bkan_epochs}`, MC=`{args.bkan_mc_samples}`",
        "",
        "## Summary Table (mean ± std across sampling seeds)",
        "",
    ]

    display_cols = [
        "task", "model", "fraction", "n_seeds",
        "rmse_target_mean", "rmse_target_std",
        "r2_target_mean", "r2_target_std",
        "mae_target_mean",
        "n_train_mean",
    ]
    existing = [c for c in display_cols if c in summary_df.columns]
    if not summary_df.empty:
        lines.append(_md_table(summary_df[existing].sort_values(
            ["task", "model", "fraction"]
        )))
    else:
        lines.append("_No results generated._")

    # Per-task key findings
    lines.extend(["", "## Key Findings", ""])
    for task in sorted(summary_df["task"].drop_duplicates()):
        sub = summary_df[summary_df["task"] == task]
        lines.append(f"### {task}")
        for model in ["dkan", "bkan", "mlp_n", "mlp_l"]:
            msub = sub[sub["model"] == model].sort_values("fraction")
            if msub.empty:
                continue
            full = msub[msub["fraction"] == 1.0]
            low = msub[msub["fraction"] == 0.01]
            if not full.empty and not low.empty:
                r2_full = float(full["r2_target_mean"].iloc[0])
                r2_1pct = float(low["r2_target_mean"].iloc[0])
                rmse_full = float(full["rmse_target_mean"].iloc[0])
                rmse_1pct = float(low["rmse_target_mean"].iloc[0])
                lines.append(
                    f"- **{model}**: 1% data R²={r2_1pct:.4f} → 100% data "
                    f"R²={r2_full:.4f}, RMSE {rmse_1pct:.4f} → {rmse_full:.4f}"
                )

    lines.extend([
        "",
        "## Output Files",
        "",
        "- `generalization_rmse_vs_fraction.png` — RMSE vs data fraction curves",
        "- `generalization_r2_vs_fraction.png` — R² vs data fraction curves",
        "- `generalization_kan_advantage.png` — KAN/MLP RMSE ratio",
        "- `metrics_by_run.csv` — Per-run raw metrics",
        "- `metrics_summary.csv` — Aggregated (mean ± std across sampling seeds)",
        "",
        "## Notes",
        "",
        "- Train/test split is fixed at 70/30 (seed=42) across all fractions.",
        "- Each fraction randomly subsamples the training set `n_seeds` times.",
        "- Test set is always the full 30% held-out data.",
        "- `rmse_target` is in the transformed target space (log10 for DC tasks).",
    ])
    (output_dir / "p0_2_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Summarize ──────────────────────────────────────────────────────────────

def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse_target", "mae_target", "r2_target",
        "rmse_raw", "mae_raw", "medape_percent", "p95ape_percent",
        "param_count", "train_time_s", "n_train",
    ]
    rows = []
    grouped = metrics_df.groupby(["task", "model", "fraction"], sort=False)
    for (task, model, fraction), group in grouped:
        row = {
            "task": task, "model": model, "fraction": fraction,
            "n_seeds": int(group["seed"].nunique()),
            "n_train_mean": float(group["n_train"].mean()),
        }
        for col in metric_cols:
            if col in group.columns:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P0-2 Generalization sweep — training data fraction experiment."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--capacitance-data", type=Path, default=DEFAULT_CAPACITANCE_DATA,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--tasks", nargs="+",
        default=["I_dark", "I_photo", "AC_Response"],
        choices=list(TASKS),
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["dkan", "bkan", "mlp_n", "mlp_l"],
        choices=["dkan", "bkan", "mlp_n", "mlp_l"],
    )
    parser.add_argument(
        "--fractions", nargs="+", type=float,
        default=FRACTIONS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="auto")
    # MLP
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-lr", type=float, default=0.002)
    parser.add_argument("--mlp-lamb", type=float, default=0.001)
    parser.add_argument("--mlp-lamb-l1", type=float, default=1.0)
    parser.add_argument("--mlp-lamb-entropy", type=float, default=2.0)
    # D-KAN
    parser.add_argument("--kan-steps", type=int, default=300)
    # B-KAN
    parser.add_argument("--bkan-epochs", type=int, default=300)
    parser.add_argument("--bkan-mc-samples", type=int, default=50)
    parser.add_argument("--bkan-batch-size", type=int, default=256)
    parser.add_argument("--bkan-early-stopping-patience", type=int, default=7)
    # Misc
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-bkan", action="store_true",
                        help="Skip BKAN (slowest model)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TQDM_DISABLE", "1")
    device = resolve_device(args.device)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    config_payload = {
        "data": str(args.data),
        "tasks": args.tasks,
        "models": args.models,
        "fractions": args.fractions,
        "seeds": args.seeds,
        "device": str(device),
        "mlp_epochs": args.mlp_epochs,
        "mlp_lr": args.mlp_lr,
        "kan_steps": args.kan_steps,
        "bkan_epochs": args.bkan_epochs,
        "bkan_mc_samples": args.bkan_mc_samples,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    metrics_rows: List[Dict] = []

    KAN_GRID, KAN_K, KAN_WIDTH = 8, 3, 8

    for task_name in args.tasks:
        spec = TASKS[task_name]
        df_raw = (
            load_capacitance_dataframe(args.capacitance_data)
            if task_name == "Capacitance"
            else load_dataframe(args.data)
        )
        df_task = prepare_task_dataframe(df_raw, spec)
        print(f"\n[{spec.name}] total valid rows: {len(df_task)}")

        # Fixed 70/30 train/test split (same across all fractions)
        df_train_full, df_test = fixed_split(df_task, seed=42)
        print(f"  Fixed split: train={len(df_train_full)}, test={len(df_test)}")

        for fraction in args.fractions:
            n_expected = max(10, int(len(df_train_full) * fraction))
            print(f"\n  fraction={fraction:.2f} (n_train≈{n_expected})")

            for sample_seed in args.seeds:
                print(f"    sample_seed={sample_seed}", end="", flush=True)

                # Subsample training data
                df_train_sub = subsample_train(df_train_full, fraction, sample_seed)
                arrays = make_scaled_arrays(df_train_sub, df_test, spec)
                y_true_target = arrays["y_test_transformed"]
                y_true_raw = arrays["y_test_raw"]

                for model_name in args.models:
                    if model_name == "bkan" and args.skip_bkan:
                        continue

                    run_start = time.perf_counter()
                    run_tag = f"frac_{fraction:.2f}_seed_{sample_seed}"

                    if model_name in MLP_ARCHES:
                        y_pred, info = train_mlp(
                            arrays, MLP_ARCHES[model_name],
                            seed=sample_seed, device=device,
                            epochs=args.mlp_epochs, lr=args.mlp_lr,
                            lamb=args.mlp_lamb,
                            lamb_l1=args.mlp_lamb_l1,
                            lamb_entropy=args.mlp_lamb_entropy,
                        )
                    elif model_name == "dkan":
                        y_pred, info = train_dkan(
                            arrays, seed=sample_seed, device=device,
                            steps=args.kan_steps, grid=KAN_GRID,
                            k=KAN_K, width=KAN_WIDTH,
                        )
                    elif model_name == "bkan":
                        y_pred, info = train_bkan(
                            df_train_sub, df_test, spec,
                            seed=sample_seed, device=device,
                            output_dir=output_dir,
                            epochs=args.bkan_epochs,
                            mc_samples=args.bkan_mc_samples,
                            grid=KAN_GRID, k=KAN_K, width=KAN_WIDTH,
                            run_tag=run_tag,
                            batch_size=args.bkan_batch_size,
                            early_stopping_patience=(
                                args.bkan_early_stopping_patience
                            ),
                        )
                    else:
                        raise ValueError(f"Unknown model: {model_name}")

                    metrics = compute_metrics(
                        y_true_target, y_true_raw, y_pred, spec,
                    )
                    row = {
                        "task": spec.name,
                        "model": model_name,
                        "fraction": fraction,
                        "seed": sample_seed,
                        **metrics,
                        **info,
                        "wall_time_s": time.perf_counter() - run_start,
                        "n_train": len(df_train_sub),
                        "n_test": len(df_test),
                    }
                    metrics_rows.append(row)
                    print(
                        f" [{model_name}] "
                        f"RMSE={metrics['rmse_target']:.5f}, "
                        f"R2={metrics['r2_target']:.5f}",
                        end="", flush=True,
                    )
                print()  # newline after all models for this seed

    # ── Save outputs ───────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = summarize_metrics(metrics_df)

    metrics_df.to_csv(output_dir / "metrics_by_run.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    save_generalization_curves(summary_df, output_dir)
    save_r2_curves(summary_df, output_dir)
    save_kane_vs_mlp_advantage_plot(summary_df, output_dir)
    write_report(summary_df, metrics_df, args, output_dir)

    print(f"\nSaved results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
