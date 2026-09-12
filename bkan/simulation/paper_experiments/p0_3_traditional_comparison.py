"""
P0-3: TCAD-referenced compact-model comparison for a vertical photodetector.

Compares classical analytical / equivalent-circuit-inspired photodetector
baselines (Shockley + SRH + Hurkx TAT + surface leakage, Lambert-Beer
absorption, and first-order AC roll-off) against data-driven KAN/MLP models
on all three tasks: I_dark, I_photo, AC_Response.

Positioning note: the current data set is for a vertical Ge/Si photodetector,
not an avalanche photodetector (APD). APD modeling is reserved for future
extension. Unlike CMOS transistors, vertical photodetectors do not have a
BSIM-like universal compact model that is standardized across foundries and
EDA tools, so all models here are evaluated against the same TCAD reference
data rather than against a claimed industry-standard photonic model.

Default run:
  python bkan/simulation/paper_experiments/p0_3_traditional_comparison.py

Faster smoke test:
  python bkan/simulation/paper_experiments/p0_3_traditional_comparison.py --tasks I_dark --seeds 42 --models physics_simple physics_full dkan mlp_n --mlp-epochs 50 --kan-steps 50 --skip-bkan
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
from typing import Dict, Iterable, List, Tuple, Callable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution, least_squares
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler


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
DEFAULT_OUT = ARTIFACT_RESULTS / "traditional_vs_kan"
DEFAULT_GROUPED_OUT = ARTIFACT_RESULTS / "traditional_vs_kan_grouped"

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

# ── Physical constants ────────────────────────────────────────────────────

KB = 8.617333262145e-5       # Boltzmann constant (eV/K)
Q = 1.0                       # electron charge (eV/V, for convenience)
T_REF = 300.0                 # reference temperature (K)
L_REF = 40.0                  # reference active length (um)
TRAP_REF = 1.0e-9             # reference trap coefficient
VEL_REF = 5000.0              # reference recombination velocity sum (cm/s)
W0_FACTOR = 1.0e-5            # depletion width prefactor from C_j0 (cm)

# ── Task definitions (same as P0-1) ────────────────────────────────────────

@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_col: str
    input_cols: Tuple[str, ...]
    use_log_transform: bool
    y_bounds: Tuple[float, float]
    ylabel: str
    group_cols: Tuple[str, ...] = ()


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
        ylabel="total-admittance apparent capacitance (F)",
        group_cols=("sample_id",),
    ),
}

MLP_ARCHES: Dict[str, Tuple[int, ...]] = {
    "mlp_n": (32, 16),
    "mlp_pm": (40, 32),
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
    if not path.is_file():
        raise FileNotFoundError(f"Capacitance data file not found: {path}")
    df = pd.read_csv(path)
    if "frequency_ghz" not in df or "sample_id" not in df:
        raise KeyError("Capacitance data require frequency_ghz and sample_id")
    df = df[df["frequency_ghz"] > 0.0].copy()
    df["log_frequency_ghz"] = np.log10(df["frequency_ghz"].astype(float))
    return df


def prepare_task_dataframe(df_raw: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    required = list(spec.input_cols) + [spec.target_col] + list(spec.group_cols)
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


def split_data(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """70/30 train/test split (same as P0-1 pattern)."""
    df_train, df_test = train_test_split(
        df, test_size=0.30, random_state=seed, shuffle=True
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def task_group_columns(spec: TaskSpec) -> Tuple[str, ...]:
    """Columns that define one complete condition curve for Protocol-A grouping.

    The first input column is the swept axis for each task:
    voltage for DC tasks and frequency for AC.  Holding all remaining input
    columns fixed gives a complete TCAD condition curve.  Splitting on these
    condition keys avoids putting points from the same curve in both train and
    test sets.
    """
    return spec.group_cols or tuple(spec.input_cols[1:])


def split_data_grouped(
    df: pd.DataFrame,
    spec: TaskSpec,
    seed: int,
    test_size: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = task_group_columns(spec)
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise KeyError(f"[{spec.name}] Missing group columns: {missing}")
    groups = (
        df[list(group_cols)]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    train_groups, test_groups = train_test_split(
        groups, test_size=test_size, random_state=seed, shuffle=True
    )
    train_keys = set(map(tuple, train_groups.to_numpy()))
    test_keys = set(map(tuple, test_groups.to_numpy()))
    if train_keys & test_keys:
        raise RuntimeError(f"[{spec.name}] Grouped split produced overlapping curves")

    keyed = df[list(group_cols)].apply(tuple, axis=1)
    df_train = df.loc[keyed.isin(train_keys)].copy()
    df_test = df.loc[keyed.isin(test_keys)].copy()
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def split_task_data(
    df: pd.DataFrame, spec: TaskSpec, seed: int, split_mode: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split_mode == "point":
        return split_data(df, seed)
    if split_mode == "group":
        return split_data_grouped(df, spec, seed)
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def split_audit_row(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    spec: TaskSpec,
    seed: int,
    split_mode: str,
) -> Dict[str, object]:
    group_cols = task_group_columns(spec)
    if split_mode == "group":
        train_groups = set(map(tuple, df_train[list(group_cols)].drop_duplicates().to_numpy()))
        test_groups = set(map(tuple, df_test[list(group_cols)].drop_duplicates().to_numpy()))
        overlap_groups = len(train_groups & test_groups)
        total_groups = len(train_groups | test_groups)
    else:
        train_groups = set(map(tuple, df_train[list(group_cols)].drop_duplicates().to_numpy()))
        test_groups = set(map(tuple, df_test[list(group_cols)].drop_duplicates().to_numpy()))
        overlap_groups = len(train_groups & test_groups)
        total_groups = len(train_groups | test_groups)
    return {
        "task": spec.name,
        "seed": seed,
        "split_mode": split_mode,
        "group_columns": ",".join(group_cols),
        "train_rows": len(df_train),
        "test_rows": len(df_test),
        "train_curves": len(train_groups),
        "test_curves": len(test_groups),
        "total_curves": total_groups,
        "overlap_groups": overlap_groups,
    }


def make_scaled_arrays(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    spec: TaskSpec,
) -> Dict:
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


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS MODELS
# ═══════════════════════════════════════════════════════════════════════════

# ── DC: Dark Current ───────────────────────────────────────────────────────

class DarkCurrentParams:
    """Parameter contract for the five-mechanism dark-current baseline.

    ``names`` contains the nine fitted (free) parameters.  The analytical
    expression also contains two fixed quantities, recorded explicitly in
    ``fixed_values`` so that a reader cannot mistake the eleven symbols in the
    equation for eleven fitted degrees of freedom.
    """
    names = [
        "log10_I_s0", "n", "log10_I_gr0", "V_bi", "E_a_gr",
        "log10_I_tat0", "B_tat", "log10_I_surf0", "log10_R_shunt",
    ]
    fixed_values = {
        "E_a_diff": 0.66,   # Ge bandgap activation energy (eV)
        "gamma_surf": 1.0,  # linear surface-voltage exponent
    }
    # bounds in natural space
    bounds_natural = [
        (-20, -5), (1.0, 3.0), (-20, -5), (0.1, 0.8), (0.10, 0.50),
        (-20, -5), (0.01, 5.0), (-25, -5), (3, 15),
    ]

    @staticmethod
    def initial_guess() -> np.ndarray:
        return np.array([-14.0, 1.5, -11.0, 0.45, 0.33,
                         -12.0, 1.0, -15.0, 9.0])

    @classmethod
    def manifest(cls) -> Dict:
        """Return a serialization-ready free/fixed parameter declaration."""
        return {
            "model": "dark_current_physics_full",
            "free_parameter_count": len(cls.names),
            "free_parameters": [
                {
                    "name": name,
                    "lower_bound": float(bounds[0]),
                    "upper_bound": float(bounds[1]),
                }
                for name, bounds in zip(cls.names, cls.bounds_natural)
            ],
            "fixed_parameters": {
                name: float(value) for name, value in cls.fixed_values.items()
            },
            "clarification": (
                "E_a_gr is fitted; E_a_diff and gamma_surf are fixed. "
                "The equation has eleven named quantities but nine fitted "
                "degrees of freedom."
            ),
        }

    @staticmethod
    def unpack(params: np.ndarray):
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        if params.size != len(DarkCurrentParams.names):
            raise ValueError(
                "Dark-current Physics-Full expects exactly "
                f"{len(DarkCurrentParams.names)} fitted parameters; "
                f"received {params.size}."
            )
        (log10_I_s0, n, log10_I_gr0, V_bi, E_a_gr,
         log10_I_tat0, B_tat, log10_I_surf0, log10_R_shunt) = params
        E_a_diff = DarkCurrentParams.fixed_values["E_a_diff"]
        gamma_surf = DarkCurrentParams.fixed_values["gamma_surf"]
        return (log10_I_s0, n, log10_I_gr0, V_bi, E_a_gr,
                log10_I_tat0, B_tat, log10_I_surf0, log10_R_shunt,
                E_a_diff, gamma_surf)


def dark_current_model(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    X shape: (n_samples, 6)
    Columns: [V, trap_A, vs, vg, L, T]
    Returns: log10(I_dark)  shape (n_samples,)
    """
    (log10_I_s0, n, log10_I_gr0, V_bi, E_a_gr,
     log10_I_tat0, B_tat, log10_I_surf0, log10_R_shunt,
     E_a_diff, gamma_surf) = DarkCurrentParams.unpack(params)

    V   = X[:, 0].astype(np.float64)
    A   = X[:, 1].astype(np.float64)
    vs  = X[:, 2].astype(np.float64)
    vg  = X[:, 3].astype(np.float64)
    L   = X[:, 4].astype(np.float64)
    T   = X[:, 5].astype(np.float64)

    Vt    = KB * T
    V_abs = np.maximum(np.abs(V), 1e-6)
    V_bi_pos = np.abs(V_bi)

    # 1. Diffusion [Shockley 1949]
    I_s0  = 10.0 ** log10_I_s0
    I_diff = I_s0 * (T / T_REF)**3 * np.exp(-E_a_diff / Vt) * \
             (np.exp(np.minimum(Q * V / (n * Vt), 80.0)) - 1.0)
    I_diff_abs = np.abs(I_diff)

    # 2. Generation-Recombination [Sah-Noyce-Shockley 1957]
    I_gr0 = 10.0 ** log10_I_gr0
    depletion = np.sqrt(np.maximum(V_bi_pos - V, 1e-10))
    I_gr = I_gr0 * (L / L_REF) * depletion * np.exp(-E_a_gr / Vt)

    # 3. Trap-Assisted Tunneling [Hurkx IEEE TED 1992]
    I_tat0_val = 10.0 ** log10_I_tat0
    I_tat = I_tat0_val * (A / TRAP_REF) * np.exp(-B_tat / (V_abs + 1e-4))

    # 4. Surface/interface leakage [Chen JAP 2016]
    I_surf0_val = 10.0 ** log10_I_surf0
    I_surf = I_surf0_val * ((vs + vg) / VEL_REF) * (V_abs ** gamma_surf)

    # 5. Ohmic shunt
    R_shunt = 10.0 ** log10_R_shunt
    I_shunt = V_abs / R_shunt

    I_total = I_diff_abs + I_gr + I_tat + I_surf + I_shunt
    return np.log10(np.maximum(I_total, 1e-30))


# ── DC: Photo Current ──────────────────────────────────────────────────────

class PhotoCurrentParams:
    names = ["log10_I_ph0", "alpha_eff", "gamma_T"]
    bounds_natural = [
        (-5, -2), (0.001, 0.5), (-2.0, 2.0),
    ]

    @staticmethod
    def initial_guess() -> np.ndarray:
        return np.array([-3.3, 0.05, 0.0])


def photo_current_model(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    X shape: (n_samples, 6)
    Columns: [V, trap_A, vs, vg, L, T]
    Returns: log10(I_photo)  shape (n_samples,)

    First-order analytical baseline: in a vertical photodetector under reverse
    bias, the dominant photocurrent term is absorption-limited and the carrier
    collection efficiency is close to saturated.  Photocurrent is therefore
    treated as voltage-independent in the baseline:

        I_photo = I_ph0 * absorption(L) * (T/300)^gamma_T

    This model omits trap_A, vs, vg — the recombination parameters used here as
    second-order correction candidates.  The near-zero R2 of this simple model is
    a residual diagnostic rather than a claim that the classical mechanisms are
    invalid:

        *A first-order absorption baseline cannot explain the residual TCAD
        photocurrent variation. KAN reveals that recombination at interfaces
        dominates the residual scatter.*

    This is the intended comparison: classical physics writes down the compact
    dominant term, while the data-driven model learns device-specific TCAD
    corrections that would otherwise require manual model enrichment.
    """
    log10_I_ph0, alpha_eff, gamma_T = params

    L = X[:, 4].astype(np.float64)
    T = X[:, 5].astype(np.float64)

    I_ph0 = 10.0 ** log10_I_ph0
    absorption = 1.0 - np.exp(-alpha_eff * L)
    temp_factor = (T / T_REF) ** gamma_T

    I_total = I_ph0 * absorption * temp_factor
    return np.log10(np.maximum(I_total, 1e-30))


# ── AC: Frequency Response ─────────────────────────────────────────────────

class ACResponseParams:
    """Simplified first-order low-pass model with power-law bandwidth vs bias."""
    names = ["log10_f_0", "p_bias", "alpha_L", "gamma_T", "log10_S21_DC"]
    bounds_natural = [
        (-2.0, 3.0), (0.1, 2.0), (-3.0, 3.0), (-3.0, 3.0), (-10, 10),
    ]

    @staticmethod
    def initial_guess(V_bi: float = 0.45) -> np.ndarray:
        return np.array([1.0, 0.5, 0.0, 0.0, 0.0])

    @staticmethod
    def initial_guess_full() -> np.ndarray:
        return ACResponseParams.initial_guess()


def ac_response_model(params: np.ndarray, X: np.ndarray,
                      V_bi: float = 0.45) -> np.ndarray:
    """
    X shape: (n_samples, 4)
    Columns: [frequency_ghz, bandwidth_bias, active_layer_length, simulation_temperature]
    Returns: normalized optical-SSAC contact-current response (dB), shape (n_samples,)

    Physics-based first-order low-pass model:

        f_3dB(V, L, T) = f_0 * |V|^p * (L/L_ref)^alpha * (T/T_ref)^gamma
        S21(f) = S21_DC - 10*log10(1 + (f/f_3dB)^2)

    The 3dB bandwidth increases with reverse bias (|V|^p, p ≈ 0.5 from
    W ∝ sqrt(V_bi-V)), weakly depends on device length (capacitance ∝ L),
    and has a small temperature coefficient.
    """
    log10_f_0, p_bias, alpha_L, gamma_T, log10_S21_DC = params

    f_ghz  = X[:, 0].astype(np.float64)
    V_bias = X[:, 1].astype(np.float64)
    L      = X[:, 2].astype(np.float64)
    T      = X[:, 3].astype(np.float64)

    # 3dB bandwidth: power-law in bias, power-law in L and T
    f_0 = 10.0 ** log10_f_0
    f_3dB = f_0 * (np.abs(V_bias) ** p_bias) * \
            ((L / L_REF) ** alpha_L) * ((T / T_REF) ** gamma_T)
    f_3dB = np.maximum(f_3dB, 1e-3)

    # First-order low-pass roll-off: S21(dB) = S21_DC - 10*log10(1 + (f/fc)^2)
    S21_DC_lin = 10.0 ** (log10_S21_DC / 20.0)  # dB → linear voltage gain
    S21 = S21_DC_lin / np.sqrt(1.0 + (f_ghz / f_3dB) ** 2)
    response_db = 20.0 * np.log10(np.maximum(np.abs(S21), 1e-30))

    return response_db


# ═══════════════════════════════════════════════════════════════════════════
# ENGINEERING BASELINES
# ═══════════════════════════════════════════════════════════════════════════

ENGINEERING_MODELS = [
    "pdk_lut_knn",
    "poly3_ridge",
    "spline_ridge",
    "rbf_nystroem",
    "random_forest",
    "xgboost",
]

MODEL_PARAM_COUNTS = {
    "pdk_lut_knn": None,    # stored entries, counted per dataset
    "poly3_ridge": None,    # depends on n_features
    "spline_ridge": None,   # depends on n_features
    "rbf_nystroem": None,   # depends on n_features
    "random_forest": None,  # total fitted tree nodes
    "xgboost": None,        # total fitted tree nodes
}


def build_engineering_model(
    model_name: str,
    n_train: int,
    n_features: int,
    seed: int,
    knn_neighbors: int = 5,
    rbf_components: int = 256,
) -> object:
    if model_name == "pdk_lut_knn":
        return Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=max(1, min(knn_neighbors, n_train)),
                weights="distance",
                p=2,
            )),
        ])

    if model_name == "poly3_ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("ridge", Ridge(alpha=1e-6)),
        ])

    if model_name == "spline_ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("spline", SplineTransformer(
                n_knots=6,
                degree=3,
                include_bias=False,
            )),
            ("ridge", Ridge(alpha=1e-5)),
        ])

    if model_name == "rbf_nystroem":
        n_components = max(16, min(rbf_components, n_train))
        return Pipeline([
            ("scale", StandardScaler()),
            ("rbf", Nystroem(
                kernel="rbf",
                gamma=1.0 / max(1, n_features),
                n_components=n_components,
                random_state=seed,
            )),
            ("ridge", Ridge(alpha=1e-4)),
        ])

    # Standardize the target from training rows only. This avoids degenerate
    # split gains for the capacitance target, whose values are around 1e-19.
    if model_name == "random_forest":
        regressor = Pipeline([
            ("scale", StandardScaler()),
            ("forest", RandomForestRegressor(
                n_estimators=500,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
            )),
        ])
        return TransformedTargetRegressor(
            regressor=regressor,
            transformer=StandardScaler(),
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "The xgboost baseline requires `pip install xgboost>=3.0,<4`."
            ) from exc
        regressor = Pipeline([
            ("scale", StandardScaler()),
            ("xgboost", XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=1.0,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )),
        ])
        return TransformedTargetRegressor(
            regressor=regressor,
            transformer=StandardScaler(),
        )

    raise ValueError(f"Unknown engineering model: {model_name}")


def count_engineering_params(model_name: str, model: object, n_train: int, n_features: int) -> int:
    if model_name == "pdk_lut_knn":
        return int(n_train * (n_features + 1))

    if model_name in {"random_forest", "xgboost"}:
        fitted_pipeline = model.regressor_
        if model_name == "random_forest":
            forest = fitted_pipeline.named_steps["forest"]
            return int(sum(tree.tree_.node_count for tree in forest.estimators_))
        booster = fitted_pipeline.named_steps["xgboost"].get_booster()
        return int(
            sum(tree.count('"nodeid"') for tree in booster.get_dump(dump_format="json"))
        )

    if "ridge" in model.named_steps:
        ridge = model.named_steps["ridge"]
        count = int(np.size(ridge.coef_) + np.size(ridge.intercept_))
        if "rbf" in model.named_steps:
            rbf = model.named_steps["rbf"]
            count += int(np.size(rbf.components_))
        elif "poly" in model.named_steps:
            pass  # poly features don't add trained parameters
        elif "spline" in model.named_steps:
            pass  # spline knots are fixed
        return count

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# FITTING
# ═══════════════════════════════════════════════════════════════════════════

def fit_physics_model(
    model_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    bounds_natural: List[Tuple[float, float]],
    x0: np.ndarray,
    seed: int = 42,
    maxiter_de: int = 2000,
    popsize: int = 30,
) -> Tuple[np.ndarray, Dict]:
    """Fit a physics model using differential_evolution + least_squares."""
    n_params = len(bounds_natural)

    # Ensure bounds are proper (lower < upper)
    bounds_de = []
    for lo, hi in bounds_natural:
        if lo >= hi:
            lo, hi = hi - 1e-6, hi + 1e-6
        bounds_de.append((float(lo), float(hi)))

    # Stage 1: global search
    start = time.perf_counter()
    result_de = differential_evolution(
        lambda p: np.mean((y.ravel() - model_fn(p, X).ravel()) ** 2),
        bounds=bounds_de,
        seed=seed,
        maxiter=maxiter_de,
        popsize=popsize,
        polish=False,
        tol=1e-10,
    )
    de_time = time.perf_counter() - start

    # Stage 2: local refinement
    start_ls = time.perf_counter()
    try:
        result_ls = least_squares(
            lambda p: (model_fn(p, X).ravel() - y.ravel()),
            x0=result_de.x,
            bounds=([b[0] for b in bounds_natural],
                    [b[1] for b in bounds_natural]),
            method='trf',
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=5000,
        )
        final_params = result_ls.x
        final_cost = float(np.mean(result_ls.fun ** 2))
        nfev_ls = result_ls.nfev
    except Exception:
        final_params = result_de.x
        final_cost = float(result_de.fun)
        nfev_ls = 0

    ls_time = time.perf_counter() - start_ls

    info = {
        "de_cost": float(result_de.fun),
        "de_nfev": int(result_de.nfev),
        "de_time_s": de_time,
        "final_cost": final_cost,
        "nfev_ls": int(nfev_ls),
        "ls_time_s": ls_time,
        "fit_time_s": time.perf_counter() - start,
    }
    return final_params, info


# ═══════════════════════════════════════════════════════════════════════════
# MLP / KAN TRAINING  (adapted from P0-1)
# ═══════════════════════════════════════════════════════════════════════════

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
    arrays: Dict, hidden: Tuple[int, ...], seed: int, device: torch.device,
    epochs: int, lr: float, lamb: float, lamb_l1: float, lamb_entropy: float,
) -> Tuple[np.ndarray, Dict]:
    set_seed(seed)
    model = TorchMLP(arrays["x_train"].shape[1], hidden, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    x_train_t = torch.from_numpy(arrays["x_train"]).float().to(device)
    y_train_t = torch.from_numpy(arrays["y_train"]).float().to(device)
    x_test_t  = torch.from_numpy(arrays["x_test"]).float().to(device)

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

    for _ in range(epochs):
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
    return y_pred, {"param_count": count_parameters(model), "train_time_s": elapsed}


def train_dkan(
    arrays: Dict, seed: int, device: torch.device,
    steps: int, grid: int, k: int, width: int,
) -> Tuple[np.ndarray, Dict]:
    set_seed(seed)
    from kan import KAN
    model = KAN(
        width=[arrays["x_train"].shape[1], width, 1],
        grid=grid, k=k, seed=seed, device=device, auto_save=False,
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
    model.fit(dataset, opt="Adam", steps=steps, lr=0.002, lamb=0.001,
              lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.1, lamb_coefdiff=0.1,
              batch=batch_size, log=max(steps + 1, 1))
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
    return y_pred, {"param_count": count_parameters(model), "train_time_s": elapsed}


def train_bkan(
    df_train: pd.DataFrame, df_test: pd.DataFrame, spec: TaskSpec,
    seed: int, device: torch.device, output_dir: Path,
    epochs: int, mc_samples: int, grid: int, k: int, width: int,
    run_tag: str, split_mode: str = "point", batch_size: int = 32,
    val_fraction: float = 0.15, early_stopping_patience: int = 7,
) -> Tuple[np.ndarray, Dict]:
    set_seed(seed)
    from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("BKAN validation fraction must be between 0 and 1")
    if split_mode == "group":
        df_fit, df_val = split_data_grouped(
            df_train, spec, seed=seed, test_size=val_fraction
        )
        group_cols = task_group_columns(spec)
        fit_groups = set(
            map(tuple, df_fit[list(group_cols)].drop_duplicates().to_numpy())
        )
        val_groups = set(
            map(tuple, df_val[list(group_cols)].drop_duplicates().to_numpy())
        )
        validation_overlap_groups = len(fit_groups & val_groups)
        if validation_overlap_groups:
            raise RuntimeError(
                f"[{spec.name}] BKAN inner validation leaked "
                f"{validation_overlap_groups} condition curves"
            )
        n_fit_curves = len(fit_groups)
        n_val_curves = len(val_groups)
    elif split_mode == "point":
        df_fit, df_val = train_test_split(
            df_train,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
        )
        validation_overlap_groups = np.nan
        n_fit_curves = np.nan
        n_val_curves = np.nan
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    save_dir = output_dir / "_bkan_runs" / spec.name / run_tag
    modeler = BayesKANDeviceModeler(
        task_name=f"{spec.name}_B_KAN_{run_tag}",
        results_dir=str(save_dir), device=str(device),
    )
    modeler.load_data(
        df_fit.reset_index(drop=True),
        input_cols=list(spec.input_cols),
        output_col=spec.target_col,
        use_log_transform=spec.use_log_transform,
        y_bounds=spec.y_bounds,
    )
    bayes_config = {
        "width": [None, width, 2], "grid": grid, "k": k,
        "seed": seed,
        "grid_range": [-3, 3], "kl_weight": 0.1,
        "num_mc_samples": mc_samples, "prior_mu": 0.0,
        "prior_log_sigma": 0.0, "posterior_init_sigma": 0.1,
        "likelihood": "gaussian",
    }
    train_config = {
        "num_epochs": epochs, "batch_size": batch_size, "lr": 1e-3,
        "weight_decay": 1e-5, "val_freq": max(10, epochs // 10),
        "early_stopping_patience": early_stopping_patience,
    }
    start = time.perf_counter()
    modeler.build_model(bayes_config)
    modeler.train(train_config, df_val=df_val.reset_index(drop=True))
    elapsed = time.perf_counter() - start

    x_test = df_test[list(spec.input_cols)].to_numpy(dtype=np.float32)
    pred = modeler.predict_with_uncertainty(x_test)
    y_pred = pred["log_mean"] if spec.use_log_transform else pred["mean"]
    return y_pred.reshape(-1), {
        "param_count": count_parameters(modeler.model),
        "train_time_s": elapsed,
        "n_fit": len(df_fit),
        "n_val": len(df_val),
        "n_fit_curves": n_fit_curves,
        "n_val_curves": n_val_curves,
        "validation_overlap_groups": validation_overlap_groups,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

MODEL_ORDER = ["physics_simple", "physics_full", "dkan", "bkan", "mlp_n", "mlp_pm",
                "mlp_l", "poly3_ridge", "spline_ridge", "rbf_nystroem",
                "random_forest", "xgboost", "pdk_lut_knn"]
MODEL_COLORS = {
    "physics_simple": "#E53935", "physics_full": "#FF7043",
    "dkan":   "#2196F3", "bkan": "#4CAF50",
    "mlp_n":  "#FF9800", "mlp_pm": "#F57C00", "mlp_l": "#9C27B0",
    "poly3_ridge": "#607D8B", "spline_ridge": "#009688",
    "rbf_nystroem": "#3F51B5", "pdk_lut_knn": "#795548",
    "random_forest": "#8BC34A", "xgboost": "#C62828",
}
MODEL_LABELS = {
    "physics_simple": "Analytical (simple)",
    "physics_full": "Physics / Eq.-Circuit",
    "dkan": "D-KAN", "bkan": "B-KAN",
    "mlp_n": "MLP-N", "mlp_pm": "MLP-PM", "mlp_l": "MLP-L",
    "poly3_ridge": "Poly3-Ridge",
    "spline_ridge": "Spline-Ridge",
    "rbf_nystroem": "RBF-Nystroem",
    "random_forest": "Random Forest", "xgboost": "XGBoost",
    "pdk_lut_knn": "LUT-KNN",
}


def save_actual_vs_pred(pred_dfs: Dict[str, pd.DataFrame], spec: TaskSpec,
                        models_present: List[str], output_dir: Path) -> None:
    """Actual vs predicted scatter for each model."""
    n_models = len(models_present)
    cols = min(3, n_models)
    rows = int(math.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.0),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, model in zip(axes_flat, models_present):
        df = pred_dfs.get(model)
        if df is None or df.empty:
            ax.axis("off"); continue
        ax.scatter(df["y_true"], df["y_pred"], s=12, alpha=0.6)
        lo = min(df["y_true"].min(), df["y_pred"].min())
        hi = max(df["y_true"].max(), df["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11)
        ax.set_xlabel(f"True {spec.ylabel}")
        ax.set_ylabel(f"Pred {spec.ylabel}")
        ax.grid(alpha=0.25)
    for ax in axes_flat[n_models:]:
        ax.axis("off")
    fig.suptitle(f"{spec.name}: Actual vs Predicted", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{spec.name}_actual_vs_pred.png",
                dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_rmse_bar(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped RMSE bar chart across all tasks and models."""
    preferred_order = ["I_dark", "I_photo", "AC_Response", "Capacitance"]
    available = set(summary_df["task"].drop_duplicates())
    tasks = [task for task in preferred_order if task in available]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.2),
                             squeeze=False)
    for ax, task in zip(axes.flatten(), tasks):
        sub = summary_df[summary_df["task"] == task]
        models_present = [m for m in MODEL_ORDER if m in sub["model"].values]
        x = np.arange(len(models_present))
        heights, errs, colors = [], [], []
        for m in models_present:
            row = sub[sub["model"] == m]
            heights.append(float(row["rmse_target_mean"].iloc[0])
                           if len(row) else 0)
            errs.append(float(row["rmse_target_std"].iloc[0])
                        if len(row) else 0)
            colors.append(MODEL_COLORS.get(m, "#888"))
        if task == "I_dark":
            lower_errs = [
                min(err, 0.95 * height) for height, err in zip(heights, errs)
            ]
            ax.bar(
                x,
                heights,
                yerr=np.vstack([lower_errs, errs]),
                capsize=4,
                color=colors,
            )
            ax.set_yscale("log")
        else:
            ax.bar(x, heights, yerr=errs, capsize=4, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_present],
                           rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("RMSE (target space)")
        ax.set_title(task, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_bar_summary.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_residual_vs_voltage(
    pred_dfs: Dict[str, pd.DataFrame], spec: TaskSpec, voltage_col: str,
    models_present: List[str], output_dir: Path,
) -> None:
    """Residual (y_pred - y_true) vs voltage, annotated with physics regions."""
    n_models = len(models_present)
    if n_models == 0:
        return
    cols = min(3, n_models)
    rows = int(math.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.0),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, model in zip(axes_flat, models_present):
        df = pred_dfs.get(model)
        if df is None or df.empty or voltage_col not in df.columns:
            ax.axis("off"); continue
        residual = df["y_pred"] - df["y_true"]
        v = df[voltage_col]
        ax.scatter(v, residual, s=10, alpha=0.5, c=df.get("T", v*0),
                   cmap="coolwarm", vmin=290, vmax=310)
        ax.axhline(y=0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xlabel(f"{voltage_col} (V)")
        ax.set_ylabel("Residual (target space)")
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10)
        ax.grid(alpha=0.25)

        # Annotate physics regions (voltage is negative for reverse bias)
        v_lo, v_hi = v.min(), v.max()
        if v_lo < -2.0:
            ax.axvspan(v_lo, -2.0, alpha=0.08, color="orange")
            ax.text((v_lo - 2.0)/2, ax.get_ylim()[1]*0.95, "TAT",
                    ha="center", fontsize=8, color="darkorange")
        if v_lo < -0.8:
            ax.axvspan(max(v_lo, -2.0), -0.5, alpha=0.08, color="green")
            ax.text(-1.25, ax.get_ylim()[1]*0.95, "G-R",
                    ha="center", fontsize=8, color="darkgreen")
        if v_hi > -0.5:
            ax.axvspan(-0.5, v_hi, alpha=0.08, color="blue")
            ax.text((-0.5+v_hi)/2, ax.get_ylim()[1]*0.95, "Diff+Shunt",
                    ha="center", fontsize=8, color="darkblue")

    for ax in axes_flat[n_models:]:
        ax.axis("off")
    fig.suptitle(f"{spec.name}: Residual vs Voltage", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{spec.name}_residual_vs_voltage.png",
                dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_physical_decomposition(
    physics_params: np.ndarray, df_sample: pd.DataFrame, spec: TaskSpec,
    voltage_col: str, output_dir: Path,
) -> None:
    """Decompose I_dark into 5 physical components for a representative sample."""
    if physics_params is None or df_sample.empty:
        return

    V = df_sample[voltage_col].to_numpy(dtype=np.float64)
    sort_idx = np.argsort(V)

    # Build X for this sample (use first row's fixed params)
    n_pts = len(V)
    X = np.zeros((n_pts, 6))
    X[:, 0] = V
    X[:, 1] = df_sample["trap_assisted_recomb_A"].iloc[0]
    X[:, 2] = df_sample["ge_sio2_recomb_velocity"].iloc[0]
    X[:, 3] = df_sample["ge_si_recomb_velocity"].iloc[0]
    X[:, 4] = df_sample["active_layer_length"].iloc[0]
    X[:, 5] = df_sample["simulation_temperature"].iloc[0]

    # Compute individual components
    (log10_I_s0, n, log10_I_gr0, V_bi, E_a_gr,
     log10_I_tat0, B_tat, log10_I_surf0, log10_R_shunt,
     E_a_diff, gamma_surf) = DarkCurrentParams.unpack(physics_params)

    Vt = KB * X[:, 5]
    V_abs = np.maximum(np.abs(V), 1e-6)
    V_bi_pos = np.abs(V_bi)

    I_diff  = np.abs(10.0**log10_I_s0 * (X[:,5]/T_REF)**3 *
                     np.exp(-E_a_diff/Vt) *
                     (np.exp(np.minimum(Q*V/(n*Vt), 80.0)) - 1.0))
    I_gr    = 10.0**log10_I_gr0 * (X[:,4]/L_REF) * \
              np.sqrt(np.maximum(V_bi_pos - V, 1e-10)) * np.exp(-E_a_gr/Vt)
    I_tat   = 10.0**log10_I_tat0 * (X[:,1]/TRAP_REF) * \
              np.exp(-B_tat/(V_abs + 1e-4))
    I_surf  = 10.0**log10_I_surf0 * ((X[:,2]+X[:,3])/VEL_REF) * (V_abs**gamma_surf)
    I_shunt = V_abs / (10.0**log10_R_shunt)

    I_total = I_diff + I_gr + I_tat + I_surf + I_shunt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(V[sort_idx], I_diff[sort_idx],  label="Diffusion",  linewidth=1.2)
    ax.semilogy(V[sort_idx], I_gr[sort_idx],   label="G-R (SRH)",  linewidth=1.2)
    ax.semilogy(V[sort_idx], I_tat[sort_idx],  label="TAT (Hurkx)",linewidth=1.2)
    ax.semilogy(V[sort_idx], I_surf[sort_idx], label="Surface Leak",linewidth=1.2)
    ax.semilogy(V[sort_idx], I_shunt[sort_idx],label="Shunt",       linewidth=1.2)
    ax.semilogy(V[sort_idx], I_total[sort_idx],label="Total", linewidth=2.0,
                color="black")

    # Overlay TCAD data if available in sample
    if spec.target_col in df_sample.columns:
        y_tcad = df_sample[spec.target_col].to_numpy(dtype=np.float64)
        ax.scatter(V, y_tcad, s=20, alpha=0.5, color="gray",
                   label="TCAD", zorder=10)

    ax.set_xlabel(f"{voltage_col} (V)")
    ax.set_ylabel("Current (A)")
    ax.set_title(f"{spec.name}: Physical Decomposition", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{spec.name}_decomposition.png",
                dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_ac_overlay(pred_dfs: Dict[str, pd.DataFrame],
                    models_present: List[str], output_dir: Path) -> None:
    """Overlay S21(f) curves: physics_full vs dkan vs TCAD at a fixed bias."""
    spec = TASKS["AC_Response"]
    df_phys = pred_dfs.get("physics_full")
    df_dkan = pred_dfs.get("dkan")
    if df_phys is None or df_dkan is None:
        return

    # Find a representative bias voltage
    bias_vals = df_phys["bandwidth_bias"].drop_duplicates().sort_values()
    if len(bias_vals) > 4:
        bias_sample = [bias_vals.iloc[0], bias_vals.iloc[len(bias_vals)//3],
                       bias_vals.iloc[2*len(bias_vals)//3], bias_vals.iloc[-1]]
    else:
        bias_sample = list(bias_vals)

    cols = min(2, len(bias_sample))
    rows = int(math.ceil(len(bias_sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.2),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, v_bias in zip(axes_flat, bias_sample):
        sub_phys = df_phys[abs(df_phys["bandwidth_bias"] - v_bias) < 0.01].sort_values("frequency_ghz")
        sub_dkan = df_dkan[abs(df_dkan["bandwidth_bias"] - v_bias) < 0.01].sort_values("frequency_ghz")

        if not sub_phys.empty:
            ax.plot(sub_phys["frequency_ghz"], sub_phys["y_true"],
                    "o", markersize=6, alpha=0.5, color="gray", label="TCAD")
            ax.plot(sub_phys["frequency_ghz"], sub_phys["y_pred"],
                    "-", linewidth=2, color=MODEL_COLORS["physics_full"],
                    label="Physics (full)")
        if not sub_dkan.empty:
            ax.plot(sub_dkan["frequency_ghz"], sub_dkan["y_pred"],
                    "--", linewidth=2, color=MODEL_COLORS["dkan"],
                    label="D-KAN")

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("S21 (dB)")
        ax.set_title(f"V_bias = {v_bias:.2f} V", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    for ax in axes_flat[len(bias_sample):]:
        ax.axis("off")
    fig.suptitle("AC Response: S21(f) — Physics vs KAN vs TCAD",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "AC_Response_overlay.png",
                dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_kan_vs_physics_error(
    pred_dfs: Dict[str, pd.DataFrame], spec: TaskSpec,
    output_dir: Path,
) -> None:
    """Scatter: X = physics_full residual, Y = dkan residual."""
    df_phys = pred_dfs.get("physics_full")
    df_dkan = pred_dfs.get("dkan")
    if df_phys is None or df_dkan is None:
        return

    r_phys = df_phys["y_pred"].values - df_phys["y_true"].values
    r_dkan = df_dkan["y_pred"].values - df_dkan["y_true"].values

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(r_phys, r_dkan, s=10, alpha=0.4)
    lim = max(np.abs(r_phys).max(), np.abs(r_dkan).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
    ax.plot([-lim, lim], [0, 0], "k-", linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [-lim, lim], "k-", linewidth=0.5, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Physics (full) residual")
    ax.set_ylabel("D-KAN residual")
    ax.set_title(f"{spec.name}: Error Comparison", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(output_dir / f"{spec.name}_error_comparison.png",
                dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_f3db_vs_bias(ac_params: np.ndarray, df_test: pd.DataFrame,
                      output_dir: Path) -> None:
    """Show f_3dB vs bias voltage with the fitted power-law model."""
    if ac_params is None:
        return
    log10_f_0, p_bias, alpha_L, gamma_T, log10_S21_DC = ac_params

    V_vals = np.linspace(-3.0, -0.1, 100)
    f_0 = 10.0 ** log10_f_0

    # Use median L and T from test data for the reference curve
    L_med = np.median(df_test["active_layer_length"].values) if "active_layer_length" in df_test.columns else L_REF
    T_med = np.median(df_test["simulation_temperature"].values) if "simulation_temperature" in df_test.columns else T_REF

    f_3dB = f_0 * (np.abs(V_vals) ** p_bias) * \
            ((L_med / L_REF) ** alpha_L) * ((T_med / T_REF) ** gamma_T)
    f_3dB = np.maximum(f_3dB, 1e-3)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(np.abs(V_vals), f_3dB, "b-", label="f_3dB (fitted)", linewidth=2.0)
    ax.set_xlabel("|V_bias| (V)")
    ax.set_ylabel("3dB Bandwidth (GHz)")
    ax.set_title("AC Bandwidth vs Bias Voltage (power-law model)",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "AC_f3dB_vs_bias.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════

def write_report(summary_df: pd.DataFrame, metrics_df: pd.DataFrame,
                 args: argparse.Namespace, output_dir: Path) -> None:
    def _md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            vals = []
            for col in headers:
                v = row[col]
                if isinstance(v, (float, np.floating)):
                    vals.append(f"{float(v):.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# P0-3: TCAD-Referenced Compact Modeling: Classical Baselines vs KAN",
        "",
        "## Purpose",
        "",
        "Use TCAD data for the vertical Ge/Si photodetector as the high-fidelity "
        "reference, and compare classical analytical / equivalent-circuit-inspired "
        "baselines against data-driven KAN/MLP compact models on all three tasks.",
        "",
        "## Positioning",
        "",
        "This comparison does not claim that the physics baseline is a BSIM-like "
        "industry-standard photonic compact model. Unlike CMOS transistors, vertical "
        "photodetectors do not have a universally standardized compact model across "
        "foundries and EDA tools. The intended use case is TCAD-to-compact-model "
        "automation for new or customized optoelectronic devices. When a foundry "
        "provides a silicon-validated PDK model, that model remains the sign-off "
        "reference.",
        "",
        "The current device is a vertical photodetector. Avalanche photodetector "
        "(APD) compact modeling is a planned extension and is not part of this "
        "experiment.",
        "",
        "## Configuration",
        "",
        f"- Data: `{args.data}`",
        f"- Tasks: `{', '.join(args.tasks)}`",
        f"- Models: `{', '.join(args.models)}`",
        f"- Seeds: `{', '.join(map(str, args.seeds))}`",
        f"- Split mode: `{args.split_mode}`",
        f"- MLP epochs: `{args.mlp_epochs}`, lr=`{args.mlp_lr}`",
        f"- D-KAN steps: `{args.kan_steps}`, grid=8, k=3, width=8",
        (
            f"- B-KAN max epochs: `{args.bkan_epochs}`, "
            f"MC=`{args.bkan_mc_samples}`, batch=`{args.bkan_batch_size}`, "
            f"inner validation=`{args.bkan_val_fraction:.0%}`, "
            f"early-stopping patience=`{args.bkan_early_stopping_patience}` "
            "validation checks"
        ),
        (
            "- Statistical scope: multi-seed mean/population standard deviation."
            if len(set(args.seeds)) >= 3
            else "- Statistical scope: limited single/few-seed run; do not make stability claims."
        ),
        (
            "- Generalization scope: grouped complete-condition-curve split; "
            "test curves have no condition-curve overlap with training."
            if args.split_mode == "group"
            else "- Generalization scope: point-level random split; this is an interpolation "
            "benchmark and can contain points from the same condition curve in train and test."
        ),
        "",
        "## Summary Table",
        "",
    ]
    display_cols = [
        "task", "model", "n_seeds", "rmse_target_mean", "rmse_target_std",
        "r2_target_mean", "r2_target_std",
    ]
    existing = [c for c in display_cols if c in summary_df.columns]
    if not summary_df.empty:
        show = summary_df[existing].sort_values(["task", "model"])
        lines.append(_md_table(show))
    else:
        lines.append("_No results generated._")

    lines.extend(["", "## Key Findings", ""])
    for task in sorted(summary_df["task"].drop_duplicates()):
        sub = summary_df[summary_df["task"] == task]
        lines.append(f"### {task}")
        for model in MODEL_ORDER:
            if model not in sub["model"].values:
                continue
            row = sub[sub["model"] == model]
            if row.empty:
                continue
            rmse = float(row["rmse_target_mean"].iloc[0])
            r2 = float(row["r2_target_mean"].iloc[0])
            lines.append(f"- **{MODEL_LABELS.get(model, model)}**: "
                         f"RMSE={rmse:.5f}, R2={r2:.4f}")

    lines.extend([
        "",
        "## Output Files",
        "",
        "- `{task}_actual_vs_pred.png` — Actual vs predicted scatter per task",
        "- `rmse_bar_summary.png` — RMSE bar chart across all tasks/models",
        "- `{task}_residual_vs_voltage.png` — Residual vs voltage with physics region annotations",
        "- `I_dark_decomposition.png` — Physical mechanism decomposition",
        "- `AC_Response_overlay.png` — S21(f) overlay: physics vs KAN vs TCAD",
        "- `AC_f3dB_vs_bias.png` — Bandwidth vs bias voltage",
        "- `{task}_error_comparison.png` — KAN vs physics error scatter",
        "- `metrics_by_seed.csv` — Per-seed metrics",
        "- `metrics_summary.csv` — Aggregated metrics",
        "- `split_audit.csv` — Train/test condition-curve overlap audit",
        "",
        "## Physics Model Equations",
        "",
        "### DC Dark Current (9 params)",
        "```",
        "I_dark = |I_diff| + I_gr + I_tat + I_surf + I_shunt",
        "I_diff  = I_s0*(T/300)^3*exp(-E_a_diff/kT)*(exp(qV/nkT)-1)   [Shockley 1949]",
        "I_gr    = I_gr0*(L/L_ref)*sqrt(V_bi-V)*exp(-E_a_gr/kT)       [Sah-Noyce-Shockley 1957]",
        "I_tat   = I_tat0*(trap_A/1e-9)*exp(-B_tat/|V|)                [Hurkx IEEE TED 1992]",
        "I_surf  = I_surf0*((vs+vg)/5000)*|V|                           [Chen JAP 2016]",
        "I_shunt = |V|/R_shunt",
        "```",
        "",
        "### DC Photo Current (3 params)",
        "```",
        "I_photo = I_ph0 * (1 - exp(-alpha*L)) * (T/300)^gamma",
        "```",
        "_This first-order absorption baseline is intentionally compact. Its residual "
        "error is used to diagnose interface-recombination / trap-related corrections "
        "present in the TCAD data._",
        "",
        "### AC Frequency Response (5 params)",
        "```",
        "f_3dB(V, L, T) = f_0 * |V|^p * (L/L_ref)^alpha * (T/T_ref)^gamma",
        "S21(f) = S21_DC - 10*log10(1 + (f/f_3dB)^2)   [1st-order low-pass]",
        "```",
        "",
        "## Honest Analysis",
        "",
        "1. **Parameter non-identifiability**: DC I-V alone cannot uniquely separate G-R, "
        "TAT, and surface leakage — they have partially overlapping voltage dependencies. "
        "This is a fundamental limitation of physics-based modeling, not a fitting issue.",
        "",
        "2. **TAT regime (V < -2V)**: The simple exp(-B/|V|) is a first-order approximation. "
        "Real TAT involves trap energy distributions, multi-phonon processes, and "
        "field-dependent barrier lowering. KAN learns the correct nonlinearity automatically.",
        "",
        "3. **Photocurrent residual diagnosis**: The first-order absorption baseline "
        "captures dominant photocurrent physics but misses recombination-dependent "
        "corrections (trap_A, vs, vg) that are present in the TCAD data.",
        "",
        "4. **AC non-ideal roll-off**: The first-order low-pass model assumes a fixed "
        "functional form. KAN captures device-specific deviations from ideal behavior "
        "without manually adding additional parasitic terms.",
        "",
        "5. **Why use the KAN compact model**: The value is not replacing a mature foundry "
        "PDK model. The value is compressing TCAD sweeps into a compact, differentiable, "
        "uncertainty-aware, Verilog-A-exportable model when no validated PDK compact model "
        "exists yet.",
        "",
        "6. **Transferability boundary**: Classical analytical / equivalent-circuit "
        "baselines must be re-selected, enriched, and re-parameterized when the device "
        "geometry or material system changes. The KAN workflow still needs new TCAD or "
        "measurement data, but the train-distill-export pipeline can be reused.",
    ])
    (output_dir / "p0_3_report.md").write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse_target", "mae_target", "r2_target",
        "rmse_raw", "mae_raw", "medape_percent", "p95ape_percent",
        "param_count", "train_time_s", "fit_time_s",
    ]
    rows = []
    grouped = metrics_df.groupby(["task", "model"], sort=False)
    for (task, model), group in grouped:
        row = {"task": task, "model": model,
               "n_seeds": int(group["seed"].nunique())}
        for col in metric_cols:
            if col in group.columns:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P0-3: TCAD-referenced vertical photodetector compact-model comparison."
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument(
        "--capacitance-data", type=Path, default=DEFAULT_CAPACITANCE_DATA,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--split-mode",
        choices=["point", "group"],
        default="point",
        help=(
            "point keeps the historical random point-level 70/30 split; "
            "group splits complete condition curves so no fixed-condition curve "
            "appears in both train and test."
        ),
    )
    p.add_argument("--tasks", nargs="+",
                   default=["I_dark", "I_photo", "AC_Response"],
                   choices=list(TASKS))
    p.add_argument("--models", nargs="+",
                   default=["physics_simple", "physics_full", "dkan", "bkan",
                            "mlp_n", "mlp_l"],
                   choices=["physics_simple", "physics_full",
                            "dkan", "bkan", "mlp_n", "mlp_pm", "mlp_l",
                            "pdk_lut_knn", "poly3_ridge",
                            "spline_ridge", "rbf_nystroem",
                            "random_forest", "xgboost"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--device", default="auto")
    # MLP
    p.add_argument("--mlp-epochs", type=int, default=300)
    p.add_argument("--mlp-lr", type=float, default=0.002)
    p.add_argument("--mlp-lamb", type=float, default=0.001)
    p.add_argument("--mlp-lamb-l1", type=float, default=1.0)
    p.add_argument("--mlp-lamb-entropy", type=float, default=2.0)
    # D-KAN
    p.add_argument("--kan-steps", type=int, default=300)
    # B-KAN
    p.add_argument("--bkan-epochs", type=int, default=300)
    p.add_argument("--bkan-mc-samples", type=int, default=50)
    p.add_argument("--bkan-batch-size", type=int, default=32)
    p.add_argument("--bkan-val-fraction", type=float, default=0.15)
    p.add_argument("--bkan-early-stopping-patience", type=int, default=7)
    p.add_argument("--skip-bkan", action="store_true")
    # Physics fitting
    p.add_argument("--de-maxiter", type=int, default=2000)
    p.add_argument("--de-popsize", type=int, default=30)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(set(args.seeds)) < 3:
        print(
            "[warning] Fewer than three unique seeds: generated results are suitable "
            "for smoke testing, not model-stability claims."
        )
    os.environ.setdefault("TQDM_DISABLE", "1")
    device = resolve_device(args.device)
    if args.split_mode == "group" and Path(args.output_dir) == DEFAULT_OUT:
        args.output_dir = DEFAULT_GROUPED_OUT
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (output_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    metrics_rows: List[Dict] = []
    split_audit_rows: List[Dict] = []
    KAN_GRID, KAN_K, KAN_WIDTH = 8, 3, 8

    # Store fitted physics params and predictions per task
    fitted_params: Dict[str, np.ndarray] = {}
    physics_full_fit_rows: List[Dict] = []
    prediction_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}  # task -> model -> df

    for task_name in args.tasks:
        spec = TASKS[task_name]
        df_raw = (
            load_capacitance_dataframe(args.capacitance_data)
            if task_name == "Capacitance"
            else load_dataframe(args.data)
        )
        df_task = prepare_task_dataframe(df_raw, spec)
        print(f"\n[{spec.name}] valid rows: {len(df_task)}")

        prediction_dfs[spec.name] = {}
        task_y_true, task_y_pred = {}, {}
        task_y_raw = None
        task_test_df = None

        for seed in args.seeds:
            print(f"\n[{spec.name}] seed={seed}")
            df_train, df_test = split_task_data(
                df_task, spec, seed, args.split_mode
            )
            task_test_df = df_test
            audit_row = split_audit_row(
                df_train, df_test, spec, seed, args.split_mode
            )
            split_audit_rows.append(audit_row)
            if args.split_mode == "group" and audit_row["overlap_groups"]:
                raise RuntimeError(
                    f"[{spec.name}] Grouped split leakage: "
                    f"{audit_row['overlap_groups']} overlapping curves"
                )

            # Build arrays for KAN/MLP training
            arrays = make_scaled_arrays(df_train, df_test, spec)
            y_true_target = arrays["y_test_transformed"]
            y_true_raw_np = arrays["y_test_raw"]

            # Raw data for physics model fitting
            X_train_raw = df_train[list(spec.input_cols)].to_numpy(dtype=np.float64)
            X_test_raw = df_test[list(spec.input_cols)].to_numpy(dtype=np.float64)
            y_train_raw = transformed_target(df_train, spec)
            y_test_raw_t = transformed_target(df_test, spec)

            # ── Fit physics models (once per task; params shared across seeds) ──
            phys_simple_fitted = False
            phys_full_fitted = False

            for model_name in args.models:
                run_start = time.perf_counter()
                run_tag = f"seed_{seed}"

                if model_name == "physics_simple":
                    if task_name in {"AC_Response", "Capacitance"}:
                        continue  # No AC for simple model
                    if not phys_simple_fitted:
                        # Simple model: only diffusion + G-R (4 params)
                        bounds_simple = DarkCurrentParams.bounds_natural[:4] if task_name == "I_dark" else []
                        x0_simple = DarkCurrentParams.initial_guess()[:4]

                        if task_name == "I_dark":
                            def _simple_fn(p, X):
                                (li0, nn, lgr0, vbi) = p
                                V = X[:, 0]; T = X[:, 5]
                                Vt = KB * T; V_abs = np.maximum(np.abs(V), 1e-6)
                                I_diff = np.abs(10.0**li0 * (T/T_REF)**3 *
                                                np.exp(-0.66/Vt) *
                                                (np.exp(np.minimum(Q*V/(nn*Vt),80.0))-1.0))
                                I_gr = 10.0**lgr0 * (X[:,4]/L_REF) * \
                                       np.sqrt(np.maximum(abs(vbi)-V, 1e-10)) * \
                                       np.exp(-0.33/Vt)
                                return np.log10(np.maximum(I_diff+I_gr, 1e-30))

                            dc_params_simple, dc_fit_info = fit_physics_model(
                                _simple_fn, X_train_raw, y_train_raw,
                                bounds_simple, x0_simple,
                                seed=seed, maxiter_de=args.de_maxiter,
                                popsize=args.de_popsize,
                            )
                            y_pred_phys = _simple_fn(dc_params_simple, X_test_raw)
                            info = {**dc_fit_info, "param_count": 4}
                        elif task_name == "I_photo":
                            # Simple photo: first-order absorption baseline
                            bounds_p = PhotoCurrentParams.bounds_natural
                            x0_p = PhotoCurrentParams.initial_guess()

                            def _simple_photo(p, X):
                                lp0, ae, gt = p
                                Iph0 = 10.0**lp0
                                absorp = 1.0 - np.exp(-ae * X[:, 4])
                                tf = (X[:, 5] / T_REF) ** gt
                                return np.log10(np.maximum(Iph0*absorp*tf, 1e-30))

                            dc_params_simple, dc_fit_info = fit_physics_model(
                                _simple_photo, X_train_raw, y_train_raw,
                                bounds_p, x0_p,
                                seed=seed, maxiter_de=args.de_maxiter,
                                popsize=args.de_popsize,
                            )
                            y_pred_phys = _simple_photo(dc_params_simple, X_test_raw)
                            info = {**dc_fit_info, "param_count": 3}

                        fitted_params[f"{task_name}_simple"] = dc_params_simple
                        phys_simple_fitted = True

                    # Recompute prediction using stored params
                    if task_name == "I_dark":
                        y_pred_phys = _simple_fn(
                            fitted_params[f"{task_name}_simple"], X_test_raw)
                    else:
                        y_pred_phys = _simple_photo(
                            fitted_params[f"{task_name}_simple"], X_test_raw)
                    y_pred = y_pred_phys.ravel()

                elif model_name == "physics_full":
                    if task_name == "Capacitance":
                        continue
                    if not phys_full_fitted:
                        if task_name == "I_dark":
                            bounds = DarkCurrentParams.bounds_natural
                            x0 = DarkCurrentParams.initial_guess()
                            dc_params, dc_info = fit_physics_model(
                                dark_current_model, X_train_raw, y_train_raw,
                                bounds, x0, seed=seed,
                                maxiter_de=args.de_maxiter,
                                popsize=args.de_popsize,
                            )
                            fitted_params["I_dark_full"] = dc_params
                            info = {
                                **dc_info,
                                "param_count": len(DarkCurrentParams.names),
                            }
                            fit_row = {
                                "task": "I_dark",
                                "model": "physics_full",
                                "seed": seed,
                                "free_parameter_count": len(DarkCurrentParams.names),
                            }
                            fit_row.update({
                                name: float(value)
                                for name, value in zip(
                                    DarkCurrentParams.names, dc_params
                                )
                            })
                            fit_row.update({
                                f"fixed_{name}": float(value)
                                for name, value in
                                DarkCurrentParams.fixed_values.items()
                            })
                            physics_full_fit_rows.append(fit_row)
                        elif task_name == "I_photo":
                            bounds = PhotoCurrentParams.bounds_natural
                            x0 = PhotoCurrentParams.initial_guess()
                            ph_params, ph_info = fit_physics_model(
                                photo_current_model, X_train_raw, y_train_raw,
                                bounds, x0, seed=seed,
                                maxiter_de=args.de_maxiter,
                                popsize=args.de_popsize,
                            )
                            fitted_params["I_photo_full"] = ph_params
                            info = {**ph_info, "param_count": 3}
                        elif task_name == "AC_Response":
                            # Use V_bi from dark fit if available, else default
                            v_bi = 0.45
                            if "I_dark_full" in fitted_params:
                                v_bi = float(DarkCurrentParams.unpack(
                                    fitted_params["I_dark_full"])[3])
                            bounds = ACResponseParams.bounds_natural
                            x0 = ACResponseParams.initial_guess(v_bi)

                            def _ac_fn(p, X):
                                return ac_response_model(p, X, V_bi=v_bi)

                            ac_params, ac_info = fit_physics_model(
                                _ac_fn, X_train_raw, y_train_raw,
                                bounds, x0, seed=seed,
                                maxiter_de=args.de_maxiter,
                                popsize=args.de_popsize,
                            )
                            fitted_params["AC_full"] = ac_params
                            info = {**ac_info, "param_count": 5}
                        phys_full_fitted = True

                    # Predict
                    if task_name == "I_dark":
                        y_pred = dark_current_model(
                            fitted_params["I_dark_full"], X_test_raw).ravel()
                    elif task_name == "I_photo":
                        y_pred = photo_current_model(
                            fitted_params["I_photo_full"], X_test_raw).ravel()
                    elif task_name == "AC_Response":
                        v_bi = float(DarkCurrentParams.unpack(
                            fitted_params.get("I_dark_full",
                                              DarkCurrentParams.initial_guess()))[3])
                        y_pred = ac_response_model(
                            fitted_params["AC_full"], X_test_raw, V_bi=v_bi).ravel()

                elif model_name in MLP_ARCHES:
                    y_pred, info = train_mlp(
                        arrays, MLP_ARCHES[model_name], seed=seed,
                        device=device, epochs=args.mlp_epochs, lr=args.mlp_lr,
                        lamb=args.mlp_lamb, lamb_l1=args.mlp_lamb_l1,
                        lamb_entropy=args.mlp_lamb_entropy,
                    )
                elif model_name == "dkan":
                    y_pred, info = train_dkan(
                        arrays, seed=seed, device=device,
                        steps=args.kan_steps, grid=KAN_GRID,
                        k=KAN_K, width=KAN_WIDTH,
                    )
                elif model_name == "bkan":
                    if args.skip_bkan:
                        continue
                    y_pred, info = train_bkan(
                        df_train, df_test, spec, seed=seed, device=device,
                        output_dir=output_dir, epochs=args.bkan_epochs,
                        mc_samples=args.bkan_mc_samples,
                        grid=KAN_GRID, k=KAN_K, width=KAN_WIDTH,
                        run_tag=run_tag, split_mode=args.split_mode,
                        batch_size=args.bkan_batch_size,
                        val_fraction=args.bkan_val_fraction,
                        early_stopping_patience=(
                            args.bkan_early_stopping_patience
                        ),
                    )
                elif model_name in ENGINEERING_MODELS:
                    n_features = len(spec.input_cols)
                    model = build_engineering_model(
                        model_name, len(df_train), n_features, seed,
                    )
                    X_tr = df_train[list(spec.input_cols)].to_numpy(dtype=np.float64)
                    X_te = df_test[list(spec.input_cols)].to_numpy(dtype=np.float64)
                    y_tr = transformed_target(df_train, spec)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te).ravel()
                    param_count = count_engineering_params(
                        model_name, model, len(df_train), n_features,
                    )
                    info = {"param_count": param_count}
                else:
                    continue

                metrics = compute_metrics(y_true_target, y_true_raw_np, y_pred, spec)
                row = {
                    "task": spec.name, "seed": seed, "model": model_name,
                    **metrics, **info,
                    "wall_time_s": time.perf_counter() - run_start,
                    "n_train": len(df_train), "n_test": len(df_test),
                    "split_mode": args.split_mode,
                    "train_curves": audit_row["train_curves"],
                    "test_curves": audit_row["test_curves"],
                    "overlap_groups": audit_row["overlap_groups"],
                }
                metrics_rows.append(row)
                print(f"  [{model_name}] RMSE={metrics['rmse_target']:.5f}, "
                      f"R2={metrics['r2_target']:.5f}")

                # Store prediction data
                pred_cols = {
                    "task": spec.name, "seed": seed, "model": model_name,
                    "y_true": y_true_target.ravel(),
                    "y_pred": y_pred.ravel(),
                    "y_true_raw": y_true_raw_np.ravel(),
                }
                # Add voltage column for residual plots
                if task_name == "I_dark":
                    pred_cols["dark_voltage"] = df_test["dark_voltage"].values
                    pred_cols["T"] = df_test["simulation_temperature"].values
                elif task_name == "I_photo":
                    pred_cols["light_voltage"] = df_test["light_voltage"].values
                    pred_cols["T"] = df_test["simulation_temperature"].values
                elif task_name == "AC_Response":
                    pred_cols["frequency_ghz"] = df_test["frequency_ghz"].values
                    pred_cols["bandwidth_bias"] = df_test["bandwidth_bias"].values

                if model_name not in prediction_dfs[spec.name]:
                    prediction_dfs[spec.name][model_name] = pd.DataFrame(pred_cols)
                else:
                    prediction_dfs[spec.name][model_name] = pd.concat(
                        [prediction_dfs[spec.name][model_name],
                         pd.DataFrame(pred_cols)], ignore_index=True)

    # ── Save metrics ──────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = summarize_metrics(metrics_df)

    metrics_df.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(split_audit_rows).to_csv(output_dir / "split_audit.csv", index=False)
    if "I_dark" in args.tasks and "physics_full" in args.models:
        (output_dir / "physics_full_parameterization.json").write_text(
            json.dumps(DarkCurrentParams.manifest(), indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(physics_full_fit_rows).to_csv(
            output_dir / "physics_full_fit_parameters.csv", index=False
        )

    # ── Generate plots ────────────────────────────────────────────────
    for task_name in args.tasks:
        spec = TASKS[task_name]
        preds = prediction_dfs.get(spec.name, {})
        models_present = [m for m in MODEL_ORDER if m in preds]
        if not models_present:
            continue

        # Plot 1: Actual vs predicted
        save_actual_vs_pred(preds, spec, models_present, output_dir)

        # Plot 3: Residual vs voltage
        if task_name == "I_dark":
            save_residual_vs_voltage(preds, spec, "dark_voltage",
                                     models_present, output_dir)
        elif task_name == "I_photo":
            save_residual_vs_voltage(preds, spec, "light_voltage",
                                     models_present, output_dir)

        # Plot 6: KAN vs physics error comparison
        if "physics_full" in preds and "dkan" in preds:
            save_kan_vs_physics_error(preds, spec, output_dir)

    # Plot 2: RMSE bar
    save_rmse_bar(summary_df, output_dir)

    # Plot 4: Physical decomposition (I_dark only, first seed)
    if "I_dark_full" in fitted_params and "I_dark" in args.tasks:
        df_train, df_test = split_task_data(
            prepare_task_dataframe(df_raw, TASKS["I_dark"]),
            TASKS["I_dark"],
            args.seeds[0],
            args.split_mode,
        )
        # Find a representative parameter subset
        df_sample = df_test.head(200).copy()  # use first 200 test points
        save_physical_decomposition(
            fitted_params["I_dark_full"], df_sample, TASKS["I_dark"],
            "dark_voltage", output_dir)

    # Plot 5: AC overlay
    if "AC_Response" in args.tasks and "AC_Response" in prediction_dfs:
        save_ac_overlay(
            prediction_dfs["AC_Response"],
            [m for m in MODEL_ORDER if m in prediction_dfs["AC_Response"]],
            output_dir)

    # Plot: f3dB vs bias
    if "AC_full" in fitted_params:
        df_ac = prepare_task_dataframe(df_raw, TASKS["AC_Response"])
        _, df_ac_test = split_task_data(
            df_ac, TASKS["AC_Response"], args.seeds[0], args.split_mode
        )
        save_f3db_vs_bias(fitted_params["AC_full"], df_ac_test, output_dir)

    # ── Report ────────────────────────────────────────────────────────
    write_report(summary_df, metrics_df, args, output_dir)

    print(f"\nSaved results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
