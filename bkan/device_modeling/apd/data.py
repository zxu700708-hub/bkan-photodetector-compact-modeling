"""Validated APD table adapters for the second-device experiment.

Only the paired DC curves and their derived photocurrent, multiplication-gain,
and scalar voltage targets are in scope.  AC, C--V, charge, spatial, and noise
tables are deliberately outside this experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_APD_ROOT = Path(
    r"<FROZEN_APD_ROOT>/Data\APD_Data_20260816_0426_957372_paper"
)

STRUCTURE_INPUTS: Tuple[str, ...] = (
    "trap_assisted_recomb_A",
    "ge_sio2_recomb_velocity",
    "ge_si_recomb_velocity",
    "simulation_temperature",
    "ge_width_scale",
)


@dataclass(frozen=True)
class APDTask:
    key: str
    name: str
    source_table: str
    target_col: str
    axis_col: str
    input_cols: Tuple[str, ...]
    use_log_transform: bool
    y_bounds: Tuple[float, float]
    ylabel: str
    primary: bool = True


APD_TASKS = {
    "dark_current": APDTask(
        key="dark_current",
        name="APD_I_dark",
        source_table="apd_dc.csv",
        target_col="current_A",
        axis_col="bias_v",
        input_cols=("bias_v", *STRUCTURE_INPUTS),
        use_log_transform=True,
        y_bounds=(-31.0, 1.0),
        ylabel=r"$\log_{10}(I_{\rm dark}/{\rm A})$",
    ),
    "photo_current": APDTask(
        key="photo_current",
        name="APD_I_photo",
        source_table="apd_dc.csv",
        target_col="current_A",
        axis_col="bias_v",
        input_cols=("bias_v", *STRUCTURE_INPUTS, "optical_power_W"),
        use_log_transform=True,
        y_bounds=(-31.0, 1.0),
        ylabel=r"$\log_{10}(I_{\rm photo}/{\rm A})$",
    ),
    "net_photocurrent": APDTask(
        key="net_photocurrent",
        name="APD_I_net",
        source_table="apd_gain.csv",
        target_col="net_photocurrent_A",
        axis_col="bias_v",
        input_cols=("bias_v", *STRUCTURE_INPUTS, "optical_power_W"),
        use_log_transform=True,
        y_bounds=(-31.0, 1.0),
        ylabel=r"$\log_{10}(I_{\rm net}/{\rm A})$",
    ),
    "multiplication_gain": APDTask(
        key="multiplication_gain",
        name="APD_M",
        source_table="apd_gain.csv",
        target_col="multiplication_gain",
        axis_col="bias_v",
        input_cols=("bias_v", *STRUCTURE_INPUTS, "optical_power_W"),
        use_log_transform=True,
        y_bounds=(-4.0, 3.0),
        ylabel=r"$\log_{10}(M)$",
    ),
    "gain_threshold_voltage": APDTask(
        key="gain_threshold_voltage",
        name="APD_V_M10",
        source_table="apd_fom.csv",
        target_col="voltage_at_gain_threshold_v",
        axis_col="optical_power_W",
        input_cols=(*STRUCTURE_INPUTS, "optical_power_W"),
        use_log_transform=False,
        y_bounds=(-20.0, 0.0),
        ylabel=r"$V_{M=10}$ (V)",
        primary=False,
    ),
    "breakdown_voltage": APDTask(
        key="breakdown_voltage",
        name="APD_V_br",
        source_table="apd_fom.csv",
        target_col="breakdown_voltage_v",
        axis_col="optical_power_W",
        input_cols=(*STRUCTURE_INPUTS, "optical_power_W"),
        use_log_transform=False,
        y_bounds=(-20.0, 0.0),
        ylabel=r"$V_{\rm br}$ (V)",
        primary=False,
    ),
}


def training_tables_dir(root: str | Path = DEFAULT_APD_ROOT) -> Path:
    root = Path(root)
    tables = root / "supplemental_apd" / "training_tables"
    if not tables.is_dir():
        raise FileNotFoundError(f"APD training-table directory not found: {tables}")
    return tables


def _read_source(root: str | Path, task: APDTask) -> pd.DataFrame:
    path = training_tables_dir(root) / task.source_table
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _canonical_dark_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one optical-power-independent dark trace per physical structure.

    Each structure was simulated at two optical powers.  The dark branch is an
    optical-power-independent reference and must not be counted as two
    independent curves.  The lower-power condition is the deterministic
    canonical representative.
    """

    dark = frame.loc[frame["mode"].eq("dark") & frame["bias_v"].lt(0.0)].copy()
    dark = dark.sort_values(
        ["structure_group_id", "bias_v", "optical_power_W", "sample_id"],
        kind="stable",
    )
    return dark.drop_duplicates(["structure_group_id", "bias_v"], keep="first")


def load_apd_task(
    task_key: str,
    root: str | Path = DEFAULT_APD_ROOT,
) -> tuple[pd.DataFrame, APDTask]:
    """Load one model-ready APD task with a structure-level group identifier."""

    if task_key not in APD_TASKS:
        raise KeyError(f"Unknown APD task: {task_key}")
    task = APD_TASKS[task_key]
    frame = _read_source(root, task)

    if task_key == "dark_current":
        frame = _canonical_dark_curves(frame)
    elif task_key == "photo_current":
        frame = frame.loc[frame["mode"].eq("illuminated")].copy()

    required = {
        "sample_id",
        "structure_group_id",
        task.target_col,
        *task.input_cols,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"[{task.name}] missing columns: {missing}")

    keep = ["sample_id", "structure_group_id", *task.input_cols, task.target_col]
    keep = list(dict.fromkeys(keep))
    frame = frame[keep].copy()
    numeric = [*task.input_cols, task.target_col]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric)
    if task.use_log_transform:
        frame = frame.loc[frame[task.target_col] > 0.0].copy()
        transformed = np.log10(frame[task.target_col].to_numpy(dtype=np.float64))
    else:
        transformed = frame[task.target_col].to_numpy(dtype=np.float64)
    valid = (transformed > task.y_bounds[0]) & (transformed < task.y_bounds[1])
    frame = frame.loc[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"[{task.name}] no usable rows")

    frame["_curve_id"] = frame["structure_group_id"].astype(str)
    if frame["structure_group_id"].nunique() != 78:
        raise ValueError(
            f"[{task.name}] expected 78 accepted structure groups, observed "
            f"{frame['structure_group_id'].nunique()}"
        )
    if frame.duplicated(["sample_id", task.axis_col]).any() and task_key != "dark_current":
        raise ValueError(f"[{task.name}] duplicate sample/axis keys")
    return frame, task
