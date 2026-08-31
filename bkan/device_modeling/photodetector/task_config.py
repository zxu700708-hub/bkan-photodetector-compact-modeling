"""Shared data and task definitions for KAN device-model research."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from device_modeling.photodetector.data_ingestion import (
    AC_RESPONSE_DB_COLUMN,
    aggregate_data_smart,
    discover_data_files,
    looks_like_combined_dataset,
    looks_like_raw_simulation_directory,
    read_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
MAIN_DIR = PROJECT_ROOT / "main"
DEFAULT_DATA = Path(r"<FROZEN_TCAD_ROOT>/Data\Data_20260617_0243_181408") / "data"
DEFAULT_CAPACITANCE_DATA = (
    REPO_ROOT
    / "artifacts"
    / "results"
    / "apparent_capacitance_correction"
    / "primary_ge_si_apparent_capacitance.csv"
)
DEFAULT_RESULTS = REPO_ROOT / "artifacts" / "results" / "device_modeling"
DARK_CURRENT_ZERO_FLOOR = 1e-10
CURRENT_ZERO_FLOOR = 1e-10
NET_PHOTOCURRENT_COLUMN = "net_photocurrent"
PHOTOCURRENT_VOLTAGE_ATOL = 1e-12
MODELED_SERIES = (
    ("dark_voltage", "dark_current"),
    ("light_voltage", "light_current"),
    ("frequency_ghz", AC_RESPONSE_DB_COLUMN),
)
DEFAULT_SPLIT_FRACTIONS = (0.70, 0.10, 0.10, 0.10)


@dataclass(frozen=True)
class TaskSpec:
    key: str
    name: str
    result_subdir: str
    target_col: str
    axis_col: str
    input_candidates: Tuple[str, ...]
    use_log_transform: bool
    y_bounds: Tuple[float, float]


TASKS = {
    "dark_current": TaskSpec(
        key="dark_current",
        name="I_dark",
        result_subdir="I-dark-bayes-results",
        target_col="dark_current",
        axis_col="dark_voltage",
        input_candidates=(
            "dark_voltage",
            "trap_assisted_recomb_A",
            "ge_sio2_recomb_velocity",
            "ge_si_recomb_velocity",
            "active_layer_length",
            "simulation_temperature",
        ),
        use_log_transform=True,
        y_bounds=(-15.0, 5.0),
    ),
    "photo_current": TaskSpec(
        key="photo_current",
        name="I_photo",
        result_subdir="I-photo-bayes-results",
        target_col=NET_PHOTOCURRENT_COLUMN,
        axis_col="light_voltage",
        input_candidates=(
            "light_voltage",
            "trap_assisted_recomb_A",
            "ge_sio2_recomb_velocity",
            "ge_si_recomb_velocity",
            "active_layer_length",
            "simulation_temperature",
        ),
        use_log_transform=True,
        y_bounds=(-15.0, 5.0),
    ),
    "ac_response": TaskSpec(
        key="ac_response",
        name="AC_Response",
        result_subdir="AC-response-bayes-results",
        target_col=AC_RESPONSE_DB_COLUMN,
        axis_col="frequency_ghz",
        input_candidates=(
            "frequency_ghz",
            "bandwidth_bias",
            "active_layer_length",
            "simulation_temperature",
        ),
        use_log_transform=False,
        y_bounds=(-np.inf, np.inf),
    ),
    "capacitance": TaskSpec(
        key="capacitance",
        name="Capacitance",
        result_subdir="Capacitance-bayes-results",
        target_col="capacitance_F",
        axis_col="bias_v",
        input_candidates=(
            "bias_v",
            "log_frequency_ghz",
            "trap_assisted_recomb_A",
            "ge_sio2_recomb_velocity",
            "ge_si_recomb_velocity",
            "active_layer_length",
            "simulation_temperature",
        ),
        use_log_transform=False,
        y_bounds=(-np.inf, np.inf),
    ),
}


def resolve_capacitance_data_path(path=DEFAULT_CAPACITANCE_DATA):
    """Resolve the corrected total-admittance apparent-capacitance table.

    ``path`` may be the CSV itself, the 20260617 dataset root, or one of the
    nested supplemental directories.  This keeps the capacitance task convenient to
    launch from the same root used by the charge-proxy validation scripts.
    """

    path = Path(path)
    if path.is_file():
        return path
    candidates = [
        path
        / "supplemental_charge_ac"
        / "supplemental_electrical"
        / "ac_cv"
        / "kan_ac_cv_training.csv",
        path / "supplemental_electrical" / "ac_cv" / "kan_ac_cv_training.csv",
        path / "ac_cv" / "kan_ac_cv_training.csv",
        path / "kan_ac_cv_training.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"AC-CV capacitance table not found from: {path}")


def load_capacitance_data(path=DEFAULT_CAPACITANCE_DATA):
    """Load the corrected apparent-capacitance target used by the paper."""

    table_path = resolve_capacitance_data_path(path)
    frame = read_table(table_path)
    required = {
        "sample_id",
        "bias_v",
        "frequency_ghz",
        "capacitance_F",
        "trap_assisted_recomb_A",
        "ge_sio2_recomb_velocity",
        "ge_si_recomb_velocity",
        "active_layer_length",
        "simulation_temperature",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing required AC-CV capacitance columns: {missing}")
    if "target_definition" not in frame.columns or not frame["target_definition"].astype(str).str.startswith(
        "C_app=Im(Y_total)/(2*pi*f)"
    ).all():
        raise ValueError(
            "The legacy AC-CV table is not a valid paper input. Use "
            "scripts/apply_apparent_capacitance_correction.py to create the "
            "Vac-corrected total-admittance apparent-capacitance table."
        )
    if "vac_V" not in frame.columns or not np.allclose(
        frame["vac_V"].astype(float).to_numpy(), 1.0e-3, rtol=0.0, atol=1.0e-15
    ):
        raise ValueError("The apparent-capacitance table must record the actual Vac=1 mV.")

    frame = frame.copy()
    frame = frame[frame["frequency_ghz"] > 0.0].copy()
    frame["log_frequency_ghz"] = np.log10(frame["frequency_ghz"].astype(float))
    frame["_curve_id"] = "capacitance_sample_" + frame["sample_id"].astype(str)
    frame.attrs["data_source_mode"] = "corrected_total_admittance_apparent_capacitance"
    frame.attrs["capacitance_source_path"] = str(table_path)
    return frame


def _paired_finite_count(frame, columns):
    values = frame[list(columns)].replace([np.inf, -np.inf], np.nan)
    return int(values.dropna().shape[0])


def _clean_directory_curves(frames):
    available_series = [
        columns for columns in MODELED_SERIES
        if all(column in frames[0][1].columns for column in columns)
    ]
    expected_counts = {}
    for columns in available_series:
        counts = pd.Series([_paired_finite_count(frame, columns) for _, frame in frames])
        modes = counts.mode()
        expected_counts[columns] = int(modes.max())

    retained = []
    report_rows = []
    for curve_id, frame in frames:
        issues = []
        row = {"curve_id": curve_id}
        for columns, expected in expected_counts.items():
            name = f"{columns[1]}_points"
            count = _paired_finite_count(frame, columns)
            row[name] = count
            if count != expected:
                issues.append(f"{columns[1]} has {count}/{expected} finite points")
        row["status"] = "removed" if issues else "retained"
        row["reason"] = "; ".join(issues)
        report_rows.append(row)
        if not issues:
            retained.append(frame)

    if not retained:
        raise ValueError("All condition curves were removed during completeness cleaning")
    cleaned = pd.concat(retained, ignore_index=True)
    cleaned = apply_dark_current_floor(cleaned)
    cleaned.attrs["cleaning_report"] = pd.DataFrame(report_rows)
    return cleaned


def apply_dark_current_floor(frame):
    frame = frame.copy()
    report = []
    for column in ("dark_current", "light_current"):
        if column not in frame.columns:
            continue
        current = frame[column].replace([np.inf, -np.inf], np.nan)
        low_mask = current.notna() & (current.abs() < CURRENT_ZERO_FLOOR)
        frame.loc[low_mask, column] = 0.0
        voltage_col = "dark_voltage" if column == "dark_current" else "light_voltage"
        zero_voltage = (
            frame[voltage_col] == 0.0
            if voltage_col in frame.columns
            else pd.Series(False, index=frame.index)
        )
        report.append(
            {
                "rule": f"abs({column}) < {CURRENT_ZERO_FLOOR:g} -> 0",
                f"finite_{column}_rows": int(current.notna().sum()),
                f"{column}_values_set_to_zero": int(low_mask.sum()),
                "zero_voltage_rows_set_to_zero": int((low_mask & zero_voltage).sum()),
                "nonzero_voltage_rows_set_to_zero": int((low_mask & ~zero_voltage).sum()),
            }
        )
    required = {"dark_voltage", "light_voltage", "dark_current", "light_current"}
    if required.issubset(frame.columns):
        paired = frame[list(required)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        voltage_mismatch = paired & ~np.isclose(
            frame["dark_voltage"].astype(float),
            frame["light_voltage"].astype(float),
            rtol=0.0,
            atol=PHOTOCURRENT_VOLTAGE_ATOL,
        )
        if voltage_mismatch.any():
            examples = frame.loc[
                voltage_mismatch, ["dark_voltage", "light_voltage"]
            ].head(3).to_dict("records")
            raise ValueError(
                "Cannot derive net photocurrent from misaligned dark/illuminated "
                f"biases; examples: {examples}"
            )

        net = pd.Series(np.nan, index=frame.index, dtype=np.float64)
        net.loc[paired] = (
            frame.loc[paired, "light_current"].astype(float)
            - frame.loc[paired, "dark_current"].astype(float)
        )
        nonpositive = paired & (net <= 0.0)
        if nonpositive.any():
            examples = frame.loc[
                nonpositive,
                ["dark_voltage", "dark_current", "light_current"],
            ].head(3).to_dict("records")
            raise ValueError(
                "Derived net photocurrent must be positive for logarithmic training; "
                f"examples: {examples}"
            )
        frame[NET_PHOTOCURRENT_COLUMN] = net
        frame.attrs["photocurrent_target_definition"] = (
            "net_photocurrent = light_current - dark_current, paired at the same "
            "device condition and bias"
        )
        frame.attrs["photocurrent_target_audit"] = {
            "paired_rows": int(paired.sum()),
            "voltage_mismatch_rows": int(voltage_mismatch.sum()),
            "nonpositive_net_rows": int(nonpositive.sum()),
        }
    if not report:
        return frame
    frame.attrs["low_current_cleaning_report"] = pd.DataFrame(report)
    return frame


def load_research_data(path=DEFAULT_DATA):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Research dataset not found: {path}")
    if path.is_dir():
        all_tables = discover_data_files(path)
        if len(all_tables) == 1:
            frame = read_table(all_tables[0])
            if looks_like_combined_dataset(frame):
                frame.attrs["data_source_mode"] = "single_table_directory"
                return apply_dark_current_floor(frame)
        if looks_like_raw_simulation_directory(path):
            frame, _ = aggregate_data_smart(path)
            frame = apply_dark_current_floor(frame)
            frame.attrs["data_source_mode"] = "raw_simulation_directory"
            return frame
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV data files found in: {path}")
        frames = []
        for file_path in files:
            frame = read_table(file_path)
            frame["_curve_id"] = file_path.stem
            frames.append((file_path.stem, frame))
        frame = _clean_directory_curves(frames)
        frame.attrs["data_source_mode"] = "curve_csv_directory"
        return frame
    frame = read_table(path)
    frame.attrs["data_source_mode"] = "single_table"
    return apply_dark_current_floor(frame)


def target_values(df, spec):
    values = df[spec.target_col].to_numpy(dtype=np.float64)
    if spec.use_log_transform:
        if np.any(values <= 0.0):
            raise ValueError(f"{spec.name} contains non-positive values that cannot be log-transformed")
        return np.log10(values)
    return values


def prepare_task_dataframe(df_raw, spec):
    missing = [name for name in (spec.target_col, spec.axis_col) if name not in df_raw.columns]
    if missing:
        raise KeyError(f"Missing required columns for {spec.name}: {missing}")

    inputs = tuple(
        column
        for column in spec.input_candidates
        if column in df_raw.columns and df_raw[column].nunique(dropna=True) > 1
    )
    if spec.axis_col not in inputs:
        raise ValueError(f"Required varying input is unavailable: {spec.axis_col}")

    metadata_columns = ["_curve_id"] if "_curve_id" in df_raw.columns else []
    columns = list(inputs) + [spec.target_col] + metadata_columns
    df = df_raw[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if spec.use_log_transform:
        df = df[df[spec.target_col] > 0.0].copy()
    if spec.key == "ac_response":
        df = df[df[spec.axis_col] > 0.0].copy()

    transformed = target_values(df, spec)
    valid = np.isfinite(transformed)
    df = df.loc[valid].reset_index(drop=True)
    if "_curve_id" not in df.columns:
        condition_cols = [column for column in inputs if column != spec.axis_col]
        if not condition_cols:
            raise ValueError(f"No condition columns available for grouped task: {spec.name}")
        df["_curve_id"] = pd.util.hash_pandas_object(
            df[condition_cols], index=False
        ).astype(str)
    if len(df) < 20:
        raise ValueError(f"Only {len(df)} usable samples remain for {spec.name}")
    return df, inputs


def _group_strata(df, group_labels, spec, n_bins=5):
    condition_cols = [
        column for column in spec.input_candidates
        if column in df.columns and column != spec.axis_col
    ]
    if not condition_cols:
        return None

    meta = df[condition_cols].copy()
    meta["_group_label"] = group_labels.to_numpy()
    group_meta = meta.groupby("_group_label", sort=False)[condition_cols].median()
    varying_cols = [
        column for column in condition_cols
        if group_meta[column].nunique(dropna=True) > 1
    ]
    if not varying_cols:
        return None

    ranked = []
    for column in varying_cols:
        values = group_meta[column].astype(float)
        ranked.append(values.rank(method="average", pct=True).to_numpy(dtype=np.float64))
    score = np.mean(np.vstack(ranked), axis=0)
    bins = min(n_bins, max(2, len(np.unique(score))))
    try:
        strata = pd.qcut(score, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    strata = pd.Series(strata, index=group_meta.index).astype("Int64")
    if strata.isna().any() or strata.nunique(dropna=True) < 2:
        return None
    counts = strata.value_counts()
    if counts.min() < 4:
        return None
    return strata.astype(str)


def _stratified_train_test_split(groups, strata, test_size, seed):
    if strata is None:
        return train_test_split(groups, test_size=test_size, random_state=seed, shuffle=True)
    group_strata = strata.loc[groups].to_numpy()
    try:
        return train_test_split(
            groups,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=group_strata,
        )
    except ValueError:
        return train_test_split(groups, test_size=test_size, random_state=seed, shuffle=True)


def validate_split_fractions(fractions):
    """Validate and normalize train/validation/calibration/test fractions."""
    fractions = tuple(float(value) for value in fractions)
    if len(fractions) != 4:
        raise ValueError(
            "Expected four split fractions: train validation calibration test"
        )
    if any(value <= 0.0 for value in fractions):
        raise ValueError("All split fractions must be positive")
    if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Split fractions must sum to 1.0, received {sum(fractions):.12g}"
        )
    return fractions


def allocate_split_group_counts(n_groups, fractions):
    """Allocate an exact integer group count with largest-remainder rounding."""
    fractions = validate_split_fractions(fractions)
    exact = np.asarray(fractions, dtype=np.float64) * int(n_groups)
    counts = np.floor(exact).astype(int)
    remaining = int(n_groups) - int(counts.sum())
    remainders = exact - counts
    order = np.argsort(-remainders, kind="stable")
    for index in order[:remaining]:
        counts[index] += 1
    if np.any(counts < 1):
        raise ValueError(
            f"Split fractions are too small for {n_groups} condition curves: "
            f"{tuple(counts)}"
        )
    if int(counts.sum()) != int(n_groups):
        raise RuntimeError("Internal split allocation error")
    return tuple(int(value) for value in counts)


def task_group_labels(df, spec):
    """Return one stable condition-curve label per row."""
    if "_curve_id" in df.columns:
        return df["_curve_id"].astype(str)
    condition_cols = [
        column for column in spec.input_candidates
        if column in df.columns and column != spec.axis_col
    ]
    if not condition_cols:
        raise ValueError(
            f"No condition columns available for grouped split: {spec.name}"
        )
    return pd.util.hash_pandas_object(
        df[condition_cols], index=False
    ).astype(str)


def split_task_dataframe(
    df,
    spec,
    seed,
    fractions=DEFAULT_SPLIT_FRACTIONS,
):
    """Split complete condition curves into train/validation/calibration/test."""
    fractions = validate_split_fractions(fractions)
    group_labels = task_group_labels(df, spec)

    unique_groups = group_labels.drop_duplicates().to_numpy()
    if len(unique_groups) < 10:
        raise ValueError(f"Only {len(unique_groups)} condition curves available for {spec.name}")
    train_count, validation_count, calibration_count, test_count = (
        allocate_split_group_counts(len(unique_groups), fractions)
    )

    strata = _group_strata(df, group_labels, spec)
    train_groups, remainder_groups = _stratified_train_test_split(
        unique_groups,
        strata,
        test_size=validation_count + calibration_count + test_count,
        seed=seed,
    )
    validation_groups, remainder_groups = _stratified_train_test_split(
        remainder_groups,
        strata,
        test_size=calibration_count + test_count,
        seed=seed,
    )
    calibration_groups, test_groups = _stratified_train_test_split(
        remainder_groups,
        strata,
        test_size=test_count,
        seed=seed,
    )
    partitions = (train_groups, validation_groups, calibration_groups, test_groups)
    return tuple(
        df.loc[group_labels.isin(groups)].reset_index(drop=True)
        for groups in partitions
    )
