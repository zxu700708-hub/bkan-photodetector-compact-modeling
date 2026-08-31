"""Data loading helpers for device-modeling experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_EXTENSIONS = (".csv", ".xls", ".xlsx")
AC_RESPONSE_DB_COLUMN = "ac_response_db"
LEGACY_AC_RESPONSE_COLUMN = "bandwidth"
MODELED_TARGET_COLUMNS = {"dark_current", "light_current", AC_RESPONSE_DB_COLUMN}
AC_TARGET_SEMANTICS = (
    "sampled normalized optical-generation SSAC contact-current-magnitude response, "
    "ac_response_db = 20*log10(|dI_contact(f)|/max_f|dI_contact|)"
)


def canonicalize_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame using the maintained device-model column names.

    Historical TCAD exports called the sampled frequency-response ordinate
    ``bandwidth``.  The values are actually the normalized current-magnitude
    response ``20*log10(|dI(f)|/max_f|dI|)`` in dB, not a scalar bandwidth or
    a port-derived S parameter.  Accept that legacy header at the input
    boundary, but expose the unambiguous maintained name everywhere else.
    """

    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    if AC_RESPONSE_DB_COLUMN not in frame.columns and LEGACY_AC_RESPONSE_COLUMN in frame.columns:
        if "frequency_ghz" not in frame.columns:
            raise ValueError(
                "Legacy column 'bandwidth' is ambiguous without 'frequency_ghz'. "
                "Use 'ac_response_db' for a sampled response curve or define a "
                "separate scalar-bandwidth task."
            )
        paired = frame[["frequency_ghz", LEGACY_AC_RESPONSE_COLUMN]].dropna()
        if (
            paired["frequency_ghz"].nunique() < 3
            or paired[LEGACY_AC_RESPONSE_COLUMN].nunique() < 3
        ):
            raise ValueError(
                "Legacy column 'bandwidth' does not look like a sampled frequency "
                "response. Refusing to relabel a possible scalar bandwidth target."
            )
        frame = frame.rename(columns={LEGACY_AC_RESPONSE_COLUMN: AC_RESPONSE_DB_COLUMN})
        frame.attrs["legacy_column_migrations"] = {
            LEGACY_AC_RESPONSE_COLUMN: AC_RESPONSE_DB_COLUMN
        }
        frame.attrs["ac_target_semantics"] = AC_TARGET_SEMANTICS
        frame.attrs["ac_perturbation"] = "small-signal optical generation (optical SSAC)"
        frame.attrs["ac_observable"] = "contact small-signal current magnitude"
    elif AC_RESPONSE_DB_COLUMN in frame.columns:
        frame.attrs["ac_target_semantics"] = AC_TARGET_SEMANTICS
        frame.attrs["ac_perturbation"] = "small-signal optical generation (optical SSAC)"
        frame.attrs["ac_observable"] = "contact small-signal current magnitude"
    return frame


def discover_data_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in DATA_EXTENSIONS:
        files.extend(path.glob(f"*{suffix}"))
    return sorted(files)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported data file format: {path}")
    return canonicalize_model_columns(frame)


def aggregate_data_smart(data_dir: str | Path, shuffle: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate raw simulation CSV/Excel files into one dataframe.

    This is the maintained device_modeling equivalent of the old
    simulation.full.aggregate_data_smart helper. It keeps the useful behavior:
    directory scanning, CSV/Excel support, header cleanup, temperature fallback,
    and potential input discovery.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory for smart aggregation: {data_dir}")

    files = discover_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV/Excel data files found in: {data_dir}")

    frames = []
    skipped = []
    for file_path in files:
        try:
            frame = read_table(file_path)
            if "Temperature" not in frame.columns and "temperature" not in frame.columns:
                frame["temperature"] = 300.0
            frame["_source_file"] = file_path.stem
            frames.append(frame)
        except Exception as exc:
            skipped.append({"file": str(file_path), "error": str(exc)})

    if not frames:
        raise ValueError(f"All data files failed to load from: {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    potential_inputs = [
        column
        for column in combined.columns
        if "current" not in column.lower()
        and not column.startswith("_")
        and combined[column].nunique(dropna=True) > 1
    ]
    if shuffle:
        combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    report = pd.DataFrame(
        {
            "source_file": [path.name for path in files],
            "status": ["loaded" if path.stem in set(combined["_source_file"].astype(str)) else "skipped" for path in files],
        }
    )
    if skipped:
        report = pd.concat([report, pd.DataFrame(skipped)], ignore_index=True, sort=False)
    combined.attrs["aggregation_report"] = report
    combined.attrs["potential_inputs"] = potential_inputs
    return combined, potential_inputs


def looks_like_combined_dataset(frame: pd.DataFrame) -> bool:
    columns = set(frame.columns)
    return len(columns & MODELED_TARGET_COLUMNS) >= 2


def looks_like_raw_simulation_directory(path: str | Path) -> bool:
    path = Path(path)
    files = discover_data_files(path)
    if not files:
        return False
    if any(file_path.suffix.lower() in {".xls", ".xlsx"} for file_path in files):
        return True
    if len(files) == 1:
        return False
    for file_path in files[:5]:
        try:
            if looks_like_combined_dataset(read_table(file_path)):
                return True
        except Exception:
            continue
    return False
