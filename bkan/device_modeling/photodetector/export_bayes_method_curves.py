from __future__ import annotations

from pathlib import Path

from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler
from device_modeling.photodetector.task_config import (
    DEFAULT_DATA,
    TASKS,
    load_research_data,
    prepare_task_dataframe,
    split_task_dataframe,
)

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
RESULTS = REPO_ROOT / "artifacts" / "results"
OUT = RESULTS / "current_visualization_report"

TASK_KEY_BY_LABEL = {
    "I_dark": "dark_current",
    "I_photo": "photo_current",
    "AC_Response": "ac_response",
}

TASK_LABELS = {
    "I_dark": "Dark current",
    "I_photo": "Net photocurrent",
    "AC_Response": "Optical SSAC response",
}

AXIS_COLUMNS = {
    "I_dark": "dark_voltage",
    "I_photo": "light_voltage",
    "AC_Response": "frequency_ghz",
}

TARGET_COLUMNS = {
    "I_dark": "dark_current",
    "I_photo": "net_photocurrent",
    "AC_Response": "ac_response_db",
}

BAYES_METHOD_CHECKPOINTS = {
    "I_photo": {
        # Historical dropout/HMC checkpoints predict illuminated total current
        # and must not be replayed against the corrected net target.
        "vi": RESULTS / "net_photocurrent_retrain" / "uq_repeated_grouped" / "seed_42" / "I-photo-bayes-results" / "model_checkpoint.pt",
    },
    "I_dark": {
        "dropout": RESULTS / "validation_bayes_methods_dark_ac" / "I-dark-bayes-results_dropout" / "I_dark_bayes_results" / "model_checkpoint.pt",
        "hmc": RESULTS / "validation_bayes_methods_dark_ac" / "I-dark-bayes-results_hmc" / "I_dark_bayes_results" / "model_checkpoint.pt",
        "vi": RESULTS / "validation_bayes_methods_dark_ac" / "I-dark-bayes-results" / "model_checkpoint.pt",
    },
    "AC_Response": {
        "dropout": RESULTS / "validation_bayes_methods_dark_ac" / "AC-response-bayes-results_dropout" / "AC_Response_bayes_results" / "model_checkpoint.pt",
        "hmc": RESULTS / "validation_bayes_methods_dark_ac" / "AC-response-bayes-results_hmc" / "AC_Response_bayes_results" / "model_checkpoint.pt",
        "vi": RESULTS / "validation_bayes_methods_dark_ac" / "AC-response-bayes-results" / "model_checkpoint.pt",
    },
}


def select_representative_curve(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    axis_col = AXIS_COLUMNS[task]
    if "_curve_id" in frame.columns:
        curve_id = frame["_curve_id"].value_counts(sort=True).index[0]
        return frame[frame["_curve_id"].eq(curve_id)].sort_values(axis_col).copy()

    excluded = {axis_col, TARGET_COLUMNS[task]}
    condition_cols = [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not condition_cols:
        return frame.sort_values(axis_col).copy()

    first_key = frame.groupby(condition_cols, sort=False).size().idxmax()
    if not isinstance(first_key, tuple):
        first_key = (first_key,)
    mask = np.ones(len(frame), dtype=bool)
    for column, value in zip(condition_cols, first_key):
        mask &= np.isclose(frame[column].to_numpy(dtype=float), float(value), rtol=1e-10, atol=1e-12)
    return frame.loc[mask].sort_values(axis_col).copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_research_data(DEFAULT_DATA)
    rows = []

    for task in TASK_LABELS:
        spec = TASKS[TASK_KEY_BY_LABEL[task]]
        frame, inputs = prepare_task_dataframe(raw, spec)
        _, _, _, test = split_task_dataframe(frame, spec, seed=42)
        curve = select_representative_curve(test, task)
        x_values = curve[list(inputs)].to_numpy(dtype=np.float32)
        condition_cols = [column for column in inputs if column != AXIS_COLUMNS[task]]

        for method, checkpoint in BAYES_METHOD_CHECKPOINTS[task].items():
            modeler = BayesKANDeviceModeler.load_model(str(checkpoint), device="cpu")
            if not hasattr(modeler, "global_aleatoric_var"):
                modeler.global_aleatoric_var = float(getattr(modeler, "hmc_noise_var", 0.0) or 0.0)
            pred = modeler.predict_with_uncertainty(x_values)
            mean = np.asarray(pred["mean"], dtype=float).reshape(-1)
            std = np.asarray(pred["std"], dtype=float).reshape(-1)
            for i, (_, source_row) in enumerate(curve.iterrows()):
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "axis_col": AXIS_COLUMNS[task],
                        "axis_value": float(source_row[AXIS_COLUMNS[task]]),
                        "actual": float(source_row[TARGET_COLUMNS[task]]),
                        "mean": float(mean[i]),
                        "std": float(std[i]),
                        **{column: float(source_row[column]) for column in condition_cols},
                    }
                )

    out_path = OUT / "bayesian_method_curve_predictions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved Bayesian method curve predictions to: {out_path}")


if __name__ == "__main__":
    main()
