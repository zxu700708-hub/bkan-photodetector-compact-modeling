"""Audit microring spectra and recover device-level metrics from held-out predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("bkan", "dkan", "mlp_l", "spline_ridge", "gmls", "autopinn")
MODEL_LABELS = {
    "bkan": "BKAN-VI",
    "dkan": "DKAN",
    "mlp_l": "MLP-L",
    "spline_ridge": "Spline-Ridge",
    "gmls": "GMLS-adapted",
    "autopinn": "AutoPINN-adapted",
}
METRIC_COLUMNS = (
    "resonance_wavelength_nm",
    "fwhm_nm",
    "quality_factor",
    "through_extinction_ratio_db",
    "drop_extinction_ratio_db",
    "through_insertion_loss_db",
    "drop_insertion_loss_db",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    return parser.parse_args()


def prediction_file(results: Path, seed: int, task: str, model: str) -> Path:
    return results / "predictions" / f"seed_{seed}" / task / model / "test_predictions.npz"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pair(results: Path, seed: int, model: str) -> dict[str, np.ndarray]:
    paths = {
        task: prediction_file(results, seed, task, model)
        for task in ("through", "drop")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing formal prediction files: " + ", ".join(missing))
    bundles: dict[str, dict[str, np.ndarray]] = {}
    for task, path in paths.items():
        # These prediction bundles are produced locally by run_ring_third_device.py.
        # Pandas emits object-dtype string arrays for structure/sample labels, so
        # loading those labels requires pickle even though all modeled values are
        # separately checked for finiteness below.
        with np.load(path, allow_pickle=True) as source:
            bundles[task] = {name: np.asarray(source[name]) for name in source.files}
    for name in (
        "structure_id",
        "bias_v",
        "wavelength_m",
        "through_db",
        "drop_db",
    ):
        left = bundles["through"][name]
        right = bundles["drop"][name]
        if left.shape != right.shape or not np.array_equal(left, right):
            raise RuntimeError(f"Through/drop test-row mismatch for seed={seed}, model={model}: {name}")
    result = dict(bundles["through"])
    result["through_prediction"] = np.asarray(bundles["through"]["prediction"], dtype=np.float64)
    result["drop_prediction"] = np.asarray(bundles["drop"]["prediction"], dtype=np.float64)
    for name in ("through_prediction", "drop_prediction", "through_db", "drop_db"):
        if not np.isfinite(result[name]).all():
            raise RuntimeError(f"Non-finite spectrum values for seed={seed}, model={model}: {name}")
    return result


def reference_lookup(data_root: Path) -> pd.DataFrame:
    path = data_root / "training_tables" / "hifi_ring_derived_metrics.csv"
    frame = pd.read_csv(path)
    required = {"structure_id", "bias_v", "metric_extraction_version", *METRIC_COLUMNS}
    if not required.issubset(frame.columns):
        raise KeyError(f"Reference metric table is missing {sorted(required - set(frame.columns))}")
    if len(frame) != 400 or frame["structure_id"].nunique() != 80:
        raise RuntimeError("Expected 400 derived-metric rows from 80 structures")
    if not frame["metric_extraction_version"].eq("local_resonance_v1").all():
        raise RuntimeError("Unexpected derived-metric extraction version")
    return frame.set_index(["structure_id", "bias_v"], verify_integrity=True)


def extract_prediction_rows(
    bundle: dict[str, np.ndarray],
    reference: pd.DataFrame,
    seed: int,
    model: str,
    spectrum_metrics,
) -> list[dict[str, object]]:
    frame = pd.DataFrame(
        {
            "structure_id": bundle["structure_id"].astype(str),
            "bias_v": bundle["bias_v"].astype(float),
            "wavelength_m": bundle["wavelength_m"].astype(float),
            "through_prediction": bundle["through_prediction"].astype(float),
            "drop_prediction": bundle["drop_prediction"].astype(float),
        }
    )
    rows: list[dict[str, object]] = []
    for structure_id, structure in frame.groupby("structure_id", sort=True):
        predicted_hint: float | None = None
        for bias in sorted(structure["bias_v"].unique(), reverse=True):
            curve = structure.loc[np.isclose(structure["bias_v"], bias)]
            key = (str(structure_id), float(bias))
            if key not in reference.index:
                raise KeyError(f"Missing reference derived metrics for {key}")
            ref = reference.loc[key]
            record: dict[str, object] = {
                "seed": seed,
                "model": model,
                "structure_id": str(structure_id),
                "bias_v": float(bias),
                "extraction_ok": False,
                "failure": "",
            }
            for column in METRIC_COLUMNS:
                record[f"reference_{column}"] = float(ref[column])
                record[f"predicted_{column}"] = float("nan")
            try:
                predicted = spectrum_metrics(
                    curve["wavelength_m"].to_numpy(dtype=float),
                    curve["through_prediction"].to_numpy(dtype=float),
                    curve["drop_prediction"].to_numpy(dtype=float),
                    resonance_hint_nm=predicted_hint,
                )
                if predicted_hint is None:
                    predicted_hint = float(predicted["resonance_wavelength_nm"])
                values = np.asarray([predicted[column] for column in METRIC_COLUMNS], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError("non-finite predicted metric")
                record["extraction_ok"] = True
                record["predicted_selected_extrema_alignment_nm"] = float(
                    predicted["selected_extrema_alignment_nm"]
                )
                for column in METRIC_COLUMNS:
                    record[f"predicted_{column}"] = float(predicted[column])
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                record["failure"] = f"{type(error).__name__}: {error}"
                record["predicted_selected_extrema_alignment_nm"] = float("nan")
            rows.append(record)
    return rows


def add_errors(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["resonance_abs_error_pm"] = 1.0e3 * (
        frame["predicted_resonance_wavelength_nm"]
        - frame["reference_resonance_wavelength_nm"]
    ).abs()
    for short, column in (("fwhm", "fwhm_nm"), ("quality_factor", "quality_factor")):
        frame[f"{short}_ape_percent"] = 100.0 * (
            frame[f"predicted_{column}"] - frame[f"reference_{column}"]
        ).abs() / frame[f"reference_{column}"].abs()
    for short, column in (
        ("through_er", "through_extinction_ratio_db"),
        ("drop_er", "drop_extinction_ratio_db"),
        ("through_il", "through_insertion_loss_db"),
        ("drop_il", "drop_insertion_loss_db"),
    ):
        frame[f"{short}_abs_error_db"] = (
            frame[f"predicted_{column}"] - frame[f"reference_{column}"]
        ).abs()
    frame["cross_port_alignment_pm"] = (
        1.0e3 * frame["predicted_selected_extrema_alignment_nm"]
    )
    frame["tuning_abs_error_pm_per_v"] = np.nan
    for _, indices in frame.groupby(["seed", "model", "structure_id"], sort=False).groups.items():
        part = frame.loc[indices]
        zero = part.iloc[int(np.argmin(np.abs(part["bias_v"].to_numpy(dtype=float))))]
        if not bool(zero["extraction_ok"]):
            continue
        predicted_zero = float(zero["predicted_resonance_wavelength_nm"])
        reference_zero = float(zero["reference_resonance_wavelength_nm"])
        for index, row in part.iterrows():
            bias = abs(float(row["bias_v"]))
            if bias == 0.0 or not bool(row["extraction_ok"]):
                continue
            predicted_tuning = (float(row["predicted_resonance_wavelength_nm"]) - predicted_zero) / bias
            reference_tuning = (float(row["reference_resonance_wavelength_nm"]) - reference_zero) / bias
            frame.loc[index, "tuning_abs_error_pm_per_v"] = 1.0e3 * abs(
                predicted_tuning - reference_tuning
            )
    return frame


def aggregate_by_seed(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, model), part in frame.groupby(["seed", "model"], sort=False):
        valid = part.loc[part["extraction_ok"]]

        def mean(name: str) -> float:
            return float(valid[name].mean()) if len(valid) else float("nan")

        def median(name: str) -> float:
            return float(valid[name].median()) if len(valid) else float("nan")

        rows.append(
            {
                "seed": int(seed),
                "model": str(model),
                "curves": int(len(part)),
                "recovered_curves": int(len(valid)),
                "recovery_fraction": float(len(valid) / len(part)),
                "resonance_mae_pm": mean("resonance_abs_error_pm"),
                "fwhm_median_ape_percent": median("fwhm_ape_percent"),
                "quality_factor_median_ape_percent": median("quality_factor_ape_percent"),
                "through_er_mae_db": mean("through_er_abs_error_db"),
                "drop_er_mae_db": mean("drop_er_abs_error_db"),
                "through_il_mae_db": mean("through_il_abs_error_db"),
                "drop_il_mae_db": mean("drop_il_abs_error_db"),
                "tuning_mae_pm_per_v": mean("tuning_abs_error_pm_per_v"),
                "cross_port_alignment_median_pm": median("cross_port_alignment_pm"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in by_seed.columns
        if column not in {"seed", "model", "curves", "recovered_curves"}
    ]
    result = by_seed.groupby("model", sort=False)[numeric].agg(["mean", "std"])
    result.columns = [f"{name}_{stat}" for name, stat in result.columns]
    return result.reset_index()


def representative_curve(results: Path, model: str = "bkan") -> tuple[dict[str, np.ndarray], str, float]:
    bundle = load_pair(results, 42, model)
    frame = pd.DataFrame(
        {
            "structure_id": bundle["structure_id"].astype(str),
            "bias_v": bundle["bias_v"].astype(float),
            "through_error2": (bundle["through_prediction"] - bundle["through_db"]) ** 2,
            "drop_error2": (bundle["drop_prediction"] - bundle["drop_db"]) ** 2,
        }
    )
    curve_errors = (
        frame.groupby(["structure_id", "bias_v"], sort=False)[["through_error2", "drop_error2"]]
        .mean()
        .mean(axis=1)
        .pow(0.5)
        .sort_values()
    )
    structure_id, bias = curve_errors.index[len(curve_errors) // 2]
    return bundle, str(structure_id), float(bias)


def plot_evidence(
    results: Path,
    reference: pd.DataFrame,
    response_summary: pd.DataFrame,
    derived_summary: pd.DataFrame,
) -> None:
    task_minima = response_summary.groupby("task")["rmse_target_mean"].min()
    scores = {}
    for model, part in response_summary.groupby("model"):
        scores[model] = float(
            np.mean([row.rmse_target_mean / task_minima.loc[row.task] for row in part.itertuples()])
        )
    comparator = min((model for model in scores if model != "bkan"), key=scores.get)
    bkan, structure_id, bias = representative_curve(results, "bkan")
    other = load_pair(results, 42, comparator)
    ids = bkan["structure_id"].astype(str)
    mask = (ids == structure_id) & np.isclose(bkan["bias_v"].astype(float), bias)
    order = np.argsort(bkan["wavelength_m"][mask])
    wavelength_nm = bkan["wavelength_m"][mask][order] * 1.0e9
    resonance_nm = float(reference.loc[(structure_id, bias), "resonance_wavelength_nm"])
    detuning_nm = wavelength_nm - resonance_nm
    window = np.abs(detuning_nm) <= 0.22

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    styles = (("TCAD", "#111111", "-"), ("BKAN-VI", "#4C78A8", "-"), (MODEL_LABELS[comparator], "#F58518", "--"))
    for ax, target, ylabel in (
        (axes[0], "through", "Through response (dB)"),
        (axes[1], "drop", "Drop response (dB)"),
    ):
        arrays = (
            bkan[f"{target}_db"][mask][order],
            bkan[f"{target}_prediction"][mask][order],
            other[f"{target}_prediction"][mask][order],
        )
        for values, (label, color, linestyle) in zip(arrays, styles):
            ax.plot(detuning_nm[window], values[window], color=color, linestyle=linestyle, linewidth=1.25, label=label)
        ax.axvline(0.0, color="#777777", linewidth=0.7, alpha=0.6)
        ax.set_xlabel("Detuning from reference resonance (nm)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(f"Through; $V={bias:g}$ V")
    axes[1].set_title("Drop; same held-out curve")

    metric_names = [
        "resonance_mae_pm_mean",
        "quality_factor_median_ape_percent_mean",
        "through_er_mae_db_mean",
        "drop_il_mae_db_mean",
    ]
    labels = ["Resonance\nMAE", "$Q$ median\nAPE", "Through ER\nMAE", "Drop IL\nMAE"]
    selected = derived_summary.set_index("model").loc[list(MODELS), metric_names].to_numpy(dtype=float)
    minima = np.nanmin(selected, axis=0)
    ratios = selected / np.where(minima > 0.0, minima, 1.0)
    image = axes[2].imshow(np.clip(ratios, 1.0, 5.0), cmap="YlOrRd", vmin=1.0, vmax=5.0, aspect="auto")
    axes[2].set_xticks(np.arange(len(labels)), labels, fontsize=7)
    recovery = derived_summary.set_index("model").loc[list(MODELS), "recovery_fraction_mean"]
    recovery_labels = [
        f"{MODEL_LABELS[model]} [{100.0 * recovery.loc[model]:.0f}%]" for model in MODELS
    ]
    axes[2].set_yticks(np.arange(len(MODELS)), recovery_labels, fontsize=7)
    axes[2].set_title("Error / best [recovery]")
    for row in range(ratios.shape[0]):
        for column in range(ratios.shape[1]):
            value = ratios[row, column]
            axes[2].text(column, row, "--" if not np.isfinite(value) else f"{value:.1f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04, label="ratio (color capped at 5)")
    for label, ax in zip("abc", axes, strict=True):
        ax.text(
            0.5,
            -0.30,
            f"({label})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=10,
            clip_on=False,
        )
    fig.tight_layout(pad=0.8)
    fig.savefig(results / "ring_third_device_evidence.pdf", bbox_inches="tight")
    fig.savefig(results / "ring_third_device_evidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    (results / "representative_curve.json").write_text(
        json.dumps(
            {
                "selection": "seed-42 BKAN combined through/drop curve RMSE closest to median",
                "structure_id": structure_id,
                "bias_v": bias,
                "comparison_model": comparator,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.results = args.results.resolve()
    helper_root = args.data_root.parents[1]
    if str(helper_root) not in sys.path:
        sys.path.insert(0, str(helper_root))
    from hifi_common import spectrum_metrics

    validation = json.loads((args.data_root / "validation_report.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("metric_extraction_version") != "local_resonance_v1":
        raise RuntimeError("The source microring dataset has not passed the frozen validation gates")
    response_path = args.results / "metrics_summary.csv"
    split_path = args.results / "split_manifest.csv"
    if not response_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("The formal spectral comparison has not completed its final summary")
    response_summary = pd.read_csv(response_path)
    split_manifest = pd.read_csv(split_path)
    expected_pairs = len(args.seeds) * len(args.models)
    if len(split_manifest) != 80 * len(args.seeds):
        raise RuntimeError("Incomplete formal split manifest")
    if split_manifest.groupby(["seed", "structure_id"])["split"].nunique().max() != 1:
        raise RuntimeError("Structure leakage detected in the formal split manifest")

    reference = reference_lookup(args.data_root)
    records: list[dict[str, object]] = []
    for seed in args.seeds:
        for model in args.models:
            records.extend(
                extract_prediction_rows(
                    load_pair(args.results, seed, model), reference, seed, model, spectrum_metrics
                )
            )
    curves = add_errors(pd.DataFrame(records))
    if len(curves) != expected_pairs * 40:
        raise RuntimeError(f"Expected {expected_pairs * 40} held-out curves, found {len(curves)}")
    by_seed = aggregate_by_seed(curves)
    summary = aggregate_summary(by_seed)
    curve_path = args.results / "derived_metrics_by_curve.csv"
    seed_path = args.results / "derived_metrics_by_seed.csv"
    summary_path = args.results / "derived_metrics_summary.csv"
    curves.to_csv(curve_path, index=False)
    by_seed.to_csv(seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_evidence(args.results, reference, response_summary, summary)
    audit = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "PASS",
        "protocol": "ring_prediction_derived_metrics_local_resonance_v1",
        "source_validation_status": validation["status"],
        "seeds": args.seeds,
        "models": args.models,
        "prediction_pairs": expected_pairs,
        "held_out_curves": int(len(curves)),
        "all_prediction_rows_finite": True,
        "through_drop_rows_aligned": True,
        "structure_split_leakage": False,
        "response_metrics_sha256": file_sha256(response_path),
        "reference_metrics_sha256": file_sha256(
            args.data_root / "training_tables" / "hifi_ring_derived_metrics.csv"
        ),
        "derived_curve_metrics_sha256": file_sha256(curve_path),
        "derived_seed_metrics_sha256": file_sha256(seed_path),
        "derived_summary_sha256": file_sha256(summary_path),
        "metric_recovery_fraction_min": float(by_seed["recovery_fraction"].min()),
        "metric_recovery_fraction_mean": float(by_seed["recovery_fraction"].mean()),
        "interpretation": "Derived metrics are secondary checks reconstructed from independently fitted through/drop spectra; extraction failures remain in the denominator.",
    }
    (args.results / "derived_metrics_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
