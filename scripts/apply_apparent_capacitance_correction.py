"""Build the canonical apparent-capacitance dataset and export overlays.

The historical 64,000-row SSAC table recorded ``vac_V=1`` although the
DEVICE perturbation amplitude was 1 mV.  Its stored ``capacitance_F`` is
exactly Im(Y_total)/(2*pi*f), so the physically scaled total-admittance
apparent capacitance is obtained by a deterministic factor of 1000.  This
script preserves the historical source, writes a corrected canonical table,
and creates corrected copies of publication-facing symbolic exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    r"<FROZEN_TCAD_ROOT>/Data\Data_20260617_0243_181408\supplemental_charge_ac"
    r"\supplemental_electrical\ac_cv\kan_ac_cv_training.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts/results/apparent_capacitance_correction"
SCALE = 1000.0
ACTUAL_VAC_V = 1.0e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_legacy(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "sample_id",
        "bias_v",
        "frequency_hz",
        "Y_real_S",
        "Y_imag_S",
        "Y_abs_S",
        "capacitance_F",
        "conductance_S",
        "vac_V",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"legacy capacitance table is missing columns: {missing}")
    if len(frame) != 64_000 or frame["sample_id"].nunique() != 160:
        raise ValueError("expected 64,000 rows from 160 DEVICE identifiers")
    vac = frame["vac_V"].to_numpy(dtype=float)
    if not np.allclose(vac, 1.0, rtol=0.0, atol=0.0):
        raise ValueError("legacy vac_V is not uniformly 1 V as expected")
    omega = 2.0 * math.pi * frame["frequency_hz"].to_numpy(dtype=float)
    stored = frame["capacitance_F"].to_numpy(dtype=float)
    reconstructed = frame["Y_imag_S"].to_numpy(dtype=float) / omega
    rel = np.abs(reconstructed - stored) / np.maximum(np.abs(stored), 1e-300)
    max_rel = float(np.max(rel))
    if max_rel > 1e-12:
        raise ValueError(f"stored capacitance is not Im(Y_total)/omega: {max_rel}")
    per_device = frame.groupby("sample_id", sort=False).size().to_numpy()
    if not np.all(per_device == 400):
        raise ValueError("each DEVICE must contain a complete 16x25 surface")
    return {
        "rows": int(len(frame)),
        "device_count": int(frame["sample_id"].nunique()),
        "rows_per_device": 400,
        "bias_count": int(frame["bias_v"].nunique()),
        "frequency_count": int(frame["frequency_hz"].nunique()),
        "legacy_vac_V": 1.0,
        "actual_vac_V": ACTUAL_VAC_V,
        "scale_factor": SCALE,
        "max_relative_identity_error": max_rel,
    }


def corrected_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.insert(
        out.columns.get_loc("capacitance_F"),
        "legacy_capacitance_F",
        out["capacitance_F"].to_numpy(dtype=float),
    )
    for column in ("Y_real_S", "Y_imag_S", "Y_abs_S", "conductance_S"):
        out[column] = out[column].astype(float) * SCALE
    out["capacitance_F"] = out["legacy_capacitance_F"] * SCALE
    out["capacitance_from_total_admittance_imag_F"] = out["capacitance_F"]
    out["legacy_recorded_vac_V"] = out["vac_V"].astype(float)
    out["vac_V"] = ACTUAL_VAC_V
    out["normalization_correction_factor"] = SCALE
    out["target_definition"] = "C_app=Im(Y_total)/(2*pi*f)"
    return out


def _scale_nested(value: Any) -> Any:
    if isinstance(value, list):
        return [_scale_nested(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) * SCALE
    raise TypeError(f"unsupported coefficient payload: {type(value)!r}")


def correct_formula_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    if "output_scale" in payload:
        payload["output_scale"] = float(payload["output_scale"]) * SCALE
    elif "intercept" in payload and (
        "coefficients" in payload or "features" in payload
    ):
        payload["intercept"] = float(payload["intercept"]) * SCALE
        if "coefficients" in payload:
            payload["coefficients"] = _scale_nested(payload["coefficients"])
        if "features" in payload:
            features = []
            for original in payload["features"]:
                feature = dict(original)
                feature["coefficients"] = _scale_nested(feature["coefficients"])
                features.append(feature)
            payload["features"] = features
    else:
        raise ValueError("unrecognized formula schema")
    payload["target_name"] = "total-admittance apparent capacitance"
    payload["target_symbol"] = "C_app"
    payload["posthoc_scale_from_legacy_source"] = SCALE
    return payload


def correct_formula_json(source: Path, destination: Path) -> str:
    payload = json.loads(source.read_text(encoding="utf-8"))
    try:
        payload = correct_formula_payload(payload)
    except ValueError as exc:
        raise ValueError(f"unrecognized formula schema: {source}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return "json"


def correct_formula_va(source: Path, destination: Path) -> str:
    text = source.read_text(encoding="utf-8")
    needle = "V(out) <+ y;"
    if needle not in text:
        raise ValueError(f"Verilog-A output assignment not found: {source}")
    replacement = (
        "// Historical SSAC source used vac_V=1 while DEVICE used 1 mV.\n"
        "  // Report the corrected total-admittance apparent capacitance.\n"
        "  V(out) <+ 1000.0*y;"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.replace(needle, replacement), encoding="utf-8")
    return "veriloga"


def export_sources() -> list[Path]:
    sources: list[Path] = []
    primary = [
        ROOT
        / "artifacts/results/symbolic_gate_replacement"
        / name
        / "capacitance_symbolic_gated_kan/pure_symbolic_formula.json"
        for name in (
            "cap_teacher_curvature_budget24_seed42",
            "cap_teacher_curvature_budget24_studentseed43",
            "cap_teacher_curvature_budget24_studentseed44",
        )
    ]
    sources.extend(path for path in primary if path.exists())
    direct = ROOT / "artifacts/results/symbolic_export_audit/direct_baselines/Capacitance"
    multi = ROOT / "artifacts/results/multi_teacher_symbolic_pareto/exports/Capacitance"
    posterior = ROOT / "artifacts/results/posterior_formula_ensemble/formulas/Capacitance"
    if direct.exists():
        sources.extend(direct.glob("*.json"))
        sources.extend(direct.glob("*.va"))
    if multi.exists():
        sources.extend(multi.rglob("formula.json"))
        sources.extend(multi.rglob("formula.va"))
    if posterior.exists():
        sources.extend(posterior.rglob("pure_symbolic_formula.json"))
    return sorted(set(sources))


def build(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    legacy = pd.read_csv(source)
    audit = validate_legacy(legacy)
    corrected = corrected_frame(legacy)
    dataset = output / "primary_ge_si_apparent_capacitance.csv"
    corrected.to_csv(dataset, index=False)

    export_rows = []
    overlay_root = output / "formula_overlays"
    for formula in export_sources():
        relative = formula.relative_to(ROOT)
        destination = overlay_root / relative
        if formula.suffix.lower() == ".json":
            kind = correct_formula_json(formula, destination)
        elif formula.suffix.lower() == ".va":
            kind = correct_formula_va(formula, destination)
        else:
            continue
        export_rows.append(
            {
                "kind": kind,
                "source": relative.as_posix(),
                "source_sha256": sha256(formula),
                "corrected": destination.relative_to(ROOT).as_posix(),
                "corrected_sha256": sha256(destination),
            }
        )

    manifest = {
        "schema_version": "apparent_capacitance_correction_v1",
        "status": "passed",
        "target_name": "total-admittance apparent capacitance",
        "target_symbol": "C_app",
        "target_definition": "Im(Y_total)/(2*pi*f)",
        "not_displacement_capacitance": True,
        "not_terminal_charge_derivative": True,
        "source": str(source),
        "source_sha256": sha256(source),
        "canonical_dataset": dataset.relative_to(ROOT).as_posix(),
        "canonical_dataset_sha256": sha256(dataset),
        "audit": audit,
        "metric_transform": {
            "target_prediction_residual_rmse_mae_crps_mpiw_interval_score": "multiply by 1000",
            "raw_gaussian_nll": f"add ln(1000)={math.log(SCALE):.15g}",
            "r2_mape_coverage_conformal_multiplier_rankings_p_values": "unchanged",
        },
        "formula_overlay_count": len(export_rows),
        "formula_overlays": export_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(export_rows).to_csv(output / "formula_overlays.csv", index=False)

    readme = output / "README.md"
    readme.write_text(
        "# Apparent-capacitance correction\n\n"
        "The canonical target is `C_app = Im(Y_total)/(2*pi*f)`. The historical "
        "table recorded `vac_V=1` although DEVICE used 0.001 V, so all "
        "admittance-derived physical outputs are scaled by 1000. This is a "
        "deterministic source-unit correction, not a new TCAD or model fit. "
        "It is not displacement-current capacitance and is not used as "
        "terminal-charge evidence. See `manifest.json` for hashes and exact "
        "metric transformations.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
