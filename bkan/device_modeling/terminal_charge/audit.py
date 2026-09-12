"""Audit production terminal-charge data against strict DEVICE evidence.

The upstream certificate checks validation gates and parameter-space coverage.
This module additionally requires numerical agreement between strict evidence
and the production Q-V table at identical parameter/bias points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PARAM_COLS = [
    "trap_assisted_recomb_A",
    "ge_sio2_recomb_velocity",
    "ge_si_recomb_velocity",
    "active_layer_length",
    "simulation_temperature",
]

STRICT_GATE_COLS = [
    "quasi_static_terminal_charge_density_identity_pass",
    "quasi_static_terminal_charge_incremental_gauss_pass",
    "quasi_static_terminal_charge_incremental_mesh_convergence_pass",
    "quasi_static_terminal_charge_incremental_method_agreement_pass",
    "compact_port_dQdV_small_signal_validation_pass",
    "quasi_static_compact_port_dQdV_small_signal_pass",
]

Q_COL = "compact_port_positive_charge_C"
BIAS_COL = "terminal_charge_bias_v"
C_COL = "compact_port_capacitance_from_two_contact_ssac_F"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_semicolon(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def certificate_evidence(certificate_path: Path) -> tuple[list[Path], list[str], dict]:
    certificate = pd.read_csv(certificate_path)
    if len(certificate) != 1:
        raise ValueError("Terminal-charge certificate must contain exactly one row.")
    row = certificate.iloc[0]
    paths = [Path(value) for value in _split_semicolon(row["validation_evidence_files"])]
    hashes = _split_semicolon(row["validation_evidence_sha256"])
    if len(paths) != len(hashes):
        raise ValueError("Certificate evidence path/hash counts differ.")
    return paths, hashes, row.to_dict()


def _unique_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [*PARAM_COLS]
    if "production_source_row" in frame.columns:
        columns.append("production_source_row")
    return frame[columns].drop_duplicates(PARAM_COLS).reset_index(drop=True)


def _parameter_scales(conditions: pd.DataFrame) -> np.ndarray:
    values = conditions[PARAM_COLS].to_numpy(dtype=np.float64)
    scales = np.ptp(values, axis=0)
    fallback = np.maximum(np.max(np.abs(values), axis=0), 1.0)
    return np.where(scales > 0.0, scales, fallback)


def map_evidence_conditions(
    production: pd.DataFrame,
    evidence: pd.DataFrame,
    exact_distance_limit: float = 1.0e-8,
) -> pd.DataFrame:
    production_conditions = _unique_conditions(production)
    evidence_conditions = evidence[PARAM_COLS].drop_duplicates().reset_index(drop=True)
    scales = _parameter_scales(production_conditions)
    production_values = production_conditions[PARAM_COLS].to_numpy(dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for evidence_group, condition in evidence_conditions.iterrows():
        values = condition[PARAM_COLS].to_numpy(dtype=np.float64)
        distances = np.sqrt(np.sum(((production_values - values) / scales) ** 2, axis=1))
        match_index = int(np.argmin(distances))
        distance = float(distances[match_index])
        if distance > exact_distance_limit:
            raise ValueError(
                f"Evidence condition {evidence_group} has no exact production match; "
                f"nearest normalized distance={distance:.6g}."
            )
        match = production_conditions.iloc[match_index]
        row: dict[str, float | int] = {
            "evidence_group": int(evidence_group),
            "parameter_match_distance": distance,
        }
        for column in PARAM_COLS:
            row[column] = float(condition[column])
        row["production_source_row"] = int(
            match.get("production_source_row", match_index)
        )
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class AuditResult:
    summary: dict[str, object]
    condition_table: pd.DataFrame
    evidence: pd.DataFrame


def audit_terminal_charge(
    production_path: Path,
    evidence_paths: Iterable[Path] | None = None,
    certificate_path: Path | None = None,
    q_relative_tolerance: float = 0.02,
    q_absolute_tolerance_C: float = 1.0e-18,
    coverage_distance_limit: float = 0.5,
    minimum_validation_conditions: int = 16,
) -> AuditResult:
    production_path = production_path.resolve()
    production = pd.read_csv(production_path)
    required_production = {
        "bias_v",
        Q_COL,
        "production_terminal_charge_extraction_eligible",
        *PARAM_COLS,
    }
    missing = sorted(required_production.difference(production.columns))
    if missing:
        raise ValueError(f"Production table is missing columns: {missing}")
    if not production["production_terminal_charge_extraction_eligible"].eq(1).all():
        raise ValueError("Production table contains extraction-ineligible rows.")
    if production.duplicated([*PARAM_COLS, "bias_v"]).any():
        raise ValueError("Production table has duplicate parameter/bias keys.")

    certificate_row: dict[str, object] | None = None
    expected_hashes: list[str] | None = None
    if certificate_path is not None:
        certified_paths, expected_hashes, certificate_row = certificate_evidence(
            certificate_path.resolve()
        )
        if str(certificate_row.get("certification_status", "")).lower() != "certified":
            raise ValueError("Certificate status is not 'certified'.")
        supplied = [Path(path).resolve() for path in evidence_paths or certified_paths]
        certified = [path.resolve() for path in certified_paths]
        if supplied != certified:
            raise ValueError("Supplied evidence paths do not exactly match the certificate.")
        evidence_paths = supplied
    else:
        evidence_paths = [Path(path).resolve() for path in evidence_paths or []]

    evidence_paths = list(evidence_paths)
    if not evidence_paths:
        raise ValueError("At least one strict validation evidence file is required.")

    evidence_frames: list[pd.DataFrame] = []
    observed_hashes: list[str] = []
    for index, path in enumerate(evidence_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        observed_hashes.append(observed_hash)
        if expected_hashes is not None and observed_hash.lower() != expected_hashes[index].lower():
            raise ValueError(f"Certificate SHA-256 mismatch for {path}")
        frame = pd.read_csv(path)
        required_evidence = {BIAS_COL, Q_COL, C_COL, *PARAM_COLS, *STRICT_GATE_COLS}
        missing = sorted(required_evidence.difference(frame.columns))
        if missing:
            raise ValueError(f"Evidence {path} is missing columns: {missing}")
        frame = frame.copy()
        frame["evidence_path"] = str(path)
        frame["evidence_file_index"] = index
        evidence_frames.append(frame)
    evidence = pd.concat(evidence_frames, ignore_index=True)

    for gate in STRICT_GATE_COLS:
        if not evidence[gate].eq(1).all():
            failed = int((~evidence[gate].eq(1)).sum())
            raise ValueError(f"Strict gate {gate} failed on {failed} evidence rows.")
    if "compact_port_negative_charge_C" in evidence.columns:
        residual = evidence[Q_COL] + evidence["compact_port_negative_charge_C"]
        if float(residual.abs().max()) > q_absolute_tolerance_C:
            raise ValueError("Strict evidence violates equal-opposite compact-port charge.")

    mapping = map_evidence_conditions(production, evidence)
    evidence = evidence.merge(mapping, on=PARAM_COLS, how="left", validate="many_to_one")
    evidence["bias_v"] = evidence[BIAS_COL].astype(float)
    production_keys = production[["production_source_row", "bias_v", Q_COL]].rename(
        columns={Q_COL: "production_charge_C"}
    )
    comparison = evidence.merge(
        production_keys,
        on=["production_source_row", "bias_v"],
        how="left",
        validate="many_to_one",
    )
    if comparison["production_charge_C"].isna().any():
        raise ValueError("Evidence contains bias points absent from the production Q-V grid.")
    comparison["charge_error_C"] = comparison[Q_COL] - comparison["production_charge_C"]
    comparison["charge_relative_error"] = comparison["charge_error_C"].abs() / np.maximum.reduce(
        [
            comparison[Q_COL].abs().to_numpy(),
            comparison["production_charge_C"].abs().to_numpy(),
            np.full(len(comparison), q_absolute_tolerance_C),
        ]
    )
    comparison["charge_agreement_pass"] = (
        comparison["charge_error_C"].abs()
        <= q_absolute_tolerance_C
        + q_relative_tolerance * np.maximum(
            comparison[Q_COL].abs(), comparison["production_charge_C"].abs()
        )
    )

    condition_table = (
        comparison.groupby("production_source_row", as_index=False)
        .agg(
            evidence_rows=(Q_COL, "size"),
            bias_min_V=("bias_v", "min"),
            bias_max_V=("bias_v", "max"),
            max_charge_abs_error_C=("charge_error_C", lambda values: float(np.max(np.abs(values)))),
            max_charge_relative_error=("charge_relative_error", "max"),
            charge_agreement_pass=("charge_agreement_pass", "all"),
            max_reported_dqdv_relative_error=(
                "compact_port_dQdV_small_signal_relative_error",
                "max",
            ),
        )
        .sort_values("production_source_row")
        .reset_index(drop=True)
    )
    bias_min = float(production["bias_v"].min())
    bias_max = float(production["bias_v"].max())
    condition_table["full_bias_range_pass"] = np.isclose(
        condition_table["bias_min_V"], bias_min
    ) & np.isclose(condition_table["bias_max_V"], bias_max)
    if not condition_table["charge_agreement_pass"].all():
        failed = condition_table.loc[
            ~condition_table["charge_agreement_pass"], "production_source_row"
        ].tolist()
        raise ValueError(f"Strict/production Q agreement failed for source rows {failed}.")
    if not condition_table["full_bias_range_pass"].all():
        raise ValueError("One or more strict validation groups do not span the production bias range.")
    if len(condition_table) < minimum_validation_conditions:
        raise ValueError(
            f"Only {len(condition_table)} strict conditions; minimum is "
            f"{minimum_validation_conditions}."
        )

    production_conditions = _unique_conditions(production)
    scales = _parameter_scales(production_conditions)
    validated = production_conditions.merge(
        condition_table[["production_source_row"]],
        on="production_source_row",
        how="inner",
    )
    all_values = production_conditions[PARAM_COLS].to_numpy(dtype=float)
    valid_values = validated[PARAM_COLS].to_numpy(dtype=float)
    coverage = np.min(
        np.sqrt(np.sum(((all_values[:, None, :] - valid_values[None, :, :]) / scales) ** 2, axis=2)),
        axis=1,
    )
    max_coverage = float(np.max(coverage))
    if max_coverage > coverage_distance_limit:
        raise ValueError(
            f"Validation coverage failed: {max_coverage:.6g} > {coverage_distance_limit:.6g}."
        )

    summary: dict[str, object] = {
        "audit_status": "passed",
        "production_path": str(production_path),
        "production_sha256": sha256_file(production_path),
        "production_rows": int(len(production)),
        "production_conditions": int(len(production_conditions)),
        "evidence_paths": [str(path) for path in evidence_paths],
        "evidence_sha256": observed_hashes,
        "validated_conditions": int(len(condition_table)),
        "strict_evidence_rows": int(len(comparison)),
        "maximum_strict_production_q_relative_error": float(
            comparison["charge_relative_error"].max()
        ),
        "maximum_strict_production_q_absolute_error_C": float(
            comparison["charge_error_C"].abs().max()
        ),
        "strict_ssac_dqdv_relative_error_median": float(
            comparison["compact_port_dQdV_small_signal_relative_error"].median()
        ),
        "strict_ssac_dqdv_relative_error_p95": float(
            comparison["compact_port_dQdV_small_signal_relative_error"].quantile(0.95)
        ),
        "strict_ssac_dqdv_relative_error_max": float(
            comparison["compact_port_dQdV_small_signal_relative_error"].max()
        ),
        "maximum_normalized_coverage_distance": max_coverage,
        "coverage_distance_limit": float(coverage_distance_limit),
        "q_relative_tolerance": float(q_relative_tolerance),
        "q_absolute_tolerance_C": float(q_absolute_tolerance_C),
        "certificate_path": str(certificate_path.resolve()) if certificate_path else None,
        "certificate_sha256": sha256_file(certificate_path.resolve()) if certificate_path else None,
    }
    return AuditResult(summary, condition_table, comparison)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--evidence", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--q-absolute-tolerance-C", type=float, default=1.0e-18)
    parser.add_argument("--coverage-distance-limit", type=float, default=0.5)
    parser.add_argument("--minimum-validation-conditions", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_terminal_charge(
        args.production,
        args.evidence,
        args.certificate,
        args.q_relative_tolerance,
        args.q_absolute_tolerance_C,
        args.coverage_distance_limit,
        args.minimum_validation_conditions,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "terminal_charge_audit.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result.condition_table.to_csv(args.output / "terminal_charge_audit_conditions.csv", index=False)
    result.evidence.to_csv(args.output / "terminal_charge_audit_evidence.csv", index=False)
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
