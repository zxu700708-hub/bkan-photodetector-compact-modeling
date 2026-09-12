#!/usr/bin/env python3
"""Verify integrity, publication scope, and frozen-claim invariants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


TEXT_SUFFIXES = {
    ".cfg", ".cff", ".csv", ".json", ".md", ".py", ".scs",
    ".sha256", ".sh", ".txt", ".va", ".yaml", ".yml",
}
FORBIDDEN_TEXT = (
    "E:" + "\\KAN",
    "E:" + "/KAN",
    "C:" + "\\Users\\",
    "E:" + "\\Data\\",
    "/mnt/e/" + "KAN",
    "/mnt/e/" + "Data",
    "/home/" + "cadence" + "zx",
    "cadence" + "zx",
)
CREDENTIAL_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*[\"']?[^\s\"']{6,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
)
FORBIDDEN_PATH_PARTS = {
    "paper", "__pycache__", ".pytest_cache", "tmp", "rubbish",
}
FORBIDDEN_PATH_TOKENS = (
    "_smoke", "trial", "discarded", "incomplete", "eight_models",
    "six_models", "reviewer", "legacy", "hmc", "proxy_charge",
)
FORBIDDEN_SUFFIXES = {".log", ".pt", ".pth", ".pkl", ".joblib", ".zip", ".tar", ".gz", ".7z"}
ALLOWED_RESULT_DIRS = {
    "apparent_capacitance_correction",
    "compact_framework_seven_models_apd",
    "compact_framework_seven_models_pd",
    "corrected_composite_spectre",
    "evidence_audit",
    "matched_grouped_comparison",
    "matched_heteroscedastic_uq_equal_budget_current",
    "multi_teacher_symbolic_pareto_10split",
    "posterior_formula_ensemble",
    "ring_compact_variation_seven_no_gmls",
    "structured_generalization_capacitance",
    "structured_ood_uq",
    "symbolic_export_audit",
    "tcad_provenance",
    "terminal_charge_model",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_manifest(root: Path) -> dict:
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        return {"passed": False, "error": "MANIFEST.sha256 missing"}
    rows = []
    failures = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        actual = sha256(path)
        rows.append(relative)
        if actual != expected:
            failures.append(
                {"path": relative, "reason": "sha256", "expected": expected, "actual": actual}
            )
    return {"passed": not failures, "files": len(rows), "failures": failures}


def check_release_hygiene(root: Path) -> dict:
    findings = []
    result_root = root / "artifacts/results"
    if result_root.is_dir():
        observed = {path.name for path in result_root.iterdir() if path.is_dir()}
        unexpected = sorted(observed - ALLOWED_RESULT_DIRS)
        if unexpected:
            findings.append({"reason": "unexpected_result_directories", "paths": unexpected})
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "replay_output" in path.parts or ".git" in path.parts:
            continue
        relative_path = path.relative_to(root)
        relative = relative_path.as_posix()
        lowered_parts = {part.lower() for part in relative_path.parts}
        lowered = relative.lower()
        if lowered_parts & FORBIDDEN_PATH_PARTS:
            findings.append({"path": relative, "reason": "forbidden_path_part"})
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            findings.append({"path": relative, "reason": "superseded_or_nonpaper_path"})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append({"path": relative, "reason": "forbidden_file_type"})
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in FORBIDDEN_TEXT:
            if token.lower() in text.lower():
                findings.append({"path": relative, "reason": "local_absolute_path", "token": token})
        for pattern in CREDENTIAL_PATTERNS:
            if pattern.search(text):
                findings.append({"path": relative, "reason": "credential_pattern", "pattern": pattern.pattern})
    required = [
        root / "README.md",
        root / "REPRODUCIBILITY.md",
        root / "DATA_AVAILABILITY.md",
        root / "CITATION.cff",
        root / "artifacts/results/evidence_audit/current_evidence_index.json",
        root / "scripts/analyze_ring_derived_metrics.py",
        root / "scripts/experiment_ring_resonance_aware_dkan.py",
        root / "scripts/manuscript_plot_palette.py",
        root / "scripts/run_ac_lowpass_gated_teacher.py",
        root / "scripts/run_multi_teacher_symbolic_pareto.py",
        root / "scripts/run_ring_third_device.py",
    ]
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
    if missing:
        findings.append({"reason": "missing_required_document_or_dependency", "paths": missing})
    return {"passed": not findings, "findings": findings}


def check_datasets(root: Path) -> dict:
    manifest_path = root / "data/dataset_manifest.json"
    if not manifest_path.is_file():
        return {"passed": False, "error": "data/dataset_manifest.json missing"}
    payload = load_json(manifest_path)
    entries = {item.get("dataset_id"): item for item in payload.get("datasets", [])}
    expected_rows = {
        "dark_current": 2400,
        "photo_current": 2560,
        "ac_response": 4480,
        "capacitance": 64000,
        "terminal_charge": 2080,
        "terminal_charge_dqdv_reference": 48,
        "condition_design": 160,
        "sampling_parameters": 5,
        "apd_dark_current": 6162,
        "apd_photo_current": 12480,
        "apd_net_photocurrent": 12480,
        "apd_multiplication_gain": 12480,
        "apd_gain_threshold_voltage": 155,
        "apd_breakdown_voltage": 155,
        "ring_development_targets": 14400,
        "ring_nominal_structures": 400,
        "ring_process_realizations": 1200,
        "ring_frozen_confirmation_ids": 80,
    }
    failures = []
    for dataset_id, expected in expected_rows.items():
        item = entries.get(dataset_id)
        if item is None:
            failures.append({"dataset": dataset_id, "reason": "missing_manifest_entry"})
            continue
        if item.get("rows") != expected:
            failures.append(
                {"dataset": dataset_id, "reason": "row_count", "expected": expected, "observed": item.get("rows")}
            )
        path = root / "data" / item.get("path", "")
        if not path.is_file():
            failures.append({"dataset": dataset_id, "reason": "missing_file"})
        elif sha256(path) != item.get("artifact_sha256"):
            failures.append({"dataset": dataset_id, "reason": "sha256"})

    ring_targets_path = root / "data/ring/ring_development_targets.csv"
    ring_ids_path = root / "data/ring/ring_frozen_confirmation_ids.csv"
    public_manifest_path = root / "data/ring/ring_public_manifest.json"
    if ring_targets_path.is_file() and ring_ids_path.is_file() and public_manifest_path.is_file():
        ring_rows = load_csv(ring_targets_path)
        confirmation_rows = load_csv(ring_ids_path)
        ring_ids = [row["nominal_structure_id"] for row in ring_rows]
        confirmation_ids = {row["nominal_structure_id"] for row in confirmation_rows}
        target_columns = set(ring_rows[0]) if ring_rows else set()
        confirmation_columns = set(confirmation_rows[0]) if confirmation_rows else set()
        forbidden_confirmation_columns = {
            "resonance_shift_pm", "quality_factor", "drop_extinction_ratio_db",
            "through_insertion_loss_db",
        }
        counts = Counter(ring_ids)
        public_manifest = load_json(public_manifest_path)
        boundary = public_manifest.get("publication_boundary", {})
        if (
            len(counts) != 320
            or set(counts.values()) != {45}
            or set(ring_ids) & confirmation_ids
            or not set(("resonance_shift_pm", "quality_factor", "drop_extinction_ratio_db", "through_insertion_loss_db")).issubset(target_columns)
            or confirmation_columns & forbidden_confirmation_columns
            or boundary.get("frozen_confirmation_targets_published") is not False
            or boundary.get("reported_comparison_used_confirmation_targets") is not False
        ):
            failures.append({"dataset": "ring", "reason": "confirmation_quarantine_or_grouping"})
    else:
        failures.append({"dataset": "ring", "reason": "missing_public_ring_files"})

    return {
        "passed": not failures and payload.get("dataset_count") == 18 and len(entries) == 18,
        "dataset_count": len(entries),
        "expected_dataset_count": 18,
        "failures": failures,
    }


def check_model_table(
    root: Path,
    relative: str,
    expected_rows: int,
    task_column: str,
    expected_tasks: set[str],
    expected_models: set[str],
    expected_seeds: set[int],
) -> dict:
    path = root / relative
    if not path.is_file():
        return {"passed": False, "error": relative + " missing"}
    rows = load_csv(path)
    tasks = {row[task_column] for row in rows}
    models = {row["model"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    passed = (
        len(rows) == expected_rows
        and tasks == expected_tasks
        and models == expected_models
        and seeds == expected_seeds
    )
    return {
        "passed": passed,
        "rows": len(rows),
        "tasks": sorted(tasks),
        "models": sorted(models),
        "seeds": sorted(seeds),
    }


def check_predictive_results(root: Path) -> dict:
    five = check_model_table(
        root,
        "artifacts/results/matched_grouped_comparison/metrics_by_seed.csv",
        200,
        "task",
        {"I_dark", "I_photo", "AC_Response", "Capacitance"},
        {"bkan", "dkan", "mlp_l", "poly3_ridge", "spline_ridge"},
        set(range(42, 52)),
    )
    seven_models = {"bkan", "dkan", "mlp_l", "spline_ridge", "gmls", "autopinn", "curve_lut"}
    pd_seven = check_model_table(
        root,
        "artifacts/results/compact_framework_seven_models_pd/metrics_by_seed.csv",
        280,
        "task",
        {"I_dark", "I_photo", "AC_Response", "Capacitance"},
        seven_models,
        set(range(42, 52)),
    )
    apd_seven = check_model_table(
        root,
        "artifacts/results/compact_framework_seven_models_apd/metrics_by_seed.csv",
        420,
        "task",
        {"APD_I_dark", "APD_I_photo", "APD_I_net", "APD_M", "APD_V_M10", "APD_V_br"},
        seven_models,
        set(range(42, 52)),
    )
    ring_models = {"bkan", "dkan", "mlp_l", "spline_ridge", "autopinn", "curve_lut", "semiempirical"}
    ring = check_model_table(
        root,
        "artifacts/results/ring_compact_variation_seven_no_gmls/metrics_by_seed.csv",
        28,
        "task_key",
        {"resonance_shift", "quality_factor", "drop_extinction", "through_insertion"},
        ring_models,
        {42},
    )
    ring_root = root / "artifacts/results/ring_compact_variation_seven_no_gmls"
    ring_audit = load_json(ring_root / "audit.json") if (ring_root / "audit.json").is_file() else {}
    ring_predictions = len(list((ring_root / "predictions").rglob("test_predictions.csv")))
    ring["prediction_files"] = ring_predictions
    ring["audit_status"] = ring_audit.get("status")
    ring["passed"] = ring["passed"] and ring_predictions == 28 and ring_audit.get("status") == "PASS"
    return {
        "passed": all(item["passed"] for item in (five, pd_seven, apd_seven, ring)),
        "prespecified_five_model_primary": five,
        "seven_model_primary": pd_seven,
        "seven_model_apd": apd_seven,
        "seven_model_ring_development": ring,
    }


def check_uq(root: Path) -> dict:
    path = root / "artifacts/results/matched_heteroscedastic_uq_equal_budget_current/acceptance_audit.json"
    if not path.is_file():
        return {"passed": False, "error": str(path.relative_to(root)) + " missing"}
    payload = load_json(path)
    checks = payload.get("checks", {})
    passed = (
        payload.get("status") == "passed"
        and payload.get("failures") == []
        and checks.get("metrics_rows", {}).get("observed") == 120
        and checks.get("baseline_rows", {}).get("observed") == 80
        and checks.get("summary_rows", {}).get("observed") == 12
        and checks.get("paired_rows", {}).get("observed") == 72
        and checks.get("split_overlap_count") == 0
        and checks.get("baseline_budget_audits_pass") is True
        and checks.get("total_update_ratio_matches_declared_budget") is True
        and checks.get("prediction_count") == 80
    )
    return {"passed": passed, "status": payload.get("status"), "checks": checks}


def check_symbolic(root: Path) -> dict:
    audit_root = root / "artifacts/results/multi_teacher_symbolic_pareto_10split"
    formulas = list(audit_root.rglob("formula.json")) if audit_root.is_dir() else []
    sources = list(audit_root.rglob("formula.va")) if audit_root.is_dir() else []
    metrics = load_csv(audit_root / "metrics_by_export.csv") if (audit_root / "metrics_by_export.csv").is_file() else []
    paired = load_csv(audit_root / "paired_teacher_vs_direct.csv") if (audit_root / "paired_teacher_vs_direct.csv").is_file() else []
    finite = all(
        row.get(column) == "1.0"
        for row in metrics
        for column in ("test_finite_rate", "dense_finite_rate", "swept_axis_derivative_finite_rate")
    )
    ensemble_root = root / "artifacts/results/posterior_formula_ensemble"
    ensemble_required = [
        ensemble_root / "formula_ensemble_calibration.csv",
        ensemble_root / "formula_metrics_by_posterior_draw.csv",
        ensemble_root / "formula_stability_summary.csv",
        ensemble_root / "protocol.json",
    ]
    passed = (
        len(formulas) == 960
        and len(sources) == 960
        and len(metrics) == 960
        and len({row.get("seed") for row in metrics}) == 10
        and len(paired) == 72
        and finite
        and all(path.is_file() for path in ensemble_required)
    )
    return {
        "passed": passed,
        "json_exports": len(formulas),
        "veriloga_exports": len(sources),
        "metric_rows": len(metrics),
        "paired_contrasts": len(paired),
        "posterior_formula_summaries_present": all(path.is_file() for path in ensemble_required),
    }


def check_terminal_and_spectre(root: Path) -> dict:
    terminal_root = root / "artifacts/results/terminal_charge_model"
    terminal = load_json(terminal_root / "terminal_charge_audit.json")
    numerical = load_json(terminal_root / "reference_validation/reference_validation_summary.json")
    base = root / "artifacts/results/corrected_composite_spectre"
    device = load_json(base / "device_acceptance_report.json")
    circuit = load_json(base / "circuit_acceptance_report.json")
    provenance = load_json(base / "clean_rerun_provenance.json")
    model_hash = provenance.get("model_sha256")
    passed = (
        terminal.get("audit_status") == "passed"
        and terminal.get("production_rows") == 2080
        and terminal.get("production_conditions") == 160
        and numerical.get("validation_status") == "passed"
        and device.get("status") == "passed"
        and circuit.get("status") == "passed"
        and device.get("model_sha256") == model_hash
        and circuit.get("checks", {}).get("source_binding", {}).get("expected_sha256") == model_hash
        and provenance.get("status") == "clean_rerun_passed"
        and provenance.get("device_decks_rerun_after_verifier_fix") is True
        and provenance.get("spectre_results_reused") is False
    )
    return {
        "passed": passed,
        "terminal_audit": terminal.get("audit_status"),
        "numerical_replay": numerical.get("validation_status"),
        "device_status": device.get("status"),
        "circuit_status": circuit.get("status"),
        "model_sha256": model_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    checks = {
        "manifest": check_manifest(root),
        "release_hygiene": check_release_hygiene(root),
        "datasets": check_datasets(root),
        "predictive_results": check_predictive_results(root),
        "matched_uq": check_uq(root),
        "symbolic": check_symbolic(root),
        "terminal_and_spectre": check_terminal_and_spectre(root),
    }
    passed = all(item.get("passed") is True for item in checks.values())
    report = {"schema_version": 2, "status": "passed" if passed else "failed", "root": ".", "checks": checks}
    output = args.output or (root / "replay_output/verification_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
