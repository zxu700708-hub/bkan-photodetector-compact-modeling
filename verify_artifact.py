#!/usr/bin/env python3
"""Verify integrity, anonymity, and key frozen-claim invariants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


TEXT_SUFFIXES = {
    ".bib", ".cfg", ".cff", ".csv", ".json", ".md", ".py", ".scs",
    ".sha256", ".sh", ".tex", ".txt", ".va", ".yaml", ".yml",
}
FORBIDDEN_TEXT = (
    "E:" + "\\KAN",
    "E:" + "/KAN",
    "C:" + "\\Users\\",
    "E:" + "\\Data\\",
    "E:" + "\\\\Data\\\\",
    "/home/" + "cadence" + "zx",
    "cadence" + "zx",
)
CREDENTIAL_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*[\"']?[^\s\"']{6,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
)
LOCAL_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:[\\/](?:Users|KAN|Data)[\\/]"),
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
            failures.append({"path": relative, "reason": "sha256", "expected": expected, "actual": actual})
    return {"passed": not failures, "files": len(rows), "failures": failures}


def scan_anonymity(root: Path) -> dict:
    findings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "replay_output" in path.parts or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if any(token.lower() in relative.lower() for token in ("paper/notes", "credentials", "id_rsa")):
            findings.append({"path": relative, "reason": "forbidden_path"})
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in FORBIDDEN_TEXT:
            if token.lower() in text.lower():
                findings.append({"path": relative, "reason": "local_identity_or_absolute_path", "token": token})
        for pattern in CREDENTIAL_PATTERNS:
            if pattern.search(text):
                findings.append({"path": relative, "reason": "credential_pattern", "pattern": pattern.pattern})
        for pattern in LOCAL_PATH_PATTERNS:
            if pattern.search(text):
                findings.append({"path": relative, "reason": "local_absolute_path", "pattern": pattern.pattern})
    return {"passed": not findings, "findings": findings}


def check_grouped_accuracy(root: Path) -> dict:
    path = root / "artifacts/results/matched_grouped_comparison/metrics_by_seed.csv"
    if not path.is_file():
        return {"passed": False, "error": str(path.relative_to(root)) + " missing"}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    tasks = {row["task"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    models = {row["model"] for row in rows}
    expected_tasks = {"I_dark", "I_photo", "AC_Response", "Capacitance"}
    expected_models = {"bkan", "dkan", "mlp_l", "poly3_ridge", "spline_ridge"}
    correction = load_json(root / "artifacts/results/apparent_capacitance_correction/manifest.json")
    capacitance_scale = float(correction.get("audit", {}).get("scale_factor", 0.0))
    capacitance_bkan = [
        float(row["rmse_target"]) * capacitance_scale
        for row in rows
        if row["task"] == "Capacitance" and row["model"] == "bkan"
    ]
    capacitance_bkan_mean = statistics.mean(capacitance_bkan) if capacitance_bkan else None
    expected_capacitance_bkan_mean = 6.853730269380608e-17
    numeric_match = (
        capacitance_bkan_mean is not None
        and abs(capacitance_bkan_mean - expected_capacitance_bkan_mean) <= 1e-28
    )
    passed = (
        tasks == expected_tasks
        and seeds == set(range(42, 52))
        and models == expected_models
        and len(rows) == 200
        and correction.get("status") == "passed"
        and capacitance_scale == 1000.0
        and numeric_match
    )
    return {
        "passed": passed,
        "rows": len(rows),
        "tasks": sorted(tasks),
        "seeds": sorted(seeds),
        "models": sorted(models),
        "capacitance_scale_applied": capacitance_scale,
        "capacitance_bkan_rmse_mean_F": capacitance_bkan_mean,
        "expected_capacitance_bkan_rmse_mean_F": expected_capacitance_bkan_mean,
        "publication_value_match": numeric_match,
    }


def check_datasets(root: Path) -> dict:
    manifest_path = root / "data/dataset_manifest.json"
    if not manifest_path.is_file():
        return {"passed": False, "error": "data/dataset_manifest.json missing"}
    payload = load_json(manifest_path)
    rows = {item.get("dataset_id"): item for item in payload.get("datasets", [])}
    expected_rows = {
        "dark_current": 2400,
        "photo_current": 2560,
        "ac_response": 4480,
        "capacitance": 64000,
        "terminal_charge": 2080,
        "terminal_charge_dqdv_reference": 48,
        "condition_design": 160,
        "sampling_parameters": 5,
    }
    expected_apd = {
        "apd_dark_current",
        "apd_photo_current",
        "apd_net_photocurrent",
        "apd_multiplication_gain",
        "apd_gain_threshold_voltage",
        "apd_breakdown_voltage",
    }
    failures = []
    for dataset_id, expected in expected_rows.items():
        item = rows.get(dataset_id)
        if item is None:
            failures.append({"dataset": dataset_id, "reason": "missing_manifest_entry"})
            continue
        if item.get("rows") != expected:
            failures.append(
                {"dataset": dataset_id, "reason": "row_count", "expected": expected, "observed": item.get("rows")}
            )
    for dataset_id in expected_apd:
        if dataset_id not in rows:
            failures.append({"dataset": dataset_id, "reason": "missing_manifest_entry"})
    for dataset_id, item in rows.items():
        relative = item.get("path", "")
        path = root / "data" / relative
        if not path.is_file():
            failures.append({"dataset": dataset_id, "reason": "missing_file", "path": relative})
            continue
        observed = sha256(path)
        if observed != item.get("artifact_sha256"):
            failures.append(
                {
                    "dataset": dataset_id,
                    "reason": "sha256",
                    "expected": item.get("artifact_sha256"),
                    "observed": observed,
                }
            )
    required_docs = [
        root / "data/README.md",
        root / "data/DATA_DICTIONARY.csv",
        root / "DATA_LICENSE.md",
        root / "CITATION.cff",
    ]
    missing_docs = [path.relative_to(root).as_posix() for path in required_docs if not path.is_file()]
    if missing_docs:
        failures.append({"reason": "missing_data_documentation", "paths": missing_docs})
    return {
        "passed": not failures and payload.get("dataset_count") == 14 and len(rows) == 14,
        "dataset_count": len(rows),
        "expected_dataset_count": 14,
        "failures": failures,
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
        and checks.get("checkpoint_count") == 80
        and checks.get("prediction_count") == 80
    )
    return {"passed": passed, "status": payload.get("status"), "failures": payload.get("failures"), "checks": checks}


def check_symbolic_exports(root: Path) -> dict:
    audit_root = root / "artifacts/results/multi_teacher_symbolic_pareto_10split"
    export_root = audit_root
    json_count = len(list(export_root.rglob("formula.json"))) if export_root.is_dir() else 0
    va_count = len(list(export_root.rglob("formula.va"))) if export_root.is_dir() else 0
    metrics_path = audit_root / "metrics_by_export.csv"
    paired_path = audit_root / "paired_teacher_vs_direct.csv"
    metrics = list(csv.DictReader(metrics_path.open(encoding="utf-8", newline=""))) if metrics_path.is_file() else []
    paired = list(csv.DictReader(paired_path.open(encoding="utf-8", newline=""))) if paired_path.is_file() else []
    seeds = {row.get("seed") for row in metrics}
    finite = all(
        row.get(column) == "1.0"
        for row in metrics
        for column in (
            "test_finite_rate",
            "dense_finite_rate",
            "swept_axis_derivative_finite_rate",
        )
    )
    passed = (
        json_count == 960
        and va_count == 960
        and len(metrics) == 960
        and len(seeds) == 10
        and len(paired) == 72
        and finite
    )
    return {
        "passed": passed,
        "json_exports": json_count,
        "veriloga_exports": va_count,
        "metric_rows": len(metrics),
        "paired_contrasts": len(paired),
        "seeds": len(seeds),
        "all_registered_finite_rates_one": finite,
    }


def check_spectre(root: Path) -> dict:
    base = root / "artifacts/results/corrected_composite_spectre"
    required = {
        "device": base / "device_acceptance_report.json",
        "circuit": base / "circuit_acceptance_report.json",
        "provenance": base / "clean_rerun_provenance.json",
    }
    if any(not path.is_file() for path in required.values()):
        return {"passed": False, "missing": [str(path.relative_to(root)) for path in required.values() if not path.is_file()]}
    device = load_json(required["device"])
    circuit = load_json(required["circuit"])
    provenance = load_json(required["provenance"])
    model_hash = provenance.get("model_sha256")
    passed = (
        device.get("status") == "passed"
        and circuit.get("status") == "passed"
        and device.get("failed_checks") == []
        and circuit.get("failed_checks") == []
        and device.get("model_sha256") == model_hash
        and circuit.get("checks", {}).get("source_binding", {}).get("expected_sha256") == model_hash
        and provenance.get("status") == "clean_rerun_passed"
        and provenance.get("device_decks_rerun_after_verifier_fix") is True
        and provenance.get("spectre_results_reused") is False
        and provenance.get("device_internal_manifest_files") == 62
        and provenance.get("circuit_internal_manifest_files") == 62
    )
    return {
        "passed": passed,
        "model_sha256": model_hash,
        "device_status": device.get("status"),
        "circuit_status": circuit.get("status"),
        "device_decks_rerun_after_verifier_fix": provenance.get("device_decks_rerun_after_verifier_fix"),
        "spectre_results_reused": provenance.get("spectre_results_reused"),
    }


def check_release_scope(root: Path) -> dict:
    required = [
        root / "README.md",
        root / "REPRODUCIBILITY.md",
        root / "DATA_AVAILABILITY.md",
        root / "CITATION.cff",
        root / "data/dataset_manifest.json",
    ]
    paper = root / "paper"
    logs = [path.relative_to(root).as_posix() for path in root.rglob("*.log")]
    return {
        "passed": all(path.is_file() for path in required) and not paper.exists() and not logs,
        "required_present": [path.relative_to(root).as_posix() for path in required if path.is_file()],
        "paper_directory_absent": not paper.exists(),
        "log_files": logs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    checks = {
        "manifest": check_manifest(root),
        "release_hygiene": scan_anonymity(root),
        "release_scope": check_release_scope(root),
        "datasets": check_datasets(root),
        "grouped_accuracy": check_grouped_accuracy(root),
        "matched_uq": check_uq(root),
        "symbolic_exports": check_symbolic_exports(root),
        "spectre_clean_rerun": check_spectre(root),
    }
    passed = all(item.get("passed") is True for item in checks.values())
    report = {"schema_version": 1, "status": "passed" if passed else "failed", "root": ".", "checks": checks}
    output = args.output or (root / "replay_output/verification_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
