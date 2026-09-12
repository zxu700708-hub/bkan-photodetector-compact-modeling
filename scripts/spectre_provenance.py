"""Shared, live Spectre provenance evaluation.

Callers recompute this state from the archived proxy/DC source snapshots,
validated copies, comparison report, testbenches, and returned PSF/log
evidence. A cached freshness status is never authoritative, and this result
never transfers to the replacement terminal-charge source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The archived Spectre bundle predates the replacement terminal-charge source.
# Keep its state machine explicitly scoped so a generic "current" result can
# never be read as validation of the replacement model.
LEGACY_PROXY_SCOPE = "LEGACY_PROXY"
LEGACY_PROXY_PASS = "LEGACY_PROXY_CURRENT_NUMERIC_PASS"
LEGACY_PROXY_STALE = "LEGACY_PROXY_STALE_MODEL_NOT_VERIFIED"
REPLACEMENT_TERMINAL_Q_PASS = "CURRENT_REPLACEMENT_SPECTRE_DC_AC_TRANSIENT_PASS"
REPLACEMENT_TERMINAL_Q_PENDING = "PENDING_SPECTRE_DC_AC_TRANSIENT"
REPLACEMENT_TERMINAL_Q_CIRCUIT_PASS = (
    "CURRENT_REPLACEMENT_SPECTRE_BOUNDED_CIRCUIT_LEVEL_PASS"
)
REPLACEMENT_TERMINAL_Q_CIRCUIT_PENDING = (
    "PENDING_SPECTRE_BOUNDED_CIRCUIT_LEVEL"
)
# Backward-compatible paper/audit label for the currently registered artifact.
# The net-photocurrent correction changed the composite source hash, so the
# archived Spectre result cannot be transferred to that source.
REPLACEMENT_TERMINAL_Q_STATUS = REPLACEMENT_TERMINAL_Q_PENDING
REPLACEMENT_TERMINAL_Q_SOURCE = (
    "bkan/device_modeling/veriloga/"
    "ge_si_photodetector_terminal_charge.va"
)

MODEL_SPECS = {
    "independent_dc": {
        "filename": "ge_si_pdet_fixed.va",
        "report_key": "independent_dc_veriloga_model",
    },
    "charge_proxy": {
        "filename": "ge_si_photodetector_charge_proxy.va",
        "report_key": "charge_proxy_veriloga_model",
    },
}

TESTBENCH_SPECS = {
    "dc": "testbench_dc_ic618.scs",
    "dc_bias_temperature": "testbench_dc_bias_temp_proxy_ic618.scs",
    "ac_bias_temperature": "testbench_ac_bias_temp_proxy_ic618.scs",
    "transient": "testbench_transient_proxy_ic618.scs",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_replacement_terminal_q_spectre(root: Path = ROOT) -> dict:
    report_path = (
        root
        / "artifacts/results/terminal_charge_spectre/acceptance_report.json"
    )
    source_path = root / REPLACEMENT_TERMINAL_Q_SOURCE
    if not report_path.is_file() or not source_path.is_file():
        return {
            "source": REPLACEMENT_TERMINAL_Q_SOURCE,
            "status": REPLACEMENT_TERMINAL_Q_PENDING,
            "spectre_compiled": False,
            "dc_verified": False,
            "ac_admittance_verified": False,
            "transient_stability_verified": False,
            "evidence_transfer_allowed": False,
            "acceptance_report": str(report_path.relative_to(root)).replace("\\", "/"),
            "failures": ["missing_current_replacement_acceptance_or_source"],
        }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    binding = report.get("checks", {}).get("source_binding", {})
    simulator = report.get("simulator_execution_status", {})
    expected_hash = sha256_file(source_path)
    failures = []
    if report.get("status") != "passed":
        failures.append("acceptance_status_not_passed")
    if report.get("evidence_scope") != "CURRENT_REPLACEMENT_TERMINAL_Q":
        failures.append("wrong_evidence_scope")
    if report.get("legacy_proxy_evidence_transfer") is not False:
        failures.append("legacy_proxy_transfer_not_forbidden")
    if report.get("model_sha256") != expected_hash:
        failures.append("current_source_hash_mismatch")
    if not binding.get("passed"):
        failures.append("returned_source_binding_failed")
    required_status = (
        "spectre_compilation",
        "spectre_dc",
        "spectre_ac_admittance",
        "spectre_transient_stability",
    )
    for name in required_status:
        if simulator.get(name) != "passed":
            failures.append(f"simulator_status:{name}")

    evidence_root_value = binding.get("evidence_root")
    evidence_root = root / str(evidence_root_value or "")
    evidence_rows = binding.get("evidence_files", [])
    if not evidence_root_value or not evidence_rows:
        failures.append("missing_bound_result_evidence")
    else:
        try:
            evidence_root.resolve().relative_to(root.resolve())
        except ValueError:
            failures.append("evidence_root_outside_repository")
        for row in evidence_rows:
            path = evidence_root / str(row.get("path", ""))
            recorded = row.get("sha256")
            if not path.is_file() or not recorded or sha256_file(path) != recorded:
                failures.append(f"evidence_hash:{row.get('path')}")

    passed = not failures
    return {
        "source": REPLACEMENT_TERMINAL_Q_SOURCE,
        "status": (
            REPLACEMENT_TERMINAL_Q_PASS
            if passed
            else REPLACEMENT_TERMINAL_Q_PENDING
        ),
        "spectre_compiled": passed,
        "dc_verified": passed,
        "ac_admittance_verified": passed,
        "transient_stability_verified": passed,
        "evidence_transfer_allowed": False,
        "acceptance_report": str(report_path.relative_to(root)).replace("\\", "/"),
        "acceptance_report_sha256": sha256_file(report_path),
        "bound_evidence_files": len(evidence_rows),
        "failures": failures,
        "note": (
            "Current replacement source passed the bound Spectre DC, AC-"
            "admittance, and transient test envelope. This is not foundry or "
            "measurement qualification."
            if passed
            else "Current replacement Spectre evidence is incomplete or stale."
        ),
    }


def evaluate_terminal_q_circuit_spectre(root: Path = ROOT) -> dict:
    """Recompute freshness of the bounded external load/TIA Spectre evidence."""

    base = root / "artifacts/results/terminal_charge_circuit_validation"
    report_path = base / "circuit_acceptance_report.json"
    package_path = base / "package_manifest.json"
    source_path = root / REPLACEMENT_TERMINAL_Q_SOURCE
    required = (report_path, package_path, source_path)
    if not all(path.is_file() for path in required):
        return {
            "status": REPLACEMENT_TERMINAL_Q_CIRCUIT_PENDING,
            "passed": False,
            "failures": ["missing_report_manifest_or_current_source"],
        }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    failures = []
    source_hash = sha256_file(source_path)
    binding = report.get("checks", {}).get("source_binding", {})
    if report.get("status") != "passed" or report.get("failed_checks"):
        failures.append("acceptance_status_not_passed")
    if report.get("scope") != "bounded_electrical_circuit_level_validation_no_measurements":
        failures.append("wrong_circuit_evidence_scope")
    if not binding.get("passed") or binding.get("returned_sha256") != source_hash:
        failures.append("returned_source_binding_failed")
    for name in (
        "logs_and_convergence",
        "bias_load",
        "tia_ac",
        "tia_transient",
        "multi_instance",
    ):
        if not report.get("checks", {}).get(name, {}).get("passed"):
            failures.append(f"circuit_check:{name}")

    unsupported = set(report.get("claims_not_supported", []))
    for claim in (
        "measured_device_agreement",
        "optical_dynamic_port_response",
        "foundry_PDK_integration",
        "device_qualification",
        "deployment_ready_operation",
    ):
        if claim not in unsupported:
            failures.append(f"missing_claim_boundary:{claim}")

    evidence_root_value = report.get("evidence_root")
    evidence_rows = report.get("evidence_files", [])
    evidence_root = root / str(evidence_root_value or "")
    if not evidence_root_value or not evidence_rows:
        failures.append("missing_bound_circuit_evidence")
    else:
        try:
            evidence_root.resolve().relative_to(root.resolve())
        except ValueError:
            failures.append("circuit_evidence_root_outside_repository")
        for row in evidence_rows:
            path = evidence_root / str(row.get("path", ""))
            recorded = row.get("sha256")
            if not path.is_file() or not recorded or sha256_file(path) != recorded:
                failures.append(f"circuit_evidence_hash:{row.get('path')}")

    archive_value = package.get("returned_archive")
    archive = root / str(archive_value or "")
    if (
        package.get("status") != "passed_bound_spectre_circuit_level_validation"
        or package.get("model_sha256") != source_hash
        or package.get("acceptance_report_sha256") != sha256_file(report_path)
        or not archive_value
        or not archive.is_file()
        or package.get("returned_archive_sha256") != sha256_file(archive)
    ):
        failures.append("package_or_returned_archive_binding_failed")
    boundary = package.get("claim_boundary", {})
    if any(boundary.get(name) is not False for name in (
        "optical_dynamic_port_validated",
        "measurement_agreement",
        "pdk_integration",
        "device_qualified",
        "deployment_ready",
    )):
        failures.append("package_claim_boundary_failed")

    passed = not failures
    return {
        "status": (
            REPLACEMENT_TERMINAL_Q_CIRCUIT_PASS
            if passed
            else REPLACEMENT_TERMINAL_Q_CIRCUIT_PENDING
        ),
        "passed": passed,
        "source_sha256": source_hash,
        "acceptance_report": str(report_path.relative_to(root)).replace("\\", "/"),
        "acceptance_report_sha256": sha256_file(report_path),
        "returned_archive_sha256": package.get("returned_archive_sha256"),
        "bound_evidence_files": len(evidence_rows),
        "bias_load_points": report.get("checks", {}).get("bias_load", {}).get("points"),
        "tia_ac_points": report.get("checks", {}).get("tia_ac", {}).get("points"),
        "tia_transient_points": report.get("checks", {}).get("tia_transient", {}).get("points"),
        "multi_instance_counts": sorted(
            int(value)
            for value in report.get("checks", {}).get("multi_instance", {}).get("instances", {})
        ),
        "failures": failures,
        "note": (
            "Bounded electrical load/TIA and 1/10/100-instance Spectre tests "
            "passed. This is not optical-port, PDK, measurement, deployment-"
            "ready, or device-qualification evidence."
            if passed
            else "Circuit-level Spectre evidence is incomplete or stale."
        ),
    }


def _bound_file_status(
    root: Path,
    current_path: Path,
    validated_path: Path,
    recorded: dict,
) -> dict:
    current_hash = sha256_file(current_path) if current_path.exists() else None
    validated_hash = (
        sha256_file(validated_path) if validated_path.exists() else None
    )
    recorded_hash = recorded.get("sha256")
    return {
        "current": {
            "path": str(current_path.relative_to(root)).replace("\\", "/"),
            "sha256": current_hash,
        },
        "spectre_validated": {
            "path": str(validated_path.relative_to(root)).replace("\\", "/"),
            "sha256": validated_hash,
        },
        "report_recorded_sha256": recorded_hash,
        "hash_match": bool(
            current_hash
            and current_hash == validated_hash
            and current_hash == recorded_hash
        ),
    }


def evaluate_spectre_freshness(root: Path = ROOT) -> dict:
    report_path = (
        root
        / "artifacts/results/spectre_validation/"
        "spectre_comparison_report.json"
    )
    veriloga = root / "bkan/device_modeling/veriloga"
    result_dir = veriloga / "result"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    recorded_inputs = report.get("validated_inputs", {})

    models = {}
    for name, config in MODEL_SPECS.items():
        models[name] = _bound_file_status(
            root,
            veriloga / config["filename"],
            result_dir / config["filename"],
            recorded_inputs.get(config["report_key"], {}),
        )

    testbenches = {}
    for name, filename in TESTBENCH_SPECS.items():
        testbenches[name] = _bound_file_status(
            root,
            veriloga / filename,
            result_dir / filename,
            recorded_inputs.get(f"testbench_{name}", {}),
        )

    evidence_rows = []
    for item in report.get("result_evidence", []):
        relative = str(item.get("path", ""))
        path = result_dir / relative
        actual_hash = sha256_file(path) if path.is_file() else None
        recorded_hash = item.get("sha256")
        evidence_rows.append(
            {
                "path": relative,
                "recorded_sha256": recorded_hash,
                "actual_sha256": actual_hash,
                "hash_match": bool(
                    actual_hash and actual_hash == recorded_hash
                ),
            }
        )
    evidence_ok = bool(evidence_rows) and all(
        row["hash_match"] for row in evidence_rows
    )

    comparisons = report.get("comparisons", {})
    required_comparisons = (
        "dc_reference",
        "dc_bias_temperature",
        "ac_bias_temperature",
        "transient",
    )
    comparison_status = {
        name: str(comparisons.get(name, {}).get("status", "MISSING"))
        for name in required_comparisons
    }
    comparisons_ok = all(
        value == "NUMERIC_PASS" for value in comparison_status.values()
    )
    fresh = bool(
        report.get("overall_status") == "NUMERIC_PASS"
        and comparisons_ok
        and all(item["hash_match"] for item in models.values())
        and all(item["hash_match"] for item in testbenches.values())
        and evidence_ok
    )
    legacy_status = LEGACY_PROXY_PASS if fresh else LEGACY_PROXY_STALE
    replacement = evaluate_replacement_terminal_q_spectre(root)
    return {
        "schema_version": 4,
        "evidence_scope": LEGACY_PROXY_SCOPE,
        "status": legacy_status,
        "legacy_proxy_status": legacy_status,
        "replacement_terminal_q": replacement,
        "report_path": str(report_path.relative_to(root)).replace("\\", "/"),
        "report_overall_status": report.get("overall_status"),
        "report_generated_at": report.get("generated_at"),
        "models": models,
        "testbenches": testbenches,
        "comparison_status": comparison_status,
        "result_evidence": evidence_rows,
        "result_evidence_ok": evidence_ok,
    }
