#!/usr/bin/env python3
"""Reaggregate the manuscript-facing summaries from frozen current evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite_numbers(value) -> tuple[int, int]:
    total = 0
    invalid = 0
    if isinstance(value, bool) or value is None:
        return total, invalid
    if isinstance(value, (int, float)):
        return 1, 0 if math.isfinite(float(value)) else 1
    if isinstance(value, dict):
        for child in value.values():
            child_total, child_invalid = finite_numbers(child)
            total += child_total
            invalid += child_invalid
    elif isinstance(value, list):
        for child in value:
            child_total, child_invalid = finite_numbers(child)
            total += child_total
            invalid += child_invalid
    return total, invalid


def aggregate_rmse(
    path: Path,
    *,
    task_column: str = "task",
    value_column: str = "rmse_target",
) -> dict:
    rows = load_csv(path)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row[task_column], row["model"])].append(float(row[value_column]))
    summary = []
    for (task, model), values in sorted(grouped.items()):
        summary.append(
            {
                "task": task,
                "model": model,
                "splits": len(values),
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            }
        )
    return {"source_rows": len(rows), "summary": summary}


def replay_predictive_tables(root: Path) -> dict:
    primary = aggregate_rmse(
        root / "artifacts/results/compact_framework_seven_models_pd/metrics_by_seed.csv"
    )
    apd = aggregate_rmse(
        root / "artifacts/results/compact_framework_seven_models_apd/metrics_by_seed.csv"
    )
    ring_path = root / "artifacts/results/ring_compact_variation_seven_no_gmls/metrics_by_seed.csv"
    ring_rows = load_csv(ring_path)
    ring_metric_columns = {
        "resonance_shift": "nonzero_bias_rmse_raw",
        "quality_factor": "median_ape_percent",
        "drop_extinction": "mae_raw",
        "through_insertion": "median_ape_percent",
    }
    ring_summary = []
    for row in sorted(ring_rows, key=lambda item: (item["task_key"], item["model"])):
        metric = ring_metric_columns[row["task_key"]]
        ring_summary.append(
            {
                "task": row["task_key"],
                "model": row["model"],
                "metric": metric,
                "value": float(row[metric]),
            }
        )
    passed = (
        primary["source_rows"] == 280
        and len(primary["summary"]) == 28
        and all(item["splits"] == 10 for item in primary["summary"])
        and apd["source_rows"] == 420
        and len(apd["summary"]) == 42
        and all(item["splits"] == 10 for item in apd["summary"])
        and len(ring_rows) == 28
        and len(ring_summary) == 28
    )
    return {
        "passed": passed,
        "primary_seven_model": primary,
        "apd_seven_model": apd,
        "ring_development_seven_model": {"source_rows": len(ring_rows), "summary": ring_summary},
    }


def replay_uq(root: Path) -> dict:
    base = root / "artifacts/results/matched_heteroscedastic_uq_equal_budget_current"
    acceptance = load_json(base / "acceptance_audit.json")
    rows = load_csv(base / "metrics_by_seed.csv")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["model"])].append(float(row["rmse_model_space"]))
    summary = [
        {
            "task": task,
            "model": model,
            "splits": len(values),
            "rmse_mean": statistics.mean(values),
        }
        for (task, model), values in sorted(grouped.items())
    ]
    checks = acceptance.get("checks", {})
    passed = (
        acceptance.get("status") == "passed"
        and acceptance.get("failures") == []
        and len(rows) == 120
        and len(summary) == 12
        and all(item["splits"] == 10 for item in summary)
        and checks.get("split_overlap_count") == 0
        and checks.get("baseline_budget_audits_pass") is True
        and checks.get("total_update_ratio_matches_declared_budget") is True
    )
    return {"passed": passed, "source_rows": len(rows), "summary": summary}


def replay_symbolic(root: Path) -> dict:
    export_root = root / "artifacts/results/multi_teacher_symbolic_pareto_10split"
    formulas = sorted(export_root.rglob("formula.json"))
    sources = sorted(export_root.rglob("formula.va"))
    numeric_values = 0
    invalid_values = 0
    parse_failures = []
    for path in formulas:
        try:
            total, invalid = finite_numbers(load_json(path))
            numeric_values += total
            invalid_values += invalid
        except Exception as exc:
            parse_failures.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
    metrics = load_csv(export_root / "metrics_by_export.csv")
    paired = load_csv(export_root / "paired_teacher_vs_direct.csv")
    ensemble = load_csv(
        root / "artifacts/results/posterior_formula_ensemble/formula_ensemble_calibration.csv"
    )
    capacitance = next((row for row in ensemble if row.get("task") == "Capacitance"), None)
    passed = (
        len(formulas) == 960
        and len(sources) == 960
        and len(metrics) == 960
        and len(paired) == 72
        and invalid_values == 0
        and not parse_failures
        and capacitance is not None
        and abs(float(capacitance["test_point_coverage_90"]) - 99.90625) < 1e-9
        and abs(float(capacitance["test_group_coverage_90"]) - 75.0) < 1e-9
    )
    return {
        "passed": passed,
        "json_exports": len(formulas),
        "veriloga_exports": len(sources),
        "metric_rows": len(metrics),
        "paired_contrasts": len(paired),
        "numeric_values_checked": numeric_values,
        "invalid_numeric_values": invalid_values,
        "capacitance_formula_ensemble": capacitance,
        "parse_failures": parse_failures,
    }


def replay_ood(root: Path) -> dict:
    base = root / "artifacts/results/structured_ood_uq"
    metric_rows = load_csv(base / "metrics_by_run_domain.csv")
    shifts = load_csv(base / "uncertainty_shift_by_run.csv")
    protocol = load_json(base / "protocol.json")
    passed = (
        bool(metric_rows)
        and bool(shifts)
        and protocol.get("seeds") == [42, 43, 44]
        and {row["task"] for row in metric_rows} == {"dark_current", "photo_current", "ac_response", "capacitance"}
    )
    return {
        "passed": passed,
        "metric_rows": len(metric_rows),
        "uncertainty_shift_rows": len(shifts),
        "seeds": protocol.get("seeds"),
    }


def replay_terminal_and_spectre(root: Path) -> dict:
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
        and numerical.get("validation_status") == "passed"
        and device.get("status") == "passed"
        and circuit.get("status") == "passed"
        and device.get("model_sha256") == model_hash
        and circuit.get("checks", {}).get("source_binding", {}).get("expected_sha256") == model_hash
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
        "predictive_tables": replay_predictive_tables(root),
        "matched_uq": replay_uq(root),
        "symbolic": replay_symbolic(root),
        "structured_ood": replay_ood(root),
        "terminal_and_spectre": replay_terminal_and_spectre(root),
    }
    passed = all(item.get("passed") is True for item in checks.values())
    report = {"schema_version": 2, "status": "passed" if passed else "failed", "checks": checks}
    output = args.output or (root / "replay_output/current_evidence_replay.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
