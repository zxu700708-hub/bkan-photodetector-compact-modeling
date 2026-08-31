#!/usr/bin/env python3
"""Replay current publication-facing summaries from frozen, non-legacy evidence."""

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


def replay_grouped(root: Path) -> dict:
    path = root / "artifacts/results/matched_grouped_comparison/metrics_by_seed.csv"
    correction_path = root / "artifacts/results/apparent_capacitance_correction/manifest.json"
    correction = load_json(correction_path)
    capacitance_scale = float(correction["audit"]["scale_factor"])
    if correction.get("status") != "passed" or capacitance_scale != 1000.0:
        raise ValueError("The frozen apparent-capacitance correction is missing or invalid")
    rows = load_csv(path)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        task = row["task"]
        value = float(row["rmse_target"])
        if task == "Capacitance":
            value *= capacitance_scale
        groups[(task, row["model"])].append(value)
    units = {
        "I_dark": "log10 target",
        "I_photo": "log10 target",
        "AC_Response": "dB",
        "Capacitance": "F",
    }
    summary = []
    for (task, model), values in sorted(groups.items()):
        summary.append(
            {
                "task": task,
                "model": model,
                "splits": len(values),
                "rmse_mean": statistics.mean(values),
                "rmse_sample_sd": statistics.stdev(values),
                "unit": units[task],
            }
        )
    passed = len(rows) == 200 and len(summary) == 20 and all(item["splits"] == 10 for item in summary)
    return {
        "passed": passed,
        "source_rows": len(rows),
        "capacitance_scale_applied": capacitance_scale,
        "summary": summary,
    }


def replay_uq(root: Path) -> dict:
    base = root / "artifacts/results/matched_heteroscedastic_uq_equal_budget_current"
    acceptance = load_json(base / "acceptance_audit.json")
    rows = load_csv(base / "metrics_by_seed.csv")
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["model"])].append(float(row["rmse_model_space"]))
    summary = [
        {"task": task, "model": model, "splits": len(values), "rmse_mean": statistics.mean(values)}
        for (task, model), values in sorted(groups.items())
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
        except Exception as exc:  # report malformed exports without hiding them
            parse_failures.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
    metrics_path = export_root / "metrics_by_export.csv"
    paired_path = export_root / "paired_teacher_vs_direct.csv"
    metric_rows = list(csv.DictReader(metrics_path.open(encoding="utf-8", newline=""))) if metrics_path.is_file() else []
    paired_rows = list(csv.DictReader(paired_path.open(encoding="utf-8", newline=""))) if paired_path.is_file() else []
    seeds = {row.get("seed") for row in metric_rows}
    passed = (
        len(formulas) == 960
        and len(sources) == 960
        and len(metric_rows) == 960
        and len(paired_rows) == 72
        and len(seeds) == 10
        and invalid_values == 0
        and not parse_failures
    )
    return {
        "passed": passed,
        "json_exports": len(formulas),
        "veriloga_exports": len(sources),
        "metric_rows": len(metric_rows),
        "paired_contrasts": len(paired_rows),
        "seeds": len(seeds),
        "numeric_values_checked": numeric_values,
        "invalid_numeric_values": invalid_values,
        "parse_failures": parse_failures,
    }


def replay_spectre(root: Path) -> dict:
    base = root / "artifacts/results/corrected_composite_spectre"
    device = load_json(base / "device_acceptance_report.json")
    circuit = load_json(base / "circuit_acceptance_report.json")
    provenance = load_json(base / "clean_rerun_provenance.json")
    model_hash = provenance.get("model_sha256")
    passed = (
        device.get("status") == "passed"
        and circuit.get("status") == "passed"
        and device.get("model_sha256") == model_hash
        and circuit.get("checks", {}).get("source_binding", {}).get("expected_sha256") == model_hash
        and provenance.get("device_decks_rerun_after_verifier_fix") is True
        and provenance.get("spectre_results_reused") is False
    )
    return {
        "passed": passed,
        "model_sha256": model_hash,
        "device_status": device.get("status"),
        "circuit_status": circuit.get("status"),
        "device_decks_rerun_after_verifier_fix": provenance.get("device_decks_rerun_after_verifier_fix"),
        "spectre_results_reused": provenance.get("spectre_results_reused"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    checks = {
        "grouped_accuracy_reaggregation": replay_grouped(root),
        "matched_uq_reaggregation": replay_uq(root),
        "symbolic_export_parse": replay_symbolic(root),
        "spectre_source_binding": replay_spectre(root),
    }
    passed = all(item.get("passed") is True for item in checks.values())
    report = {"schema_version": 1, "status": "passed" if passed else "failed", "checks": checks}
    output = args.output or (root / "replay_output/current_evidence_replay.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
