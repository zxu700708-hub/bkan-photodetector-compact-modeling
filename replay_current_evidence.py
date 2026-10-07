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
    base = root / "artifacts/experiments/physics_balanced_export_20260917"
    export_root = base / "consensus_3split_compact"
    formulas = sorted(export_root.rglob("selected_formula.json"))
    expected_counts = {"I_dark": 10, "I_photo": 8, "Q_terminal": 20}
    numeric_values = 0
    invalid_values = 0
    parse_failures = []
    coefficient_counts = []
    dense_audit_points = 0
    for path in formulas:
        try:
            payload = load_json(path)
            total, invalid = finite_numbers(payload)
            numeric_values += total
            invalid_values += invalid
            coefficient_counts.append(
                (payload.get("task"), payload.get("seed"), payload.get("coefficient_count"))
            )
            dense_audit_points += int(
                payload.get("selection", {}).get("numerical_audit", {}).get("dense_points", 0)
            )
        except Exception as exc:
            parse_failures.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
    aggregate = load_csv(base / "aggregate.csv")
    replay = load_csv(base / "independent_replay_audit.csv")
    expected_triplets = {
        (task, seed, count)
        for task, count in expected_counts.items()
        for seed in (42, 43, 44)
    }
    passed = (
        len(formulas) == 9
        and set(coefficient_counts) == expected_triplets
        and dense_audit_points == 238080
        and len(aggregate) == 3
        and len(replay) == 9
        and max(float(row["prediction_max_abs_error"]) for row in replay) <= 1e-14
        and max(float(row["derivative_max_abs_error"]) for row in replay) <= 1e-9
        and invalid_values == 0
        and not parse_failures
    )
    return {
        "passed": passed,
        "selected_formulas": len(formulas),
        "coefficient_counts": expected_counts,
        "dense_audit_points": dense_audit_points,
        "aggregate": aggregate,
        "independent_replay_rows": len(replay),
        "numeric_values_checked": numeric_values,
        "invalid_numeric_values": invalid_values,
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
    base = root / "artifacts/results/physics_sparse_results_py36_20260921"
    device = load_json(base / "device/acceptance_report.json")
    circuit = load_json(base / "circuits/circuit_acceptance_report.json")
    source_binding = circuit.get("checks", {}).get("source_binding", {})
    model_hash = device.get("model_sha256")
    coarse = device.get("transient", {}).get("coarse_nrmse", {})
    tia_ac = circuit.get("checks", {}).get("tia_ac", {})
    tia_transient = circuit.get("checks", {}).get("tia_transient", {})
    passed = (
        device.get("status") == "passed"
        and circuit.get("status") == "passed"
        and device.get("checks", {}).get("source_binding") is True
        and source_binding.get("passed") is True
        and source_binding.get("expected_sha256") == model_hash
        and source_binding.get("returned_sha256") == model_hash
        and device.get("dc", {}).get("points_per_trace") == 183
        and device.get("ac", {}).get("points_per_trace") == 1815
        and max(coarse.values()) < 0.08
        and tia_ac.get("max_output_complex_error_V") < 5e-8
        and tia_transient.get("output_nrmse") < 0.05
    )
    return {
        "passed": passed,
        "device_status": device.get("status"),
        "circuit_status": circuit.get("status"),
        "model_sha256": model_hash,
        "maximum_coarse_transient_nrmse": max(coarse.values()),
        "tia_ac_maximum_output_error_V": tia_ac.get("max_output_complex_error_V"),
        "tia_transient_output_nrmse": tia_transient.get("output_nrmse"),
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
