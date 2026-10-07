"""Select one validation-stable extraction configuration per task and replay it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run_experiment as experiment  # noqa: E402


VALUE_SLACK = 1.05
TAIL_SLACK = 1.10
DERIVATIVE_SLACK = 1.05
CONFIG_COLUMNS = ["budget", "teacher_weight", "derivative_weight", "ridge_alpha"]


def validation_constraints(task: str) -> dict[str, float]:
    limits = {
        "validation_rmse_ratio": VALUE_SLACK,
        "validation_relative_p95_ratio": TAIL_SLACK,
        "validation_group_rmse_p95_ratio": TAIL_SLACK,
    }
    if task == "Q_terminal":
        limits.update(
            validation_derivative_rmse_ratio=DERIVATIVE_SLACK,
            validation_derivative_p95_ratio=DERIVATIVE_SLACK,
        )
    else:
        limits.update(
            validation_conductance_nrmse_ratio=DERIVATIVE_SLACK,
            validation_conductance_relative_p95_ratio=DERIVATIVE_SLACK,
        )
    return limits


def choose_config(source: Path, task: str, seeds: list[int]) -> tuple[dict, pd.DataFrame]:
    limits = validation_constraints(task)
    tables = []
    for seed in seeds:
        table = pd.read_csv(source / task / f"seed_{seed}" / "validation_candidates.csv")
        table["seed"] = seed
        for column, limit in limits.items():
            table[f"scaled_{column}"] = table[column] / limit
        table["consensus_score"] = table[[f"scaled_{column}" for column in limits]].max(axis=1)
        table["consensus_eligible"] = table["consensus_score"] <= 1.0
        tables.append(table)
    all_rows = pd.concat(tables, ignore_index=True)
    grouped = (
        all_rows.groupby(CONFIG_COLUMNS, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            eligible_seeds=("consensus_eligible", "sum"),
            maximum_scaled_ratio=("consensus_score", "max"),
            mean_scaled_ratio=("consensus_score", "mean"),
        )
    )
    eligible = grouped[
        (grouped["seeds"] == len(seeds))
        & (grouped["eligible_seeds"] == len(seeds))
    ].sort_values(["budget", "mean_scaled_ratio", "maximum_scaled_ratio", "teacher_weight"])
    if eligible.empty:
        raise RuntimeError(f"No cross-split consensus candidate for {task}")
    selected = eligible.iloc[0].to_dict()
    config = {column: selected[column] for column in CONFIG_COLUMNS}
    config["budget"] = int(config["budget"])
    return config, grouped


def replay_one(task: str, seed: int, config: dict, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    data = (
        experiment.prepare_charge(seed, destination)
        if task == "Q_terminal"
        else experiment.prepare_current(task, seed, destination)
    )
    coefficient = experiment.fit_fixed_config(data, config)
    payload = experiment.make_payload(data, coefficient, config)
    audit = experiment.numerical_audit(data, payload)
    if not audit["passed"]:
        raise RuntimeError(f"Consensus numerical audit failed for {task}, seed {seed}")
    payload["selection"] = {
        "rule": "fixed cross-split validation consensus",
        "value_rmse_slack": VALUE_SLACK,
        "complete_response_tail_slack": TAIL_SLACK,
        "derivative_slack": DERIVATIVE_SLACK,
        "development_seeds": [42, 43, 44],
        "mandatory_columns": list(data["mandatory"]),
        "numerical_audit": audit,
        "test_used_by_selection_script": False,
        "evidence_boundary": "post hoc development; historical test results existed before this run",
    }
    experiment.dump(destination / "selected_formula.json", payload)
    frozen = json.loads((destination / "selected_formula.json").read_text(encoding="utf-8"))
    if task == "Q_terminal":
        prediction, _ = experiment.base.charge_sparse_eval(frozen, data["frames"]["test"])
        _, derivative = experiment.base.charge_sparse_eval(frozen, data["cframes"]["test"])
    else:
        prediction, derivative = experiment.current_formula_eval(
            frozen, data["frames"]["test"]
        )
    baseline_prediction, baseline_derivative = experiment.baseline_predictions(data, "test")
    selected_metrics = experiment.metrics(data, "test", prediction, derivative)
    baseline_metrics = experiment.metrics(
        data, "test", baseline_prediction, baseline_derivative
    )
    result = {
        "task": task,
        "seed": seed,
        "baseline_count": int(data["baseline_count"]),
        "selected_count": int(payload["coefficient_count"]),
        **config,
        **audit,
        **{f"baseline_test_{key}": value for key, value in baseline_metrics.items()},
        **{f"selected_test_{key}": value for key, value in selected_metrics.items()},
        "formula": str((destination / "selected_formula.json").relative_to(ROOT)),
    }
    experiment.dump(destination / "test_metrics.json", result)
    table = data["frames"]["test"][data["inputs"]].copy()
    table["group"] = data["labels"]["test"]
    table["target"] = data["ys"]["test"]
    table["baseline"] = baseline_prediction
    table["selected"] = prediction
    table.to_csv(destination / "test_predictions.csv", index=False)
    if task == "Q_terminal":
        derivative_table = data["cframes"]["test"][data["inputs"]].copy()
        derivative_table["target"] = data["dy"]["test"]
        derivative_table["baseline"] = baseline_derivative
        derivative_table["selected"] = derivative
        derivative_table.to_csv(destination / "derivative_test_predictions.csv", index=False)
    else:
        table["target_derivative"] = data["dy"]["test"]
        table["baseline_derivative"] = baseline_derivative
        table["selected_derivative"] = derivative
        table.to_csv(destination / "test_predictions.csv", index=False)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = {
        "status": "running",
        "source": str(source.relative_to(ROOT)),
        "seeds": args.seeds,
        "value_rmse_slack": VALUE_SLACK,
        "complete_response_tail_slack": TAIL_SLACK,
        "derivative_slack": DERIVATIVE_SLACK,
        "selection": "minimum common budget, then mean scaled validation ratio",
        "scope": "post hoc grouped development; no deployment or manuscript replacement",
    }
    experiment.dump(output / "protocol.json", protocol)
    results = []
    selected_configs = {}
    with threadpool_limits(limits=1):
        for task in ("I_dark", "I_photo", "Q_terminal"):
            config, table = choose_config(source, task, args.seeds)
            selected_configs[task] = config
            table.to_csv(output / f"{task}_consensus_candidates.csv", index=False)
            for seed in args.seeds:
                results.append(
                    replay_one(task, seed, config, output / task / f"seed_{seed}")
                )
                pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
    protocol["selected_configs"] = selected_configs
    protocol["status"] = "complete"
    experiment.dump(output / "protocol.json", protocol)


if __name__ == "__main__":
    main()
