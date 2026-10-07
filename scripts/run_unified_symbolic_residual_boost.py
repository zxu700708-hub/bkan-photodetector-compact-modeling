"""Add one common sparse symbolic residual layer to frozen task formulas.

This is a development experiment and does not edit the manuscript or retrain a
surrogate. Every task uses the same standardized degree-three polynomial
library, residual budgets, group-cross-fitted structure selection, stable-term
rule, shrinkage grid, and grouped safeguard. The existing validation-selected
formula is frozen. Training groups select the residual structure. All
calibration groups independently decide whether the correction is retained.
Test rows are evaluated only after that decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sympy
from sklearn.linear_model import OrthogonalMatchingPursuit, Ridge
from sklearn.preprocessing import PolynomialFeatures


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_output_aware_progressive_symbolification as base_audit  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "artifacts/results/global_soft_group_safeguarded_all_tasks_10seed/metrics.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts/results/unified_symbolic_residual_boost"
TASKS = ("I_dark", "I_photo", "AC_Response", "Capacitance")
PAPER_FILES = (
    ROOT / "paper/main_manuscript.tex",
    ROOT / "paper/supplementary_information.tex",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--budgets", nargs="+", type=int, default=[0, 1, 2, 4, 8, 12])
    parser.add_argument(
        "--shrinkages", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--cross-validation-folds", type=int, default=5)
    parser.add_argument("--minimum-support-frequency", type=float, default=0.60)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--selection-slack", type=float, default=0.01)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.05)
    parser.add_argument("--minimum-group-win-fraction", type=float, default=0.50)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.90)
    parser.add_argument("--dense-points", type=int, default=2000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Tasks and seeds must be unique")
    if args.degree < 1 or not args.budgets or min(args.budgets) < 0:
        raise ValueError("Invalid polynomial degree or residual budget")
    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Residual budgets must be unique")
    if (
        not args.shrinkages
        or len(set(args.shrinkages)) != len(args.shrinkages)
        or min(args.shrinkages) < 0.0
        or max(args.shrinkages) > 1.0
        or 0.0 not in args.shrinkages
    ):
        raise ValueError("Shrinkages must be unique values in [0, 1] including zero")
    if args.cross_validation_folds < 3:
        raise ValueError("At least three group-cross-validation folds are required")
    if not 0.5 <= args.minimum_support_frequency <= 1.0:
        raise ValueError("Minimum support frequency must lie in [0.5, 1]")
    if args.ridge_alpha < 0.0:
        raise ValueError("Ridge alpha must be nonnegative")
    if not 0.0 <= args.selection_slack < 1.0:
        raise ValueError("Selection slack must lie in [0, 1)")
    if not 0.0 <= args.minimum_relative_improvement < 1.0:
        raise ValueError("Minimum relative improvement must lie in [0, 1)")
    if not 0.5 <= args.minimum_group_win_fraction <= 1.0:
        raise ValueError("Minimum group-win fraction must lie in [0.5, 1]")
    if args.bootstrap_samples < 100 or not 0.5 < args.bootstrap_confidence < 1.0:
        raise ValueError("Invalid bootstrap controls")
    if args.dense_points < 100:
        raise ValueError("Dense audit requires at least 100 points")


def loader_args() -> SimpleNamespace:
    return SimpleNamespace(
        data=base_audit.DEFAULT_DATA,
        net_photocurrent_data=(
            base_audit.DEFAULT_NET_ROOT / "device_modeling/cleaned_data.csv"
        ),
        capacitance_data=base_audit.task_config.DEFAULT_CAPACITANCE_DATA,
        bkan_root=base_audit.shared_audit.DEFAULT_BKAN,
        capacitance_bkan_root=base_audit.shared_audit.DEFAULT_CAP_BKAN,
        net_root=base_audit.DEFAULT_NET_ROOT,
    )


def grouped_folds(
    frame: pd.DataFrame,
    specification,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    labels = base_audit.task_config.task_group_labels(frame, specification).to_numpy()
    groups = pd.unique(labels).copy()
    rng = np.random.default_rng(32452843 + int(seed) * 49999)
    rng.shuffle(groups)
    return labels, groups


def feature_matrices(
    raw: dict[str, np.ndarray],
    degree: int,
) -> tuple[dict[str, np.ndarray], dict]:
    train = raw["train"]
    input_mean = train.mean(axis=0)
    input_scale = train.std(axis=0, ddof=0)
    input_scale = np.where(input_scale > 0.0, input_scale, 1.0)
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    train_poly = polynomial.fit_transform((train - input_mean) / input_scale)
    feature_mean = train_poly.mean(axis=0)
    feature_scale = train_poly.std(axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    matrices = {}
    for split, values in raw.items():
        poly = polynomial.transform((values - input_mean) / input_scale)
        matrices[split] = (poly - feature_mean) / feature_scale
    metadata = {
        "input_mean": input_mean,
        "input_scale": input_scale,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "powers": polynomial.powers_.astype(int),
    }
    return matrices, metadata


def fit_candidate(
    features: np.ndarray,
    residual: np.ndarray,
    budget: int,
) -> dict:
    if budget == 0:
        return {
            "budget": 0,
            "intercept": float(np.mean(residual)),
            "indices": np.empty(0, dtype=int),
            "coefficients": np.empty(0, dtype=np.float64),
        }
    model = OrthogonalMatchingPursuit(
        n_nonzero_coefs=min(int(budget), features.shape[1]),
        fit_intercept=True,
    )
    model.fit(features, residual)
    indices = np.flatnonzero(np.abs(model.coef_) > 1e-12)
    return {
        "budget": int(budget),
        "intercept": float(model.intercept_),
        "indices": indices.astype(int),
        "coefficients": model.coef_[indices].astype(np.float64),
    }


def residual_prediction(candidate: dict, features: np.ndarray) -> np.ndarray:
    result = np.full(len(features), candidate["intercept"], dtype=np.float64)
    if len(candidate["indices"]):
        result += features[:, candidate["indices"]] @ candidate["coefficients"]
    return result


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(actual)) ** 2)))


def group_average_rmse(
    actual: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
) -> float:
    group_mse = [
        float(np.mean((prediction[labels == group] - actual[labels == group]) ** 2))
        for group in pd.unique(labels)
    ]
    return float(np.sqrt(np.mean(group_mse)))


def cross_fitted_candidate(
    features: np.ndarray,
    base_prediction: np.ndarray,
    actual: np.ndarray,
    target_scale: float,
    labels: np.ndarray,
    budgets: list[int],
    shrinkages: list[float],
    fold_count: int,
    minimum_support_frequency: float,
    ridge_alpha: float,
    seed: int,
    slack: float,
) -> tuple[dict, list[dict], dict]:
    groups = pd.unique(labels).copy()
    if len(groups) < fold_count:
        raise RuntimeError("Fewer training groups than requested cross-validation folds")
    rng = np.random.default_rng(49979687 + int(seed) * 104729)
    rng.shuffle(groups)
    folds = [group for group in np.array_split(groups, fold_count) if len(group)]
    residual = (actual - base_prediction) / target_scale
    diagnostics = []
    support_by_budget: dict[int, list[np.ndarray]] = {}
    for budget in sorted(budgets):
        out_of_fold_residual = np.full(len(features), np.nan, dtype=np.float64)
        fold_supports = []
        fold_sizes = []
        for fold_groups in folds:
            held_mask = np.isin(labels, fold_groups)
            fitted = fit_candidate(features[~held_mask], residual[~held_mask], budget)
            out_of_fold_residual[held_mask] = residual_prediction(
                fitted, features[held_mask]
            )
            fold_supports.append(fitted["indices"])
            fold_sizes.append(int(np.sum(held_mask)))
        if not np.isfinite(out_of_fold_residual).all():
            raise RuntimeError("Cross-fitted residual predictions are incomplete")
        support_by_budget[int(budget)] = fold_supports
        for shrinkage in sorted(shrinkages):
            prediction = (
                base_prediction
                + float(shrinkage) * target_scale * out_of_fold_residual
            )
            diagnostics.append(
                {
                    "budget": int(budget),
                    "shrinkage": float(shrinkage),
                    "cross_fitted_group_rmse": group_average_rmse(
                        actual, prediction, labels
                    ),
                    "fold_sizes": fold_sizes,
                    "fold_realized_terms": [
                        int(len(indices)) for indices in fold_supports
                    ],
                }
            )
    best = min(item["cross_fitted_group_rmse"] for item in diagnostics)
    eligible = [
        item
        for item in diagnostics
        if item["cross_fitted_group_rmse"] <= best * (1.0 + slack)
    ]
    choice = min(
        eligible,
        key=lambda item: (
            item["budget"],
            item["shrinkage"],
            item["cross_fitted_group_rmse"],
        ),
    )
    budget = int(choice["budget"])
    fold_supports = support_by_budget[budget]
    support_counts = np.zeros(features.shape[1], dtype=int)
    for indices in fold_supports:
        support_counts[indices] += 1
    support_frequency = support_counts / len(fold_supports)
    stable_indices = np.flatnonzero(
        support_frequency >= float(minimum_support_frequency)
    )
    if budget > 0 and len(stable_indices) > budget:
        order = np.lexsort((stable_indices, -support_frequency[stable_indices]))
        stable_indices = stable_indices[order[:budget]]
    if len(stable_indices):
        model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
        model.fit(features[:, stable_indices], residual)
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        intercept = float(model.intercept_)
    else:
        coefficients = np.empty(0, dtype=np.float64)
        intercept = float(np.mean(residual))
    selected = {
        "budget": budget,
        "shrinkage": float(choice["shrinkage"]),
        "intercept": intercept,
        "indices": stable_indices.astype(int),
        "coefficients": coefficients,
    }
    stability = {
        "fold_count": len(folds),
        "fold_groups": [[str(value) for value in fold] for fold in folds],
        "minimum_support_frequency": float(minimum_support_frequency),
        "selected_feature_frequencies": support_frequency[stable_indices].tolist(),
        "candidate_budget": budget,
        "candidate_shrinkage": float(choice["shrinkage"]),
        "candidate_cross_fitted_group_rmse": float(
            choice["cross_fitted_group_rmse"]
        ),
        "best_cross_fitted_group_rmse": float(best),
    }
    return selected, diagnostics, stability


def grouped_guard(
    actual: np.ndarray,
    base_prediction: np.ndarray,
    corrected_prediction: np.ndarray,
    labels: np.ndarray,
    guard_groups: np.ndarray,
    target_scale: float,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    guard_mask = np.isin(labels, guard_groups)
    base_rmse = rmse(actual[guard_mask], base_prediction[guard_mask])
    corrected_rmse = rmse(actual[guard_mask], corrected_prediction[guard_mask])
    relative_improvement = (base_rmse - corrected_rmse) / max(base_rmse, 1e-30)
    deltas = []
    for group in guard_groups:
        mask = labels == group
        base_mse = float(np.mean(((base_prediction[mask] - actual[mask]) / target_scale) ** 2))
        corrected_mse = float(
            np.mean(((corrected_prediction[mask] - actual[mask]) / target_scale) ** 2)
        )
        deltas.append(corrected_mse - base_mse)
    deltas = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(86028121 + int(seed) * 65537)
    indices = rng.integers(
        0, len(deltas), size=(args.bootstrap_samples, len(deltas))
    )
    bootstrap_upper = float(
        np.quantile(deltas[indices].mean(axis=1), args.bootstrap_confidence)
    )
    win_fraction = float(np.mean(deltas < 0.0))
    accepted = bool(
        relative_improvement >= args.minimum_relative_improvement
        and win_fraction >= args.minimum_group_win_fraction
        and bootstrap_upper < 0.0
    )
    return {
        "base_rmse": base_rmse,
        "corrected_rmse": corrected_rmse,
        "relative_improvement": float(relative_improvement),
        "minimum_relative_improvement": float(args.minimum_relative_improvement),
        "group_win_fraction": win_fraction,
        "minimum_group_win_fraction": float(args.minimum_group_win_fraction),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_confidence": float(args.bootstrap_confidence),
        "bootstrap_mean_delta_upper": bootstrap_upper,
        "accepted": accepted,
    }


def symbolic_residual(
    candidate: dict,
    metadata: dict,
    target_scale: float,
    variables: list[sympy.Symbol],
) -> sympy.Expr:
    value = sympy.Float(candidate["intercept"], 17)
    for feature_index, coefficient in zip(
        candidate["indices"], candidate["coefficients"]
    ):
        power = metadata["powers"][feature_index]
        term = sympy.Integer(1)
        for index, exponent in enumerate(power):
            if exponent:
                normalized = (
                    variables[index] - sympy.Float(metadata["input_mean"][index], 17)
                ) / sympy.Float(metadata["input_scale"][index], 17)
                term *= normalized ** int(exponent)
        standardized = (
            term - sympy.Float(metadata["feature_mean"][feature_index], 17)
        ) / sympy.Float(metadata["feature_scale"][feature_index], 17)
        value += sympy.Float(coefficient, 17) * standardized
    return (
        sympy.Float(candidate["shrinkage"], 17)
        * sympy.Float(target_scale, 17)
        * value
    )


def replay_expression(payload: dict, raw: np.ndarray) -> np.ndarray:
    variables = [sympy.Symbol(name) for name in payload["symbol_order"]]
    expression = sympy.sympify(payload["formula"])
    evaluator = sympy.lambdify(variables, expression, modules="numpy", cse=True)
    with np.errstate(all="ignore"):
        values = evaluator(*[raw[:, index] for index in range(raw.shape[1])])
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(len(raw), float(values), dtype=np.float64)
    return values.reshape(-1)


def main() -> None:
    args = parse_args()
    validate_args(args)
    before_hashes = {path: sha256(path) for path in PAPER_FILES}
    source = pd.read_csv(args.input)
    source = source[
        source["task"].isin(args.tasks)
        & source["seed"].isin(args.seeds)
        & source["variant"].eq("global_soft_safeguarded")
        & source["source"].eq("bkan")
    ].copy()
    expected = len(args.tasks) * len(args.seeds)
    if len(source) != expected or source.duplicated(["task", "seed"]).any():
        raise RuntimeError(f"Expected {expected} unique source formulas, found {len(source)}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    loader = loader_args()
    for task in args.tasks:
        for seed in args.seeds:
            record = source[source["task"].eq(task) & source["seed"].eq(seed)].iloc[0]
            formula_path = ROOT / Path(record["formula_path"])
            base_payload = json.loads(formula_path.read_text(encoding="utf-8"))
            _, frame, inputs, specification, baseline_spec, partitions = (
                base_audit._load_task(loader, task, seed)
            )
            split_names = ("train", "validation", "calibration", "test")
            raw = {
                name: split_frame[list(inputs)].to_numpy(dtype=np.float64)
                for name, split_frame in zip(split_names, partitions)
            }
            actual = base_audit._physical_targets(partitions, baseline_spec)
            base_prediction = {
                name: replay_expression(base_payload, raw[name]) for name in split_names
            }
            target_scale = max(float(np.std(actual["train"], ddof=0)), 1e-30)
            residual_train = (
                actual["train"] - base_prediction["train"]
            ) / target_scale
            features, feature_metadata = feature_matrices(raw, args.degree)
            train_labels, train_groups = grouped_folds(
                partitions[0], specification, seed
            )
            selected, candidate_diagnostics, stability = cross_fitted_candidate(
                features["train"],
                base_prediction["train"],
                actual["train"],
                target_scale,
                train_labels,
                sorted(args.budgets),
                sorted(args.shrinkages),
                args.cross_validation_folds,
                args.minimum_support_frequency,
                args.ridge_alpha,
                seed,
                args.selection_slack,
            )
            calibration_labels, calibration_groups = grouped_folds(
                partitions[2], specification, seed
            )
            corrected_calibration = (
                base_prediction["calibration"]
                + selected["shrinkage"]
                * target_scale
                * residual_prediction(selected, features["calibration"])
            )
            guard = grouped_guard(
                actual["calibration"],
                base_prediction["calibration"],
                corrected_calibration,
                calibration_labels,
                calibration_groups,
                target_scale,
                seed,
                args,
            )

            variables = [sympy.Symbol(name) for name in base_payload["symbol_order"]]
            base_expression = sympy.sympify(base_payload["formula"])
            correction = symbolic_residual(selected, feature_metadata, target_scale, variables)
            expression = base_expression + correction if guard["accepted"] else base_expression
            payload = {
                "schema": "unified_sparse_symbolic_residual_v2",
                "task": task,
                "seed": int(seed),
                "source_formula": str(formula_path.relative_to(ROOT)),
                "input_order": list(inputs),
                "symbol_order": base_payload["symbol_order"],
                "formula": str(expression),
                "count_ops": int(sympy.count_ops(expression, visual=False)),
                "base_count_ops": int(base_payload["count_ops"]),
                "residual_degree": int(args.degree),
                "residual_budget_grid": sorted(args.budgets),
                "residual_shrinkage_grid": sorted(args.shrinkages),
                "selected_budget": int(selected["budget"]),
                "selected_shrinkage": float(selected["shrinkage"]),
                "selected_indices": selected["indices"].tolist(),
                "selected_coefficients": selected["coefficients"].tolist(),
                "selected_intercept": float(selected["intercept"]),
                "target_scale": target_scale,
                "candidate_diagnostics": candidate_diagnostics,
                "cross_fitted_selection": stability,
                "group_counts": {
                    "training": int(len(train_groups)),
                    "calibration_guard": int(len(calibration_groups)),
                },
                "guard": guard,
                "test_accessed_during_selection": False,
            }
            output_path = args.output / "formulas" / f"seed_{seed}" / task / "formula.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            prediction = replay_expression(payload, raw["test"])
            direct_prediction = base_prediction["test"]
            if guard["accepted"]:
                direct_prediction = (
                    direct_prediction
                    + selected["shrinkage"]
                    * target_scale
                    * residual_prediction(selected, features["test"])
                )
            replay_error = np.abs(prediction - direct_prediction)
            dense_raw = base_audit._dense_raw(frame, inputs, args.dense_points, seed + 7001)
            dense_prediction = replay_expression(payload, dense_raw)
            base_metrics = base_audit._metrics(actual["test"], base_prediction["test"])
            final_metrics = base_audit._metrics(actual["test"], prediction)
            shape = base_audit._shape_diagnostics(
                task, partitions[3], prediction, specification
            )
            row = {
                "task": task,
                "seed": int(seed),
                "residual_accepted": bool(guard["accepted"]),
                "selected_budget": int(selected["budget"]),
                "selected_shrinkage": float(selected["shrinkage"]),
                "realized_residual_terms": int(len(selected["indices"]) + 1),
                "base_formula_ops": int(base_payload["count_ops"]),
                "formula_ops": int(payload["count_ops"]),
                "base_formula_to_tcad_rmse": base_metrics["rmse"],
                "formula_to_tcad_rmse": final_metrics["rmse"],
                "formula_to_tcad_r2": final_metrics["r2"],
                "formula_to_tcad_p95_abs_error": final_metrics["p95_abs_error"],
                "formula_to_tcad_max_abs_error": final_metrics["max_abs_error"],
                "test_delta": final_metrics["rmse"] - base_metrics["rmse"],
                "guard_relative_improvement": guard["relative_improvement"],
                "guard_group_win_fraction": guard["group_win_fraction"],
                "guard_bootstrap_upper": guard["bootstrap_mean_delta_upper"],
                "replay_max_abs_error": float(np.max(replay_error)),
                "test_finite": int(np.isfinite(prediction).sum()),
                "test_points": len(prediction),
                "dense_finite": int(np.isfinite(dense_prediction).sum()),
                "dense_points": len(dense_prediction),
                "formula_path": str(output_path.relative_to(ROOT)),
                **shape,
            }
            rows.append(row)
            print(
                f"[{task} seed={seed}] accepted={row['residual_accepted']} "
                f"budget={row['selected_budget']} base={base_metrics['rmse']:.6g} "
                f"corrected={final_metrics['rmse']:.6g}"
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    after_hashes = {path: sha256(path) for path in PAPER_FILES}
    if before_hashes != after_hashes:
        raise RuntimeError("A manuscript hash changed during residual-boost experiment")
    metadata = {
        "schema": "unified_sparse_symbolic_residual_experiment_v2",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": len(metrics),
        "paper_sha256": {
            str(path.relative_to(ROOT)): digest
            for path, digest in after_hashes.items()
        },
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
