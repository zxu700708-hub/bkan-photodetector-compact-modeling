"""Audit symbolic stability over fixed VI posterior draws without retraining BKAN.

Each draw is copied once into the existing deterministic KAN teacher.  The
current task-specific symbolic family and frozen term budget are then fitted
using training groups only, with validation used for optimization checkpoints.
The original calibration groups remain untouched until formula-ensemble
conformal scaling; the original test groups are evaluated once.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
for entry in (ROOT, BKAN_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler
from device_modeling.photodetector.common import set_seed
from device_modeling.photodetector.posterior_export import (
    posterior_draw_to_deterministic,
    predict_fixed_teacher,
)
from device_modeling.photodetector.run_research import DEFAULT_20260617_DATA
from device_modeling.photodetector.symbolic_gated_kan import (
    evaluate_exported_pure_symbolic_formula,
    train_symbolic_gated,
)
from device_modeling.photodetector.task_config import (
    DEFAULT_CAPACITANCE_DATA,
    TASKS,
    load_capacitance_data,
    load_research_data,
    prepare_task_dataframe,
    split_task_dataframe,
    target_values,
    task_group_labels,
)
import scripts.run_ac_lowpass_gated_teacher as ac_lowpass


OUTPUT = ROOT / "artifacts/results/posterior_formula_ensemble"
DATA = ROOT / "artifacts/results/device_modeling/cleaned_data.csv"
FORMULA_ALPHA = 0.10
TASK_MAP = {
    "I_dark": "dark_current",
    "I_photo": "photo_current",
    "AC_Response": "ac_response",
    "Capacitance": "capacitance",
}
CHECKPOINTS = {
    "I_dark": ROOT / "artifacts/results/device_modeling/I-dark-bayes-results/model_checkpoint.pt",
    "I_photo": ROOT / "artifacts/results/net_photocurrent_retrain/uq_repeated_grouped/seed_42/I-photo-bayes-results/model_checkpoint.pt",
    "AC_Response": ROOT / "artifacts/results/device_modeling/AC-response-bayes-results/model_checkpoint.pt",
    "Capacitance": ROOT / "artifacts/results/device_modeling_capacitance_full/Capacitance-bayes-results/model_checkpoint.pt",
}
CONFIGS = {
    "I_dark": ROOT / "artifacts/results/symbolic_gate_replacement/dark_teacher_budget36/dark_current_symbolic_gated_kan/run_config.json",
    "I_photo": ROOT / "artifacts/results/symbolic_gate_replacement/photo_teacher_budget72_refit/photo_current_symbolic_gated_kan/run_config.json",
    "Capacitance": ROOT / "artifacts/results/symbolic_gate_replacement/cap_teacher_curvature_budget24_seed42/capacitance_symbolic_gated_kan/run_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_MAP), default=list(TASK_MAP))
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--draw-seed-base", type=int, default=26000)
    parser.add_argument("--distillation-seed", type=int, default=314159)
    parser.add_argument(
        "--symbolic-step-scale",
        type=float,
        default=1.0,
        help="Frozen multiplier for symbolic optimization stages; architecture, library, and term budgets are unchanged.",
    )
    parser.add_argument(
        "--task-step-scale",
        nargs="*",
        default=[],
        metavar="TASK=VALUE",
        help="Task-specific symbolic step multipliers, e.g. I_photo=0.3 Capacitance=0.2.",
    )
    parser.add_argument("--allow-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--draw-indices", type=int, nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--fit-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--capacitance-data", type=Path, default=DEFAULT_CAPACITANCE_DATA)
    parser.add_argument(
        "--split-fractions", type=float, nargs=4,
        default=(0.70, 0.10, 0.10, 0.10),
        metavar=("TRAIN", "VALIDATION", "CALIBRATION", "TEST"),
        help="Frozen group split; it must match the loaded checkpoint scaler.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ac-warmup-steps", type=int, default=600)
    parser.add_argument("--ac-gate-steps", type=int, default=1800)
    parser.add_argument("--ac-refit-steps", type=int, default=2200)
    return parser.parse_args()


def _coerce(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _symbolic_args(task: str, distillation_seed: int, step_scale: float) -> argparse.Namespace:
    raw = json.loads(CONFIGS[task].read_text(encoding="utf-8"))
    values = {key: _coerce(value) for key, value in raw.items()}
    values["seed"] = int(distillation_seed)
    # Isolate posterior-function variation: no observed-target, derivative, or
    # curve-shape term is allowed to pull a sampled teacher back to the mean.
    values["bayesian_distill_weight"] = 1.0
    values["observed_loss_weight"] = 0.0
    values["derivative_loss_weight"] = 0.0
    values["curve_shape_loss_weight"] = 0.0
    values["pure_symbolic_student"] = True
    values["ridge_init"] = True
    for key in ("stage1_steps", "stage2_steps", "stage3_steps"):
        values[key] = max(10, int(round(int(values[key]) * step_scale)))
    values["val_freq"] = max(5, min(int(values.get("val_freq", 25)), values["stage2_steps"] // 4))
    return argparse.Namespace(**values)


def _task_step_scale(task: str, args: argparse.Namespace) -> float:
    for item in args.task_step_scale:
        name, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid --task-step-scale value: {item}")
        if name == task:
            return float(value)
    return float(args.symbolic_step_scale)


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def _load_task(task: str, args: argparse.Namespace):
    spec = TASKS[TASK_MAP[task]]
    raw = load_capacitance_data(args.capacitance_data) if task == "Capacitance" else load_research_data(args.data)
    data, inputs = prepare_task_dataframe(raw, spec)
    partitions = split_task_dataframe(data, spec, 42, fractions=tuple(args.split_fractions))
    return spec, inputs, partitions


def _teacher_for(modeler, teacher, frame: pd.DataFrame, inputs: tuple[str, ...]) -> np.ndarray:
    return predict_fixed_teacher(modeler, teacher, frame[list(inputs)].to_numpy(dtype=np.float32))


def _formula_terms(payload: dict) -> set[str]:
    if payload.get("family") == "gated_lowpass":
        return {f"{row['submodel']}:{row['feature']}" for row in payload["active_terms"]}
    terms: set[str] = set()
    for branch in payload.get("branches", []):
        for row in branch.get("terms", []):
            terms.add(f"mechanism:{branch.get('name', 'branch')}:{row['term']}")
    for label in ("residual_branch", "generic_branch"):
        for row in payload.get(label, {}).get("terms", []):
            terms.add(f"{label}:{row['term']}")
    return terms


def _evaluate(payload: dict, x_raw: np.ndarray) -> np.ndarray:
    if payload.get("family") == "gated_lowpass":
        return ac_lowpass._evaluate(payload, x_raw)
    return evaluate_exported_pure_symbolic_formula(payload, x_raw)


def _fit_ac_formula(
    teacher_train: np.ndarray,
    train: pd.DataFrame,
    inputs: tuple[str, ...],
    output: Path,
    args: argparse.Namespace,
) -> dict:
    set_seed(args.distillation_seed)
    x_fit = train[list(inputs)].to_numpy(dtype=np.float64)
    features, conditions = ac_lowpass._make_features(x_fit, inputs)
    phi = ac_lowpass._condition_design(x_fit, inputs, features)
    frequency = x_fit[:, inputs.index("frequency_ghz")]
    n_features = phi.shape[1]
    theta0 = np.zeros(3 * n_features + 1, dtype=np.float64)
    theta0[0] = float(np.percentile(teacher_train, 95))
    theta0[n_features] = float(np.percentile(teacher_train, 5))
    theta0[2 * n_features] = np.log(6.0)
    theta0[-1] = np.log(1.5)
    initial = least_squares(
        lambda theta: ac_lowpass._predict_numpy(theta, phi, frequency) - teacher_train,
        theta0,
        loss="soft_l1",
        f_scale=0.08,
        max_nfev=10000,
    ).x
    student = ac_lowpass.GatedLowpass(initial, n_features, max_terms=10)
    phi_t = torch.tensor(phi, dtype=torch.float32)
    frequency_t = torch.tensor(frequency.reshape(-1, 1), dtype=torch.float32)
    teacher_t = torch.tensor(teacher_train.reshape(-1, 1), dtype=torch.float32)
    optimizer = torch.optim.Adam(student.parameters(), lr=2e-2)

    def fit(steps: int, gated: bool, start: float, end: float, penalty: float) -> None:
        for step in range(steps):
            fraction = step / max(steps - 1, 1)
            temperature = start * ((end / start) ** fraction)
            optimizer.zero_grad()
            prediction = student(phi_t, frequency_t, temperature)
            loss = torch.mean((prediction - teacher_t) ** 2)
            if gated:
                loss = loss + penalty * torch.sigmoid(student.gate_logits / temperature).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 20.0)
            optimizer.step()

    fit(args.ac_warmup_steps, False, 1.0, 1.0, 0.0)
    fit(args.ac_gate_steps, True, 2.0, 0.15, 2e-4)
    student.prune()
    fit(args.ac_refit_steps, False, 0.1, 0.1, 0.0)
    active = student.hard_mask.detach().cpu().numpy() > 0
    coefficient = (student.coeff * student.hard_mask).detach().cpu().numpy().astype(np.float64)
    names = [item["name"] for item in features]
    payload = {
        "family": "gated_lowpass",
        "inputs": list(inputs),
        "condition_inputs": conditions,
        "feature_definitions": features,
        "formula": "floor(z) + (y0(z)-floor(z))/(1+(frequency_ghz/exp(logfc(z)))^p)",
        "coefficients": {head: coefficient[index].tolist() for index, head in enumerate(("y0", "floor", "logfc"))},
        "log_p": float(student.log_p.detach().cpu()),
        "p": float(0.5 + np.exp(np.clip(float(student.log_p.detach().cpu()), np.log(0.05), np.log(20.0)))),
        "active_terms": [
            {"submodel": head, "feature": names[column], "coefficient": float(coefficient[row, column])}
            for row, head in enumerate(("y0", "floor", "logfc"))
            for column in range(n_features) if active[row, column]
        ],
        "kan_contribution_at_inference": 0.0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pure_symbolic_formula.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(__import__("math").erf)(z / np.sqrt(2.0)))


def _crps(y: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> float:
    z = (y - mean) / sigma
    values = sigma * (z * (2.0 * _normal_cdf(z) - 1.0) + 2.0 * _normal_pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(values))


def _interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.05) -> float:
    score = upper - lower
    score += (2.0 / alpha) * (lower - y) * (y < lower)
    score += (2.0 / alpha) * (y - upper) * (y > upper)
    return float(np.mean(score))


def _group_scores(y: np.ndarray, mean: np.ndarray, sigma: np.ndarray, groups: np.ndarray) -> np.ndarray:
    scores = np.abs(y - mean) / sigma
    return pd.DataFrame({"group": groups, "score": scores}).groupby("group", sort=False)["score"].max().to_numpy()


def _higher_quantile(scores: np.ndarray, alpha: float = FORMULA_ALPHA) -> tuple[float, int]:
    """Return the finite split-conformal order statistic without rank clipping.

    A finite order statistic exists only when
    ``ceil((n + 1) * (1 - alpha)) <= n``.  Silently clipping an unavailable
    rank to the largest observed score would overstate the nominal coverage.
    """

    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("At least one finite calibration score is required")
    rank = int(np.ceil((finite.size + 1) * (1.0 - alpha)))
    if rank > finite.size:
        raise ValueError(
            f"No finite {(1.0 - alpha):.1%} conformal order statistic: "
            f"n={finite.size}, required rank={rank}. Increase calibration groups "
            "or lower the nominal coverage."
        )
    return float(np.sort(finite)[rank - 1]), rank


def _ensemble_metrics(task: str, frames, prediction_sets: dict[str, list[np.ndarray]]) -> dict:
    spec, _, (_, _, calibration, test) = frames
    y_cal = target_values(calibration, spec)
    y_test = target_values(test, spec)
    cal = np.stack(prediction_sets["calibration"], axis=0)
    tst = np.stack(prediction_sets["test"], axis=0)
    cal_mean, test_mean = cal.mean(axis=0), tst.mean(axis=0)
    ddof = 1 if cal.shape[0] > 1 else 0
    cal_std, test_std = cal.std(axis=0, ddof=ddof), tst.std(axis=0, ddof=ddof)
    positive = cal_std[cal_std > 0]
    floor = max((float(np.median(positive)) * 0.1 if len(positive) else 0.0), 1e-12 if task != "Capacitance" else 1e-24)
    cal_sigma = np.maximum(cal_std, floor)
    test_sigma = np.maximum(test_std, floor)
    cal_groups = task_group_labels(calibration, spec).to_numpy()
    test_groups = task_group_labels(test, spec).to_numpy()
    q, rank = _higher_quantile(
        _group_scores(y_cal, cal_mean, cal_sigma, cal_groups), FORMULA_ALPHA
    )
    lower, upper = test_mean - q * test_sigma, test_mean + q * test_sigma
    group_covered = pd.DataFrame({"group": test_groups, "covered": (y_test >= lower) & (y_test <= upper)}).groupby("group")["covered"].all()
    # Proper Gaussian scores describe the unmodified between-formula Gaussian
    # approximation.  Conformalization changes only the reported interval; it
    # does not turn q * sigma into a newly calibrated posterior distribution.
    raw_gaussian_sigma = test_sigma
    nll = 0.5 * np.log(2.0 * np.pi * raw_gaussian_sigma**2) + 0.5 * (
        (y_test - test_mean) / raw_gaussian_sigma
    ) ** 2
    return {
        "task": task,
        "posterior_draws": int(tst.shape[0]),
        "calibration_groups": int(pd.Series(cal_groups).nunique()),
        "test_groups": int(pd.Series(test_groups).nunique()),
        "nominal_coverage": 1.0 - FORMULA_ALPHA,
        "conformal_rank": rank,
        "sigma_floor": floor,
        "conformal_q": q,
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_mean))),
        "test_r2": float(r2_score(y_test, test_mean)),
        "test_point_coverage_90": float(np.mean((y_test >= lower) & (y_test <= upper)) * 100.0),
        "test_group_coverage_90": float(group_covered.mean() * 100.0),
        "test_mpiw": float(np.mean(upper - lower)),
        "raw_gaussian_nll": float(np.mean(nll)),
        "raw_gaussian_crps": _crps(y_test, test_mean, raw_gaussian_sigma),
        "interval_score_90": _interval_score(
            y_test, lower, upper, alpha=FORMULA_ALPHA
        ),
    }


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    if not args.allow_smoke and not 20 <= args.draws <= 50:
        raise ValueError("--draws must be between 20 and 50 for the reviewer stability audit")
    args.output.mkdir(parents=True, exist_ok=True)
    draw_rows, term_rows, pair_rows, ensemble_rows = [], [], [], []
    for task in args.tasks:
        print(f"[posterior-formula] task={task} draws={args.draws}", flush=True)
        spec, inputs, partitions = _load_task(task, args)
        train, validation, calibration, test = partitions
        checkpoint = CHECKPOINTS[task]
        modeler = BayesKANDeviceModeler.load_model(str(checkpoint), device=args.device)
        train_mean = train[list(inputs)].to_numpy(dtype=np.float64).mean(axis=0)
        scaler_delta = float(np.max(np.abs(np.asarray(modeler.scaler_x.mean_) - train_mean) / np.maximum(np.abs(np.asarray(modeler.scaler_x.mean_)), 1.0)))
        if list(modeler.input_cols) != list(inputs) or scaler_delta > 1e-7:
            raise RuntimeError(f"{task}: checkpoint/split mismatch, scaler delta={scaler_delta:.3g}")
        formulas, term_sets = [], []
        predictions = {"calibration": [], "test": []}
        draw_indices = args.draw_indices if args.draw_indices is not None else list(range(args.draws))
        for draw_index in draw_indices:
            if draw_index < 0 or draw_index >= args.draws:
                raise ValueError(f"draw index out of range: {draw_index}")
            draw_seed = args.draw_seed_base + draw_index
            print(f"  draw={draw_index + 1}/{args.draws} seed={draw_seed}", flush=True)
            teacher = posterior_draw_to_deterministic(modeler.model, draw_seed, args.device)
            teacher_targets = {
                "train": _teacher_for(modeler, teacher, train, inputs),
                "validation": _teacher_for(modeler, teacher, validation, inputs),
                "test": _teacher_for(modeler, teacher, test, inputs),
            }
            draw_root = args.output / "formulas" / task / f"draw_{draw_index:02d}_seed_{draw_seed}"
            formula_path = (
                draw_root / "ac_response_lowpass_gated/pure_symbolic_formula.json"
                if task == "AC_Response"
                else draw_root / f"{spec.key}_symbolic_gated_kan/pure_symbolic_formula.json"
            )
            if formula_path.exists():
                payload = json.loads(formula_path.read_text(encoding="utf-8"))
            elif task == "AC_Response":
                payload = _fit_ac_formula(teacher_targets["train"], train, inputs, draw_root / "ac_response_lowpass_gated", args)
            else:
                symbolic_args = _symbolic_args(task, args.distillation_seed, _task_step_scale(task, args))
                train_symbolic_gated(
                    spec,
                    inputs,
                    train,
                    validation,
                    test,
                    symbolic_args,
                    torch.device(args.device),
                    draw_root,
                    teacher_targets=teacher_targets,
                )
                formula_dir = draw_root / f"{spec.key}_symbolic_gated_kan"
                payload = json.loads((formula_dir / "pure_symbolic_formula.json").read_text(encoding="utf-8"))
            formulas.append(payload)
            terms = _formula_terms(payload)
            term_sets.append(terms)
            for term in sorted(terms):
                term_rows.append({"task": task, "draw_index": draw_index, "draw_seed": draw_seed, "term": term})
            cal_prediction = _evaluate(payload, calibration[list(inputs)].to_numpy(dtype=np.float64))
            test_prediction = _evaluate(payload, test[list(inputs)].to_numpy(dtype=np.float64))
            predictions["calibration"].append(cal_prediction)
            predictions["test"].append(test_prediction)
            y_test = target_values(test, spec)
            teacher_test = teacher_targets["test"]
            draw_rows.append({
                "task": task,
                "draw_index": draw_index,
                "draw_seed": draw_seed,
                "active_terms": len(terms),
                "formula_to_sampled_teacher_rmse": float(np.sqrt(mean_squared_error(teacher_test, test_prediction))),
                "formula_to_sampled_teacher_r2": float(r2_score(teacher_test, test_prediction)),
                "sampled_teacher_to_tcad_rmse": float(np.sqrt(mean_squared_error(y_test, teacher_test))),
                "sampled_teacher_to_tcad_r2": float(r2_score(y_test, teacher_test)),
                "formula_to_tcad_rmse": float(np.sqrt(mean_squared_error(y_test, test_prediction))),
                "formula_to_tcad_r2": float(r2_score(y_test, test_prediction)),
            })
        if args.fit_only:
            continue
        for left, right in combinations(range(args.draws), 2):
            union = term_sets[left] | term_sets[right]
            intersection = term_sets[left] & term_sets[right]
            prediction_rmse = float(np.sqrt(np.mean((predictions["test"][left] - predictions["test"][right]) ** 2)))
            pair_rows.append({
                "task": task,
                "left_draw": left,
                "right_draw": right,
                "term_jaccard": float(len(intersection) / len(union)) if union else 1.0,
                "test_prediction_disagreement_rmse": prediction_rmse,
            })
        frames = (spec, inputs, partitions)
        ensemble_rows.append(_ensemble_metrics(task, frames, predictions))

    if args.fit_only:
        return 0
    draw_frame = pd.DataFrame(draw_rows)
    term_frame = pd.DataFrame(term_rows)
    pair_frame = pd.DataFrame(pair_rows)
    ensemble_frame = pd.DataFrame(ensemble_rows)
    frequency = term_frame.groupby(["task", "term"]).size().rename("draw_count").reset_index()
    frequency["frequency"] = frequency["draw_count"] / args.draws
    if pair_frame.empty:
        stability = pd.DataFrame([{
            "task": task,
            "pair_count": 0,
            "term_jaccard_mean": np.nan,
            "term_jaccard_min": np.nan,
            "prediction_disagreement_rmse_mean": np.nan,
            "prediction_disagreement_rmse_max": np.nan,
        } for task in args.tasks])
    else:
        stability = pair_frame.groupby("task").agg(
            pair_count=("term_jaccard", "size"),
            term_jaccard_mean=("term_jaccard", "mean"),
            term_jaccard_min=("term_jaccard", "min"),
            prediction_disagreement_rmse_mean=("test_prediction_disagreement_rmse", "mean"),
            prediction_disagreement_rmse_max=("test_prediction_disagreement_rmse", "max"),
        ).reset_index()
    draw_frame.to_csv(args.output / "formula_metrics_by_posterior_draw.csv", index=False)
    frequency.to_csv(args.output / "term_selection_frequency.csv", index=False)
    pair_frame.to_csv(args.output / "pairwise_formula_stability.csv", index=False)
    stability.to_csv(args.output / "formula_stability_summary.csv", index=False)
    ensemble_frame.to_csv(args.output / "formula_ensemble_calibration.csv", index=False)
    protocol = {
        "posterior_draw_count": args.draws,
        "draw_seed_base": args.draw_seed_base,
        "distillation_seed_fixed": args.distillation_seed,
        "symbolic_step_scale": args.symbolic_step_scale,
        "task_step_scale": args.task_step_scale,
        "bkan_retrained": False,
        "architecture_changed": False,
        "split_fractions": list(map(float, args.split_fractions)),
        "formula_fit_groups": f"{100*args.split_fractions[0]:g}% train only",
        "formula_selection_groups": f"{100*args.split_fractions[1]:g}% validation only",
        "formula_conformal_groups": f"{100*args.split_fractions[2]:g}% calibration only",
        "final_evaluation_groups": f"{100*args.split_fractions[3]:g}% test only",
        "formula_nominal_coverage": 1.0 - FORMULA_ALPHA,
        "formula_conformal_rank_rule": "ceil((n+1)*(1-alpha)); no clipping",
        "posterior_variation_isolation": "distillation RNG fixed across draws; observed/derivative/shape losses disabled",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    lines = [
        "# Posterior-Sampled Formula Ensemble Audit",
        "",
        f"- Fixed VI posterior draws per task: {args.draws}",
        "- BKAN architecture/training: unchanged; checkpoints loaded read-only",
        "- Formula optimizer seed: fixed across posterior draws",
        "- Calibration groups are not used for formula fitting or selection",
        "",
        "## Stability",
        "",
        _markdown(stability),
        "",
        "## Formula-ensemble calibration",
        "",
        _markdown(ensemble_frame),
        "",
    ]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(stability.to_string(index=False))
    print(ensemble_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
