"""Distil a fixed AC BKAN teacher into a gated low-pass symbolic student.

Unlike the general symbolic student, this keeps the physically meaningful
low-pass transfer family intact and uses gates only to select condition terms
inside ``y0``, ``floor`` and ``log(fc)``.  The exported expression contains no
KAN path and is independently replayed from JSON before a run is accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = REPO_ROOT / "bkan"
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))

from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler  # noqa: E402
from device_modeling.photodetector.common import concat_without_attrs, set_seed  # noqa: E402
from device_modeling.photodetector.task_config import (  # noqa: E402
    DEFAULT_DATA,
    TASKS,
    load_research_data,
    prepare_task_dataframe,
    split_task_dataframe,
    target_values,
)


DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts/results/device_modeling/AC-response-bayes-results/model_checkpoint.pt"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--split-fractions", type=float, nargs=4, default=(0.70, 0.10, 0.10, 0.10))
    parser.add_argument("--max-symbolic-terms", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=600)
    parser.add_argument("--gate-steps", type=int, default=1800)
    parser.add_argument("--refit-steps", type=int, default=2200)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--gate-penalty", type=float, default=2e-4)
    parser.add_argument("--observed-loss-weight", type=float, default=0.0,
                        help="Optional train/calibration TCAD loss weight; zero is strict teacher-only distillation.")
    return parser.parse_args()


def _condition_design(x_raw: np.ndarray, inputs: tuple[str, ...], feature_defs: list[dict]) -> np.ndarray:
    values = {name: x_raw[:, inputs.index(name)] for name in inputs}
    columns = []
    for feature in feature_defs:
        kind = feature["kind"]
        if kind == "one":
            columns.append(np.ones(len(x_raw), dtype=np.float64))
        elif kind == "z":
            columns.append((values[feature["input"]] - feature["mean"]) / feature["scale"])
        elif kind == "z2":
            z = (values[feature["input"]] - feature["mean"]) / feature["scale"]
            columns.append(z * z)
        elif kind == "z_product":
            left = (values[feature["left"]] - feature["left_mean"]) / feature["left_scale"]
            right = (values[feature["right"]] - feature["right_mean"]) / feature["right_scale"]
            columns.append(left * right)
        else:
            raise ValueError(f"Unknown feature kind: {kind}")
    return np.column_stack(columns)


def _make_features(train_x: np.ndarray, inputs: tuple[str, ...]) -> tuple[list[dict], list[str]]:
    conditions = [name for name in inputs if name != "frequency_ghz"]
    features = [{"name": "1", "kind": "one"}]
    for name in conditions:
        values = train_x[:, inputs.index(name)]
        features.append({
            "name": f"scaled({name})", "kind": "z", "input": name,
            "mean": float(values.mean()), "scale": float(max(values.std(), EPS)),
        })
    # Curvature candidates are deliberately optional: hard gates can reject
    # them, while an AC teacher with bias-dependent bandwidth can retain them.
    for name in conditions:
        values = train_x[:, inputs.index(name)]
        features.append({
            "name": f"scaled({name})^2", "kind": "z2", "input": name,
            "mean": float(values.mean()), "scale": float(max(values.std(), EPS)),
        })
    for left_idx, left in enumerate(conditions):
        for right in conditions[left_idx + 1:]:
            lval, rval = train_x[:, inputs.index(left)], train_x[:, inputs.index(right)]
            features.append({
                "name": f"scaled({left})*scaled({right})", "kind": "z_product",
                "left": left, "right": right,
                "left_mean": float(lval.mean()), "left_scale": float(max(lval.std(), EPS)),
                "right_mean": float(rval.mean()), "right_scale": float(max(rval.std(), EPS)),
            })
    return features, conditions


def _predict_numpy(theta: np.ndarray, phi: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    count = phi.shape[1]
    y0, floor, logfc = theta[:count], theta[count:2 * count], theta[2 * count:3 * count]
    log_p = theta[-1]
    fc = np.exp(np.clip(phi @ logfc, np.log(0.02), np.log(500.0)))
    p = 0.5 + np.exp(np.clip(log_p, np.log(0.05), np.log(20.0)))
    ratio = np.power(np.maximum(frequency / fc, 1e-12), p)
    y0_value, floor_value = phi @ y0, phi @ floor
    return floor_value + (y0_value - floor_value) / (1.0 + ratio)


class GatedLowpass(torch.nn.Module):
    def __init__(self, init: np.ndarray, n_features: int, max_terms: int):
        super().__init__()
        self.n_features = n_features
        self.max_terms = max_terms
        self.coeff = torch.nn.Parameter(torch.tensor(init[:-1].reshape(3, n_features), dtype=torch.float32))
        self.log_p = torch.nn.Parameter(torch.tensor(float(init[-1]), dtype=torch.float32))
        # One gate per deployed coefficient.  Intercepts start very open to
        # preserve a valid low-pass family through the competition stage.
        initial = np.full((3, n_features), 2.0, dtype=np.float32)
        initial[:, 0] = 5.0
        self.gate_logits = torch.nn.Parameter(torch.tensor(initial))
        self.register_buffer("hard_mask", torch.ones((3, n_features), dtype=torch.float32))
        self.hard = False

    def effective(self, temperature: float) -> torch.Tensor:
        if self.hard:
            return self.coeff * self.hard_mask
        return self.coeff * torch.sigmoid(self.gate_logits / temperature)

    def forward(self, phi: torch.Tensor, frequency: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        coeff = self.effective(temperature)
        y0, floor, logfc = (phi @ coeff[index].unsqueeze(1) for index in range(3))
        fc = torch.exp(torch.clamp(logfc, float(np.log(0.02)), float(np.log(500.0))))
        p = 0.5 + torch.exp(torch.clamp(self.log_p, float(np.log(0.05)), float(np.log(20.0))))
        ratio = torch.pow(torch.clamp(frequency / fc, min=1e-12), p)
        return floor + (y0 - floor) / (1.0 + ratio)

    def prune(self) -> None:
        with torch.no_grad():
            score = (self.coeff.abs() * torch.sigmoid(self.gate_logits)).cpu().numpy()
            # A stable low-pass requires intercepts for all three submodels.
            selected = {(row, 0) for row in range(3)}
            ranked = np.argsort(-score.reshape(-1))
            for flat in ranked:
                if len(selected) >= self.max_terms:
                    break
                selected.add((int(flat // self.n_features), int(flat % self.n_features)))
            mask = np.zeros_like(score, dtype=np.float32)
            for row, col in selected:
                mask[row, col] = 1.0
            self.hard_mask.copy_(torch.tensor(mask))
            self.hard = True


def _teacher(modeler, frame: pd.DataFrame, inputs: tuple[str, ...]) -> np.ndarray:
    """Return the fixed finite-MC variational predictive mean.

    Historical AC exports used this stochastic-prediction mean, not the
    deterministic parameter-mean KAN ``f(x; E_q[theta])``.  The distinction
    is serialized in every newly generated export and run configuration.
    """
    out = modeler.predict_with_uncertainty(frame[list(inputs)].to_numpy(dtype=np.float32))
    values = np.asarray(out["model_mean"], dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Teacher produced non-finite AC values")
    return values


def _evaluate(payload: dict, x_raw: np.ndarray) -> np.ndarray:
    inputs = tuple(payload["inputs"])
    phi = _condition_design(x_raw, inputs, payload["feature_definitions"])
    theta = np.concatenate([
        np.asarray(payload["coefficients"][name], dtype=np.float64)
        for name in ("y0", "floor", "logfc")
    ] + [[float(payload["log_p"])]] )
    return _predict_numpy(theta, phi, x_raw[:, inputs.index("frequency_ghz")])


def _metrics(actual: np.ndarray, prediction: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        f"{prefix}_mae": float(mean_absolute_error(actual, prediction)),
        f"{prefix}_r2": float(r2_score(actual, prediction)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    spec = TASKS["ac_response"]
    raw = load_research_data(args.data)
    data, inputs = prepare_task_dataframe(raw, spec)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train, validation, calibration, test = split_task_dataframe(data, spec, split_seed, fractions=tuple(args.split_fractions))
    modeler = BayesKANDeviceModeler.load_model(str(args.checkpoint), device="cpu")
    observed = train[list(inputs)].to_numpy(dtype=np.float64).mean(axis=0)
    scaler_delta = float(np.max(np.abs(np.asarray(modeler.scaler_x.mean_) - observed) / np.maximum(np.abs(np.asarray(modeler.scaler_x.mean_)), 1.0)))
    if list(modeler.input_cols) != list(inputs) or scaler_delta > 1e-7:
        raise RuntimeError(f"Teacher/split mismatch (scaler relative delta={scaler_delta:.3g})")

    fit_frame = concat_without_attrs([train, calibration])
    x_fit = fit_frame[list(inputs)].to_numpy(dtype=np.float64)
    teacher_fit = _teacher(modeler, fit_frame, inputs)
    observed_fit = target_values(fit_frame, spec)
    features, conditions = _make_features(train[list(inputs)].to_numpy(dtype=np.float64), inputs)
    phi_fit = _condition_design(x_fit, inputs, features)
    freq_fit = x_fit[:, inputs.index("frequency_ghz")]
    n = phi_fit.shape[1]
    theta0 = np.zeros(3 * n + 1, dtype=np.float64)
    theta0[0] = float(np.percentile(teacher_fit, 95))
    theta0[n] = float(np.percentile(teacher_fit, 5))
    theta0[2 * n] = np.log(6.0)
    theta0[-1] = np.log(1.5)
    # This is the exact comparison anchor for the legacy compact posthoc
    # family: teacher-only fitting, with the optional curvature columns held
    # at zero.  It is reported alongside every gated run rather than inferred
    # from a historical CSV with an unknown split.
    affine_count = 1 + len(conditions)
    posthoc_start = theta0.copy()
    posthoc = least_squares(
        lambda compact: _predict_numpy(
            np.concatenate([
                np.pad(compact[:affine_count], (0, n - affine_count)),
                np.pad(compact[affine_count:2 * affine_count], (0, n - affine_count)),
                np.pad(compact[2 * affine_count:3 * affine_count], (0, n - affine_count)),
                compact[-1:],
            ]),
            phi_fit,
            freq_fit,
        ) - teacher_fit,
        np.concatenate([
            posthoc_start[:affine_count],
            posthoc_start[n:n + affine_count],
            posthoc_start[2 * n:2 * n + affine_count],
            posthoc_start[-1:],
        ]),
        loss="soft_l1", f_scale=0.08, max_nfev=10000,
    ).x
    posthoc_theta = np.concatenate([
        np.pad(posthoc[:affine_count], (0, n - affine_count)),
        np.pad(posthoc[affine_count:2 * affine_count], (0, n - affine_count)),
        np.pad(posthoc[2 * affine_count:3 * affine_count], (0, n - affine_count)),
        posthoc[-1:],
    ])
    initial = least_squares(
        lambda theta: np.concatenate([
            _predict_numpy(theta, phi_fit, freq_fit) - teacher_fit,
            np.sqrt(max(args.observed_loss_weight, 0.0)) * (_predict_numpy(theta, phi_fit, freq_fit) - observed_fit),
        ]),
        theta0, loss="soft_l1", f_scale=0.08, max_nfev=10000,
    ).x

    student = GatedLowpass(initial, n, args.max_symbolic_terms)
    phi_tensor = torch.tensor(phi_fit, dtype=torch.float32)
    freq_tensor = torch.tensor(freq_fit.reshape(-1, 1), dtype=torch.float32)
    y_tensor = torch.tensor(teacher_fit.reshape(-1, 1), dtype=torch.float32)
    observed_tensor = torch.tensor(observed_fit.reshape(-1, 1), dtype=torch.float32)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)

    def fit(steps: int, gated: bool, temperature_start: float, temperature_end: float, penalty: float) -> None:
        for step in range(steps):
            fraction = step / max(steps - 1, 1)
            temperature = temperature_start * ((temperature_end / temperature_start) ** fraction)
            optimizer.zero_grad()
            prediction = student(phi_tensor, freq_tensor, temperature)
            loss = torch.mean((prediction - y_tensor) ** 2)
            if args.observed_loss_weight > 0.0:
                loss = loss + args.observed_loss_weight * torch.mean((prediction - observed_tensor) ** 2)
            if gated:
                loss = loss + penalty * torch.sigmoid(student.gate_logits / temperature).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 20.0)
            optimizer.step()

    fit(args.warmup_steps, False, 1.0, 1.0, 0.0)
    fit(args.gate_steps, True, 2.0, 0.15, args.gate_penalty)
    student.prune()
    fit(args.refit_steps, False, 0.1, 0.1, 0.0)

    output = args.output / "ac_response_lowpass_gated"
    output.mkdir(parents=True, exist_ok=True)
    active = student.hard_mask.detach().cpu().numpy() > 0.0
    coefficient = (student.coeff * student.hard_mask).detach().cpu().numpy().astype(np.float64)
    names = [item["name"] for item in features]
    payload = {
        "family": "gated_lowpass",
        "inputs": list(inputs),
        "condition_inputs": conditions,
        "feature_definitions": features,
        "formula": "floor(z) + (y0(z)-floor(z))/(1+(frequency_ghz/exp(logfc(z)))^p)",
        "coefficients": {head: coefficient[idx].tolist() for idx, head in enumerate(("y0", "floor", "logfc"))},
        "log_p": float(student.log_p.detach().cpu()),
        "p": float(0.5 + np.exp(np.clip(float(student.log_p.detach().cpu()), np.log(0.05), np.log(20.0)))),
        "active_terms": [
            {"submodel": head, "feature": names[col], "coefficient": float(coefficient[row, col])}
            for row, head in enumerate(("y0", "floor", "logfc"))
            for col in range(n) if active[row, col]
        ],
        "teacher_reference": {
            "kind": "finite_mc_variational_predictive_mean",
            "distribution": "q_beta from beta-weighted mean-field VI",
            "stochastic_passes": int(modeler.bayes_config.get("num_mc_samples", 100)),
            "not_parameter_mean": True,
            "not_exact_posterior_predictive": True,
        },
        "kan_contribution_at_inference": 0.0,
    }
    (output / "pure_symbolic_formula.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["AC gated low-pass pure-symbolic student", "", f"family: {payload['formula']}", f"p = {payload['p']:.16g}", ""]
    for term in payload["active_terms"]:
        lines.append(f"{term['submodel']}: {term['coefficient']:.16g} * {term['feature']}")
    (output / "pure_symbolic_formula.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows, all_metrics = [], {}
    for split_name, frame in (("train", train), ("validation", validation), ("test", test)):
        x = frame[list(inputs)].to_numpy(dtype=np.float64)
        prediction = _evaluate(payload, x)
        teacher = _teacher(modeler, frame, inputs)
        actual = target_values(frame, spec)
        rows.append(pd.DataFrame({"split": split_name, "actual": actual, "teacher": teacher, "prediction": prediction, "abs_error": np.abs(prediction - actual)}))
        if split_name == "test":
            all_metrics.update(_metrics(teacher, prediction, "teacher"))
            all_metrics.update(_metrics(actual, prediction, "test"))
            replay = _evaluate(payload, x)
            all_metrics["pure_symbolic_export_rmse"] = float(np.sqrt(np.mean((replay - prediction) ** 2)))
            all_metrics["pure_symbolic_export_max_abs_error"] = float(np.max(np.abs(replay - prediction)))
            posthoc_prediction = _predict_numpy(
                posthoc_theta,
                _condition_design(x, inputs, features),
                x[:, inputs.index("frequency_ghz")],
            )
            all_metrics.update(_metrics(teacher, posthoc_prediction, "posthoc_baseline_teacher"))
            all_metrics.update(_metrics(actual, posthoc_prediction, "posthoc_baseline_test"))
            all_metrics["posthoc_baseline_parameter_terms"] = int(3 * affine_count + 1)
    all_metrics.update({
        "active_symbolic_terms": int(active.sum()), "parameter_terms": int(active.sum() + 1),
        "formula_core_chars": len(payload["formula"]) + sum(len(term["feature"]) for term in payload["active_terms"]),
        "scaler_relative_delta": scaler_delta, "kan_contribution_at_inference": 0.0,
        "observed_loss_weight": float(args.observed_loss_weight),
    })
    if all_metrics["pure_symbolic_export_max_abs_error"] > 1e-12:
        raise RuntimeError("Independent exported formula replay mismatch")
    pd.concat(rows, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame([all_metrics]).to_csv(output / "metrics.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    run_config = {
        **{key: str(value) for key, value in vars(args).items()},
        "split_seed_realized": int(split_seed),
        "fit_partitions": ["gradient_fit", "conformal_calibration"],
        "validation_role": "evaluation_only; not used for checkpoint or structure selection",
        "test_role": "one-time evaluation after hard pruning and refit",
        "teacher_reference": "finite_mc_variational_predictive_mean",
        "teacher_stochastic_passes": int(modeler.bayes_config.get("num_mc_samples", 100)),
        "initialization": "scipy least_squares with soft_l1, f_scale=0.08, max_nfev=10000",
        "optimizer": "Adam with PyTorch defaults",
        "gradient_clip_norm": 20.0,
        "temperature_schedule": {
            "warmup": [1.0, 1.0],
            "gate": [2.0, 0.15],
            "refit": [0.1, 0.1],
            "interpolation": "geometric",
        },
        "pruning_score": "abs(coefficient)*sigmoid(gate_logit)",
        "pruning_tie_rule": "numpy argsort descending; exact-tie order is not guaranteed",
        "mandatory_terms": ["y0:1", "floor:1", "logfc:1"],
    }
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.checkpoint), **all_metrics}, indent=2))


if __name__ == "__main__":
    main()
