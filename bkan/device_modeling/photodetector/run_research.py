"""Unified deterministic KAN and Bayesian KAN device-model entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BKAN_ROOT = Path(__file__).resolve().parents[2]
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))
REPO_ROOT = BKAN_ROOT.parent

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from kan import KAN
from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler
from device_modeling.photodetector.common import concat_without_attrs, set_seed, without_attrs
from device_modeling.photodetector.diagnostics import (
    plot_loss_curve,
    plot_ood_detection,
    plot_reliability_diagram,
    plot_uncertainty_analysis,
)
from device_modeling.photodetector.symbolic_gated_kan import train_symbolic_gated
from device_modeling.photodetector.task_config import (
    DARK_CURRENT_ZERO_FLOOR,
    TASKS,
    DEFAULT_CAPACITANCE_DATA,
    load_capacitance_data,
    load_research_data,
    prepare_task_dataframe,
    split_task_dataframe,
    task_group_labels,
    target_values,
)


DEFAULT_20260617_ROOT = Path(r"<FROZEN_TCAD_ROOT>/Data\Data_20260617_0243_181408")
DEFAULT_20260617_DATA = DEFAULT_20260617_ROOT / "data"
DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "results"
DEFAULT_RESULTS = DEFAULT_RUN_ROOT / "device_modeling"


def regression_metrics(y_true, y_pred):
    return {
        "rmse_model_space": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_model_space": float(mean_absolute_error(y_true, y_pred)),
        "r2_model_space": float(r2_score(y_true, y_pred)),
    }


def _task_float_override(items, spec, default):
    names = {spec.key, spec.name, spec.result_subdir}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid TASK=VALUE override: {item}")
        task_name, value = item.split("=", 1)
        if task_name in names:
            return float(value)
    return default


def _normalize_metric_values(metrics):
    normalized = {}
    for key, value in metrics.items():
        key = key.lower()
        if isinstance(value, (int, float, np.integer, np.floating)):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def conformal_group_columns(frame, spec):
    """Return the independent grouped-split unit used for conformal scores."""
    if "_curve_id" not in frame.columns:
        raise ValueError(
            f"{spec.name}: grouped conformal calibration requires _curve_id"
        )
    return ["_curve_id"]


def derivative_targets(frame, spec, inputs):
    """Finite-difference d(target)/d(voltage) references for complete curves."""
    if "voltage" not in spec.axis_col:
        return None
    derivatives = np.full(len(frame), np.nan, dtype=np.float64)
    if "_curve_id" in frame.columns:
        grouped = frame.groupby("_curve_id", sort=False)
    else:
        condition_cols = [column for column in inputs if column != spec.axis_col]
        grouped = frame.groupby(condition_cols, sort=False)
    for _, group in grouped:
        ordered = group.sort_values(spec.axis_col)
        axis = ordered[spec.axis_col].to_numpy(dtype=np.float64)
        values = target_values(ordered, spec)
        if len(axis) < 3 or len(np.unique(axis)) != len(axis):
            raise ValueError(f"Cannot construct voltage derivative reference for {spec.name}")
        derivative = np.gradient(values, axis, edge_order=2)
        derivatives[ordered.index.to_numpy()] = derivative
    if not np.all(np.isfinite(derivatives)):
        raise ValueError(f"Non-finite voltage derivative reference for {spec.name}")
    return derivatives


def _compact_loss(model, x_values, y_values, derivative_scaled, axis_index, args, training):
    needs_derivative = derivative_scaled is not None
    x_model = x_values.detach().clone().requires_grad_(needs_derivative)
    prediction = model(x_model)
    value_loss = F.mse_loss(prediction, y_values)
    derivative_loss = torch.zeros((), device=x_values.device)
    if needs_derivative:
        gradient = torch.autograd.grad(
            prediction.sum(), x_model, create_graph=training, retain_graph=training
        )[0][:, [axis_index]]
        derivative_loss = F.smooth_l1_loss(
            gradient, derivative_scaled, beta=args.derivative_huber_delta
        )
    return value_loss + args.derivative_weight * derivative_loss, value_loss, derivative_loss


def _clone_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _new_deterministic_model(input_count, grid, args, device):
    return KAN(
        width=[input_count, args.width, 1],
        grid=grid,
        k=args.spline_order,
        seed=args.seed,
        device=device,
        symbolic_enabled=False,
        auto_save=False,
    )


def _train_grid_candidate(
    parent_model,
    grid,
    inputs,
    x_train,
    y_train,
    x_val,
    y_val,
    train_derivative,
    val_derivative,
    axis_index,
    args,
    device,
):
    model = _new_deterministic_model(len(inputs), grid, args, device)
    if parent_model is not None:
        model.initialize_from_another_model(parent_model, x_train)
        initialized_state = _clone_state(model)
        model = _new_deterministic_model(len(inputs), grid, args, device)
        model.load_state_dict(initialized_state)
    optimizer = torch.optim.LBFGS(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lbfgs_learning_rate,
        max_iter=args.lbfgs_inner_iterations,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    history = []
    best_state = None
    best_score = float("inf")
    best_iteration = 0
    stale_iterations = 0

    for iteration in range(1, args.steps_per_grid + 1):
        model.train()

        def closure():
            optimizer.zero_grad()
            total_loss, _, _ = _compact_loss(
                model, x_train, y_train, train_derivative, axis_index, args, training=True
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite compact-model objective at grid={grid}")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
            return total_loss

        optimizer.step(closure)
        model.eval()
        train_total, train_value, train_deriv = _compact_loss(
            model, x_train, y_train, train_derivative, axis_index, args, training=False
        )
        val_total, val_value, val_deriv = _compact_loss(
            model, x_val, y_val, val_derivative, axis_index, args, training=False
        )
        score = float(val_total.detach().cpu())
        history.append(
            {
                "grid": grid,
                "iteration": iteration,
                "train_total_loss": float(train_total.detach().cpu()),
                "train_value_loss": float(train_value.detach().cpu()),
                "train_derivative_loss": float(train_deriv.detach().cpu()),
                "validation_total_loss": score,
                "validation_value_loss": float(val_value.detach().cpu()),
                "validation_derivative_loss": float(val_deriv.detach().cpu()),
            }
        )
        if score < best_score:
            best_score = score
            best_state = _clone_state(model)
            best_iteration = iteration
            stale_iterations = 0
        else:
            stale_iterations += 1
        if stale_iterations >= args.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError(f"No finite state obtained while training grid={grid}")
    model.load_state_dict(best_state)
    return model, best_score, best_iteration, history


def train_deterministic(spec, inputs, train, validation, test, args, device, output_dir):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(train[list(inputs)].to_numpy(dtype=np.float32)).astype(np.float32)
    y_train = y_scaler.fit_transform(target_values(train, spec).reshape(-1, 1)).astype(np.float32)
    x_val = x_scaler.transform(validation[list(inputs)].to_numpy(dtype=np.float32)).astype(np.float32)
    y_val = y_scaler.transform(target_values(validation, spec).reshape(-1, 1)).astype(np.float32)
    axis_index = list(inputs).index(spec.axis_col)
    train_derivative_raw = derivative_targets(train, spec, inputs)
    val_derivative_raw = derivative_targets(validation, spec, inputs)
    derivative_scale = float(x_scaler.scale_[axis_index] / y_scaler.scale_[0])
    train_derivative = (
        torch.from_numpy((train_derivative_raw * derivative_scale).reshape(-1, 1).astype(np.float32)).to(device)
        if train_derivative_raw is not None else None
    )
    val_derivative = (
        torch.from_numpy((val_derivative_raw * derivative_scale).reshape(-1, 1).astype(np.float32)).to(device)
        if val_derivative_raw is not None else None
    )
    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    x_val_tensor = torch.from_numpy(x_val).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)

    set_seed(args.seed)
    model = None
    chosen_grid = None
    selected_score = float("inf")
    training_history = []
    grid_records = []
    for grid in args.grid_schedule:
        candidate, score, best_iteration, stage_history = _train_grid_candidate(
            model,
            grid,
            inputs,
            x_train_tensor,
            y_train_tensor,
            x_val_tensor,
            y_val_tensor,
            train_derivative,
            val_derivative,
            axis_index,
            args,
            device,
        )
        accepted = model is None or score < selected_score * (1.0 - args.min_relative_improvement)
        grid_records.append(
            {
                "grid": grid,
                "best_iteration": best_iteration,
                "best_validation_objective": score,
                "accepted": accepted,
            }
        )
        training_history.extend(stage_history)
        if not accepted:
            break
        model = candidate
        chosen_grid = grid
        selected_score = score

    task_dir = output_dir / f"{spec.key}_deterministic_kan"
    task_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    metrics = {}
    prediction_frames = []
    for split_name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        x_values = x_scaler.transform(frame[list(inputs)].to_numpy(dtype=np.float32)).astype(np.float32)
        x_tensor = torch.from_numpy(x_values).to(device)
        reference_derivative = derivative_targets(frame, spec, inputs)
        if reference_derivative is None:
            with torch.no_grad():
                prediction_scaled = model(x_tensor).detach().cpu().numpy()
            prediction_derivative = None
        else:
            x_tensor = x_tensor.detach().clone().requires_grad_(True)
            prediction_tensor = model(x_tensor)
            gradient_scaled = torch.autograd.grad(prediction_tensor.sum(), x_tensor)[0][:, axis_index]
            prediction_scaled = prediction_tensor.detach().cpu().numpy()
            prediction_derivative = (
                gradient_scaled.detach().cpu().numpy()
                * float(y_scaler.scale_[0] / x_scaler.scale_[axis_index])
            )
        prediction = y_scaler.inverse_transform(prediction_scaled).reshape(-1)
        actual = target_values(frame, spec)
        physical_prediction = (
            np.power(10.0, np.clip(prediction, -300.0, 300.0))
            if spec.use_log_transform
            else prediction
        )
        split_metrics = regression_metrics(actual, prediction)
        if reference_derivative is not None:
            split_metrics["derivative_mae_model_space"] = float(
                mean_absolute_error(reference_derivative, prediction_derivative)
            )
            split_metrics["derivative_rmse_model_space"] = float(
                np.sqrt(mean_squared_error(reference_derivative, prediction_derivative))
            )
        split_metrics["mae_physical"] = float(
            mean_absolute_error(frame[spec.target_col].to_numpy(dtype=np.float64), physical_prediction)
        )
        metrics.update({f"{split_name}_{key}": value for key, value in split_metrics.items()})
        predictions = without_attrs(frame)
        predictions["split"] = split_name
        predictions["actual_model_space"] = actual
        predictions["prediction_model_space"] = prediction
        predictions["prediction_physical"] = physical_prediction
        predictions["abs_error_model_space"] = np.abs(prediction - actual)
        if reference_derivative is not None:
            predictions["reference_derivative_model_space"] = reference_derivative
            predictions["prediction_derivative_model_space"] = prediction_derivative
            predictions["abs_derivative_error_model_space"] = np.abs(
                prediction_derivative - reference_derivative
            )
        prediction_frames.append(predictions)

    concat_without_attrs(prediction_frames).to_csv(
        task_dir / "predictions.csv", index=False
    )
    pd.DataFrame(training_history).to_csv(task_dir / "training_history.csv", index=False)
    pd.DataFrame(grid_records).to_csv(task_dir / "grid_selection.csv", index=False)
    torch.save(
        {
            "model_state": model.state_dict(),
            "width": [len(inputs), args.width, 1],
            "grid": chosen_grid,
            "k": args.spline_order,
            "seed": args.seed,
            "inputs": list(inputs),
            "target_col": spec.target_col,
            "use_log_transform": spec.use_log_transform,
            "y_bounds": spec.y_bounds,
            "target_space": "log10" if spec.use_log_transform else "raw",
            "dark_current_zero_floor": DARK_CURRENT_ZERO_FLOOR if spec.key == "dark_current" else None,
            "x_scaler_mean": x_scaler.mean_,
            "x_scaler_scale": x_scaler.scale_,
            "y_scaler_mean": y_scaler.mean_,
            "y_scaler_scale": y_scaler.scale_,
            "training_method": "validation_selected_grid_refinement_full_batch_lbfgs",
            "derivative_weight": args.derivative_weight if train_derivative is not None else 0.0,
            "derivative_huber_delta": args.derivative_huber_delta,
            "max_gradient_norm": args.max_gradient_norm,
        },
        task_dir / "model_checkpoint.pt",
    )
    metrics["train_samples"] = int(len(train))
    metrics["validation_samples"] = int(len(validation))
    metrics["test_samples"] = int(len(test))
    metrics["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    metrics["selected_grid"] = int(chosen_grid)
    metrics["validation_compact_objective"] = float(selected_score)
    return metrics


def train_bayesian(
    spec,
    inputs,
    train,
    validation,
    calibration,
    test,
    args,
    device,
    output_dir,
    return_modeler=False,
):
    set_seed(args.seed)
    method = args.bayes_method
    task_leaf = spec.result_subdir if method == "vi" else f"{spec.result_subdir}_{method}"
    task_dir = output_dir / task_leaf
    hmc_step_size = _task_float_override(args.hmc_task_step_size, spec, args.hmc_step_size)
    modeler = BayesKANDeviceModeler(task_name=spec.name, results_dir=str(task_dir), device=device)
    modeler.load_data(
        train,
        input_cols=list(inputs),
        output_col=spec.target_col,
        use_log_transform=spec.use_log_transform,
        y_bounds=spec.y_bounds,
    )
    output_width = 2 if method == "vi" else 1
    modeler.build_model(
        {
            "width": [None, args.width, output_width],
            "seed": args.seed,
            "grid": args.grid,
            "k": args.spline_order,
            "grid_range": [-3, 3],
            "kl_weight": args.kl_weight,
            "num_mc_samples": args.mc_samples,
            "prior_mu": 0.0,
            "prior_log_sigma": 0.0,
            "posterior_init_sigma": 0.1,
            "likelihood": "gaussian",
            "inference_method": method,
            "seed": args.seed,
            "prediction_seed": args.prediction_seed,
            "dropout_rate": args.dropout_rate,
            "hmc_samples": args.hmc_samples,
            "hmc_burn_in": args.hmc_burn_in,
            "hmc_leapfrog_steps": args.hmc_leapfrog_steps,
            "hmc_step_size": hmc_step_size,
            "hmc_thinning": args.hmc_thinning,
            "hmc_prior_scale": args.hmc_prior_scale,
            "hmc_parameter_scope": args.hmc_parameter_scope,
            "hmc_mass_matrix": args.hmc_mass_matrix,
            "hmc_chains": args.hmc_chains,
            "hmc_chain_init_jitter": args.hmc_chain_init_jitter,
            "hmc_seed": args.seed,
            "hmc_defer_sampling": method == "hmc" and args.hmc_noise_source != "train",
        }
    )
    modeler.train(
        {
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.learning_rate,
            "weight_decay": 1e-5,
            "val_freq": max(1, min(10, args.epochs)),
            "train_mc_samples": args.train_mc_samples,
            "validation_mc_samples": args.validation_mc_samples,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "validation_seed": args.prediction_seed,
        },
        df_val=validation,
    )
    calibration_frame = concat_without_attrs([validation, calibration])
    if method == "hmc" and args.hmc_noise_source != "train":
        if args.hmc_noise_source == "validation":
            noise_frame = validation
        elif args.hmc_noise_source == "calibration":
            noise_frame = calibration
        elif args.hmc_noise_source == "validation_calibration":
            noise_frame = calibration_frame
        else:
            raise ValueError(f"Unknown HMC noise source: {args.hmc_noise_source}")
        noise_var = modeler.estimate_residual_noise_var(noise_frame)
        modeler.global_aleatoric_var = max(
            noise_var * args.hmc_noise_scale,
            args.hmc_noise_floor,
        )
        print(
            f"  HMC noise variance from {args.hmc_noise_source}: "
            f"{modeler.global_aleatoric_var:.6g} "
            f"(scale={args.hmc_noise_scale:g}, floor={args.hmc_noise_floor:g})"
        )
        modeler._fit_hmc_posterior()
    if method == "hmc":
        if args.hmc_calibration_source == "validation":
            method_calibration_frame = validation
        elif args.hmc_calibration_source == "calibration":
            method_calibration_frame = calibration
        elif args.hmc_calibration_source == "validation_calibration":
            method_calibration_frame = calibration_frame
        else:
            raise ValueError(
                f"Unknown HMC calibration source: {args.hmc_calibration_source}"
            )
    else:
        method_calibration_frame = calibration
    # Conformal calibration must use the same independent unit as the grouped
    # train/validation/calibration/test split.  In particular, capacitance rows
    # are split by DEVICE sample (``_curve_id``); grouping the calibration
    # scores by the remaining inputs would instead create 25 correlated C--V
    # sub-curves per DEVICE and incorrectly inflate 24 calibration devices to
    # 600 conformal scores.
    calibration_group_cols = conformal_group_columns(
        method_calibration_frame,
        spec,
    )
    modeler.calibrate_uncertainty(
        df_calib=method_calibration_frame,
        calibration_unit=args.conformal_unit,
        group_cols=calibration_group_cols if args.conformal_unit == "curve" else None,
    )
    metrics = modeler.evaluate_on_test_set(df_test=test)
    hmc_function_diagnostics = (
        modeler.hmc_function_space_diagnostics(
            test[list(inputs)].to_numpy(dtype=np.float32)
        )
        if method == "hmc"
        else {}
    )
    if args.diagnostics:
        try:
            if modeler.experiment is not None and modeler.experiment.history.get("train_loss"):
                plot_loss_curve(modeler)
            plot_uncertainty_analysis(modeler, test)
            ood_result = plot_ood_detection(modeler, train, test)
            plot_reliability_diagram(modeler, test)
            metrics["OOD_Ratio"] = float(ood_result["ood_ratio"])
        except Exception as exc:
            print(f"Warning: Bayesian diagnostics skipped for {spec.name}: {exc}")
    derivative_reference = derivative_targets(test, spec, inputs)
    if derivative_reference is not None and method == "vi":
        derivative_key = "dlogI_dX_mean" if spec.use_log_transform else "dI_dX_mean"
        derivative_prediction = modeler.predict_derivative_with_uncertainty(
            test[list(inputs)].to_numpy(dtype=np.float32),
            spec.axis_col,
            n_mc=args.mc_samples,
        )[derivative_key]
        metrics["Derivative_MAE_Model_Space"] = float(
            mean_absolute_error(derivative_reference, derivative_prediction)
        )
        metrics["Derivative_RMSE_Model_Space"] = float(
            np.sqrt(mean_squared_error(derivative_reference, derivative_prediction))
        )
    modeler.save_model()
    metrics = _normalize_metric_values(metrics)
    metrics["inference_method"] = method
    metrics["comparison_eligible"] = True
    metrics["comparison_status"] = "eligible"
    if hasattr(modeler, "hmc_acceptance_rate"):
        metrics["hmc_acceptance_rate"] = float(modeler.hmc_acceptance_rate)
    if method == "hmc":
        metrics["hmc_step_size"] = float(hmc_step_size)
        metrics["hmc_leapfrog_steps"] = int(args.hmc_leapfrog_steps)
        metrics["hmc_samples"] = int(args.hmc_samples)
        metrics["hmc_burn_in"] = int(args.hmc_burn_in)
        metrics["hmc_thinning"] = int(args.hmc_thinning)
        metrics["hmc_prior_scale"] = float(args.hmc_prior_scale)
        metrics["hmc_parameter_scope"] = args.hmc_parameter_scope
        metrics["hmc_mass_matrix"] = args.hmc_mass_matrix
        metrics["hmc_chains"] = int(args.hmc_chains)
        metrics["hmc_chain_init_jitter"] = float(args.hmc_chain_init_jitter)
        metrics["hmc_chain_acceptance_rates"] = json.dumps(
            modeler.hmc_chain_acceptance_rates
        )
        metrics["hmc_noise_source"] = args.hmc_noise_source
        metrics["hmc_calibration_source"] = args.hmc_calibration_source
        metrics["hmc_noise_var"] = float(getattr(modeler, "global_aleatoric_var", np.nan))
        for key, value in modeler.hmc_diagnostics.items():
            metrics[f"hmc_{key}"] = value
        for key, value in modeler.hmc_mass_matrix_diagnostics.items():
            metrics[f"hmc_mass_{key}"] = value
        for key, value in hmc_function_diagnostics.items():
            metrics[f"hmc_function_{key}"] = value
        rhat = modeler.hmc_diagnostics.get("rhat_max", np.inf)
        ess = modeler.hmc_diagnostics.get("ess_bulk_min", 0.0)
        function_rhat = hmc_function_diagnostics.get("rhat_max", np.inf)
        function_ess = hmc_function_diagnostics.get("ess_bulk_min", 0.0)
        hmc_eligible = (
            args.hmc_chains >= 2
            and rhat <= args.hmc_rhat_threshold
            and ess >= args.hmc_ess_threshold
            and function_rhat <= args.hmc_rhat_threshold
            and function_ess >= args.hmc_ess_threshold
        )
        metrics["comparison_eligible"] = hmc_eligible
        metrics["comparison_status"] = (
            "eligible"
            if hmc_eligible
            else (
                "excluded_multichain_not_verified"
                if args.hmc_chains < 2
                else "excluded_nonconverged"
            )
        )
        metrics["hmc_diagnostic_status"] = (
            "passed" if hmc_eligible else (
                "not_verified" if args.hmc_chains < 2 else "failed"
            )
        )
    if return_modeler:
        return metrics, modeler
    return metrics


def train_bayesian_comparison(spec, inputs, train, validation, calibration, test, args, device, output_dir):
    rows = []
    original_method = args.bayes_method
    for method in ("dropout", "hmc", "vi"):
        args.bayes_method = method
        metrics = train_bayesian(spec, inputs, train, validation, calibration, test, args, device, output_dir)
        rows.append({"task": spec.name, "model": f"bayesian_kan_{method}", **metrics})
    args.bayes_method = original_method
    return rows


def bayesian_precision_weights(modeler, frame, inputs, args):
    prediction = modeler.predict_with_uncertainty(frame[list(inputs)].to_numpy(dtype=np.float32))
    mean = np.asarray(prediction["mean"], dtype=np.float64).reshape(-1)
    sigma = np.asarray(prediction["model_std"], dtype=np.float64).reshape(-1)
    if len(mean) != len(frame) or len(sigma) != len(frame):
        raise ValueError("Bayesian prediction length does not match the requested frame")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sigma)):
        raise ValueError("Bayesian predictions contain non-finite values")

    positive = sigma[sigma > 0.0]
    if len(positive) == 0:
        raise ValueError("Bayesian model_std is zero for all samples; precision weighting is undefined")
    sigma_floor = max(float(np.median(positive)) * args.bayesian_weight_floor_ratio, 1e-12)
    precision = 1.0 / (sigma * sigma + sigma_floor * sigma_floor)
    precision /= float(np.mean(precision))
    precision = np.clip(precision, args.bayesian_weight_min, args.bayesian_weight_max)
    precision /= float(np.mean(precision))
    return precision.astype(np.float32), sigma.astype(np.float32), mean.astype(np.float32)


def train_bayesian_symbolic(spec, inputs, train, validation, calibration, test, args, device, output_dir):
    bayesian_metrics, modeler = train_bayesian(
        spec,
        inputs,
        train,
        validation,
        calibration,
        test,
        args,
        device,
        output_dir,
        return_modeler=True,
    )
    symbolic_train = concat_without_attrs([train, calibration])
    sample_weights = {}
    uncertainty = {}
    teacher_targets = {}
    for split_name, frame in (
        ("train", symbolic_train),
        ("validation", validation),
        ("test", test),
    ):
        sample_weights[split_name], uncertainty[split_name], teacher_targets[split_name] = bayesian_precision_weights(
            modeler, frame, inputs, args
        )
    symbolic_metrics = train_symbolic_gated(
        spec,
        inputs,
        symbolic_train,
        validation,
        test,
        args,
        device,
        output_dir,
        sample_weights=sample_weights,
        uncertainty=uncertainty,
        teacher_targets=teacher_targets,
    )
    return {
        **{f"bayesian_{key}": value for key, value in bayesian_metrics.items()},
        **symbolic_metrics,
    }


def _split_manifest(spec, inputs, partitions):
    condition_cols = [
        column for column in inputs if column != spec.axis_col
    ]
    rows = []
    for split_name, frame in zip(
        ("train", "validation", "calibration", "test"),
        partitions,
    ):
        labels = task_group_labels(frame, spec)
        work = frame.copy()
        work["_group_label"] = labels.to_numpy()
        for group_label, group in work.groupby("_group_label", sort=True):
            row = {
                "task": spec.name,
                "split": split_name,
                "group_id": str(group_label),
                "n_points": int(len(group)),
            }
            for column in condition_cols:
                values = group[column].dropna()
                if values.nunique(dropna=True) <= 1:
                    row[column] = values.iloc[0] if len(values) else np.nan
                elif pd.api.types.is_numeric_dtype(values):
                    row[f"{column}_min"] = float(values.min())
                    row[f"{column}_max"] = float(values.max())
                    row[f"{column}_nunique"] = int(values.nunique(dropna=True))
                else:
                    row[f"{column}_nunique"] = int(values.nunique(dropna=True))
            rows.append(row)
    return pd.DataFrame(rows)


def run_task(spec, raw, args, device):
    data, inputs = prepare_task_dataframe(raw, spec)
    partitions = split_task_dataframe(
        data,
        spec,
        args.seed,
        fractions=args.split_fractions,
    )
    train, validation, calibration, test = partitions
    split_manifest = _split_manifest(spec, inputs, partitions)
    split_manifest.to_csv(
        args.output / f"{spec.key}_split_manifest.csv",
        index=False,
    )
    curve_counts = {
        f"{split_name}_curve_count": int(
            task_group_labels(frame, spec).nunique()
        )
        for split_name, frame in zip(
            ("train", "validation", "calibration", "test"),
            partitions,
        )
    }
    sample_counts = {
        f"{split_name}_samples": int(len(frame))
        for split_name, frame in zip(
            ("train", "validation", "calibration", "test"),
            partitions,
        )
    }
    print(
        f"\n[{spec.name}] samples={len(data)}, inputs={list(inputs)}, "
        f"split={len(train)}/{len(validation)}/{len(calibration)}/{len(test)}, "
        "curves="
        f"{curve_counts['train_curve_count']}/"
        f"{curve_counts['validation_curve_count']}/"
        f"{curve_counts['calibration_curve_count']}/"
        f"{curve_counts['test_curve_count']}"
    )

    rows = []
    if args.model in {"deterministic", "both", "all"}:
        metrics = train_deterministic(spec, inputs, train, validation, test, args, device, args.output)
        rows.append({"task": spec.name, "model": "deterministic_kan", **metrics})
    if args.model in {"bayesian", "both", "all"}:
        metrics = train_bayesian(
            spec, inputs, train, validation, calibration, test, args, device, args.output
        )
        rows.append({"task": spec.name, "model": f"bayesian_kan_{args.bayes_method}", **metrics})
    if args.model == "bayesian_compare":
        rows.extend(
            train_bayesian_comparison(
                spec, inputs, train, validation, calibration, test, args, device, args.output
            )
        )
    if args.model in {"symbolic_gated", "all"}:
        symbolic_train = concat_without_attrs([train, calibration])
        metrics = train_symbolic_gated(
            spec, inputs, symbolic_train, validation, test, args, device, args.output
        )
        rows.append({"task": spec.name, "model": "symbolic_gated_kan", **metrics})
    if args.model in {"bayesian_symbolic", "all"}:
        metrics = train_bayesian_symbolic(
            spec, inputs, train, validation, calibration, test, args, device, args.output
        )
        rows.append({"task": spec.name, "model": "bayesian_symbolic_gated_kan", **metrics})
    split_names = ("train", "validation", "calibration", "test")
    split_fractions = {
        f"{name}_fraction": float(value)
        for name, value in zip(split_names, args.split_fractions)
    }
    for row in rows:
        row.update(curve_counts)
        for key, value in sample_counts.items():
            row.setdefault(key, value)
        row.update(split_fractions)
        row["split_seed"] = int(args.seed)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument(
        "--model",
        choices=["deterministic", "bayesian", "bayesian_compare", "both", "symbolic_gated", "bayesian_symbolic", "all"],
        default="both",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_20260617_DATA,
        help="Main device-modeling curve directory or combined table.",
    )
    parser.add_argument(
        "--capacitance-data",
        type=Path,
        default=DEFAULT_CAPACITANCE_DATA,
        help=(
            "AC-CV capacitance CSV for the capacitance task, or a root directory containing "
            "supplemental_charge_ac/supplemental_electrical/ac_cv."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-fractions",
        type=float,
        nargs=4,
        metavar=("TRAIN", "VALIDATION", "CALIBRATION", "TEST"),
        default=[0.70, 0.10, 0.10, 0.10],
        help=(
            "Condition-curve split fractions. Repeated-UQ experiments use "
            "0.65 0.10 0.15 0.10 so calibration has at least 19 curves."
        ),
    )
    parser.add_argument("--epochs", type=int, default=300, help="Bayesian KAN training epochs")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--grid", type=int, default=8, help="Bayesian KAN spline grid size")
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument(
        "--grid-schedule",
        type=int,
        nargs="+",
        default=[2, 4, 8, 12],
        help="Deterministic KAN grids considered in coarse-to-fine validation order",
    )
    parser.add_argument("--steps-per-grid", type=int, default=30)
    parser.add_argument("--lbfgs-inner-iterations", type=int, default=5)
    parser.add_argument("--lbfgs-learning-rate", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--min-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--derivative-weight", type=float, default=0.05)
    parser.add_argument("--derivative-huber-delta", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--train-mc-samples", type=int, default=3)
    parser.add_argument("--validation-mc-samples", type=int, default=20)
    parser.add_argument(
        "--prediction-seed",
        type=int,
        default=1729,
        help="Fixed seed for reproducible posterior predictive draws.",
    )
    parser.add_argument(
        "--conformal-unit",
        choices=["point", "curve"],
        default="curve",
        help="Use curve maxima for simultaneous curve coverage, or point scores for marginal coverage.",
    )
    parser.add_argument("--bayes-method", choices=["vi", "dropout", "hmc"], default="vi")
    parser.add_argument("--dropout-rate", type=float, default=0.05)
    parser.add_argument("--hmc-samples", type=int, default=120)
    parser.add_argument("--hmc-burn-in", type=int, default=80)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=8)
    parser.add_argument("--hmc-step-size", type=float, default=1e-3)
    parser.add_argument(
        "--hmc-task-step-size",
        nargs="*",
        default=[],
        metavar="TASK=STEP",
        help="Task-specific HMC step-size overrides, e.g. dark_current=5e-4 I_photo=3e-3.",
    )
    parser.add_argument("--hmc-thinning", type=int, default=1)
    parser.add_argument("--hmc-prior-scale", type=float, default=1.0)
    parser.add_argument(
        "--hmc-parameter-scope",
        choices=["all", "last_layer", "last_layer_coef"],
        default="all",
    )
    parser.add_argument(
        "--hmc-mass-matrix",
        choices=["identity", "hessian"],
        default="identity",
    )
    parser.add_argument("--hmc-chains", type=int, default=4)
    parser.add_argument("--hmc-chain-init-jitter", type=float, default=1e-3)
    parser.add_argument(
        "--hmc-noise-source",
        choices=["train", "validation", "calibration", "validation_calibration"],
        default="validation",
        help="Residual split used to set the fixed HMC observation-noise variance.",
    )
    parser.add_argument(
        "--hmc-calibration-source",
        choices=["validation", "calibration", "validation_calibration"],
        default="calibration",
        help="Held-out split used for HMC conformal interval calibration.",
    )
    parser.add_argument("--hmc-noise-scale", type=float, default=1.0)
    parser.add_argument("--hmc-noise-floor", type=float, default=1e-6)
    parser.add_argument("--hmc-rhat-threshold", type=float, default=1.05)
    parser.add_argument("--hmc-ess-threshold", type=float, default=100.0)
    parser.add_argument("--stage1-steps", type=int, default=300)
    parser.add_argument("--stage2-steps", type=int, default=500)
    parser.add_argument("--stage3-steps", type=int, default=250)
    parser.add_argument("--residual-weight", type=float, default=1e-3)
    parser.add_argument("--complexity-weight", type=float, default=2e-4)
    parser.add_argument("--binary-gate-weight", type=float, default=2e-4)
    parser.add_argument(
        "--derivative-loss-weight",
        type=float,
        default=0.05,
        help="Weight of the finite-difference derivative loss in symbolic-gated training.",
    )
    parser.add_argument(
        "--curve-shape-loss-weight",
        type=float,
        default=0.02,
        help="Weight of the photo-current curve mean/swing loss in symbolic-gated training.",
    )
    parser.add_argument("--max-symbolic-terms", type=int, default=12)
    parser.add_argument("--min-symbolic-probability", type=float, default=0.02)
    parser.add_argument("--bayesian-weight-floor-ratio", type=float, default=0.1)
    parser.add_argument("--bayesian-weight-min", type=float, default=0.2)
    parser.add_argument("--bayesian-weight-max", type=float, default=5.0)
    parser.add_argument("--bayesian-distill-weight", type=float, default=0.25)
    parser.add_argument(
        "--generic-symbolic-terms",
        action="store_true",
        help="Add generic math terms to the symbolic feature library for mixed-library experiments",
    )
    parser.add_argument(
        "--max-generic-symbolic-terms",
        type=int,
        default=4,
        help="Maximum generic residual terms retained when --generic-symbolic-terms is enabled",
    )
    parser.add_argument(
        "--generic-complexity-multiplier",
        type=float,
        default=5.0,
        help="Extra complexity penalty multiplier for generic residual gates",
    )
    parser.add_argument(
        "--flat-gates",
        action="store_true",
        help="Use flat single-pool gates (no mechanisms, no hierarchy) for all tasks.",
    )
    parser.add_argument(
        "--pure-symbolic-student",
        action="store_true",
        help=(
            "Exclude the KAN branch from symbolic-gated inference.  The KAN/BKAN may supply "
            "training targets, but the exported student contains only the selected symbolic terms."
        ),
    )
    parser.add_argument(
        "--ridge-init",
        action="store_true",
        help="Initialize gate coefficients from ridge regression fit.",
    )
    parser.add_argument(
        "--disable-photo-mechanisms",
        nargs="*",
        default=[],
        choices=[
            "photogeneration",
            "photogeneration_collection",
            "field_collection",
            "surface_recombination",
            "trap_recombination",
            "forward_injection",
        ],
        help="Photo-current symbolic mechanisms to remove for ablation experiments",
    )
    parser.add_argument(
        "--photo-coupling-mode",
        choices=["separate", "coupled"],
        default="separate",
        help="Photo-current mechanism layout: separate photogeneration/collection or one coupled branch",
    )
    parser.add_argument("--val-freq", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Generate optional Bayesian diagnostic plots, including uncertainty, OOD, and reliability.",
    )
    parser.add_argument(
        "--charge-mode",
        choices=["off", "validate", "train", "both"],
        default="off",
        help="Run integrated AC-CV/Q-V proxy validation and/or charge-proxy model training.",
    )
    parser.add_argument(
        "--charge-root",
        type=Path,
        default=DEFAULT_20260617_ROOT,
        help="Root directory containing supplemental_charge_ac.",
    )
    parser.add_argument(
        "--charge-output",
        type=Path,
        default=None,
        help="Output directory for charge-proxy model training; defaults under --output.",
    )
    parser.add_argument(
        "--charge-validation-output",
        type=Path,
        default=None,
        help="Output directory for charge proxy data validation; defaults under --output.",
    )
    parser.add_argument(
        "--charge-veriloga-output",
        type=Path,
        default=None,
        help="Generated charge-proxy Verilog-A path; defaults under --charge-output.",
    )
    parser.add_argument("--charge-degree", type=int, default=3)
    parser.add_argument("--charge-cvf-degree", type=int, default=3)
    parser.add_argument("--charge-dc-degree", type=int, default=3)
    parser.add_argument("--charge-alpha", type=float, default=1.0e-5)
    parser.add_argument(
        "--charge-dark-formula",
        type=Path,
        default=None,
        help="Optional dark-current symbolic formula path for charge-proxy Verilog-A export.",
    )
    parser.add_argument(
        "--charge-photo-formula",
        type=Path,
        default=None,
        help="Optional photo-current symbolic formula path for charge-proxy Verilog-A export.",
    )
    return parser.parse_args()


def _default_charge_parent(output: Path) -> Path:
    return output.parent if output.name == "device_modeling" else output


def _resolve_charge_paths(args) -> None:
    parent = _default_charge_parent(args.output)
    if args.charge_output is None:
        args.charge_output = parent / "charge_model"
    if args.charge_validation_output is None:
        args.charge_validation_output = parent / "charge_validation"
    if args.charge_veriloga_output is None:
        args.charge_veriloga_output = args.charge_output / "charge_proxy.va"
    if args.charge_dark_formula is None:
        args.charge_dark_formula = args.output / TASKS["dark_current"].result_subdir / "symbolic_formula.txt"
    if args.charge_photo_formula is None:
        args.charge_photo_formula = args.output / TASKS["photo_current"].result_subdir / "symbolic_formula.txt"


def run_integrated_charge_flows(args) -> dict[str, dict[str, Path]]:
    if args.charge_mode == "off":
        return {}

    from device_modeling.charge_proxy.train import run_charge_proxy_training
    from device_modeling.charge_proxy.validate import run_charge_proxy_validation

    _resolve_charge_paths(args)
    results: dict[str, dict[str, Path]] = {}

    if args.charge_mode in {"validate", "both"}:
        validation_args = argparse.Namespace(
            root=args.charge_root,
            output=args.charge_validation_output,
        )
        print(f"\nRunning charge proxy data validation: root={args.charge_root}")
        results["validation"] = run_charge_proxy_validation(validation_args)

    if args.charge_mode in {"train", "both"}:
        training_args = argparse.Namespace(
            root=args.charge_root,
            output=args.charge_output,
            veriloga_output=args.charge_veriloga_output,
            degree=args.charge_degree,
            cvf_degree=args.charge_cvf_degree,
            dc_degree=args.charge_dc_degree,
            alpha=args.charge_alpha,
            seed=args.seed,
            dark_formula=args.charge_dark_formula,
            photo_formula=args.charge_photo_formula,
        )
        print(f"\nRunning charge proxy model training: root={args.charge_root}")
        results["training"] = run_charge_proxy_training(training_args)

    return results


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    raw_cache = {}

    def load_raw_for_task(spec):
        if spec.key == "capacitance":
            if "capacitance" not in raw_cache:
                raw_cache["capacitance"] = load_capacitance_data(args.capacitance_data)
                print(
                    "Capacitance data: "
                    f"mode={raw_cache['capacitance'].attrs.get('data_source_mode')}, "
                    f"rows={len(raw_cache['capacitance'])}, "
                    f"source={raw_cache['capacitance'].attrs.get('capacitance_source_path')}"
                )
            return raw_cache["capacitance"]

        if "main" not in raw_cache:
            raw_cache["main"] = load_research_data(args.data)
            raw = raw_cache["main"]
            aggregation_report = raw.attrs.get("aggregation_report")
            if aggregation_report is not None:
                aggregation_report.to_csv(args.output / "data_aggregation_report.csv", index=False)
                raw.to_csv(args.output / "aggregated_data.csv", index=False)
                print(f"Data aggregation: mode={raw.attrs.get('data_source_mode', 'unknown')}, rows={len(raw)}")
            cleaning_report = raw.attrs.get("cleaning_report")
            if cleaning_report is not None:
                cleaning_report.to_csv(args.output / "data_cleaning_report.csv", index=False)
                raw.to_csv(args.output / "cleaned_data.csv", index=False)
                retained = int((cleaning_report["status"] == "retained").sum())
                removed = int((cleaning_report["status"] == "removed").sum())
                print(f"Data cleaning: retained={retained} condition curves, removed={removed}")
            low_current_report = raw.attrs.get("low_current_cleaning_report")
            if low_current_report is not None:
                low_current_report.to_csv(args.output / "low_current_cleaning_report.csv", index=False)
                raw.to_csv(args.output / "cleaned_data.csv", index=False)
                print(low_current_report.to_string(index=False))
            photocurrent_audit = raw.attrs.get("photocurrent_target_audit")
            if photocurrent_audit is not None:
                payload = {
                    "definition": raw.attrs.get("photocurrent_target_definition"),
                    **photocurrent_audit,
                }
                (args.output / "photocurrent_target_audit.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
                print(f"Photocurrent target audit: {payload}")
            if raw.attrs.get("ac_target_semantics") is not None:
                ac_payload = {
                    "definition": raw.attrs.get("ac_target_semantics"),
                    "perturbation": raw.attrs.get("ac_perturbation"),
                    "observable": raw.attrs.get("ac_observable"),
                    "electrical_port_perturbation": False,
                    "terminal_dynamic_branch": False,
                }
                (args.output / "ac_target_audit.json").write_text(
                    json.dumps(ac_payload, indent=2), encoding="utf-8"
                )
                print(f"AC target audit: {ac_payload}")
        return raw_cache["main"]

    rows = []
    for task_key in args.task:
        spec = TASKS[task_key]
        rows.extend(run_task(spec, load_raw_for_task(spec), args, device))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nSummary saved to: {args.output / 'summary.csv'}")
    charge_results = run_integrated_charge_flows(args)
    with (args.output / "run_config.json").open("w", encoding="utf-8") as handle:
        config = {key: str(value) for key, value in vars(args).items()}
        config["charge_results"] = {
            name: {key: str(value) for key, value in paths.items()}
            for name, paths in charge_results.items()
        }
        json.dump(config, handle, indent=2)


if __name__ == "__main__":
    main()
