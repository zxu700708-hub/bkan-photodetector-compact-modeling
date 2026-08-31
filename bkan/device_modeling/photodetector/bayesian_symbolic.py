"""Symbolic extraction and formula verification for Bayesian KAN models."""

from __future__ import annotations

import os
import re

import numpy as np
import sympy
import torch


PHYSICS_INFORMED_SYMBOLIC_STAGES = [
    {
        "name": "Level 1 - safe polynomial probe",
        "lib": ["0", "x", "x^2", "sqrt"],
        "r2_threshold": 0.92,
        "note": "Start with simple, relatively stable semiconductor-friendly trends.",
    },
    {
        "name": "Level 2 - core exponential mechanism",
        "lib": ["exp", "1/x"],
        "r2_threshold": 0.96,
        "note": "Admit exponential and reciprocal forms only after stronger evidence.",
    },
    {
        "name": "Level 3 - advanced correction",
        "lib": ["log", "x^3", "abs"],
        "r2_threshold": 0.98,
        "note": "Capture secondary curvature, compression, and magnitude effects.",
    },
    {
        "name": "Level 4 - mathematical fallback",
        "lib": ["sin", "cos", "tanh", "gaussian"],
        "r2_threshold": 0.998,
        "note": "Use flexible patches only when the fit is almost exact.",
    },
]

AC_RESPONSE_SYMBOLIC_STAGES = [
    {
        "name": "AC Level 1 - monotone baseline",
        "lib": ["0", "x", "x^2"],
        "r2_threshold": 0.90,
        "note": "Start with smooth polynomial trends and avoid singular functions.",
    },
    {
        "name": "AC Level 2 - roll-off curvature",
        "lib": ["exp", "log", "tanh", "arctan"],
        "r2_threshold": 0.94,
        "note": "Prefer bounded or smooth roll-off functions for frequency response.",
    },
    {
        "name": "AC Level 3 - gentle correction",
        "lib": ["x^3", "gaussian"],
        "r2_threshold": 0.985,
        "note": "Allow secondary curvature only after a strong local fit.",
    },
]

PHYSICS_INFORMED_SYMBOLIC_PROS = [
    "Improves interpretability by trying low-complexity physical forms before flexible patches.",
    "Reduces noise fitting by requiring higher R2 for exponential, reciprocal, and fallback functions.",
    "Matches common semiconductor behavior such as polynomial corrections and thermally activated exponential trends.",
]

PHYSICS_INFORMED_SYMBOLIC_CONS = [
    "A correct but subtle exponential/log mechanism can be missed if the local edge fit does not pass its threshold.",
    "Stage order is a prior; it can bias formula selection when multiple functions have similar R2.",
    "The formula represents the deterministic posterior-mean KAN, not a full posterior over symbolic structures.",
]


def _is_ac_response(modeler):
    name = str(modeler.task_name).lower().replace("_", "-")
    return (
        name in {"ac-response", "ac-response-db", "bandwidth"}
        or "frequency_ghz" in (modeler.input_cols or [])
    )


def _make_symbol_name(name, index):
    symbol = re.sub(r"\W+", "_", str(name)).strip("_")
    if not symbol:
        symbol = f"x_{index + 1}"
    if symbol[0].isdigit():
        symbol = f"x_{symbol}"
    return symbol


def _model_space_label(modeler):
    return f"Model-space {modeler.output_col}"


def _predict_model_space(modeler, X):
    pred = modeler.predict_with_uncertainty(np.asarray(X, dtype=np.float32))
    missing = [key for key in ("model_mean", "model_lower", "model_upper") if key not in pred]
    if missing:
        raise KeyError("Prediction result missing keys: " + ", ".join(missing))
    return pred


def _fit_ac_response_compact_formula(modeler, verbose=True):
    """Fit a low-pass compact expression to the Bayesian posterior mean."""

    if not modeler.input_cols or "frequency_ghz" not in modeler.input_cols:
        return None

    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        if verbose:
            print(f"[AC Compact] scipy.optimize unavailable, skipping: {exc}")
        return None

    freq_col = "frequency_ghz"
    condition_cols = [
        col
        for col in ("active_layer_length", "simulation_temperature", "bandwidth_bias")
        if col in modeler.input_cols
    ]

    X_raw = modeler.scaler_x.inverse_transform(modeler.X_train.detach().cpu().numpy())
    y_target = _predict_model_space(modeler, X_raw)["model_mean"].reshape(-1)

    f_idx = modeler.input_cols.index(freq_col)
    frequency = np.maximum(X_raw[:, f_idx].astype(np.float64), 1e-9)

    z_cols = []
    z_stats = {}
    for col in condition_cols:
        idx = modeler.input_cols.index(col)
        raw = X_raw[:, idx].astype(np.float64)
        mean = float(np.mean(raw))
        std = float(np.std(raw))
        if std <= 0.0 or not np.isfinite(std):
            std = 1.0
        z_cols.append((col, (raw - mean) / std))
        z_stats[col] = (mean, std)

    n_cond = len(z_cols)

    def unpack(theta):
        pos = 0
        y0_coef = theta[pos:pos + 1 + n_cond]
        pos += 1 + n_cond
        floor_coef = theta[pos:pos + 1 + n_cond]
        pos += 1 + n_cond
        logfc_coef = theta[pos:pos + 1 + n_cond]
        pos += 1 + n_cond
        log_p = theta[pos]
        return y0_coef, floor_coef, logfc_coef, log_p

    def affine(coef):
        out = np.full_like(frequency, coef[0], dtype=np.float64)
        for k, (_, z_value) in enumerate(z_cols):
            out = out + coef[k + 1] * z_value
        return out

    def predict(theta):
        y0_coef, floor_coef, logfc_coef, log_p = unpack(theta)
        y0 = affine(y0_coef)
        y_floor = affine(floor_coef)
        fc = np.exp(np.clip(affine(logfc_coef), np.log(0.02), np.log(500.0)))
        p = 0.5 + np.exp(np.clip(log_p, np.log(0.05), np.log(20.0)))
        ratio = np.power(np.maximum(frequency / fc, 1e-12), p)
        return y_floor + (y0 - y_floor) / (1.0 + ratio)

    y_hi = float(np.nanpercentile(y_target, 95))
    y_lo = float(np.nanpercentile(y_target, 5))
    theta0 = np.zeros(3 * (1 + n_cond) + 1, dtype=np.float64)
    theta0[0] = y_hi
    theta0[1 + n_cond] = y_lo
    theta0[2 * (1 + n_cond)] = np.log(6.0)
    theta0[-1] = np.log(1.5)

    def residual(theta):
        res = predict(theta) - y_target
        res[~np.isfinite(res)] = 1e6
        return res

    result = least_squares(residual, theta0, loss="soft_l1", f_scale=0.08, max_nfev=8000)
    theta = result.x
    pred = predict(theta)
    valid = np.isfinite(pred) & np.isfinite(y_target)
    rmse = float(np.sqrt(np.mean((pred[valid] - y_target[valid]) ** 2))) if np.any(valid) else np.nan
    mae = float(np.mean(np.abs(pred[valid] - y_target[valid]))) if np.any(valid) else np.nan

    symbols = {
        col: sympy.Symbol(_make_symbol_name(col, index))
        for index, col in enumerate(modeler.input_cols)
    }
    f_sym = symbols[freq_col]
    y0_coef, floor_coef, logfc_coef, log_p = unpack(theta)

    def affine_sym(coef):
        expr = sympy.Float(float(coef[0]))
        for k, (col, _) in enumerate(z_cols):
            mean, std = z_stats[col]
            expr += sympy.Float(float(coef[k + 1])) * (
                (symbols[col] - sympy.Float(mean)) / sympy.Float(std)
            )
        return expr

    y0_expr = affine_sym(y0_coef)
    floor_expr = affine_sym(floor_coef)
    fc_expr = sympy.exp(affine_sym(logfc_coef))
    p_expr = sympy.Float(0.5 + np.exp(np.clip(log_p, np.log(0.05), np.log(20.0))))
    formula = floor_expr + (y0_expr - floor_expr) / (1 + (f_sym / fc_expr) ** p_expr)

    modeler._model_formula = formula
    modeler._linear_formula = sympy.Pow(10, formula, evaluate=False) if modeler.use_log_transform else formula
    modeler._readable_symbols = [symbols[col] for col in modeler.input_cols]
    modeler._variable_map = {str(symbols[col]): col for col in modeler.input_cols}
    modeler._ac_compact_formula_info = {
        "rmse": rmse,
        "mae": mae,
        "success": bool(result.success),
        "message": str(result.message),
        "condition_cols": condition_cols,
    }

    save_path = os.path.join(modeler.save_dir, "ac_response_compact_formula.txt")
    with open(save_path, "w", encoding="utf-8") as handle:
        handle.write(f"AC response compact formula [{modeler.task_name}]\n")
        handle.write("=" * 44 + "\n\n")
        handle.write("Formula family:\n")
        handle.write("  y = y_floor + (y0 - y_floor) / (1 + (frequency_ghz / fc)^p)\n\n")
        handle.write(f"Fit target: Bayesian posterior mean in model space for {modeler.output_col}\n")
        handle.write(f"RMSE={rmse:.6g}, MAE={mae:.6g}, optimizer_success={result.success}\n")
        handle.write(f"Condition variables: {condition_cols if condition_cols else 'none'}\n\n")
        handle.write(f"{modeler.output_col}_model_space = {formula}\n")

    if verbose:
        print(f"[AC Compact] low-pass compact fit: RMSE={rmse:.4f}, MAE={mae:.4f}")
        print(f"[AC Compact] saved: {save_path}")

    return formula


def extract_symbolic_formula(
    modeler,
    weight_simple=0.8,
    r2_threshold=0.0,
    verbose=1,
    simplify=True,
    prune_node_th=0.01,
    prune_edge_th=0.03,
):
    """Export the Bayesian posterior mean to a deterministic symbolic KAN."""

    if modeler.model is None or modeler.X_train is None:
        raise RuntimeError("Model has not been trained yet.")

    print("\n[Symbolic] Exporting Bayesian posterior mean to deterministic KAN...")
    symbolic_model = modeler.model.to_deterministic_kan(
        device=modeler.device,
        symbolic_enabled=True,
        auto_save=False,
    )
    os.makedirs(getattr(symbolic_model, "save_folder", "./model"), exist_ok=True)

    with torch.no_grad():
        symbolic_model(modeler.X_train)

    is_ac = _is_ac_response(modeler)
    symbolic_stages = AC_RESPONSE_SYMBOLIC_STAGES if is_ac else PHYSICS_INFORMED_SYMBOLIC_STAGES
    extraction_mode = "ac_response_safe" if is_ac else "physics_informed_safe"

    try:
        symbolic_model = symbolic_model.prune(node_th=prune_node_th, edge_th=prune_edge_th)
        for stage in symbolic_stages:
            threshold = max(stage["r2_threshold"], r2_threshold)
            if verbose:
                print(f" -> {stage['name']}: lib={stage['lib']}, r2_threshold={threshold}")
            symbolic_model.auto_symbolic(
                lib=stage["lib"],
                r2_threshold=threshold,
                verbose=verbose,
                weight_simple=weight_simple,
            )
    except Exception as exc:
        extraction_mode = "identity_fallback"
        if verbose:
            print(f"[Symbolic] stepwise extraction failed, using identity fallback: {exc}")
        for layer_idx in range(len(symbolic_model.width_in) - 1):
            for input_idx in range(symbolic_model.width_in[layer_idx]):
                for output_idx in range(symbolic_model.width_out[layer_idx + 1]):
                    mask = float(symbolic_model.act_fun[layer_idx].mask[input_idx][output_idx].detach().cpu())
                    if mask == 0.0:
                        symbolic_model.fix_symbolic(
                            layer_idx, input_idx, output_idx, "0",
                            fit_params_bool=False, verbose=False, log_history=False,
                        )
                        continue
                    try:
                        symbolic_model.fix_symbolic(
                            layer_idx, input_idx, output_idx, "x",
                            fit_params_bool=True, verbose=False, log_history=False,
                        )
                    except Exception:
                        symbolic_model.fix_symbolic(
                            layer_idx, input_idx, output_idx, "0",
                            fit_params_bool=False, verbose=False, log_history=False,
                        )

    readable_symbols = [
        sympy.Symbol(_make_symbol_name(col, index))
        for index, col in enumerate(modeler.input_cols)
    ]
    variable_map = {str(symbol): col for symbol, col in zip(readable_symbols, modeler.input_cols)}

    formulas, _ = symbolic_model.symbolic_formula(
        var=readable_symbols,
        normalizer=[modeler.data_stats["X_mean"], modeler.data_stats["X_std"]],
        output_normalizer=([modeler.data_stats["y_mean"]], [modeler.data_stats["y_std"]]),
        simplify=simplify,
    )

    model_formula = formulas[0]
    linear_formula = sympy.Pow(10, model_formula, evaluate=False) if modeler.use_log_transform else model_formula
    free_variables = sorted([str(variable) for variable in model_formula.free_symbols])

    save_path = os.path.join(modeler.save_dir, "symbolic_formula.txt")
    with open(save_path, "w", encoding="utf-8") as handle:
        handle.write(f"Bayesian Symbolic KAN formula extraction [{modeler.task_name}]\n")
        handle.write("=" * 48 + "\n\n")
        handle.write(f"Extraction mode: {extraction_mode}\n\n")
        handle.write("Physics-informed safe stepwise strategy:\n")
        for stage in symbolic_stages:
            threshold = max(stage["r2_threshold"], r2_threshold)
            handle.write(f"  {stage['name']}: lib={stage['lib']}, r2_threshold={threshold}\n")
            handle.write(f"    {stage['note']}\n")
        handle.write("\nAdvantages:\n")
        for item in PHYSICS_INFORMED_SYMBOLIC_PROS:
            handle.write(f"  - {item}\n")
        handle.write("\nLimitations:\n")
        for item in PHYSICS_INFORMED_SYMBOLIC_CONS:
            handle.write(f"  - {item}\n")
        handle.write("\nVariable map:\n")
        for symbol, column in variable_map.items():
            handle.write(f"  {symbol}: {column}\n")
        handle.write("\nFree variables in formula:\n")
        handle.write(f"  {', '.join(free_variables) if free_variables else 'none'}\n")
        handle.write(f"\nFormula in model space:\n  {model_formula}\n\n")
        handle.write(f"Formula in physical space:\n  {modeler.output_col} = {linear_formula}\n")

    modeler.symbolic_model = symbolic_model
    modeler._model_formula = model_formula
    modeler._log_formula = model_formula
    modeler._linear_formula = linear_formula
    modeler._readable_symbols = readable_symbols
    modeler._variable_map = variable_map

    if is_ac:
        ac_formula = _fit_ac_response_compact_formula(modeler, verbose=verbose > 0)
        if ac_formula is not None:
            model_formula = ac_formula
            linear_formula = sympy.Pow(10, model_formula, evaluate=False) if modeler.use_log_transform else model_formula
            free_variables = sorted([str(variable) for variable in model_formula.free_symbols])
            extraction_mode += " + low_pass_compact"
            with open(save_path, "w", encoding="utf-8") as handle:
                handle.write(f"AC Response Compact Formula extraction [{modeler.task_name}] *LOW-PASS*\n")
                handle.write("=" * 56 + "\n\n")
                handle.write(f"Extraction mode: {extraction_mode}\n\n")
                handle.write("AC-specific strategy:\n")
                handle.write("  1. Avoid free reciprocal terms in KAN symbolic extraction.\n")
                handle.write("  2. Fit a monotone low-pass compact form to the Bayesian posterior mean.\n")
                handle.write("  3. Let y0, y_floor, and fc vary affinely with physical condition variables.\n\n")
                if hasattr(modeler, "_ac_compact_formula_info"):
                    info = modeler._ac_compact_formula_info
                    handle.write(f"Compact fit RMSE={info['rmse']:.6g}, MAE={info['mae']:.6g}\n")
                    handle.write(f"Optimizer success: {info['success']} ({info['message']})\n")
                    handle.write(f"Condition variables: {info['condition_cols']}\n\n")
                handle.write("Variable map:\n")
                for symbol, column in modeler._variable_map.items():
                    handle.write(f"  {symbol}: {column}\n")
                handle.write("\nFree variables in formula:\n")
                handle.write(f"  {', '.join(free_variables) if free_variables else 'none'}\n")
                handle.write(f"\nFormula in model space:\n  {model_formula}\n\n")
                handle.write(f"Formula in physical space:\n  {modeler.output_col} = {linear_formula}\n")

    return {
        "model": symbolic_model,
        "model_formula": model_formula,
        "log_formula": model_formula,
        "linear_formula": linear_formula,
        "extraction_mode": extraction_mode,
        "strategy_stages": symbolic_stages,
        "strategy_pros": PHYSICS_INFORMED_SYMBOLIC_PROS,
        "strategy_cons": PHYSICS_INFORMED_SYMBOLIC_CONS,
    }


def _formula_function(modeler):
    from sympy import lambdify

    if not hasattr(modeler, "_model_formula") or not hasattr(modeler, "_readable_symbols"):
        raise RuntimeError("No symbolic formula found. Run extract_symbolic_formula first.")

    formula = modeler._model_formula
    readable_symbols = modeler._readable_symbols
    free_symbols = sorted(formula.free_symbols, key=str)
    sym_to_col = {symbol: col for symbol, col in zip(readable_symbols, modeler.input_cols)}
    formula_fn = lambdify(free_symbols, formula, modules="numpy")
    return formula, free_symbols, sym_to_col, formula_fn


def _eval_formula_rows(modeler, X_rows, free_symbols, sym_to_col, formula_fn):
    values = np.full(len(X_rows), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(X_rows):
        args = []
        for symbol in free_symbols:
            col = sym_to_col.get(symbol)
            if col is None:
                args.append(np.nan)
            else:
                args.append(float(row[modeler.input_cols.index(col)]))
        try:
            value = float(formula_fn(*args))
            values[row_idx] = value if np.isfinite(value) else np.nan
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError, TypeError):
            values[row_idx] = np.nan
    return values


def plot_formula_verification(modeler, df_test=None, voltage_col=None, df_curves=None):
    """Verify symbolic formula against Bayesian posterior mean in model space."""

    if df_test is None or df_test.empty:
        print("[Formula Verification] Missing test data, skipping.")
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    _, free_symbols, sym_to_col, formula_fn = _formula_function(modeler)
    result_dir = modeler.save_dir
    inputs = modeler.input_cols
    target_label = _model_space_label(modeler)
    generated = []

    X_test = df_test[inputs].to_numpy(dtype=np.float64)
    pred = _predict_model_space(modeler, X_test)
    bnn_mean = pred["model_mean"]
    formula_values = _eval_formula_rows(modeler, X_test, free_symbols, sym_to_col, formula_fn)
    target_values = modeler._transform_target(df_test[modeler.output_col].to_numpy(dtype=np.float64)).reshape(-1)

    pointwise = df_test[inputs + [modeler.output_col]].copy().reset_index(drop=True)
    pointwise["tcad_model_space"] = target_values
    pointwise["bnn_model_space"] = bnn_mean
    pointwise["formula_model_space"] = formula_values
    pointwise["formula_minus_bnn"] = formula_values - bnn_mean
    pointwise["bnn_minus_tcad"] = bnn_mean - target_values
    pointwise["formula_minus_tcad"] = formula_values - target_values
    pointwise["formula_is_finite"] = np.isfinite(formula_values)
    pointwise_path = os.path.join(result_dir, "Formula_Pointwise_Verification.csv")
    pointwise.to_csv(pointwise_path, index=False)
    generated.append(pointwise_path)

    valid = np.isfinite(formula_values) & np.isfinite(bnn_mean)
    fig, ax = plt.subplots(figsize=(8, 8))
    if np.any(valid):
        x_valid = bnn_mean[valid]
        y_valid = formula_values[valid]
        lo = float(min(np.min(x_valid), np.min(y_valid)))
        hi = float(max(np.max(x_valid), np.max(y_valid)))
        rmse = float(np.sqrt(np.mean((y_valid - x_valid) ** 2)))
        mae = float(np.mean(np.abs(y_valid - x_valid)))
        ax.scatter(x_valid, y_valid, alpha=0.4, s=16, color="#4c78a8", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="y = x")
    else:
        rmse = np.nan
        mae = np.nan
        ax.text(0.5, 0.5, "No finite formula values", transform=ax.transAxes, ha="center", va="center")
    nan_rate = 1.0 - float(np.sum(valid)) / max(len(formula_values), 1)
    ax.set_xlabel(f"Bayesian posterior mean ({target_label})", fontweight="bold")
    ax.set_ylabel(f"Symbolic formula ({target_label})", fontweight="bold")
    ax.set_title(
        f"[{modeler.task_name}] Formula vs Bayesian KAN\nRMSE={rmse:.4g}  MAE={mae:.4g}  NaN={nan_rate:.1%}",
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    scatter_path = os.path.join(result_dir, "Formula_Verification_Scatter.png")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(scatter_path)

    if voltage_col is None or voltage_col not in inputs:
        voltage_col = inputs[0]
    v_idx = inputs.index(voltage_col)
    df_all = df_test.copy()
    if df_curves is not None and not df_curves.empty:
        df_curve_source = df_curves.copy()
    else:
        df_curve_source = df_test.copy()

    for sweep_var in inputs:
        if sweep_var == voltage_col:
            continue
        sweep_values = df_all[sweep_var].to_numpy(dtype=np.float64)
        if len(sweep_values) < 2 or float(np.nanmax(sweep_values)) <= float(np.nanmin(sweep_values)):
            continue
        sweep_min = float(np.nanpercentile(sweep_values, 1))
        sweep_max = float(np.nanpercentile(sweep_values, 99))
        if sweep_max <= sweep_min:
            sweep_min = float(np.nanmin(sweep_values))
            sweep_max = float(np.nanmax(sweep_values))

        voltage_values = df_all[voltage_col].to_numpy(dtype=np.float64)
        v_levels = [float(np.nanquantile(voltage_values, q)) for q in (0.1, 0.5, 0.9)]
        v_labels = ["Low Bias", "Mid Bias", "High Bias"]
        v_colors = plt.cm.viridis(np.linspace(0.1, 0.9, 3))
        fixed = {
            col: float(df_all[col].median())
            for col in inputs
            if col not in (voltage_col, sweep_var)
        }
        sweep_idx = inputs.index(sweep_var)
        x_sweep = np.linspace(sweep_min, sweep_max, 300, dtype=np.float64)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for axis, v_value, v_label, color in zip(axes, v_levels, v_labels, v_colors):
            X_infer = np.zeros((len(x_sweep), len(inputs)), dtype=np.float32)
            for col_idx, col_name in enumerate(inputs):
                if col_idx == sweep_idx:
                    X_infer[:, col_idx] = x_sweep.astype(np.float32)
                elif col_idx == v_idx:
                    X_infer[:, col_idx] = v_value
                else:
                    X_infer[:, col_idx] = fixed[col_name]
            bnn = _predict_model_space(modeler, X_infer)
            formula_curve = _eval_formula_rows(modeler, X_infer, free_symbols, sym_to_col, formula_fn)
            valid_curve = np.isfinite(formula_curve) & np.isfinite(bnn["model_mean"])
            axis.plot(x_sweep, bnn["model_mean"], color=color, linewidth=2.4, label="Bayesian mean")
            axis.fill_between(x_sweep, bnn["model_lower"], bnn["model_upper"], color=color, alpha=0.10)
            if np.any(valid_curve):
                axis.plot(x_sweep[valid_curve], formula_curve[valid_curve], color="#e45756", linewidth=1.8, linestyle="--", label="Formula")
            if np.sum(valid_curve) > 5:
                local_rmse = float(np.sqrt(np.mean((formula_curve[valid_curve] - bnn["model_mean"][valid_curve]) ** 2)))
                status = f"RMSE={local_rmse:.3g}"
            else:
                status = "insufficient finite values"
            axis.set_xlabel(sweep_var, fontweight="bold")
            axis.set_ylabel(target_label, fontweight="bold")
            axis.set_title(f"{v_label} {voltage_col}={v_value:.3g} [{status}]", fontsize=10, fontweight="bold")
            axis.grid(True, linestyle="--", alpha=0.35)
            axis.legend(loc="best", fontsize=7)
        fig.suptitle(f"[{modeler.task_name}] Formula vs Bayesian KAN: Sweep {sweep_var}", fontweight="bold")
        fig.tight_layout()
        sweep_path = os.path.join(result_dir, f"Formula_Verification_Sweep_{sweep_var}.png")
        fig.savefig(sweep_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        generated.append(sweep_path)

    condition_cols = [col for col in inputs if col != voltage_col]
    df_curve_source = df_curve_source.dropna(subset=inputs + [modeler.output_col]).copy()
    real_curves = []
    if condition_cols:
        grouped = df_curve_source.groupby(condition_cols, sort=True, dropna=True)
        for condition_values, group in grouped:
            condition_tuple = condition_values if isinstance(condition_values, tuple) else (condition_values,)
            group = group.sort_values(voltage_col).copy()
            v_values = group[voltage_col].to_numpy(dtype=np.float64)
            v_values = v_values[np.isfinite(v_values)]
            if len(np.unique(v_values)) >= 4:
                real_curves.append((len(np.unique(v_values)), float(np.nanmax(v_values) - np.nanmin(v_values)), condition_tuple, group))
    else:
        group = df_curve_source.sort_values(voltage_col).copy()
        v_values = group[voltage_col].to_numpy(dtype=np.float64)
        if len(np.unique(v_values[np.isfinite(v_values)])) >= 4:
            real_curves.append((len(np.unique(v_values)), float(np.nanmax(v_values) - np.nanmin(v_values)), tuple(), group))

    real_curves = sorted(real_curves, key=lambda item: (item[0], item[1]), reverse=True)[:6]
    if real_curves:
        cols = min(2, len(real_curves))
        rows = int(np.ceil(len(real_curves) / cols))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(real_curves)))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5), squeeze=False)
        axes = axes.flatten()
        summary_rows = []
        for curve_idx, ((_, _, condition_tuple, group), color) in enumerate(zip(real_curves, colors)):
            axis = axes[curve_idx]
            condition = dict(zip(condition_cols, condition_tuple))
            v_values = group[voltage_col].to_numpy(dtype=np.float64)
            v_sweep = np.linspace(float(np.nanmin(v_values)), float(np.nanmax(v_values)), 400)
            X_infer = np.zeros((len(v_sweep), len(inputs)), dtype=np.float32)
            for col_idx, col_name in enumerate(inputs):
                if col_name == voltage_col:
                    X_infer[:, col_idx] = v_sweep.astype(np.float32)
                else:
                    X_infer[:, col_idx] = float(condition[col_name])
            bnn = _predict_model_space(modeler, X_infer)
            formula_curve = _eval_formula_rows(modeler, X_infer, free_symbols, sym_to_col, formula_fn)
            valid_curve = np.isfinite(formula_curve) & np.isfinite(bnn["model_mean"])
            if np.sum(valid_curve) > 5:
                curve_rmse = float(np.sqrt(np.mean((formula_curve[valid_curve] - bnn["model_mean"][valid_curve]) ** 2)))
                curve_mae = float(np.mean(np.abs(formula_curve[valid_curve] - bnn["model_mean"][valid_curve])))
            else:
                curve_rmse = np.nan
                curve_mae = np.nan
            y_tcad = modeler._transform_target(group[modeler.output_col].to_numpy(dtype=np.float64)).reshape(-1)
            axis.plot(v_sweep, bnn["model_mean"], color=color, linewidth=2.4, label="Bayesian mean")
            axis.fill_between(v_sweep, bnn["model_lower"], bnn["model_upper"], color=color, alpha=0.10)
            if np.any(valid_curve):
                axis.plot(v_sweep[valid_curve], formula_curve[valid_curve], color="#e45756", linewidth=1.8, linestyle="--", label="Formula")
            axis.scatter(group[voltage_col], y_tcad, s=18, alpha=0.65, color=color, edgecolors="none", label=f"TCAD (n={len(group)})")
            cond_text = ", ".join(f"{col}={float(value):.4g}" for col, value in condition.items()) if condition else "all data"
            axis.text(0.02, 0.02, cond_text, transform=axis.transAxes, fontsize=7, ha="left", va="bottom", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
            axis.set_xlabel(voltage_col, fontweight="bold")
            axis.set_ylabel(target_label, fontweight="bold")
            axis.set_title(f"Real condition #{curve_idx + 1}\nFormula-BNN RMSE={curve_rmse:.3g}", fontsize=10, fontweight="bold")
            axis.grid(True, linestyle="--", alpha=0.35)
            axis.legend(loc="best", fontsize=7)
            row = {
                "curve_index": curve_idx + 1,
                "n_points": len(group),
                "formula_bnn_rmse_sweep": curve_rmse,
                "formula_bnn_mae_sweep": curve_mae,
                "formula_valid_count_sweep": int(np.sum(valid_curve)),
                "formula_nan_rate_sweep": float(1.0 - np.sum(valid_curve) / max(len(valid_curve), 1)),
            }
            row.update(condition)
            summary_rows.append(row)
        for empty_idx in range(len(real_curves), len(axes)):
            fig.delaxes(axes[empty_idx])
        fig.suptitle(f"[{modeler.task_name}] Formula Verification: Real Conditions", fontweight="bold")
        fig.tight_layout()
        iv_path = os.path.join(result_dir, "Formula_Verification_IV_Real_Conditions.png")
        fig.savefig(iv_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        generated.append(iv_path)
        summary_path = os.path.join(result_dir, "Formula_IV_Real_Condition_Summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        generated.append(summary_path)

    print(f"[Formula Verification] Saved {len(generated)} artifacts to {result_dir}")
    return generated
