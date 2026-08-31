"""Model-space diagnostics for Bayesian KAN device models."""

from __future__ import annotations

import os

import numpy as np


def _frame_inputs(modeler, frame_or_array):
    if hasattr(frame_or_array, "loc"):
        return frame_or_array[modeler.input_cols].to_numpy(dtype=np.float32)
    return np.asarray(frame_or_array, dtype=np.float32)


def _frame_target_model_space(modeler, frame):
    y_raw = frame[modeler.output_col].to_numpy(dtype=np.float32)
    return modeler._transform_target(y_raw).reshape(-1)


def _require_prediction_keys(prediction, keys):
    missing = [key for key in keys if key not in prediction]
    if missing:
        raise KeyError(
            "Bayesian prediction is missing model-space diagnostics keys: "
            + ", ".join(missing)
        )


def plot_loss_curve(modeler):
    """Save the Bayesian training loss curve when VI history is available."""

    if modeler.experiment is None or not modeler.experiment.history.get("train_loss"):
        raise ValueError("No training history found for loss-curve plotting")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = modeler.experiment.history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    skip = 5 if len(history["train_loss"]) > 15 else 0
    epochs = range(skip, len(history["train_loss"]))
    val_freq = getattr(modeler.experiment, "val_freq", 10) or 10
    val_epochs = range(skip, len(history["train_loss"]), val_freq)
    val_slice = skip // val_freq

    def validation_points(values):
        y_values = values[val_slice:]
        x_values = list(val_epochs)[val_slice:val_slice + len(y_values)]
        if len(x_values) != len(y_values):
            stop = max(skip, len(history["train_loss"]) - 1)
            x_values = np.linspace(skip, stop, len(y_values)) if y_values else []
        return x_values, y_values

    axes[0, 0].plot(epochs, history["train_loss"][skip:], label="Train ELBO", color="#2980b9", linewidth=2)
    if history.get("val_loss"):
        val_x, val_y = validation_points(history["val_loss"])
        axes[0, 0].plot(
            val_x,
            val_y,
            label="Val ELBO",
            color="#e67e22",
            linewidth=2,
            linestyle="--",
        )
    axes[0, 0].set_title("Total ELBO Loss", fontsize=12, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    if history.get("train_nll"):
        axes[0, 1].plot(epochs, history["train_nll"][skip:], label="Train NLL", color="#27ae60", linewidth=2)
    axes[0, 1].set_title("Negative Log Likelihood", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    if history.get("train_kl"):
        axes[1, 0].plot(epochs, history["train_kl"][skip:], label="Train KL", color="#8e44ad", linewidth=2)
    axes[1, 0].set_title("KL Divergence", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    if history.get("val_mse"):
        val_x, val_y = validation_points(history["val_mse"])
        axes[1, 1].plot(
            val_x,
            val_y,
            label="Val MSE",
            color="#c0392b",
            linewidth=2,
        )
        axes[1, 1].legend()
    axes[1, 1].set_title("Validation MSE", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"[{modeler.task_name}] Bayesian KAN Training History", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(modeler.save_dir, "Training_Loss_Curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def detect_ood(modeler, X_test, X_train=None, threshold=None, percentile=95.0):
    """Flag likely out-of-distribution points using model-space predictive std."""

    X_test_np = _frame_inputs(modeler, X_test)
    test_pred = modeler.predict_with_uncertainty(X_test_np)
    _require_prediction_keys(test_pred, ("model_std",))
    test_score = np.asarray(test_pred["model_std"], dtype=np.float64).reshape(-1)

    train_score = None
    if X_train is not None:
        X_train_np = _frame_inputs(modeler, X_train)
        train_pred = modeler.predict_with_uncertainty(X_train_np)
        _require_prediction_keys(train_pred, ("model_std",))
        train_score = np.asarray(train_pred["model_std"], dtype=np.float64).reshape(-1)
        train_score = train_score[np.isfinite(train_score)]
        if threshold is None and train_score.size:
            threshold = float(np.percentile(train_score, percentile))

    finite_test = test_score[np.isfinite(test_score)]
    if threshold is None:
        if finite_test.size == 0:
            threshold = float("inf")
        else:
            threshold = float(np.mean(finite_test) + 2.0 * np.std(finite_test))

    ood_flags = test_score > threshold
    return {
        "uncertainty_score": test_score,
        "model_std": test_score,
        "train_uncertainty_score": train_score,
        "threshold": threshold,
        "ood_flags": ood_flags,
        "ood_ratio": float(np.mean(ood_flags)) if len(ood_flags) else 0.0,
    }


def plot_uncertainty_analysis(modeler, df_raw):
    """Save residual and uncertainty diagnostics in target/model space."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = _frame_inputs(modeler, df_raw)
    y_model = _frame_target_model_space(modeler, df_raw)
    valid_mask = (y_model > modeler.y_bounds[0]) & (y_model < modeler.y_bounds[1])
    X = X[valid_mask]
    y_model = y_model[valid_mask]

    pred = modeler.predict_with_uncertainty(X)
    _require_prediction_keys(pred, ("model_mean", "model_std"))
    y_mean = np.asarray(pred["model_mean"], dtype=np.float64).reshape(-1)
    y_std = np.asarray(pred["model_std"], dtype=np.float64).reshape(-1)
    epistemic = np.asarray(pred.get("epistemic", np.zeros_like(y_std)), dtype=np.float64).reshape(-1)
    aleatoric = np.asarray(pred.get("aleatoric", np.zeros_like(y_std)), dtype=np.float64).reshape(-1)
    residuals = y_model - y_mean
    abs_residuals = np.abs(residuals)

    x_axis_idx = next((i for i, col in enumerate(modeler.input_cols) if "V" in col.upper()), 0)
    x_axis_name = modeler.input_cols[x_axis_idx]
    x_axis = X[:, x_axis_idx]
    if len(x_axis) == 0:
        raise ValueError("No valid rows available for uncertainty diagnostics")

    if float(np.nanmax(x_axis)) > float(np.nanmin(x_axis)):
        n_bins = min(30, max(12, len(np.unique(x_axis)) // 3 or 12))
        bin_edges = np.linspace(float(np.nanmin(x_axis)), float(np.nanmax(x_axis)), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_ids = np.digitize(x_axis, bin_edges[1:-1], right=False)
        counts = np.bincount(bin_ids, minlength=n_bins)
    else:
        n_bins = 1
        bin_centers = np.array([float(x_axis[0])])
        bin_ids = np.zeros(len(x_axis), dtype=int)
        counts = np.array([len(x_axis)])

    def binned_mean(values):
        sums = np.bincount(bin_ids, weights=values, minlength=n_bins)
        means = np.full(n_bins, np.nan)
        means[counts > 0] = sums[counts > 0] / counts[counts > 0]
        return means

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].hist(residuals, bins=30, alpha=0.7, edgecolor="black", color="#4c78a8")
    axes[0, 0].set_title("Residuals Distribution", fontweight="bold")

    axes[0, 1].scatter(epistemic, abs_residuals, alpha=0.45, s=18, color="#f58518")
    axes[0, 1].set_title("Error vs Epistemic", fontweight="bold")

    axes[0, 2].scatter(aleatoric, abs_residuals, alpha=0.45, s=18, color="#54a24b")
    axes[0, 2].set_title("Error vs Aleatoric", fontweight="bold")

    axes[1, 0].scatter(y_model, y_mean, alpha=0.5, s=20, label="Prediction", color="#4c78a8")
    lo = float(min(np.nanmin(y_model), np.nanmin(y_mean)))
    hi = float(max(np.nanmax(y_model), np.nanmax(y_mean)))
    axes[1, 0].plot([lo, hi], [lo, hi], "r--", label="Perfect")
    axes[1, 0].set_title("Prediction vs Actual", fontweight="bold")
    axes[1, 0].legend()

    axes[1, 1].hist(y_std, bins=30, alpha=0.55, edgecolor="black", label="Total", color="#4c78a8")
    axes[1, 1].hist(epistemic, bins=30, alpha=0.45, edgecolor="black", label="Epistemic", color="#f58518")
    axes[1, 1].hist(aleatoric, bins=30, alpha=0.45, edgecolor="black", label="Aleatoric", color="#54a24b")
    axes[1, 1].set_title("Uncertainty Distribution", fontweight="bold")
    axes[1, 1].legend()

    ax_main = axes[1, 2]
    ax_density = ax_main.twinx()
    width = 0.9 if len(bin_centers) == 1 else (bin_centers[1] - bin_centers[0]) * 0.9
    ax_density.bar(bin_centers, counts, width=width, color="#d3d3d3", alpha=0.45, label="Sample Count")
    ax_main.plot(bin_centers, binned_mean(y_std), color="#4c78a8", linewidth=2.2, label="Total")
    ax_main.plot(bin_centers, binned_mean(epistemic), color="#f58518", linewidth=2.0, label="Epistemic")
    ax_main.plot(bin_centers, binned_mean(aleatoric), color="#54a24b", linewidth=2.0, label="Aleatoric")
    ax_main.plot(bin_centers, binned_mean(abs_residuals), color="#e45756", linewidth=2.0, linestyle="--", label="Abs Error")
    ax_main.set_title(f"Uncertainty vs {x_axis_name}", fontweight="bold")
    lines_main, labels_main = ax_main.get_legend_handles_labels()
    lines_density, labels_density = ax_density.get_legend_handles_labels()
    ax_main.legend(lines_main + lines_density, labels_main + labels_density, loc="upper left", fontsize=9)

    fig.suptitle(f"[{modeler.task_name}] Model-Space Uncertainty Diagnosis", fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(modeler.save_dir, "Uncertainty_Analysis.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ood_detection(modeler, df_train, df_test):
    """Save an uncertainty-threshold OOD diagnostic plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ood_res = detect_ood(modeler, df_test, X_train=df_train)
    scores = ood_res["uncertainty_score"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(scores, bins=30, alpha=0.7, edgecolor="black", color="#4c78a8")
    axes[0].axvline(
        ood_res["threshold"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold={ood_res['threshold']:.3g}",
    )
    axes[0].set_title("OOD Score Distribution", fontweight="bold")
    axes[0].set_xlabel("Model-space predictive std")
    axes[0].legend()

    n_ood = int(np.sum(ood_res["ood_flags"]))
    n_id = int(len(ood_res["ood_flags"]) - n_ood)
    axes[1].pie(
        [n_id, n_ood],
        labels=["ID Samples", "OOD Samples"],
        colors=["#54a24b", "#e45756"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05, 0.1),
    )
    axes[1].set_title(f"OOD Detection Results (Rate={ood_res['ood_ratio']:.1%})", fontweight="bold")

    fig.suptitle(f"[{modeler.task_name}] OOD Detection", fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(modeler.save_dir, "OOD_Detection.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ood_res


def plot_reliability_diagram(modeler, df_test):
    """Save a model-space reliability diagram."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    X_test = _frame_inputs(modeler, df_test)
    y_test = _frame_target_model_space(modeler, df_test)
    pred = modeler.predict_with_uncertainty(X_test)
    _require_prediction_keys(pred, ("model_mean", "model_std"))
    y_mean = np.asarray(pred["model_mean"], dtype=np.float64).reshape(-1)
    y_std = np.asarray(pred["model_std"], dtype=np.float64).reshape(-1)

    finite = np.isfinite(y_test) & np.isfinite(y_mean) & np.isfinite(y_std) & (y_std > 0.0)
    if not np.any(finite):
        raise ValueError("No finite test predictions available for reliability diagram")

    y_test = y_test[finite]
    y_mean = y_mean[finite]
    y_std = y_std[finite]
    coverage_frame = df_test.reset_index(drop=True).loc[finite].reset_index(drop=True)
    calibration_unit = getattr(modeler, "_calib_unit", "point")
    group_cols = [
        col for col in getattr(modeler, "_calib_group_cols", [])
        if col in coverage_frame.columns
    ]

    def empirical_coverage(covered):
        if calibration_unit == "curve" and group_cols:
            grouped = coverage_frame[group_cols].copy()
            grouped["_covered"] = covered
            return float(
                grouped.groupby(group_cols, dropna=False)["_covered"].all().mean()
            )
        return float(np.mean(covered))

    theoretical_conf = np.linspace(0.05, 0.99, 20)
    raw_coverage = []
    calibrated_coverage = []
    calibration_scores = np.asarray(
        getattr(modeler, "_calibration_scores", []), dtype=np.float64
    )
    for confidence in theoretical_conf:
        z_value = stats.norm.ppf((1.0 + confidence) / 2.0)
        raw_in_bound = (
            (y_test >= y_mean - z_value * y_std)
            & (y_test <= y_mean + z_value * y_std)
        )
        raw_coverage.append(empirical_coverage(raw_in_bound))
        if len(calibration_scores):
            calibrated_z = modeler._conformal_quantile(
                calibration_scores, confidence * 100.0
            )
        else:
            calibrated_z = z_value * getattr(modeler, "_calib_z", 1.96) / 1.96
        calibrated_in_bound = (
            (y_test >= y_mean - calibrated_z * y_std)
            & (y_test <= y_mean + calibrated_z * y_std)
        )
        calibrated_coverage.append(empirical_coverage(calibrated_in_bound))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration")
    ax.plot(
        theoretical_conf,
        raw_coverage,
        "o-",
        color="#e45756",
        linewidth=2.5,
        markersize=8,
        label="Raw VI",
    )
    ax.plot(
        theoretical_conf,
        calibrated_coverage,
        "s-",
        color="#54a24b",
        linewidth=2.5,
        markersize=7,
        label=f"Post-conformal ({calibration_unit})",
    )
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.set_xlabel(f"Nominal {calibration_unit.capitalize()} Confidence Level", fontweight="bold")
    ax.set_ylabel("Empirical Coverage", fontweight="bold")
    ax.set_title(f"[{modeler.task_name}] Reliability Diagram", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(modeler.save_dir, "Reliability_Diagram.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
