"""Prediction plotting helpers for Bayesian device models."""

from __future__ import annotations

import os

import numpy as np


def plot_predictions_with_uncertainty(modeler, df_raw, x_col_name, step_col_name, use_bayes=True):
    """Plot representative curves with Bayesian confidence intervals."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inputs = modeler.input_cols
    if x_col_name not in inputs or step_col_name not in inputs:
        return

    unique_vals = np.sort(df_raw[step_col_name].unique())
    if len(unique_vals) > 6:
        sample_idx = np.linspace(0, len(unique_vals) - 1, 6, dtype=int)
        unique_vals = unique_vals[sample_idx]

    n_plots = len(unique_vals)
    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))

    x_idx = inputs.index(x_col_name)
    step_idx = inputs.index(step_col_name)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_plots))

    for scale in ["log", "linear"]:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
        axes = axes.flatten()

        for i, step_value in enumerate(unique_vals):
            ax = axes[i]
            df_subset = df_raw[np.isclose(df_raw[step_col_name], step_value, rtol=1e-3, atol=1e-12)]

            for other_col in [col for col in inputs if col not in [x_col_name, step_col_name]]:
                if df_subset.empty:
                    break
                mode_series = df_subset[other_col].mode()
                if not mode_series.empty:
                    df_subset = df_subset[
                        np.isclose(df_subset[other_col], mode_series.iloc[0], rtol=1e-3)
                    ]

            if df_subset.empty:
                continue

            real_x = df_subset[x_col_name].values
            real_y_raw = df_subset[modeler.output_col].values

            if scale == "log" and modeler.use_log_transform:
                real_y = np.log10(np.maximum(real_y_raw, 1e-30))
                y_label = f"Log10({modeler.output_col})"
            else:
                real_y = real_y_raw
                y_label = modeler.output_col

            check_y = (
                np.log10(np.maximum(real_y_raw, 1e-30))
                if modeler.use_log_transform
                else real_y_raw
            )
            valid_mask = (check_y > modeler.y_bounds[0]) & (check_y < modeler.y_bounds[1])
            real_x = real_x[valid_mask]
            real_y = real_y[valid_mask]

            ax.scatter(
                real_x,
                real_y,
                s=30,
                alpha=0.6,
                color=colors[i],
                label=f"TCAD Data (n={len(real_x)})",
                zorder=2,
            )

            if len(real_x) > 1:
                x_range = real_x.max() - real_x.min()
                x_pred = np.linspace(
                    real_x.min() - x_range * 0.3,
                    real_x.max() + x_range * 0.3,
                    150,
                ).reshape(-1, 1)
            else:
                x_pred = np.linspace(-5, 5, 100).reshape(-1, 1)

            X_infer = np.zeros((len(x_pred), len(inputs)))
            X_infer[:, x_idx] = x_pred.flatten()
            X_infer[:, step_idx] = step_value

            for col_idx, col_name in enumerate(inputs):
                if col_idx in [x_idx, step_idx]:
                    continue
                mode_series = df_subset[col_name].mode()
                X_infer[:, col_idx] = (
                    mode_series.iloc[0] if not mode_series.empty else df_raw[col_name].mean()
                )

            if use_bayes:
                pred_res = modeler.predict_with_uncertainty(X_infer)
                y_mean = pred_res["log_mean"] if scale == "log" else pred_res["mean"]
                y_upper = pred_res["log_upper"] if scale == "log" else pred_res["upper"]
                y_lower = pred_res["log_lower"] if scale == "log" else pred_res["lower"]

                ax.plot(x_pred, y_mean, color=colors[i], linewidth=2.5, label="Bayesian Mean", zorder=3)
                ax.fill_between(
                    x_pred.flatten(),
                    y_lower,
                    y_upper,
                    alpha=0.2,
                    color=colors[i],
                    label="95% CI",
                    zorder=1,
                )

                if len(real_x) > 5:
                    axins = ax.inset_axes([0.05, 0.05, 0.35, 0.35])
                    axins.scatter(real_x, real_y, s=20, alpha=0.6, color=colors[i])
                    axins.plot(x_pred, y_mean, color=colors[i], linewidth=2)
                    axins.fill_between(x_pred.flatten(), y_lower, y_upper, alpha=0.2, color=colors[i])
                    mid_idx = len(real_x) // 2
                    zx, zy = real_x[mid_idx], real_y[mid_idx]
                    zs_x = (real_x.max() - real_x.min()) * 0.02
                    zs_y = 0.05 if scale == "log" else zy * 0.02
                    axins.set_xlim(zx - zs_x, zx + zs_x)
                    axins.set_ylim(zy - zs_y, zy + zs_y)
                    axins.set_xticks([])
                    axins.set_yticks([])
                    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.3)

            ax.set_xlabel(x_col_name, fontsize=10, fontweight="bold")
            ax.set_ylabel(y_label, fontsize=10, fontweight="bold")
            ax.set_title(f"{step_col_name} = {step_value:.4g}", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best", fontsize=8)

        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(
            f"[{modeler.task_name}] Prediction vs Data ({scale.capitalize()})",
            fontsize=14,
            fontweight="bold",
            y=1.00,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(modeler.save_dir, f"Predictions_{scale.capitalize()}_{step_col_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
