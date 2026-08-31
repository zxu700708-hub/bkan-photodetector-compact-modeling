"""Freeze one VI posterior draw as a deterministic KAN teacher.

This module changes no trained parameter or architecture.  It samples the
already-fitted variational coefficient and scale distributions exactly once,
copies that draw into the existing deterministic KAN representation, and then
keeps the sampled teacher fixed for all downstream rows.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch


def posterior_draw_to_deterministic(bayes_model, seed: int, device: str | torch.device = "cpu"):
    """Return a deterministic mean-channel KAN containing one fixed VI draw."""
    target = torch.device(device)
    deterministic = bayes_model.to_deterministic_kan(
        device=target,
        symbolic_enabled=False,
        auto_save=False,
    )
    cuda_devices = []
    if target.type == "cuda":
        cuda_devices = [target.index if target.index is not None else torch.cuda.current_device()]
    context = torch.random.fork_rng(devices=cuda_devices) if cuda_devices else nullcontext()
    with context:
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        samples = [layer.sample_parameters() for layer in bayes_model.layers]

    with torch.no_grad():
        for index, ((coef, scale_base, scale_sp), bayes_layer, det_layer) in enumerate(
            zip(samples, bayes_model.layers, deterministic.act_fun)
        ):
            det_layer.grid.copy_(bayes_layer.grid.to(target))
            mean_channel_only = index == len(bayes_model.layers) - 1 and bayes_model.width[-1] == 2
            if mean_channel_only:
                det_layer.coef.copy_(coef[:, 0:1, :].to(target))
                det_layer.scale_base.copy_(scale_base[:, 0:1].to(target))
                det_layer.scale_sp.copy_(scale_sp[:, 0:1].to(target))
                det_layer.mask.copy_(bayes_layer.mask[:, 0:1].to(target))
            else:
                det_layer.coef.copy_(coef.to(target))
                det_layer.scale_base.copy_(scale_base.to(target))
                det_layer.scale_sp.copy_(scale_sp.to(target))
                det_layer.mask.copy_(bayes_layer.mask.to(target))
    return deterministic.eval()


def predict_fixed_teacher(modeler, deterministic, x_raw: np.ndarray) -> np.ndarray:
    """Predict in model/output space using one frozen posterior-draw teacher."""
    x_raw = np.asarray(x_raw, dtype=np.float32)
    x_scaled = modeler.scaler_x.transform(x_raw).astype(np.float32)
    device = next(deterministic.parameters()).device
    with torch.no_grad():
        prediction_scaled = deterministic(torch.tensor(x_scaled, device=device))
        if isinstance(prediction_scaled, tuple):
            prediction_scaled = prediction_scaled[0]
    values = prediction_scaled.detach().cpu().numpy().reshape(-1, 1)
    prediction = modeler.scaler_y.inverse_transform(values).reshape(-1)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("Frozen posterior-draw teacher produced non-finite predictions")
    return prediction.astype(np.float64)
