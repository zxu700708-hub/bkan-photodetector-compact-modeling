"""Heteroscedastic MLP uncertainty baselines.

The models in this module deliberately use the same two-channel Gaussian
likelihood and mean-to-NLL warm-up used by :class:`BayesianExperiment`.  They
provide an architecture-independent MC-dropout baseline and the members used
by a deep ensemble without depending on the KAN implementation.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kan.bayes_loss import variance_from_logits


@dataclass(frozen=True)
class MLPTrainingConfig:
    """Frozen per-member optimization budget."""

    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    train_mc_samples: int = 3
    validation_mc_samples: int = 20
    validation_frequency: int = 10
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1.0e-4
    warmup_epochs: int = 50
    transition_epochs: int = 30
    gradient_clip_norm: float = 1.0
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.train_mc_samples < 1 or self.validation_mc_samples < 1:
            raise ValueError("Monte Carlo sample counts must be positive")
        if self.validation_frequency < 1:
            raise ValueError("validation_frequency must be positive")


class HeteroscedasticMLP(nn.Module):
    """ReLU MLP with mean and unconstrained variance-logit outputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_widths: Sequence[int] = (64, 32),
        dropout_rate: float = 0.05,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if not hidden_widths or any(int(width) < 1 for width in hidden_widths):
            raise ValueError("hidden_widths must contain positive widths")
        if not 0.0 <= float(dropout_rate) < 1.0:
            raise ValueError("dropout_rate must lie in [0, 1)")

        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_widths:
            width = int(width)
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(float(dropout_rate)))
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_widths = tuple(int(width) for width in hidden_widths)
        self.dropout_rate = float(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for one isolated fit."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def heteroscedastic_gaussian_nll(
    output: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL without the additive constant, matching BKAN training."""

    if output.shape[-1] != 2:
        raise ValueError("heteroscedastic output must have two channels")
    mean = output[..., 0:1]
    variance = variance_from_logits(output[..., 1:2])
    return 0.5 * (torch.log(variance) + (target - mean).pow(2) / variance).mean()


def _expected_validation_nll(
    model: HeteroscedasticMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    samples: int,
    seed: int,
) -> float:
    previous_mode = model.training
    model.train(model.dropout_rate > 0.0)
    devices = [inputs.device] if inputs.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if inputs.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            losses = [
                heteroscedastic_gaussian_nll(model(inputs), targets)
                for _ in range(max(1, int(samples)))
            ]
    model.train(previous_mode)
    return float(torch.stack(losses).mean().item())


def train_heteroscedastic_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    hidden_widths: Sequence[int],
    dropout_rate: float,
    seed: int,
    validation_seed: int,
    device: torch.device | str,
    config: MLPTrainingConfig,
) -> tuple[HeteroscedasticMLP, dict]:
    """Fit one member using validation NLL for checkpoint selection."""

    config.validate()
    set_reproducible_seed(seed)
    device = torch.device(device)
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_validation_tensor = torch.as_tensor(
        x_validation, dtype=torch.float32, device=device
    )
    y_validation_tensor = torch.as_tensor(
        y_validation, dtype=torch.float32, device=device
    ).reshape(-1, 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=generator,
    )

    model = HeteroscedasticMLP(
        input_dim=x_train_tensor.shape[1],
        hidden_widths=hidden_widths,
        dropout_rate=dropout_rate,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config.scheduler_step_size),
        gamma=float(config.scheduler_gamma),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_nll = math.inf
    checks_without_improvement = 0
    epochs_completed = 0
    optimizer_updates = 0
    history: list[dict] = []

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        running_loss = 0.0
        running_nll = 0.0
        running_mse = 0.0
        batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            effective_samples = (
                int(config.train_mc_samples) if model.dropout_rate > 0.0 else 1
            )
            outputs = [model(x_batch) for _ in range(effective_samples)]
            nll = torch.stack(
                [heteroscedastic_gaussian_nll(output, y_batch) for output in outputs]
            ).mean()
            mse = torch.stack(
                [torch.mean((output[..., 0:1] - y_batch).pow(2)) for output in outputs]
            ).mean()

            if epoch <= int(config.warmup_epochs):
                loss = mse
            else:
                progress = min(
                    1.0,
                    (epoch - int(config.warmup_epochs))
                    / max(1, int(config.transition_epochs)),
                )
                mse_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
                loss = mse_weight * mse + (1.0 - mse_weight) * nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.gradient_clip_norm)
            )
            optimizer.step()
            optimizer_updates += 1
            running_loss += float(loss.item())
            running_nll += float(nll.item())
            running_mse += float(mse.item())
            batches += 1

        scheduler.step()
        epochs_completed = epoch
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batches),
            "train_nll": running_nll / max(1, batches),
            "train_mse": running_mse / max(1, batches),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if epoch % int(config.validation_frequency) == 0:
            validation_nll = _expected_validation_nll(
                model,
                x_validation_tensor,
                y_validation_tensor,
                config.validation_mc_samples,
                validation_seed,
            )
            row["validation_nll"] = validation_nll
            if validation_nll < best_validation_nll - float(
                config.early_stopping_min_delta
            ):
                best_validation_nll = validation_nll
                best_state = copy.deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            history.append(row)
            if (
                config.early_stopping_patience > 0
                and checks_without_improvement
                >= int(config.early_stopping_patience)
            ):
                break
        else:
            history.append(row)

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": int(seed),
        "epochs_completed": int(epochs_completed),
        "optimizer_updates": int(optimizer_updates),
        "best_validation_nll": float(best_validation_nll),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "dropout_rate": float(dropout_rate),
        "hidden_widths": list(int(width) for width in hidden_widths),
        "training_config": asdict(config),
        "history": history,
    }


def predictive_moments(
    models: Iterable[HeteroscedasticMLP],
    inputs: np.ndarray,
    *,
    method: str,
    mc_samples: int,
    prediction_seed: int,
    device: torch.device | str,
) -> dict[str, np.ndarray]:
    """Return standardized predictive components for dropout or an ensemble."""

    model_list = list(models)
    if not model_list:
        raise ValueError("At least one fitted model is required")
    if method not in {"mc_dropout", "deep_ensemble"}:
        raise ValueError(f"Unknown predictive method: {method}")
    if method == "mc_dropout" and len(model_list) != 1:
        raise ValueError("MC dropout expects exactly one fitted model")

    device = torch.device(device)
    x_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    previous_modes = [model.training for model in model_list]
    devices = [device] if device.type == "cuda" else []
    outputs: list[torch.Tensor] = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(prediction_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(prediction_seed))
        with torch.no_grad():
            if method == "mc_dropout":
                model_list[0].train()
                outputs = [model_list[0](x_tensor) for _ in range(int(mc_samples))]
            else:
                for model in model_list:
                    model.eval()
                    outputs.append(model(x_tensor))
    for model, previous_mode in zip(model_list, previous_modes):
        model.train(previous_mode)

    stacked = torch.stack(outputs)
    mean_samples = stacked[..., 0:1]
    aleatoric_samples = variance_from_logits(stacked[..., 1:2])
    epistemic_variance = torch.var(mean_samples, dim=0, unbiased=False)
    aleatoric_variance = torch.mean(aleatoric_samples, dim=0)
    total_variance = epistemic_variance + aleatoric_variance + 1.0e-8
    return {
        "mean_samples": mean_samples.cpu().numpy(),
        "aleatoric_variance_samples": aleatoric_samples.cpu().numpy(),
        "mean": torch.mean(mean_samples, dim=0).cpu().numpy(),
        "epistemic_variance": epistemic_variance.cpu().numpy(),
        "aleatoric_variance": aleatoric_variance.cpu().numpy(),
        "total_variance": total_variance.cpu().numpy(),
    }
