"""Symbolic-gated KAN training with an in-model physical term branch.

This is not post-hoc symbolic regression.  For photo current, the symbolic
branch is hierarchical:

    I_photo = main_photogeneration_collection
              * surface_loss_correction
              * trap_loss_correction
              + small_residual

The symbolic gates and physical coefficients are optimized in the same training
objective as the KAN residual.  Training follows three stages:

1. numerical_stability: train KAN only.
2. symbolic_competition: enable physical gates and penalize KAN residual use.
3. structure_freeze: hard-prune low-probability symbolic terms and refit.

``pure_symbolic_student`` changes the deployment contract: the KAN is then
excluded from every forward pass, so the retained gate terms are the complete
student model rather than a correction added to a neural residual.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from kan import KAN
from device_modeling.photodetector.common import concat_without_attrs, safe_scale, set_seed, without_attrs
from device_modeling.photodetector.task_config import TASKS, TaskSpec, target_values
from device_modeling.photodetector.physical_library import PhysicalFeatureLibrary, finite_difference_derivative


EPS = 1e-12
TARGET_SCALE_EPS = 1e-30
LOG2 = math.log(2.0)


def _symbolic_drive_expression(drive: str, axis: str, reverse_scale: float, reverse_norm: float) -> str:
    """Return the deployable counterpart of ``mechanism_drives``."""
    reverse = (
        f"clip(softplus(-{axis}/{reverse_scale:.16g}) - log(2), 0, inf)"
        f" / {reverse_norm:.16g}"
    )
    forward = (
        f"clip(softplus({axis}/{reverse_scale:.16g}) - log(2), 0, inf)"
        f" / {reverse_norm:.16g}"
    )
    if drive == "reverse_zero":
        return reverse
    if drive == "forward_zero":
        return forward
    if drive == "reverse":
        return f"sigmoid(4*({reverse}))"
    if drive == "forward":
        return f"sigmoid(4*({forward}))"
    return "1"


@dataclass(frozen=True)
class StageConfig:
    name: str
    steps: int
    learning_rate: float
    symbolic_weight: float
    residual_weight: float
    complexity_weight: float
    binary_weight: float
    temperature_start: float
    temperature_end: float


@dataclass(frozen=True)
class TrainTensors:
    x: torch.Tensor
    y: torch.Tensor
    dy: torch.Tensor | None
    curve_id: torch.Tensor | None
    weight: torch.Tensor | None
    teacher_y: torch.Tensor | None


@dataclass(frozen=True)
class MechanismSpec:
    name: str
    drive: str
    sign: float


def _temperature(step: int, steps: int, start: float, end: float) -> float:
    if steps <= 1:
        return end
    ratio = step / float(steps - 1)
    return float(start * ((end / start) ** ratio))


def _sensitivity_table(x_raw: np.ndarray, y: np.ndarray, inputs: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    y_centered = y - np.mean(y)
    y_norm = np.linalg.norm(y_centered)
    for index, name in enumerate(inputs):
        col = x_raw[:, index]
        col_centered = col - np.mean(col)
        denom = np.linalg.norm(col_centered) * y_norm
        corr = abs(float(np.dot(col_centered, y_centered) / denom)) if denom > EPS else 0.0
        rows.append(
            {
                "input": name,
                "mean_abs_teacher_gradient": corr,
                "input_range": float(np.ptp(col)),
                "range_scaled_sensitivity": corr * float(np.ptp(col)),
                "normalized_sensitivity": corr,
            }
        )
    values = np.asarray([row["normalized_sensitivity"] for row in rows], dtype=np.float64)
    max_value = max(float(values.max()), EPS)
    for row in rows:
        row["normalized_sensitivity"] = float(row["normalized_sensitivity"] / max_value)
    return pd.DataFrame(rows)


class SymbolicGatedKAN(torch.nn.Module):
    def __init__(
        self,
        spec: TaskSpec,
        inputs: tuple[str, ...],
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        y_scale: float,
        train_x_raw: np.ndarray,
        train_y: np.ndarray,
        width: int,
        grid: int,
        spline_order: int,
        seed: int,
        device: torch.device,
        generic_symbolic_terms: bool = False,
        max_generic_symbolic_terms: int = 4,
        generic_complexity_multiplier: float = 5.0,
        disabled_photo_mechanisms: tuple[str, ...] = tuple(),
        photo_coupling_mode: str = "separate",
        photo_loss_branch_max_terms: int | None = None,
        photo_residual_min_terms: int = 0,
        flat_gates: bool = False,
        coeff_init: np.ndarray | None = None,
        pure_symbolic_student: bool = False,
    ):
        super().__init__()
        self.spec = spec
        self.inputs = inputs
        self.y_scale = float(max(abs(y_scale), TARGET_SCALE_EPS))
        self.device = device
        self.generic_symbolic_terms = generic_symbolic_terms
        self.max_generic_symbolic_terms = int(max_generic_symbolic_terms)
        self.generic_complexity_multiplier = float(generic_complexity_multiplier)
        self.disabled_photo_mechanisms = set(disabled_photo_mechanisms)
        self.photo_coupling_mode = photo_coupling_mode
        self.photo_loss_branch_max_terms = (
            None if photo_loss_branch_max_terms is None else max(1, int(photo_loss_branch_max_terms))
        )
        self.photo_residual_min_terms = max(0, int(photo_residual_min_terms))
        self.flat_gates = flat_gates
        # In this mode the KAN is deliberately excluded from the forward pass.
        # It may still exist as a training-time teacher elsewhere in the
        # workflow, but it is not part of the deployable student expression.
        self.pure_symbolic_student = bool(pure_symbolic_student)
        self.hard_structure = False
        self.kan_adapter_enabled = False
        self.register_buffer("kan_adapter_x_mean", None)
        self.register_buffer("kan_adapter_x_scale", None)
        self.register_buffer("kan_adapter_y_mean", None)
        self.register_buffer("kan_adapter_y_scale", None)
        self.kan = KAN(
            width=[len(inputs), width, 1],
            grid=grid,
            k=spline_order,
            seed=seed,
            device=device,
            symbolic_enabled=False,
            auto_save=False,
        )
        sensitivity = _sensitivity_table(train_x_raw, train_y, inputs)
        self.library = PhysicalFeatureLibrary(
            spec=spec,
            inputs=inputs,
            x_mean=x_mean,
            x_scale=x_scale,
            train_x_raw=train_x_raw,
            sensitivity=sensitivity,
            generic_symbolic_terms=generic_symbolic_terms,
        )
        n_terms = len(self.library.terms)
        priors = np.clip(0.45 + 0.10 * self.library.term_priors, 0.05, 0.95)
        logits = np.log(priors / (1.0 - priors))

        if flat_gates:
            # Flat mode: single pool, no mechanisms, no hierarchy
            self.mechanisms = []
            mechanism_count = 1
            self.gate_logits = torch.nn.Parameter(torch.tensor(logits.reshape(1, -1), dtype=torch.float32, device=device))
            if coeff_init is not None:
                self.coeff = torch.nn.Parameter(torch.tensor(coeff_init.reshape(1, -1), dtype=torch.float32, device=device))
            else:
                self.coeff = torch.nn.Parameter(0.02 * torch.randn(1, n_terms, device=device))
            self.bias = torch.nn.Parameter(torch.zeros(1, device=device))
            self.residual_gate_logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32, device=device))
            self.residual_coeff = torch.nn.Parameter(0.02 * torch.randn(n_terms, device=device))
            self.residual_bias = torch.nn.Parameter(torch.zeros(1, device=device))
            self.generic_gate_logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32, device=device))
            self.generic_coeff = torch.nn.Parameter(0.02 * torch.randn(n_terms, device=device))
            term_mask = np.ones((1, n_terms), dtype=np.float32)
            generic_mask = np.asarray(
                [1.0 if term.name.startswith("generic_") else 0.0 for term in self.library.terms],
                dtype=np.float32,
            ) if generic_symbolic_terms else np.zeros(n_terms, dtype=np.float32)
            residual_term_mask = np.ones(n_terms, dtype=np.float32)
            self.register_buffer("term_mask", torch.tensor(term_mask, dtype=torch.float32, device=device))
            self.register_buffer("active_mask", torch.tensor(term_mask, dtype=torch.float32, device=device))
            self.register_buffer("generic_term_mask", torch.tensor(generic_mask, dtype=torch.float32, device=device))
            self.register_buffer("residual_term_mask", torch.tensor(residual_term_mask, dtype=torch.float32, device=device))
            self.register_buffer("residual_active_mask", torch.tensor(residual_term_mask, dtype=torch.float32, device=device))
            self.register_buffer("generic_active_mask", torch.tensor(generic_mask, dtype=torch.float32, device=device))
            self.mechanism_scale = torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False)
            self.register_buffer("mechanism_sign", torch.ones(1, dtype=torch.float32, device=device))
            self.register_buffer("photo_main_mask", torch.ones(1, dtype=torch.float32, device=device))
            self.register_buffer("photo_surface_loss_mask", torch.zeros(1, dtype=torch.float32, device=device))
            self.register_buffer("photo_trap_loss_mask", torch.zeros(1, dtype=torch.float32, device=device))
            self.register_buffer("photo_additive_mask", torch.zeros(1, dtype=torch.float32, device=device))
            self.reverse_scale = 0.2
            self.reverse_norm = 1.0
            self.generic_symbolic_terms = generic_symbolic_terms
            self.max_generic_symbolic_terms = int(max_generic_symbolic_terms)
            self.generic_complexity_multiplier = float(generic_complexity_multiplier)
            self.disabled_photo_mechanisms = set()
            self.photo_coupling_mode = "separate"
            self.photo_loss_branch_max_terms = None
            self.photo_residual_min_terms = 0
            return
        self.mechanisms = self._build_mechanisms(train_x_raw, train_y)
        mechanism_count = len(self.mechanisms)
        tiled_logits = np.tile(logits.reshape(1, -1), (mechanism_count, 1))
        self.gate_logits = torch.nn.Parameter(torch.tensor(tiled_logits, dtype=torch.float32, device=device))
        self.coeff = torch.nn.Parameter(0.02 * torch.randn(mechanism_count, n_terms, device=device))
        self.bias = torch.nn.Parameter(torch.tensor(self._initial_mechanism_biases(), dtype=torch.float32, device=device))
        self.residual_gate_logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32, device=device))
        self.residual_coeff = torch.nn.Parameter(0.02 * torch.randn(n_terms, device=device))
        self.residual_bias = torch.nn.Parameter(torch.zeros(1, device=device))
        self.generic_gate_logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32, device=device))
        self.generic_coeff = torch.nn.Parameter(0.02 * torch.randn(n_terms, device=device))
        term_mask = self._build_mechanism_term_mask()
        generic_mask = np.asarray(
            [1.0 if term.name.startswith("generic_") else 0.0 for term in self.library.terms],
            dtype=np.float32,
        )
        physical_mask = 1.0 - generic_mask
        residual_term_mask = physical_mask if generic_symbolic_terms else np.ones(n_terms, dtype=np.float32)
        self.register_buffer("term_mask", torch.tensor(term_mask, dtype=torch.float32, device=device))
        self.register_buffer("active_mask", torch.tensor(term_mask, dtype=torch.float32, device=device))
        self.register_buffer("generic_term_mask", torch.tensor(generic_mask, dtype=torch.float32, device=device))
        self.register_buffer("residual_term_mask", torch.tensor(residual_term_mask, dtype=torch.float32, device=device))
        self.register_buffer("residual_active_mask", torch.tensor(residual_term_mask, dtype=torch.float32, device=device))
        self.register_buffer("generic_active_mask", torch.tensor(generic_mask if generic_symbolic_terms else np.zeros(n_terms, dtype=np.float32), dtype=torch.float32, device=device))
        mechanism_scale_init = np.asarray(self._initial_mechanism_scales(), dtype=np.float32)
        self.mechanism_scale = torch.nn.Parameter(
            torch.tensor(mechanism_scale_init, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "mechanism_sign",
            torch.tensor([mechanism.sign for mechanism in self.mechanisms], dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "photo_main_mask",
            torch.tensor(
                [1.0 if self._photo_mechanism_role(mechanism.name) == "main" else 0.0 for mechanism in self.mechanisms],
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "photo_surface_loss_mask",
            torch.tensor(
                [1.0 if self._photo_mechanism_role(mechanism.name) == "surface_loss" else 0.0 for mechanism in self.mechanisms],
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "photo_trap_loss_mask",
            torch.tensor(
                [1.0 if self._photo_mechanism_role(mechanism.name) == "trap_loss" else 0.0 for mechanism in self.mechanisms],
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "photo_additive_mask",
            torch.tensor(
                [1.0 if self._photo_mechanism_role(mechanism.name) == "additive" else 0.0 for mechanism in self.mechanisms],
                dtype=torch.float32,
                device=device,
            ),
        )
        self.hard_structure = False
        self.reverse_scale = 0.2
        self.reverse_norm = float(F.softplus(torch.tensor(3.0 / self.reverse_scale)).item() - LOG2)

    def _observed_sign(self, train_x_raw: np.ndarray, train_y: np.ndarray, selector: np.ndarray, default: float) -> float:
        values = train_y[selector]
        if values.size == 0:
            values = train_y
        mean_value = float(np.mean(values)) if values.size else default
        if abs(mean_value) < EPS:
            return float(default)
        return 1.0 if mean_value > 0.0 else -1.0

    def _build_mechanisms(self, train_x_raw: np.ndarray, train_y: np.ndarray) -> tuple[MechanismSpec, ...]:
        if self.spec.key not in {"dark_current", "photo_current"}:
            return (MechanismSpec("symbolic_response", "all", 1.0),)

        axis = train_x_raw[:, self.inputs.index(self.spec.axis_col)]
        reverse_sign = self._observed_sign(train_x_raw, train_y, axis < 0.0, default=1.0)
        forward_sign = self._observed_sign(train_x_raw, train_y, axis > 0.0, default=-reverse_sign)
        total_sign = self._observed_sign(train_x_raw, train_y, np.ones_like(axis, dtype=bool), default=reverse_sign)
        has_forward_data = bool(np.any(axis > 0.0))

        if self.spec.key == "dark_current":
            mechanisms = [MechanismSpec("reverse_leakage", "reverse_zero", reverse_sign)]
            if has_forward_data:
                mechanisms.append(MechanismSpec("forward_injection", "forward_zero", forward_sign))
            return tuple(mechanisms)
        if self.spec.key == "photo_current":
            if self.photo_coupling_mode == "coupled":
                mechanisms = [
                    MechanismSpec("photogeneration_collection", "reverse", total_sign),
                    MechanismSpec("surface_recombination", "reverse", -total_sign),
                    MechanismSpec("trap_recombination", "reverse", -total_sign),
                ]
            else:
                mechanisms = [
                    MechanismSpec("photogeneration", "all", total_sign),
                    MechanismSpec("field_collection", "reverse", total_sign),
                    MechanismSpec("surface_recombination", "reverse", -total_sign),
                    MechanismSpec("trap_recombination", "reverse", -total_sign),
                ]
            if has_forward_data:
                mechanisms.append(MechanismSpec("forward_injection", "forward", forward_sign))
            return tuple(
                mechanism
                for mechanism in mechanisms
                if mechanism.name not in self.disabled_photo_mechanisms
            )
        return (MechanismSpec("symbolic_current", "all", total_sign),)

    def _initial_mechanism_biases(self) -> list[float]:
        if self.spec.key == "dark_current":
            return [0.0 for _ in self.mechanisms]
        if self.spec.key == "photo_current":
            return [
                0.0 if mechanism.name in {"photogeneration", "photogeneration_collection"} else -3.0
                for mechanism in self.mechanisms
            ]
        return [0.0 for _ in self.mechanisms]

    def _initial_mechanism_scales(self) -> list[float]:
        if self.spec.key == "dark_current":
            return [mechanism.sign for mechanism in self.mechanisms]
        if self.spec.key == "photo_current":
            scales = {
                "photogeneration": 1.0,
                "photogeneration_collection": 1.0,
                "field_collection": 0.2,
                "surface_recombination": 0.1,
                "trap_recombination": 0.1,
                "forward_injection": 0.1,
            }
            return [mechanism.sign * scales.get(mechanism.name, 0.1) for mechanism in self.mechanisms]
        return [mechanism.sign for mechanism in self.mechanisms]

    @staticmethod
    def _photo_mechanism_role(mechanism: str) -> str:
        if mechanism in {"photogeneration", "field_collection", "photogeneration_collection"}:
            return "main"
        if mechanism == "surface_recombination":
            return "surface_loss"
        if mechanism == "trap_recombination":
            return "trap_loss"
        if mechanism == "forward_injection":
            return "additive"
        return "other"

    def _build_mechanism_term_mask(self) -> np.ndarray:
        rows = []
        for mechanism in self.mechanisms:
            rows.append([1.0 if self._term_allowed(mechanism.name, term.name) else 0.0 for term in self.library.terms])
        return np.asarray(rows, dtype=np.float32)

    def _term_allowed(self, mechanism: str, term_name: str) -> bool:
        if self.spec.key != "photo_current":
            return True

        common = {
            "one",
            f"{self.spec.axis_col}_z",
            f"{self.spec.axis_col}_z2",
            "temperature_z",
            "inverse_temperature",
            f"{self.spec.axis_col}_over_temperature",
            "length_z",
            "inverse_length",
            f"{self.spec.axis_col}_over_length",
        }
        voltage_reverse = {
            f"soft_reverse_{self.spec.axis_col}",
            f"tanh_reverse_{self.spec.axis_col}",
            f"reverse_saturation_{self.spec.axis_col}",
            f"reverse_gate_{self.spec.axis_col}",
            "reverse_drive_over_length",
        }
        voltage_forward = {
            f"forward_gate_{self.spec.axis_col}",
        }
        collection = {
            "collection_saturation_length",
            "length_collection_saturation",
        }
        surface = {
            "ge_sio2_velocity_z",
            "ge_si_velocity_z",
            "ge_sio2_velocity_over_length",
            "ge_si_velocity_over_length",
            "surface_recomb_sum_z",
            "surface_recomb_over_length",
            "reverse_drive_surface_recomb",
            "thermal_surface_recomb_over_length",
            "reverse_thermal_surface_recomb",
        }
        trap = {
            "trap_A_z",
            "reverse_drive_trap_A",
            "trap_A_times_length",
            "trap_A_over_temperature",
        }
        allowed_by_mechanism = {
            "photogeneration": common | collection,
            "photogeneration_collection": common | voltage_reverse | collection,
            "field_collection": common | voltage_reverse | collection,
            "surface_recombination": common | voltage_reverse | surface,
            "trap_recombination": common | voltage_reverse | trap,
            "forward_injection": common | voltage_forward,
        }
        return term_name in allowed_by_mechanism.get(mechanism, common)

    def gates(self, temperature: float) -> torch.Tensor:
        if self.hard_structure:
            return self.active_mask
        return torch.sigmoid(self.gate_logits / temperature) * self.active_mask * self.term_mask

    def residual_gates(self, temperature: float) -> torch.Tensor:
        if self.hard_structure:
            return self.residual_active_mask
        return torch.sigmoid(self.residual_gate_logits / temperature) * self.residual_active_mask * self.residual_term_mask

    def generic_gates(self, temperature: float) -> torch.Tensor:
        if self.hard_structure:
            return self.generic_active_mask
        return torch.sigmoid(self.generic_gate_logits / temperature) * self.generic_active_mask * self.generic_term_mask

    def symbolic_part(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        phi = self.library.matrix(x)
        if self.flat_gates:
            weights = self.coeff[0] * self.gates(temperature)[0]
            residual = phi @ (self.residual_coeff * self.residual_gates(temperature)).unsqueeze(1) + self.residual_bias
            generic = phi @ (self.generic_coeff * self.generic_gates(temperature)).unsqueeze(1)
            return phi @ weights.unsqueeze(1) + self.bias[0] + residual + generic
        weights = self.coeff * self.gates(temperature)
        raw = phi @ weights.T + self.bias.unsqueeze(0)
        if self.spec.key not in {"dark_current", "photo_current"}:
            return raw[:, [0]]
        amplitudes = F.softplus(raw)
        drives = self.mechanism_drives(x)
        if self.spec.key == "photo_current":
            return self._photo_hierarchical_part(phi, amplitudes, drives, temperature)
        signed = amplitudes * drives * self.mechanism_scale.unsqueeze(0)
        residual = phi @ (self.residual_coeff * self.residual_gates(temperature)).unsqueeze(1) + self.residual_bias
        generic = phi @ (self.generic_coeff * self.generic_gates(temperature)).unsqueeze(1)
        return signed.sum(dim=1, keepdim=True) + residual + generic

    def _photo_hierarchical_part(
        self,
        phi: torch.Tensor,
        amplitudes: torch.Tensor,
        drives: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        magnitudes = amplitudes * drives * torch.clamp(self.mechanism_scale.abs(), min=EPS).unsqueeze(0)
        signs = self.mechanism_sign.unsqueeze(0)
        main = (magnitudes * signs * self.photo_main_mask.unsqueeze(0)).sum(dim=1, keepdim=True)
        additive = (magnitudes * signs * self.photo_additive_mask.unsqueeze(0)).sum(dim=1, keepdim=True)

        surface_loss = (magnitudes * self.photo_surface_loss_mask.unsqueeze(0)).sum(dim=1, keepdim=True)
        trap_loss = (magnitudes * self.photo_trap_loss_mask.unsqueeze(0)).sum(dim=1, keepdim=True)
        surface_factor = torch.exp(-torch.clamp(surface_loss, min=0.0, max=8.0))
        trap_factor = torch.exp(-torch.clamp(trap_loss, min=0.0, max=8.0))

        residual = phi @ (self.residual_coeff * self.residual_gates(temperature)).unsqueeze(1) + self.residual_bias
        generic = phi @ (self.generic_coeff * self.generic_gates(temperature)).unsqueeze(1)
        return main * surface_factor * trap_factor + additive + residual + generic

    def configure_kan_adapter(self, adapter: dict[str, np.ndarray | float]) -> None:
        """Map this model's normalized coordinates to a pretrained KAN's space."""
        self.kan_adapter_x_mean = torch.as_tensor(adapter["x_mean"], dtype=torch.float32, device=self.device).reshape(1, -1)
        self.kan_adapter_x_scale = torch.as_tensor(adapter["x_scale"], dtype=torch.float32, device=self.device).reshape(1, -1)
        self.kan_adapter_y_mean = torch.as_tensor(adapter["y_mean"], dtype=torch.float32, device=self.device).reshape(1, 1)
        self.kan_adapter_y_scale = torch.as_tensor(adapter["y_scale"], dtype=torch.float32, device=self.device).reshape(1, 1)
        self.kan_adapter_enabled = True

    def kan_part(self, x: torch.Tensor) -> torch.Tensor:
        if self.kan_adapter_enabled:
            raw = self.library.raw_from_z(x)
            teacher_x = (raw - self.kan_adapter_x_mean) / self.kan_adapter_x_scale
            teacher_model_space = self.kan(teacher_x) * self.kan_adapter_y_scale + self.kan_adapter_y_mean
            return teacher_model_space / self.y_scale
        return self.kan(x)

    def raw_output(self, x: torch.Tensor, temperature: float = 1.0, symbolic_weight: float = 1.0) -> torch.Tensor:
        if self.pure_symbolic_student:
            return symbolic_weight * self.symbolic_part(x, temperature)
        kan = self.kan_part(x)
        return kan + symbolic_weight * self.symbolic_part(x, temperature)

    def reverse_drive(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.library.raw_from_z(x)
        voltage = raw[:, self.inputs.index(self.spec.axis_col):self.inputs.index(self.spec.axis_col) + 1]
        return torch.clamp(F.softplus(-voltage / self.reverse_scale) - LOG2, min=0.0) / self.reverse_norm

    def forward_drive(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.library.raw_from_z(x)
        voltage = raw[:, self.inputs.index(self.spec.axis_col):self.inputs.index(self.spec.axis_col) + 1]
        return torch.clamp(F.softplus(voltage / self.reverse_scale) - LOG2, min=0.0) / self.reverse_norm

    def mechanism_drives(self, x: torch.Tensor) -> torch.Tensor:
        drives = []
        reverse = self.reverse_drive(x)
        forward = self.forward_drive(x)
        ones = torch.ones_like(reverse)
        for mechanism in self.mechanisms:
            if mechanism.drive == "reverse_zero":
                drives.append(reverse)
            elif mechanism.drive == "forward_zero":
                drives.append(forward)
            elif mechanism.drive == "reverse":
                drives.append(torch.sigmoid(reverse * 4.0))
            elif mechanism.drive == "forward":
                drives.append(torch.sigmoid(forward * 4.0))
            else:
                drives.append(ones)
        return torch.cat(drives, dim=1)

    def forward(self, x: torch.Tensor, temperature: float = 1.0, symbolic_weight: float = 1.0) -> torch.Tensor:
        return self.raw_output(x, temperature=temperature, symbolic_weight=symbolic_weight)

    def complexity(self, temperature: float) -> torch.Tensor:
        if self.spec.key in {"dark_current", "photo_current"}:
            mechanism_complexity = self.gates(temperature).mean()
            residual = self.residual_gates(temperature)
            residual_denominator = torch.clamp(self.residual_term_mask.sum(), min=1.0)
            residual_complexity = residual.sum() / residual_denominator
            if self.spec.key == "photo_current" and self.generic_symbolic_terms:
                generic = self.generic_gates(temperature)
                generic_denominator = torch.clamp(self.generic_term_mask.sum(), min=1.0)
                generic_complexity = generic.sum() / generic_denominator
                return (
                    mechanism_complexity
                    + residual_complexity
                    + self.generic_complexity_multiplier * generic_complexity
                )
            return torch.cat([self.gates(temperature).reshape(-1), residual]).mean()
        return self.gates(temperature).mean()

    def binary_penalty(self, temperature: float) -> torch.Tensor:
        p = self.gates(temperature)
        if self.spec.key in {"dark_current", "photo_current"}:
            residual = self.residual_gates(temperature)
            if self.spec.key == "photo_current" and self.generic_symbolic_terms:
                residual = residual[self.residual_term_mask > 0]
                generic = self.generic_gates(temperature)[self.generic_term_mask > 0]
                p = torch.cat([p.reshape(-1), residual, generic])
            else:
                p = torch.cat([p.reshape(-1), residual])
        return (p * (1.0 - p)).mean()

    def residual_penalty(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan_part(x).pow(2).mean()

    def prune_symbolic(
        self,
        max_terms: int,
        min_probability: float,
        temperature: float = 0.2,
        max_generic_terms: int | None = None,
    ) -> None:
        with torch.no_grad():
            probabilities = torch.sigmoid(self.gate_logits / temperature)
            scores = probabilities * self.coeff.abs()
            mask = torch.zeros_like(scores)
            if self.flat_gates:
                residual_probabilities = torch.sigmoid(self.residual_gate_logits / temperature)
                residual_scores = residual_probabilities * self.residual_coeff.abs()
                generic_probabilities = torch.sigmoid(self.generic_gate_logits / temperature)
                generic_scores = generic_probabilities * self.generic_coeff.abs()
                generic_allowed = self.generic_term_mask > 0

                # Flat mode has three additive symbolic branches.  They must
                # share one budget; independently retaining max_terms in each
                # branch made a nominal 12/48-term model much denser.
                branch_scores = torch.cat(
                    [scores.reshape(-1), residual_scores, generic_scores]
                )
                branch_probabilities = torch.cat(
                    [probabilities.reshape(-1), residual_probabilities, generic_probabilities]
                )
                allowed = torch.cat(
                    [
                        self.term_mask.reshape(-1) > 0,
                        self.residual_term_mask > 0,
                        generic_allowed if self.generic_symbolic_terms else torch.zeros_like(generic_allowed),
                    ]
                )
                eligible = torch.where((branch_probabilities >= min_probability) & allowed)[0]
                if eligible.numel() == 0:
                    eligible = torch.where(allowed)[0]
                ranked = eligible[torch.argsort(branch_scores[eligible], descending=True)]
                keep = ranked[: min(max_terms, ranked.numel())]

                n_main = scores.numel()
                n_residual = residual_scores.numel()
                mask.reshape(-1)[keep[keep < n_main]] = 1.0
                residual_mask = torch.zeros_like(residual_scores)
                residual_keep = keep[(keep >= n_main) & (keep < n_main + n_residual)] - n_main
                residual_mask[residual_keep] = 1.0
                generic_mask = torch.zeros_like(self.generic_coeff)
                if self.generic_symbolic_terms:
                    generic_keep = keep[keep >= n_main + n_residual] - n_main - n_residual
                    generic_mask[generic_keep] = 1.0
                self.active_mask.copy_(mask)
                self.residual_active_mask.copy_(residual_mask)
                self.generic_active_mask.copy_(generic_mask)
                self.hard_structure = True
                return
            if self.spec.key in {"dark_current", "photo_current"}:
                # The residual is part of the deployed current formula, so it
                # must consume the same term budget as the physical mechanism
                # branches.  Previously dark current left every residual term
                # active, turning a nominal 12-term model into a 31-term one.
                divisor = scores.shape[0] + 1
                per_mechanism = max(1, max_terms // divisor)
                for mechanism_index in range(scores.shape[0]):
                    allowed = self.term_mask[mechanism_index] > 0
                    eligible = torch.where((probabilities[mechanism_index] >= min_probability) & allowed)[0]
                    keep_count = per_mechanism
                    if eligible.numel() < keep_count:
                        eligible = torch.where(allowed)[0]
                    ranked = eligible[torch.argsort(scores[mechanism_index, eligible], descending=True)]
                    keep = ranked[: min(keep_count, ranked.numel())]
                    mask[mechanism_index, keep] = 1.0
                if self.spec.key == "photo_current":
                    residual_probabilities = torch.sigmoid(self.residual_gate_logits / temperature)
                    residual_scores = residual_probabilities * self.residual_coeff.abs() * self.residual_term_mask
                    generic_probabilities = torch.sigmoid(self.generic_gate_logits / temperature)
                    generic_scores = generic_probabilities * self.generic_coeff.abs() * self.generic_term_mask
                    generic_limit = max(0, int(max_generic_terms if max_generic_terms is not None else self.max_generic_symbolic_terms)) if self.generic_symbolic_terms else 0

                    # Preserve one interpretable anchor per mechanism, then
                    # allocate every remaining slot globally.  Fixed equal
                    # per-mechanism allocation was especially harmful for
                    # photo current: low-impact loss branches consumed the
                    # same capacity as the collection branch and residual.
                    mask.zero_()
                    candidates = []
                    for mechanism_index in range(scores.shape[0]):
                        allowed = self.term_mask[mechanism_index] > 0
                        eligible = torch.where((probabilities[mechanism_index] >= min_probability) & allowed)[0]
                        if eligible.numel() == 0:
                            eligible = torch.where(allowed)[0]
                        if eligible.numel() == 0:
                            continue
                        ranked = eligible[torch.argsort(scores[mechanism_index, eligible], descending=True)]
                        anchor = int(ranked[0])
                        mask[mechanism_index, anchor] = 1.0
                        for index in ranked[1:]:
                            candidates.append((float(scores[mechanism_index, index]), "mechanism", mechanism_index, int(index)))

                    residual_allowed = self.residual_term_mask > 0
                    residual_eligible = torch.where((residual_probabilities >= min_probability) & residual_allowed)[0]
                    if residual_eligible.numel() == 0:
                        residual_eligible = torch.where(residual_allowed)[0]
                    for index in residual_eligible:
                        candidates.append((float(residual_scores[index]), "residual", 0, int(index)))

                    if generic_limit:
                        generic_allowed = self.generic_term_mask > 0
                        generic_eligible = torch.where((generic_probabilities >= min_probability) & generic_allowed)[0]
                        if generic_eligible.numel() == 0:
                            generic_eligible = torch.where(generic_allowed)[0]
                        for index in generic_eligible:
                            # Generic corrections remain eligible but pay the
                            # same relative complexity surcharge as training.
                            candidates.append((float(generic_scores[index]) / self.generic_complexity_multiplier, "generic", 0, int(index)))

                    residual_mask = torch.zeros_like(residual_scores)
                    generic_mask = torch.zeros_like(generic_scores)
                    generic_count = 0
                    remaining = max(0, max_terms - int(mask.sum().detach().cpu()))
                    # A pure photo formula needs a small additive correction
                    # for effects not represented by the two multiplicative
                    # loss factors.  Without this reservation, nearly tied
                    # surface/trap scores can consume the budget differently
                    # in each seed and make teacher fidelity unstable.
                    residual_reserve = min(self.photo_residual_min_terms, remaining)
                    if residual_reserve:
                        residual_ranked = residual_eligible[
                            torch.argsort(residual_scores[residual_eligible], descending=True)
                        ]
                        residual_keep = residual_ranked[:residual_reserve]
                        residual_mask[residual_keep] = 1.0
                        remaining -= int(residual_keep.numel())
                    branch_counts = {
                        index: int(mask[index].sum().detach().cpu())
                        for index in range(scores.shape[0])
                    }
                    for _, branch, mechanism_index, term_index in sorted(candidates, reverse=True):
                        if remaining <= 0:
                            break
                        if branch == "generic":
                            if generic_count >= generic_limit:
                                continue
                            generic_mask[term_index] = 1.0
                            generic_count += 1
                        elif branch == "residual":
                            if residual_mask[term_index] > 0:
                                continue
                            residual_mask[term_index] = 1.0
                        else:
                            role = self._photo_mechanism_role(self.mechanisms[mechanism_index].name)
                            if (
                                role in {"surface_loss", "trap_loss"}
                                and self.photo_loss_branch_max_terms is not None
                                and branch_counts[mechanism_index] >= self.photo_loss_branch_max_terms
                            ):
                                continue
                            mask[mechanism_index, term_index] = 1.0
                            branch_counts[mechanism_index] += 1
                        remaining -= 1
                    self.residual_active_mask.copy_(residual_mask)
                    self.generic_active_mask.copy_(generic_mask)
                else:
                    residual_probabilities = torch.sigmoid(self.residual_gate_logits / temperature)
                    residual_scores = residual_probabilities * self.residual_coeff.abs() * self.residual_term_mask
                    residual_allowed = self.residual_term_mask > 0
                    residual_keep_count = max(0, max_terms - int(mask.sum().detach().cpu()))
                    residual_mask = torch.zeros_like(residual_scores)
                    if residual_keep_count > 0:
                        residual_eligible = torch.where(
                            (residual_probabilities >= min_probability) & residual_allowed
                        )[0]
                        if residual_eligible.numel() < residual_keep_count:
                            residual_eligible = torch.where(residual_allowed)[0]
                        residual_ranked = residual_eligible[
                            torch.argsort(residual_scores[residual_eligible], descending=True)
                        ]
                        residual_mask[residual_ranked[: min(residual_keep_count, residual_ranked.numel())]] = 1.0
                    self.residual_active_mask.copy_(residual_mask)
                    self.generic_active_mask.zero_()
            else:
                flat_probabilities = probabilities.reshape(-1)
                flat_scores = scores.reshape(-1)
                eligible = torch.where(flat_probabilities >= min_probability)[0]
                if eligible.numel() == 0:
                    eligible = torch.arange(flat_scores.numel(), device=flat_scores.device)
                ranked = eligible[torch.argsort(flat_scores[eligible], descending=True)]
                keep = ranked[: min(max_terms, ranked.numel())]
                if keep.numel() == 0:
                    keep = torch.argmax(flat_scores).reshape(1)
                mask.reshape(-1)[keep] = 1.0
            self.active_mask.copy_(mask * self.term_mask)
            self.hard_structure = True

    def gate_table(self, temperature: float = 0.2) -> pd.DataFrame:
        with torch.no_grad():
            probabilities = torch.sigmoid(self.gate_logits / temperature).detach().cpu().numpy()
            gates = self.gates(temperature).detach().cpu().numpy()
            residual_probabilities = torch.sigmoid(self.residual_gate_logits / temperature).detach().cpu().numpy()
            residual_gates = self.residual_gates(temperature).detach().cpu().numpy()
            generic_probabilities = torch.sigmoid(self.generic_gate_logits / temperature).detach().cpu().numpy()
            generic_gates = self.generic_gates(temperature).detach().cpu().numpy()
            coeffs = self.coeff.detach().cpu().numpy()
            residual_coeffs = self.residual_coeff.detach().cpu().numpy()
            generic_coeffs = self.generic_coeff.detach().cpu().numpy()
            mechanism_scale = self.mechanism_scale.detach().cpu().numpy()
        rows = []
        if self.flat_gates:
            for term_index, term in enumerate(self.library.terms):
                rows.append(
                    {
                        "mechanism_index": 0,
                        "mechanism": "flat",
                        "photo_role": "flat",
                        "mechanism_drive": "none",
                        "mechanism_initial_sign": 1.0,
                        "mechanism_scale": 1.0,
                        "term_index": term_index,
                        "term_name": term.name,
                        "expression": term.expression,
                        "dependencies": ",".join(term.dependencies),
                        "probability": float(probabilities[0, term_index]),
                        "active_gate": float(gates[0, term_index]),
                        "coefficient_model_space": float(coeffs[0, term_index]),
                    }
                )
            return pd.DataFrame(rows)
        for mechanism_index, mechanism in enumerate(self.mechanisms):
            role = self._photo_mechanism_role(mechanism.name) if self.spec.key == "photo_current" else "mechanism"
            for term_index, term in enumerate(self.library.terms):
                rows.append(
                    {
                        "mechanism_index": mechanism_index,
                        "mechanism": mechanism.name,
                        "photo_role": role,
                        "mechanism_drive": mechanism.drive,
                        "mechanism_initial_sign": float(mechanism.sign),
                        "mechanism_scale": float(mechanism_scale[mechanism_index]),
                        "term_index": term_index,
                        "term_name": term.name,
                        "expression": term.expression,
                        "dependencies": ",".join(term.dependencies),
                        "probability": float(probabilities[mechanism_index, term_index]),
                        "active_gate": float(gates[mechanism_index, term_index]),
                        "coefficient_model_space": float(coeffs[mechanism_index, term_index]),
                    }
                )
        if self.spec.key in {"dark_current", "photo_current"}:
            for term_index, term in enumerate(self.library.terms):
                rows.append(
                    {
                        "mechanism_index": len(self.mechanisms),
                        "mechanism": "higher_order_symbolic_residual",
                        "photo_role": "residual",
                        "mechanism_drive": "all",
                        "mechanism_initial_sign": 0.0,
                        "mechanism_scale": 1.0,
                        "term_index": term_index,
                        "term_name": term.name,
                        "expression": term.expression,
                        "dependencies": ",".join(term.dependencies),
                        "probability": float(residual_probabilities[term_index]),
                        "active_gate": float(residual_gates[term_index]),
                        "coefficient_model_space": float(residual_coeffs[term_index]),
                    }
                )
            if self.spec.key == "photo_current" and self.generic_symbolic_terms:
                for term_index, term in enumerate(self.library.terms):
                    rows.append(
                        {
                            "mechanism_index": len(self.mechanisms) + 1,
                            "mechanism": "generic_symbolic_residual",
                            "photo_role": "residual",
                            "mechanism_drive": "all",
                            "mechanism_initial_sign": 0.0,
                            "mechanism_scale": 1.0,
                            "term_index": term_index,
                            "term_name": term.name,
                            "expression": term.expression,
                            "dependencies": ",".join(term.dependencies),
                            "probability": float(generic_probabilities[term_index]),
                            "active_gate": float(generic_gates[term_index]),
                            "coefficient_model_space": float(generic_coeffs[term_index]),
                        }
                    )
        return pd.DataFrame(rows).sort_values(["active_gate", "probability"], ascending=False)

    def export_pure_symbolic_formula(self, output_dir: Path) -> dict[str, object]:
        """Write an auditable, KAN-free formula for every pure symbolic student.

        Current tasks retain their trained hierarchical composition instead of
        being flattened into an unrelated additive fit.  The JSON artifact is
        deliberately structural: it records the normalized feature terms,
        gates, coefficients, mechanism drives, and loss factors needed for an
        independent numerical replay.
        """
        if not self.pure_symbolic_student:
            raise RuntimeError("Pure-symbolic export requested for a model that still uses KAN output")

        with torch.no_grad():
            gates = self.gates(temperature=0.2).detach().cpu().numpy()
            coefficients = self.coeff.detach().cpu().numpy()
            residual_gates = self.residual_gates(temperature=0.2).detach().cpu().numpy()
            residual_coefficients = self.residual_coeff.detach().cpu().numpy()
            generic_gates = self.generic_gates(temperature=0.2).detach().cpu().numpy()
            generic_coefficients = self.generic_coeff.detach().cpu().numpy()
            biases = self.bias.detach().cpu().numpy()
            residual_bias = float(self.residual_bias.detach().cpu())
            mechanism_scale = self.mechanism_scale.detach().cpu().numpy()

        def selected_terms(branch_gates, branch_coefficients):
            rows = []
            for index, term in enumerate(self.library.terms):
                gate = float(branch_gates[index])
                if gate <= 0.0:
                    continue
                rows.append(
                    {
                        "term_index": index,
                        "term": term.name,
                        "expression": term.expression,
                        "normalized_expression": (
                            f"(({term.expression}) - {self.library.centers[index]:.16g})"
                            f" / {self.library.scales[index]:.16g}"
                        ),
                        "coefficient_normalized_output": float(branch_coefficients[index] * gate),
                        "dependencies": list(term.dependencies),
                    }
                )
            return rows

        def linear_formula(bias, rows):
            pieces = [f"{float(bias):.16g}"]
            pieces.extend(
                f"({row['coefficient_normalized_output']:.16g})*({row['normalized_expression']})"
                for row in rows
            )
            return " + ".join(pieces)

        branches = []
        if self.flat_gates:
            main_terms = selected_terms(gates[0], coefficients[0])
            residual_terms = selected_terms(residual_gates, residual_coefficients)
            generic_terms = selected_terms(generic_gates, generic_coefficients)
            normalized_formula = " + ".join(
                [linear_formula(biases[0], main_terms), linear_formula(residual_bias, residual_terms)]
                + ([linear_formula(0.0, generic_terms)] if generic_terms else [])
            )
            branches.append({"name": "flat", "bias": float(biases[0]), "terms": main_terms})
            residual_branch = {"bias": residual_bias, "terms": residual_terms}
            generic_branch = {"bias": 0.0, "terms": generic_terms}
        else:
            for mechanism_index, mechanism in enumerate(self.mechanisms):
                rows = selected_terms(gates[mechanism_index], coefficients[mechanism_index])
                branches.append(
                    {
                        "name": mechanism.name,
                        "role": self._photo_mechanism_role(mechanism.name),
                        "drive": mechanism.drive,
                        "sign": float(mechanism.sign),
                        "scale": float(mechanism_scale[mechanism_index]),
                        "bias": float(biases[mechanism_index]),
                        "terms": rows,
                    }
                )
            residual_branch = {
                "bias": residual_bias,
                "terms": selected_terms(residual_gates, residual_coefficients),
            }
            generic_branch = {
                "bias": 0.0,
                "terms": selected_terms(generic_gates, generic_coefficients),
            }
            if self.spec.key not in {"dark_current", "photo_current"}:
                # These two branches are intentionally unused by the
                # non-current forward path; do not advertise dead terms in a
                # deployable formula.
                residual_branch = {"bias": 0.0, "terms": []}
                generic_branch = {"bias": 0.0, "terms": []}
                normalized_formula = linear_formula(branches[0]["bias"], branches[0]["terms"])
            elif self.spec.key == "dark_current":
                mechanism_formulas = []
                for branch in branches:
                    drive = _symbolic_drive_expression(branch["drive"], self.spec.axis_col, self.reverse_scale, self.reverse_norm)
                    mechanism_formulas.append(
                        f"({branch['scale']:.16g})*softplus({linear_formula(branch['bias'], branch['terms'])})*({drive})"
                    )
                normalized_formula = " + ".join(
                    mechanism_formulas + [linear_formula(residual_branch["bias"], residual_branch["terms"]), linear_formula(0.0, generic_branch["terms"])]
                )
            else:
                def mechanism_expression(branch):
                    drive = _symbolic_drive_expression(branch["drive"], self.spec.axis_col, self.reverse_scale, self.reverse_norm)
                    return (
                        f"abs({branch['scale']:.16g})*softplus({linear_formula(branch['bias'], branch['terms'])})"
                        f"*({drive})"
                    )

                main = [f"({branch['sign']:.16g})*({mechanism_expression(branch)})" for branch in branches if branch["role"] == "main"]
                surface = [mechanism_expression(branch) for branch in branches if branch["role"] == "surface_loss"]
                trap = [mechanism_expression(branch) for branch in branches if branch["role"] == "trap_loss"]
                additive = [f"({branch['sign']:.16g})*({mechanism_expression(branch)})" for branch in branches if branch["role"] == "additive"]
                normalized_formula = (
                    f"({' + '.join(main) if main else '0'})"
                    f"*exp(-clip({' + '.join(surface) if surface else '0'}, 0, 8))"
                    f"*exp(-clip({' + '.join(trap) if trap else '0'}, 0, 8))"
                    f" + ({' + '.join(additive) if additive else '0'})"
                    f" + ({linear_formula(residual_branch['bias'], residual_branch['terms'])})"
                    f" + ({linear_formula(0.0, generic_branch['terms'])})"
                )

        model_space_formula = f"{self.y_scale:.16g} * ({normalized_formula})"
        physical_formula = (
            f"10**({model_space_formula})"
            if self.spec.use_log_transform
            else model_space_formula
        )
        scale_map = {
            name: {"mean": float(self.library.x_mean[index]), "scale": float(self.library.x_scale[index])}
            for index, name in enumerate(self.inputs)
        }
        selected = [
            row
            for branch in branches
            for row in branch["terms"]
        ] + residual_branch["terms"] + generic_branch["terms"]
        payload = {
            "kind": "pure_symbolic_student",
            "task_key": self.spec.key,
            "task": self.spec.name,
            "uses_kan_at_inference": False,
            "output_space": "log10" if self.spec.use_log_transform else "linear",
            "output_scale": float(self.y_scale),
            "normalized_feature_definition": "scaled(x) = (x - mean[x]) / scale[x]",
            "input_scalers": scale_map,
            "feature_centers": self.library.centers.tolist(),
            "feature_scales": self.library.scales.tolist(),
            "generic_symbolic_terms": bool(self.generic_symbolic_terms),
            "flat_gates": bool(self.flat_gates),
            "reverse_scale": float(self.reverse_scale),
            "reverse_norm": float(self.reverse_norm),
            "branches": branches,
            "residual_branch": residual_branch,
            "generic_branch": generic_branch,
            "selected_term_count": int(sum(len(branch["terms"]) for branch in branches) + len(residual_branch["terms"]) + len(generic_branch["terms"])),
            "selected_terms": selected,
            "model_space_formula": model_space_formula,
            "formula": physical_formula,
        }
        json_path = output_dir / "pure_symbolic_formula.json"
        text_path = output_dir / "pure_symbolic_formula.txt"
        structured_path = output_dir / "pure_symbolic_formula_structured.txt"

        # The expanded formula is convenient for auditing but repeats the
        # same normalized physical feature in every mechanism branch.  Emit a
        # CSE-style deployment view as well: each phi_i is evaluated once and
        # branches consume only its scalar value.  This is the relevant cost
        # representation for a compact-model or Verilog-A backend.
        unique_terms = {}
        for row in selected:
            unique_terms.setdefault(int(row["term_index"]), row)

        def structured_linear(branch):
            pieces = [f"{float(branch['bias']):.16g}"]
            pieces.extend(
                f"({row['coefficient_normalized_output']:.16g})*phi_{int(row['term_index'])}"
                for row in branch["terms"]
            )
            return " + ".join(pieces)

        structured_lines = [
            "Pure symbolic student: structured deployment graph",
            "=" * 50,
            "KAN contribution at inference: 0",
            f"Unique normalized features: {len(unique_terms)}; branch coefficients: {len(selected)}",
            "",
            "# Evaluate each normalized feature once",
        ]
        structured_lines.extend(
            f"phi_{index} = {row['normalized_expression']}  # {row['term']}"
            for index, row in sorted(unique_terms.items())
        )
        structured_lines.append("\n# Branch graph")
        if self.flat_gates:
            structured_lines.append(f"flat = {structured_linear(branches[0])}")
            structured_lines.append(f"residual = {structured_linear(residual_branch)}")
            if generic_branch["terms"]:
                structured_lines.append(f"generic = {structured_linear(generic_branch)}")
                structured_lines.append("normalized_output = flat + residual + generic")
            else:
                structured_lines.append("normalized_output = flat + residual")
        elif self.spec.key == "photo_current":
            branch_names = {}
            for branch in branches:
                label = branch["name"]
                drive = _symbolic_drive_expression(branch["drive"], self.spec.axis_col, self.reverse_scale, self.reverse_norm)
                expression = (
                    f"abs({branch['scale']:.16g})*softplus({structured_linear(branch)})*({drive})"
                )
                branch_names[label] = expression
                structured_lines.append(f"{label} = {expression}")
            main = " + ".join(
                f"({branch['sign']:.16g})*{branch['name']}"
                for branch in branches if branch["role"] == "main"
            ) or "0"
            surface = " + ".join(branch["name"] for branch in branches if branch["role"] == "surface_loss") or "0"
            trap = " + ".join(branch["name"] for branch in branches if branch["role"] == "trap_loss") or "0"
            additive = " + ".join(
                f"({branch['sign']:.16g})*{branch['name']}"
                for branch in branches if branch["role"] == "additive"
            ) or "0"
            structured_lines.append(f"residual = {structured_linear(residual_branch)}")
            if generic_branch["terms"]:
                structured_lines.append(f"generic = {structured_linear(generic_branch)}")
            structured_lines.append(
                f"normalized_output = ({main})*exp(-clip({surface},0,8))*exp(-clip({trap},0,8)) + ({additive}) + residual"
                + (" + generic" if generic_branch["terms"] else "")
            )
        else:
            structured_lines.append(f"normalized_output = {structured_linear(branches[0])}")
        structured_lines.append(f"model_space_output = {self.y_scale:.16g} * normalized_output")
        if self.spec.use_log_transform:
            structured_lines.append("physical_output = 10**(model_space_output)")
        else:
            structured_lines.append("physical_output = model_space_output")
        payload["deployment_cost"] = {
            "unique_normalized_features": len(unique_terms),
            "branch_coefficients": len(selected),
            "branches": len(branches),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        text_path.write_text(
            "Pure symbolic student formula\n" + "=" * 30 + "\n\n"
            + "KAN contribution at inference: 0\n"
            + "scaled(x) = (x - mean[x]) / scale[x]\n"
            + (
                f"Formula in model space:\nlog10({self.spec.target_col}) = {model_space_formula}\n\n"
                f"Formula in physical space:\n{self.spec.target_col} = {physical_formula}\n\n"
                if self.spec.use_log_transform
                else f"Output formula:\n{self.spec.target_col} = {physical_formula}\n\n"
            )
            + "Input scalers:\n"
            + "\n".join(
                f"  {name}: mean={values['mean']:.16g}, scale={values['scale']:.16g}"
                for name, values in scale_map.items()
            )
            + "\n\nSelected terms:\n"
            + "\n".join(
                f"  {row['coefficient_normalized_output']:.16g} * {row['expression']}"
                for row in selected
            )
            + "\n",
            encoding="utf-8",
        )
        structured_path.write_text("\n".join(structured_lines) + "\n", encoding="utf-8")
        return {
            "json": str(json_path), "text": str(text_path), "structured": str(structured_path),
            "selected_term_count": len(selected),
            "unique_feature_count": len(unique_terms),
        }


def evaluate_exported_pure_symbolic_formula(
    payload: dict[str, object],
    x_raw: np.ndarray,
) -> np.ndarray:
    """Numerically replay a pure-symbolic JSON export without a trained model.

    The replay reconstructs only the public feature library and the serialized
    formula graph.  It never reads a KAN weight or gate from the live model.
    This makes it suitable for export verification and external integration
    tests while preserving the exact feature normalization used at training.
    """
    task_key = str(payload["task_key"])
    spec = TASKS[task_key]
    inputs = tuple(payload["input_scalers"].keys())
    raw = np.asarray(x_raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(inputs):
        raise ValueError(f"Expected raw inputs with shape (n, {len(inputs)}), got {raw.shape}")
    scalers = payload["input_scalers"]
    x_mean = np.asarray([scalers[name]["mean"] for name in inputs], dtype=np.float64)
    x_scale = np.asarray([scalers[name]["scale"] for name in inputs], dtype=np.float64)
    x_scale = safe_scale(x_scale, eps=EPS)
    sensitivity = pd.DataFrame({"input": inputs, "normalized_sensitivity": np.ones(len(inputs))})
    # Feature scalers are overwritten immediately.  A single valid row is
    # sufficient to reconstruct the task-dependent term catalogue.
    library = PhysicalFeatureLibrary(
        spec=spec,
        inputs=inputs,
        x_mean=x_mean,
        x_scale=x_scale,
        train_x_raw=raw[:1],
        sensitivity=sensitivity,
        generic_symbolic_terms=bool(payload.get("generic_symbolic_terms", False)),
    )
    library.centers = np.asarray(payload["feature_centers"], dtype=np.float64)
    library.scales = np.asarray(payload["feature_scales"], dtype=np.float64)
    if len(library.terms) != len(library.centers):
        raise ValueError("Exported feature normalizers do not match the reconstructed feature library")
    z = torch.as_tensor((raw - x_mean) / x_scale, dtype=torch.float32)
    with torch.no_grad():
        phi = library.matrix(z).detach().cpu().numpy().astype(np.float64)

    def linear(branch: dict[str, object]) -> np.ndarray:
        out = np.full(len(raw), float(branch["bias"]), dtype=np.float64)
        for term in branch.get("terms", []):
            out += float(term["coefficient_normalized_output"]) * phi[:, int(term["term_index"])]
        return out

    branches = list(payload["branches"])
    residual = linear(payload["residual_branch"])
    generic = linear(payload["generic_branch"])
    if bool(payload["flat_gates"]):
        normalized = linear(branches[0]) + residual + generic
    elif spec.key not in {"dark_current", "photo_current"}:
        normalized = linear(branches[0])
    else:
        axis = raw[:, inputs.index(spec.axis_col)]
        reverse_scale = float(payload["reverse_scale"])
        reverse_norm = float(payload["reverse_norm"])
        reverse = np.maximum(np.logaddexp(0.0, -axis / reverse_scale) - LOG2, 0.0) / reverse_norm
        forward = np.maximum(np.logaddexp(0.0, axis / reverse_scale) - LOG2, 0.0) / reverse_norm

        def drive(name: str) -> np.ndarray:
            if name == "reverse_zero":
                return reverse
            if name == "forward_zero":
                return forward
            if name == "reverse":
                return 1.0 / (1.0 + np.exp(-4.0 * reverse))
            if name == "forward":
                return 1.0 / (1.0 + np.exp(-4.0 * forward))
            return np.ones(len(raw), dtype=np.float64)

        values = []
        for branch in branches:
            amplitude = np.logaddexp(0.0, linear(branch))
            values.append((branch, amplitude * drive(str(branch["drive"]))))
        if task_key == "dark_current":
            normalized = sum(float(branch["scale"]) * value for branch, value in values) + residual + generic
        else:
            main = sum(float(branch["sign"]) * abs(float(branch["scale"])) * value for branch, value in values if branch["role"] == "main")
            surface = sum(abs(float(branch["scale"])) * value for branch, value in values if branch["role"] == "surface_loss")
            trap = sum(abs(float(branch["scale"])) * value for branch, value in values if branch["role"] == "trap_loss")
            additive = sum(float(branch["sign"]) * abs(float(branch["scale"])) * value for branch, value in values if branch["role"] == "additive")
            normalized = main * np.exp(-np.clip(surface, 0.0, 8.0)) * np.exp(-np.clip(trap, 0.0, 8.0)) + additive + residual + generic
    return normalized * float(payload["output_scale"])


def _tensorize(
    frame: pd.DataFrame,
    spec: TaskSpec,
    inputs: tuple[str, ...],
    x_scaler: StandardScaler,
    y_scale: float,
    device: torch.device,
    derivative_scale: float,
    weights: np.ndarray | None = None,
    teacher: np.ndarray | None = None,
) -> TrainTensors:
    frame = without_attrs(frame)
    x = x_scaler.transform(frame[list(inputs)].to_numpy(dtype=np.float64)).astype(np.float32)
    y = (target_values(frame, spec) / y_scale).reshape(-1, 1).astype(np.float32)
    dy = None
    try:
        dy_values = finite_difference_derivative(frame, spec, inputs)
        dy = (dy_values * derivative_scale / y_scale).reshape(-1, 1).astype(np.float32)
    except ValueError:
        dy = None
    curve_id = None
    if spec.key == "photo_current":
        if "_curve_id" in frame.columns:
            labels = frame["_curve_id"].astype(str)
        else:
            condition_cols = [column for column in inputs if column != spec.axis_col]
            labels = frame[list(condition_cols)].astype(str).agg("|".join, axis=1)
        curve_id = pd.factorize(labels, sort=False)[0].astype(np.int64)
    weight_tensor = None
    if weights is not None:
        weight_values = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        if len(weight_values) != len(frame):
            raise ValueError(f"Expected {len(frame)} Bayesian weights, got {len(weight_values)}")
        if not np.all(np.isfinite(weight_values)) or np.any(weight_values <= 0):
            raise ValueError("Bayesian weights must be finite positive values")
        weight_tensor = torch.from_numpy(weight_values).to(device)
    teacher_tensor = None
    if teacher is not None:
        teacher_values = (np.asarray(teacher, dtype=np.float64).reshape(-1, 1) / y_scale).astype(np.float32)
        if len(teacher_values) != len(frame):
            raise ValueError(f"Expected {len(frame)} Bayesian teacher targets, got {len(teacher_values)}")
        if not np.all(np.isfinite(teacher_values)):
            raise ValueError("Bayesian teacher targets must be finite values")
        teacher_tensor = torch.from_numpy(teacher_values).to(device)
    return TrainTensors(
        x=torch.from_numpy(x).to(device),
        y=torch.from_numpy(y).to(device),
        dy=torch.from_numpy(dy).to(device) if dy is not None else None,
        curve_id=torch.from_numpy(curve_id).to(device) if curve_id is not None else None,
        weight=weight_tensor,
        teacher_y=teacher_tensor,
    )


def _curve_shape_loss(pred: torch.Tensor, target: torch.Tensor, curve_id: torch.Tensor | None) -> torch.Tensor:
    if curve_id is None:
        return torch.zeros((), device=pred.device)
    losses = []
    for identifier in torch.unique(curve_id):
        mask = curve_id == identifier
        if int(mask.sum().detach().cpu()) < 2:
            continue
        pred_curve = pred[mask]
        target_curve = target[mask]
        mean_loss = (pred_curve.mean() - target_curve.mean()).pow(2)
        swing_loss = ((pred_curve.max() - pred_curve.min()) - (target_curve.max() - target_curve.min())).pow(2)
        losses.append(mean_loss + swing_loss)
    if not losses:
        return torch.zeros((), device=pred.device)
    return torch.stack(losses).mean()


def _loss(
    model: SymbolicGatedKAN,
    tensors: TrainTensors,
    stage: StageConfig,
    temperature: float,
    training: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    derivative_weight = float(getattr(model, "derivative_loss_weight", 0.05))
    need_derivative = tensors.dy is not None and derivative_weight > 0.0
    x = tensors.x.detach().clone().requires_grad_(need_derivative)
    pred = model(x, temperature=temperature, symbolic_weight=stage.symbolic_weight)
    value_beta = float(getattr(model, "symbolic_loss_beta", 0.05))
    if value_beta <= 0.0:
        raise ValueError("symbolic_loss_beta must be positive")
    if tensors.weight is None:
        value = F.smooth_l1_loss(pred, tensors.y, beta=value_beta)
    else:
        value_per_sample = F.smooth_l1_loss(pred, tensors.y, beta=value_beta, reduction="none")
        value = (value_per_sample * tensors.weight).sum() / tensors.weight.sum()
    distillation = torch.zeros((), device=x.device)
    distill_weight = float(getattr(model, "bayesian_distill_weight", 0.0))
    if tensors.teacher_y is not None and distill_weight > 0.0:
        distill_per_sample = F.smooth_l1_loss(pred, tensors.teacher_y, beta=value_beta, reduction="none")
        if tensors.weight is None:
            distillation = distill_per_sample.mean()
        else:
            distillation = (distill_per_sample * tensors.weight).sum() / tensors.weight.sum()
    curve_shape = _curve_shape_loss(pred, tensors.y, tensors.curve_id) if model.spec.key == "photo_current" else torch.zeros((), device=x.device)
    derivative = torch.zeros((), device=x.device)
    if need_derivative:
        grad = torch.autograd.grad(
            pred.sum(),
            x,
            create_graph=training,
            retain_graph=training,
        )[0][:, [model.inputs.index(model.spec.axis_col)]]
        if tensors.weight is None:
            derivative = F.smooth_l1_loss(grad, tensors.dy, beta=0.05)
        else:
            derivative_per_sample = F.smooth_l1_loss(grad, tensors.dy, beta=0.05, reduction="none")
            derivative = (derivative_per_sample * tensors.weight).sum() / tensors.weight.sum()
    residual = model.residual_penalty(tensors.x)
    complexity = model.complexity(temperature)
    binary = model.binary_penalty(temperature)
    observed_weight = float(getattr(model, "observed_loss_weight", 1.0))
    total = (
        observed_weight * value
        + distill_weight * distillation
        + derivative_weight * derivative
        + float(getattr(model, "curve_shape_loss_weight", 0.02)) * curve_shape
        + stage.residual_weight * residual
        + stage.complexity_weight * complexity
        + stage.binary_weight * binary
    )
    return total, {
        "total": float(total.detach().cpu()),
        "value": float(value.detach().cpu()),
        "observed_weight": observed_weight,
        "bayesian_distillation": float(distillation.detach().cpu()),
        "derivative": float(derivative.detach().cpu()),
        "curve_shape": float(curve_shape.detach().cpu()),
        "kan_residual": float(residual.detach().cpu()),
        "complexity": float(complexity.detach().cpu()),
        "binary": float(binary.detach().cpu()),
    }


def _train_stage(
    model: SymbolicGatedKAN,
    train: TrainTensors,
    validation: TrainTensors,
    stage: StageConfig,
    val_freq: int,
    max_gradient_norm: float,
) -> pd.DataFrame:
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=stage.learning_rate)
    rows = []
    best_state = None
    best_score = float("inf")
    for step in range(1, stage.steps + 1):
        temperature = _temperature(step - 1, stage.steps, stage.temperature_start, stage.temperature_end)
        model.train()
        optimizer.zero_grad()
        loss, metrics = _loss(model, train, stage, temperature, training=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in stage {stage.name}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        if step == 1 or step % val_freq == 0 or step == stage.steps:
            model.eval()
            val_loss, val_metrics = _loss(model, validation, stage, temperature, training=False)
            score = float(val_loss.detach().cpu())
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            rows.append(
                {
                    "stage": stage.name,
                    "step": step,
                    "temperature": temperature,
                    **{f"train_{key}": value for key, value in metrics.items()},
                    **{f"validation_{key}": value for key, value in val_metrics.items()},
                    "active_terms": int((model.gates(temperature) > 0).sum().detach().cpu()),
                }
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def _predict(
    model: SymbolicGatedKAN,
    tensors: TrainTensors,
    y_scale: float,
    symbolic_weight: float = 1.0,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(
            tensors.x,
            temperature=0.1,
            symbolic_weight=symbolic_weight,
        ).detach().cpu().numpy().reshape(-1)
    return pred * y_scale


def _select_symbolic_weight(
    model: SymbolicGatedKAN,
    validation: TrainTensors,
    candidates: tuple[float, ...],
) -> float:
    """Choose the symbolic residual strength using validation data only."""
    model.eval()
    best_weight = float(candidates[0])
    best_mse = float("inf")
    with torch.no_grad():
        for weight in candidates:
            prediction = model(validation.x, temperature=0.1, symbolic_weight=float(weight))
            mse = float(torch.mean((prediction - validation.y).pow(2)).detach().cpu())
            if mse < best_mse:
                best_mse = mse
                best_weight = float(weight)
    return best_weight


def _plot_outputs(
    result_dir: Path,
    history: pd.DataFrame,
    gates: pd.DataFrame,
    predictions: pd.DataFrame,
    spec: TaskSpec,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for stage, group in history.groupby("stage", sort=False):
        ax.plot(group.index, group["validation_total"], label=stage)
    ax.set_xlabel("Logged step")
    ax.set_ylabel("Validation objective")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "training_history.png", dpi=220)
    plt.close(fig)

    top = gates.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top))))
    ax.barh(top["term_name"], top["active_gate"])
    ax.set_xlabel("Gate value")
    ax.set_title("Active physical-symbol terms")
    fig.tight_layout()
    fig.savefig(result_dir / "symbolic_gates.png", dpi=220)
    plt.close(fig)

    test = predictions[predictions["split"] == "test"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test["actual"], test["prediction"], s=18, alpha=0.65)
    lo = min(test["actual"].min(), test["prediction"].min())
    hi = max(test["actual"].max(), test["prediction"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel(f"Actual {spec.target_col}")
    ax.set_ylabel(f"Predicted {spec.target_col}")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(result_dir / "test_prediction_scatter.png", dpi=220)
    plt.close(fig)

    if spec.axis_col in predictions.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sample = test.sort_values(spec.axis_col)
        ax.scatter(sample[spec.axis_col], sample["actual"], s=16, alpha=0.55, label="actual")
        ax.scatter(sample[spec.axis_col], sample["prediction"], s=16, alpha=0.55, label="prediction")
        ax.set_xlabel(spec.axis_col)
        ax.set_ylabel(spec.target_col)
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(result_dir / "test_axis_prediction.png", dpi=220)
        plt.close(fig)


def train_symbolic_gated(
    spec: TaskSpec,
    inputs: tuple[str, ...],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    args,
    device: torch.device,
    output_dir: Path,
    sample_weights: dict[str, np.ndarray] | None = None,
    uncertainty: dict[str, np.ndarray] | None = None,
    teacher_targets: dict[str, np.ndarray] | None = None,
    kan_init_state: dict[str, torch.Tensor] | None = None,
    kan_adapter: dict[str, np.ndarray | float] | None = None,
    zero_symbolic_init: bool = False,
    freeze_kan: bool = False,
) -> dict[str, float]:
    set_seed(args.seed)
    result_dir = output_dir / f"{spec.key}_symbolic_gated_kan"
    result_dir.mkdir(parents=True, exist_ok=True)

    x_scaler = StandardScaler()
    x_train_raw = train[list(inputs)].to_numpy(dtype=np.float64)
    x_scaler.fit(x_train_raw)
    x_scaler.scale_ = safe_scale(x_scaler.scale_, eps=EPS)
    y_train = target_values(train, spec)
    y_scale = max(float(np.max(np.abs(y_train))), TARGET_SCALE_EPS)
    axis_index = list(inputs).index(spec.axis_col)
    derivative_scale = 1.0 / float(x_scaler.scale_[axis_index])

    coeff_init = None
    if bool(getattr(args, "ridge_init", False)):
        from sklearn.linear_model import Ridge as RidgeReg
        from device_modeling.photodetector.physical_library import PhysicalFeatureLibrary as PFL
        from device_modeling.photodetector.symbolic_gated_kan import _sensitivity_table
        temp_lib = PFL(
            spec, inputs, x_scaler.mean_, x_scaler.scale_,
            x_train_raw,
            _sensitivity_table(x_train_raw, y_train, inputs),
            generic_symbolic_terms=bool(getattr(args, "generic_symbolic_terms", False)),
        )
        z = ((x_train_raw - x_scaler.mean_) / x_scaler.scale_).astype(np.float32)
        phi = temp_lib.matrix(torch.tensor(z, device=device)).detach().cpu().numpy()
        ridge = RidgeReg(alpha=1.0, fit_intercept=False)
        ridge.fit(phi, y_train)
        coeff_init = ridge.coef_.astype(np.float64)
        print(f"[Ridge init] fitted {len(coeff_init)} coefficients, "
              f"R2={ridge.score(phi, y_train):.4f}")

    model = SymbolicGatedKAN(
        spec=spec,
        inputs=inputs,
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_scale=y_scale,
        train_x_raw=x_train_raw,
        train_y=y_train,
        width=args.width,
        grid=args.grid,
        spline_order=args.spline_order,
        seed=args.seed,
        device=device,
        generic_symbolic_terms=bool(getattr(args, "generic_symbolic_terms", False)),
        max_generic_symbolic_terms=int(getattr(args, "max_generic_symbolic_terms", 4)),
        generic_complexity_multiplier=float(getattr(args, "generic_complexity_multiplier", 5.0)),
        disabled_photo_mechanisms=tuple(getattr(args, "disable_photo_mechanisms", []) or []),
        photo_coupling_mode=str(getattr(args, "photo_coupling_mode", "separate")),
        photo_loss_branch_max_terms=getattr(args, "photo_loss_branch_max_terms", None),
        photo_residual_min_terms=int(getattr(args, "photo_residual_min_terms", 0)),
        flat_gates=bool(getattr(args, "flat_gates", False)),
        coeff_init=coeff_init,
        pure_symbolic_student=bool(getattr(args, "pure_symbolic_student", False)),
    ).to(device)
    if kan_init_state is not None:
        model.kan.load_state_dict(kan_init_state, strict=True)
    if kan_adapter is not None:
        model.configure_kan_adapter(kan_adapter)
    if zero_symbolic_init:
        with torch.no_grad():
            model.coeff.zero_()
            model.residual_coeff.zero_()
            model.generic_coeff.zero_()
            model.bias.zero_()
            model.residual_bias.zero_()
    if freeze_kan or model.pure_symbolic_student:
        for parameter in model.kan.parameters():
            parameter.requires_grad_(False)

    sample_weights = sample_weights or {}
    uncertainty = uncertainty or {}
    teacher_targets = teacher_targets or {}
    model.bayesian_distill_weight = float(getattr(args, "bayesian_distill_weight", 0.0))
    # A value of zero turns this into strict posterior-mean distillation.  It
    # is useful when comparing an extractor against post-hoc conversion of
    # the same teacher, whereas a positive value deliberately performs joint
    # TCAD/teacher fitting and is reported as such.
    model.observed_loss_weight = float(getattr(args, "observed_loss_weight", 1.0))
    if model.observed_loss_weight < 0.0:
        raise ValueError("observed_loss_weight must be non-negative")
    model.symbolic_loss_beta = float(getattr(args, "symbolic_loss_beta", 0.05))
    # These were previously hard-coded in _loss.  Keeping them configurable
    # makes it possible to evaluate whether derivative/curve regularization
    # helps symbolic fidelity rather than silently conflating it with gating.
    model.derivative_loss_weight = float(getattr(args, "derivative_loss_weight", 0.05))
    model.curve_shape_loss_weight = float(getattr(args, "curve_shape_loss_weight", 0.02))
    train_t = _tensorize(
        train,
        spec,
        inputs,
        x_scaler,
        y_scale,
        device,
        derivative_scale,
        sample_weights.get("train"),
        teacher_targets.get("train"),
    )
    val_t = _tensorize(
        validation,
        spec,
        inputs,
        x_scaler,
        y_scale,
        device,
        derivative_scale,
        sample_weights.get("validation"),
        teacher_targets.get("validation"),
    )
    test_t = _tensorize(
        test,
        spec,
        inputs,
        x_scaler,
        y_scale,
        device,
        derivative_scale,
        sample_weights.get("test"),
        teacher_targets.get("test"),
    )

    stage1 = StageConfig(
        "symbolic_warmup" if model.pure_symbolic_student else "numerical_stability",
        args.stage1_steps,
        args.learning_rate,
        # A pure student has no KAN branch to warm up.  It must learn its
        # symbolic coefficients from the first optimization stage.
        symbolic_weight=1.0 if model.pure_symbolic_student else 0.0,
        residual_weight=0.0,
        complexity_weight=0.0,
        binary_weight=0.0,
        temperature_start=2.0,
        temperature_end=2.0,
    )
    stage2 = StageConfig(
        "symbolic_competition",
        args.stage2_steps,
        args.learning_rate,
        symbolic_weight=1.0,
        residual_weight=args.residual_weight,
        complexity_weight=args.complexity_weight,
        binary_weight=args.binary_gate_weight,
        temperature_start=2.0,
        temperature_end=0.35,
    )
    stage3 = StageConfig(
        "structure_freeze",
        args.stage3_steps,
        args.learning_rate * 0.5,
        symbolic_weight=1.0,
        residual_weight=args.residual_weight * 5.0,
        complexity_weight=0.0,
        binary_weight=0.0,
        temperature_start=0.2,
        temperature_end=0.2,
    )

    histories = [
        _train_stage(model, train_t, val_t, stage1, args.val_freq, args.max_gradient_norm),
        _train_stage(model, train_t, val_t, stage2, args.val_freq, args.max_gradient_norm),
    ]
    model.prune_symbolic(
        args.max_symbolic_terms,
        args.min_symbolic_probability,
        max_generic_terms=getattr(args, "max_generic_symbolic_terms", None),
    )
    histories.append(_train_stage(model, train_t, val_t, stage3, args.val_freq, args.max_gradient_norm))
    history = concat_without_attrs(histories)

    candidates = (1.0,) if model.pure_symbolic_student else tuple(
        float(value) for value in getattr(args, "symbolic_weight_candidates", (1.0,))
    )
    if not candidates or any(value < 0.0 or value > 1.0 for value in candidates):
        raise ValueError("symbolic_weight_candidates must be non-empty values in [0, 1].")
    selected_symbolic_weight = _select_symbolic_weight(model, val_t, candidates)

    prediction_frames = []
    split_tensors = {"train": train_t, "validation": val_t, "test": test_t}
    for split_name, frame in (("train", train), ("validation", validation), ("test", test)):
        prediction = _predict(
            model,
            split_tensors[split_name],
            y_scale,
            symbolic_weight=selected_symbolic_weight,
        )
        actual = target_values(frame, spec)
        out = without_attrs(frame)
        out["split"] = split_name
        out["actual"] = actual
        out["prediction"] = prediction
        out["abs_error"] = np.abs(prediction - actual)
        if split_name in sample_weights:
            out["bayesian_precision_weight"] = sample_weights[split_name]
        if split_name in uncertainty:
            out["bayesian_model_std"] = uncertainty[split_name]
        if split_name in teacher_targets:
            out["bayesian_teacher_mean"] = teacher_targets[split_name]
        prediction_frames.append(out)
    predictions = concat_without_attrs(prediction_frames)
    test_pred = predictions[predictions["split"] == "test"]
    metrics = {
        "test_rmse": float(np.sqrt(mean_squared_error(test_pred["actual"], test_pred["prediction"]))),
        "test_mae": float(mean_absolute_error(test_pred["actual"], test_pred["prediction"])),
        "test_r2": float(r2_score(test_pred["actual"], test_pred["prediction"])),
        "active_symbolic_terms": float(model.gate_table()["active_gate"].gt(0).sum()),
        "selected_symbolic_weight": selected_symbolic_weight,
        "y_scale": float(y_scale),
        "pure_symbolic_student": bool(model.pure_symbolic_student),
        "kan_contribution_at_inference": 0.0 if model.pure_symbolic_student else 1.0,
    }
    if sample_weights:
        all_weights = np.concatenate([np.asarray(values, dtype=np.float64) for values in sample_weights.values()])
        metrics.update(
            {
                "bayesian_weight_min": float(all_weights.min()),
                "bayesian_weight_median": float(np.median(all_weights)),
                "bayesian_weight_max": float(all_weights.max()),
            }
        )
    if teacher_targets:
        metrics["bayesian_distill_weight"] = float(getattr(args, "bayesian_distill_weight", 0.0))
        teacher_test = np.asarray(teacher_targets.get("test", []), dtype=np.float64).reshape(-1)
        if len(teacher_test) == len(test_pred):
            teacher_test = teacher_test
            prediction = test_pred["prediction"].to_numpy(dtype=np.float64)
            metrics["teacher_rmse"] = float(np.sqrt(mean_squared_error(teacher_test, prediction)))
            metrics["teacher_mae"] = float(mean_absolute_error(teacher_test, prediction))
            metrics["teacher_r2"] = float(r2_score(teacher_test, prediction))

    gates = model.gate_table()
    history.to_csv(result_dir / "training_history.csv", index=False)
    gates.to_csv(result_dir / "symbolic_gates.csv", index=False)
    predictions.to_csv(result_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(result_dir / "metrics.csv", index=False)
    _plot_outputs(result_dir, history, gates, predictions, spec)

    torch.save(
        {
            "model_state": model.state_dict(),
            "inputs": list(inputs),
            "target_col": spec.target_col,
            "axis_col": spec.axis_col,
            "x_mean": x_scaler.mean_,
            "x_scale": x_scaler.scale_,
            "y_scale": y_scale,
            "gate_table": gates.to_dict(orient="records"),
            "config": {
                "width": args.width,
                "grid": args.grid,
                "spline_order": args.spline_order,
                "stage1_steps": args.stage1_steps,
                "stage2_steps": args.stage2_steps,
                "stage3_steps": args.stage3_steps,
                "bayesian_precision_weighted": bool(sample_weights),
                "bayesian_distill_weight": float(getattr(args, "bayesian_distill_weight", 0.0)),
                "observed_loss_weight": float(getattr(args, "observed_loss_weight", 1.0)),
                "pure_symbolic_student": bool(model.pure_symbolic_student),
            },
        },
        result_dir / "model_checkpoint.pt",
    )
    serialized_config = {key: str(value) for key, value in vars(args).items()}
    serialized_config.update(
        {
            "zero_symbolic_init": str(bool(zero_symbolic_init)),
            "freeze_kan": str(bool(freeze_kan or model.pure_symbolic_student)),
            "optimizer": "Adam with PyTorch defaults; fresh optimizer per stage",
            "fit_partition_role": "caller-provided fitting rows",
            "validation_role": "best objective checkpoint within every stage; strict '<' tie rule; symbolic-weight selection",
            "test_role": "one-time evaluation after structure and weight freeze",
            "temperature_interpolation": "geometric",
            "coefficient_initialization": "0.02*torch.randn unless ridge_init or zero_symbolic_init",
            "gate_prior": "clip(0.45+0.10*normalized_term_sensitivity,0.05,0.95); optimize logits",
            "pruning_score": "sigmoid(gate_logit/0.2)*abs(coefficient)",
        }
    )
    with (result_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(serialized_config, handle, indent=2)
    if model.pure_symbolic_student:
        export_info = model.export_pure_symbolic_formula(result_dir)
        payload = json.loads(Path(export_info["json"]).read_text(encoding="utf-8"))
        exported_prediction = evaluate_exported_pure_symbolic_formula(
            payload,
            test[list(inputs)].to_numpy(dtype=np.float64),
        )
        model_prediction = test_pred["prediction"].to_numpy(dtype=np.float64)
        export_delta = exported_prediction - model_prediction
        export_rmse = float(np.sqrt(np.mean(export_delta ** 2)))
        export_max_abs = float(np.max(np.abs(export_delta)))
        # The model path is float32 while the independent replay accumulates
        # the serialized coefficients in float64.  A relative 1e-6 allowance
        # is well below fitting error yet avoids treating normal float32 round
        # off as an export mismatch.
        export_tolerance = max(1e-6 * float(y_scale), 1e-30)
        verification = without_attrs(test[list(inputs)]).copy()
        verification["model_prediction"] = model_prediction
        verification["exported_formula_prediction"] = exported_prediction
        verification["formula_minus_model"] = export_delta
        verification.to_csv(result_dir / "pure_symbolic_formula_verification.csv", index=False)
        (result_dir / "pure_symbolic_formula_verification.json").write_text(
            json.dumps(
                {
                    "sample_count": int(len(verification)),
                    "rmse": export_rmse,
                    "max_abs_error": export_max_abs,
                    "tolerance": export_tolerance,
                    "passed": bool(export_max_abs <= export_tolerance),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if export_max_abs > export_tolerance:
            raise RuntimeError(
                "Pure-symbolic export numerical replay diverged from the model: "
                f"max_abs_error={export_max_abs:.6g}, tolerance={export_tolerance:.6g}"
            )
        metrics["pure_symbolic_formula_path"] = export_info["text"]
        metrics["pure_symbolic_exported_terms"] = float(export_info["selected_term_count"])
        metrics["pure_symbolic_export_rmse"] = export_rmse
        metrics["pure_symbolic_export_max_abs_error"] = export_max_abs
        pd.DataFrame([metrics]).to_csv(result_dir / "metrics.csv", index=False)
    return metrics
