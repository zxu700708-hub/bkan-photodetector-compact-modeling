"""Physical symbolic feature library used by the final KAN symbolic trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch

from device_modeling.photodetector.task_config import TaskSpec, target_values


LOG2 = math.log(2.0)
EPS = 1e-12


@dataclass(frozen=True)
class FeatureTerm:
    name: str
    expression: str
    dependencies: tuple[str, ...]
    func: Callable[[torch.Tensor, torch.Tensor, tuple[str, ...]], torch.Tensor]


@dataclass(frozen=True)
class TeacherBundle:
    x_mean: np.ndarray
    x_scale: np.ndarray
    device: torch.device


@dataclass(frozen=True)
class DistillTensors:
    x_z: torch.Tensor
    y: torch.Tensor
    dy: torch.Tensor | None
    weight: torch.Tensor


def _as_float_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)


def _safe_scale(values: np.ndarray | float) -> np.ndarray | float:
    scale = np.asarray(values, dtype=np.float64).copy()
    scale[~np.isfinite(scale)] = 1.0
    scale[np.abs(scale) < EPS] = 1.0
    if scale.ndim == 0:
        return float(scale)
    return scale


def _column(raw: torch.Tensor, inputs: tuple[str, ...], name: str) -> torch.Tensor:
    return raw[:, inputs.index(name):inputs.index(name) + 1]


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(torch.clamp(x, -40.0, 40.0))


def finite_difference_derivative(frame: pd.DataFrame, spec: TaskSpec, inputs: Iterable[str]) -> np.ndarray:
    derivatives = np.full(len(frame), np.nan, dtype=np.float64)
    if "_curve_id" in frame.columns:
        groups = frame.groupby("_curve_id", sort=False)
    else:
        condition_cols = [column for column in inputs if column != spec.axis_col]
        groups = frame.groupby(condition_cols, sort=False)

    for _, group in groups:
        ordered = group.sort_values(spec.axis_col)
        axis = ordered[spec.axis_col].to_numpy(dtype=np.float64)
        values = target_values(ordered, spec)
        if len(axis) < 3 or len(np.unique(axis)) != len(axis):
            continue
        derivatives[ordered.index.to_numpy()] = np.gradient(values, axis, edge_order=2)

    if not np.all(np.isfinite(derivatives)):
        raise ValueError(f"Cannot build finite-difference derivative references for {spec.name}")
    return derivatives


def make_dense_axis_frame(
    frame: pd.DataFrame,
    spec: TaskSpec,
    inputs: tuple[str, ...],
    axis_points: int,
) -> pd.DataFrame:
    if "_curve_id" in frame.columns:
        groups = frame.groupby("_curve_id", sort=False)
    else:
        condition_cols = [column for column in inputs if column != spec.axis_col]
        groups = frame.groupby(condition_cols, sort=False)

    rows: list[pd.DataFrame] = []
    for _, group in groups:
        first = group.iloc[0]
        observed_axis = np.sort(group[spec.axis_col].to_numpy(dtype=np.float64))
        if len(observed_axis) == 0:
            continue
        if spec.key == "ac_response":
            dense = np.exp(np.linspace(np.log(observed_axis.min()), np.log(observed_axis.max()), axis_points))
        else:
            dense = np.linspace(observed_axis.min(), observed_axis.max(), axis_points)
        axis_values = np.unique(np.concatenate([observed_axis, dense]))
        dense_frame = pd.DataFrame({column: first[column] for column in inputs}, index=range(len(axis_values)))
        dense_frame[spec.axis_col] = axis_values
        rows.append(dense_frame)

    if not rows:
        raise ValueError(f"No dense samples could be generated for {spec.name}")
    return pd.concat(rows, ignore_index=True)


def tensorize_distillation(
    bundle: TeacherBundle,
    x_raw: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray | None,
    weights: np.ndarray | None = None,
) -> DistillTensors:
    z = ((x_raw - bundle.x_mean) / bundle.x_scale).astype(np.float32)
    if weights is None:
        weights = np.ones(len(x_raw), dtype=np.float32)
    return DistillTensors(
        x_z=_as_float_tensor(z, bundle.device),
        y=_as_float_tensor(y.reshape(-1, 1), bundle.device),
        dy=_as_float_tensor(dy.reshape(-1, 1), bundle.device) if dy is not None else None,
        weight=_as_float_tensor(weights.reshape(-1, 1), bundle.device),
    )


def build_observed_tensors(
    bundle: TeacherBundle,
    frame: pd.DataFrame,
    spec: TaskSpec,
    inputs: tuple[str, ...],
    output_scale: float,
) -> DistillTensors:
    x_raw = frame[list(inputs)].to_numpy(dtype=np.float64)
    y = target_values(frame, spec) / output_scale
    dy = finite_difference_derivative(frame, spec, inputs) / output_scale
    return tensorize_distillation(bundle, x_raw, y, dy)


def compute_parameter_sensitivity(
    inputs: tuple[str, ...],
    x_raw: np.ndarray,
    grad_raw: np.ndarray,
) -> pd.DataFrame:
    ranges = np.ptp(x_raw, axis=0)
    mean_abs = np.mean(np.abs(grad_raw), axis=0)
    range_scaled = mean_abs * np.maximum(ranges, EPS)
    normalized = range_scaled / max(float(np.max(range_scaled)), EPS)
    rows = []
    for index, name in enumerate(inputs):
        rows.append(
            {
                "input": name,
                "mean_abs_teacher_gradient": float(mean_abs[index]),
                "input_range": float(ranges[index]),
                "range_scaled_sensitivity": float(range_scaled[index]),
                "normalized_sensitivity": float(normalized[index]),
            }
        )
    return pd.DataFrame(rows).sort_values("normalized_sensitivity", ascending=False)


class PhysicalFeatureLibrary:
    """Differentiable physical-parameter term library with train-set scaling."""

    def __init__(
        self,
        spec: TaskSpec,
        inputs: tuple[str, ...],
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        train_x_raw: np.ndarray,
        sensitivity: pd.DataFrame,
        generic_symbolic_terms: bool = False,
    ):
        self.spec = spec
        self.inputs = inputs
        self.x_mean = np.asarray(x_mean, dtype=np.float64)
        self.x_scale = np.asarray(x_scale, dtype=np.float64)
        self.generic_symbolic_terms = generic_symbolic_terms
        axis_values = train_x_raw[:, self.inputs.index(self.spec.axis_col)]
        self.has_forward_axis = bool(np.any(axis_values > 0.0))
        self.has_reverse_axis = bool(np.any(axis_values < 0.0))
        self.terms = self._build_terms()
        self.centers, self.scales = self._fit_term_scalers(train_x_raw)
        self.sensitivity = sensitivity.set_index("input")["normalized_sensitivity"].to_dict()
        self.term_priors = self._build_term_priors()
        self.axis_center, self.axis_scale = self._fit_axis_scaler(train_x_raw)

    def raw_from_z(self, z: torch.Tensor) -> torch.Tensor:
        mean = _as_float_tensor(self.x_mean.reshape(1, -1), z.device)
        scale = _as_float_tensor(self.x_scale.reshape(1, -1), z.device)
        return z * scale + mean

    def _build_terms(self) -> list[FeatureTerm]:
        axis = self.spec.axis_col
        terms: list[FeatureTerm] = [
            FeatureTerm("one", "1", tuple(), lambda raw, z, inputs: torch.ones((raw.shape[0], 1), device=raw.device)),
        ]

        if self.spec.key != "ac_response":
            terms.extend(
                [
                    FeatureTerm(
                        f"{axis}_z",
                        f"scaled({axis})",
                        (axis,),
                        lambda raw, z, inputs: z[:, inputs.index(axis):inputs.index(axis) + 1],
                    ),
                    FeatureTerm(
                        f"{axis}_z2",
                        f"scaled({axis})^2",
                        (axis,),
                        lambda raw, z, inputs: z[:, inputs.index(axis):inputs.index(axis) + 1].pow(2),
                    ),
                    FeatureTerm(
                        f"soft_reverse_{axis}",
                        f"softplus(-{axis}/0.2)",
                        (axis,),
                        lambda raw, z, inputs: _softplus(-_column(raw, inputs, axis) / 0.2),
                    ),
                    FeatureTerm(
                        f"tanh_reverse_{axis}",
                        f"tanh(-{axis}/1.0)",
                        (axis,),
                        lambda raw, z, inputs: torch.tanh(-_column(raw, inputs, axis)),
                    ),
                    FeatureTerm(
                        f"reverse_saturation_{axis}",
                        f"1-exp(-(softplus(-{axis}/0.2)-log(2)))",
                        (axis,),
                        lambda raw, z, inputs: 1.0
                        - torch.exp(-torch.clamp(_softplus(-_column(raw, inputs, axis) / 0.2) - LOG2, min=0.0)),
                    ),
                ]
            )
            terms.extend(
                [
                    FeatureTerm(
                        f"forward_gate_{axis}",
                        f"sigmoid({axis}/0.2)",
                        (axis,),
                        lambda raw, z, inputs: torch.sigmoid(_column(raw, inputs, axis) / 0.2),
                    ),
                    FeatureTerm(
                        f"reverse_gate_{axis}",
                        f"sigmoid(-{axis}/0.2)",
                        (axis,),
                        lambda raw, z, inputs: torch.sigmoid(-_column(raw, inputs, axis) / 0.2),
                    ),
                ]
            )

        include_axis_interactions = self.spec.key != "ac_response"

        if "simulation_temperature" in self.inputs:
            terms.extend(
                [
                    FeatureTerm(
                        "temperature_z",
                        "scaled(simulation_temperature)",
                        ("simulation_temperature",),
                        lambda raw, z, inputs: z[:, inputs.index("simulation_temperature"):inputs.index("simulation_temperature") + 1],
                    ),
                    FeatureTerm(
                        "inverse_temperature",
                        "1/simulation_temperature",
                        ("simulation_temperature",),
                        lambda raw, z, inputs: 1.0 / torch.clamp(_column(raw, inputs, "simulation_temperature"), min=1.0),
                    ),
                ]
            )
            if include_axis_interactions:
                terms.append(
                    FeatureTerm(
                        f"{axis}_over_temperature",
                        f"{axis}/simulation_temperature",
                        (axis, "simulation_temperature"),
                        lambda raw, z, inputs: _column(raw, inputs, axis)
                        / torch.clamp(_column(raw, inputs, "simulation_temperature"), min=1.0),
                    )
                )

        if "active_layer_length" in self.inputs:
            terms.extend(
                [
                    FeatureTerm(
                        "length_z",
                        "scaled(active_layer_length)",
                        ("active_layer_length",),
                        lambda raw, z, inputs: z[:, inputs.index("active_layer_length"):inputs.index("active_layer_length") + 1],
                    ),
                    FeatureTerm(
                        "inverse_length",
                        "1/active_layer_length",
                        ("active_layer_length",),
                        lambda raw, z, inputs: 1.0 / torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6),
                    ),
                ]
            )
            if include_axis_interactions:
                terms.append(
                    FeatureTerm(
                        f"{axis}_over_length",
                        f"{axis}/active_layer_length",
                        (axis, "active_layer_length"),
                        lambda raw, z, inputs: _column(raw, inputs, axis)
                        / torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6),
                    )
                )

        if "trap_assisted_recomb_A" in self.inputs:
            terms.append(
                FeatureTerm(
                    "trap_A_z",
                    "scaled(trap_assisted_recomb_A)",
                    ("trap_assisted_recomb_A",),
                    lambda raw, z, inputs: z[:, inputs.index("trap_assisted_recomb_A"):inputs.index("trap_assisted_recomb_A") + 1],
                )
            )

        if "ge_sio2_recomb_velocity" in self.inputs:
            terms.append(
                FeatureTerm(
                    "ge_sio2_velocity_z",
                    "scaled(ge_sio2_recomb_velocity)",
                    ("ge_sio2_recomb_velocity",),
                    lambda raw, z, inputs: z[:, inputs.index("ge_sio2_recomb_velocity"):inputs.index("ge_sio2_recomb_velocity") + 1],
                )
            )

        if "ge_si_recomb_velocity" in self.inputs:
            terms.append(
                FeatureTerm(
                    "ge_si_velocity_z",
                    "scaled(ge_si_recomb_velocity)",
                    ("ge_si_recomb_velocity",),
                    lambda raw, z, inputs: z[:, inputs.index("ge_si_recomb_velocity"):inputs.index("ge_si_recomb_velocity") + 1],
                )
            )

        if "ge_sio2_recomb_velocity" in self.inputs and "active_layer_length" in self.inputs:
            terms.append(
                FeatureTerm(
                    "ge_sio2_velocity_over_length",
                    "ge_sio2_recomb_velocity/active_layer_length",
                    ("ge_sio2_recomb_velocity", "active_layer_length"),
                    lambda raw, z, inputs: _column(raw, inputs, "ge_sio2_recomb_velocity")
                    / torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6),
                )
            )

        if "ge_si_recomb_velocity" in self.inputs and "active_layer_length" in self.inputs:
            terms.append(
                FeatureTerm(
                    "ge_si_velocity_over_length",
                    "ge_si_recomb_velocity/active_layer_length",
                    ("ge_si_recomb_velocity", "active_layer_length"),
                    lambda raw, z, inputs: _column(raw, inputs, "ge_si_recomb_velocity")
                    / torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6),
                )
            )

        if self.spec.key == "photo_current" and "active_layer_length" in self.inputs:
            terms.append(
                FeatureTerm(
                    f"reverse_drive_over_length",
                    f"(softplus(-{axis}/0.2)-log(2))/active_layer_length",
                    (axis, "active_layer_length"),
                    lambda raw, z, inputs: (
                        _softplus(-_column(raw, inputs, axis) / 0.2) - LOG2
                    )
                    / torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6),
                )
            )

        if (
            self.spec.key == "photo_current"
            and "ge_sio2_recomb_velocity" in self.inputs
            and "ge_si_recomb_velocity" in self.inputs
            and "active_layer_length" in self.inputs
        ):
            terms.extend(
                [
                    FeatureTerm(
                        "surface_recomb_sum_z",
                        "scaled(ge_sio2_recomb_velocity + ge_si_recomb_velocity)",
                        ("ge_sio2_recomb_velocity", "ge_si_recomb_velocity"),
                        lambda raw, z, inputs: (
                            _column(raw, inputs, "ge_sio2_recomb_velocity")
                            + _column(raw, inputs, "ge_si_recomb_velocity")
                        ),
                    ),
                    FeatureTerm(
                        "surface_recomb_over_length",
                        "(ge_sio2_recomb_velocity + ge_si_recomb_velocity)/active_layer_length",
                        ("ge_sio2_recomb_velocity", "ge_si_recomb_velocity", "active_layer_length"),
                        lambda raw, z, inputs: (
                            _column(raw, inputs, "ge_sio2_recomb_velocity")
                            + _column(raw, inputs, "ge_si_recomb_velocity")
                        )
                        / torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6),
                    ),
                    FeatureTerm(
                        "reverse_drive_surface_recomb",
                        "(softplus(-light_voltage/0.2)-log(2))*(ge_sio2_recomb_velocity + ge_si_recomb_velocity)/active_layer_length",
                        (axis, "ge_sio2_recomb_velocity", "ge_si_recomb_velocity", "active_layer_length"),
                        lambda raw, z, inputs: (
                            (_softplus(-_column(raw, inputs, axis) / 0.2) - LOG2)
                            * (
                                _column(raw, inputs, "ge_sio2_recomb_velocity")
                                + _column(raw, inputs, "ge_si_recomb_velocity")
                            )
                            / torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6)
                        ),
                    ),
                ]
            )

        if self.spec.key == "photo_current" and "trap_assisted_recomb_A" in self.inputs:
            terms.append(
                FeatureTerm(
                    "reverse_drive_trap_A",
                    "(softplus(-light_voltage/0.2)-log(2))*trap_assisted_recomb_A",
                    (axis, "trap_assisted_recomb_A"),
                    lambda raw, z, inputs: (
                        _softplus(-_column(raw, inputs, axis) / 0.2) - LOG2
                    )
                    * _column(raw, inputs, "trap_assisted_recomb_A"),
                )
            )

        if self.spec.key == "photo_current" and "active_layer_length" in self.inputs:
            terms.extend(
                [
                    FeatureTerm(
                        "collection_saturation_length",
                        "(1-exp(-max(softplus(-light_voltage/0.2)-log(2),0)))*active_layer_length",
                        (axis, "active_layer_length"),
                        lambda raw, z, inputs: (
                            1.0
                            - torch.exp(
                                -torch.clamp(_softplus(-_column(raw, inputs, axis) / 0.2) - LOG2, min=0.0)
                            )
                        )
                        * _column(raw, inputs, "active_layer_length"),
                    ),
                    FeatureTerm(
                        "length_collection_saturation",
                        "1-exp(-active_layer_length/40)",
                        ("active_layer_length",),
                        lambda raw, z, inputs: 1.0
                        - torch.exp(-torch.clamp(_column(raw, inputs, "active_layer_length"), min=0.0) / 40.0),
                    ),
                ]
            )

        if (
            self.spec.key == "photo_current"
            and "ge_sio2_recomb_velocity" in self.inputs
            and "ge_si_recomb_velocity" in self.inputs
            and "active_layer_length" in self.inputs
            and "simulation_temperature" in self.inputs
        ):
            terms.extend(
                [
                    FeatureTerm(
                        "thermal_surface_recomb_over_length",
                        "(ge_sio2_recomb_velocity + ge_si_recomb_velocity)/(active_layer_length*simulation_temperature)",
                        ("ge_sio2_recomb_velocity", "ge_si_recomb_velocity", "active_layer_length", "simulation_temperature"),
                        lambda raw, z, inputs: (
                            _column(raw, inputs, "ge_sio2_recomb_velocity")
                            + _column(raw, inputs, "ge_si_recomb_velocity")
                        )
                        / (
                            torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6)
                            * torch.clamp(_column(raw, inputs, "simulation_temperature"), min=1.0)
                        ),
                    ),
                    FeatureTerm(
                        "reverse_thermal_surface_recomb",
                        "(softplus(-light_voltage/0.2)-log(2))*(ge_sio2_recomb_velocity + ge_si_recomb_velocity)/(active_layer_length*simulation_temperature)",
                        (axis, "ge_sio2_recomb_velocity", "ge_si_recomb_velocity", "active_layer_length", "simulation_temperature"),
                        lambda raw, z, inputs: (
                            (_softplus(-_column(raw, inputs, axis) / 0.2) - LOG2)
                            * (
                                _column(raw, inputs, "ge_sio2_recomb_velocity")
                                + _column(raw, inputs, "ge_si_recomb_velocity")
                            )
                            / (
                                torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6)
                                * torch.clamp(_column(raw, inputs, "simulation_temperature"), min=1.0)
                            )
                        ),
                    ),
                ]
            )

        if (
            self.spec.key == "photo_current"
            and "trap_assisted_recomb_A" in self.inputs
            and "active_layer_length" in self.inputs
        ):
            terms.append(
                FeatureTerm(
                    "trap_A_times_length",
                    "trap_assisted_recomb_A*active_layer_length",
                    ("trap_assisted_recomb_A", "active_layer_length"),
                    lambda raw, z, inputs: _column(raw, inputs, "trap_assisted_recomb_A")
                    * _column(raw, inputs, "active_layer_length"),
                )
            )

        if (
            self.spec.key == "photo_current"
            and "trap_assisted_recomb_A" in self.inputs
            and "simulation_temperature" in self.inputs
        ):
            terms.append(
                FeatureTerm(
                    "trap_A_over_temperature",
                    "trap_assisted_recomb_A/simulation_temperature",
                    ("trap_assisted_recomb_A", "simulation_temperature"),
                    lambda raw, z, inputs: _column(raw, inputs, "trap_assisted_recomb_A")
                    / torch.clamp(_column(raw, inputs, "simulation_temperature"), min=1.0),
                )
            )

        if self.spec.key == "photo_current" and self.generic_symbolic_terms:
            for column in self.inputs:
                terms.extend(
                    [
                        FeatureTerm(
                            f"generic_square_{column}",
                            f"scaled({column})^2",
                            (column,),
                            lambda raw, z, inputs, name=column: z[:, inputs.index(name):inputs.index(name) + 1].pow(2),
                        ),
                        FeatureTerm(
                            f"generic_cube_{column}",
                            f"scaled({column})^3",
                            (column,),
                            lambda raw, z, inputs, name=column: z[:, inputs.index(name):inputs.index(name) + 1].pow(3),
                        ),
                        FeatureTerm(
                            f"generic_tanh_{column}",
                            f"tanh(scaled({column}))",
                            (column,),
                            lambda raw, z, inputs, name=column: torch.tanh(
                                z[:, inputs.index(name):inputs.index(name) + 1]
                            ),
                        ),
                        FeatureTerm(
                            f"generic_softplus_{column}",
                            f"softplus(scaled({column}))",
                            (column,),
                            lambda raw, z, inputs, name=column: _softplus(
                                z[:, inputs.index(name):inputs.index(name) + 1]
                            ),
                        ),
                        FeatureTerm(
                            f"generic_gaussian_{column}",
                            f"exp(-scaled({column})^2)",
                            (column,),
                            lambda raw, z, inputs, name=column: torch.exp(
                                -z[:, inputs.index(name):inputs.index(name) + 1].pow(2)
                            ),
                        ),
                        FeatureTerm(
                            f"generic_gaussian_low_{column}",
                            f"exp(-(scaled({column}) + 1)^2)",
                            (column,),
                            lambda raw, z, inputs, name=column: torch.exp(
                                -(z[:, inputs.index(name):inputs.index(name) + 1] + 1.0).pow(2)
                            ),
                        ),
                        FeatureTerm(
                            f"generic_gaussian_high_{column}",
                            f"exp(-(scaled({column}) - 1)^2)",
                            (column,),
                            lambda raw, z, inputs, name=column: torch.exp(
                                -(z[:, inputs.index(name):inputs.index(name) + 1] - 1.0).pow(2)
                            ),
                        ),
                        FeatureTerm(
                            f"generic_exp_pos_{column}",
                            f"exp(clip(scaled({column}), -4, 4))",
                            (column,),
                            lambda raw, z, inputs, name=column: torch.exp(torch.clamp(
                                z[:, inputs.index(name):inputs.index(name) + 1], -4.0, 4.0
                            )),
                        ),
                        FeatureTerm(
                            f"generic_exp_neg_{column}",
                            f"exp(-clip(scaled({column}), -4, 4))",
                            (column,),
                            lambda raw, z, inputs, name=column: torch.exp(-torch.clamp(
                                z[:, inputs.index(name):inputs.index(name) + 1], -4.0, 4.0
                            )),
                        ),
                        FeatureTerm(
                            f"generic_logabs_{column}",
                            f"log1p(abs(scaled({column})))",
                            (column,),
                            lambda raw, z, inputs, name=column: torch.log1p(
                                z[:, inputs.index(name):inputs.index(name) + 1].abs()
                            ),
                        ),
                    ]
                )

            for left_index, left in enumerate(self.inputs):
                for right in self.inputs[left_index + 1:]:
                    terms.append(
                        FeatureTerm(
                            f"generic_{left}_times_{right}",
                            f"scaled({left})*scaled({right})",
                            (left, right),
                            lambda raw, z, inputs, a=left, b=right: (
                                z[:, inputs.index(a):inputs.index(a) + 1]
                                * z[:, inputs.index(b):inputs.index(b) + 1]
                            ),
                        )
                    )

        # Capacitance-versus-bias curves are usually smooth but strongly
        # asymmetric around depletion.  The generic polynomial pool used for
        # photo current is a poor prior here: it neither exposes the location
        # of the knee nor distinguishes forward/reverse curvature.  These
        # bounded terms give the gate model an explicit, deployable curvature
        # vocabulary while retaining the raw bias as the only swept variable.
        if self.spec.key == "capacitance" and "bias_v" in self.inputs:
            bias = "bias_v"
            terms.extend(
                [
                    FeatureTerm(
                        "bias_curvature_z3",
                        "scaled(bias_v)^3",
                        (bias,),
                        lambda raw, z, inputs: z[:, inputs.index(bias):inputs.index(bias) + 1].pow(3),
                    ),
                    FeatureTerm(
                        "bias_reverse_softplus2",
                        "softplus(-bias_v/0.2)^2",
                        (bias,),
                        lambda raw, z, inputs: _softplus(-_column(raw, inputs, bias) / 0.2).pow(2),
                    ),
                    FeatureTerm(
                        "bias_forward_softplus2",
                        "softplus(bias_v/0.2)^2",
                        (bias,),
                        lambda raw, z, inputs: _softplus(_column(raw, inputs, bias) / 0.2).pow(2),
                    ),
                    FeatureTerm(
                        "bias_depletion_knee",
                        "1/sqrt(1 + (bias_v/0.45)^2)",
                        (bias,),
                        lambda raw, z, inputs: torch.rsqrt(
                            1.0 + (_column(raw, inputs, bias) / 0.45).pow(2)
                        ),
                    ),
                    FeatureTerm(
                        "bias_reverse_exponential",
                        "exp(-clip(bias_v, -3, 3))",
                        (bias,),
                        lambda raw, z, inputs: torch.exp(
                            -torch.clamp(_column(raw, inputs, bias), -3.0, 3.0)
                        ),
                    ),
                    FeatureTerm(
                        "bias_signed_log_curvature",
                        "sign(bias_v)*log1p(abs(bias_v)/0.1)",
                        (bias,),
                        lambda raw, z, inputs: torch.sign(_column(raw, inputs, bias))
                        * torch.log1p(_column(raw, inputs, bias).abs() / 0.1),
                    ),
                ]
            )
            if "log_frequency_ghz" in self.inputs:
                terms.extend(
                    [
                        FeatureTerm(
                            "bias_curvature_times_log_frequency",
                            "scaled(bias_v)^2*scaled(log_frequency_ghz)",
                            (bias, "log_frequency_ghz"),
                            lambda raw, z, inputs: (
                                z[:, inputs.index(bias):inputs.index(bias) + 1].pow(2)
                                * z[:, inputs.index("log_frequency_ghz"):inputs.index("log_frequency_ghz") + 1]
                            ),
                        ),
                        FeatureTerm(
                            "depletion_knee_times_log_frequency",
                            "1/sqrt(1 + (bias_v/0.45)^2)*scaled(log_frequency_ghz)",
                            (bias, "log_frequency_ghz"),
                            lambda raw, z, inputs: (
                                torch.rsqrt(1.0 + (_column(raw, inputs, bias) / 0.45).pow(2))
                                * z[:, inputs.index("log_frequency_ghz"):inputs.index("log_frequency_ghz") + 1]
                            ),
                        ),
                    ]
                )

        if self.spec.key == "ac_response" and "frequency_ghz" in self.inputs:
            terms.extend(
                [
                    FeatureTerm(
                        "log_frequency_z",
                        "scaled(log10(frequency_ghz))",
                        ("frequency_ghz",),
                        lambda raw, z, inputs: torch.log10(
                            torch.clamp(_column(raw, inputs, "frequency_ghz"), min=EPS)
                        ),
                    ),
                    FeatureTerm(
                        "frequency_z",
                        "scaled(frequency_ghz)",
                        ("frequency_ghz",),
                        lambda raw, z, inputs: z[:, inputs.index("frequency_ghz"):inputs.index("frequency_ghz") + 1],
                    ),
                    FeatureTerm(
                        "frequency_z2",
                        "scaled(frequency_ghz)^2",
                        ("frequency_ghz",),
                        lambda raw, z, inputs: z[:, inputs.index("frequency_ghz"):inputs.index("frequency_ghz") + 1].pow(2),
                    ),
                    FeatureTerm(
                        "lowpass_frequency_gate",
                        "1/(1 + frequency_ghz/10)",
                        ("frequency_ghz",),
                        lambda raw, z, inputs: 1.0
                        / (1.0 + torch.clamp(_column(raw, inputs, "frequency_ghz"), min=0.0) / 10.0),
                    ),
                    FeatureTerm(
                        "lowpass_frequency_gate2",
                        "1/sqrt(1 + (frequency_ghz/10)^2)",
                        ("frequency_ghz",),
                        lambda raw, z, inputs: torch.rsqrt(
                            1.0 + torch.clamp(_column(raw, inputs, "frequency_ghz"), min=0.0).pow(2) / 100.0
                        ),
                    ),
                ]
            )

            if "active_layer_length" in self.inputs:
                terms.extend(
                    [
                        FeatureTerm(
                            "frequency_times_length",
                            "frequency_ghz*active_layer_length",
                            ("frequency_ghz", "active_layer_length"),
                            lambda raw, z, inputs: _column(raw, inputs, "frequency_ghz")
                            * _column(raw, inputs, "active_layer_length"),
                        ),
                        FeatureTerm(
                            "frequency_over_length",
                            "frequency_ghz/active_layer_length",
                            ("frequency_ghz", "active_layer_length"),
                            lambda raw, z, inputs: _column(raw, inputs, "frequency_ghz")
                            / torch.clamp(_column(raw, inputs, "active_layer_length"), min=1e-6),
                        ),
                        FeatureTerm(
                            "length_lowpass_frequency",
                            "active_layer_length/(1 + frequency_ghz/10)",
                            ("frequency_ghz", "active_layer_length"),
                            lambda raw, z, inputs: _column(raw, inputs, "active_layer_length")
                            / (1.0 + torch.clamp(_column(raw, inputs, "frequency_ghz"), min=0.0) / 10.0),
                        ),
                    ]
                )

            if "simulation_temperature" in self.inputs:
                terms.extend(
                    [
                        FeatureTerm(
                            "frequency_over_temperature",
                            "frequency_ghz/simulation_temperature",
                            ("frequency_ghz", "simulation_temperature"),
                            lambda raw, z, inputs: _column(raw, inputs, "frequency_ghz")
                            / torch.clamp(_column(raw, inputs, "simulation_temperature"), min=1.0),
                        ),
                        FeatureTerm(
                            "temperature_lowpass_frequency",
                            "simulation_temperature/(1 + frequency_ghz/10)",
                            ("frequency_ghz", "simulation_temperature"),
                            lambda raw, z, inputs: _column(raw, inputs, "simulation_temperature")
                            / (1.0 + torch.clamp(_column(raw, inputs, "frequency_ghz"), min=0.0) / 10.0),
                        ),
                    ]
                )

        return terms

    def _fit_term_scalers(self, train_x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = ((train_x_raw - self.x_mean) / self.x_scale).astype(np.float32)
        raw_tensor = torch.from_numpy(train_x_raw.astype(np.float32))
        z_tensor = torch.from_numpy(z)
        values = []
        for term in self.terms:
            values.append(term.func(raw_tensor, z_tensor, self.inputs).detach().cpu().numpy())
        matrix = np.concatenate(values, axis=1)
        centers = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        centers[0] = 0.0
        scales[0] = 1.0
        scales = _safe_scale(scales)
        return centers.astype(np.float64), np.asarray(scales, dtype=np.float64)

    def _fit_axis_scaler(self, train_x_raw: np.ndarray) -> tuple[float, float]:
        axis_values = train_x_raw[:, self.inputs.index(self.spec.axis_col)]
        if self.spec.key == "ac_response":
            axis_values = np.log10(np.maximum(axis_values, EPS))
        center = float(np.mean(axis_values))
        scale = float(np.std(axis_values))
        return center, float(_safe_scale(scale))

    def _build_term_priors(self) -> np.ndarray:
        priors = []
        for term in self.terms:
            if not term.dependencies:
                priors.append(1.0)
            else:
                priors.append(max(float(self.sensitivity.get(dep, 0.0)) for dep in term.dependencies))
        values = np.asarray(priors, dtype=np.float64)
        return values / max(float(values.max()), EPS)

    def matrix(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.raw_from_z(z)
        centers = _as_float_tensor(self.centers.reshape(1, -1), z.device)
        scales = _as_float_tensor(self.scales.reshape(1, -1), z.device)
        values = [term.func(raw, z, self.inputs) for term in self.terms]
        return (torch.cat(values, dim=1) - centers) / scales

    def axis_value(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.raw_from_z(z)
        axis = _column(raw, self.inputs, self.spec.axis_col)
        if self.spec.key == "ac_response":
            axis = torch.log10(torch.clamp(axis, min=EPS))
        return (axis - self.axis_center) / self.axis_scale

    def feature_table(self) -> pd.DataFrame:
        rows = []
        for index, term in enumerate(self.terms):
            rows.append(
                {
                    "term_index": index,
                    "term_name": term.name,
                    "expression": term.expression,
                    "dependencies": ",".join(term.dependencies),
                    "center": float(self.centers[index]),
                    "scale": float(self.scales[index]),
                    "kan_sensitivity_prior": float(self.term_priors[index]),
                }
            )
        return pd.DataFrame(rows)


