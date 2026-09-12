"""Compact-model construction baselines shared by the PD and APD studies.

The implementations are method adaptations for the present tabular device
data.  They are deliberately named ``gmls`` and ``autopinn_adapted`` in
metadata so that results are not confused with source-code reproductions of
the cited diode and transistor studies.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class GeneralizedMovingLeastSquaresRegressor:
    """Quadratic local-polynomial GMLS regressor for scattered device data.

    Inputs and targets are standardized from training rows only.  At every
    query, a Gaussian-weighted quadratic polynomial is fitted to the nearest
    training samples and evaluated at the query origin.  A small local ridge
    term protects nearly singular neighborhoods.
    """

    def __init__(
        self,
        *,
        degree: int = 2,
        neighbor_factor: int = 4,
        min_neighbors: int = 48,
        ridge: float = 1.0e-8,
        weight_scale: float = 2.0,
        clip_margin: float | None = None,
    ) -> None:
        if degree not in (1, 2):
            raise ValueError("GMLS supports degree 1 or 2")
        if neighbor_factor < 1 or min_neighbors < 1:
            raise ValueError("GMLS neighborhood sizes must be positive")
        if ridge <= 0.0 or weight_scale <= 0.0:
            raise ValueError("GMLS ridge and weight scale must be positive")
        self.degree = int(degree)
        self.neighbor_factor = int(neighbor_factor)
        self.min_neighbors = int(min_neighbors)
        self.ridge = float(ridge)
        self.weight_scale = float(weight_scale)
        if clip_margin is not None and clip_margin < 0.0:
            raise ValueError("GMLS clip margin must be nonnegative")
        self.clip_margin = clip_margin

    @staticmethod
    def _quadratic_term_count(n_features: int) -> int:
        return 1 + n_features + n_features * (n_features + 1) // 2

    def _basis(self, delta: np.ndarray) -> np.ndarray:
        pieces = [np.ones((len(delta), 1), dtype=np.float64), delta]
        if self.degree == 2:
            quadratic = [
                (delta[:, i] * delta[:, j]).reshape(-1, 1)
                for i in range(delta.shape[1])
                for j in range(i, delta.shape[1])
            ]
            pieces.extend(quadratic)
        return np.hstack(pieces)

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if x.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("GMLS requires aligned two-dimensional X and y")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("GMLS training data must be finite")
        self.x_scaler_ = StandardScaler().fit(x)
        self.y_scaler_ = StandardScaler().fit(y)
        self.x_train_ = self.x_scaler_.transform(x)
        self.y_train_ = self.y_scaler_.transform(y).reshape(-1)
        self.y_min_ = float(self.y_train_.min())
        self.y_max_ = float(self.y_train_.max())
        n_features = x.shape[1]
        linear_terms = 1 + n_features
        quadratic_terms = self._quadratic_term_count(n_features)
        self.basis_terms_ = linear_terms if self.degree == 1 else quadratic_terms
        requested = max(self.min_neighbors, self.neighbor_factor * self.basis_terms_)
        self.n_neighbors_ = min(len(x), requested)
        self.neighbors_ = NearestNeighbors(
            n_neighbors=self.n_neighbors_, algorithm="auto"
        ).fit(self.x_train_)
        self.n_features_in_ = n_features
        self.n_training_rows_ = len(x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "neighbors_"):
            raise RuntimeError("GMLS must be fitted before prediction")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("GMLS prediction features do not match training")
        x_scaled = self.x_scaler_.transform(x)
        distances, indices = self.neighbors_.kneighbors(x_scaled)
        predictions = np.empty(len(x_scaled), dtype=np.float64)
        penalty = np.eye(self.basis_terms_, dtype=np.float64) * self.ridge
        penalty[0, 0] = self.ridge * 1.0e-4
        for row, query in enumerate(x_scaled):
            local_x = self.x_train_[indices[row]]
            delta = local_x - query
            basis = self._basis(delta)
            bandwidth = max(float(distances[row, -1]), 1.0e-8)
            weights = np.exp(
                -self.weight_scale * np.square(distances[row] / bandwidth)
            )
            weighted_basis = basis * weights[:, None]
            lhs = basis.T @ weighted_basis + penalty
            rhs = basis.T @ (weights * self.y_train_[indices[row]])
            try:
                coefficients = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            predictions[row] = coefficients[0]
        if self.clip_margin is not None:
            span = max(self.y_max_ - self.y_min_, 1.0)
            predictions = np.clip(
                predictions,
                self.y_min_ - self.clip_margin * span,
                self.y_max_ + self.clip_margin * span,
            )
        return self.y_scaler_.inverse_transform(predictions[:, None]).reshape(-1)


def fit_predict_gmls(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    spec,
) -> tuple[np.ndarray, dict[str, object]]:
    started = time.perf_counter()
    x_train = train[list(spec.input_cols)].to_numpy(dtype=np.float64)
    x_validation = validation[list(spec.input_cols)].to_numpy(dtype=np.float64)
    x_test = test[list(spec.input_cols)].to_numpy(dtype=np.float64)
    y_train = _transformed_target(train, spec)
    y_validation = _transformed_target(validation, spec)
    candidate_specs = (
        {"degree": 1, "neighbor_factor": 4},
        {"degree": 2, "neighbor_factor": 4},
        {"degree": 2, "neighbor_factor": 8},
    )
    candidate_records: list[dict[str, object]] = []
    selected: tuple[float, GeneralizedMovingLeastSquaresRegressor] | None = None
    for candidate in candidate_specs:
        model = GeneralizedMovingLeastSquaresRegressor(
            **candidate,
            min_neighbors=48,
            ridge=1.0e-4,
            weight_scale=1.0,
            clip_margin=0.10,
        ).fit(x_train, y_train)
        validation_prediction = model.predict(x_validation)
        validation_mse = float(np.mean(np.square(validation_prediction - y_validation)))
        candidate_records.append(
            {
                **candidate,
                "neighbors": int(model.n_neighbors_),
                "validation_mse": validation_mse,
            }
        )
        if selected is None or validation_mse < selected[0]:
            selected = (validation_mse, model)
    assert selected is not None
    best_validation_mse, model = selected
    prediction = model.predict(x_test)
    return prediction, {
        "method_variant": "validation_selected_gaussian_gmls_adapted",
        "param_count": int(len(train) * (len(spec.input_cols) + 1)),
        "stored_training_values": int(len(train) * (len(spec.input_cols) + 1)),
        "selected_degree": int(model.degree),
        "local_basis_terms": int(model.basis_terms_),
        "neighbors": int(model.n_neighbors_),
        "best_validation_mse": best_validation_mse,
        "candidate_validation": json.dumps(candidate_records, sort_keys=True),
        "train_time_s": float(time.perf_counter() - started),
    }


class CurveTablePchipRegressor:
    """Curve-object lookup table with shape-preserving axis interpolation.

    Each distinct set of non-axis inputs defines one stored response curve.
    PCHIP evaluates those curves on the requested swept-axis coordinates, and
    inverse-distance weights interpolate between nearby characterized device
    conditions.  This is deliberately different from row-wise KNN: complete
    curves are the stored objects and the swept-axis response remains smooth.
    """

    def __init__(self, *, axis_col: str, input_cols: Sequence[str]) -> None:
        if axis_col not in input_cols:
            raise ValueError(f"Axis {axis_col!r} is absent from the model inputs")
        self.axis_col = str(axis_col)
        self.input_cols = tuple(input_cols)
        self.condition_cols = tuple(
            column for column in self.input_cols if column != self.axis_col
        )

    def fit(self, frame: pd.DataFrame, target: np.ndarray):
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(frame) != len(target) or len(frame) < 2:
            raise ValueError("Curve LUT requires aligned training rows")
        work = frame[list(self.input_cols)].copy()
        work["__target"] = target
        if not np.isfinite(work.to_numpy(dtype=np.float64)).all():
            raise ValueError("Curve LUT training data must be finite")

        group_key: str | list[str]
        if self.condition_cols:
            group_key = list(self.condition_cols)
        else:
            work["__single_condition"] = 0.0
            group_key = "__single_condition"

        condition_rows: list[np.ndarray] = []
        interpolators: list[tuple[np.ndarray, np.ndarray, PchipInterpolator | None]] = []
        stored_points = 0
        for key, group in work.groupby(group_key, sort=False, dropna=False):
            collapsed = (
                group.groupby(self.axis_col, as_index=False, sort=True)["__target"]
                .mean()
                .sort_values(self.axis_col)
            )
            axis = collapsed[self.axis_col].to_numpy(dtype=np.float64)
            values = collapsed["__target"].to_numpy(dtype=np.float64)
            if len(axis) == 0:
                continue
            interpolator = (
                PchipInterpolator(axis, values, extrapolate=False)
                if len(axis) >= 2
                else None
            )
            if self.condition_cols:
                key_values = key if isinstance(key, tuple) else (key,)
                condition_rows.append(np.asarray(key_values, dtype=np.float64))
            else:
                condition_rows.append(np.zeros(1, dtype=np.float64))
            interpolators.append((axis, values, interpolator))
            stored_points += len(axis)
        if not interpolators:
            raise ValueError("Curve LUT found no usable training curves")

        self.condition_values_ = np.vstack(condition_rows)
        self.condition_scaler_ = StandardScaler().fit(self.condition_values_)
        self.condition_scaled_ = self.condition_scaler_.transform(
            self.condition_values_
        )
        self.interpolators_ = interpolators
        self.n_curves_ = len(interpolators)
        self.stored_points_ = int(stored_points)
        return self

    @staticmethod
    def _curve_values(
        record: tuple[np.ndarray, np.ndarray, PchipInterpolator | None],
        axis_query: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        axis, values, interpolator = record
        clipped = np.clip(axis_query, axis[0], axis[-1])
        boundary_count = int(np.count_nonzero(clipped != axis_query))
        if interpolator is None:
            return np.full(len(axis_query), values[0], dtype=np.float64), boundary_count
        return np.asarray(interpolator(clipped), dtype=np.float64), boundary_count

    def predict_frame(
        self,
        frame: pd.DataFrame,
        *,
        neighbor_curves: int,
        distance_power: float,
    ) -> tuple[np.ndarray, dict[str, int]]:
        if not hasattr(self, "interpolators_"):
            raise RuntimeError("Curve LUT must be fitted before prediction")
        if neighbor_curves < 1 or distance_power <= 0.0:
            raise ValueError("Curve LUT neighbor settings must be positive")
        work = frame[list(self.input_cols)].reset_index(drop=True)
        if not np.isfinite(work.to_numpy(dtype=np.float64)).all():
            raise ValueError("Curve LUT query data must be finite")
        predictions = np.empty(len(work), dtype=np.float64)
        boundary_count = 0
        if self.condition_cols:
            grouped = work.groupby(list(self.condition_cols), sort=False, dropna=False)
        else:
            work = work.assign(__single_condition=0.0)
            grouped = work.groupby("__single_condition", sort=False, dropna=False)

        k = min(int(neighbor_curves), self.n_curves_)
        for key, indices in grouped.indices.items():
            index_array = np.asarray(indices, dtype=np.int64)
            if self.condition_cols:
                key_values = key if isinstance(key, tuple) else (key,)
                query_condition = np.asarray(key_values, dtype=np.float64)[None, :]
            else:
                query_condition = np.zeros((1, 1), dtype=np.float64)
            query_scaled = self.condition_scaler_.transform(query_condition)[0]
            distances = np.linalg.norm(
                self.condition_scaled_ - query_scaled[None, :], axis=1
            )
            nearest = np.argsort(distances, kind="stable")[:k]
            nearest_distances = distances[nearest]
            exact = nearest_distances <= 1.0e-12
            if exact.any():
                weights = exact.astype(np.float64)
            else:
                weights = np.power(np.maximum(nearest_distances, 1.0e-12), -distance_power)
            weights /= weights.sum()
            axis_query = work.loc[index_array, self.axis_col].to_numpy(dtype=np.float64)
            curve_matrix = np.empty((k, len(index_array)), dtype=np.float64)
            for row, curve_index in enumerate(nearest):
                values, count = self._curve_values(
                    self.interpolators_[int(curve_index)], axis_query
                )
                curve_matrix[row] = values
                boundary_count += count
            predictions[index_array] = weights @ curve_matrix
        return predictions, {"boundary_hold_evaluations": boundary_count}


def fit_predict_curve_lut(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    spec,
    *,
    axis_col: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select and evaluate the curve-object PCHIP lookup-table baseline."""

    started = time.perf_counter()
    model = CurveTablePchipRegressor(
        axis_col=axis_col, input_cols=tuple(spec.input_cols)
    ).fit(train, _transformed_target(train, spec))
    validation_target = _transformed_target(validation, spec)
    candidates = ((1, 1.0), (2, 1.0), (4, 1.0), (4, 2.0), (8, 2.0))
    records: list[dict[str, object]] = []
    selected: tuple[float, int, float] | None = None
    for neighbor_curves, distance_power in candidates:
        prediction, diagnostics = model.predict_frame(
            validation,
            neighbor_curves=neighbor_curves,
            distance_power=distance_power,
        )
        mse = float(np.mean(np.square(prediction - validation_target)))
        records.append(
            {
                "neighbor_curves": min(neighbor_curves, model.n_curves_),
                "distance_power": distance_power,
                "validation_mse": mse,
                **diagnostics,
            }
        )
        candidate = (mse, neighbor_curves, distance_power)
        if selected is None or candidate[0] < selected[0]:
            selected = candidate
    assert selected is not None
    best_mse, best_neighbors, best_power = selected
    prediction, diagnostics = model.predict_frame(
        test,
        neighbor_curves=best_neighbors,
        distance_power=best_power,
    )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Curve LUT produced non-finite predictions")
    storage_values = model.stored_points_ + model.condition_values_.size
    return prediction, {
        "method_variant": "curve_object_pchip_lut_condition_idw",
        "param_count": int(storage_values),
        "stored_training_values": int(storage_values),
        "stored_curve_points": int(model.stored_points_),
        "stored_curves": int(model.n_curves_),
        "selected_neighbor_curves": int(min(best_neighbors, model.n_curves_)),
        "selected_distance_power": float(best_power),
        "best_validation_mse": float(best_mse),
        "candidate_validation": json.dumps(records, sort_keys=True),
        **diagnostics,
        "train_time_s": float(time.perf_counter() - started),
    }


class SemiEmpiricalCompactRegressor:
    """Ridge-calibrated library of compact-model mechanism features.

    The feature library represents reverse-bias field dependence, avalanche
    onset, low-pass/relaxation responses, temperature activation, optical
    power, and geometry/process modulation.  It is an explicit semi-empirical
    compact response surface, not a claim of reproducing a foundry model.
    """

    def __init__(self, *, task_name: str, axis_col: str, input_cols: Sequence[str]):
        if axis_col not in input_cols:
            raise ValueError(f"Axis {axis_col!r} is absent from the model inputs")
        self.task_name = str(task_name)
        self.axis_col = str(axis_col)
        self.input_cols = tuple(input_cols)

    def _fit_feature_state(self, frame: pd.DataFrame) -> None:
        values = frame[list(self.input_cols)].to_numpy(dtype=np.float64)
        self.input_scaler_ = StandardScaler().fit(values)
        self.positive_columns_ = tuple(
            column
            for column in self.input_cols
            if np.all(frame[column].to_numpy(dtype=np.float64) > 0.0)
        )
        self.log_scalers_ = {}
        for column in self.positive_columns_:
            logged = np.log10(frame[column].to_numpy(dtype=np.float64))[:, None]
            self.log_scalers_[column] = StandardScaler().fit(logged)

        axis = frame[self.axis_col].to_numpy(dtype=np.float64)
        self.axis_scale_ = max(float(np.max(np.abs(axis))), 1.0e-12)
        reverse = np.maximum(-axis / self.axis_scale_, 0.0)
        nonzero_reverse = np.unique(reverse[reverse > 0.0])
        if len(nonzero_reverse):
            self.reverse_knots_ = np.unique(
                np.quantile(nonzero_reverse, [0.35, 0.55, 0.70, 0.82, 0.90])
            )
        else:
            self.reverse_knots_ = np.empty(0, dtype=np.float64)

        frequency_columns = [
            column
            for column in self.input_cols
            if "frequency" in column.lower()
        ]
        self.frequency_cutoffs_ = {}
        for column in frequency_columns:
            raw = frame[column].to_numpy(dtype=np.float64)
            frequency = np.power(10.0, raw) if column.startswith("log_") else raw
            positive = np.unique(frequency[frequency > 0.0])
            if len(positive):
                self.frequency_cutoffs_[column] = np.geomspace(
                    max(float(positive.min()), 1.0e-12),
                    max(float(positive.max()), float(positive.min()) * 1.001),
                    6,
                )

    def _features(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        raw = frame[list(self.input_cols)].to_numpy(dtype=np.float64)
        z = self.input_scaler_.transform(raw)
        pieces = [np.ones((len(frame), 1), dtype=np.float64), z, np.square(z)]
        names = ["intercept"]
        names.extend(f"z:{column}" for column in self.input_cols)
        names.extend(f"z2:{column}" for column in self.input_cols)

        log_features: list[np.ndarray] = []
        log_names: list[str] = []
        for column in self.positive_columns_:
            logged = np.log10(frame[column].to_numpy(dtype=np.float64))[:, None]
            scaled = self.log_scalers_[column].transform(logged)
            log_features.append(scaled)
            log_names.append(f"log10:{column}")
        if log_features:
            pieces.append(np.hstack(log_features))
            names.extend(log_names)

        axis_index = self.input_cols.index(self.axis_col)
        axis = frame[self.axis_col].to_numpy(dtype=np.float64)
        reverse = np.maximum(-axis / self.axis_scale_, 0.0)
        reverse_features = [
            reverse,
            np.sqrt(reverse + 0.05) - math.sqrt(0.05),
            np.log1p(reverse),
            np.square(reverse),
            np.power(reverse, 3),
        ]
        reverse_names = ["reverse", "sqrt_reverse", "log_reverse", "reverse2", "reverse3"]
        for knot in self.reverse_knots_:
            hinge = np.maximum(reverse - knot, 0.0)
            reverse_features.extend((hinge, np.square(hinge)))
            reverse_names.extend((f"avalanche_hinge:{knot:.6g}", f"avalanche_hinge2:{knot:.6g}"))
        reverse_matrix = np.column_stack(reverse_features)
        pieces.append(reverse_matrix)
        names.extend(reverse_names)

        for column, cutoffs in self.frequency_cutoffs_.items():
            values = frame[column].to_numpy(dtype=np.float64)
            frequency = np.power(10.0, values) if column.startswith("log_") else values
            frequency = np.maximum(frequency, 1.0e-30)
            response_features = []
            response_names = []
            for cutoff in cutoffs:
                ratio2 = np.square(frequency / cutoff)
                response_features.extend(
                    (-np.log10(1.0 + ratio2), 1.0 / (1.0 + ratio2))
                )
                response_names.extend(
                    (f"lowpass_db:{column}:{cutoff:.6g}", f"relaxation:{column}:{cutoff:.6g}")
                )
            pieces.append(np.column_stack(response_features))
            names.extend(response_names)

        condition_indices = [
            index for index in range(len(self.input_cols)) if index != axis_index
        ]
        if condition_indices:
            interaction_base = reverse_matrix[:, : min(reverse_matrix.shape[1], 9)]
            interactions = np.einsum(
                "ni,nj->nij", interaction_base, z[:, condition_indices]
            ).reshape(len(frame), -1)
            pieces.append(interactions)
            for reverse_name in reverse_names[: interaction_base.shape[1]]:
                names.extend(
                    f"{reverse_name}*z:{self.input_cols[index]}"
                    for index in condition_indices
                )
        return np.hstack(pieces), names

    def fit(self, frame: pd.DataFrame, target: np.ndarray, *, alpha: float):
        target = np.asarray(target, dtype=np.float64).reshape(-1, 1)
        if len(frame) != len(target):
            raise ValueError("Semi-empirical model requires aligned training rows")
        self._fit_feature_state(frame)
        features, self.feature_names_ = self._features(frame)
        self.feature_scaler_ = StandardScaler().fit(features)
        self.target_scaler_ = StandardScaler().fit(target)
        self.model_ = Ridge(alpha=float(alpha), fit_intercept=False).fit(
            self.feature_scaler_.transform(features),
            self.target_scaler_.transform(target).reshape(-1),
        )
        self.alpha_ = float(alpha)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features, names = self._features(frame)
        if names != self.feature_names_:
            raise RuntimeError("Semi-empirical feature contract changed after fitting")
        scaled = self.model_.predict(self.feature_scaler_.transform(features))
        return self.target_scaler_.inverse_transform(scaled[:, None]).reshape(-1)


def fit_predict_semiempirical_compact(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    spec,
    *,
    axis_col: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Validation-select a task-specific semi-empirical compact model."""

    started = time.perf_counter()
    train_target = _transformed_target(train, spec)
    validation_target = _transformed_target(validation, spec)
    candidates = (1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0)
    records: list[dict[str, float]] = []
    selected: tuple[float, SemiEmpiricalCompactRegressor] | None = None
    for alpha in candidates:
        model = SemiEmpiricalCompactRegressor(
            task_name=spec.name,
            axis_col=axis_col,
            input_cols=tuple(spec.input_cols),
        ).fit(train, train_target, alpha=alpha)
        validation_prediction = model.predict(validation)
        mse = float(np.mean(np.square(validation_prediction - validation_target)))
        records.append({"alpha": alpha, "validation_mse": mse})
        if selected is None or mse < selected[0]:
            selected = (mse, model)
    assert selected is not None
    best_mse, model = selected
    prediction = model.predict(test)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Semi-empirical compact model produced non-finite predictions")
    return prediction, {
        "method_variant": "ridge_calibrated_semiempirical_compact_feature_library",
        "param_count": int(model.model_.coef_.size),
        "feature_count": int(len(model.feature_names_)),
        "selected_alpha": float(model.alpha_),
        "best_validation_mse": float(best_mse),
        "candidate_validation": json.dumps(records, sort_keys=True),
        "mechanism_features": json.dumps(model.feature_names_),
        "train_time_s": float(time.perf_counter() - started),
    }


class _SmoothCompactMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        layers: list[torch.nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend((torch.nn.Linear(previous, width), torch.nn.Tanh()))
            previous = width
        layers.append(torch.nn.Linear(previous, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _transformed_target(frame: pd.DataFrame, spec) -> np.ndarray:
    values = frame[spec.target_col].to_numpy(dtype=np.float64)
    if spec.use_log_transform:
        return np.log10(np.maximum(values, 1.0e-30))
    return values


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def train_autopinn_adapted(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    spec,
    *,
    axis_col: str,
    monotonic_sign: int | None,
    seed: int,
    device: torch.device,
    epochs: int = 300,
    learning_rate: float = 2.0e-3,
    candidates: Sequence[Sequence[int]] = ((16, 16), (32, 16), (32, 32)),
    smoothness_weight: float = 1.0e-5,
    monotonic_weight: float = 1.0e-2,
    collocation_size: int = 64,
    validation_frequency: int = 10,
    patience: int = 8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Train a physics-informed, automatically selected smooth NN adaptation.

    Architecture and checkpoint selection use the supplied validation groups.
    The physical adapter applies a second-derivative smoothness penalty on the
    swept axis and, where a direction is physically prescribed, a monotonicity
    penalty.  Test rows are evaluated once after selection.
    """

    if axis_col not in spec.input_cols:
        raise ValueError(f"Axis {axis_col!r} is absent from the model inputs")
    if monotonic_sign not in (None, -1, 1):
        raise ValueError("monotonic_sign must be None, -1, or 1")
    if epochs < validation_frequency or not candidates:
        raise ValueError("AutoPINN needs candidates and enough training epochs")
    started = time.perf_counter()
    x_scaler = StandardScaler().fit(
        train[list(spec.input_cols)].to_numpy(dtype=np.float64)
    )
    y_scaler = StandardScaler().fit(_transformed_target(train, spec).reshape(-1, 1))

    def scaled_x(frame: pd.DataFrame) -> np.ndarray:
        return x_scaler.transform(
            frame[list(spec.input_cols)].to_numpy(dtype=np.float64)
        ).astype(np.float32)

    x_train = scaled_x(train)
    x_validation = scaled_x(validation)
    x_test = scaled_x(test)
    y_train = y_scaler.transform(
        _transformed_target(train, spec).reshape(-1, 1)
    ).astype(np.float32)
    y_validation = y_scaler.transform(
        _transformed_target(validation, spec).reshape(-1, 1)
    ).astype(np.float32)
    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    x_validation_tensor = torch.from_numpy(x_validation).to(device)
    y_validation_tensor = torch.from_numpy(y_validation).to(device)
    x_test_tensor = torch.from_numpy(x_test).to(device)
    lower = torch.from_numpy(x_train.min(axis=0)).to(device)
    upper = torch.from_numpy(x_train.max(axis=0)).to(device)
    axis_index = list(spec.input_cols).index(axis_col)
    criterion = torch.nn.MSELoss()
    candidate_records: list[dict[str, object]] = []
    selected: tuple[float, _SmoothCompactMLP, tuple[int, ...], int] | None = None

    for candidate_index, hidden_values in enumerate(candidates):
        hidden = tuple(int(value) for value in hidden_values)
        candidate_seed = int(seed + 1009 * candidate_index)
        torch.manual_seed(candidate_seed)
        np.random.seed(candidate_seed)
        model = _SmoothCompactMLP(x_train.shape[1], hidden, candidate_seed).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = math.inf
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        checks_without_improvement = 0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train_tensor)
            data_loss = criterion(prediction, y_train_tensor)
            random_unit = torch.rand(
                (min(collocation_size, len(train)), x_train.shape[1]), device=device
            )
            collocation = (lower + random_unit * (upper - lower)).requires_grad_(True)
            collocation_prediction = model(collocation)
            first_all = torch.autograd.grad(
                collocation_prediction.sum(), collocation, create_graph=True
            )[0]
            first_axis = first_all[:, axis_index]
            second_axis = torch.autograd.grad(
                first_axis.sum(), collocation, create_graph=True
            )[0][:, axis_index]
            physics_loss = smoothness_weight * torch.mean(second_axis.square())
            if monotonic_sign is not None:
                violation = torch.relu(-float(monotonic_sign) * first_axis)
                physics_loss = physics_loss + monotonic_weight * torch.mean(
                    violation.square()
                )
            loss = data_loss + physics_loss
            loss.backward()
            optimizer.step()

            if epoch % validation_frequency == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    validation_loss = float(
                        criterion(model(x_validation_tensor), y_validation_tensor).item()
                    )
                if validation_loss < best_loss - 1.0e-8:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    break
        model.load_state_dict(best_state)
        candidate_records.append(
            {
                "hidden": "x".join(map(str, hidden)),
                "validation_mse_scaled": best_loss,
                "best_epoch": best_epoch,
                "param_count": _parameter_count(model),
            }
        )
        candidate_result = (best_loss, model, hidden, best_epoch)
        if selected is None or candidate_result[0] < selected[0]:
            selected = candidate_result

    assert selected is not None
    best_loss, best_model, best_hidden, best_epoch = selected
    best_model.eval()
    diagnostic_input = x_test_tensor.detach().clone().requires_grad_(True)
    diagnostic_output = best_model(diagnostic_input)
    diagnostic_gradient = torch.autograd.grad(
        diagnostic_output.sum(), diagnostic_input, create_graph=True
    )[0][:, axis_index]
    diagnostic_curvature = torch.autograd.grad(
        diagnostic_gradient.sum(), diagnostic_input
    )[0][:, axis_index]
    prediction_scaled = diagnostic_output.detach().cpu().numpy()
    gradient_values = diagnostic_gradient.detach().cpu().numpy()
    curvature_values = diagnostic_curvature.detach().cpu().numpy()
    prediction = y_scaler.inverse_transform(prediction_scaled).reshape(-1)
    if not np.isfinite(prediction).all():
        raise RuntimeError("AutoPINN-adapted produced non-finite predictions")
    return prediction, {
        "method_variant": "autopinn_adapted_smooth_monotone_architecture_search",
        "param_count": _parameter_count(best_model),
        "selected_hidden": "x".join(map(str, best_hidden)),
        "best_epoch": int(best_epoch),
        "best_validation_mse_scaled": float(best_loss),
        "candidate_validation": json.dumps(candidate_records, sort_keys=True),
        "constraint_axis": axis_col,
        "monotonic_sign": monotonic_sign,
        "test_axis_gradient_abs_p95_scaled": float(
            np.percentile(np.abs(gradient_values), 95)
        ),
        "test_axis_curvature_abs_p95_scaled": float(
            np.percentile(np.abs(curvature_values), 95)
        ),
        "test_monotonic_violation_fraction": (
            float(np.mean(float(monotonic_sign) * gradient_values < -1.0e-6))
            if monotonic_sign is not None
            else None
        ),
        "train_time_s": float(time.perf_counter() - started),
    }
