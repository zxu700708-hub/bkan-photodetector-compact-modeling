"""Audit one unified, output-aware KAN symbolification procedure on all tasks.

This is a development experiment.  It does not edit the manuscript or replace
the frozen symbolic exports.  Every task uses the same primitive library,
pruning thresholds, candidate fitter, progressive replacement rule, and final
symbolic-affine refit.  Task-specific information is limited to the data,
grouped split, input columns, and target transform already registered by the
benchmark.

Two extraction policies share exactly the same fitted edge proposals:

* ``local_edgewise`` chooses each edge by its isolated activation-fit R2.
* ``output_aware`` greedily chooses the edge/primitive pair that minimizes the
  complete network's TCAD validation RMSE.
* ``output_aware_anchored`` adds complete-output fidelity to the frozen teacher
  so early hybrid replacements cannot compensate for numeric edges removed
  later in the trajectory.
* ``output_aware_screened`` additionally limits each edge to locally plausible
  primitives before applying the anchored complete-output criterion.
* ``global_soft_hardened`` jointly trains a soft primitive mixture over every
  retained edge, hardens the whole graph at once, and refines it with
  fully-symbolic coordinate rollback.
* ``global_soft_safeguarded`` also retains the local symbolic solution and
  switches to the global formula only when its validation improvement clears
  a common effect-size threshold and a grouped bootstrap stability check.

All policies end in a fully symbolic graph.  By default, calibration groups
select the replacement trajectory, validation groups stop the final refit, and
test groups are opened once.  The exported SymPy expression is reloaded from
JSON and evaluated without a KAN for the replay audit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
import sympy
import torch
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_multi_teacher_symbolic_pareto as shared_audit  # noqa: E402
from device_modeling.photodetector.bayesian_modeler import (  # noqa: E402
    BayesKANDeviceModeler,
)
from device_modeling.photodetector import task_config as task_config  # noqa: E402
from kan import KAN  # noqa: E402
from kan.utils import SYMBOLIC_LIB, fit_params  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts/results/output_aware_progressive_symbolification"
DEFAULT_DATA = ROOT / "artifacts/results/device_modeling/cleaned_data.csv"
DEFAULT_NET_ROOT = ROOT / "artifacts/results/net_photocurrent_retrain"
DEFAULT_TASKS = tuple(shared_audit.TASK_KEY)
DEFAULT_SOURCES = ("bkan", "dkan")
DEFAULT_VARIANTS = (
    "local_edgewise",
    "output_aware",
    "output_aware_anchored",
    "output_aware_screened",
    "global_soft_hardened",
    "global_soft_safeguarded",
)
DEFAULT_LIBRARY = ("0", "x", "x^2", "x^3", "exp", "tanh", "arctan", "gaussian")
CAPACITANCE_LEGACY_SCALE = shared_audit.CAPACITANCE_LEGACY_SCALE
PAPER_FILES = (
    ROOT / "paper/main_manuscript.tex",
    ROOT / "paper/supplementary_information.tex",
)


@dataclass(frozen=True)
class EdgeProposal:
    name: str
    params: tuple[float, float, float, float]
    local_r2: float
    complexity: float


@dataclass
class ModelBundle:
    source: str
    model: KAN
    x_scaler: object
    y_scaler: object
    output_scale: float
    tensors: dict[str, torch.Tensor]
    targets_scaled: dict[str, torch.Tensor]
    targets_physical: dict[str, np.ndarray]
    teacher_predictions: dict[str, np.ndarray]
    metadata: dict


@dataclass(frozen=True)
class ConstraintDescriptor:
    axis_index: int
    monotonic_direction: int
    monotonic_confidence: float
    reference_side: str | None
    reference_target_scaled: float | None
    reference_std_scaled: float | None
    lower_bound_scaled: float
    upper_bound_scaled: float


@dataclass(frozen=True)
class ConstraintBatch:
    lower_input: torch.Tensor
    upper_input: torch.Tensor
    reference_input: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--net-photocurrent-data",
        type=Path,
        default=DEFAULT_NET_ROOT / "device_modeling/cleaned_data.csv",
    )
    parser.add_argument(
        "--capacitance-data",
        type=Path,
        default=task_config.DEFAULT_CAPACITANCE_DATA,
    )
    parser.add_argument("--bkan-root", type=Path, default=shared_audit.DEFAULT_BKAN)
    parser.add_argument(
        "--capacitance-bkan-root",
        type=Path,
        default=shared_audit.DEFAULT_CAP_BKAN,
    )
    parser.add_argument("--net-root", type=Path, default=DEFAULT_NET_ROOT)
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--sources", nargs="+", choices=DEFAULT_SOURCES, default=list(DEFAULT_SOURCES)
    )
    parser.add_argument(
        "--variants", nargs="+", choices=DEFAULT_VARIANTS, default=list(DEFAULT_VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--library", nargs="+", choices=tuple(SYMBOLIC_LIB), default=list(DEFAULT_LIBRARY))
    parser.add_argument("--dkan-steps", type=int, default=300)
    parser.add_argument("--node-threshold", type=float, default=0.01)
    parser.add_argument("--edge-threshold", type=float, default=0.03)
    parser.add_argument("--proposal-points", type=int, default=4096)
    parser.add_argument("--proposal-grid", type=int, default=11)
    parser.add_argument("--proposal-iterations", type=int, default=2)
    parser.add_argument("--proposal-range", type=float, default=3.0)
    parser.add_argument("--local-complexity-weight", type=float, default=1e-3)
    parser.add_argument("--output-complexity-weight", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--local-r2-gap", type=float, default=0.02)
    parser.add_argument(
        "--selection-split",
        choices=("validation", "calibration", "both"),
        default="calibration",
    )
    parser.add_argument("--refit-steps", type=int, default=200)
    parser.add_argument("--refit-lr", type=float, default=1e-3)
    parser.add_argument("--refit-batch", type=int, default=2048)
    parser.add_argument("--refit-every", type=int, default=10)
    parser.add_argument("--refit-patience", type=int, default=6)
    parser.add_argument("--soft-steps", type=int, default=300)
    parser.add_argument("--soft-batch", type=int, default=2048)
    parser.add_argument("--soft-gate-lr", type=float, default=0.02)
    parser.add_argument("--soft-affine-lr", type=float, default=0.001)
    parser.add_argument("--soft-temperature-start", type=float, default=2.0)
    parser.add_argument("--soft-temperature-end", type=float, default=0.10)
    parser.add_argument("--soft-every", type=int, default=10)
    parser.add_argument("--soft-patience", type=int, default=8)
    parser.add_argument("--soft-complexity-weight", type=float, default=1e-3)
    parser.add_argument("--soft-entropy-weight", type=float, default=1e-3)
    parser.add_argument("--soft-affine-anchor-weight", type=float, default=1e-4)
    parser.add_argument("--constraint-weight", type=float, default=0.2)
    parser.add_argument("--reference-weight", type=float, default=0.2)
    parser.add_argument("--bound-weight", type=float, default=0.02)
    parser.add_argument("--monotonic-confidence", type=float, default=0.90)
    parser.add_argument("--reference-std-threshold", type=float, default=0.03)
    parser.add_argument("--hardening-sweeps", type=int, default=2)
    parser.add_argument("--hardening-topk", type=int, default=4)
    parser.add_argument("--hardening-tolerance", type=float, default=1e-7)
    parser.add_argument("--safeguard-min-relative-improvement", type=float, default=0.05)
    parser.add_argument("--safeguard-bootstrap-samples", type=int, default=2000)
    parser.add_argument("--safeguard-confidence", type=float, default=0.90)
    parser.add_argument("--safeguard-min-group-win-fraction", type=float, default=0.50)
    parser.add_argument("--dense-points", type=int, default=2000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "output",
        "data",
        "net_photocurrent_data",
        "capacitance_data",
        "bkan_root",
        "capacitance_bkan_root",
        "net_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for path in (args.data, args.net_photocurrent_data, args.capacitance_data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Tasks must be unique")
    if len(set(args.sources)) != len(args.sources):
        raise ValueError("Sources must be unique")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Variants must be unique")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if len(set(args.library)) != len(args.library):
        raise ValueError("Primitive names must be unique")
    if "0" not in args.library or "x" not in args.library:
        raise ValueError("The common library must contain 0 and x")
    if args.dkan_steps < 1 or args.refit_steps < 0:
        raise ValueError("Training step counts must be nonnegative, with DKAN steps positive")
    if args.proposal_points < 32 or args.dense_points < 100:
        raise ValueError("Proposal and dense sample counts are too small")
    if args.proposal_grid < 3 or args.proposal_iterations < 1:
        raise ValueError("Candidate grid settings are invalid")
    if args.refit_batch < 1 or args.refit_every < 1 or args.refit_patience < 1:
        raise ValueError("Refit controls must be positive")
    if args.soft_steps < 1 or args.soft_batch < 1:
        raise ValueError("Soft-gate step and batch counts must be positive")
    if args.soft_every < 1 or args.soft_patience < 1:
        raise ValueError("Soft-gate validation controls must be positive")
    if args.soft_gate_lr <= 0.0 or args.soft_affine_lr <= 0.0:
        raise ValueError("Soft-gate learning rates must be positive")
    if args.soft_temperature_start <= 0.0 or args.soft_temperature_end <= 0.0:
        raise ValueError("Soft-gate temperatures must be positive")
    if args.soft_temperature_start < args.soft_temperature_end:
        raise ValueError("Soft-gate temperature must anneal downward")
    for name in (
        "soft_complexity_weight",
        "soft_entropy_weight",
        "soft_affine_anchor_weight",
        "constraint_weight",
        "reference_weight",
        "bound_weight",
        "hardening_tolerance",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if not 0.5 <= args.monotonic_confidence <= 1.0:
        raise ValueError("Monotonic confidence must lie in [0.5, 1]")
    if args.reference_std_threshold < 0.0:
        raise ValueError("Reference standard-deviation threshold must be nonnegative")
    if args.hardening_sweeps < 0 or args.hardening_topk < 1:
        raise ValueError("Hardening controls are invalid")
    if not 0.0 <= args.safeguard_min_relative_improvement < 1.0:
        raise ValueError("Safeguard relative improvement must lie in [0, 1)")
    if args.safeguard_bootstrap_samples < 100:
        raise ValueError("Safeguard bootstrap requires at least 100 samples")
    if not 0.5 < args.safeguard_confidence < 1.0:
        raise ValueError("Safeguard confidence must lie in (0.5, 1)")
    if not 0.5 <= args.safeguard_min_group_win_fraction <= 1.0:
        raise ValueError("Safeguard group-win fraction must lie in [0.5, 1]")
    if args.anchor_weight < 0.0:
        raise ValueError("Anchor weight must be nonnegative")
    if args.local_r2_gap < 0.0:
        raise ValueError("Local R2 gap must be nonnegative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_args(args: argparse.Namespace, task: str, seed: int) -> SimpleNamespace:
    photo = task == "I_photo"
    return SimpleNamespace(
        data=args.net_photocurrent_data if photo else args.data,
        capacitance_data=args.capacitance_data,
        bkan_root=(args.net_root / "uq_repeated_grouped" if photo else args.bkan_root),
        capacitance_bkan_root=args.capacitance_bkan_root,
        seed=seed,
    )


def _load_task(args: argparse.Namespace, task: str, seed: int):
    task_args = _task_args(args, task, seed)
    frame, inputs, specification, baseline_spec = shared_audit._load_task(task, task_args)
    partitions = task_config.split_task_dataframe(
        frame, specification, seed, fractions=(0.65, 0.10, 0.15, 0.10)
    )
    return task_args, frame, inputs, specification, baseline_spec, partitions


def _physical_targets(partitions: Iterable[pd.DataFrame], baseline_spec) -> dict[str, np.ndarray]:
    return {
        name: shared_audit._target(frame, baseline_spec)
        for name, frame in zip(("train", "validation", "calibration", "test"), partitions)
    }


def _tensor_inputs(partitions, inputs, scaler) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(
            scaler.transform(frame[list(inputs)].to_numpy(dtype=np.float32)).astype(np.float32)
        ).float()
        for name, frame in zip(("train", "validation", "calibration", "test"), partitions)
    }


def _scaled_targets(
    physical: dict[str, np.ndarray], y_scaler, output_scale: float
) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(
            y_scaler.transform((values / output_scale).reshape(-1, 1)).astype(np.float32)
        ).float()
        for name, values in physical.items()
    }


def _selection_tensors(
    bundle: ModelBundle, selection_split: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if selection_split == "both":
        return (
            torch.cat(
                [bundle.tensors["validation"], bundle.tensors["calibration"]], dim=0
            ),
            torch.cat(
                [
                    bundle.targets_scaled["validation"],
                    bundle.targets_scaled["calibration"],
                ],
                dim=0,
            ),
        )
    return bundle.tensors[selection_split], bundle.targets_scaled[selection_split]


def _model_output(model: KAN, values: torch.Tensor) -> torch.Tensor:
    output = model(values)
    if isinstance(output, tuple):
        output = output[0]
    return output.reshape(-1, 1)


def _to_physical(bundle: ModelBundle, scaled: np.ndarray) -> np.ndarray:
    values = bundle.y_scaler.inverse_transform(np.asarray(scaled).reshape(-1, 1)).reshape(-1)
    return values.astype(np.float64) * bundle.output_scale


def _predict_physical(bundle: ModelBundle, model: KAN, split: str) -> np.ndarray:
    with torch.no_grad():
        scaled = _model_output(model, bundle.tensors[split]).detach().cpu().numpy()
    return _to_physical(bundle, scaled)


def _load_bkan_bundle(
    args: argparse.Namespace,
    task: str,
    seed: int,
    task_args,
    inputs,
    specification,
    partitions,
    targets_physical,
) -> ModelBundle:
    checkpoint = shared_audit._bkan_checkpoint(task, seed, specification, task_args)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    modeler = BayesKANDeviceModeler.load_model(str(checkpoint), device="cpu")
    if list(modeler.input_cols) != list(inputs):
        raise RuntimeError(f"BKAN input mismatch for {task}")
    train = partitions[0]
    observed_mean = train[list(inputs)].to_numpy(dtype=np.float64).mean(axis=0)
    expected_mean = np.asarray(modeler.scaler_x.mean_, dtype=np.float64)
    scaler_delta = float(
        np.max(np.abs(observed_mean - expected_mean) / np.maximum(np.abs(expected_mean), 1.0))
    )
    if scaler_delta > 1e-7:
        raise RuntimeError(f"BKAN checkpoint/split mismatch for {task}: {scaler_delta:.3g}")
    output_scale = 1.0
    if task == "Capacitance":
        observed_y_mean = float(np.mean(targets_physical["train"]))
        checkpoint_y_mean = float(np.asarray(modeler.scaler_y.mean_).reshape(-1)[0])
        ratio = observed_y_mean / checkpoint_y_mean
        if np.isclose(ratio, CAPACITANCE_LEGACY_SCALE, rtol=1e-7, atol=0.0):
            output_scale = CAPACITANCE_LEGACY_SCALE
        elif not np.isclose(ratio, 1.0, rtol=1e-7, atol=0.0):
            raise RuntimeError(f"Unregistered capacitance scale ratio: {ratio:.12g}")
    model = modeler.model.to_deterministic_kan(
        device="cpu", symbolic_enabled=True, auto_save=False
    ).eval()
    tensors = _tensor_inputs(partitions, inputs, modeler.scaler_x)
    targets_scaled = _scaled_targets(targets_physical, modeler.scaler_y, output_scale)
    bundle = ModelBundle(
        source="bkan",
        model=model,
        x_scaler=modeler.scaler_x,
        y_scaler=modeler.scaler_y,
        output_scale=output_scale,
        tensors=tensors,
        targets_scaled=targets_scaled,
        targets_physical=targets_physical,
        teacher_predictions={},
        metadata={
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "scaler_mean_max_relative_delta": scaler_delta,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        },
    )
    bundle.teacher_predictions = {
        split: _predict_physical(bundle, model, split)
        for split in ("train", "validation", "test")
    }
    return bundle


def _fit_dkan_bundle(
    args: argparse.Namespace,
    task: str,
    seed: int,
    inputs,
    partitions,
    baseline_spec,
    targets_physical,
) -> ModelBundle:
    train, validation = partitions[:2]
    x_scaler, y_scaler, x_train, y_train = shared_audit._scaled_training_arrays(
        train, inputs, baseline_spec
    )
    x_validation = x_scaler.transform(
        validation[list(inputs)].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    y_validation = y_scaler.transform(
        targets_physical["validation"].reshape(-1, 1)
    ).astype(np.float32)
    shared_audit.set_seed(seed)
    model = KAN(
        width=[len(inputs), 8, 1],
        grid=8,
        k=3,
        seed=seed,
        device="cpu",
        symbolic_enabled=True,
        auto_save=False,
    )
    dataset = {
        "train_input": torch.from_numpy(x_train).float(),
        "train_label": torch.from_numpy(y_train).float(),
        "test_input": torch.from_numpy(x_validation).float(),
        "test_label": torch.from_numpy(y_validation).float(),
    }
    batch = min(256, len(x_train), len(x_validation))
    started = time.perf_counter()
    model.fit(
        dataset,
        opt="Adam",
        steps=args.dkan_steps,
        lr=0.002,
        lamb=0.001,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.1,
        lamb_coefdiff=0.1,
        batch=batch,
        log=args.dkan_steps + 1,
    )
    train_time = time.perf_counter() - started
    model.eval()
    tensors = _tensor_inputs(partitions, inputs, x_scaler)
    targets_scaled = _scaled_targets(targets_physical, y_scaler, 1.0)
    bundle = ModelBundle(
        source="dkan",
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        output_scale=1.0,
        tensors=tensors,
        targets_scaled=targets_scaled,
        targets_physical=targets_physical,
        teacher_predictions={},
        metadata={
            "train_time_s": train_time,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        },
    )
    bundle.teacher_predictions = {
        split: _predict_physical(bundle, model, split)
        for split in ("train", "validation", "test")
    }
    return bundle


def _active_edges(model: KAN) -> list[tuple[int, int, int]]:
    edges = []
    for layer_index, layer in enumerate(model.act_fun):
        for input_index in range(layer.mask.shape[0]):
            for output_index in range(layer.mask.shape[1]):
                if float(layer.mask[input_index, output_index].detach().cpu()) > 0.0:
                    edges.append((layer_index, input_index, output_index))
    return edges


def _clone_numeric_model(model: KAN) -> KAN:
    clone = KAN(
        width=copy.deepcopy(model.width),
        grid=copy.deepcopy(model.grid),
        k=copy.deepcopy(model.k),
        mult_arity=copy.deepcopy(model.mult_arity),
        base_fun=model.base_fun_name,
        symbolic_enabled=True,
        affine_trainable=model.affine_trainable,
        grid_eps=model.grid_eps,
        grid_range=copy.deepcopy(model.grid_range),
        sp_trainable=model.sp_trainable,
        sb_trainable=model.sb_trainable,
        seed=0,
        save_act=True,
        auto_save=False,
        first_init=False,
        device=model.device,
    )
    clone.load_state_dict(model.state_dict())
    return clone.eval()


def _prepare_numeric_graph(bundle: ModelBundle, args: argparse.Namespace) -> KAN:
    model = _clone_numeric_model(bundle.model)
    with torch.no_grad():
        _model_output(model, bundle.tensors["train"])
    model.attribute()
    if any(int(width[1]) != 0 for width in model.width):
        raise RuntimeError("This audit currently requires additive KAN nodes")
    original_masks = [layer.mask.detach().clone() for layer in model.act_fun]
    for hidden_layer in range(1, len(model.width) - 1):
        keep = model.node_scores[hidden_layer] > args.node_threshold
        model.act_fun[hidden_layer - 1].mask.data[:, ~keep] = 0.0
        model.act_fun[hidden_layer].mask.data[~keep, :] = 0.0
    model.prune_edge(threshold=args.edge_threshold, log_history=False)
    if len(model.act_fun) != 2 or model.act_fun[1].mask.shape[1] != 1:
        raise RuntimeError("Path-preserving pruning expects one hidden layer and one output")
    first_scores = model.edge_scores[0].detach()
    second_scores = model.edge_scores[1].detach().reshape(-1)
    for input_index in range(model.act_fun[0].mask.shape[0]):
        admissible = original_masks[0][input_index] * original_masks[1][:, 0]
        path_scores = first_scores[:, input_index] * second_scores * admissible
        hidden_index = int(torch.argmax(path_scores).detach().cpu())
        model.act_fun[0].mask.data[input_index, hidden_index] = original_masks[0][
            input_index, hidden_index
        ]
        model.act_fun[1].mask.data[hidden_index, 0] = original_masks[1][hidden_index, 0]
    model.auto_save = False
    model.eval()
    return model


def _protected_input_paths(model: KAN) -> set[tuple[int, int, int]]:
    first_scores = model.edge_scores[0].detach()
    second_scores = model.edge_scores[1].detach().reshape(-1)
    protected: set[tuple[int, int, int]] = set()
    for input_index in range(model.act_fun[0].mask.shape[0]):
        active = model.act_fun[0].mask[input_index] * model.act_fun[1].mask[:, 0]
        path_scores = first_scores[:, input_index] * second_scores * active
        hidden_index = int(torch.argmax(path_scores).detach().cpu())
        protected.add((0, input_index, hidden_index))
        protected.add((1, hidden_index, 0))
    return protected


def _proposal_sample(values: torch.Tensor, maximum: int, seed: int) -> torch.Tensor:
    if len(values) <= maximum:
        return values
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(values), size=maximum, replace=False))
    return values[torch.from_numpy(indices).long()]


def _fit_proposals(
    model: KAN,
    train_input: torch.Tensor,
    library: list[str],
    args: argparse.Namespace,
) -> dict[tuple[int, int, int], list[EdgeProposal]]:
    with torch.no_grad():
        _model_output(model, train_input)
    proposals: dict[tuple[int, int, int], list[EdgeProposal]] = {}
    span = float(args.proposal_range)
    for edge in _active_edges(model):
        layer_index, input_index, output_index = edge
        x = model.acts[layer_index][:, input_index].detach()
        y = model.spline_postacts[layer_index][:, output_index, input_index].detach()
        edge_proposals: list[EdgeProposal] = []
        for name in library:
            complexity = float(SYMBOLIC_LIB[name][2])
            if name == "0":
                edge_proposals.append(EdgeProposal(name, (1.0, 0.0, 0.0, 0.0), 0.0, complexity))
                continue
            try:
                params, r2 = fit_params(
                    x,
                    y,
                    SYMBOLIC_LIB[name][0],
                    a_range=(-span, span),
                    b_range=(-span, span),
                    grid_number=args.proposal_grid,
                    iteration=args.proposal_iterations,
                    verbose=False,
                    device=model.device,
                )
                params_np = params.detach().cpu().numpy().astype(np.float64)
                r2_value = float(r2.detach().cpu())
                if np.all(np.isfinite(params_np)) and np.isfinite(r2_value):
                    edge_proposals.append(
                        EdgeProposal(name, tuple(float(value) for value in params_np), r2_value, complexity)
                    )
            except Exception:
                continue
        if not any(proposal.name == "x" for proposal in edge_proposals):
            raise RuntimeError(f"Identity proposal failed for edge {edge}")
        proposals[edge] = edge_proposals
    return proposals


def _apply_proposal(model: KAN, edge: tuple[int, int, int], proposal: EdgeProposal) -> None:
    layer_index, input_index, output_index = edge
    model.fix_symbolic(
        layer_index,
        input_index,
        output_index,
        proposal.name,
        fit_params_bool=False,
        verbose=False,
        log_history=False,
    )
    model.symbolic_fun[layer_index].affine.data[output_index, input_index] = torch.tensor(
        proposal.params,
        dtype=model.symbolic_fun[layer_index].affine.dtype,
        device=model.device,
    )


def _restore_numeric(model: KAN, edge: tuple[int, int, int]) -> None:
    model.set_mode(*edge, mode="n")


def _scaled_rmse(model: KAN, values: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        prediction = _model_output(model, values)
    if not bool(torch.isfinite(prediction).all()):
        return math.inf
    return float(torch.sqrt(torch.mean((prediction - target) ** 2)).detach().cpu())


def _select_local(
    model: KAN,
    proposals: dict[tuple[int, int, int], list[EdgeProposal]],
    protected_edges: set[tuple[int, int, int]],
    args: argparse.Namespace,
) -> list[dict]:
    trajectory = []
    for step, edge in enumerate(sorted(proposals), start=1):
        candidates = [
            proposal
            for proposal in proposals[edge]
            if proposal.name != "0" or edge not in protected_edges
        ]
        chosen = min(
            candidates,
            key=lambda item: (
                1.0 - float(np.clip(item.local_r2, -1.0, 1.0))
                + args.local_complexity_weight * item.complexity,
                item.complexity,
                item.name,
            ),
        )
        _apply_proposal(model, edge, chosen)
        trajectory.append(
            {
                "step": step,
                "edge": list(edge),
                "primitive": chosen.name,
                "local_r2": chosen.local_r2,
                "complexity": chosen.complexity,
            }
        )
    return trajectory


def _select_output_aware(
    model: KAN,
    proposals: dict[tuple[int, int, int], list[EdgeProposal]],
    validation_input: torch.Tensor,
    validation_target: torch.Tensor,
    protected_edges: set[tuple[int, int, int]],
    args: argparse.Namespace,
    reference_prediction: torch.Tensor | None = None,
    local_r2_gap: float | None = None,
) -> list[dict]:
    remaining = set(proposals)
    trajectory = []
    while remaining:
        best = None
        for edge in sorted(remaining):
            best_local_r2 = max(proposal.local_r2 for proposal in proposals[edge])
            for proposal in proposals[edge]:
                if proposal.name == "0" and edge in protected_edges:
                    continue
                if (
                    local_r2_gap is not None
                    and proposal.local_r2 < best_local_r2 - local_r2_gap
                ):
                    continue
                _apply_proposal(model, edge, proposal)
                with torch.no_grad():
                    prediction = _model_output(model, validation_input)
                if bool(torch.isfinite(prediction).all()):
                    rmse = float(
                        torch.sqrt(torch.mean((prediction - validation_target) ** 2))
                        .detach()
                        .cpu()
                    )
                    fidelity_rmse = (
                        float(
                            torch.sqrt(torch.mean((prediction - reference_prediction) ** 2))
                            .detach()
                            .cpu()
                        )
                        if reference_prediction is not None
                        else 0.0
                    )
                else:
                    rmse = math.inf
                    fidelity_rmse = math.inf
                _restore_numeric(model, edge)
                score = (
                    rmse
                    + args.anchor_weight * fidelity_rmse
                    + args.output_complexity_weight * proposal.complexity
                )
                candidate = (
                    score,
                    rmse,
                    fidelity_rmse,
                    proposal.complexity,
                    edge,
                    proposal.name,
                    proposal,
                )
                if best is None or candidate[:6] < best[:6]:
                    best = candidate
        if best is None or not np.isfinite(best[1]):
            raise RuntimeError("No finite output-aware replacement remains")
        score, rmse, fidelity_rmse, _, edge, _, proposal = best
        _apply_proposal(model, edge, proposal)
        remaining.remove(edge)
        trajectory.append(
            {
                "step": len(trajectory) + 1,
                "edge": list(edge),
                "primitive": proposal.name,
                "local_r2": proposal.local_r2,
                "complexity": proposal.complexity,
                "validation_scaled_rmse": rmse,
                "reference_fidelity_scaled_rmse": fidelity_rmse,
                "selection_score": score,
            }
        )
    return trajectory


def _curve_indices(
    frame: pd.DataFrame,
    inputs: tuple[str, ...],
    axis_col: str,
) -> list[np.ndarray]:
    condition_columns = [column for column in inputs if column != axis_col]
    if condition_columns:
        labels = pd.util.hash_pandas_object(
            frame[condition_columns], index=False
        ).to_numpy()
    else:
        labels = np.zeros(len(frame), dtype=np.int64)
    axis = frame[axis_col].to_numpy(dtype=np.float64)
    groups = []
    for label in pd.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            continue
        groups.append(indices[np.argsort(axis[indices], kind="stable")])
    return groups


def _infer_constraint_descriptor(
    frame: pd.DataFrame,
    values: torch.Tensor,
    targets: torch.Tensor,
    inputs: tuple[str, ...],
    specification,
    args: argparse.Namespace,
) -> ConstraintDescriptor:
    del values
    groups = _curve_indices(frame, inputs, specification.axis_col)
    target = targets.detach().cpu().numpy().reshape(-1).astype(np.float64)
    differences = []
    for indices in groups:
        differences.extend(np.diff(target[indices]).tolist())
    differences_array = np.asarray(differences, dtype=np.float64)
    tolerance = max(1e-8, 1e-6 * float(np.ptp(target)))
    active = differences_array[np.abs(differences_array) > tolerance]
    positive_fraction = float(np.mean(active > 0.0)) if len(active) else 0.5
    negative_fraction = float(np.mean(active < 0.0)) if len(active) else 0.5
    confidence = max(positive_fraction, negative_fraction)
    direction = 0
    if len(active) and confidence >= args.monotonic_confidence:
        direction = 1 if positive_fraction >= negative_fraction else -1

    reference_side = None
    reference_target = None
    reference_std = None
    endpoint_candidates = []
    if len(groups) >= 4:
        for side, position in (("lower", 0), ("upper", -1)):
            endpoint_values = np.asarray(
                [target[indices[position]] for indices in groups], dtype=np.float64
            )
            endpoint_candidates.append(
                (
                    float(np.std(endpoint_values, ddof=1)),
                    side,
                    float(np.mean(endpoint_values)),
                )
            )
        endpoint_candidates.sort()
        candidate_std, candidate_side, candidate_mean = endpoint_candidates[0]
        if candidate_std <= args.reference_std_threshold:
            reference_side = candidate_side
            reference_target = candidate_mean
            reference_std = candidate_std

    target_range = max(float(np.ptp(target)), 1e-6)
    margin = 0.05 * target_range
    return ConstraintDescriptor(
        axis_index=inputs.index(specification.axis_col),
        monotonic_direction=direction,
        monotonic_confidence=confidence,
        reference_side=reference_side,
        reference_target_scaled=reference_target,
        reference_std_scaled=reference_std,
        lower_bound_scaled=float(np.min(target) - margin),
        upper_bound_scaled=float(np.max(target) + margin),
    )


def _constraint_batch(
    frame: pd.DataFrame,
    values: torch.Tensor,
    inputs: tuple[str, ...],
    specification,
    descriptor: ConstraintDescriptor,
) -> ConstraintBatch:
    groups = _curve_indices(frame, inputs, specification.axis_col)
    lower_indices = []
    upper_indices = []
    if descriptor.monotonic_direction != 0:
        for indices in groups:
            lower_indices.extend(indices[:-1].tolist())
            upper_indices.extend(indices[1:].tolist())
    reference_indices = []
    if descriptor.reference_side is not None:
        position = 0 if descriptor.reference_side == "lower" else -1
        reference_indices = [int(indices[position]) for indices in groups]
    feature_count = values.shape[1]

    def rows(indices: list[int]) -> torch.Tensor:
        if not indices:
            return torch.empty((0, feature_count), dtype=values.dtype)
        return values[torch.tensor(indices, dtype=torch.long)]

    return ConstraintBatch(
        lower_input=rows(lower_indices),
        upper_input=rows(upper_indices),
        reference_input=rows(reference_indices),
    )


def _limit_constraint_batch(
    batch: ConstraintBatch,
    maximum: int,
    generator: np.random.Generator | None = None,
) -> ConstraintBatch:
    def select(values: torch.Tensor, paired_indices: np.ndarray | None = None):
        if len(values) <= maximum:
            return values
        if paired_indices is not None:
            return values[torch.from_numpy(paired_indices).long()]
        if generator is None:
            indices = np.linspace(0, len(values) - 1, maximum).round().astype(int)
        else:
            indices = np.sort(generator.choice(len(values), maximum, replace=False))
        return values[torch.from_numpy(indices).long()]

    pair_indices = None
    if len(batch.lower_input) > maximum:
        if generator is None:
            pair_indices = np.linspace(
                0, len(batch.lower_input) - 1, maximum
            ).round().astype(int)
        else:
            pair_indices = np.sort(
                generator.choice(len(batch.lower_input), maximum, replace=False)
            )
    return ConstraintBatch(
        lower_input=select(batch.lower_input, pair_indices),
        upper_input=select(batch.upper_input, pair_indices),
        reference_input=select(batch.reference_input),
    )


class SoftPrimitiveGraph(torch.nn.Module):
    """Differentiable primitive mixtures with frozen KAN node affines."""

    def __init__(
        self,
        model: KAN,
        proposals: dict[tuple[int, int, int], list[EdgeProposal]],
        protected_edges: set[tuple[int, int, int]],
    ) -> None:
        super().__init__()
        self.edges = sorted(proposals)
        self.edge_candidates: list[list[EdgeProposal]] = []
        self.logits = torch.nn.ParameterList()
        self.affines = torch.nn.ParameterList()
        self.initial_affines: list[torch.Tensor] = []
        self.complexities: list[torch.Tensor] = []
        self.zero_indices: list[int | None] = []
        self.layer_edges: list[list[int]] = [[] for _ in model.act_fun]
        self.layer_output_dims = [int(layer.mask.shape[1]) for layer in model.act_fun]
        self.subnode_scale = [value.detach().clone() for value in model.subnode_scale]
        self.subnode_bias = [value.detach().clone() for value in model.subnode_bias]
        self.node_scale = [value.detach().clone() for value in model.node_scale]
        self.node_bias = [value.detach().clone() for value in model.node_bias]

        for edge_index, edge in enumerate(self.edges):
            candidates = [
                proposal
                for proposal in proposals[edge]
                if proposal.name != "0" or edge not in protected_edges
            ]
            if not candidates:
                raise RuntimeError(f"No admissible soft candidates for edge {edge}")
            self.edge_candidates.append(candidates)
            local_r2 = torch.tensor(
                [float(np.clip(item.local_r2, -1.0, 1.0)) for item in candidates],
                dtype=torch.float32,
            )
            best_r2 = torch.max(local_r2)
            initial_logits = torch.clamp((local_r2 - best_r2) / 0.02, min=-6.0)
            initial_logits -= 0.02 * torch.tensor(
                [item.complexity for item in candidates], dtype=torch.float32
            )
            affine = torch.tensor(
                [item.params for item in candidates], dtype=torch.float32
            )
            self.logits.append(torch.nn.Parameter(initial_logits))
            self.affines.append(torch.nn.Parameter(affine.clone()))
            self.initial_affines.append(affine.clone())
            self.complexities.append(
                torch.tensor([item.complexity for item in candidates], dtype=torch.float32)
            )
            self.zero_indices.append(
                next(
                    (index for index, item in enumerate(candidates) if item.name == "0"),
                    None,
                )
            )
            self.layer_edges[edge[0]].append(edge_index)

    def gate_probabilities(self, temperature: float) -> list[torch.Tensor]:
        return [
            torch.softmax(logits / max(float(temperature), 1e-6), dim=0)
            for logits in self.logits
        ]

    def forward(
        self, values: torch.Tensor, temperature: float, hard: bool = False
    ) -> torch.Tensor:
        current = values
        probabilities = self.gate_probabilities(temperature)
        for layer_index, edge_indices in enumerate(self.layer_edges):
            columns: list[list[torch.Tensor]] = [
                [] for _ in range(self.layer_output_dims[layer_index])
            ]
            for edge_index in edge_indices:
                _, input_index, output_index = self.edges[edge_index]
                edge_input = current[:, input_index]
                affine = self.affines[edge_index]
                candidate_values = []
                for candidate_index, proposal in enumerate(
                    self.edge_candidates[edge_index]
                ):
                    if proposal.name == "0":
                        candidate_values.append(torch.zeros_like(edge_input))
                        continue
                    a, b, c, d = affine[candidate_index]
                    transformed = a * edge_input + b
                    candidate_values.append(
                        c * SYMBOLIC_LIB[proposal.name][0](transformed) + d
                    )
                stacked = torch.stack(candidate_values, dim=1)
                if hard:
                    choice = int(torch.argmax(probabilities[edge_index]).detach().cpu())
                    edge_output = stacked[:, choice]
                else:
                    edge_output = torch.sum(
                        stacked * probabilities[edge_index][None, :], dim=1
                    )
                columns[output_index].append(edge_output)
            summed = []
            for parts in columns:
                if parts:
                    summed.append(torch.stack(parts, dim=0).sum(dim=0))
                else:
                    summed.append(torch.zeros(len(current), dtype=current.dtype))
            current = torch.stack(summed, dim=1)
            current = (
                self.subnode_scale[layer_index][None, :] * current
                + self.subnode_bias[layer_index][None, :]
            )
            current = (
                self.node_scale[layer_index][None, :] * current
                + self.node_bias[layer_index][None, :]
            )
        return current.reshape(-1, 1)

    def regularizers(self, temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        complexity_terms = []
        entropy_terms = []
        affine_terms = []
        for index, probabilities in enumerate(self.gate_probabilities(temperature)):
            complexity = self.complexities[index].to(probabilities.device)
            complexity_terms.append(
                torch.sum(probabilities * complexity) / max(float(torch.max(complexity)), 1.0)
            )
            if len(probabilities) > 1:
                entropy_terms.append(
                    -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-8)))
                    / math.log(len(probabilities))
                )
            affine_anchor = self.initial_affines[index].to(self.affines[index].device)
            scale = 1.0 + torch.abs(affine_anchor)
            affine_terms.append(
                torch.mean(((self.affines[index] - affine_anchor) / scale) ** 2)
            )
        zero = self.logits[0].sum() * 0.0
        return (
            torch.stack(complexity_terms).mean() if complexity_terms else zero,
            torch.stack(entropy_terms).mean() if entropy_terms else zero,
            torch.stack(affine_terms).mean() if affine_terms else zero,
        )

    def constrain_affines(self) -> None:
        with torch.no_grad():
            for index, affine in enumerate(self.affines):
                affine[:, 0].clamp_(-10.0, 10.0)
                affine[:, 1].clamp_(-10.0, 10.0)
                affine[:, 2].clamp_(-50.0, 50.0)
                affine[:, 3].clamp_(-50.0, 50.0)
                zero_index = self.zero_indices[index]
                if zero_index is not None:
                    affine[zero_index].zero_()

    def trained_proposals(self) -> dict[tuple[int, int, int], list[EdgeProposal]]:
        result = {}
        for index, edge in enumerate(self.edges):
            params = self.affines[index].detach().cpu().numpy()
            result[edge] = [
                EdgeProposal(
                    proposal.name,
                    tuple(float(value) for value in params[candidate_index]),
                    proposal.local_r2,
                    proposal.complexity,
                )
                for candidate_index, proposal in enumerate(self.edge_candidates[index])
            ]
        return result


def _constraint_components(
    predict,
    main_prediction: torch.Tensor,
    batch: ConstraintBatch,
    descriptor: ConstraintDescriptor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = main_prediction.sum() * 0.0
    monotonic = zero
    if descriptor.monotonic_direction != 0 and len(batch.lower_input):
        lower = predict(batch.lower_input)
        upper = predict(batch.upper_input)
        signed_difference = descriptor.monotonic_direction * (upper - lower)
        monotonic = torch.mean(torch.relu(-signed_difference) ** 2)
    reference = zero
    if descriptor.reference_target_scaled is not None and len(batch.reference_input):
        reference_prediction = predict(batch.reference_input)
        reference = torch.mean(
            (reference_prediction - descriptor.reference_target_scaled) ** 2
        )
    bounds = torch.mean(
        torch.relu(descriptor.lower_bound_scaled - main_prediction) ** 2
        + torch.relu(main_prediction - descriptor.upper_bound_scaled) ** 2
    )
    return monotonic, reference, bounds


def _objective_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reference_prediction: torch.Tensor,
    constraint_components: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    complexity: float,
    args: argparse.Namespace,
) -> float:
    monotonic, reference, bounds = constraint_components
    values = (
        torch.sqrt(torch.mean((prediction - target) ** 2))
        + args.anchor_weight
        * torch.sqrt(torch.mean((prediction - reference_prediction) ** 2))
        + args.constraint_weight * torch.sqrt(monotonic.clamp_min(0.0))
        + args.reference_weight * torch.sqrt(reference.clamp_min(0.0))
        + args.bound_weight * torch.sqrt(bounds.clamp_min(0.0))
    )
    return float(values.detach().cpu()) + args.soft_complexity_weight * complexity


def _soft_graph_consistency_check(
    graph: SoftPrimitiveGraph,
    base_model: KAN,
    values: torch.Tensor,
) -> float:
    hard_model = _clone_numeric_model(base_model)
    trained = graph.trained_proposals()
    for edge_index, edge in enumerate(graph.edges):
        choice = int(torch.argmax(graph.logits[edge_index]).detach().cpu())
        _apply_proposal(hard_model, edge, trained[edge][choice])
    _symbolify_inactive_edges(hard_model)
    _assert_fully_symbolic(hard_model)
    with torch.no_grad():
        graph_prediction = graph(values, temperature=0.1, hard=True)
        model_prediction = _model_output(hard_model, values)
    error = float(torch.max(torch.abs(graph_prediction - model_prediction)).detach().cpu())
    if error > 2e-5:
        raise RuntimeError(f"Soft graph does not match hard KAN semantics: {error:.6g}")
    return error


def _train_global_soft_graph(
    graph: SoftPrimitiveGraph,
    base_model: KAN,
    bundle: ModelBundle,
    train_constraints: ConstraintBatch,
    validation_constraints: ConstraintBatch,
    descriptor: ConstraintDescriptor,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    train_input = bundle.tensors["train"]
    train_target = bundle.targets_scaled["train"]
    validation_input = bundle.tensors["validation"]
    validation_target = bundle.targets_scaled["validation"]
    with torch.no_grad():
        train_reference = _model_output(base_model, train_input).detach()
        validation_reference = _model_output(base_model, validation_input).detach()
    validation_constraints = _limit_constraint_batch(
        validation_constraints, maximum=4096, generator=None
    )
    optimizer = torch.optim.Adam(
        [
            {"params": graph.logits, "lr": args.soft_gate_lr},
            {"params": graph.affines, "lr": args.soft_affine_lr},
        ]
    )
    generator = np.random.default_rng(seed + 7001)

    def validation_score() -> tuple[float, dict[str, float]]:
        with torch.no_grad():
            prediction = graph(
                validation_input,
                temperature=args.soft_temperature_end,
                hard=False,
            )
            components = _constraint_components(
                lambda x: graph(
                    x, temperature=args.soft_temperature_end, hard=False
                ),
                prediction,
                validation_constraints,
                descriptor,
            )
            complexity, entropy, affine_anchor = graph.regularizers(
                args.soft_temperature_end
            )
            score = _objective_score(
                prediction,
                validation_target,
                validation_reference,
                components,
                float(complexity.detach().cpu()),
                args,
            )
            details = {
                "tcad_rmse": float(
                    torch.sqrt(torch.mean((prediction - validation_target) ** 2)).cpu()
                ),
                "teacher_rmse": float(
                    torch.sqrt(torch.mean((prediction - validation_reference) ** 2)).cpu()
                ),
                "monotonic_rmse": float(torch.sqrt(components[0]).cpu()),
                "reference_rmse": float(torch.sqrt(components[1]).cpu()),
                "bound_rmse": float(torch.sqrt(components[2]).cpu()),
                "complexity": float(complexity.cpu()),
                "entropy": float(entropy.cpu()),
                "affine_anchor": float(affine_anchor.cpu()),
            }
        return score, details

    initial_score, initial_details = validation_score()
    best_score = initial_score
    best_details = initial_details
    best_step = 0
    best_state = {
        name: value.detach().clone() for name, value in graph.state_dict().items()
    }
    history = [{"step": 0, "temperature": args.soft_temperature_start, "score": initial_score, **initial_details}]
    stale = 0
    steps_run = 0
    minimum_stop_step = int(math.ceil(0.7 * args.soft_steps))
    for step in range(1, args.soft_steps + 1):
        fraction = (step - 1) / max(args.soft_steps - 1, 1)
        temperature = args.soft_temperature_start * (
            args.soft_temperature_end / args.soft_temperature_start
        ) ** fraction
        size = min(args.soft_batch, len(train_input))
        indices = np.sort(generator.choice(len(train_input), size=size, replace=False))
        index_tensor = torch.from_numpy(indices).long()
        constraint_sample = _limit_constraint_batch(
            train_constraints,
            maximum=max(64, min(args.soft_batch // 2, 2048)),
            generator=generator,
        )
        prediction = graph(train_input[index_tensor], temperature=temperature, hard=False)
        components = _constraint_components(
            lambda x: graph(x, temperature=temperature, hard=False),
            prediction,
            constraint_sample,
            descriptor,
        )
        complexity, entropy, affine_anchor = graph.regularizers(temperature)
        loss = (
            torch.mean((prediction - train_target[index_tensor]) ** 2)
            + args.anchor_weight
            * torch.mean((prediction - train_reference[index_tensor]) ** 2)
            + args.constraint_weight * components[0]
            + args.reference_weight * components[1]
            + args.bound_weight * components[2]
            + args.soft_complexity_weight * complexity
            + args.soft_entropy_weight * entropy
            + args.soft_affine_anchor_weight * affine_anchor
        )
        if not bool(torch.isfinite(loss)):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph.parameters(), max_norm=1.0)
        optimizer.step()
        graph.constrain_affines()
        steps_run = step
        if step % args.soft_every != 0 and step != args.soft_steps:
            continue
        score, details = validation_score()
        history.append(
            {"step": step, "temperature": temperature, "score": score, **details}
        )
        if score + 1e-7 < best_score:
            best_score = score
            best_details = details
            best_step = step
            best_state = {
                name: value.detach().clone()
                for name, value in graph.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if step >= minimum_stop_step and stale >= args.soft_patience:
            break
    graph.load_state_dict(best_state)
    probabilities = graph.gate_probabilities(args.soft_temperature_end)
    max_probabilities = [float(torch.max(value).detach().cpu()) for value in probabilities]
    return {
        "steps_run": steps_run,
        "best_step": best_step,
        "initial_validation_score": initial_score,
        "best_validation_score": best_score,
        "initial_validation": initial_details,
        "best_validation": best_details,
        "mean_max_gate_probability": float(np.mean(max_probabilities)),
        "min_max_gate_probability": float(np.min(max_probabilities)),
        "history": history,
    }


def _hard_model_score(
    model: KAN,
    validation_input: torch.Tensor,
    validation_target: torch.Tensor,
    validation_reference: torch.Tensor,
    constraints: ConstraintBatch,
    descriptor: ConstraintDescriptor,
    complexity: float,
    args: argparse.Namespace,
) -> float:
    with torch.no_grad():
        prediction = _model_output(model, validation_input)
        components = _constraint_components(
            lambda x: _model_output(model, x),
            prediction,
            constraints,
            descriptor,
        )
    if not bool(torch.isfinite(prediction).all()):
        return math.inf
    return _objective_score(
        prediction,
        validation_target,
        validation_reference,
        components,
        complexity,
        args,
    )


def _safeguard_validation_metrics(
    model: KAN,
    base_model: KAN,
    bundle: ModelBundle,
    constraints: ConstraintBatch,
    descriptor: ConstraintDescriptor,
    args: argparse.Namespace,
) -> dict[str, float]:
    validation_input = bundle.tensors["validation"]
    validation_target = bundle.targets_scaled["validation"]
    constraints = _limit_constraint_batch(constraints, maximum=4096, generator=None)
    with torch.no_grad():
        prediction = _model_output(model, validation_input)
        reference_prediction = _model_output(base_model, validation_input)
        components = _constraint_components(
            lambda x: _model_output(model, x),
            prediction,
            constraints,
            descriptor,
        )
        target_rmse = torch.sqrt(torch.mean((prediction - validation_target) ** 2))
        teacher_rmse = torch.sqrt(
            torch.mean((prediction - reference_prediction) ** 2)
        )
        monotonic_rmse = torch.sqrt(components[0].clamp_min(0.0))
        reference_rmse = torch.sqrt(components[1].clamp_min(0.0))
        bound_rmse = torch.sqrt(components[2].clamp_min(0.0))
        score = (
            target_rmse
            + args.constraint_weight * monotonic_rmse
            + args.reference_weight * reference_rmse
            + args.bound_weight * bound_rmse
        )
    return {
        "selection_score": float(score.detach().cpu()),
        "tcad_rmse": float(target_rmse.detach().cpu()),
        "teacher_rmse": float(teacher_rmse.detach().cpu()),
        "monotonic_rmse": float(monotonic_rmse.detach().cpu()),
        "reference_rmse": float(reference_rmse.detach().cpu()),
        "bound_rmse": float(bound_rmse.detach().cpu()),
    }


def _grouped_safeguard_evidence(
    global_model: KAN,
    local_model: KAN,
    bundle: ModelBundle,
    validation_frame: pd.DataFrame,
    specification,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Test whether a validation improvement is stable across response groups."""
    with torch.no_grad():
        global_prediction = (
            _model_output(global_model, bundle.tensors["validation"])
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        local_prediction = (
            _model_output(local_model, bundle.tensors["validation"])
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    target = bundle.targets_scaled["validation"].detach().cpu().numpy().reshape(-1)
    labels = task_config.task_group_labels(validation_frame, specification).to_numpy()
    if len(labels) != len(target):
        raise RuntimeError("Validation labels and predictions have different lengths")

    group_deltas = []
    for label in pd.unique(labels):
        mask = labels == label
        global_mse = float(np.mean((global_prediction[mask] - target[mask]) ** 2))
        local_mse = float(np.mean((local_prediction[mask] - target[mask]) ** 2))
        group_deltas.append(global_mse - local_mse)
    deltas = np.asarray(group_deltas, dtype=np.float64)
    if not len(deltas) or not np.all(np.isfinite(deltas)):
        raise RuntimeError("Nonfinite or empty grouped safeguard evidence")

    rng = np.random.default_rng(7919 + 104729 * int(seed))
    indices = rng.integers(
        0,
        len(deltas),
        size=(args.safeguard_bootstrap_samples, len(deltas)),
    )
    bootstrap_means = deltas[indices].mean(axis=1)
    upper = float(np.quantile(bootstrap_means, args.safeguard_confidence))
    win_fraction = float(np.mean(deltas < 0.0))
    stability_pass = bool(
        upper < 0.0
        and win_fraction >= args.safeguard_min_group_win_fraction
    )
    return {
        "n_groups": int(len(deltas)),
        "mean_group_mse_delta_global_minus_local": float(np.mean(deltas)),
        "median_group_mse_delta_global_minus_local": float(np.median(deltas)),
        "group_win_fraction": win_fraction,
        "bootstrap_samples": int(args.safeguard_bootstrap_samples),
        "bootstrap_confidence": float(args.safeguard_confidence),
        "bootstrap_mean_delta_upper": upper,
        "minimum_group_win_fraction": float(
            args.safeguard_min_group_win_fraction
        ),
        "stability_pass": stability_pass,
    }


def _harden_global_soft_graph(
    graph: SoftPrimitiveGraph,
    base_model: KAN,
    bundle: ModelBundle,
    validation_constraints: ConstraintBatch,
    descriptor: ConstraintDescriptor,
    args: argparse.Namespace,
) -> tuple[KAN, list[dict], dict]:
    model = _clone_numeric_model(base_model)
    trained = graph.trained_proposals()
    probabilities = graph.gate_probabilities(args.soft_temperature_end)
    choices = {}
    confidence = {}
    for edge_index, edge in enumerate(graph.edges):
        choice = int(torch.argmax(probabilities[edge_index]).detach().cpu())
        choices[edge] = choice
        confidence[edge] = float(torch.max(probabilities[edge_index]).detach().cpu())
        _apply_proposal(model, edge, trained[edge][choice])
    _symbolify_inactive_edges(model)
    _assert_fully_symbolic(model)
    validation_constraints = _limit_constraint_batch(
        validation_constraints, maximum=4096, generator=None
    )
    validation_input = bundle.tensors["validation"]
    validation_target = bundle.targets_scaled["validation"]
    with torch.no_grad():
        validation_reference = _model_output(base_model, validation_input).detach()

    def total_complexity() -> float:
        if not choices:
            return 0.0
        return float(
            np.mean(
                [trained[edge][choice].complexity for edge, choice in choices.items()]
            )
            / max(max(item.complexity for values in trained.values() for item in values), 1.0)
        )

    current_score = _hard_model_score(
        model,
        validation_input,
        validation_target,
        validation_reference,
        validation_constraints,
        descriptor,
        total_complexity(),
        args,
    )
    initial_score = current_score
    trajectory = []
    for sweep in range(1, args.hardening_sweeps + 1):
        changes = 0
        for edge in sorted(graph.edges, key=lambda item: (confidence[item], item)):
            edge_index = graph.edges.index(edge)
            probability = probabilities[edge_index].detach().cpu().numpy()
            ranked = list(np.argsort(-probability)[: args.hardening_topk])
            current_choice = choices[edge]
            if current_choice not in ranked:
                ranked.append(current_choice)
            local_choice = int(
                np.argmax([proposal.local_r2 for proposal in trained[edge]])
            )
            if local_choice not in ranked:
                ranked.append(local_choice)
            best_choice = current_choice
            best_score = current_score
            for candidate_choice in ranked:
                candidate_choice = int(candidate_choice)
                choices[edge] = candidate_choice
                _apply_proposal(model, edge, trained[edge][candidate_choice])
                score = _hard_model_score(
                    model,
                    validation_input,
                    validation_target,
                    validation_reference,
                    validation_constraints,
                    descriptor,
                    total_complexity(),
                    args,
                )
                if score + args.hardening_tolerance < best_score:
                    best_choice = candidate_choice
                    best_score = score
            accepted = best_choice != current_choice
            choices[edge] = best_choice
            _apply_proposal(model, edge, trained[edge][best_choice])
            if accepted:
                changes += 1
                current_score = best_score
            trajectory.append(
                {
                    "sweep": sweep,
                    "edge": list(edge),
                    "old_primitive": trained[edge][current_choice].name,
                    "primitive": trained[edge][best_choice].name,
                    "accepted": accepted,
                    "validation_score": current_score,
                    "gate_probability": float(probability[best_choice]),
                    "local_r2": trained[edge][best_choice].local_r2,
                    "complexity": trained[edge][best_choice].complexity,
                }
            )
        if changes == 0:
            break
    _assert_fully_symbolic(model)
    return model, trajectory, {
        "initial_validation_score": initial_score,
        "final_validation_score": current_score,
        "accepted_changes": int(sum(item["accepted"] for item in trajectory)),
        "sweeps_run": int(max((item["sweep"] for item in trajectory), default=0)),
    }


def _symbolify_inactive_edges(model: KAN) -> None:
    for layer_index, layer in enumerate(model.act_fun):
        for input_index in range(layer.mask.shape[0]):
            for output_index in range(layer.mask.shape[1]):
                numeric = float(layer.mask[input_index, output_index].detach().cpu())
                symbolic = float(
                    model.symbolic_fun[layer_index].mask[output_index, input_index].detach().cpu()
                )
                if numeric == 0.0 and symbolic == 0.0:
                    model.fix_symbolic(
                        layer_index,
                        input_index,
                        output_index,
                        "0",
                        fit_params_bool=False,
                        verbose=False,
                        log_history=False,
                    )
                    model.symbolic_fun[layer_index].affine.data[output_index, input_index].zero_()


def _assert_fully_symbolic(model: KAN) -> None:
    numeric = sum(float(torch.sum(torch.abs(layer.mask)).detach().cpu()) for layer in model.act_fun)
    expected = sum(layer.mask.numel() for layer in model.act_fun)
    symbolic = sum(
        int(torch.sum(layer.mask > 0.0).detach().cpu()) for layer in model.symbolic_fun
    )
    if numeric != 0.0 or symbolic != expected:
        raise RuntimeError(
            f"Graph is not fully symbolic: numeric mask sum={numeric}, symbolic={symbolic}/{expected}"
        )


def _refit_symbolic_affines(
    model: KAN,
    active_edges: list[tuple[int, int, int]],
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    validation_input: torch.Tensor,
    validation_target: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    if args.refit_steps == 0:
        return {
            "steps_run": 0,
            "best_step": 0,
            "initial_validation_scaled_rmse": _scaled_rmse(
                model, validation_input, validation_target
            ),
            "best_validation_scaled_rmse": _scaled_rmse(
                model, validation_input, validation_target
            ),
        }
    parameters = [layer.affine for layer in model.symbolic_fun]
    masks = [torch.zeros_like(parameter) for parameter in parameters]
    for layer_index, input_index, output_index in active_edges:
        masks[layer_index][output_index, input_index, :] = 1.0
    optimizer = torch.optim.Adam(parameters, lr=args.refit_lr)
    generator = np.random.default_rng(seed)
    initial = _scaled_rmse(model, validation_input, validation_target)
    best_rmse = initial
    best_step = 0
    best_state = [parameter.detach().clone() for parameter in parameters]
    stale = 0
    steps_run = 0
    for step in range(1, args.refit_steps + 1):
        size = min(args.refit_batch, len(train_input))
        indices = generator.choice(len(train_input), size=size, replace=False)
        index_tensor = torch.from_numpy(indices).long()
        prediction = _model_output(model, train_input[index_tensor])
        loss = torch.mean((prediction - train_target[index_tensor]) ** 2)
        if not bool(torch.isfinite(loss)):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter, mask in zip(parameters, masks):
            if parameter.grad is not None:
                parameter.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            for parameter, mask in zip(parameters, masks):
                parameter[..., 0].clamp_(-10.0, 10.0)
                parameter[..., 1].clamp_(-10.0, 10.0)
                parameter[..., 2].clamp_(-50.0, 50.0)
                parameter[..., 3].clamp_(-50.0, 50.0)
                parameter.mul_(mask)
        steps_run = step
        if step % args.refit_every != 0 and step != args.refit_steps:
            continue
        validation_rmse = _scaled_rmse(model, validation_input, validation_target)
        if validation_rmse + 1e-7 < best_rmse:
            best_rmse = validation_rmse
            best_step = step
            best_state = [parameter.detach().clone() for parameter in parameters]
            stale = 0
        else:
            stale += 1
        if stale >= args.refit_patience:
            break
    with torch.no_grad():
        for parameter, state in zip(parameters, best_state):
            parameter.copy_(state)
    return {
        "steps_run": steps_run,
        "best_step": best_step,
        "initial_validation_scaled_rmse": initial,
        "best_validation_scaled_rmse": best_rmse,
    }


def _refit_symbolic_affines_unified(
    model: KAN,
    active_edges: list[tuple[int, int, int]],
    base_model: KAN,
    bundle: ModelBundle,
    train_constraints: ConstraintBatch,
    validation_constraints: ConstraintBatch,
    descriptor: ConstraintDescriptor,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    train_input = bundle.tensors["train"]
    train_target = bundle.targets_scaled["train"]
    validation_input = bundle.tensors["validation"]
    validation_target = bundle.targets_scaled["validation"]
    with torch.no_grad():
        train_reference = _model_output(base_model, train_input).detach()
        validation_reference = _model_output(base_model, validation_input).detach()
    validation_constraints = _limit_constraint_batch(
        validation_constraints, maximum=4096, generator=None
    )
    parameters = [layer.affine for layer in model.symbolic_fun]
    masks = [torch.zeros_like(parameter) for parameter in parameters]
    for layer_index, input_index, output_index in active_edges:
        masks[layer_index][output_index, input_index, :] = 1.0
    affine_anchors = [parameter.detach().clone() for parameter in parameters]

    def validation_metrics() -> tuple[float, float]:
        score = _hard_model_score(
            model,
            validation_input,
            validation_target,
            validation_reference,
            validation_constraints,
            descriptor,
            complexity=0.0,
            args=args,
        )
        return score, _scaled_rmse(model, validation_input, validation_target)

    initial_score, initial_rmse = validation_metrics()
    if args.refit_steps == 0:
        return {
            "steps_run": 0,
            "best_step": 0,
            "initial_validation_scaled_rmse": initial_rmse,
            "best_validation_scaled_rmse": initial_rmse,
            "initial_validation_objective": initial_score,
            "best_validation_objective": initial_score,
            "objective": "tcad+teacher+inferred_constraints",
        }

    optimizer = torch.optim.Adam(parameters, lr=args.refit_lr)
    generator = np.random.default_rng(seed + 9001)
    best_score = initial_score
    best_rmse = initial_rmse
    best_step = 0
    best_state = [parameter.detach().clone() for parameter in parameters]
    stale = 0
    steps_run = 0
    for step in range(1, args.refit_steps + 1):
        size = min(args.refit_batch, len(train_input))
        indices = np.sort(generator.choice(len(train_input), size=size, replace=False))
        index_tensor = torch.from_numpy(indices).long()
        prediction = _model_output(model, train_input[index_tensor])
        constraint_sample = _limit_constraint_batch(
            train_constraints,
            maximum=max(64, min(args.refit_batch // 2, 2048)),
            generator=generator,
        )
        components = _constraint_components(
            lambda x: _model_output(model, x),
            prediction,
            constraint_sample,
            descriptor,
        )
        affine_anchor = torch.stack(
            [
                torch.mean(
                    (
                        (parameter - anchor)
                        / (1.0 + torch.abs(anchor))
                    )
                    ** 2
                )
                for parameter, anchor in zip(parameters, affine_anchors)
            ]
        ).mean()
        loss = (
            torch.mean((prediction - train_target[index_tensor]) ** 2)
            + args.anchor_weight
            * torch.mean((prediction - train_reference[index_tensor]) ** 2)
            + args.constraint_weight * components[0]
            + args.reference_weight * components[1]
            + args.bound_weight * components[2]
            + args.soft_affine_anchor_weight * affine_anchor
        )
        if not bool(torch.isfinite(loss)):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter, mask in zip(parameters, masks):
            if parameter.grad is not None:
                parameter.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            for parameter, mask in zip(parameters, masks):
                parameter[..., 0].clamp_(-10.0, 10.0)
                parameter[..., 1].clamp_(-10.0, 10.0)
                parameter[..., 2].clamp_(-50.0, 50.0)
                parameter[..., 3].clamp_(-50.0, 50.0)
                parameter.mul_(mask)
        steps_run = step
        if step % args.refit_every != 0 and step != args.refit_steps:
            continue
        validation_score, validation_rmse = validation_metrics()
        if validation_score + 1e-7 < best_score:
            best_score = validation_score
            best_rmse = validation_rmse
            best_step = step
            best_state = [parameter.detach().clone() for parameter in parameters]
            stale = 0
        else:
            stale += 1
        if stale >= args.refit_patience:
            break
    with torch.no_grad():
        for parameter, state in zip(parameters, best_state):
            parameter.copy_(state)
    return {
        "steps_run": steps_run,
        "best_step": best_step,
        "initial_validation_scaled_rmse": initial_rmse,
        "best_validation_scaled_rmse": best_rmse,
        "initial_validation_objective": initial_score,
        "best_validation_objective": best_score,
        "objective": "tcad+teacher+inferred_constraints",
    }


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    residual = prediction - actual
    return {
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(r2_score(actual, prediction)),
        "p95_abs_error": float(np.quantile(np.abs(residual), 0.95)),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def _dense_raw(frame: pd.DataFrame, inputs: tuple[str, ...], points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = frame[list(inputs)].min(axis=0).to_numpy(dtype=np.float64)
    upper = frame[list(inputs)].max(axis=0).to_numpy(dtype=np.float64)
    unit = (rng.permutation(points)[:, None] + rng.random((points, len(inputs)))) / points
    for column in range(len(inputs)):
        rng.shuffle(unit[:, column])
    return lower + unit * (upper - lower)


def _make_formula(
    model: KAN,
    bundle: ModelBundle,
    inputs: tuple[str, ...],
) -> tuple[sympy.Expr, list[sympy.Symbol]]:
    variables = [sympy.Symbol(f"x{index}") for index in range(len(inputs))]
    formulas, _ = model.symbolic_formula(
        var=variables,
        normalizer=[bundle.x_scaler.mean_, bundle.x_scaler.scale_],
        output_normalizer=([bundle.y_scaler.mean_[0]], [bundle.y_scaler.scale_[0]]),
        simplify=False,
    )
    expression = sympy.Float(bundle.output_scale) * formulas[0]
    return expression, variables


def _formula_payload(
    expression: sympy.Expr,
    variables: list[sympy.Symbol],
    inputs: tuple[str, ...],
    task: str,
    seed: int,
    source: str,
    variant: str,
    library: list[str],
    trajectory: list[dict],
    refit: dict,
) -> dict:
    free_symbols = {str(symbol) for symbol in expression.free_symbols}
    used_inputs = [
        input_name
        for input_name, variable in zip(inputs, variables)
        if str(variable) in free_symbols
    ]
    return {
        "schema": "unified_output_aware_symbolic_v1",
        "task": task,
        "seed": seed,
        "source": source,
        "variant": variant,
        "input_order": list(inputs),
        "symbol_order": [str(variable) for variable in variables],
        "primitive_library": list(library),
        "formula": str(expression),
        "count_ops": int(sympy.count_ops(expression, visual=False)),
        "formula_characters": len(str(expression)),
        "used_inputs": used_inputs,
        "uses_all_inputs": len(used_inputs) == len(inputs),
        "trajectory": trajectory,
        "refit": refit,
    }


def _replay_formula(payload_path: Path, raw: np.ndarray) -> np.ndarray:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    variables = [sympy.Symbol(name) for name in payload["symbol_order"]]
    expression = sympy.sympify(payload["formula"])
    evaluator = sympy.lambdify(variables, expression, modules="numpy", cse=True)
    with np.errstate(all="ignore"):
        values = evaluator(*[raw[:, index] for index in range(raw.shape[1])])
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(len(raw), float(values), dtype=np.float64)
    return values.reshape(-1)


def _shape_diagnostics(
    task: str,
    test: pd.DataFrame,
    prediction: np.ndarray,
    specification,
) -> dict:
    diagnostics = {
        "monotonic_pairs": 0,
        "monotonic_violation_fraction": np.nan,
        "positive_fraction": float(np.mean(prediction > 0.0)),
        "negative_fraction": float(np.mean(prediction < 0.0)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }
    if task == "Capacitance":
        return diagnostics
    labels = task_config.task_group_labels(test, specification).to_numpy()
    axis = test[specification.axis_col].to_numpy(dtype=np.float64)
    violations = 0
    pairs = 0
    tolerance = max(1e-8, 1e-6 * float(np.ptp(prediction)))
    for label in pd.unique(labels):
        indices = np.flatnonzero(labels == label)
        order = indices[np.argsort(axis[indices])]
        differences = np.diff(prediction[order])
        violations += int(np.sum(differences > tolerance))
        pairs += len(differences)
    diagnostics["monotonic_pairs"] = pairs
    diagnostics["monotonic_violation_fraction"] = (
        float(violations / pairs) if pairs else np.nan
    )
    return diagnostics


def _write_report(output: Path, rows: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Unified output-aware progressive symbolification",
        "",
        "All rows use the same primitive library, candidate fitter, pruning thresholds, "
        "selection objective, symbolic-affine refit, and replay checks. The registered "
        f"`{args.selection_split}` groups select replacement trajectories; validation groups "
        "stop the final refit. Test groups are evaluated only after each expression is frozen.",
        "",
        f"Common library: `{', '.join(args.library)}`.",
        "",
        "| Task | Seed | Source | Policy | Active edges | Inputs used | Ops | Teacher RMSE | Formula RMSE | Formula R2 | Formula-network RMSE | Replay max | Dense finite |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows.sort_values(["task", "seed", "source", "variant"]).to_dict("records"):
        lines.append(
            f"| {row['task']} | {row['seed']} | {row['source']} | {row['variant']} | "
            f"{row['active_edges']} | {row['used_input_count']}/{row['input_count']} | "
            f"{row['formula_ops']} | {row['teacher_to_tcad_rmse']:.6g} | "
            f"{row['formula_to_tcad_rmse']:.6g} | {row['formula_to_tcad_r2']:.6g} | "
            f"{row['formula_to_network_rmse']:.6g} | {row['replay_max_abs_error']:.3g} | "
            f"{row['dense_finite']}/{row['dense_points']} |"
        )
    policies = sorted(set(rows["variant"]) - {"local_edgewise"})
    if "local_edgewise" in set(rows["variant"]) and policies:
        pivot = rows.pivot_table(
            index=["task", "seed", "source"],
            columns="variant",
            values="formula_to_tcad_rmse",
            aggfunc="first",
        )
        lines.extend(
            [
                "",
                "## Matched policy comparison",
                "",
            ]
        )
        for policy in policies:
            matched = pivot[["local_edgewise", policy]].dropna().copy()
            matched["relative_change"] = (
                matched[policy] / matched["local_edgewise"] - 1.0
            )
            wins = int(np.sum(matched["relative_change"] < 0.0))
            lines.extend(
                [
                    f"`{policy}` improves {wins}/{len(matched)} matched task-seed-source cases.",
                    "",
                    "| Task | Seed | Source | Policy | Local RMSE | Policy RMSE | Relative change |",
                    "| --- | ---: | --- | --- | ---: | ---: | ---: |",
                ]
            )
            for index, row in matched.reset_index().sort_values(
                ["task", "seed", "source"]
            ).iterrows():
                del index
                lines.append(
                    f"| {row['task']} | {int(row['seed'])} | {row['source']} | {policy} | "
                    f"{row['local_edgewise']:.6g} | {row[policy]:.6g} | "
                    f"{100.0 * row['relative_change']:.2f}% |"
                )
            lines.append("")
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    paper_hashes = {str(path): _sha256(path) for path in PAPER_FILES}
    args.output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    manifest_frames: list[pd.DataFrame] = []
    started = time.perf_counter()

    for seed in args.seeds:
        for task in args.tasks:
            (
                task_args,
                full_frame,
                inputs,
                specification,
                baseline_spec,
                partitions,
            ) = _load_task(args, task, seed)
            manifest_frames.append(
                shared_audit._split_manifest(task, seed, specification, partitions)
            )
            targets_physical = _physical_targets(partitions, baseline_spec)
            source_bundles = []
            if "bkan" in args.sources:
                source_bundles.append(
                    _load_bkan_bundle(
                        args,
                        task,
                        seed,
                        task_args,
                        inputs,
                        specification,
                        partitions,
                        targets_physical,
                    )
                )
            if "dkan" in args.sources:
                source_bundles.append(
                    _fit_dkan_bundle(
                        args,
                        task,
                        seed,
                        inputs,
                        partitions,
                        baseline_spec,
                        targets_physical,
                    )
                )

            test_raw = partitions[3][list(inputs)].to_numpy(dtype=np.float64)
            dense_raw = _dense_raw(full_frame, inputs, args.dense_points, seed + 1009)
            for bundle in source_bundles:
                base_model = _prepare_numeric_graph(bundle, args)
                active_edges = _active_edges(base_model)
                protected_edges = _protected_input_paths(base_model)
                selection_input, selection_target = _selection_tensors(
                    bundle, args.selection_split
                )
                proposal_input = _proposal_sample(
                    bundle.tensors["train"], args.proposal_points, seed + 17
                )
                proposal_started = time.perf_counter()
                proposals = _fit_proposals(base_model, proposal_input, list(args.library), args)
                proposal_seconds = time.perf_counter() - proposal_started
                pruned_prediction = _predict_physical(bundle, base_model, "test")
                proposal_records = {
                    "/".join(str(value) for value in edge): [proposal.__dict__ for proposal in values]
                    for edge, values in proposals.items()
                }
                proposal_path = (
                    args.output
                    / "proposals"
                    / f"seed_{seed}"
                    / task
                    / f"{bundle.source}.json"
                )
                proposal_path.parent.mkdir(parents=True, exist_ok=True)
                proposal_path.write_text(
                    json.dumps(proposal_records, indent=2, allow_nan=False), encoding="utf-8"
                )

                for variant in args.variants:
                    symbolic_model = _clone_numeric_model(base_model)
                    extraction_started = time.perf_counter()
                    soft_training = None
                    hardening = None
                    safeguard = None
                    constraint_descriptor = None
                    soft_graph_consistency = None
                    train_constraints = None
                    validation_constraints = None
                    if variant == "local_edgewise":
                        trajectory = _select_local(
                            symbolic_model, proposals, protected_edges, args
                        )
                    elif variant in (
                        "global_soft_hardened",
                        "global_soft_safeguarded",
                    ):
                        constraint_descriptor = _infer_constraint_descriptor(
                            partitions[0],
                            bundle.tensors["train"],
                            bundle.targets_scaled["train"],
                            inputs,
                            specification,
                            args,
                        )
                        train_constraints = _constraint_batch(
                            partitions[0],
                            bundle.tensors["train"],
                            inputs,
                            specification,
                            constraint_descriptor,
                        )
                        validation_constraints = _constraint_batch(
                            partitions[1],
                            bundle.tensors["validation"],
                            inputs,
                            specification,
                            constraint_descriptor,
                        )
                        soft_graph = SoftPrimitiveGraph(
                            base_model, proposals, protected_edges
                        )
                        consistency_points = min(256, len(bundle.tensors["validation"]))
                        soft_graph_consistency = _soft_graph_consistency_check(
                            soft_graph,
                            base_model,
                            bundle.tensors["validation"][:consistency_points],
                        )
                        soft_training = _train_global_soft_graph(
                            soft_graph,
                            base_model,
                            bundle,
                            train_constraints,
                            validation_constraints,
                            constraint_descriptor,
                            seed,
                            args,
                        )
                        symbolic_model, trajectory, hardening = _harden_global_soft_graph(
                            soft_graph,
                            base_model,
                            bundle,
                            validation_constraints,
                            constraint_descriptor,
                            args,
                        )
                    else:
                        reference_prediction = None
                        if variant in (
                            "output_aware_anchored",
                            "output_aware_screened",
                        ):
                            with torch.no_grad():
                                reference_prediction = _model_output(
                                    base_model, selection_input
                                ).detach()
                        trajectory = _select_output_aware(
                            symbolic_model,
                            proposals,
                            selection_input,
                            selection_target,
                            protected_edges,
                            args,
                            reference_prediction=reference_prediction,
                            local_r2_gap=(
                                args.local_r2_gap
                                if variant == "output_aware_screened"
                                else None
                            ),
                        )
                    _symbolify_inactive_edges(symbolic_model)
                    _assert_fully_symbolic(symbolic_model)
                    if variant in (
                        "global_soft_hardened",
                        "global_soft_safeguarded",
                    ):
                        refit = _refit_symbolic_affines_unified(
                            symbolic_model,
                            active_edges,
                            base_model,
                            bundle,
                            train_constraints,
                            validation_constraints,
                            constraint_descriptor,
                            seed,
                            args,
                        )
                    else:
                        refit = _refit_symbolic_affines(
                            symbolic_model,
                            active_edges,
                            bundle.tensors["train"],
                            bundle.targets_scaled["train"],
                            bundle.tensors["validation"],
                            bundle.targets_scaled["validation"],
                            seed,
                            args,
                        )
                    _assert_fully_symbolic(symbolic_model)
                    if variant == "global_soft_safeguarded":
                        global_validation = _safeguard_validation_metrics(
                            symbolic_model,
                            base_model,
                            bundle,
                            validation_constraints,
                            constraint_descriptor,
                            args,
                        )
                        fallback_model = _clone_numeric_model(base_model)
                        fallback_trajectory = _select_local(
                            fallback_model, proposals, protected_edges, args
                        )
                        _symbolify_inactive_edges(fallback_model)
                        _assert_fully_symbolic(fallback_model)
                        fallback_refit = _refit_symbolic_affines(
                            fallback_model,
                            active_edges,
                            bundle.tensors["train"],
                            bundle.targets_scaled["train"],
                            bundle.tensors["validation"],
                            bundle.targets_scaled["validation"],
                            seed,
                            args,
                        )
                        fallback_validation = _safeguard_validation_metrics(
                            fallback_model,
                            base_model,
                            bundle,
                            validation_constraints,
                            constraint_descriptor,
                            args,
                        )
                        grouped_evidence = _grouped_safeguard_evidence(
                            symbolic_model,
                            fallback_model,
                            bundle,
                            partitions[1],
                            specification,
                            seed,
                            args,
                        )
                        local_score = fallback_validation["selection_score"]
                        global_score = global_validation["selection_score"]
                        relative_improvement = (
                            (local_score - global_score) / max(abs(local_score), 1e-12)
                        )
                        score_pass = bool(
                            global_score + args.hardening_tolerance < local_score
                        )
                        effect_size_pass = bool(
                            relative_improvement
                            >= args.safeguard_min_relative_improvement
                        )
                        selected_solution = "global_soft"
                        if not (
                            score_pass
                            and effect_size_pass
                            and grouped_evidence["stability_pass"]
                        ):
                            selected_solution = "local_fallback"
                            symbolic_model = fallback_model
                            trajectory = fallback_trajectory
                            refit = fallback_refit
                        safeguard = {
                            "selected_solution": selected_solution,
                            "global_soft": global_validation,
                            "local_fallback": fallback_validation,
                            "relative_validation_improvement": float(
                                relative_improvement
                            ),
                            "minimum_relative_improvement": float(
                                args.safeguard_min_relative_improvement
                            ),
                            "score_pass": score_pass,
                            "effect_size_pass": effect_size_pass,
                            "grouped_evidence": grouped_evidence,
                            "selection_data": "validation_groups_only",
                            "test_accessed_during_selection": False,
                        }
                        _assert_fully_symbolic(symbolic_model)
                    extraction_seconds = time.perf_counter() - extraction_started
                    formula_prediction = _predict_physical(bundle, symbolic_model, "test")
                    if not np.all(np.isfinite(formula_prediction)):
                        raise RuntimeError(f"Nonfinite symbolic model output: {task}/{bundle.source}/{variant}")

                    expression, variables = _make_formula(symbolic_model, bundle, inputs)
                    payload = _formula_payload(
                        expression,
                        variables,
                        inputs,
                        task,
                        seed,
                        bundle.source,
                        variant,
                        list(args.library),
                        trajectory,
                        refit,
                    )
                    payload["selection_split"] = args.selection_split
                    if variant in (
                        "global_soft_hardened",
                        "global_soft_safeguarded",
                    ):
                        payload["schema"] = "unified_global_soft_gate_symbolic_v2"
                        payload["soft_training"] = soft_training
                        payload["hardening"] = hardening
                        payload["soft_graph_consistency_max_abs_error"] = (
                            soft_graph_consistency
                        )
                        payload["constraints"] = {
                            "axis_index": constraint_descriptor.axis_index,
                            "monotonic_direction": constraint_descriptor.monotonic_direction,
                            "monotonic_confidence": constraint_descriptor.monotonic_confidence,
                            "reference_side": constraint_descriptor.reference_side,
                            "reference_target_scaled": constraint_descriptor.reference_target_scaled,
                            "reference_std_scaled": constraint_descriptor.reference_std_scaled,
                            "lower_bound_scaled": constraint_descriptor.lower_bound_scaled,
                            "upper_bound_scaled": constraint_descriptor.upper_bound_scaled,
                        }
                        if safeguard is not None:
                            payload["safeguard"] = safeguard
                    formula_path = (
                        args.output
                        / "formulas"
                        / f"seed_{seed}"
                        / task
                        / f"{bundle.source}_{variant}.json"
                    )
                    formula_path.parent.mkdir(parents=True, exist_ok=True)
                    formula_path.write_text(
                        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
                    )
                    replay_prediction = _replay_formula(formula_path, test_raw)
                    dense_prediction = _replay_formula(formula_path, dense_raw)
                    if replay_prediction.shape != formula_prediction.shape:
                        raise RuntimeError("Independent formula replay shape mismatch")
                    teacher_metrics = _metrics(
                        targets_physical["test"], bundle.teacher_predictions["test"]
                    )
                    pruned_metrics = _metrics(targets_physical["test"], pruned_prediction)
                    formula_metrics = _metrics(targets_physical["test"], replay_prediction)
                    network_metrics = _metrics(bundle.teacher_predictions["test"], replay_prediction)
                    replay_error = np.abs(replay_prediction - formula_prediction)
                    shape = _shape_diagnostics(
                        task, partitions[3], replay_prediction, specification
                    )
                    row = {
                        "task": task,
                        "seed": seed,
                        "source": bundle.source,
                        "variant": variant,
                        "common_library": ",".join(args.library),
                        "active_edges": len(active_edges),
                        "protected_edges": len(protected_edges),
                        "formula_ops": payload["count_ops"],
                        "formula_characters": payload["formula_characters"],
                        "used_inputs": ",".join(payload["used_inputs"]),
                        "used_input_count": len(payload["used_inputs"]),
                        "input_count": len(inputs),
                        "uses_all_inputs": payload["uses_all_inputs"],
                        "teacher_to_tcad_rmse": teacher_metrics["rmse"],
                        "teacher_to_tcad_r2": teacher_metrics["r2"],
                        "pruned_teacher_to_tcad_rmse": pruned_metrics["rmse"],
                        "formula_to_tcad_rmse": formula_metrics["rmse"],
                        "formula_to_tcad_mae": formula_metrics["mae"],
                        "formula_to_tcad_r2": formula_metrics["r2"],
                        "formula_to_tcad_p95_abs_error": formula_metrics["p95_abs_error"],
                        "formula_to_tcad_max_abs_error": formula_metrics["max_abs_error"],
                        "formula_to_network_rmse": network_metrics["rmse"],
                        "error_inflation_ratio": formula_metrics["rmse"]
                        / max(teacher_metrics["rmse"], np.finfo(float).tiny),
                        "replay_max_abs_error": float(np.max(replay_error)),
                        "replay_rmse": float(np.sqrt(np.mean(replay_error * replay_error))),
                        "test_finite": int(np.sum(np.isfinite(replay_prediction))),
                        "test_points": len(replay_prediction),
                        "dense_finite": int(np.sum(np.isfinite(dense_prediction))),
                        "dense_points": len(dense_prediction),
                        "proposal_time_s": proposal_seconds,
                        "extraction_refit_time_s": extraction_seconds,
                        "formula_path": str(formula_path.relative_to(ROOT)),
                        "proposal_path": str(proposal_path.relative_to(ROOT)),
                        "soft_best_step": (
                            soft_training["best_step"] if soft_training is not None else np.nan
                        ),
                        "soft_best_validation_score": (
                            soft_training["best_validation_score"]
                            if soft_training is not None
                            else np.nan
                        ),
                        "soft_mean_max_gate_probability": (
                            soft_training["mean_max_gate_probability"]
                            if soft_training is not None
                            else np.nan
                        ),
                        "hardening_accepted_changes": (
                            hardening["accepted_changes"] if hardening is not None else np.nan
                        ),
                        "hardening_validation_score_change": (
                            hardening["final_validation_score"]
                            - hardening["initial_validation_score"]
                            if hardening is not None
                            else np.nan
                        ),
                        "safeguard_selected_solution": (
                            safeguard["selected_solution"]
                            if safeguard is not None
                            else ""
                        ),
                        "safeguard_relative_validation_improvement": (
                            safeguard["relative_validation_improvement"]
                            if safeguard is not None
                            else np.nan
                        ),
                        "safeguard_effect_size_pass": (
                            safeguard["effect_size_pass"]
                            if safeguard is not None
                            else ""
                        ),
                        "safeguard_group_win_fraction": (
                            safeguard["grouped_evidence"]["group_win_fraction"]
                            if safeguard is not None
                            else np.nan
                        ),
                        "safeguard_bootstrap_upper": (
                            safeguard["grouped_evidence"][
                                "bootstrap_mean_delta_upper"
                            ]
                            if safeguard is not None
                            else np.nan
                        ),
                        "safeguard_stability_pass": (
                            safeguard["grouped_evidence"]["stability_pass"]
                            if safeguard is not None
                            else ""
                        ),
                        "inferred_monotonic_direction": (
                            constraint_descriptor.monotonic_direction
                            if constraint_descriptor is not None
                            else np.nan
                        ),
                        "inferred_monotonic_confidence": (
                            constraint_descriptor.monotonic_confidence
                            if constraint_descriptor is not None
                            else np.nan
                        ),
                        "inferred_reference_side": (
                            constraint_descriptor.reference_side
                            if constraint_descriptor is not None
                            else ""
                        ),
                        **shape,
                        **{f"source_{key}": value for key, value in bundle.metadata.items()},
                    }
                    metric_rows.append(row)
                    print(
                        f"[{task} seed={seed} {bundle.source} {variant}] "
                        f"teacher={teacher_metrics['rmse']:.6g}, formula={formula_metrics['rmse']:.6g}, "
                        f"ops={payload['count_ops']}, replay={row['replay_max_abs_error']:.3g}"
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    pd.concat(manifest_frames, ignore_index=True).to_csv(
        args.output / "split_manifest.csv", index=False
    )
    _write_report(args.output, metrics, args)
    after_hashes = {str(path): _sha256(path) for path in PAPER_FILES}
    if after_hashes != paper_hashes:
        raise RuntimeError("A manuscript file changed during the development experiment")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "sympy": sympy.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "paper_sha256_before_after": {
            path: {"before": paper_hashes[path], "after": after_hashes[path]}
            for path in paper_hashes
        },
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
