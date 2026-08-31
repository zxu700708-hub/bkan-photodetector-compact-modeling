"""Shared helpers for the device-modeling workflow."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for repeatable local experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def without_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a DataFrame without pandas attrs metadata."""

    clean = frame.copy()
    clean.attrs.clear()
    return clean


def concat_without_attrs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames after dropping attrs to avoid pandas warnings."""

    return pd.concat([without_attrs(frame) for frame in frames], ignore_index=True)


def safe_scale(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Replace invalid or near-zero scaler values with 1.0."""

    scale = np.asarray(values, dtype=np.float64).copy()
    scale[~np.isfinite(scale)] = 1.0
    scale[np.abs(scale) < eps] = 1.0
    return scale
