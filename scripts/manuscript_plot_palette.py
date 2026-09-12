"""Shared color system for the manuscript's quantitative figures.

The colors are matched to the dominant blue/green/orange/magenta accents in
Fig. 1.  Keeping the semantic mappings here prevents a model or evidence type
from silently changing color between figures.
"""

from __future__ import annotations


# Core Fig. 1 accents.
INK = "#162B50"
MUTED = "#5E636D"
BLUE = "#4472C4"
GREEN = "#00B050"
ORANGE = "#ED7D31"
MAGENTA = "#D186AF"
PURPLE = "#9C54BE"
GRAY = "#A2ACAE"

# Low-saturation companions used for grids and status fills.
GRID = "#D9E2F2"
PALE_GREEN = "#C6E0B4"
PALE_RED = "#F4CCCC"
PASS_TEXT = "#006B3C"
FAIL_TEXT = "#A61C1C"


MODEL_COLORS = {
    "bkan": GREEN,
    "bkan_vi": GREEN,
    "bkan_parameter_mean": GREEN,
    "dkan": BLUE,
    "mlp_l": MAGENTA,
    "spline_ridge": ORANGE,
    "curve_lut": PURPLE,
    "poly3_ridge": INK,
    "tcad_direct": INK,
}

UQ_COLORS = {
    "bkan": GREEN,
    "mc_dropout": ORANGE,
    "ensemble": BLUE,
}
