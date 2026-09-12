"""Run the seven-model comparison with GMLS-adapted excluded.

This wrapper preserves the eight-model runner and its historical outputs while
reusing the identical dataset split, training budget, metrics, and audit logic.
The frozen-confirmation partition remains quarantined.
"""

from __future__ import annotations

from pathlib import Path
import sys

import run_ring_compact_variation_eight as runner


runner.MODELS = tuple(model for model in runner.MODELS if model != "gmls")
runner.DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "results"
    / "ring_compact_variation_seven_no_gmls"
)


if __name__ == "__main__":
    # Match the completed eight-model exploratory run unless explicitly
    # overridden on the command line.
    if "--seeds" not in sys.argv:
        sys.argv.extend(["--seeds", "42"])
    if "--batch-size" not in sys.argv:
        sys.argv.extend(["--batch-size", "512"])
    raise SystemExit(runner.main())
