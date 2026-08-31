from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))

from device_modeling.photodetector.symbolic_gated_kan import (  # noqa: E402
    evaluate_exported_pure_symbolic_formula,
)
from device_modeling.photodetector.task_config import (  # noqa: E402
    TASKS,
    apply_dark_current_floor,
    prepare_task_dataframe,
)
from device_modeling.veriloga.export_net_photocurrent import (  # noqa: E402
    FUNCTION_BEGIN,
    evaluate_deployment_graph,
    generate_function,
    update_veriloga,
)


FORMULA = (
    ROOT
    / "artifacts/results/net_photocurrent_retrain/symbolic_fixed_teacher/"
    "seed_42/photo_current_symbolic_gated_kan/pure_symbolic_formula.json"
)
DATA = ROOT / "artifacts/results/device_modeling/cleaned_data.csv"
SOURCE = ROOT / "bkan/device_modeling/veriloga/ge_si_photodetector_terminal_charge.va"


def _payload_and_inputs() -> tuple[dict, np.ndarray]:
    payload = json.loads(FORMULA.read_text(encoding="utf-8"))
    prepared, _ = prepare_task_dataframe(
        apply_dark_current_floor(pd.read_csv(DATA)), TASKS["photo_current"]
    )
    inputs = tuple(payload["input_scalers"])
    return payload, prepared[list(inputs)].to_numpy(dtype=np.float64)


class NetPhotocurrentVerilogAExportTests(unittest.TestCase):
    def test_deployment_graph_replays_exported_net_photocurrent(self) -> None:
        payload, raw = _payload_and_inputs()
        reference = evaluate_exported_pure_symbolic_formula(payload, raw)
        replay, physical = evaluate_deployment_graph(payload, raw)
        self.assertTrue(np.isfinite(physical).all())
        self.assertLess(
            float(np.max(np.abs(replay - reference))),
            1e-6 * float(np.max(np.abs(reference))),
        )

    def test_generated_source_replaces_illuminated_total_assignment(self) -> None:
        payload, _ = _payload_and_inputs()
        function = generate_function(payload)
        source = SOURCE.read_text(encoding="utf-8")
        updated = update_veriloga(source, function)
        self.assertIn(FUNCTION_BEGIN, updated)
        self.assertIn("net_photocurrent_model(vd_safe", updated)
        self.assertNotIn("photo_current = smax0(pow(10", updated)
        self.assertIn("dark_current + light_power * photo_current", updated)


if __name__ == "__main__":
    unittest.main()
