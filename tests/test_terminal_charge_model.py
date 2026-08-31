from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "bkan"
    / "device_modeling"
    / "terminal_charge"
)
sys.path.insert(0, str(MODULE_DIR))
import train  # noqa: E402


class TerminalChargeModelTests(unittest.TestCase):
    def test_replacement_simulator_status_cannot_inherit_proxy_evidence(self) -> None:
        status = train.SIMULATOR_EXECUTION_STATUS
        self.assertEqual(status["spectre_compilation"], "pending")
        self.assertEqual(status["spectre_dc"], "pending")
        self.assertEqual(status["spectre_ac_admittance"], "pending")
        self.assertEqual(status["spectre_transient_stability"], "pending")
        self.assertFalse(status["legacy_proxy_evidence_transfer"])

    def test_replacement_source_is_not_the_retired_proxy_template(self) -> None:
        source = (Path(__file__).resolve().parents[1] / train.DEFAULT_TEMPLATE).read_text(
            encoding="utf-8"
        )
        self.assertIn("module ge_si_photodetector_terminal_charge", source)
        self.assertIn("Locally audited terminal-charge", source)
        self.assertNotIn("module ge_si_photodetector_charge_proxy", source)
        self.assertNotIn("q_proxy", source)

    def test_voltage_basis_is_exactly_zero_at_reference(self) -> None:
        basis, derivative = train.voltage_design(
            np.asarray([0.0]), np.asarray(train.DEFAULT_KNOTS)
        )
        np.testing.assert_array_equal(basis, np.zeros_like(basis))
        self.assertTrue(np.isfinite(derivative).all())

    def test_joint_fit_recovers_reference_constrained_quadratic(self) -> None:
        rows = []
        for source_row, offset in enumerate(np.linspace(-1.0, 1.0, 12)):
            parameters = {
                "trap_assisted_recomb_A": 1.0e-9 + offset * 1.0e-10,
                "ge_sio2_recomb_velocity": 2500.0 + offset * 100.0,
                "ge_si_recomb_velocity": 500.0 + offset * 20.0,
                "active_layer_length": 40.0 + offset * 2.0,
                "simulation_temperature": 300.0 + offset * 5.0,
            }
            gain = 1.0e-15 * (1.0 + 0.1 * offset)
            for voltage in np.linspace(-3.0, 0.0, 13):
                rows.append(
                    {
                        **parameters,
                        "production_source_row": source_row,
                        "bias_v": voltage,
                        train.Q_COL: gain * (voltage + 0.1 * voltage**2),
                        train.C_COL: gain * (1.0 + 0.2 * voltage),
                    }
                )
        frame = pd.DataFrame(rows)
        derivative = frame[frame["bias_v"].isin([-3.0, -1.5, 0.0])].copy()
        model = train.fit_model(frame, derivative, 1.0e-10, 1.0)
        np.testing.assert_allclose(
            model.predict(frame), frame[train.Q_COL], rtol=1.0e-5, atol=1.0e-22
        )
        np.testing.assert_allclose(
            model.predict_derivative(derivative),
            derivative[train.C_COL],
            rtol=1.0e-5,
            atol=1.0e-22,
        )
        zero = frame.copy()
        zero["bias_v"] = 0.0
        np.testing.assert_array_equal(model.predict(zero), np.zeros(len(zero)))


if __name__ == "__main__":
    unittest.main()
