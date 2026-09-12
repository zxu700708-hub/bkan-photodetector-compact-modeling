import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from spectre_provenance import (  # noqa: E402
    LEGACY_PROXY_PASS,
    LEGACY_PROXY_SCOPE,
    LEGACY_PROXY_STALE,
    REPLACEMENT_TERMINAL_Q_STATUS,
    REPLACEMENT_TERMINAL_Q_CIRCUIT_PENDING,
    evaluate_terminal_q_circuit_spectre,
    evaluate_spectre_freshness,
    sha256_file,
)


class SpectreProvenanceTest(unittest.TestCase):
    def test_corrected_source_invalidates_archived_circuit_evidence(self):
        status = evaluate_terminal_q_circuit_spectre(ROOT)
        self.assertFalse(status["passed"], status)
        self.assertEqual(status["status"], REPLACEMENT_TERMINAL_Q_CIRCUIT_PENDING)
        self.assertIn("returned_source_binding_failed", status["failures"])
        self.assertEqual(status["bound_evidence_files"], 49)
        self.assertEqual(status["bias_load_points"], 57)
        self.assertEqual(status["tia_ac_points"], 101)
        self.assertEqual(status["tia_transient_points"], 5001)
        self.assertEqual(status["multi_instance_counts"], [1, 10, 100])

    def test_changed_circuit_psf_invalidates_circuit_status(self):
        def changed_evidence(path):
            if path.name == "tia_ac.ac" and "terminal_charge_circuit_run_20260820" in path.parts:
                return "e" * 64
            return sha256_file(path)

        with patch("spectre_provenance.sha256_file", side_effect=changed_evidence):
            status = evaluate_terminal_q_circuit_spectre(ROOT)
        self.assertFalse(status["passed"])
        self.assertIn("circuit_evidence_hash:results/tia_ac.raw/tia_ac.ac", status["failures"])

    def test_current_requires_both_model_hashes(self):
        baseline = evaluate_spectre_freshness(ROOT)
        self.assertEqual(baseline["status"], LEGACY_PROXY_PASS)
        self.assertEqual(baseline["evidence_scope"], LEGACY_PROXY_SCOPE)
        self.assertEqual(
            baseline["replacement_terminal_q"]["status"],
            REPLACEMENT_TERMINAL_Q_STATUS,
        )
        self.assertFalse(
            baseline["replacement_terminal_q"]["evidence_transfer_allowed"]
        )

        current_dc = (
            ROOT / "bkan/device_modeling/veriloga/ge_si_pdet_fixed.va"
        )

        def changed_current_model(path):
            if path == current_dc:
                return "0" * 64
            return sha256_file(path)

        with patch(
            "spectre_provenance.sha256_file",
            side_effect=changed_current_model,
        ):
            status = evaluate_spectre_freshness(ROOT)
        self.assertEqual(status["status"], LEGACY_PROXY_STALE)
        self.assertFalse(status["models"]["independent_dc"]["hash_match"])

    def test_changed_psf_evidence_invalidates_status(self):
        def changed_evidence(path):
            if path.name == "dc_dark.dc" and "result" in path.parts:
                return "f" * 64
            return sha256_file(path)

        with patch(
            "spectre_provenance.sha256_file",
            side_effect=changed_evidence,
        ):
            status = evaluate_spectre_freshness(ROOT)
        self.assertEqual(status["status"], LEGACY_PROXY_STALE)
        self.assertFalse(status["result_evidence_ok"])


if __name__ == "__main__":
    unittest.main()
