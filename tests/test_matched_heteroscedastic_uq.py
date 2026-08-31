from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "bkan", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from device_modeling.photodetector.heteroscedastic_mlp import (  # noqa: E402
    HeteroscedasticMLP,
    MLPTrainingConfig,
    heteroscedastic_gaussian_nll,
    predictive_moments,
    train_heteroscedastic_mlp,
)
from kan.bayes_loss import variance_from_logits  # noqa: E402
from run_matched_heteroscedastic_uq import (  # noqa: E402
    _member_training_budgets,
    _holm_adjust,
    _proper_scores,
    corrected_repeated_split_statistics,
    gaussian_crps,
)


class HeteroscedasticMLPTests(unittest.TestCase):
    def test_equal_budget_ensemble_distributes_exact_total_horizon(self):
        source = MLPTrainingConfig(
            epochs=103,
            batch_size=8,
            warmup_epochs=50,
            transition_epochs=30,
            scheduler_step_size=50,
            validation_frequency=10,
            early_stopping_patience=0,
        )
        budgets = _member_training_budgets(
            "equal_budget_deep_ensemble_mlp", source, ensemble_size=5
        )
        self.assertEqual([budget.epochs for budget in budgets], [21, 21, 21, 20, 20])
        self.assertEqual(sum(budget.epochs for budget in budgets), source.epochs)
        self.assertTrue(all(budget.validation_frequency <= budget.epochs for budget in budgets))
        self.assertTrue(
            all(
                budget.warmup_epochs + budget.transition_epochs <= budget.epochs
                for budget in budgets
            )
        )

    def test_per_member_reference_retains_source_budget(self):
        source = MLPTrainingConfig(epochs=12, batch_size=4)
        budgets = _member_training_budgets(
            "deep_ensemble_mlp", source, ensemble_size=3
        )
        self.assertEqual([budget.epochs for budget in budgets], [12, 12, 12])

    def test_two_channel_gaussian_head_has_positive_variance(self):
        model = HeteroscedasticMLP(3, hidden_widths=(8, 4), dropout_rate=0.1)
        inputs = torch.randn(7, 3)
        targets = torch.randn(7, 1)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (7, 2))
        variance = variance_from_logits(output[:, 1:2])
        self.assertTrue(torch.all(variance > 0.0))
        loss = heteroscedastic_gaussian_nll(output, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_fit_uses_exact_requested_update_horizon(self):
        rng = np.random.default_rng(7)
        x_train = rng.normal(size=(24, 2)).astype(np.float32)
        y_train = (
            0.7 * x_train[:, 0] - 0.2 * x_train[:, 1]
            + rng.normal(scale=0.05, size=24)
        ).astype(np.float32)
        x_validation = rng.normal(size=(12, 2)).astype(np.float32)
        y_validation = (
            0.7 * x_validation[:, 0] - 0.2 * x_validation[:, 1]
        ).astype(np.float32)
        config = MLPTrainingConfig(
            epochs=6,
            batch_size=8,
            learning_rate=2.0e-3,
            train_mc_samples=2,
            validation_mc_samples=3,
            validation_frequency=2,
            early_stopping_patience=0,
            warmup_epochs=2,
            transition_epochs=2,
        )
        model, info = train_heteroscedastic_mlp(
            x_train,
            y_train,
            x_validation,
            y_validation,
            hidden_widths=(8, 4),
            dropout_rate=0.1,
            seed=101,
            validation_seed=1729,
            device="cpu",
            config=config,
        )
        self.assertIsInstance(model, HeteroscedasticMLP)
        self.assertEqual(info["epochs_completed"], 6)
        self.assertEqual(info["optimizer_updates"], 6 * math.ceil(24 / 8))
        self.assertTrue(np.isfinite(info["best_validation_nll"]))

    def test_dropout_and_ensemble_moments_are_reproducible(self):
        torch.manual_seed(9)
        dropout = HeteroscedasticMLP(2, (6, 4), dropout_rate=0.2)
        inputs = np.linspace(-1.0, 1.0, 10, dtype=np.float32).reshape(5, 2)
        first = predictive_moments(
            [dropout],
            inputs,
            method="mc_dropout",
            mc_samples=12,
            prediction_seed=33,
            device="cpu",
        )
        second = predictive_moments(
            [dropout],
            inputs,
            method="mc_dropout",
            mc_samples=12,
            prediction_seed=33,
            device="cpu",
        )
        np.testing.assert_allclose(first["mean"], second["mean"], rtol=0.0, atol=0.0)
        self.assertEqual(first["mean_samples"].shape, (12, 5, 1))
        self.assertTrue(np.all(first["aleatoric_variance"] > 0.0))
        self.assertTrue(np.any(first["epistemic_variance"] > 0.0))

        ensemble = [
            HeteroscedasticMLP(2, (6, 4), dropout_rate=0.0),
            HeteroscedasticMLP(2, (6, 4), dropout_rate=0.0),
            HeteroscedasticMLP(2, (6, 4), dropout_rate=0.0),
        ]
        ensemble_result = predictive_moments(
            ensemble,
            inputs,
            method="deep_ensemble",
            mc_samples=99,
            prediction_seed=44,
            device="cpu",
        )
        self.assertEqual(ensemble_result["mean_samples"].shape, (3, 5, 1))
        self.assertTrue(np.all(ensemble_result["total_variance"] > 0.0))

    def test_proper_score_and_dependence_correction_sanity(self):
        actual = np.array([-1.0, 0.0, 1.0])
        exact = gaussian_crps(actual, actual, np.full(3, 0.1))
        shifted = gaussian_crps(actual, actual + 1.0, np.full(3, 0.1))
        self.assertLess(float(exact.mean()), float(shifted.mean()))
        low, high, standard_error, p_value = corrected_repeated_split_statistics(
            np.array([0.1, 0.2, 0.15, 0.18]), 0.65, 0.10
        )
        self.assertLess(low, high)
        self.assertGreater(standard_error, 0.0)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        raw = pd.Series([0.01, 0.04, 0.03])
        adjusted = _holm_adjust(raw)
        self.assertTrue(np.all(adjusted.to_numpy() >= raw.to_numpy()))
        self.assertTrue(np.all(adjusted.to_numpy() <= 1.0))

    def test_proper_scores_apply_deterministic_target_scale(self):
        frame = pd.DataFrame(
            {
                "actual_model_space": [1.0, 2.0],
                "prediction_mean_model_space": [1.1, 1.8],
                "prediction_std_model_space": [0.2, 0.3],
                "calibrated_lower_model_space": [0.5, 1.0],
                "calibrated_upper_model_space": [1.5, 2.6],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            frame.to_csv(path, index=False)
            base = _proper_scores(path)
            scaled = _proper_scores(path, target_scale=1000.0)
        self.assertAlmostEqual(
            scaled["raw_gaussian_nll_task_space"]
            - base["raw_gaussian_nll_task_space"],
            math.log(1000.0),
        )
        self.assertAlmostEqual(
            scaled["raw_gaussian_crps_task_space"],
            1000.0 * base["raw_gaussian_crps_task_space"],
        )


if __name__ == "__main__":
    unittest.main()
