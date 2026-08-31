import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))

from device_modeling.photodetector.posterior_export import posterior_draw_to_deterministic
from kan.BayesMultKAN import BayesMultKAN


class PosteriorExportTests(unittest.TestCase):
    def test_draw_is_reproducible_and_seed_sensitive(self):
        model = BayesMultKAN(width=[2, 3, 2], grid=3, k=2, seed=7, device="cpu")
        x = torch.tensor([[-0.4, 0.2], [0.1, 0.7]], dtype=torch.float32)
        first = posterior_draw_to_deterministic(model, 101)
        repeat = posterior_draw_to_deterministic(model, 101)
        other = posterior_draw_to_deterministic(model, 102)
        with torch.no_grad():
            y_first = first(x)
            y_repeat = repeat(x)
            y_other = other(x)
        self.assertTrue(torch.allclose(y_first, y_repeat))
        self.assertFalse(torch.allclose(y_first, y_other))
        self.assertEqual(y_first.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
