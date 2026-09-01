import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from housing_ml.pipeline import RegressionMetrics, ablation_delta, choose_k


class PipelineTest(unittest.TestCase):
    def test_k_selection(self):
        rng = np.random.default_rng(42)
        matrix = np.r_[rng.normal(0, 0.1, (10, 2)), rng.normal(10, 0.1, (10, 2))]
        best, scores = choose_k(matrix, candidates=[2, 3])
        self.assertEqual(best, 2)
        self.assertEqual(set(scores), {2, 3})

    def test_ablation_is_signed(self):
        self.assertEqual(ablation_delta(RegressionMetrics(100, 0.5), RegressionMetrics(99, 0.5)), 1)


if __name__ == "__main__":
    unittest.main()
