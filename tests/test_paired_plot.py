import math
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from paired_plot import aggregate_curves, binned_means


class TestBinnedMeans(unittest.TestCase):
    def test_assigns_to_correct_bins(self):
        edges = [0.0, 10.0, 20.0]
        steps = [1, 5, 9, 12, 18]
        rets = [10, 20, 30, 100, 200]
        out = binned_means(steps, rets, edges)
        self.assertAlmostEqual(out[0], 20.0)   # mean(10,20,30)
        self.assertAlmostEqual(out[1], 150.0)  # mean(100,200)

    def test_empty_bin_is_nan(self):
        out = binned_means([1, 2], [5, 7], [0.0, 10.0, 20.0])
        self.assertAlmostEqual(out[0], 6.0)
        self.assertTrue(math.isnan(out[1]))

    def test_right_edge_included_in_last_bin(self):
        out = binned_means([20.0], [42.0], [0.0, 10.0, 20.0])
        self.assertTrue(math.isnan(out[0]))
        self.assertAlmostEqual(out[1], 42.0)


class TestAggregateCurves(unittest.TestCase):
    def test_mean_and_ci_per_bin(self):
        per_seed = [
            [100.0, 200.0],
            [110.0, 220.0],
            [90.0, 180.0],
        ]
        means, los, his = aggregate_curves(per_seed)
        self.assertAlmostEqual(means[0], 100.0)
        self.assertAlmostEqual(means[1], 200.0)
        self.assertLess(los[0], means[0])
        self.assertGreater(his[1], means[1])

    def test_nan_bins_ignored_then_all_nan(self):
        nan = float("nan")
        per_seed = [[nan, 50.0], [nan, 70.0]]
        means, los, his = aggregate_curves(per_seed)
        self.assertTrue(math.isnan(means[0]))
        self.assertAlmostEqual(means[1], 60.0)


if __name__ == "__main__":
    unittest.main()
