import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from aggregate_runs import aggregate, mean_ci95


class TestMeanCI95(unittest.TestCase):
    def test_known_values(self):
        mean, lo, hi = mean_ci95([1, 2, 3, 4, 5])
        self.assertAlmostEqual(mean, 3.0)
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        # symmetric around the mean
        self.assertAlmostEqual(mean - lo, hi - mean)

    def test_single_value_zero_width(self):
        self.assertEqual(mean_ci95([42.0]), (42.0, 42.0, 42.0))

    def test_empty(self):
        self.assertEqual(mean_ci95([]), (0.0, 0.0, 0.0))


class TestAggregate(unittest.TestCase):
    def _seed(self, seed, ret, solved, fss):
        return {
            "seed": seed,
            "mean_return_last_100": ret,
            "solved": solved,
            "first_solve_step": fss,
        }

    def test_all_solved(self):
        per = [
            self._seed(0, 246.0, True, 250000),
            self._seed(1, 230.0, True, 300000),
            self._seed(2, 255.0, True, 240000),
            self._seed(3, 210.0, True, 280000),
            self._seed(4, 248.0, True, 260000),
        ]
        agg = aggregate(per)
        self.assertEqual(agg["n_seeds"], 5)
        self.assertEqual(agg["n_solved"], 5)
        self.assertTrue(agg["all_solved"])
        self.assertTrue(agg["criterion_met"])
        self.assertAlmostEqual(agg["mean_return_last_100"]["mean"], 237.8)
        self.assertEqual(agg["first_solve_step"]["n_solved_seeds"], 5)

    def test_four_of_five_meets_criterion(self):
        per = [
            self._seed(0, 246.0, True, 250000),
            self._seed(1, 230.0, True, 300000),
            self._seed(2, 255.0, True, 240000),
            self._seed(3, 210.0, True, 280000),
            self._seed(4, -120.0, False, None),
        ]
        agg = aggregate(per)
        self.assertEqual(agg["n_solved"], 4)
        self.assertFalse(agg["all_solved"])
        self.assertTrue(agg["criterion_met"])
        # first_solve_step aggregated only over solved seeds
        self.assertEqual(agg["first_solve_step"]["n_solved_seeds"], 4)

    def test_three_of_five_fails_criterion(self):
        per = [
            self._seed(0, 246.0, True, 250000),
            self._seed(1, 230.0, True, 300000),
            self._seed(2, 255.0, True, 240000),
            self._seed(3, -50.0, False, None),
            self._seed(4, -120.0, False, None),
        ]
        agg = aggregate(per)
        self.assertEqual(agg["n_solved"], 3)
        self.assertFalse(agg["criterion_met"])

    def test_no_seed_solved_first_solve_step_none(self):
        per = [self._seed(0, -100.0, False, None), self._seed(1, -90.0, False, None)]
        agg = aggregate(per)
        self.assertIsNone(agg["first_solve_step"]["mean"])
        self.assertEqual(agg["first_solve_step"]["n_solved_seeds"], 0)


if __name__ == "__main__":
    unittest.main()
