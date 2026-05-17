import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import aggregate, aggregate_runs, mean_ci95


class TestMeanCI95(unittest.TestCase):
    def test_acceptance_example(self):
        # T3.2 acceptance: mean_ci95([1,2,3,4,5]) gives a coherent CI.
        mean, lo, hi = mean_ci95([1, 2, 3, 4, 5])
        self.assertAlmostEqual(mean, 3.0)
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        self.assertAlmostEqual(mean - lo, hi - mean)  # symmetric
        # sample std = 1.5811, sem = 0.7071, z=1.96 → half-width ≈ 1.386
        self.assertAlmostEqual(hi - mean, 1.3859, places=3)

    def test_single_and_empty(self):
        self.assertEqual(mean_ci95([7.0]), (7.0, 7.0, 7.0))
        self.assertEqual(mean_ci95([]), (0.0, 0.0, 0.0))

    def test_returns_floats(self):
        for v in mean_ci95([10, 20, 30]):
            self.assertIsInstance(v, float)


def _summary(ret, solved, fss):
    return {
        "mean_return_last_100": ret,
        "solved": solved,
        "first_solve_step": fss,
        "eval_episodes": 100,
    }


class TestAggregate(unittest.TestCase):
    def test_five_solved(self):
        per = [
            {"seed": i, **_summary(r, True, s)}
            for i, (r, s) in enumerate(
                [(259.5, 110000), (274.6, 140000), (273.0, 90000),
                 (279.7, 110000), (249.1, 210000)]
            )
        ]
        agg = aggregate(per)
        self.assertEqual(agg["n_solved"], 5)
        self.assertTrue(agg["all_solved"])
        self.assertTrue(agg["criterion_met"])
        self.assertAlmostEqual(agg["mean_return_last_100"]["mean"], 267.18, places=1)
        self.assertEqual(agg["first_solve_step"]["n_solved_seeds"], 5)

    def test_four_of_five_passes(self):
        per = [
            {"seed": 0, **_summary(246.0, True, 250000)},
            {"seed": 1, **_summary(230.0, True, 300000)},
            {"seed": 2, **_summary(255.0, True, 240000)},
            {"seed": 3, **_summary(210.0, True, 280000)},
            {"seed": 4, **_summary(-120.0, False, None)},
        ]
        agg = aggregate(per)
        self.assertEqual(agg["n_solved"], 4)
        self.assertFalse(agg["all_solved"])
        self.assertTrue(agg["criterion_met"])
        self.assertEqual(agg["first_solve_step"]["n_solved_seeds"], 4)

    def test_three_of_five_fails(self):
        per = [{"seed": i, **_summary(r, sv, f)} for i, (r, sv, f) in enumerate(
            [(246.0, True, 250000), (230.0, True, 300000), (255.0, True, 240000),
             (-50.0, False, None), (-120.0, False, None)])]
        self.assertFalse(aggregate(per)["criterion_met"])

    def test_no_solved_first_solve_none(self):
        per = [{"seed": 0, **_summary(-100.0, False, None)}]
        agg = aggregate(per)
        self.assertIsNone(agg["first_solve_step"]["mean"])
        self.assertEqual(agg["first_solve_step"]["n_solved_seeds"], 0)


class TestAggregateRuns(unittest.TestCase):
    def test_globs_and_sorts_by_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "dqn_baseline"
            for seed, ret in ((2, 255.0), (0, 246.0), (10, 230.0)):
                d = base / str(seed)
                d.mkdir(parents=True)
                with (d / "summary.json").open("w") as f:
                    json.dump(_summary(ret, True, 100000 + seed), f)
            agg = aggregate_runs(str(base / "*" / "summary.json"))
        self.assertEqual([r["seed"] for r in agg["per_seed"]], [0, 2, 10])
        self.assertEqual(agg["n_solved"], 3)
        self.assertTrue(agg["criterion_met"] is False)  # 3 < 4

    def test_no_match_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                aggregate_runs(str(Path(tmp) / "nope" / "*" / "summary.json"))


if __name__ == "__main__":
    unittest.main()
