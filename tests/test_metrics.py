import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import (
    aggregate,
    aggregate_runs,
    betai,
    cohens_d,
    compare_samples,
    mean_ci95,
    welch_ttest,
)


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


class TestBetai(unittest.TestCase):
    def test_closed_forms(self):
        # I_x(1,1) = x
        for x in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(betai(1.0, 1.0, x), x, places=6)
        # I_x(1/2,1/2) = (2/pi) * arcsin(sqrt(x))
        for x in (0.05, 0.3, 0.75):
            self.assertAlmostEqual(
                betai(0.5, 0.5, x), 2.0 / math.pi * math.asin(math.sqrt(x)), places=6
            )
        self.assertEqual(betai(2.0, 3.0, 0.0), 0.0)
        self.assertEqual(betai(2.0, 3.0, 1.0), 1.0)

    def test_t_to_p_matches_normal_for_large_df(self):
        # df -> inf, |t| = 1.959964  =>  two-sided p -> 0.05
        t, df = 1.959963985, 1.0e7
        p = betai(df / 2.0, 0.5, df / (df + t * t))
        self.assertAlmostEqual(p, 0.05, places=4)


class TestWelchAndCohen(unittest.TestCase):
    def setUp(self):
        # Equal-variance case: Welch reduces to the pooled t-test.
        self.a = [10.0, 12.0, 14.0, 16.0, 18.0]   # mean 14, var 10
        self.b = [20.0, 22.0, 24.0, 26.0, 28.0]   # mean 24, var 10

    def test_identical_samples(self):
        t, df, p = welch_ttest(self.a, self.a)
        self.assertEqual(t, 0.0)
        self.assertAlmostEqual(p, 1.0, places=6)

    def test_known_t_df_p(self):
        t, df, p = welch_ttest(self.a, self.b)
        self.assertAlmostEqual(t, -5.0, places=6)
        self.assertAlmostEqual(df, 8.0, places=6)
        self.assertTrue(0.0009 < p < 0.0013, p)  # table: t=5,df=8 -> ~0.00106

    def test_cohens_d_pooled(self):
        self.assertAlmostEqual(cohens_d(self.a, self.b), -10.0 / math.sqrt(10.0), places=6)

    def test_compare_samples_verdict(self):
        c = compare_samples(self.a, self.b)
        self.assertAlmostEqual(c["diff"], 10.0, places=6)
        self.assertTrue(c["ci95_disjoint"])
        self.assertTrue(c["significant"])
        self.assertIn("SIGNIFICATIF", c["verdict"])

    def test_too_few_samples_raise(self):
        with self.assertRaises(ValueError):
            welch_ttest([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            cohens_d([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
