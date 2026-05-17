import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import stats  # noqa: E402


def _summary(ret, solved=True, fss=100000):
    return {
        "mean_return_last_100": ret, "std_return": 20.0, "solved": solved,
        "first_solve_step": fss, "wall_time_s": 1000.0,
        "termination_breakdown": {"land": 0.9, "crash": 0.05, "truncated": 0.05},
    }


def _eval(mean):
    return {
        "mean_return": mean, "std_return": 50.0, "solved": mean >= 200,
        "ci95_return": [mean - 10, mean + 10], "min_return": mean - 80,
        "max_return": mean + 40, "median_return": mean, "episodes": 100,
        "returns": [mean] * 100,
        "termination_breakdown": {"land": 0.8, "crash": 0.1, "truncated": 0.1},
    }


class TestStats(unittest.TestCase):
    def _build(self, root: Path):
        for s, r in enumerate([260.0, 270.0]):
            d = root / "dqn_baseline" / str(s)
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps(_summary(r)))
        for s, r in enumerate([235.0, 245.0]):
            d = root / "dqn_buffer10k" / str(s)
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps(_summary(r)))
        rd = root / "random" / "0"
        rd.mkdir(parents=True)
        (rd / "eval_20260101T000000.json").write_text(json.dumps(_eval(-185.0)))

    def test_runs_console_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            self._build(root)
            argv = ["stats.py", "--runs-root", str(root),
                    "--seeds", "0", "1", "--no-files"]
            with patch.object(sys, "argv", argv):
                self.assertEqual(stats.main(), 0)

    def test_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            out = Path(tmp) / "stats"
            self._build(root)
            argv = ["stats.py", "--runs-root", str(root), "--seeds", "0", "1",
                    "--out-dir", str(out)]
            with patch.object(sys, "argv", argv):
                self.assertEqual(stats.main(), 0)
            for f in ("comparison.csv", "comparison.md",
                      "significance.csv", "significance.md"):
                self.assertTrue((out / f).exists(), f)
            self.assertIn("baseline_vs_dqn_buffer10k",
                          (out / "significance.csv").read_text())

    def test_no_runs_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["stats.py", "--runs-root", str(Path(tmp) / "empty"),
                    "--no-files"]
            with patch.object(sys, "argv", argv):
                self.assertEqual(stats.main(), 84)


if __name__ == "__main__":
    unittest.main()
