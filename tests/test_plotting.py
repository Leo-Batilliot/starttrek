import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.plotting import (
    plot_episode_length,
    plot_epsilon,
    plot_loss,
    plot_returns,
    plot_termination_breakdown,
)


METRIC_COLUMNS = [
    "episode",
    "global_step",
    "return",
    "length",
    "epsilon",
    "loss_mean",
    "loss_std",
    "termination_reason",
    "terminated",
    "truncated",
    "wall_time_s",
]


def _write_metrics(path: Path, n: int = 200) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        w.writeheader()
        for i in range(1, n + 1):
            eps = max(0.05, 1.0 - i * 0.005)
            ret = -150 + i * 2.0
            loss = "" if i < 10 else f"{max(0.001, 0.5 - i * 0.002):.4f}"
            reason = "land" if i > 150 else ("crash" if i % 2 == 0 else "out_of_view")
            w.writerow({
                "episode": i,
                "global_step": i * 100,
                "return": ret,
                "length": 100 + i % 50,
                "epsilon": eps,
                "loss_mean": loss,
                "loss_std": "" if loss == "" else "0.05",
                "termination_reason": reason,
                "terminated": True if reason != "truncated" else False,
                "truncated": reason == "truncated",
                "wall_time_s": i * 1.5,
            })


class TestPlotting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.csv = self.dir / "metrics.csv"
        _write_metrics(self.csv, n=200)

    def tearDown(self):
        self.tmp.cleanup()

    def _assert_png(self, path: Path) -> None:
        self.assertTrue(path.exists(), f"{path} should exist")
        self.assertGreater(path.stat().st_size, 0, f"{path} should not be empty")
        header = path.read_bytes()[:8]
        self.assertEqual(header[:4], b"\x89PNG", f"{path} should start with PNG magic")

    def test_plot_returns(self):
        out = self.dir / "returns.png"
        plot_returns(self.csv, out)
        self._assert_png(out)

    def test_plot_returns_creates_parent_dir(self):
        out = self.dir / "nested" / "returns.png"
        plot_returns(self.csv, out)
        self._assert_png(out)

    def test_plot_loss(self):
        out = self.dir / "loss.png"
        plot_loss(self.csv, out)
        self._assert_png(out)

    def test_plot_episode_length(self):
        out = self.dir / "episode_length.png"
        plot_episode_length(self.csv, out)
        self._assert_png(out)

    def test_plot_epsilon(self):
        out = self.dir / "epsilon.png"
        plot_epsilon(self.csv, out)
        self._assert_png(out)

    def test_plot_termination_breakdown(self):
        out = self.dir / "breakdown.png"
        plot_termination_breakdown(self.csv, out)
        self._assert_png(out)

    def test_missing_csv_raises(self):
        with self.assertRaises(FileNotFoundError):
            plot_returns(self.dir / "does_not_exist.csv", self.dir / "x.png")

    def test_missing_column_raises(self):
        bad = self.dir / "bad.csv"
        with bad.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["episode", "wrong_column"])
            w.writerow([1, 10])
        with self.assertRaises(ValueError):
            plot_returns(bad, self.dir / "x.png")


if __name__ == "__main__":
    unittest.main()
