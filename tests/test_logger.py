import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import Logger


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


def make_row(
    episode: int,
    *,
    ret: float = -100.0,
    eps: float = 1.0,
    loss_mean=None,
    term: str = "crash",
) -> dict:
    return {
        "episode": episode,
        "global_step": episode * 100,
        "return": ret,
        "length": 100,
        "epsilon": eps,
        "loss_mean": loss_mean,
        "loss_std": None if loss_mean is None else 0.05,
        "termination_reason": term,
        "terminated": term != "truncated",
        "truncated": term == "truncated",
        "wall_time_s": episode * 1.5,
    }


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        self.csv_path = self.run_dir / "metrics.csv"
        self.events_path = self.run_dir / "events.log"

    def tearDown(self):
        self.tmp.cleanup()


    def test_files_are_created(self):
        logger = Logger(self.run_dir, METRIC_COLUMNS)
        logger.close()
        self.assertTrue(self.csv_path.exists(), "metrics.csv should exist")
        self.assertTrue(self.events_path.exists(), "events.log should exist")


    def test_csv_header_is_columns_in_order(self):
        logger = Logger(self.run_dir, METRIC_COLUMNS)
        logger.close()
        with self.csv_path.open() as f:
            header = next(csv.reader(f))
        self.assertEqual(header, METRIC_COLUMNS)

    def test_log_writes_one_row_per_call(self):
        logger = Logger(self.run_dir, METRIC_COLUMNS)
        logger.log(**make_row(episode=1))
        logger.log(**make_row(episode=2))
        logger.log(**make_row(episode=3))
        logger.close()
        with self.csv_path.open() as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 4, "expected header + 3 rows")
        self.assertEqual(rows[1][0], "1")
        self.assertEqual(rows[3][0], "3")

    def test_100_calls_produce_valid_csv(self):
        logger = Logger(self.run_dir, METRIC_COLUMNS)
        for i in range(1, 101):
            logger.log(
                **make_row(
                    episode=i,
                    eps=max(0.05, 1.0 - i * 0.01),
                    ret=-150 + i * 2.0,
                    loss_mean=None if i < 10 else 0.5 - i * 0.002,
                    term="land" if i > 80 else "crash",
                )
            )
        logger.close()

        with self.csv_path.open() as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, METRIC_COLUMNS)
            rows = list(reader)
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows[0]["episode"], "1")
        self.assertEqual(rows[-1]["episode"], "100")
        self.assertEqual(rows[-1]["termination_reason"], "land")

    def test_missing_field_writes_empty_cell(self):
        logger = Logger(self.run_dir, METRIC_COLUMNS)
        logger.log(
            episode=1,
            global_step=100,
            length=100,
            termination_reason="crash",
            terminated=True,
            truncated=False,
            wall_time_s=1.0,
            **{"return": -50.0},
        )
        logger.close()
        with self.csv_path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["loss_mean"], "")
        self.assertEqual(rows[0]["loss_std"], "")
        self.assertEqual(rows[0]["epsilon"], "")
        self.assertEqual(rows[0]["episode"], "1")

    def test_extra_field_does_not_break(self):
        logger = Logger(self.run_dir, METRIC_COLUMNS)
        row = make_row(episode=1)
        row["debug_extra"] = 42  # not in METRIC_COLUMNS
        logger.log(**row)
        logger.close()
        with self.csv_path.open() as f:
            header = next(csv.reader(f))
        self.assertEqual(header, METRIC_COLUMNS)


    def test_events_log_mirrors_csv_count(self):
        logger = Logger(self.run_dir, METRIC_COLUMNS)
        for i in range(1, 6):
            logger.log(**make_row(episode=i))
        logger.close()
        non_empty = [
            line for line in self.events_path.read_text().splitlines() if line.strip()
        ]
        self.assertGreaterEqual(
            len(non_empty), 5, "events.log should have at least 1 line per log() call"
        )

    def test_events_log_contains_episode_numbers(self):
        logger = Logger(self.run_dir, METRIC_COLUMNS)
        for i in (1, 2, 42):
            logger.log(**make_row(episode=i))
        logger.close()
        text = self.events_path.read_text()
        self.assertIn("42", text, "events.log should mention episode 42")
        self.assertIn("1", text)



def demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        logger = Logger(run_dir, METRIC_COLUMNS)
        for i in range(1, 11):
            logger.log(
                **make_row(
                    episode=i,
                    eps=max(0.05, 1.0 - i * 0.05),
                    ret=-150 + i * 18.0,
                    loss_mean=None if i < 3 else 0.5 - i * 0.02,
                    term="land" if i > 7 else "crash",
                )
            )
        logger.close()

        print(f"\n=== run_dir: {run_dir} ===\n")
        print("--- metrics.csv ---")
        print((run_dir / "metrics.csv").read_text())
        print("--- events.log ---")
        print((run_dir / "events.log").read_text())


if __name__ == "__main__":
        unittest.main()
