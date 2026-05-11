import csv
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.noop_agent import NoOpAgent
from src.env.factory import build_env
from src.training.trainer import METRIC_COLUMNS, Trainer
from src.utils.config import load_config, save_config
from src.utils.logger import Logger
from src.utils.seeding import seed_all


SMOKE_CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


class TestSmokeEndToEnd(unittest.TestCase):
    def test_smoke_config_loads(self):
        cfg = load_config(SMOKE_CFG_PATH)
        self.assertEqual(cfg.run_name, "smoke")
        self.assertEqual(cfg.agent.name, "noop")
        self.assertEqual(cfg.env.id, "LunarLander-v3")

    def test_trainer_runs_end_to_end_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(SMOKE_CFG_PATH)
            short_training = replace(cfg.training, total_steps=400, save_every=200, eval_every=200)
            cfg = replace(cfg, training=short_training)

            seed_all(seed=cfg.seed)
            run_dir = Path(tmp) / cfg.run_name / str(cfg.seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            save_config(cfg, run_dir / "config.yaml")

            env = build_env(cfg.env, seed=cfg.seed)
            try:
                agent = NoOpAgent(cfg.agent, n_actions=env.action_space.n)
                logger = Logger(run_dir, METRIC_COLUMNS)
                try:
                    trainer = Trainer(cfg, env, agent, logger, run_dir)
                    summary = trainer.run()
                finally:
                    logger.close()
            finally:
                env.close()

            self.assertEqual(summary["total_steps"], 400)
            self.assertGreater(summary["episodes"], 0, "at least one episode must finish in 400 steps")

            for child in ("config.yaml", "metrics.csv", "events.log"):
                self.assertTrue((run_dir / child).exists(), f"missing {child}")
            self.assertTrue((run_dir / "checkpoints" / "final.pt").exists())

            with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, METRIC_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), summary["episodes"])
            for row in rows:
                self.assertIn(row["termination_reason"], {
                    "land", "crash", "out_of_view", "sleep", "truncated", "unknown",
                })


if __name__ == "__main__":
    unittest.main()
