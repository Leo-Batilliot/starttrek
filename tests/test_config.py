import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.utils.config import (
    AgentConfig,
    EnvConfig,
    EpsilonConfig,
    EvalConfig,
    NetworkConfig,
    RunConfig,
    TargetUpdateConfig,
    TrainingConfig,
    WrapperConfig,
    load_config,
    save_config,
)


REF_YAML = Path("configs/baseline/dqn.yaml")


class TestConfigLoad(unittest.TestCase):
    def test_load_returns_runconfig(self):
        cfg = load_config(REF_YAML)
        self.assertIsInstance(cfg, RunConfig)

    def test_load_typed_fields(self):
        cfg = load_config(REF_YAML)
        self.assertEqual(cfg.run_name, "dqn_baseline")
        self.assertEqual(cfg.seed, 0)

        self.assertIsInstance(cfg.env, EnvConfig)
        self.assertEqual(cfg.env.id, "LunarLander-v3")
        self.assertEqual(len(cfg.env.wrappers), 1)
        self.assertIsInstance(cfg.env.wrappers[0], WrapperConfig)
        self.assertEqual(cfg.env.wrappers[0].name, "RecordEpisodeStats")

        self.assertIsInstance(cfg.agent, AgentConfig)
        self.assertEqual(cfg.agent.name, "dqn")
        self.assertIsInstance(cfg.agent.network, NetworkConfig)
        self.assertEqual(cfg.agent.network.hidden_sizes, (128, 128))
        self.assertEqual(cfg.agent.network.activation, "relu")
        self.assertAlmostEqual(cfg.agent.gamma, 0.99)
        self.assertAlmostEqual(cfg.agent.lr, 5.0e-4)

        self.assertIsInstance(cfg.agent.target_update, TargetUpdateConfig)
        self.assertEqual(cfg.agent.target_update.kind, "hard")
        self.assertEqual(cfg.agent.target_update.every_steps, 1000)

        self.assertIsInstance(cfg.agent.epsilon, EpsilonConfig)
        self.assertEqual(cfg.agent.epsilon.start, 1.0)
        self.assertEqual(cfg.agent.epsilon.end, 0.05)

        self.assertIsInstance(cfg.training, TrainingConfig)
        self.assertEqual(cfg.training.total_steps, 500_000)

        self.assertIsInstance(cfg.eval, EvalConfig)
        self.assertEqual(cfg.eval.episodes, 100)
        self.assertTrue(cfg.eval.greedy)


class TestConfigMissingKey(unittest.TestCase):
    def _write(self, data) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(data, f)
        f.close()
        return f.name

    def test_missing_key_raises_with_yaml_path(self):
        bad = yaml.safe_load(REF_YAML.read_text())
        del bad["agent"]["epsilon"]["end"]
        path = self._write(bad)
        with self.assertRaises(ValueError) as ctx:
            load_config(path)
        msg = str(ctx.exception)
        self.assertIn(path, msg)
        self.assertIn("agent.epsilon.end", msg)

    def test_missing_top_level_key_raises(self):
        bad = yaml.safe_load(REF_YAML.read_text())
        del bad["training"]
        path = self._write(bad)
        with self.assertRaises(ValueError) as ctx:
            load_config(path)
        self.assertIn("training", str(ctx.exception))


class TestConfigRoundtrip(unittest.TestCase):
    def test_save_then_load_equals_original(self):
        cfg = load_config(REF_YAML)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            out = f.name
        save_config(cfg, out)
        cfg2 = load_config(out)
        self.assertEqual(cfg, cfg2)

    def test_roundtrip_preserves_tuple_fields(self):
        cfg = load_config(REF_YAML)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            out = f.name
        save_config(cfg, out)
        cfg2 = load_config(out)
        self.assertIsInstance(cfg2.agent.network.hidden_sizes, tuple)
        self.assertIsInstance(cfg2.env.wrappers, tuple)
        self.assertEqual(cfg.agent.network.hidden_sizes, cfg2.agent.network.hidden_sizes)


if __name__ == "__main__":
    unittest.main()
