import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import Agent, Transition
from src.agents.random_agent import RandomAgent
from src.agents.registry import AGENTS
from src.utils.config import (
    AgentConfig,
    EpsilonConfig,
    NetworkConfig,
    TargetUpdateConfig,
)
from src.utils.seeding import seed_all


def _agent_cfg() -> AgentConfig:
    return AgentConfig(
        name="random",
        network=NetworkConfig(hidden_sizes=(8,), activation="relu"),
        gamma=0.99,
        lr=1e-3,
        batch_size=1,
        buffer_size=1,
        min_buffer_size=0,
        target_update=TargetUpdateConfig(kind="hard", every_steps=1, tau=0.0),
        epsilon=EpsilonConfig(start=1.0, end=1.0, decay_steps=1),
        grad_clip=1.0,
    )


class TestRandomAgent(unittest.TestCase):
    def test_registered_under_random(self):
        self.assertIn("random", AGENTS)
        self.assertIs(AGENTS["random"], RandomAgent)

    def test_is_an_agent(self):
        seed_all(seed=0)
        agent = RandomAgent(_agent_cfg(), n_actions=4)
        self.assertIsInstance(agent, Agent)

    def test_act_returns_in_range(self):
        seed_all(seed=0)
        agent = RandomAgent(_agent_cfg(), n_actions=4)
        obs = np.zeros(8, dtype=np.float32)
        actions = [agent.act(obs) for _ in range(200)]
        for a in actions:
            self.assertIsInstance(a, int)
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, 4)

    def test_act_explores_all_actions(self):
        seed_all(seed=0)
        agent = RandomAgent(_agent_cfg(), n_actions=4)
        obs = np.zeros(8, dtype=np.float32)
        counts = Counter(agent.act(obs) for _ in range(2000))
        self.assertEqual(set(counts.keys()), {0, 1, 2, 3})
        for action, count in counts.items():
            self.assertGreater(count, 200, f"action {action} sampled {count} times out of 2000 — distribution looks broken")

    def test_reproducible_with_seed_all(self):
        seed_all(seed=42)
        agent_a = RandomAgent(_agent_cfg(), n_actions=4)
        seq_a = [agent_a.act(np.zeros(8, dtype=np.float32)) for _ in range(50)]

        seed_all(seed=42)
        agent_b = RandomAgent(_agent_cfg(), n_actions=4)
        seq_b = [agent_b.act(np.zeros(8, dtype=np.float32)) for _ in range(50)]

        self.assertEqual(seq_a, seq_b)

    def test_different_seeds_differ(self):
        seed_all(seed=0)
        seq_a = [RandomAgent(_agent_cfg(), n_actions=4).act(np.zeros(8, dtype=np.float32)) for _ in range(50)]
        seed_all(seed=1)
        seq_b = [RandomAgent(_agent_cfg(), n_actions=4).act(np.zeros(8, dtype=np.float32)) for _ in range(50)]
        self.assertNotEqual(seq_a, seq_b)

    def test_observe_and_learn_are_safe(self):
        seed_all(seed=0)
        agent = RandomAgent(_agent_cfg(), n_actions=4)
        t = Transition(
            obs=np.zeros(8, dtype=np.float32),
            action=0,
            reward=1.0,
            next_obs=np.zeros(8, dtype=np.float32),
            terminated=False,
            truncated=False,
        )
        agent.observe(t)
        self.assertEqual(agent.learn(), {})

    def test_save_load_roundtrip(self):
        seed_all(seed=0)
        agent = RandomAgent(_agent_cfg(), n_actions=4)

        obs = np.zeros(8, dtype=np.float32)
        for _ in range(17):
            agent.act(obs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "agent.pt"
            agent.save(path)
            self.assertTrue(path.exists())

            next_actions_before = [agent.act(obs) for _ in range(30)]

            seed_all(seed=999)
            restored = RandomAgent(_agent_cfg(), n_actions=4)
            restored.load(path)
            next_actions_after = [restored.act(obs) for _ in range(30)]

        self.assertEqual(next_actions_before, next_actions_after)

    def test_load_rejects_n_actions_mismatch(self):
        seed_all(seed=0)
        agent = RandomAgent(_agent_cfg(), n_actions=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.pt"
            agent.save(path)

            other = RandomAgent(_agent_cfg(), n_actions=2)
            with self.assertRaises(ValueError):
                other.load(path)

    def test_invalid_n_actions_raises(self):
        with self.assertRaises(ValueError):
            RandomAgent(_agent_cfg(), n_actions=0)


if __name__ == "__main__":
    unittest.main()
