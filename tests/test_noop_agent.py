import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import Agent, Transition
from src.agents.noop_agent import NoOpAgent
from src.agents.registry import AGENTS
from src.utils.config import (
    AgentConfig,
    EpsilonConfig,
    NetworkConfig,
    TargetUpdateConfig,
)


def _agent_cfg() -> AgentConfig:
    return AgentConfig(
        name="noop",
        network=NetworkConfig(hidden_sizes=(8,), activation="relu"),
        gamma=0.99,
        lr=1e-3,
        batch_size=1,
        buffer_size=1,
        min_buffer_size=0,
        target_update=TargetUpdateConfig(kind="hard", every_steps=1, tau=0.0),
        epsilon=EpsilonConfig(start=0.0, end=0.0, decay_steps=1),
        grad_clip=1.0,
    )


class TestNoOpAgent(unittest.TestCase):
    def test_registered_under_noop(self):
        self.assertIn("noop", AGENTS)
        self.assertIs(AGENTS["noop"], NoOpAgent)

    def test_is_an_agent(self):
        agent = NoOpAgent(_agent_cfg(), n_actions=4)
        self.assertIsInstance(agent, Agent)

    def test_act_returns_zero(self):
        agent = NoOpAgent(_agent_cfg(), n_actions=4)
        obs = np.zeros(8, dtype=np.float32)
        self.assertEqual(agent.act(obs), 0)
        self.assertEqual(agent.act(obs, greedy=True), 0)

    def test_observe_and_learn_are_safe(self):
        agent = NoOpAgent(_agent_cfg(), n_actions=4)
        t = Transition(
            obs=np.zeros(8, dtype=np.float32),
            action=0,
            reward=1.0,
            next_obs=np.zeros(8, dtype=np.float32),
            terminated=False,
            truncated=False,
        )
        agent.observe(t)
        metrics = agent.learn()
        self.assertEqual(metrics, {})

    def test_save_creates_file_and_load_no_op(self):
        agent = NoOpAgent(_agent_cfg(), n_actions=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "ckpt.pt"
            agent.save(path)
            self.assertTrue(path.exists())
            agent.load(path)

    def test_invalid_n_actions_raises(self):
        with self.assertRaises(ValueError):
            NoOpAgent(_agent_cfg(), n_actions=0)

    def test_exposes_epsilon_attribute(self):
        agent = NoOpAgent(_agent_cfg(), n_actions=4)
        self.assertTrue(hasattr(agent, "epsilon"))


if __name__ == "__main__":
    unittest.main()
