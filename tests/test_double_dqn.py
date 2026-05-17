import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.dqn import DQNAgent
from src.agents.double_dqn import DoubleDQNAgent
from src.agents.registry import AGENTS
from src.utils.config import (
    AgentConfig,
    EpsilonConfig,
    NetworkConfig,
    TargetUpdateConfig,
)
from src.agents.base import Transition


def _cfg() -> AgentConfig:
    return AgentConfig(
        name="double_dqn",
        network=NetworkConfig(hidden_sizes=(16,), activation="relu"),
        gamma=0.99, lr=5e-4, batch_size=8, buffer_size=64, min_buffer_size=0,
        target_update=TargetUpdateConfig(kind="hard", every_steps=1000, tau=0.0),
        epsilon=EpsilonConfig(start=1.0, end=0.05, decay_steps=100),
        grad_clip=10.0,
    )


class TestDoubleDQN(unittest.TestCase):
    def test_registered_and_subclass(self):
        self.assertIn("double_dqn", AGENTS)
        self.assertTrue(issubclass(AGENTS["double_dqn"], DQNAgent))

    def test_target_rule_is_decoupled_selection(self):
        ag = DoubleDQNAgent(_cfg(), n_actions=4)
        # Make target != online so the two rules genuinely differ.
        with torch.no_grad():
            for p in ag.target_net.parameters():
                p.add_(0.7)
        x = torch.randn(11, 8)
        got = ag._next_state_values(x)
        # Reference Double DQN formula computed independently.
        with torch.no_grad():
            a = ag.q_net(x).argmax(dim=1, keepdim=True)
            ref = ag.target_net(x).gather(1, a).squeeze(1)
            vanilla = ag.target_net(x).max(dim=1).values
        self.assertEqual(tuple(got.shape), (11,))
        self.assertTrue(torch.allclose(got, ref))
        # Double DQN evaluation is <= the vanilla max (reduces overestimation).
        self.assertTrue(bool((got <= vanilla + 1e-6).all()))
        self.assertTrue(bool((got - vanilla).abs().max() > 1e-6))

    def test_dqn_hook_unchanged_is_plain_max(self):
        dq = DQNAgent(_cfg(), n_actions=4)
        x = torch.randn(5, 8)
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(dq._next_state_values(x),
                               dq.target_net(x).max(dim=1).values)
            )

    def test_learn_runs_and_returns_metrics(self):
        ag = DoubleDQNAgent(_cfg(), n_actions=4)
        rng = np.random.default_rng(0)
        for _ in range(64):
            ag.observe(Transition(
                obs=rng.standard_normal(8).astype(np.float32),
                action=int(rng.integers(0, 4)),
                reward=float(rng.standard_normal()),
                next_obs=rng.standard_normal(8).astype(np.float32),
                terminated=bool(rng.integers(0, 2)),
                truncated=False,
            ))
        m = ag.learn()
        self.assertIn("loss", m)
        self.assertTrue(np.isfinite(m["loss"]))


if __name__ == "__main__":
    unittest.main()
