import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import Agent, Transition
from src.agents.dqn import DQNAgent
from src.agents.registry import AGENTS
from src.utils.config import (
    AgentConfig,
    EpsilonConfig,
    NetworkConfig,
    TargetUpdateConfig,
)
from src.utils.seeding import seed_all


def _cfg(
    *,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 100,
    hidden: tuple[int, ...] = (16,),
    buffer: int = 64,
) -> AgentConfig:
    return AgentConfig(
        name="dqn",
        network=NetworkConfig(hidden_sizes=hidden, activation="relu"),
        gamma=0.99,
        lr=5e-4,
        batch_size=8,
        buffer_size=buffer,
        min_buffer_size=0,
        target_update=TargetUpdateConfig(kind="hard", every_steps=1000, tau=0.0),
        epsilon=EpsilonConfig(start=eps_start, end=eps_end, decay_steps=eps_decay),
        grad_clip=10.0,
    )


def _obs(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(8).astype(np.float32)


def _transition() -> Transition:
    return Transition(
        obs=np.zeros(8, dtype=np.float32),
        action=1,
        reward=0.5,
        next_obs=np.ones(8, dtype=np.float32),
        terminated=False,
        truncated=False,
    )


class TestDQNAgentSkeleton(unittest.TestCase):
    def test_registered_under_dqn(self):
        self.assertIn("dqn", AGENTS)
        self.assertIs(AGENTS["dqn"], DQNAgent)

    def test_is_an_agent(self):
        seed_all(seed=0)
        self.assertIsInstance(DQNAgent(_cfg(), n_actions=4), Agent)

    def test_invalid_n_actions_raises(self):
        seed_all(seed=0)
        with self.assertRaises(ValueError):
            DQNAgent(_cfg(), n_actions=0)

    def test_invalid_obs_dim_raises(self):
        seed_all(seed=0)
        with self.assertRaises(ValueError):
            DQNAgent(_cfg(), n_actions=4, obs_dim=0)

    def test_builds_frozen_target_copy(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(), n_actions=4)
        for q_p, t_p in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            self.assertTrue(torch.equal(q_p, t_p))
            self.assertFalse(t_p.requires_grad)

    def test_act_greedy_returns_argmax_of_q(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(), n_actions=4)
        rng = np.random.default_rng(123)
        for _ in range(50):
            obs = _obs(rng)
            with torch.no_grad():
                qv = agent.q_net(torch.from_numpy(obs).unsqueeze(0)).squeeze(0)
            expected = int(torch.argmax(qv).item())
            self.assertEqual(agent.act(obs, greedy=True), expected)

    def test_act_greedy_returns_int_in_range(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(), n_actions=4)
        rng = np.random.default_rng(1)
        for _ in range(100):
            a = agent.act(_obs(rng), greedy=True)
            self.assertIsInstance(a, int)
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, 4)

    def test_act_greedy_is_deterministic(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(), n_actions=4)
        obs = np.full(8, 0.3, dtype=np.float32)
        actions = {agent.act(obs, greedy=True) for _ in range(20)}
        self.assertEqual(len(actions), 1)

    def test_epsilon_zero_acts_greedy(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(eps_start=0.0, eps_end=0.0, eps_decay=1), n_actions=4)
        rng = np.random.default_rng(7)
        for _ in range(100):
            obs = _obs(rng)
            self.assertEqual(agent.act(obs, greedy=False), agent.act(obs, greedy=True))

    def test_epsilon_one_explores_all_actions(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(eps_start=1.0, eps_end=1.0, eps_decay=1), n_actions=4)
        obs = np.zeros(8, dtype=np.float32)
        counts = Counter(agent.act(obs, greedy=False) for _ in range(2000))
        self.assertEqual(set(counts), {0, 1, 2, 3})
        for action, c in counts.items():
            self.assertGreater(c, 200, f"action {action}: {c}/2000 — exploration looks broken")

    def test_act_rejects_wrong_shape(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(), n_actions=4)
        with self.assertRaises(ValueError):
            agent.act(np.zeros(7, dtype=np.float32))

    def test_observe_pushes_to_buffer(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(), n_actions=4)
        self.assertEqual(len(agent.buffer), 0)
        for i in range(5):
            agent.observe(_transition())
            self.assertEqual(len(agent.buffer), i + 1)

    def test_observe_decays_epsilon_per_schedule(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(eps_start=1.0, eps_end=0.0, eps_decay=10), n_actions=4)
        self.assertEqual(agent.epsilon, 1.0)
        for _ in range(5):
            agent.observe(_transition())
        self.assertAlmostEqual(agent.epsilon, 0.5)
        for _ in range(100):
            agent.observe(_transition())
        self.assertEqual(agent.epsilon, 0.0)

    def test_exposes_initial_epsilon(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(eps_start=0.9), n_actions=4)
        self.assertEqual(agent.epsilon, 0.9)

    def test_learn_is_safe_and_empty(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(), n_actions=4)
        agent.observe(_transition())
        self.assertEqual(agent.learn(), {})

    def test_save_load_roundtrip(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(eps_start=1.0, eps_end=0.0, eps_decay=10), n_actions=4)
        for _ in range(7):
            agent.observe(_transition())

        rng = np.random.default_rng(42)
        probe = [_obs(rng) for _ in range(10)]
        before = [agent.act(o, greedy=True) for o in probe]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "dqn.pt"
            agent.save(path)
            self.assertTrue(path.exists())

            seed_all(seed=999)
            restored = DQNAgent(_cfg(eps_start=1.0, eps_end=0.0, eps_decay=10), n_actions=4)
            restored.load(path)

        after = [restored.act(o, greedy=True) for o in probe]
        self.assertEqual(before, after)
        self.assertEqual(restored._step_count, 7)
        self.assertAlmostEqual(restored.epsilon, 0.3)

    def test_load_rejects_n_actions_mismatch(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(), n_actions=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dqn.pt"
            agent.save(path)
            other = DQNAgent(_cfg(), n_actions=2)
            with self.assertRaises(ValueError):
                other.load(path)

    def test_load_rejects_obs_dim_mismatch(self):
        seed_all(seed=0)
        agent = DQNAgent(_cfg(), n_actions=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dqn.pt"
            agent.save(path)
            other = DQNAgent(_cfg(), n_actions=4, obs_dim=6)
            with self.assertRaises(ValueError):
                other.load(path)


if __name__ == "__main__":
    unittest.main()
