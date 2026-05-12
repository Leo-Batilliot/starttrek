import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import Agent, Transition
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.registry import AGENTS
from src.utils.config import (
    AgentConfig,
    EpsilonConfig,
    NetworkConfig,
    TargetUpdateConfig,
)


def _agent_cfg() -> AgentConfig:
    return AgentConfig(
        name="heuristic",
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


def _obs(x=0.0, y=1.0, vx=0.0, vy=0.0, theta=0.0, theta_dot=0.0, leg1=0.0, leg2=0.0) -> np.ndarray:
    return np.array([x, y, vx, vy, theta, theta_dot, leg1, leg2], dtype=np.float32)


class TestHeuristicAgent(unittest.TestCase):
    def test_registered_under_heuristic(self):
        self.assertIn("heuristic", AGENTS)
        self.assertIs(AGENTS["heuristic"], HeuristicAgent)

    def test_is_an_agent(self):
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        self.assertIsInstance(agent, Agent)

    def test_invalid_n_actions_raises(self):
        with self.assertRaises(ValueError):
            HeuristicAgent(_agent_cfg(), n_actions=2)
        with self.assertRaises(ValueError):
            HeuristicAgent(_agent_cfg(), n_actions=0)

    def test_act_returns_int_in_range(self):
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        rng = np.random.default_rng(0)
        for _ in range(200):
            o = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
            a = agent.act(o)
            self.assertIsInstance(a, int)
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, 4)

    def test_act_is_deterministic(self):
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        rng = np.random.default_rng(123)
        observations = [rng.uniform(-1.0, 1.0, size=8).astype(np.float32) for _ in range(100)]
        seq_a = [agent.act(o) for o in observations]
        seq_b = [agent.act(o) for o in observations]
        seq_c = [agent.act(o, greedy=True) for o in observations]
        self.assertEqual(seq_a, seq_b)
        self.assertEqual(seq_a, seq_c)

    def test_act_rejects_wrong_shape(self):
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        with self.assertRaises(ValueError):
            agent.act(np.zeros(7, dtype=np.float32))

    def test_falling_too_fast_fires_main(self):
        # À l'altitude cible (hover_targ=0 pour x=0) mais vy fortement négatif :
        # hover_todo = -vy * 0.5 = 0.5 > 0.05 → action 2 (main engine).
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        a = agent.act(_obs(x=0.0, y=0.0, vy=-1.0))
        self.assertEqual(a, 2)

    def test_drifting_right_fires_correcting_thruster(self):
        # x > 0, lander droit, pas de chute → angle_targ > 0 → angle_todo > 0 → action 1.
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        a = agent.act(_obs(x=0.5, y=1.0, vx=0.2, vy=0.0, theta=0.0, theta_dot=0.0))
        self.assertEqual(a, 1)

    def test_drifting_left_fires_correcting_thruster(self):
        # x < 0 → angle_targ < 0 → angle_todo < 0 → action 3.
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        a = agent.act(_obs(x=-0.5, y=1.0, vx=-0.2, vy=0.0, theta=0.0, theta_dot=0.0))
        self.assertEqual(a, 3)

    def test_stable_centered_does_nothing(self):
        # Centré, sans dérive, en vol stationnaire idéal → action 0.
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        a = agent.act(_obs(x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, theta_dot=0.0))
        self.assertEqual(a, 0)

    def test_legs_contact_neutralizes_angle(self):
        # Une fois posé (legs=1), on ne corrige plus l'orientation.
        # Sans vy, hover_todo == 0 et angle_todo == 0 → action 0.
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        a = agent.act(_obs(x=0.3, theta=0.3, theta_dot=0.5, leg1=1.0, leg2=1.0, vy=0.0))
        self.assertEqual(a, 0)

    def test_observe_and_learn_are_safe(self):
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        t = Transition(
            obs=_obs(),
            action=0,
            reward=0.0,
            next_obs=_obs(),
            terminated=False,
            truncated=False,
        )
        agent.observe(t)
        self.assertEqual(agent.learn(), {})

    def test_save_load_roundtrip(self):
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "agent.pt"
            agent.save(path)
            self.assertTrue(path.exists())
            agent.load(path)
            # La politique reste identique (stateless).
            a_before = agent.act(_obs(x=0.5, vx=0.2))
            a_after = agent.act(_obs(x=0.5, vx=0.2))
            self.assertEqual(a_before, a_after)

    def test_exposes_epsilon_attribute(self):
        agent = HeuristicAgent(_agent_cfg(), n_actions=4)
        self.assertTrue(hasattr(agent, "epsilon"))
        self.assertEqual(agent.epsilon, 0.0)


if __name__ == "__main__":
    unittest.main()
