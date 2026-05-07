import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import Agent, Transition
from src.training.evaluator import (
    SOLVED_MIN_EPISODES,
    SOLVED_THRESHOLD,
    evaluate,
)
from src.utils.termination import TerminationReason


class _FakeEnv:
    """One-step episode env where each step ends with a scripted (reward, terminated, truncated, info)."""

    def __init__(self, scripted: list[tuple[float, bool, bool, dict]]) -> None:
        self.action_space = MagicMock()
        self.observation_space = MagicMock()
        self._scripted = scripted
        self._idx = 0
        self._step_in_ep = 0

    def reset(self, seed: int | None = None):
        self._step_in_ep = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self._step_in_ep += 1
        reward, terminated, truncated, info = self._scripted[self._idx]
        self._idx += 1
        return np.zeros(8, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self):
        pass


class _DummyAgent(Agent):
    def __init__(self):
        self.greedy_calls: list[bool] = []

    def act(self, obs, *, greedy: bool = False):
        self.greedy_calls.append(greedy)
        return 0

    def observe(self, transition: Transition) -> None:
        pass

    def learn(self) -> dict:
        return {}

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


def _scripted(returns: list[float], reason: str, terminated: bool = True, truncated: bool = False):
    return [(r, terminated, truncated, {"termination_reason": reason}) for r in returns]


class TestEvaluateSchema(unittest.TestCase):
    def test_returns_expected_keys(self):
        env = _FakeEnv(_scripted([10.0, -50.0, 100.0], "land"))
        agent = _DummyAgent()
        out = evaluate(agent, env, episodes=3, greedy=True)
        for key in (
            "episodes", "greedy", "mean_return", "std_return", "ci95_return",
            "min_return", "max_return", "median_return", "mean_length",
            "termination_breakdown", "solved", "returns", "lengths", "videos",
        ):
            self.assertIn(key, out, f"missing key '{key}'")

    def test_episode_count(self):
        env = _FakeEnv(_scripted([1.0, 2.0, 3.0, 4.0, 5.0], "land"))
        out = evaluate(_DummyAgent(), env, episodes=5, greedy=True)
        self.assertEqual(out["episodes"], 5)
        self.assertEqual(len(out["returns"]), 5)
        self.assertEqual(len(out["lengths"]), 5)


class TestStatistics(unittest.TestCase):
    def test_mean_and_std(self):
        env = _FakeEnv(_scripted([0.0, 100.0, 200.0, 300.0], "land"))
        out = evaluate(_DummyAgent(), env, episodes=4, greedy=True)
        self.assertAlmostEqual(out["mean_return"], 150.0)
        self.assertGreater(out["std_return"], 0.0)
        self.assertEqual(out["min_return"], 0.0)
        self.assertEqual(out["max_return"], 300.0)

    def test_ci95_brackets_mean(self):
        env = _FakeEnv(_scripted([10.0, 20.0, 30.0, 40.0, 50.0], "land"))
        out = evaluate(_DummyAgent(), env, episodes=5, greedy=True)
        lo, hi = out["ci95_return"]
        self.assertLess(lo, out["mean_return"])
        self.assertGreater(hi, out["mean_return"])

    def test_single_episode_zero_std(self):
        env = _FakeEnv(_scripted([42.0], "land"))
        out = evaluate(_DummyAgent(), env, episodes=1, greedy=True)
        self.assertEqual(out["std_return"], 0.0)
        self.assertEqual(out["ci95_return"][0], out["ci95_return"][1])


class TestTerminationBreakdown(unittest.TestCase):
    def test_breakdown_sums_to_one(self):
        env = _FakeEnv(_scripted([1.0, 2.0, 3.0, 4.0], "crash"))
        out = evaluate(_DummyAgent(), env, episodes=4, greedy=True)
        self.assertAlmostEqual(sum(out["termination_breakdown"].values()), 1.0)

    def test_breakdown_proportions(self):
        scripted = [
            (100.0, True, False, {"termination_reason": "land"}),
            (-100.0, True, False, {"termination_reason": "crash"}),
            (-100.0, True, False, {"termination_reason": "crash"}),
            (0.0, False, True, {"termination_reason": "truncated"}),
        ]
        env = _FakeEnv(scripted)
        out = evaluate(_DummyAgent(), env, episodes=4, greedy=True)
        b = out["termination_breakdown"]
        self.assertAlmostEqual(b["land"], 0.25)
        self.assertAlmostEqual(b["crash"], 0.50)
        self.assertAlmostEqual(b["truncated"], 0.25)
        self.assertAlmostEqual(b["sleep"], 0.0)
        self.assertAlmostEqual(b["out_of_view"], 0.0)
        self.assertAlmostEqual(b["unknown"], 0.0)

    def test_breakdown_contains_all_reasons(self):
        env = _FakeEnv(_scripted([1.0], "land"))
        out = evaluate(_DummyAgent(), env, episodes=1, greedy=True)
        self.assertEqual(
            set(out["termination_breakdown"].keys()),
            {r.value for r in TerminationReason},
        )


class TestSolvedFlag(unittest.TestCase):
    def test_solved_false_when_episodes_below_threshold(self):
        env = _FakeEnv(_scripted([300.0] * 50, "land"))
        out = evaluate(_DummyAgent(), env, episodes=50, greedy=True)
        self.assertFalse(out["solved"])

    def test_solved_true_when_mean_above_threshold_and_episodes_enough(self):
        n = SOLVED_MIN_EPISODES
        env = _FakeEnv(_scripted([SOLVED_THRESHOLD + 10.0] * n, "land"))
        out = evaluate(_DummyAgent(), env, episodes=n, greedy=True)
        self.assertTrue(out["solved"])

    def test_solved_false_when_mean_below_threshold(self):
        n = SOLVED_MIN_EPISODES
        env = _FakeEnv(_scripted([SOLVED_THRESHOLD - 10.0] * n, "land"))
        out = evaluate(_DummyAgent(), env, episodes=n, greedy=True)
        self.assertFalse(out["solved"])


class TestGreedyPropagation(unittest.TestCase):
    def test_greedy_true_passed_to_agent(self):
        env = _FakeEnv(_scripted([1.0, 2.0], "land"))
        agent = _DummyAgent()
        evaluate(agent, env, episodes=2, greedy=True)
        self.assertTrue(all(agent.greedy_calls))

    def test_greedy_false_passed_to_agent(self):
        env = _FakeEnv(_scripted([1.0, 2.0], "land"))
        agent = _DummyAgent()
        evaluate(agent, env, episodes=2, greedy=False)
        self.assertTrue(all(c is False for c in agent.greedy_calls))


class TestErrors(unittest.TestCase):
    def test_zero_episodes_raises(self):
        with self.assertRaises(ValueError):
            evaluate(_DummyAgent(), _FakeEnv([]), episodes=0, greedy=True)

    def test_negative_episodes_raises(self):
        with self.assertRaises(ValueError):
            evaluate(_DummyAgent(), _FakeEnv([]), episodes=-1, greedy=True)

    def test_record_video_with_zero_video_every_raises(self):
        with self.assertRaises(ValueError):
            evaluate(
                _DummyAgent(), _FakeEnv([]),
                episodes=1, greedy=True,
                record_video=True, video_every=0,
            )


if __name__ == "__main__":
    unittest.main()
