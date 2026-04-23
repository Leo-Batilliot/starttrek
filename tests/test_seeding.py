import random
import unittest

import numpy as np
import torch
import gymnasium as gym

from src.utils.seeding import seed_all


class TestSeeding(unittest.TestCase):
    def test_python_random_is_deterministic(self):
        seed_all(seed=42)
        a = [random.random() for _ in range(10)]
        seed_all(seed=42)
        b = [random.random() for _ in range(10)]
        self.assertEqual(a, b)

    def test_numpy_is_deterministic(self):
        seed_all(seed=42)
        a = np.random.rand(10).tolist()
        seed_all(seed=42)
        b = np.random.rand(10).tolist()
        self.assertEqual(a, b)

    def test_torch_cpu_is_deterministic(self):
        seed_all(seed=42)
        a = torch.rand(10).tolist()
        seed_all(seed=42)
        b = torch.rand(10).tolist()
        self.assertEqual(a, b)

    def test_different_seeds_give_different_rng_outputs(self):
        seed_all(seed=0)
        a = np.random.rand(5).tolist()
        seed_all(seed=1)
        b = np.random.rand(5).tolist()
        self.assertNotEqual(a, b)

    @staticmethod
    def _rollout(seed: int, n_steps: int = 30) -> list[tuple]:
        seed_all(seed=seed)

        env = gym.make("LunarLander-v3")
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)

        trace = []
        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            trace.append((
                obs.copy(),
                int(action),
                float(reward),
                bool(terminated),
                bool(truncated),
            ))
            if terminated or truncated:
                break
        env.close()
        return trace

    def test_env_rollout_is_reproducible(self):
        trace_a = self._rollout(seed=42, n_steps=30)
        trace_b = self._rollout(seed=42, n_steps=30)

        self.assertEqual(len(trace_a), len(trace_b),"episode terminations diverged between runs")

        for i, ((oa, aa, ra, ta, tra), (ob, ab, rb, tb, trb)) in enumerate(zip(trace_a, trace_b)):
            self.assertEqual(aa, ab, f"action mismatch at step {i}")
            np.testing.assert_allclose(oa, ob, rtol=1e-4, atol=1e-5, err_msg=f"obs diverged at step {i}")
            self.assertAlmostEqual(ra, rb, places=4, msg=f"reward mismatch at step {i}")
            self.assertEqual(ta, tb, msg=f"terminated mismatch at step {i}")
            self.assertEqual(tra, trb, msg=f"truncated mismatch at step {i}")

    def test_different_seeds_produce_different_traces(self):
        trace_a = self._rollout(seed=0, n_steps=20)
        trace_b = self._rollout(seed=1, n_steps=20)
        actions_a = [t[1] for t in trace_a]
        actions_b = [t[1] for t in trace_b]
        self.assertNotEqual(actions_a, actions_b)

    def test_negative_seed_raises(self):
        with self.assertRaises(ValueError):
            seed_all(seed=-1)

    def test_non_int_seed_raises(self):
        with self.assertRaises(ValueError):
            seed_all(seed=3.14)


if __name__ == "__main__":
    unittest.main()