import unittest

import numpy as np
import torch

from src.agents.base import Transition
from src.memory.replay_buffer import ReplayBuffer


def _make_transition(value: float, terminated: bool = False, truncated: bool = False) -> Transition:
    return Transition(
        obs=np.full(8, value, dtype=np.float32),
        action=int(value) % 4,
        reward=float(value),
        next_obs=np.full(8, value + 0.5, dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
    )


class TestReplayBufferShapes(unittest.TestCase):

    def test_sample_shapes_and_dtypes(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(8,))
        for i in range(50):
            buf.push(_make_transition(float(i)))

        batch = buf.sample(batch_size=32)

        self.assertEqual(batch["obs"].shape, (32, 8))
        self.assertEqual(batch["next_obs"].shape, (32, 8))
        self.assertEqual(batch["action"].shape, (32,))
        self.assertEqual(batch["reward"].shape, (32,))
        self.assertEqual(batch["done"].shape, (32,))

        self.assertEqual(batch["obs"].dtype, torch.float32)
        self.assertEqual(batch["next_obs"].dtype, torch.float32)
        self.assertEqual(batch["action"].dtype, torch.int64)
        self.assertEqual(batch["reward"].dtype, torch.float32)
        self.assertEqual(batch["done"].dtype, torch.float32)

    def test_sample_keys(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(8,))
        for i in range(10):
            buf.push(_make_transition(float(i)))

        batch = buf.sample(batch_size=4)
        self.assertEqual(
            set(batch.keys()), {"obs", "action", "reward", "next_obs", "done"}
        )


class TestReplayBufferContent(unittest.TestCase):

    def test_no_nan_in_sample(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(8,))
        for i in range(50):
            buf.push(_make_transition(float(i)))

        batch = buf.sample(batch_size=32)
        for key, tensor in batch.items():
            self.assertFalse(
                torch.isnan(tensor.float()).any(),
                msg=f"NaN found in '{key}'",
            )

    def test_done_uses_terminated_not_truncated(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(8,))
        buf.push(_make_transition(1.0, terminated=True, truncated=False))
        buf.push(_make_transition(2.0, terminated=False, truncated=True))
        buf.push(_make_transition(3.0, terminated=False, truncated=False))

        np.random.seed(0)
        for _ in range(50):
            batch = buf.sample(batch_size=3)
            rewards = batch["reward"].numpy()
            dones = batch["done"].numpy()
            for r, d in zip(rewards, dones):
                if r == 1.0:
                    self.assertEqual(d, 1.0, "terminated=True doit donner done=1")
                else:
                    self.assertEqual(d, 0.0, f"reward={r}: truncated/normal doit donner done=0")


class TestReplayBufferLen(unittest.TestCase):

    def test_len_grows_then_caps(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(8,))
        self.assertEqual(len(buf), 0)

        for i in range(5):
            buf.push(_make_transition(float(i)))
        self.assertEqual(len(buf), 5)

        for i in range(5, 10):
            buf.push(_make_transition(float(i)))
        self.assertEqual(len(buf), 10)

        for i in range(10, 25):
            buf.push(_make_transition(float(i)))
        self.assertEqual(len(buf), 10, "len doit être plafonné à capacity")


class TestReplayBufferEviction(unittest.TestCase):

    def test_fifo_eviction_oldest_overwritten(self):
        capacity = 5
        buf = ReplayBuffer(capacity=capacity, obs_shape=(8,))

        for i in range(capacity + 3):
            buf.push(_make_transition(float(i)))

        np.random.seed(42)
        seen_rewards: set[float] = set()
        for _ in range(200):
            batch = buf.sample(batch_size=5)
            seen_rewards.update(batch["reward"].numpy().tolist())

        self.assertEqual(seen_rewards, {3.0, 4.0, 5.0, 6.0, 7.0})

    def test_data_integrity_per_slot(self):
        buf = ReplayBuffer(capacity=4, obs_shape=(8,))
        buf.push(_make_transition(7.0))

        batch = buf.sample(batch_size=1)
        self.assertEqual(batch["reward"].item(), 7.0)
        self.assertEqual(batch["action"].item(), 7 % 4)
        np.testing.assert_array_equal(
            batch["obs"][0].numpy(), np.full(8, 7.0, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            batch["next_obs"][0].numpy(), np.full(8, 7.5, dtype=np.float32)
        )


class TestReplayBufferGuards(unittest.TestCase):

    def test_invalid_capacity_raises(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=0, obs_shape=(8,))
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=-1, obs_shape=(8,))

    def test_empty_obs_shape_raises(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=10, obs_shape=())

    def test_sample_more_than_size_raises(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(8,))
        for i in range(3):
            buf.push(_make_transition(float(i)))
        with self.assertRaises(ValueError):
            buf.sample(batch_size=5)

    def test_sample_zero_or_negative_raises(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(8,))
        buf.push(_make_transition(0.0))
        with self.assertRaises(ValueError):
            buf.sample(batch_size=0)
        with self.assertRaises(ValueError):
            buf.sample(batch_size=-1)


if __name__ == "__main__":
    unittest.main()
