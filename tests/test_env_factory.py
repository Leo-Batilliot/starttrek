import sys
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.factory import build_env
from src.env.wrappers import ObsNormalize, RewardClip, WRAPPERS
from src.utils.config import EnvConfig, WrapperConfig


def make_cfg(*wrappers: WrapperConfig, env_id: str = "LunarLander-v3") -> EnvConfig:
    return EnvConfig(id=env_id, kwargs={}, wrappers=tuple(wrappers))


class TestBuildEnv(unittest.TestCase):
    def test_obs_and_action_space_are_lunarlander(self):
        env = build_env(make_cfg(), seed=0)
        try:
            self.assertEqual(env.observation_space.shape, (8,))
            self.assertEqual(env.action_space.n, 4)
        finally:
            env.close()

    def test_unwrapped_is_lunar_lander(self):
        env = build_env(make_cfg(), seed=0)
        try:
            self.assertEqual(type(env.unwrapped).__name__, "LunarLander")
        finally:
            env.close()

    def test_no_wrappers_means_no_extra_layers(self):
        env = build_env(make_cfg(), seed=0)
        try:
            chain = []
            cur = env
            while hasattr(cur, "env") and cur is not cur.unwrapped:
                chain.append(type(cur).__name__)
                cur = cur.env
            self.assertNotIn("ObsNormalize", chain)
            self.assertNotIn("RewardClip", chain)
            self.assertNotIn("RecordEpisodeStatistics", chain)
        finally:
            env.close()

    def test_wrappers_applied_in_declared_order(self):
        cfg = make_cfg(
            WrapperConfig(name="RewardClip", kwargs={"low": -1.0, "high": 1.0}),
            WrapperConfig(name="ObsNormalize", kwargs={}),
        )
        env = build_env(cfg, seed=0)
        try:
            self.assertIsInstance(env, ObsNormalize)
            self.assertIsInstance(env.env, RewardClip)
        finally:
            env.close()

    def test_record_episode_stats_alias_resolves(self):
        cfg = make_cfg(WrapperConfig(name="RecordEpisodeStats", kwargs={}))
        env = build_env(cfg, seed=0)
        try:
            self.assertIsInstance(env, gym.wrappers.RecordEpisodeStatistics)
        finally:
            env.close()

    def test_unknown_wrapper_raises_value_error(self):
        cfg = make_cfg(WrapperConfig(name="DoesNotExist", kwargs={}))
        with self.assertRaises(ValueError) as ctx:
            build_env(cfg, seed=0)
        self.assertIn("DoesNotExist", str(ctx.exception))

    def test_seed_makes_first_step_reproducible(self):
        cfg = make_cfg()
        env_a = build_env(cfg, seed=42)
        env_b = build_env(cfg, seed=42)
        try:
            obs_a, _ = env_a.reset(seed=42)
            obs_b, _ = env_b.reset(seed=42)
            np.testing.assert_allclose(obs_a, obs_b, rtol=1e-5, atol=1e-6)
            env_a.action_space.seed(42)
            env_b.action_space.seed(42)
            for _ in range(20):
                a_a = env_a.action_space.sample()
                a_b = env_b.action_space.sample()
                self.assertEqual(a_a, a_b)
        finally:
            env_a.close()
            env_b.close()

    def test_render_mode_propagates(self):
        env = build_env(make_cfg(), seed=0, render_mode="rgb_array")
        try:
            self.assertEqual(env.unwrapped.render_mode, "rgb_array")
            frame = env.render()
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.ndim, 3)
        finally:
            env.close()


class TestRewardClip(unittest.TestCase):
    def test_clip_above(self):
        env = RewardClip(gym.make("LunarLander-v3"), low=-1.0, high=1.0)
        clipped = env.reward(5.0)
        self.assertEqual(clipped, 1.0)
        env.close()

    def test_clip_below(self):
        env = RewardClip(gym.make("LunarLander-v3"), low=-1.0, high=1.0)
        clipped = env.reward(-7.5)
        self.assertEqual(clipped, -1.0)
        env.close()

    def test_clip_passthrough_in_range(self):
        env = RewardClip(gym.make("LunarLander-v3"), low=-1.0, high=1.0)
        self.assertEqual(env.reward(0.3), 0.3)
        env.close()

    def test_clip_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            RewardClip(gym.make("LunarLander-v3"), low=2.0, high=1.0)


class TestObsNormalize(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        base = gym.make("LunarLander-v3")
        env = ObsNormalize(base)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (8,))
        self.assertEqual(obs.dtype, np.float32)
        env.close()

    def test_running_stats_progress(self):
        base = gym.make("LunarLander-v3")
        env = ObsNormalize(base)
        env.reset(seed=0)
        env.action_space.seed(0)
        for _ in range(50):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
        self.assertGreater(env._count, 1)
        env.close()


class TestRegistry(unittest.TestCase):
    def test_required_wrappers_registered(self):
        for name in ("RecordEpisodeStats", "ObsNormalize", "RewardClip"):
            self.assertIn(name, WRAPPERS)


if __name__ == "__main__":
    unittest.main()
