from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class RewardClip(gym.RewardWrapper):
    def __init__(self, env: gym.Env, *, low: float = -1.0, high: float = 1.0) -> None:
        if low > high:
            raise ValueError(f"RewardClip: low ({low}) must be <= high ({high}).")
        super().__init__(env)
        self._low: float = float(low)
        self._high: float = float(high)

    def reward(self, reward: SupportsFloat) -> float:
        return float(np.clip(float(reward), self._low, self._high))


class ObsNormalize(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, *, eps: float = 1e-8) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                f"ObsNormalize requires a Box observation space, got "
                f"{type(env.observation_space).__name__}."
            )
        shape = env.observation_space.shape
        self._eps: float = float(eps)
        self._count: int = 0
        self._mean: np.ndarray = np.zeros(shape, dtype=np.float64)
        self._m2: np.ndarray = np.zeros(shape, dtype=np.float64)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (obs - self._mean)
        var = self._m2 / self._count if self._count > 1 else np.ones_like(self._m2)
        std = np.sqrt(var) + self._eps
        return ((obs - self._mean) / std).astype(np.float32)


WRAPPERS: dict[str, type[gym.Wrapper]] = {
    "RecordEpisodeStats": gym.wrappers.RecordEpisodeStatistics,
    "ObsNormalize": ObsNormalize,
    "RewardClip": RewardClip,
}
