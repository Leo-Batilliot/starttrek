from __future__ import annotations

import numpy as np
import torch

from src.agents.base import Transition


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_dtype: np.dtype = np.int64,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}.")
        if not obs_shape:
            raise ValueError("obs_shape must be a non-empty tuple.")

        # Init
        self._capacity: int = int(capacity)
        self._obs_shape: tuple[int, ...] = tuple(obs_shape)
        self._action_dtype: np.dtype = np.dtype(action_dtype)

        self._obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.empty(capacity, dtype=self._action_dtype)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._dones = np.empty(capacity, dtype=np.float32)

        self._idx: int = 0
        self._size: int = 0

    def push(self, transition: Transition) -> None:
        i = self._idx
        self._obs[i] = transition.obs
        self._next_obs[i] = transition.next_obs
        self._actions[i] = transition.action
        self._rewards[i] = transition.reward
        self._dones[i] = float(transition.terminated)

        self._idx = (self._idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {self._size}."
            )

        idx = np.random.randint(0, self._size, size=batch_size)

        return {
            "obs": torch.from_numpy(self._obs[idx]),
            "action": torch.from_numpy(self._actions[idx]),
            "reward": torch.from_numpy(self._rewards[idx]),
            "next_obs": torch.from_numpy(self._next_obs[idx]),
            "done": torch.from_numpy(self._dones[idx]),
        }

    def __len__(self) -> int:
        return self._size
