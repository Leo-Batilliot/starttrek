from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool

ActionType = Union[int, np.ndarray]

class Agent(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray, *, greedy: bool = False) -> ActionType:
        """"""

    @abstractmethod
    def observe(self, transition: Transition) -> None:
        """"""

    @abstractmethod
    def learn(self) -> dict[str, float]:
        """"""

    @abstractmethod
    def save(self, path: Path) -> None:
        """"""

    @abstractmethod
    def load(self, path: Path) -> None:
        """"""