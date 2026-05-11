from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agents.base import Agent, Transition
from src.agents.registry import register_agent
from src.utils.config import AgentConfig


@register_agent("noop")
class NoOpAgent(Agent):
    def __init__(self, cfg: AgentConfig, *, n_actions: int) -> None:
        if n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {n_actions}")
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.epsilon: float = 0.0

    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        return 0

    def observe(self, transition: Transition) -> None:
        return None

    def learn(self) -> dict[str, float]:
        return {}

    def save(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    def load(self, path: Path) -> None:
        return None
