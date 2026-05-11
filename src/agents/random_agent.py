from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.agents.base import Agent, Transition
from src.agents.registry import register_agent
from src.utils.config import AgentConfig


@register_agent("random")
class RandomAgent(Agent):
    def __init__(self, cfg: AgentConfig, *, n_actions: int) -> None:
        if n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {n_actions}")
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self._seed: int = int(np.random.randint(0, 2**31 - 1))
        self._rng: np.random.Generator = np.random.default_rng(self._seed)
        self.epsilon: float = 1.0

    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        return int(self._rng.integers(0, self.n_actions))

    def observe(self, transition: Transition) -> None:
        return None

    def learn(self) -> dict[str, float]:
        return {}

    def save(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "seed": self._seed,
            "bit_generator_state": self._rng.bit_generator.state,
            "n_actions": self.n_actions,
        }
        with p.open("wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        if state.get("n_actions") != self.n_actions:
            raise ValueError(
                f"checkpoint n_actions={state.get('n_actions')} does not match "
                f"agent n_actions={self.n_actions}"
            )
        self._seed = int(state["seed"])
        self._rng.bit_generator.state = state["bit_generator_state"]
