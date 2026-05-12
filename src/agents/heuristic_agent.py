from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agents.base import Agent, Transition
from src.agents.registry import register_agent
from src.utils.config import AgentConfig


ANGLE_TARG_GAIN_X = 0.5
ANGLE_TARG_GAIN_VX = 1.0
ANGLE_TARG_CLIP = 0.4
HOVER_TARG_GAIN = 0.55
ANGLE_KP = 0.5
ANGLE_KD = 1.0
HOVER_KP = 0.5
HOVER_KD = 0.5
HOVER_TRIGGER = 0.05
ANGLE_TRIGGER = 0.05


@register_agent("heuristic")
class HeuristicAgent(Agent):
    def __init__(self, cfg: AgentConfig, *, n_actions: int) -> None:
        if n_actions != 4:
            raise ValueError(f"HeuristicAgent expects discrete LunarLander with n_actions=4, got {n_actions}")
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.epsilon: float = 0.0

    @staticmethod
    def _heuristic(obs: np.ndarray) -> int:
        x, _y, v_x, v_y, theta, theta_dot, leg1, leg2 = (float(v) for v in obs)

        angle_targ = x * ANGLE_TARG_GAIN_X + v_x * ANGLE_TARG_GAIN_VX
        if angle_targ > ANGLE_TARG_CLIP:
            angle_targ = ANGLE_TARG_CLIP
        elif angle_targ < -ANGLE_TARG_CLIP:
            angle_targ = -ANGLE_TARG_CLIP
        hover_targ = HOVER_TARG_GAIN * abs(x)

        angle_todo = (angle_targ - theta) * ANGLE_KP - theta_dot * ANGLE_KD
        hover_todo = (hover_targ - obs[1]) * HOVER_KP - v_y * HOVER_KD

        if leg1 or leg2:
            angle_todo = 0.0
            hover_todo = -v_y * HOVER_KD

        if hover_todo > abs(angle_todo) and hover_todo > HOVER_TRIGGER:
            return 2
        if angle_todo < -ANGLE_TRIGGER:
            return 3
        if angle_todo > ANGLE_TRIGGER:
            return 1
        return 0

    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        if obs.shape != (8,):
            raise ValueError(f"HeuristicAgent expects obs shape (8,), got {obs.shape}")
        return self._heuristic(obs)

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
