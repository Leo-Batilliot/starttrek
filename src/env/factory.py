from __future__ import annotations

import gymnasium as gym

from src.env.wrappers import WRAPPERS
from src.utils.config import EnvConfig


def build_env(cfg: EnvConfig, *, seed: int, render_mode: str | None = None) -> gym.Env:
    env: gym.Env = gym.make(cfg.id, render_mode=render_mode, **cfg.kwargs)

    for i, wrapper_cfg in enumerate(cfg.wrappers):
        if wrapper_cfg.name not in WRAPPERS:
            raise ValueError(
                f"Unknown wrapper '{wrapper_cfg.name}' at wrappers[{i}]. "
                f"Available: {sorted(WRAPPERS)}"
            )
        env = WRAPPERS[wrapper_cfg.name](env, **wrapper_cfg.kwargs)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
