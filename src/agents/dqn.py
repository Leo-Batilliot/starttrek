from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.agents.base import Agent, Transition
from src.agents.registry import register_agent
from src.memory.replay_buffer import ReplayBuffer
from src.networks.mlp import build_mlp
from src.training.schedules import linear_epsilon
from src.utils.config import AgentConfig


LUNARLANDER_OBS_DIM = 8


@register_agent("dqn")
class DQNAgent(Agent):
    def __init__(
        self,
        cfg: AgentConfig,
        *,
        n_actions: int,
        obs_dim: int = LUNARLANDER_OBS_DIM,
    ) -> None:
        if n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {n_actions}")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {obs_dim}")

        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.obs_dim = int(obs_dim)

        self.device = torch.device("cpu")

        hidden = list(cfg.network.hidden_sizes)
        self.q_net = build_mlp(
            self.obs_dim, self.n_actions, hidden, cfg.network.activation
        ).to(self.device)
        self.target_net = build_mlp(
            self.obs_dim, self.n_actions, hidden, cfg.network.activation
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.buffer = ReplayBuffer(
            capacity=cfg.buffer_size, obs_shape=(self.obs_dim,)
        )

        self._rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))

        self._step_count = 0
        self.epsilon: float = self._epsilon_at(self._step_count)

    def _epsilon_at(self, step: int) -> float:
        eps = self.cfg.epsilon
        return linear_epsilon(
            step, start=eps.start, end=eps.end, decay_steps=eps.decay_steps
        )

    def _q_values(self, obs: np.ndarray) -> torch.Tensor:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.shape != (self.obs_dim,):
            raise ValueError(
                f"DQNAgent expects obs shape ({self.obs_dim},), got {arr.shape}"
            )
        with torch.no_grad():
            t = torch.from_numpy(arr).to(self.device).unsqueeze(0)  # (1, obs_dim)
            return self.q_net(t).squeeze(0)  # (n_actions,)

    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        q = self._q_values(obs)
        greedy_action = int(torch.argmax(q).item())
        if greedy:
            return greedy_action
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return greedy_action

    def observe(self, transition: Transition) -> None:
        self.buffer.push(transition)
        self._step_count += 1
        self.epsilon = self._epsilon_at(self._step_count)

    def learn(self) -> dict[str, float]:
        return {}

    def save(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "step_count": self._step_count,
                "n_actions": self.n_actions,
                "obs_dim": self.obs_dim,
            },
            p,
        )

    def load(self, path: Path) -> None:
        state = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        if state.get("n_actions") != self.n_actions:
            raise ValueError(
                f"checkpoint n_actions={state.get('n_actions')} does not match "
                f"agent n_actions={self.n_actions}"
            )
        if state.get("obs_dim") != self.obs_dim:
            raise ValueError(
                f"checkpoint obs_dim={state.get('obs_dim')} does not match "
                f"agent obs_dim={self.obs_dim}"
            )
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self._step_count = int(state.get("step_count", 0))
        self.epsilon = self._epsilon_at(self._step_count)
