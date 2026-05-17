from __future__ import annotations

import torch

from src.agents.dqn import DQNAgent
from src.agents.registry import register_agent


@register_agent("double_dqn")
class DoubleDQNAgent(DQNAgent):
    def _next_state_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
        return self.target_net(next_obs).gather(1, next_actions).squeeze(1)
