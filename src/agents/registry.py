from __future__ import annotations

from typing import Callable, TypeVar

from src.agents.base import Agent


AGENTS: dict[str, type[Agent]] = {}


T = TypeVar("T", bound=type[Agent])


def register_agent(name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        if not issubclass(cls, Agent):
            raise TypeError(
                f"@register_agent('{name}'): {cls.__name__} must inherit from Agent."
            )
        if name in AGENTS:
            raise ValueError(
                f"@register_agent('{name}'): name already registered to "
                f"{AGENTS[name].__name__}."
            )
        AGENTS[name] = cls
        return cls
    return decorator