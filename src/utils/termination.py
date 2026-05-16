from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class TerminationReason(str, Enum):
    LAND = "land"
    CRASH = "crash"
    OUT_OF_VIEW = "out_of_view"
    SLEEP = "sleep"
    TRUNCATED = "truncated"
    UNKNOWN = "unknown"


_TERMINAL_REWARD_MAGNITUDE: float = 80.0
_OUT_OF_VIEW_X: float = 1.0

_IDX_X: int = 0
_IDX_LEFT_LEG: int = 6
_IDX_RIGHT_LEG: int = 7


def classify(
    info: dict[str, Any],
    reward: float,
    state: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> TerminationReason:
    if isinstance(info, dict) and "termination_reason" in info:
        try:
            return TerminationReason(info["termination_reason"])
        except ValueError:
            pass

    if truncated:
        return TerminationReason.TRUNCATED

    if not terminated:
        return TerminationReason.UNKNOWN

    arr = np.asarray(state, dtype=np.float32)
    if arr.shape != (8,):
        raise ValueError(f"state must have shape (8,), got {arr.shape}")

    x = float(arr[_IDX_X])
    any_leg = (
        float(arr[_IDX_LEFT_LEG]) >= 0.5 or float(arr[_IDX_RIGHT_LEG]) >= 0.5
    )

    if reward >= _TERMINAL_REWARD_MAGNITUDE:
        return TerminationReason.LAND if any_leg else TerminationReason.SLEEP

    if reward <= -_TERMINAL_REWARD_MAGNITUDE:

        if abs(x) >= _OUT_OF_VIEW_X:
            return TerminationReason.OUT_OF_VIEW
        return TerminationReason.CRASH

    if abs(x) >= _OUT_OF_VIEW_X:
        return TerminationReason.OUT_OF_VIEW
    return TerminationReason.UNKNOWN
