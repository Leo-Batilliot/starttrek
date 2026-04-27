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


_OUT_OF_VIEW_X: float = 1.0
_LAND_REWARD_MIN: float = 80.0
_CRASH_REWARD_MAX: float = -80.0
_SLEEP_LIN_VELOCITY_MAX: float = 0.05
_SLEEP_ANG_VELOCITY_MAX: float = 0.05

_IDX_X: int = 0
_IDX_VX: int = 2
_IDX_VY: int = 3
_IDX_THETA_DOT: int = 5
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
    vx = float(arr[_IDX_VX])
    vy = float(arr[_IDX_VY])
    theta_dot = float(arr[_IDX_THETA_DOT])
    left_leg = float(arr[_IDX_LEFT_LEG]) >= 0.5
    right_leg = float(arr[_IDX_RIGHT_LEG]) >= 0.5

    if (left_leg and right_leg) and reward >= _LAND_REWARD_MIN:
        return TerminationReason.LAND

    if abs(x) >= _OUT_OF_VIEW_X:
        return TerminationReason.OUT_OF_VIEW

    if reward <= _CRASH_REWARD_MAX:
        return TerminationReason.CRASH

    if (
        not (left_leg or right_leg)
        and abs(vx) <= _SLEEP_LIN_VELOCITY_MAX
        and abs(vy) <= _SLEEP_LIN_VELOCITY_MAX
        and abs(theta_dot) <= _SLEEP_ANG_VELOCITY_MAX
    ):
        return TerminationReason.SLEEP

    return TerminationReason.UNKNOWN
