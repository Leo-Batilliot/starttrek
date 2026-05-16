from __future__ import annotations

__all__ = ["linear_epsilon"]


def linear_epsilon(step: int, *, start: float, end: float, decay_steps: int) -> float:
    """Linearly anneal ε from `start` to `end` over `decay_steps`, then hold at `end`.

    Used by ε-greedy agents (DQN, Phase 2). The schedule is a *pure function* of
    the training step so it stays reproducible and unit-testable in isolation
    (no hidden state, no dependency on the agent or the env).

    Args:
        step: current training step (number of env transitions observed). Values
            ≤ 0 return `start`; values ≥ `decay_steps` return `end` (clamped).
        start: ε at step 0 (typically 1.0 — full exploration).
        end: ε once `decay_steps` is reached (typically 0.05 — residual exploration).
        decay_steps: number of steps over which ε is linearly interpolated. > 0.

    Returns:
        The interpolated ε for `step`: a float going monotonically from `start`
        to `end` (constant when `start == end`).
    """
    if decay_steps <= 0:
        raise ValueError(f"decay_steps must be > 0, got {decay_steps}.")
    if step <= 0:
        return float(start)
    if step >= decay_steps:
        return float(end)
    frac = step / decay_steps
    return float(start + frac * (end - start))
