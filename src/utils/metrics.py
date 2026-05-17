from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Sequence

# z for a 95% CI under a normal approximation — same constant as
# evaluator._CI95_Z, kept consistent so aggregated CIs match per-run CIs.
_CI95_Z = 1.96

# T3.1 acceptance: at least this many seeds (out of n) must be solved.
SOLVED_CRITERION_MIN = 4


def mean_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean and 95% confidence interval (normal approx, sample std ddof=1).

    Returns ``(mean, lo, hi)``. ``n == 0`` → all zeros; ``n == 1`` → zero-width
    interval ``(v, v, v)`` (no dispersion estimable from one sample).
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(values) / n
    if n < 2:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    return (mean, mean - _CI95_Z * sem, mean + _CI95_Z * sem)


def aggregate(per_seed: list[dict]) -> dict:
    """Aggregate per-seed run summaries. Pure (no I/O) → testable in isolation.

    Each item must expose ``seed``, ``mean_return_last_100``, ``solved`` and
    ``first_solve_step`` (the last may be ``None`` for unsolved seeds).
    """
    returns = [s["mean_return_last_100"] for s in per_seed]
    solved_flags = [bool(s["solved"]) for s in per_seed]
    solve_steps = [
        s["first_solve_step"]
        for s in per_seed
        if s.get("first_solve_step") is not None
    ]

    ret_mean, ret_lo, ret_hi = mean_ci95(returns)
    if solve_steps:
        fss_mean, fss_lo, fss_hi = mean_ci95([float(x) for x in solve_steps])
    else:
        fss_mean = fss_lo = fss_hi = None

    n = len(per_seed)
    n_solved = sum(solved_flags)
    return {
        "n_seeds": n,
        "n_solved": n_solved,
        "all_solved": n_solved == n,
        "criterion_met": n_solved >= SOLVED_CRITERION_MIN,
        "mean_return_last_100": {
            "mean": ret_mean,
            "ci95": [ret_lo, ret_hi],
            "min": min(returns) if returns else None,
            "max": max(returns) if returns else None,
        },
        "first_solve_step": {
            "mean": fss_mean,
            "ci95": [fss_lo, fss_hi],
            "n_solved_seeds": len(solve_steps),
        },
        "per_seed": [
            {
                "seed": s["seed"],
                "mean_return_last_100": s["mean_return_last_100"],
                "solved": bool(s["solved"]),
                "first_solve_step": s.get("first_solve_step"),
            }
            for s in per_seed
        ],
    }


def aggregate_runs(pattern: str) -> dict:
    """Glob ``pattern`` for per-seed ``summary.json`` files and aggregate them.

    ``pattern`` matches the summary files, e.g.
    ``runs/dqn_baseline/*/summary.json``. The seed is taken from each file's
    parent directory name (int when numeric, for stable ordering).
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no summary.json matched pattern: {pattern!r}")

    per_seed: list[dict] = []
    for p in paths:
        sp = Path(p)
        with sp.open(encoding="utf-8") as f:
            s = json.load(f)
        name = sp.parent.name
        s["seed"] = int(name) if name.lstrip("-").isdigit() else name
        per_seed.append(s)

    per_seed.sort(key=lambda d: (isinstance(d["seed"], str), d["seed"]))
    return aggregate(per_seed)
