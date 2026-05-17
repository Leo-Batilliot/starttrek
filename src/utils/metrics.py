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


# --------------------------------------------------------------------------- #
# Significance testing (pure Python — no scipy; §2.1 PROJECT.md).
#
# Welch's t-test (unequal variances) + two-sided p-value via the regularized
# incomplete beta function (Numerical Recipes `betai`/`betacf`), and Cohen's d.
# The project's headline conclusions rest on CI95 overlap/disjunction (robust
# to small n); these add a formal test alongside it.
# --------------------------------------------------------------------------- #


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """Welch's t-test. Returns ``(t, df, p_two_sided)``.

    Two-sided p from the Student-t survival function via ``betai`` (no scipy).
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("welch_ttest needs >= 2 samples in each group")
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    se2 = v1 / n1 + v2 / n2
    if se2 == 0.0:
        if m1 == m2:
            return (0.0, float(n1 + n2 - 2), 1.0)
        return (math.inf if m1 > m2 else -math.inf, float(n1 + n2 - 2), 0.0)
    t = (m1 - m2) / math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = betai(df / 2.0, 0.5, df / (df + t * t))
    return (t, df, p)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised mean difference (pooled SD)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("cohens_d needs >= 2 samples in each group")
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return 0.0 if sp == 0.0 else (m1 - m2) / sp


def compare_samples(a: Sequence[float], b: Sequence[float]) -> dict:
    """Full pairwise comparison of two seed-level samples (a = reference).

    Combines the mean difference, CI95 overlap test (the project's primary
    criterion), Welch's t-test and Cohen's d into one verdict.
    """
    ma, lo_a, hi_a = mean_ci95(a)
    mb, lo_b, hi_b = mean_ci95(b)
    t, df, p = welch_ttest(a, b)
    d = cohens_d(a, b)
    ci_disjoint = (hi_a < lo_b) or (hi_b < lo_a)
    significant = p < 0.05
    return {
        "mean_a": ma,
        "mean_b": mb,
        "diff": mb - ma,
        "pct": (mb - ma) / ma * 100.0 if ma != 0 else float("nan"),
        "ci95_a": [lo_a, hi_a],
        "ci95_b": [lo_b, hi_b],
        "ci95_disjoint": ci_disjoint,
        "welch_t": t,
        "welch_df": df,
        "p_value": p,
        "cohens_d": d,
        "significant": significant,
        "verdict": (
            "SIGNIFICATIF" if significant else "NON significatif"
        ) + (" (IC95 disjoints)" if ci_disjoint else " (IC95 recouvrants)"),
    }
