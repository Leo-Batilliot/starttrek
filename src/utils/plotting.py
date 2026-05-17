from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import mean_ci95
from src.utils.termination import TerminationReason


_REASON_ORDER: tuple[str, ...] = (
    TerminationReason.LAND.value,
    TerminationReason.CRASH.value,
    TerminationReason.OUT_OF_VIEW.value,
    TerminationReason.SLEEP.value,
    TerminationReason.TRUNCATED.value,
    TerminationReason.UNKNOWN.value,
)


def _read_columns(csv_path: Path | str, columns: list[str]) -> dict[str, list[str]]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"metrics csv not found: {p}")
    out: dict[str, list[str]] = {c: [] for c in columns}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"metrics csv has no header: {p}")
        for col in columns:
            if col not in reader.fieldnames:
                raise ValueError(f"column '{col}' missing from {p}; got {reader.fieldnames}")
        for row in reader:
            for col in columns:
                out[col].append(row[col])
    return out


def _floats(values: list[str]) -> np.ndarray:
    parsed: list[float] = []
    for v in values:
        if v == "" or v is None:
            parsed.append(float("nan"))
        else:
            parsed.append(float(v))
    return np.asarray(parsed, dtype=np.float64)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr.size == 0:
        return arr.copy()
    w = min(window, arr.size)
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(arr, kernel, mode="valid")


def _save_fig(fig, out_path: Path | str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_returns(csv_path: Path | str, out_path: Path | str, *, window: int = 100) -> Path:
    data = _read_columns(csv_path, ["episode", "return"])
    episodes = _floats(data["episode"])
    returns = _floats(data["return"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, returns, color="steelblue", alpha=0.4, linewidth=0.8, label="return")
    if returns.size >= 2 and window > 1:
        smoothed = _rolling_mean(returns, window)
        x_smoothed = episodes[window - 1 :] if returns.size >= window else episodes[-smoothed.size :]
        ax.plot(x_smoothed, smoothed, color="darkorange", linewidth=1.8, label=f"rolling mean ({window})")
    ax.axhline(200.0, color="green", linestyle="--", linewidth=0.8, label="solved (200)")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("Episodic return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    return _save_fig(fig, out_path)


def plot_episode_length(csv_path: Path | str, out_path: Path | str, *, window: int = 100) -> Path:
    data = _read_columns(csv_path, ["episode", "length"])
    episodes = _floats(data["episode"])
    lengths = _floats(data["length"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, lengths, color="teal", alpha=0.4, linewidth=0.8, label="length")
    if lengths.size >= 2 and window > 1:
        smoothed = _rolling_mean(lengths, window)
        x_smoothed = episodes[window - 1 :] if lengths.size >= window else episodes[-smoothed.size :]
        ax.plot(x_smoothed, smoothed, color="darkorange", linewidth=1.8, label=f"rolling mean ({window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode length (steps)")
    ax.set_title("Episode length")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    return _save_fig(fig, out_path)


def plot_loss(csv_path: Path | str, out_path: Path | str) -> Path:
    data = _read_columns(csv_path, ["episode", "loss_mean"])
    episodes = _floats(data["episode"])
    loss = _floats(data["loss_mean"])
    mask = ~np.isnan(loss)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if mask.any():
        ax.plot(episodes[mask], loss[mask], color="crimson", linewidth=1.0)
    ax.set_xlabel("episode")
    ax.set_ylabel("loss (mean)")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, out_path)


def plot_epsilon(csv_path: Path | str, out_path: Path | str) -> Path:
    data = _read_columns(csv_path, ["episode", "epsilon"])
    episodes = _floats(data["episode"])
    epsilon = _floats(data["epsilon"])
    mask = ~np.isnan(epsilon)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if mask.any():
        ax.plot(episodes[mask], epsilon[mask], color="purple", linewidth=1.2)
    ax.set_xlabel("episode")
    ax.set_ylabel("epsilon")
    ax.set_title("Exploration schedule (epsilon)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, out_path)


def plot_termination_breakdown(csv_path: Path | str, out_path: Path | str) -> Path:
    data = _read_columns(csv_path, ["termination_reason"])
    reasons = [r for r in data["termination_reason"] if r != ""]
    counts = Counter(reasons)
    total = sum(counts.values()) or 1

    labels = list(_REASON_ORDER)
    values = [counts.get(label, 0) / total for label in labels]
    colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f", "#bcbd22"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("fraction of episodes")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Termination reason breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    return _save_fig(fig, out_path)


def _bin_edges(total_steps: float, bins: int) -> list[float]:
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    return [i * total_steps / bins for i in range(bins + 1)]


def _binned_means(
    steps: Sequence[float], returns: Sequence[float], edges: Sequence[float]
) -> list[float]:
    n_bins = len(edges) - 1
    sums = [0.0] * n_bins
    counts = [0] * n_bins
    for st, ret in zip(steps, returns):
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if (lo <= st < hi) or (i == n_bins - 1 and st == hi):
                sums[i] += ret
                counts[i] += 1
                break
    return [
        (sums[i] / counts[i]) if counts[i] > 0 else float("nan")
        for i in range(n_bins)
    ]


def _aggregate_curves(
    per_seed_binned: list[list[float]],
) -> tuple[list[float], list[float], list[float]]:
    n_bins = len(per_seed_binned[0])
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for b in range(n_bins):
        vals = [seed[b] for seed in per_seed_binned if not math.isnan(seed[b])]
        if vals:
            m, lo, hi = mean_ci95(vals)
        else:
            m = lo = hi = float("nan")
        means.append(m)
        los.append(lo)
        his.append(hi)
    return means, los, his


def _load_seed_curves(
    run_dirs: Iterable[Path | str], edges: Sequence[float]
) -> list[list[float]]:
    per_seed: list[list[float]] = []
    for d in run_dirs:
        csv_path = Path(d) / "metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"metrics csv not found: {csv_path}")
        steps: list[float] = []
        rets: list[float] = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for col in ("global_step", "return"):
                if reader.fieldnames is None or col not in reader.fieldnames:
                    raise ValueError(f"column '{col}' missing from {csv_path}")
            for row in reader:
                steps.append(float(row["global_step"]))
                rets.append(float(row["return"]))
        per_seed.append(_binned_means(steps, rets, edges))
    if not per_seed:
        raise ValueError("no run dirs provided")
    return per_seed


def _draw_series(
    ax,
    run_dirs: Sequence[Path | str],
    edges: Sequence[float],
    centers: Sequence[float],
    *,
    label: str,
    color: str,
) -> None:
    per_seed = _load_seed_curves(run_dirs, edges)
    means, los, his = _aggregate_curves(per_seed)
    ax.plot(centers, means, color=color, linewidth=1.8, label=label)
    ax.fill_between(centers, los, his, color=color, alpha=0.2)


def plot_returns_multi_seed(
    run_dirs: Sequence[Path | str],
    out_path: Path | str,
    *,
    label: str = "DQN",
    color: str = "steelblue",
    total_steps: float = 500000,
    bins: int = 25,
    title: str | None = None,
) -> Path:
    edges = _bin_edges(total_steps, bins)
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(bins)]
    n = len(list(run_dirs))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _draw_series(
        ax, run_dirs, edges, centers,
        label=f"{label} (mean, {n} seeds)", color=color,
    )
    # Re-label the CI band via a proxy entry so the legend stays readable.
    ax.fill_between([], [], [], color=color, alpha=0.2, label="95% CI")
    ax.axhline(200.0, color="green", linestyle="--", linewidth=0.8, label="solved (200)")
    ax.set_xlabel("training step")
    ax.set_ylabel("episodic return (mean ± 95% CI)")
    ax.set_title(title or f"{label} — return over training ({n} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    return _save_fig(fig, out_path)


def plot_returns_paired(
    series: Sequence[tuple[Sequence[Path | str], str, str]],
    out_path: Path | str,
    *,
    total_steps: float = 500000,
    bins: int = 25,
    title: str = "Paired training curves (mean ± 95% CI, seeds 0..4)",
) -> Path:
    if not series:
        raise ValueError("plot_returns_paired needs at least one series")
    edges = _bin_edges(total_steps, bins)
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(bins)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run_dirs, label, color in series:
        _draw_series(ax, run_dirs, edges, centers, label=label, color=color)
    ax.axhline(200.0, color="green", linestyle="--", linewidth=0.8, label="solved (200)")
    ax.set_xlabel("training step")
    ax.set_ylabel("episodic return (mean ± 95% CI, seeds 0..4)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    return _save_fig(fig, out_path)
