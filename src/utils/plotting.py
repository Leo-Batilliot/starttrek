from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
