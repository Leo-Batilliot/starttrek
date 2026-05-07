from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from src.agents.base import Agent
from src.utils.termination import TerminationReason, classify


SOLVED_THRESHOLD: float = 200.0
SOLVED_MIN_EPISODES: int = 100
_CI95_Z: float = 1.96

_REASONS: tuple[str, ...] = (
    TerminationReason.LAND.value,
    TerminationReason.CRASH.value,
    TerminationReason.OUT_OF_VIEW.value,
    TerminationReason.SLEEP.value,
    TerminationReason.TRUNCATED.value,
    TerminationReason.UNKNOWN.value,
)


def evaluate(
    agent: Agent,
    env: gym.Env,
    *,
    episodes: int,
    greedy: bool,
    record_video: bool = False,
    video_dir: Path | None = None,
    video_every: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")
    if record_video and video_every <= 0:
        raise ValueError(f"video_every must be > 0 when record_video=True, got {video_every}")

    returns: list[float] = []
    lengths: list[int] = []
    reasons: list[str] = []
    saved_videos: list[str] = []

    for ep_idx in range(episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        ep_return = 0.0
        ep_length = 0
        terminated = False
        truncated = False
        last_reward = 0.0
        last_obs = obs
        last_info: dict = {}

        capture = record_video and (ep_idx % video_every == 0)
        frames: list[np.ndarray] = []
        if capture:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

        while not (terminated or truncated):
            action = agent.act(obs, greedy=greedy)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            last_reward = float(reward)
            last_obs = next_obs
            last_info = info
            obs = next_obs

            if capture:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))

        reason = classify(last_info, last_reward, last_obs, bool(terminated), bool(truncated))
        returns.append(ep_return)
        lengths.append(ep_length)
        reasons.append(reason.value)

        if capture and frames and video_dir is not None:
            path = _save_video(frames, Path(video_dir), ep_idx)
            if path is not None:
                saved_videos.append(str(path))

    return _aggregate(returns, lengths, reasons, greedy=greedy, videos=saved_videos)


def _aggregate(
    returns: list[float],
    lengths: list[int],
    reasons: list[str],
    *,
    greedy: bool,
    videos: list[str],
) -> dict[str, Any]:
    n = len(returns)
    arr = np.asarray(returns, dtype=np.float64)
    mean_return = float(arr.mean())
    std_return = float(arr.std(ddof=1)) if n >= 2 else 0.0
    sem = std_return / math.sqrt(n) if n >= 2 else 0.0
    ci95 = [mean_return - _CI95_Z * sem, mean_return + _CI95_Z * sem]

    counts = Counter(reasons)
    breakdown = {r: counts.get(r, 0) / n for r in _REASONS}

    solved = (n >= SOLVED_MIN_EPISODES) and (mean_return >= SOLVED_THRESHOLD)

    return {
        "episodes": n,
        "greedy": greedy,
        "mean_return": mean_return,
        "std_return": std_return,
        "ci95_return": ci95,
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "median_return": float(np.median(arr)),
        "mean_length": float(np.mean(lengths)),
        "termination_breakdown": breakdown,
        "solved": solved,
        "returns": [float(r) for r in returns],
        "lengths": [int(l) for l in lengths],
        "videos": videos,
    }


def _save_video(frames: list[np.ndarray], video_dir: Path, ep_idx: int) -> Path | None:
    try:
        from src.utils.video import save_video  # T0.13
    except ImportError:
        return None
    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / f"episode_{ep_idx:04d}.mp4"
    save_video(frames, path)
    return path
