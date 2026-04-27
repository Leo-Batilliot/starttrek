from __future__ import annotations

import time
import sys

from collections import Counter
from pathlib import Path

import numpy as np

from src.env.factory import build_env
from src.utils.config import load_config, save_config
from src.utils.logger import Logger
from src.utils.seeding import seed_all
from src.utils.termination import classify


METRIC_COLUMNS = [
    "episode", "global_step", "return", "length", "epsilon",
    "loss_mean", "loss_std", "termination_reason",
    "terminated", "truncated", "wall_time_s",
]


def main(config_path: str = "configs/smoke.yaml", render: bool = False) -> None:
    # 0.4 Load Config
    cfg = load_config(config_path)
    seed = cfg.seed
    n_episodes = cfg.eval.episodes

    print("=" * 70)
    print(f"PHASE 0 — run_name={cfg.run_name}, seed={seed}, episodes={n_episodes}")
    print("=" * 70)
    print(f"[T0.4] load_config({config_path})  ✓")

    # 0.3 Seeding
    seed_all(seed=seed)
    print(f"[T0.3] seed_all({seed})  ✓")

    # 0.7 Env Wrapper
    render_mode = "human" if render else None
    env = build_env(cfg.env, seed=seed, render_mode=render_mode)
    print(f"[T0.7] build_env  → obs={env.observation_space.shape}, " f"action={env.action_space.n}, render_mode={render_mode}")

    # 0.5 — Logger + csv
    run_dir = Path("runs") / cfg.run_name / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir, METRIC_COLUMNS)
    print(f"[T0.5] Logger     → {run_dir}/")

    # 0.4 — Save config
    save_config(cfg, run_dir / "config.yaml")
    print(f"[T0.4] save_config → {run_dir / 'config.yaml'}")
    print()

    # tmp (0.8 && 1.1)
    global_step = 0
    breakdown: Counter[str] = Counter()
    returns: list[float] = []
    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_length = 0
        last_reward = 0.0
        last_obs = obs
        last_info: dict = {}
        terminated = truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            global_step += 1
            last_reward = float(reward)
            last_obs = obs
            last_info = info

        # 0.6 — classification de la terminaison
        reason = classify(last_info, last_reward, last_obs, terminated, truncated)
        breakdown[reason.value] += 1
        returns.append(ep_return)

        logger.log(
            episode=ep,
            global_step=global_step,
            length=ep_length,
            epsilon=1.0,
            loss_mean=None,
            loss_std=None,
            termination_reason=reason.value,
            terminated=bool(terminated),
            truncated=bool(truncated),
            wall_time_s=round(time.time() - t0, 3),
            **{"return": round(ep_return, 3)},
        )
        print(f"  ep {ep:>2}: return={ep_return:>8.2f}  length={ep_length:>3}  "
              f"reason={reason.value}")

    logger.close()
    env.close()

    print()
    print("─" * 70)
    print(f"Mean return : {np.mean(returns):>8.2f}")
    print(f"Std  return : {np.std(returns):>8.2f}")
    print(f"Total steps : {global_step}")
    print("Termination breakdown:")
    for reason_name, count in breakdown.most_common():
        pct = 100.0 * count / n_episodes
        print(f"  {reason_name:>12}: {count:>3} ({pct:>5.1f}%)")
    print("─" * 70)
    print(f"Artefacts persistés dans : {run_dir}")
    print(f"  ├─ metrics.csv  ({(run_dir / 'metrics.csv').stat().st_size} octets)")
    print(f"  └─ events.log   ({(run_dir / 'events.log').stat().st_size} octets)")


if __name__ == "__main__":
    render = "--render" in sys.argv
    main(render=render)
