from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.agents.base import Agent, Transition
from src.env.factory import build_env
from src.training.evaluator import SOLVED_THRESHOLD, evaluate
from src.utils.config import RunConfig
from src.utils.logger import Logger
from src.utils.termination import classify

METRIC_COLUMNS = [
    "episode", "global_step", "return", "length",
    "epsilon", "loss_mean", "loss_std",
    "termination_reason", "terminated", "truncated", "wall_time_s",
]


class Trainer:
    def __init__(self, cfg: RunConfig, env: gym.Env, agent: Agent, logger: Logger, run_dir: Path ) -> None:
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.logger = logger
        self.run_dir = run_dir

        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        cfg = self.cfg.training
        total_steps = cfg.total_steps
        eval_every = cfg.eval_every
        save_every = cfg.save_every

        obs, _ = self.env.reset(seed=self.cfg.seed)
        episode_idx = 0
        ep_return = 0.0
        ep_length = 0
        ep_losses: list[float] = []
        last_obs = obs
        last_reward = 0.0
        last_info: dict = {}

        last_returns: deque[float] = deque(maxlen=100)
        t_start = time.time()

        eval_env = build_env(self.cfg.env, seed=self.cfg.seed)
        eval_episodes = cfg.eval_episodes
        best_eval_return = -float("inf")
        best_step = 0
        best_ckpt = self.ckpt_dir / "best.pt"
        # First periodic-eval step whose greedy mean return ≥ solved threshold
        # (cheap learning-curve proxy over training.eval_episodes; None if never).
        first_solve_step: int | None = None

        for global_step in range(1, total_steps + 1):
            action = self.agent.act(obs, greedy=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            transition = Transition(
                obs=obs.copy(),
                action=int(action) if np.isscalar(action) else action,
                reward=float(reward),
                next_obs=next_obs.copy(),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            self.agent.observe(transition)

            metrics = self.agent.learn()
            if "loss" in metrics:
                ep_losses.append(metrics["loss"])

            ep_return += float(reward)
            ep_length += 1
            last_obs = next_obs
            last_reward = float(reward)
            last_info = info

            if terminated or truncated:
                episode_idx += 1
                reason = classify(last_info, last_reward, last_obs, bool(terminated), bool(truncated))

                if ep_losses:
                    loss_mean = float(np.mean(ep_losses))
                    loss_std = float(np.std(ep_losses))
                else:
                    loss_mean = None
                    loss_std = None

                epsilon = getattr(self.agent, "epsilon", None)

                self.logger.log(
                    episode=episode_idx,
                    global_step=global_step,
                    length=ep_length,
                    epsilon=epsilon,
                    loss_mean=loss_mean,
                    loss_std=loss_std,
                    termination_reason=reason.value,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    wall_time_s=round(time.time() - t_start, 3),
                    **{"return": round(ep_return, 3)},
                )

                last_returns.append(ep_return)

                obs, _ = self.env.reset(seed=self.cfg.seed + episode_idx)
                ep_return = 0.0
                ep_length = 0
                ep_losses = []
            else:
                obs = next_obs

            if global_step % save_every == 0:
                ckpt_path = self.ckpt_dir / f"step_{global_step}.pt"
                self.agent.save(ckpt_path)

            if global_step % eval_every == 0:
                periodic = evaluate(
                    self.agent, eval_env,
                    episodes=eval_episodes, greedy=True, seed=self.cfg.seed,
                )
                mret = periodic["mean_return"]
                if first_solve_step is None and mret >= SOLVED_THRESHOLD:
                    first_solve_step = global_step
                if mret > best_eval_return:
                    best_eval_return = mret
                    best_step = global_step
                    self.agent.save(best_ckpt)
                    print(f"[eval] step={global_step} mean_return={mret:.2f} *new best*")
                else:
                    print(
                        f"[eval] step={global_step} mean_return={mret:.2f} "
                        f"(best={best_eval_return:.2f} @ step {best_step})"
                    )

        final_ckpt = self.ckpt_dir / "final.pt"
        self.agent.save(final_ckpt)

        verdict_ckpt = best_ckpt if best_ckpt.exists() else final_ckpt
        self.agent.load(verdict_ckpt)

        eval_cfg = self.cfg.eval
        eval_result = evaluate(
            self.agent,
            eval_env,
            episodes=eval_cfg.episodes,
            greedy=eval_cfg.greedy,
            seed=self.cfg.seed,
        )
        eval_env.close()

        summary = {
            "total_steps": total_steps,
            "episodes": episode_idx,
            "train_mean_return_last_100": (
                float(np.mean(last_returns)) if last_returns else 0.0
            ),
            "best_step": best_step,
            "best_eval_return": (
                best_eval_return if best_eval_return != -float("inf") else None
            ),
            "first_solve_step": first_solve_step,
            "verdict_checkpoint": verdict_ckpt.name,
            "mean_return_last_100": eval_result["mean_return"],
            "solved": eval_result["solved"],
            "eval_episodes": eval_result["episodes"],
            "eval_greedy": eval_result["greedy"],
            "std_return": eval_result["std_return"],
            "ci95_return": eval_result["ci95_return"],
            "termination_breakdown": eval_result["termination_breakdown"],
            "wall_time_s": round(time.time() - t_start, 3),
        }

        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary