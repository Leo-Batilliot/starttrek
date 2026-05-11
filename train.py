from __future__ import annotations

import argparse
import sys

from dataclasses import replace
from pathlib import Path

from src.agents.registry import AGENTS
from src.env.factory import build_env
from src.training.trainer import Trainer, METRIC_COLUMNS
from src.utils.config import load_config, save_config
from src.utils.logger import Logger
from src.utils.seeding import seed_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent on a Gymnasium env")

    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None,help="Override seed defined in YAML")
    parser.add_argument("--out-dir", type=str, default="runs", help="Root directory to save results")

    return parser.parse_args()

def main() -> int:
    args = parse_args()

    # Load config (0.4)
    cfg = load_config(args.config)

    # Override the seed if the user has specified
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)

    # Seed all (0.3)
    seed = cfg.seed
    seed_all(seed=seed)

    # Run directory
    run_dir = Path(args.out_dir) / cfg.run_name / str(cfg.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save associete config
    save_config(cfg, run_dir / "config.yaml")

    # Build env 0.7
    env = build_env(cfg.env, seed=cfg.seed)

    if cfg.agent.name not in AGENTS:
        available = ", ".join(sorted(AGENTS.keys())) or "(none registered)"
        print(f"ERROR: agent '{cfg.agent.name}' not registered. \n Available: {available}", file=sys.stderr)
        return 84

    agent_class = AGENTS[cfg.agent.name]

    agent = agent_class(cfg.agent, n_actions=env.action_space.n)

    # Build logger
    logger = Logger(run_dir, METRIC_COLUMNS)

    # Print run banner
    print("=" * 70)
    print(f"  run_name : {cfg.run_name}")
    print(f"  seed     : {cfg.seed}")
    print(f"  agent    : {cfg.agent.name}")
    print(f"  env      : {cfg.env.id}")
    print(f"  steps    : {cfg.training.total_steps}")
    print(f"  out_dir  : {run_dir}")
    print("=" * 70)

    try:
        trainer = Trainer(cfg, env, agent, logger, run_dir)
        summary = trainer.run()
    finally:
        logger.close()
        env.close()

    # Sumarry
    print()
    print("─" * 70)
    print(f"  Episodes              : {summary['episodes']}")
    print(f"  Total steps           : {summary['total_steps']}")
    print(f"  Mean return (last 100): {summary['mean_return_last_100']:.2f}")
    print(f"  Wall time             : {summary['wall_time_s']:.1f}s")
    print(f"  Artefacts             : {run_dir}")
    print("─" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())