from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.agents.registry import AGENTS
from src.env.factory import build_env
from src.training.evaluator import evaluate
from src.utils.config import load_config
from src.utils.seeding import seed_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained agent on its env")

    parser.add_argument("--run", required=True, help="Path to run directory (contains config.yaml)")
    parser.add_argument("--episodes", type=int, default=None, help="Override eval.episodes from config")
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=None, help="Force greedy policy (overrides config). Use --no-greedy to disable.")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=None, help="Force video recording (overrides config). Use --no-record-video to disable.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file (default: <run>/checkpoints/final.pt)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed used for env reset (default: cfg.seed)")

    return parser.parse_args()


def _resolve_checkpoint(run_dir: Path, override: str | None) -> Path | None:
    if override is not None:
        p = Path(override)
        if not p.exists():
            print(f"ERROR: checkpoint '{p}' not found.", file=sys.stderr)
            return None
        return p

    final = run_dir / "checkpoints" / "final.pt"
    if final.exists():
        return final

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        steps = sorted(ckpt_dir.glob("step_*.pt"))
        if steps:
            return steps[-1]

    return None


def main() -> int:
    args = parse_args()

    run_dir = Path(args.run)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        print(f"ERROR: config not found at '{cfg_path}'.", file=sys.stderr)
        return 84

    cfg = load_config(cfg_path)

    episodes = args.episodes if args.episodes is not None else cfg.eval.episodes
    greedy = args.greedy if args.greedy is not None else cfg.eval.greedy
    record_video = args.record_video if args.record_video is not None else cfg.eval.record_video
    seed = args.seed if args.seed is not None else cfg.seed

    seed_all(seed=seed)

    render_mode = "rgb_array" if record_video else None
    env = build_env(cfg.env, seed=seed, render_mode=render_mode)

    if cfg.agent.name not in AGENTS:
        available = ", ".join(sorted(AGENTS.keys())) or "(none registered)"
        print(f"ERROR: agent '{cfg.agent.name}' not registered.\n Available: {available}", file=sys.stderr)
        env.close()
        return 84

    agent_class = AGENTS[cfg.agent.name]
    agent = agent_class(cfg.agent, n_actions=env.action_space.n)

    ckpt = _resolve_checkpoint(run_dir, args.checkpoint)
    if ckpt is not None:
        agent.load(ckpt)

    print("=" * 70)
    print(f"  run       : {run_dir}")
    print(f"  agent     : {cfg.agent.name}")
    print(f"  env       : {cfg.env.id}")
    print(f"  episodes  : {episodes}")
    print(f"  greedy    : {greedy}")
    print(f"  video     : {record_video}")
    print(f"  ckpt      : {ckpt if ckpt is not None else '(none — using fresh agent)'}")
    print("=" * 70)

    video_dir = run_dir / "videos" if record_video else None

    try:
        result = evaluate(
            agent,
            env,
            episodes=episodes,
            greedy=greedy,
            record_video=record_video,
            video_dir=video_dir,
            video_every=cfg.eval.video_every,
            seed=seed,
        )
    finally:
        env.close()

    out_path = run_dir / f"eval_{time.strftime("%Y%m%dT%H%M%S")}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print("─" * 70)
    print(f"  episodes              : {result['episodes']}")
    print(f"  mean_return           : {result['mean_return']:.2f}")
    print(f"  std_return            : {result['std_return']:.2f}")
    print(f"  ci95_return           : [{result['ci95_return'][0]:.2f}, {result['ci95_return'][1]:.2f}]")
    print(f"  solved (≥200 over 100): {result['solved']}")
    print(f"  termination_breakdown : {result['termination_breakdown']}")
    print(f"  output                : {out_path}")
    print("─" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
