from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.registry import AGENTS
from src.env.factory import build_env
from src.training.evaluator import evaluate
from src.utils.config import load_config
from src.utils.logger import Logger
from src.utils.seeding import seed_all
from src.utils.termination import TerminationReason

_ROOT = Path(__file__).resolve().parents[1]

_BASELINES: tuple[tuple[str, str], ...] = (
    ("random", "configs/baseline/random.yaml"),
    ("heuristic", "configs/baseline/heuristic.yaml"),
)

_REASONS: tuple[str, ...] = tuple(r.value for r in TerminationReason)
_LOG_COLUMNS: list[str] = ["agent", "episodes", "mean_return", *_REASONS, "unknown_count", "verdict"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate termination.classify on real trajectories (T1.3)")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per baseline (default: 100)")
    parser.add_argument("--out-dir", type=str, default="runs/validate_termination", help="Where events.log / metrics.csv are written")
    return parser.parse_args()


def _evaluate_baseline(config_path: Path, episodes: int) -> dict:
    cfg = load_config(config_path)
    seed_all(seed=cfg.seed)
    env = build_env(cfg.env, seed=cfg.seed)
    try:
        agent_cls = AGENTS[cfg.agent.name]
        agent = agent_cls(cfg.agent, n_actions=env.action_space.n)
        return evaluate(agent, env, episodes=episodes, greedy=True, seed=cfg.seed)
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        print(f"ERROR: --episodes must be > 0, got {args.episodes}", file=sys.stderr)
        return 84

    out_dir = (_ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    logger = Logger(out_dir, _LOG_COLUMNS, flush_every=1)

    overall_ok = True
    print("=" * 72)
    print(f"  T1.3 — termination.classify validation ({args.episodes} episodes / baseline)")
    print("=" * 72)

    try:
        for name, rel_path in _BASELINES:
            result = _evaluate_baseline(_ROOT / rel_path, args.episodes)
            breakdown = result["termination_breakdown"]
            n = result["episodes"]
            unknown_count = round(breakdown[TerminationReason.UNKNOWN.value] * n)

            no_unknown = unknown_count == 0
            if name == "random":
                crash = breakdown[TerminationReason.CRASH.value]
                oov = breakdown[TerminationReason.OUT_OF_VIEW.value]
                plausible = (crash + oov) >= 0.5
                plausible_msg = f"crash+out_of_view={crash + oov:.2f} (expected majority)"
            else:
                land = breakdown[TerminationReason.LAND.value]
                plausible = land > 0.0
                plausible_msg = f"land={land:.2f} (expected non-zero)"

            ok = no_unknown and plausible
            overall_ok = overall_ok and ok
            verdict = "PASS" if ok else "FAIL"

            logger.log(
                agent=name,
                episodes=n,
                mean_return=round(result["mean_return"], 2),
                **{r: round(breakdown[r], 4) for r in _REASONS},
                unknown_count=unknown_count,
                verdict=verdict,
            )

            print()
            print(f"  [{name}]  {verdict}")
            print(f"    episodes        : {n}")
            print(f"    mean_return     : {result['mean_return']:.2f}")
            print(f"    breakdown       : { {r: round(breakdown[r], 3) for r in _REASONS} }")
            print(f"    UNKNOWN count   : {unknown_count}  (criterion: == 0)")
            print(f"    plausibility    : {plausible_msg} -> {'OK' if plausible else 'NOT OK'}")

        final = "PASS" if overall_ok else "FAIL"
        logger.log(agent="ALL", episodes=args.episodes, verdict=final)
    finally:
        logger.close()

    print()
    print("-" * 72)
    print(f"  OVERALL: {'PASS' if overall_ok else 'FAIL'}   (log: {out_dir / 'events.log'})")
    print("-" * 72)

    return 0 if overall_ok else 84


if __name__ == "__main__":
    sys.exit(main())
