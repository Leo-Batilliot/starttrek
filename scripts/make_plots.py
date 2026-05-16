from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.plotting import (
    plot_episode_length,
    plot_epsilon,
    plot_loss,
    plot_returns,
    plot_termination_breakdown,
)

_ROOT = Path(__file__).resolve().parents[1]

# (filename in assets/plots, plotting function) — the 5 plots required by T2.5.
_PLOTS = (
    ("returns_per_episode.png", plot_returns),
    ("loss.png", plot_loss),
    ("epsilon.png", plot_epsilon),
    ("episode_length.png", plot_episode_length),
    ("termination_breakdown.png", plot_termination_breakdown),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the T2.5 training plots from a run's metrics.csv"
    )
    parser.add_argument(
        "--run",
        default="runs/dqn_baseline/0",
        help="Run directory containing metrics.csv (default: runs/dqn_baseline/0)",
    )
    parser.add_argument(
        "--out-dir",
        default="assets/plots",
        help="Where the PNGs are written (default: assets/plots)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = _ROOT / run_dir
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        print(f"ERROR: metrics.csv not found at '{csv_path}'.", file=sys.stderr)
        return 84

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  T2.5 — training plots from {csv_path}")
    print("=" * 70)
    for filename, fn in _PLOTS:
        path = fn(csv_path, out_dir / filename)
        print(f"  wrote {path}  ({path.stat().st_size} B)")
    print("-" * 70)
    print(f"  {len(_PLOTS)} plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
