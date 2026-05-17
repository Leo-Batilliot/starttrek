from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.plotting import (  # noqa: E402
    _aggregate_curves as aggregate_curves,
    _binned_means as binned_means,
    plot_returns_paired,
)

_ROOT = Path(__file__).resolve().parents[1]

__all__ = ["binned_means", "aggregate_curves", "main"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired returns plot: ablation vs baseline (T3.3)")
    p.add_argument("--baseline", default="runs/dqn_baseline")
    p.add_argument("--ablation", default="runs/dqn_buffer10k")
    p.add_argument("--label-baseline", default="baseline (buffer 100k)")
    p.add_argument("--label-ablation", default="ablation (buffer 10k)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--total-steps", type=int, default=500000)
    p.add_argument("--bins", type=int, default=25)
    p.add_argument(
        "--title",
        default="Paired training curves — replay buffer ablation",
    )
    p.add_argument("--out", default="assets/plots/paired_buffer10k_vs_baseline.png")
    return p.parse_args()


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def _seed_dirs(base: Path, seeds: list[int]) -> list[Path]:
    return [base / str(s) for s in seeds]


def main() -> int:
    args = parse_args()
    baseline = _abs(args.baseline)
    ablation = _abs(args.ablation)
    for base in (baseline, ablation):
        if not base.exists():
            print(f"ERROR: run dir not found: {base}", file=sys.stderr)
            return 84

    series = [
        (_seed_dirs(baseline, args.seeds), args.label_baseline, "steelblue"),
        (_seed_dirs(ablation, args.seeds), args.label_ablation, "darkorange"),
    ]
    out = plot_returns_paired(
        series,
        _abs(args.out),
        total_steps=args.total_steps,
        bins=args.bins,
        title=args.title,
    )
    print(f"wrote {out} ({out.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
