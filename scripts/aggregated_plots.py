from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.plotting import plot_returns_multi_seed, plot_returns_paired  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]

_RUNS = (
    ("runs/dqn_baseline", "baseline (buffer 100k, 2×128)", "steelblue", "agg_returns_baseline.png"),
    ("runs/dqn_buffer10k", "ablation A (buffer 10k)", "darkorange", "agg_returns_buffer10k.png"),
    ("runs/dqn_depth3", "ablation B (3×128)", "seagreen", "agg_returns_depth3.png"),
    ("runs/double_dqn", "Double DQN", "crimson", "agg_returns_double_dqn.png"),
    ("runs/dqn_softtarget", "Target soft (Polyak)", "purple", "agg_returns_softtarget.png"),
)

_PAIRED = (
    (
        "runs/dqn_buffer10k",
        "ablation (buffer 10k)",
        "darkorange",
        "paired_buffer10k_vs_baseline.png",
        "Paired training curves — replay buffer ablation",
    ),
    (
        "runs/dqn_depth3",
        "ablation (3 hidden layers)",
        "seagreen",
        "paired_depth3_vs_baseline.png",
        "Paired training curves — network depth ablation",
    ),
    (
        "runs/double_dqn",
        "Double DQN",
        "crimson",
        "paired_double_dqn_vs_baseline.png",
        "Paired training curves — DQN vs Double DQN",
    ),
    (
        "runs/dqn_softtarget",
        "target soft (Polyak)",
        "purple",
        "paired_softtarget_vs_baseline.png",
        "Paired training curves — hard vs soft target update",
    ),
)

_BASELINE = "runs/dqn_baseline"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate T3.6 aggregated + paired plots")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--total-steps", type=int, default=500000)
    p.add_argument("--bins", type=int, default=25)
    p.add_argument("--out-dir", default="assets/plots")
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Warn and skip a plot whose run dirs are absent instead of failing",
    )
    return p.parse_args()


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def _seed_dirs(base: str, seeds: list[int]) -> list[Path]:
    return [_abs(base) / str(s) for s in seeds]


def _missing(base: str, seeds: list[int]) -> list[Path]:
    return [d for d in _seed_dirs(base, seeds) if not (d / "metrics.csv").exists()]


def main() -> int:
    args = parse_args()
    out_dir = _abs(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped: list[str] = []

    print("=" * 70)
    print("  T3.6 — aggregated + paired return plots (mean ± 95% CI)")
    print("=" * 70)

    for base, label, color, fname in _RUNS:
        miss = _missing(base, args.seeds)
        if miss:
            msg = f"{base}: missing metrics.csv for {[str(m) for m in miss]}"
            if args.skip_missing:
                print(f"  SKIP  {fname}  ({msg})")
                skipped.append(fname)
                continue
            print(f"ERROR: {msg}", file=sys.stderr)
            return 84
        out = plot_returns_multi_seed(
            _seed_dirs(base, args.seeds),
            out_dir / fname,
            label=label,
            color=color,
            total_steps=args.total_steps,
            bins=args.bins,
        )
        print(f"  wrote {out}  ({out.stat().st_size} B)")
        written.append(out)

    for abl_base, abl_label, abl_color, fname, title in _PAIRED:
        miss = _missing(_BASELINE, args.seeds) + _missing(abl_base, args.seeds)
        if miss:
            msg = f"{abl_base}/baseline: missing {[str(m) for m in miss]}"
            if args.skip_missing:
                print(f"  SKIP  {fname}  ({msg})")
                skipped.append(fname)
                continue
            print(f"ERROR: {msg}", file=sys.stderr)
            return 84
        out = plot_returns_paired(
            [
                (_seed_dirs(_BASELINE, args.seeds), "baseline (2×128, buffer 100k)", "steelblue"),
                (_seed_dirs(abl_base, args.seeds), abl_label, abl_color),
            ],
            out_dir / fname,
            total_steps=args.total_steps,
            bins=args.bins,
            title=title,
        )
        print(f"  wrote {out}  ({out.stat().st_size} B)")
        written.append(out)

    print("-" * 70)
    print(f"  {len(written)} plots written to {out_dir}"
          + (f"  ({len(skipped)} skipped: {skipped})" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
