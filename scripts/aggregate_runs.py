from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import aggregate, mean_ci95

_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate multi-seed DQN baseline runs (T3.1)"
    )
    p.add_argument("--base", default="runs/dqn_baseline", help="Base run dir")
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds"
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output path (default: <base>/summary_aggregated.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base)
    if not base.is_absolute():
        base = _ROOT / base

    per_seed: list[dict] = []
    missing: list[int] = []
    for seed in args.seeds:
        sp = base / str(seed) / "summary.json"
        if not sp.exists():
            missing.append(seed)
            continue
        with sp.open(encoding="utf-8") as f:
            s = json.load(f)
        s["seed"] = seed
        per_seed.append(s)

    if missing:
        print(f"ERROR: missing summary.json for seeds {missing} under {base}", file=sys.stderr)
        return 84
    if not per_seed:
        print(f"ERROR: no runs found under {base}", file=sys.stderr)
        return 84

    agg = aggregate(per_seed)

    out = Path(args.out) if args.out else base / "summary_aggregated.json"
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print("=" * 70)
    print(f"  T3.1 — multi-seed aggregation ({agg['n_seeds']} seeds)")
    print("=" * 70)
    print(f"  {'seed':>4}  {'mean_return_100':>16}  {'solved':>7}  {'first_solve_step':>16}")
    for r in agg["per_seed"]:
        print(
            f"  {r['seed']:>4}  {r['mean_return_last_100']:>16.2f}  "
            f"{str(r['solved']):>7}  {str(r['first_solve_step']):>16}"
        )
    rm = agg["mean_return_last_100"]
    print("-" * 70)
    print(
        f"  mean_return_last_100: {rm['mean']:.2f}  "
        f"CI95 [{rm['ci95'][0]:.2f}, {rm['ci95'][1]:.2f}]"
    )
    print(f"  solved: {agg['n_solved']}/{agg['n_seeds']}  "
          f"(criterion ≥4/5: {'PASS' if agg['criterion_met'] else 'FAIL'})")
    print(f"  written: {out}")
    print("-" * 70)
    return 0 if agg["criterion_met"] else 84


if __name__ == "__main__":
    sys.exit(main())
