from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import mean_ci95  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


def binned_means(steps: list[float], returns: list[float], edges: list[float]) -> list[float]:
    n_bins = len(edges) - 1
    sums = [0.0] * n_bins
    counts = [0] * n_bins
    for st, ret in zip(steps, returns):
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if (lo <= st < hi) or (i == n_bins - 1 and st == hi):
                sums[i] += ret
                counts[i] += 1
                break
    return [
        (sums[i] / counts[i]) if counts[i] > 0 else float("nan")
        for i in range(n_bins)
    ]


def aggregate_curves(per_seed_binned: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    n_bins = len(per_seed_binned[0])
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for b in range(n_bins):
        vals = [
            seed[b]
            for seed in per_seed_binned
            if not math.isnan(seed[b])
        ]
        if vals:
            m, lo, hi = mean_ci95(vals)
        else:
            m = lo = hi = float("nan")
        means.append(m)
        los.append(lo)
        his.append(hi)
    return means, los, his


def _load_curve(
    base: Path, seeds: list[int], edges: list[float]
) -> list[list[float]]:
    per_seed: list[list[float]] = []
    for s in seeds:
        csv_path = base / str(s) / "metrics.csv"
        steps: list[float] = []
        rets: list[float] = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                steps.append(float(row["global_step"]))
                rets.append(float(row["return"]))
        per_seed.append(binned_means(steps, rets, edges))
    return per_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired returns plot: ablation vs baseline (T3.3)")
    p.add_argument("--baseline", default="runs/dqn_baseline")
    p.add_argument("--ablation", default="runs/dqn_buffer10k")
    p.add_argument("--label-baseline", default="baseline (buffer 100k)")
    p.add_argument("--label-ablation", default="ablation (buffer 10k)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--total-steps", type=int, default=500000)
    p.add_argument("--bins", type=int, default=25)
    p.add_argument("--out", default="assets/plots/paired_buffer10k_vs_baseline.png")
    return p.parse_args()


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def main() -> int:
    args = parse_args()
    edges = [i * args.total_steps / args.bins for i in range(args.bins + 1)]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(args.bins)]

    series = [
        (_abs(args.baseline), args.label_baseline, "steelblue"),
        (_abs(args.ablation), args.label_ablation, "darkorange"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for base, label, color in series:
        if not base.exists():
            print(f"ERROR: run dir not found: {base}", file=sys.stderr)
            return 84
        per_seed = _load_curve(base, args.seeds, edges)
        means, los, his = aggregate_curves(per_seed)
        ax.plot(centers, means, color=color, linewidth=1.8, label=label)
        ax.fill_between(centers, los, his, color=color, alpha=0.2)

    ax.axhline(200.0, color="green", linestyle="--", linewidth=0.8, label="solved (200)")
    ax.set_xlabel("training step")
    ax.set_ylabel("episodic return (mean ± 95% CI, 5 seeds)")
    ax.set_title("Paired training curves — replay buffer ablation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    out = _abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
