"""
    python scripts/comparative_video.py [--seed 1] [--out assets/videos/comparative.mp4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.registry import AGENTS  # noqa: E402
from src.env.factory import build_env  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeding import seed_all  # noqa: E402
from src.utils.termination import classify  # noqa: E402
from src.utils.video import save_video  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]

# (label, config, checkpoint relative to root or None)
_AGENTS = (
    ("Aleatoire", "configs/baseline/random.yaml", None),
    ("Heuristique", "configs/baseline/heuristic.yaml", None),
    ("DQN entraine", "configs/baseline/dqn.yaml", "assets/checkpoints/dqn_best.pt"),
)
_HEADER_H = 64


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else _ROOT / q


def _rollout(label: str, cfg_path: str, ckpt: str | None, seed: int):
    cfg = load_config(_abs(cfg_path))
    seed_all(seed=seed)
    env = build_env(cfg.env, seed=seed, render_mode="rgb_array")
    agent = AGENTS[cfg.agent.name](cfg.agent, n_actions=env.action_space.n)
    if ckpt is not None:
        agent.load(_abs(ckpt))
    obs, _ = env.reset(seed=seed)
    frames = [np.asarray(env.render())]
    ret, done = 0.0, False
    last = (obs, {}, 0.0, False, False)
    while not done:
        a = agent.act(obs, greedy=True)
        obs, r, term, trunc, info = env.step(a)
        ret += float(r)
        frames.append(np.asarray(env.render()))
        last = (obs, info, float(r), bool(term), bool(trunc))
        done = term or trunc
    reason = classify(last[1], last[2], last[0], last[3], last[4]).value
    fps = int(env.metadata.get("render_fps", 50))
    env.close()
    print(f"  {label:<14} return={ret:7.1f}  {reason:<11} ({len(frames)} frames)")
    return frames, ret, reason, fps


def _header(width: int, panels: list[tuple[str, float, str]]) -> np.ndarray:
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, _HEADER_H / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(panels)
    for i, (label, ret, reason) in enumerate(panels):
        cx = (i + 0.5) / n
        ax.text(cx, 0.62, label, color="white", fontsize=15, weight="bold",
                ha="center", va="center")
        ax.text(cx, 0.26, f"retour {ret:.0f}  ·  {reason}", color="#bbbbbb",
                fontsize=11, ha="center", va="center")
        if i:
            ax.axvline(i / n, color="#444444", lw=1)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparative rollout video (T5.10)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="assets/videos/comparative.mp4")
    p.add_argument("--max-seconds", type=float, default=24.0)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    rolls, fps = [], 50
    print("=" * 60)
    print("  Comparative rollout — random / heuristic / trained DQN")
    print("=" * 60)
    for label, cfg, ckpt in _AGENTS:
        ck = _abs(ckpt) if ckpt else None
        if ckpt and not ck.exists():
            print(f"ERROR: checkpoint not found: {ck}", file=sys.stderr)
            return 84
        frames, ret, reason, fps = _rollout(label, cfg, ckpt, a.seed)
        rolls.append((label, frames, ret, reason))

    h, w = rolls[0][1][0].shape[:2]
    cap = int(a.max_seconds * fps)
    T = min(max(len(f) for _, f, _, _ in rolls), cap)

    def at(frames, t):
        return frames[t] if t < len(frames) else frames[-1]

    header = _header(w * len(rolls),
                     [(lb, rt, rs) for lb, _, rt, rs in rolls])
    out_frames = []
    for t in range(T):
        strip = np.hstack([at(f, t) for _, f, _, _ in rolls])
        out_frames.append(np.vstack([header, strip]))

    out = _abs(a.out)
    save_video(out_frames, out, fps=fps)
    mb = out.stat().st_size / 1e6
    print("-" * 60)
    print(f"  wrote {out}  ({len(out_frames)} frames, "
          f"{len(out_frames)/fps:.1f}s, {mb:.2f} MB)")
    if mb > 10:
        print("WARN: > 10 MB", file=sys.stderr)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
