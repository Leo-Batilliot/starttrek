from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.registry import AGENTS  # noqa: E402
from src.env.factory import build_env  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeding import seed_all  # noqa: E402
from src.utils.termination import classify  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


def _abs(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Defense demo: evaluate the backed-up best agent")
    p.add_argument("--checkpoint", default="assets/checkpoints/dqn_best.pt")
    p.add_argument("--config", default="configs/baseline/dqn.yaml")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--record", default=None, help="Write an MP4 to this path (rgb_array)")
    p.add_argument("--success-only", action="store_true",
                   help="With --record: keep only landed episodes in the video")
    p.add_argument("--max-seconds", type=float, default=30.0,
                   help="With --record: cap the video duration")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    cfg_path = _abs(a.config)
    ckpt = _abs(a.checkpoint)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 84
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}\n"
              f"       (expected the committed backup; see assets/checkpoints/)",
              file=sys.stderr)
        return 84

    cfg = load_config(cfg_path)
    seed_all(seed=a.seed)
    record = a.record is not None
    env = build_env(cfg.env, seed=a.seed, render_mode="rgb_array" if record else None)
    fps = int(env.metadata.get("render_fps", 50)) if hasattr(env, "metadata") else 50

    agent = AGENTS[cfg.agent.name](cfg.agent, n_actions=env.action_space.n)
    agent.load(ckpt)

    print("=" * 64)
    print("  DEFENSE DEMO — backed-up best DQN agent")
    print(f"  checkpoint : {ckpt.relative_to(_ROOT) if ckpt.is_relative_to(_ROOT) else ckpt}")
    print(f"  config     : {cfg.agent.name} / {cfg.env.id} / hidden={list(cfg.agent.network.hidden_sizes)}")
    print(f"  episodes   : {a.episodes}  greedy={a.greedy}  seed={a.seed}")
    print("=" * 64)

    returns: list[float] = []
    kept_frames: list[np.ndarray] = []
    max_frames = int(a.max_seconds * fps)
    t0 = time.time()

    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + ep)
        done = False
        ep_ret = 0.0
        ep_frames: list[np.ndarray] = []
        last_obs, last_info, last_r, term, trunc = obs, {}, 0.0, False, False
        while not done:
            if record:
                f = env.render()
                if f is not None:
                    ep_frames.append(np.asarray(f))
            action = agent.act(obs, greedy=a.greedy)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += float(r)
            last_obs, last_info, last_r = obs, info, float(r)
            done = term or trunc
        reason = classify(last_info, last_r, last_obs, bool(term), bool(trunc)).value
        returns.append(ep_ret)
        landed = reason == "land"
        print(f"  ep {ep:>2} | return {ep_ret:8.2f} | {reason:<11}"
              f" | {'✓ landed' if landed else ''}")
        if record and (landed or not a.success_only):
            if len(kept_frames) + len(ep_frames) <= max_frames:
                kept_frames.extend(ep_frames)

    env.close()
    arr = np.asarray(returns, dtype=np.float64)
    print("-" * 64)
    print(f"  episodes {len(returns)} | mean {arr.mean():8.2f} | "
          f"min {arr.min():.1f} | max {arr.max():.1f} | "
          f"≥200: {int((arr >= 200).sum())}/{len(returns)}")
    print(f"  wall time : {time.time() - t0:.1f}s")

    if record:
        if not kept_frames:
            print("WARN: no frames captured (no qualifying episode)", file=sys.stderr)
            return 84
        from src.utils.video import save_video
        out = _abs(a.record)
        save_video(kept_frames, out, fps=fps)
        mb = out.stat().st_size / 1e6
        print(f"  video     : {out}  ({len(kept_frames)} frames, "
              f"{len(kept_frames)/fps:.1f}s, {mb:.2f} MB)")
        if mb > 10:
            print("WARN: video > 10 MB (T4.3 vise ≤ 10 MB)", file=sys.stderr)
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
