from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as iio
import numpy as np


DEFAULT_FPS: int = 30


def save_video(frames: Iterable[np.ndarray], path: Path | str, *, fps: int = DEFAULT_FPS) -> Path:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    out_path = Path(path)
    if out_path.suffix == "":
        raise ValueError(f"path must have a file extension (e.g. .mp4), got {out_path!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_list = list(frames)
    if not frames_list:
        raise ValueError("frames is empty; cannot encode a zero-frame video.")

    first = np.asarray(frames_list[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"frames must have shape (H, W, 3), got {first.shape}")
    h, w = int(first.shape[0]), int(first.shape[1])

    prepared: list[np.ndarray] = []
    for idx, frame in enumerate(frames_list):
        arr = np.asarray(frame)
        if arr.shape != (h, w, 3):
            raise ValueError(
                f"frame {idx} has shape {arr.shape}, expected ({h}, {w}, 3)"
            )
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        prepared.append(arr)

    iio.mimwrite(str(out_path), prepared, fps=fps, macro_block_size=1)
    return out_path
