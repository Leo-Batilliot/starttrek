from __future__ import annotations

import os
import numpy as np
import torch
import random


def seed_all(*, seed: int) -> None:

    # Check seed
    if not isinstance(seed, int):
        raise ValueError(f"Seed must be int, got {seed}")
    if seed < 0:
        raise ValueError(f"Seed must be positive, got {seed}")

    # os var
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Random stdlib
    random.seed(seed)

    # Random numpy
    np.random.seed(seed)

    # Random PyTorch (CPU)
    torch.manual_seed(seed)

    # Random PyTorch (CUDA = all CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as _:
        pass