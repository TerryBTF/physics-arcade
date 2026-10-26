"""Mapping between the benchmark's 30 fps timeline and Cosmos's 16 fps, 93-frame window.

Kept free of torch so the dry run (dry_run.py) can exercise it with the benchmark's
own environment.
"""

from __future__ import annotations

import numpy as np

BENCH_FPS, COSMOS_FPS = 30, 16
NUM_FRAMES, NUM_COND = 93, 5


def cond_indices(n_ctx: int) -> list[int]:
    """Context frames closest to a 16 fps grid ending at the last context frame."""
    last = n_ctx - 1
    return [int(np.clip(round(last - (NUM_COND - 1 - i) * BENCH_FPS / COSMOS_FPS), 0, last)) for i in range(NUM_COND)]


def cosmos_frame_times(n_ctx: int) -> np.ndarray:
    """Time (s) of each of the 93 Cosmos frames on the benchmark timeline."""
    last_t = (n_ctx - 1) / BENCH_FPS
    return last_t + (np.arange(NUM_FRAMES) - (NUM_COND - 1)) / COSMOS_FPS


def future_to_cosmos_index(n_ctx: int, n_future: int) -> np.ndarray:
    """Index into the 93 generated frames for each scored future frame."""
    last = n_ctx - 1
    t = (np.arange(n_ctx, n_ctx + n_future) - last) / BENCH_FPS  # seconds after last context frame
    return np.clip(np.round(NUM_COND - 1 + t * COSMOS_FPS), 0, NUM_FRAMES - 1).astype(int)


def to_benchmark_frames(frames: list, n_ctx: int, n_future: int, size: tuple[int, int] = (640, 360)) -> np.ndarray:
    """93 generated PIL frames -> (n_future, H, W, 3) uint8 at the benchmark's size and frame rate."""
    from PIL import Image

    return np.stack([np.asarray(frames[k].convert("RGB").resize(size, Image.BICUBIC)) for k in future_to_cosmos_index(n_ctx, n_future)])
