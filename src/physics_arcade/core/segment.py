"""Map rendered RGB frames back to class indices.

Flat frames use the nearest palette color. Cartoon frames use a 6-bit-per-channel
color -> class table fitted on held-out episodes (`fit_cartoon_lut`), trained on raw,
H.264-decoded and bicubic-resampled frames so codec bleeding and resampling ringing
at edges are part of the training data.
"""

from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import imageio.v3 as iio
import numpy as np
from PIL import Image

from .game import Game


def _index(frame: np.ndarray) -> np.ndarray:
    f = frame[..., :3] >> 2
    return (f[..., 0].astype(np.int64) << 12) | (f[..., 1].astype(np.int64) << 6) | f[..., 2]


@lru_cache(maxsize=None)
def _flat_lut(palette: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    pal = np.array(palette, dtype=np.int32)
    q = np.arange(64, dtype=np.int32) * 4 + 2
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 1, 3)
    return ((grid - pal) ** 2).sum(-1).argmin(-1).astype(np.uint8)


_cartoon_luts: dict[str, np.ndarray] = {}


def lut_for(game: Game, style: str) -> np.ndarray:
    if style == "flat":
        return _flat_lut(tuple(tuple(c) for c in game.palette.values()))
    if game.name not in _cartoon_luts:
        if not game.lut_path.exists():
            raise FileNotFoundError(f"{game.lut_path} missing; run `physics-arcade fit-cartoon-lut --game {game.name}`")
        _cartoon_luts[game.name] = np.load(game.lut_path)
    return _cartoon_luts[game.name]


def segment(game: Game, frame: np.ndarray, style: str = "flat") -> np.ndarray:
    """(H, W, 3) RGB -> (H, W) class indices in game.classes order."""
    return lut_for(game, style)[_index(frame)]


def _color_counts(frames: np.ndarray, masks: np.ndarray, n_classes: int) -> np.ndarray:
    flat = _index(frames).reshape(-1) * n_classes + masks.reshape(-1)
    return np.bincount(flat, minlength=64**3 * n_classes).reshape(64**3, n_classes)


def _lut_from_counts(counts: np.ndarray) -> np.ndarray:
    """Majority class per populated bin; empty bins copy the nearest populated bin, so
    unseen colors (e.g. from a generative model) still get a sensible label."""
    from scipy.spatial import cKDTree

    seen = counts.sum(1) > 0
    lut = counts.argmax(1).astype(np.uint8)
    grid = np.stack(np.unravel_index(np.arange(64**3), (64, 64, 64)), axis=1)
    _, nn = cKDTree(grid[seen]).query(grid[~seen])
    lut[~seen] = lut[seen][nn]
    return lut


def _lut_counts(args: tuple[str, int, str]) -> np.ndarray:
    from ..games import get_game

    name, seed, difficulty = args
    game = get_game(name)
    ep = game.generate(seed, difficulty, "cartoon")
    buf = io.BytesIO()
    iio.imwrite(buf, np.stack(ep.display_frames), extension=".mp4", fps=game.fps, codec="libx264", quality=9, macro_block_size=1)
    decoded = iio.imread(buf.getvalue(), extension=".mp4")
    sel = slice(0, None, 6)
    raw = np.stack(ep.display_frames[sel])
    masks = np.stack([segment(game, f) for f in ep.frames[sel]])
    h, w = raw.shape[1:3]
    resampled = np.stack(
        [np.asarray(Image.fromarray(f).resize((1280, 704), Image.BICUBIC).resize((w, h), Image.BICUBIC)) for f in decoded[sel]]
    )
    return _color_counts(np.concatenate([raw, decoded[sel], resampled]), np.concatenate([masks] * 3), len(game.classes))


def fit_cartoon_lut(game: Game, n_episodes: int = 60, seed_offset: int = 900_000, workers: int = 8) -> dict:
    """Fit on held-out seeds spread over all difficulties and save to game.lut_path."""
    jobs = [(game.name, seed_offset + i, game.difficulties[i % len(game.difficulties)]) for i in range(n_episodes)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        counts = sum(pool.map(_lut_counts, jobs))
    lut = _lut_from_counts(counts)
    np.save(game.lut_path, lut)
    _cartoon_luts.pop(game.name, None)
    correct = counts[np.arange(len(lut)), lut].sum()
    return {"pixels": int(counts.sum()), "pixel_accuracy_on_fit_set": float(correct / counts.sum()), "out": str(game.lut_path)}
