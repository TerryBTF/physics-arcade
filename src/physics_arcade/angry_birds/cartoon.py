"""Cartoon renderer: textured, game-like frames drawn from the same world state as
the flat renderer. All artwork is original and procedural.

Ground-truth class masks always come from the flat renderer. Each class keeps its
own hue family (outlines and textures are darker or lighter shades of the base
color), so a color lookup table fitted on rendered frames (see `fit_lut`) can
still map cartoon pixels back to classes for scoring.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .sim import GROUND_Y, HEIGHT, SLING_POS, WIDTH, World

SS = 3  # supersampling factor; frames are drawn at SS x size and downscaled for anti-aliasing
LUT_PATH = Path(__file__).with_name("cartoon_lut.npy")

COLORS = {
    "wood": {"base": (196, 136, 66), "grain": (166, 108, 46), "edge": (122, 76, 32)},
    "stone": {"base": (142, 142, 152), "speck": (112, 112, 124), "edge": (86, 86, 98)},
    "ice": {"base": (176, 226, 246), "shine": (222, 246, 255), "edge": (110, 180, 216)},
    "pig": {
        "base": (128, 214, 62),
        "edge": (58, 122, 30),
        "snout": (172, 232, 112),
        "nostril": (70, 132, 36),
        "eye": (228, 246, 216),
        "pupil": (24, 52, 14),
    },
    "bird": {
        "base": (214, 34, 38),
        "edge": (120, 14, 18),
        "belly": (242, 214, 184),
        "beak": (246, 176, 34),
        "eye": (250, 242, 236),
        "pupil": (30, 10, 10),
    },
}


def _px(x: float, y: float) -> tuple[float, float]:
    """World coordinates (y up) to supersampled image coordinates (y down)."""
    return x * SS, (HEIGHT - y) * SS


@lru_cache(maxsize=1)
def _background() -> Image.Image:
    w, h = WIDTH * SS, HEIGHT * SS
    gy = (HEIGHT - GROUND_Y) * SS
    top, bottom = np.array([96, 168, 232]), np.array([196, 228, 248])
    t = np.linspace(0, 1, h)[:, None]
    sky = (top * (1 - t) + bottom * t).astype(np.uint8)
    img = Image.fromarray(np.repeat(sky[:, None, :], w, axis=1))
    d = ImageDraw.Draw(img)

    # Clouds: clusters of white circles.
    for cx, cy, s in [(70, 300, 1.0), (260, 320, 0.8), (430, 290, 1.1), (590, 330, 0.7)]:
        for dx, dy, r in [(-18, 0, 12), (0, 6, 16), (18, 0, 12), (8, -4, 11), (-8, -4, 10)]:
            x, y = _px(cx + dx * s, cy + dy * s)
            rr = r * s * SS
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(250, 252, 252))

    # Distant hills in a muted blue-green.
    hills = [(0, GROUND_Y)]
    for x in range(0, WIDTH + 20, 10):
        hills.append((x, GROUND_Y + 34 + 18 * math.sin(x / 70) + 10 * math.sin(x / 23 + 1)))
    hills.append((WIDTH + 20, GROUND_Y))
    d.polygon([_px(*p) for p in hills], fill=(138, 190, 176))

    # Ground: dirt with a grass strip and pebbles.
    d.rectangle([0, gy, w, h], fill=(128, 98, 82))
    rng = np.random.default_rng(7)
    for _ in range(120):
        x, y = rng.uniform(0, WIDTH), rng.uniform(2, GROUND_Y - 10)
        r = rng.uniform(1.0, 2.2) * SS
        px, py = _px(x, y)
        d.ellipse([px - r, py - r, px + r, py + r], fill=(104, 80, 68))
    d.rectangle([0, gy, w, gy + 7 * SS], fill=(66, 150, 84))
    for x in range(0, WIDTH, 4):
        tip = rng.uniform(2, 5)
        d.polygon([_px(x, GROUND_Y), _px(x + 2, GROUND_Y + tip), _px(x + 4, GROUND_Y)], fill=(66, 150, 84))

    # Slingshot: a Y-shaped wooden fork.
    sx, sy = SLING_POS
    fork, band = (78, 52, 36), (52, 32, 26)
    d.polygon([_px(sx - 4, GROUND_Y), _px(sx + 4, GROUND_Y), _px(sx + 3, sy - 22), _px(sx - 3, sy - 22)], fill=fork)
    d.line([_px(sx, sy - 22), _px(sx - 9, sy + 4)], fill=fork, width=5 * SS)
    d.line([_px(sx, sy - 22), _px(sx + 8, sy + 4)], fill=fork, width=5 * SS)
    d.line([_px(sx - 9, sy + 2), _px(sx + 8, sy + 2)], fill=band, width=2 * SS)
    return img


def _local(obj, pts):
    return [_px(*obj.body.local_to_world(p)) for p in pts]


def _block(d: ImageDraw.ImageDraw, obj) -> None:
    c = COLORS[obj.kind]
    w, h = obj.size
    hw, hh = w / 2, h / 2
    d.polygon(_local(obj, [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]), fill=c["edge"])
    inset = 1.4
    d.polygon(_local(obj, [(-hw + inset, -hh + inset), (hw - inset, -hh + inset), (hw - inset, hh - inset), (-hw + inset, hh - inset)]), fill=c["base"])
    if obj.kind == "wood":
        # Grain lines along the long axis.
        if w >= h:
            for f in (-0.3, 0.05, 0.35):
                d.line(_local(obj, [(-hw + 3, f * h), (hw - 3, f * h)]), fill=c["grain"], width=SS)
        else:
            for f in (-0.3, 0.05, 0.35):
                d.line(_local(obj, [(f * w, -hh + 3), (f * w, hh - 3)]), fill=c["grain"], width=SS)
    elif obj.kind == "stone":
        rng = np.random.default_rng(int(w * 1000 + h * 10))
        for _ in range(max(2, int(w * h / 60))):
            x, y = rng.uniform(-hw + 3, hw - 3), rng.uniform(-hh + 3, hh - 3)
            r = rng.uniform(0.8, 1.6)
            d.polygon(_local(obj, [(x - r, y), (x, y + r), (x + r, y), (x, y - r)]), fill=c["speck"])
    else:  # ice: a diagonal highlight stripe
        s = min(w, h) * 0.18
        d.polygon(_local(obj, [(-hw + 2, hh - 2 - 2 * s), (-hw + 2 + 2 * s, hh - 2), (-hw + 2 + 3 * s, hh - 2), (-hw + 2, hh - 2 - 3 * s)]), fill=c["shine"])


def _circle(d, center, r, fill):
    x, y = center
    d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def _pig(d: ImageDraw.ImageDraw, obj) -> None:
    c = COLORS["pig"]
    r = obj.size[0]
    _circle(d, _px(*obj.body.position), r * SS, c["edge"])
    _circle(d, _px(*obj.body.position), (r - 1.3) * SS, c["base"])
    at = lambda x, y: _px(*obj.body.local_to_world((x * r, y * r)))  # noqa: E731
    _circle(d, at(0.0, -0.1), 0.38 * r * SS, c["snout"])
    _circle(d, at(-0.13, -0.1), 0.09 * r * SS, c["nostril"])
    _circle(d, at(0.13, -0.1), 0.09 * r * SS, c["nostril"])
    for sx in (-0.42, 0.42):
        _circle(d, at(sx, 0.35), 0.2 * r * SS, c["eye"])
        _circle(d, at(sx, 0.33), 0.09 * r * SS, c["pupil"])


def _bird(d: ImageDraw.ImageDraw, obj) -> None:
    c = COLORS["bird"]
    r = obj.size[0]
    # Face along the velocity so the bird looks where it flies.
    v = obj.body.velocity
    ang = math.atan2(v.y, v.x) if v.length > 20 else 0.0
    cx, cy = obj.body.position

    def at(x: float, y: float) -> tuple[float, float]:
        ca, sa = math.cos(ang), math.sin(ang)
        return _px(cx + (x * ca - y * sa) * r, cy + (x * sa + y * ca) * r)

    _circle(d, _px(cx, cy), r * SS, c["edge"])
    _circle(d, _px(cx, cy), (r - 1.2) * SS, c["base"])
    _circle(d, at(0.1, -0.45), 0.45 * r * SS, c["belly"])
    d.polygon([at(0.75, 0.15), at(1.3, -0.05), at(0.75, -0.3)], fill=c["beak"])
    _circle(d, at(0.42, 0.28), 0.22 * r * SS, c["eye"])
    _circle(d, at(0.5, 0.26), 0.1 * r * SS, c["pupil"])
    d.line([at(0.15, 0.55), at(0.7, 0.42)], fill=c["edge"], width=max(1, int(0.18 * r * SS)))


def render_cartoon(world: World) -> Image.Image:
    img = _background().copy()
    d = ImageDraw.Draw(img)
    for o in world.objects:
        if not o.alive:
            continue
        if o.kind == "pig":
            _pig(d, o)
        elif o.kind == "bird":
            _bird(d, o)
        else:
            _block(d, o)
    return img.resize((WIDTH, HEIGHT), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Segmentation of cartoon frames


def fit_lut(frames: np.ndarray, masks: np.ndarray, n_classes: int) -> np.ndarray:
    """Learn a 6-bit-per-channel color -> class table from (frames, masks) pairs.

    Each populated color bin takes its majority class; empty bins take the class
    of the nearest populated bin so unseen colors (e.g. from a generative model)
    still get a sensible label.
    """
    from scipy.spatial import cKDTree

    q = frames.reshape(-1, 3) >> 2
    idx = (q[:, 0].astype(np.int64) << 12) | (q[:, 1].astype(np.int64) << 6) | q[:, 2]
    counts = np.zeros((64**3, n_classes), dtype=np.int64)
    np.add.at(counts, (idx, masks.reshape(-1).astype(np.int64)), 1)
    seen = counts.sum(1) > 0
    lut = counts.argmax(1).astype(np.uint8)
    grid = np.stack(np.unravel_index(np.arange(64**3), (64, 64, 64)), axis=1)
    _, nn = cKDTree(grid[seen]).query(grid[~seen])
    lut[~seen] = lut[seen][nn]
    return lut


@lru_cache(maxsize=1)
def cartoon_lut() -> np.ndarray:
    if not LUT_PATH.exists():
        raise FileNotFoundError(f"{LUT_PATH} missing; run `physics-arcade fit-cartoon-lut`")
    return np.load(LUT_PATH)


def build_lut(n_episodes: int = 60, seed_offset: int = 900_000, workers: int = 8) -> dict:
    """Fit the cartoon LUT on held-out seeds and save it next to this module.

    Frames are round-tripped through the same H.264 settings the dataset uses, so
    codec color bleeding at edges is part of the training data.
    """
    from concurrent.futures import ProcessPoolExecutor

    from .render import CLASSES

    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_lut_samples, range(seed_offset, seed_offset + n_episodes)))
    frames = np.concatenate([p[0] for p in parts])
    masks = np.concatenate([p[1] for p in parts])
    lut = fit_lut(frames, masks, len(CLASSES))
    np.save(LUT_PATH, lut)
    q = frames >> 2
    pred = lut[(q[..., 0].astype(np.int64) << 12) | (q[..., 1].astype(np.int64) << 6) | q[..., 2]]
    return {"frames": len(frames), "pixel_accuracy_on_fit_set": float((pred == masks).mean()), "out": str(LUT_PATH)}


def _lut_samples(seed: int) -> tuple[np.ndarray, np.ndarray]:
    import io

    import imageio.v3 as iio

    from .episode import generate_episode
    from .render import segment
    from .sim import FPS

    ep = generate_episode(seed, "cartoon")
    buf = io.BytesIO()
    iio.imwrite(buf, np.stack(ep.display_frames), extension=".mp4", fps=FPS, codec="libx264", quality=9, macro_block_size=1)
    decoded = iio.imread(buf.getvalue(), extension=".mp4")
    sel = slice(0, None, 6)
    raw = np.stack(ep.display_frames[sel])
    masks = np.stack([segment(f) for f in ep.frames[sel]])
    return np.concatenate([raw, decoded[sel]]), np.concatenate([masks, masks])
