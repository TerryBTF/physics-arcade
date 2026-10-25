"""Flat-color renderer. Every object class has exactly one color so frames can be
segmented back into class masks for automatic evaluation."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sim import GROUND_Y, HEIGHT, SLING_POS, WIDTH, World

PALETTE: dict[str, tuple[int, int, int]] = {
    "sky": (150, 200, 240),
    "ground": (90, 140, 60),
    "sling": (60, 40, 20),
    "bird": (220, 30, 30),
    "pig": (120, 230, 60),
    "wood": (190, 130, 60),
    "stone": (120, 120, 130),
    "ice": (200, 240, 255),
}
CLASSES = list(PALETTE)
DYNAMIC_CLASSES = ["bird", "pig", "wood", "stone", "ice"]


def _flip(x: float, y: float) -> tuple[float, float]:
    return x, HEIGHT - y


def render(world: World) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), PALETTE["sky"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, HEIGHT - GROUND_Y, WIDTH, HEIGHT], fill=PALETTE["ground"])
    sx, sy = SLING_POS
    d.rectangle([*_flip(sx - 3, sy - 12), *_flip(sx + 3, GROUND_Y)], fill=PALETTE["sling"])
    for o in world.objects:
        if not o.alive:
            continue
        color = PALETTE[o.kind]
        if o.kind in ("pig", "bird"):
            x, y = _flip(*o.body.position)
            r = o.size[0]
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)
        else:
            pts = [_flip(*o.body.local_to_world(v)) for v in o.shape.get_vertices()]
            d.polygon(pts, fill=color)
    return img


def label_pigs(img: Image.Image, world: World) -> Image.Image:
    """Copy of `img` with each pig's id written above it (for the QA prompt)."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=14)
    for p in world.pigs:
        if not p.alive:
            continue
        x, y = _flip(*p.body.position)
        r = p.size[0]
        text = str(p.pig_id)
        box = d.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        tx, ty = x - tw / 2, y - r - th - 8
        d.rectangle([tx - 3, ty - 2, tx + tw + 3, ty + th + 5], fill=(255, 255, 255), outline=(0, 0, 0))
        d.text((tx, ty - box[1] + 1), text, fill=(0, 0, 0), font=font)
    return img


def segment(frame: np.ndarray) -> np.ndarray:
    """Map an RGB frame (H, W, 3) to class indices (H, W) by nearest palette color."""
    f = frame[..., :3] >> 2
    return _LUT[(f[..., 0].astype(np.int32) << 12) | (f[..., 1].astype(np.int32) << 6) | f[..., 2]]


def _build_lut() -> np.ndarray:
    """Nearest palette class for every color quantized to 6 bits per channel."""
    pal = np.array([PALETTE[c] for c in CLASSES], dtype=np.int32)
    q = np.arange(64, dtype=np.int32) * 4 + 2
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 1, 3)
    return ((grid - pal) ** 2).sum(-1).argmin(-1).astype(np.uint8)


_LUT = _build_lut()
