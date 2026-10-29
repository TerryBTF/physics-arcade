"""Domino rendering: flat (one color per class, source of ground-truth masks) and cartoon."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sim import BALL_R, FLOOR_Y, HEIGHT, WIDTH, Scene

PALETTE: dict[str, tuple[int, int, int]] = {
    "wall": (215, 235, 225),
    "floor": (150, 140, 130),
    "platform": (170, 85, 60),
    "ball": (60, 110, 220),
    "light": (245, 238, 215),
    "heavy": (40, 50, 90),
}
DYNAMIC_CLASSES = ["ball", "light", "heavy"]
LUT_PATH = Path(__file__).with_name("cartoon_lut.npy")
SS = 3
CARTOON = {
    "light": {"base": (245, 238, 215), "edge": (200, 190, 165), "pip": (130, 40, 40)},
    "heavy": {"base": (40, 50, 90), "edge": (25, 30, 60), "pip": (230, 190, 60)},
}


def _flip(x: float, y: float, s: float = 1.0) -> tuple[float, float]:
    return x * s, (HEIGHT - y) * s


def _poly(scene_obj, s: float = 1.0):
    return [_flip(*scene_obj.body.local_to_world(v), s) for v in scene_obj.shape.get_vertices()]


def render(scene: Scene) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), PALETTE["wall"])
    d = ImageDraw.Draw(img)
    d.rectangle([*_flip(0, FLOOR_Y), *_flip(WIDTH, 0)], fill=PALETTE["floor"])
    for x0, x1, top in scene.platforms:
        d.rectangle([*_flip(x0, top), *_flip(x1, FLOOR_Y)], fill=PALETTE["platform"])
    for dm in scene.dominoes:
        d.polygon(_poly(dm), fill=PALETTE[dm.material])
    if scene.ball is not None:
        x, y = _flip(*scene.ball.body.position)
        d.ellipse([x - BALL_R, y - BALL_R, x + BALL_R, y + BALL_R], fill=PALETTE["ball"])
    return img


@lru_cache(maxsize=8)
def _cartoon_background(platforms: tuple[tuple[float, float, float], ...]) -> Image.Image:
    s = SS
    img = Image.new("RGB", (WIDTH * s, HEIGHT * s), (215, 235, 225))
    d = ImageDraw.Draw(img)
    for x in range(0, WIDTH, 24):  # wallpaper stripes
        d.rectangle([*_flip(x, HEIGHT, s), *_flip(x + 10, FLOOR_Y, s)], fill=(205, 228, 216))
    d.rectangle([*_flip(0, FLOOR_Y + 8, s), *_flip(WIDTH, FLOOR_Y, s)], fill=(185, 210, 198))  # skirting board
    d.rectangle([*_flip(0, FLOOR_Y, s), *_flip(WIDTH, 0, s)], fill=(150, 140, 130))
    for y in range(int(FLOOR_Y) - 12, 0, -12):  # floorboards
        d.line([_flip(0, y, s), _flip(WIDTH, y, s)], fill=(130, 120, 112), width=s)
    rng = np.random.default_rng(5)
    for y in range(int(FLOOR_Y) - 12, -12, -12):
        for x in rng.uniform(0, WIDTH, 4):
            d.line([_flip(x, y, s), _flip(x, y + 12, s)], fill=(130, 120, 112), width=s)
    for x0, x1, top in platforms:  # brick steps
        d.rectangle([*_flip(x0, top, s), *_flip(x1, FLOOR_Y, s)], fill=(190, 170, 150))
        for row, y in enumerate(np.arange(FLOOR_Y, top, 8.0)):
            y_top = min(y + 7, top - 1)
            if y_top - (y + 1) < 1:  # no room left for a brick in the last row
                continue
            off = 0 if row % 2 else 9
            for bx in np.arange(x0 - off, x1, 18.0):
                a, b = max(bx + 1, x0 + 1), min(bx + 17, x1 - 1)
                if b > a:
                    d.rectangle([*_flip(a, y_top, s), *_flip(b, y + 1, s)], fill=(170, 85, 60))
    return img


def render_cartoon(scene: Scene) -> Image.Image:
    s = SS
    img = _cartoon_background(tuple(scene.platforms)).copy()
    d = ImageDraw.Draw(img)
    for dm in scene.dominoes:
        c = CARTOON[dm.material]
        w, h = dm.size

        def box(x0, y0, x1, y1, fill):
            d.polygon([_flip(*dm.body.local_to_world(p), s) for p in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]], fill=fill)

        box(-w / 2, -h / 2, w / 2, h / 2, c["edge"])
        box(-w / 2 + 1, -h / 2 + 1, w / 2 - 1, h / 2 - 1, c["base"])
        box(-w / 2 + 1, -0.5, w / 2 - 1, 0.5, c["edge"])  # centre line
        for py in (-0.3 * h, 0.3 * h):
            x, y = _flip(*dm.body.local_to_world((0, py)), s)
            r = 1.2 * s
            d.ellipse([x - r, y - r, x + r, y + r], fill=c["pip"])
    if scene.ball is not None:
        bx, by = scene.ball.body.position
        for off, rr, col in ((0, BALL_R, (35, 70, 160)), (0.5, BALL_R - 1.2, (60, 110, 220)), (2.2, 2.0, (140, 180, 245))):
            x, y = _flip(bx - off, by + off, s)
            d.ellipse([x - rr * s, y - rr * s, x + rr * s, y + rr * s], fill=col)
    return img.resize((WIDTH, HEIGHT), Image.LANCZOS)


def label_dominoes(img: Image.Image, scene: Scene) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12)
    for dm in scene.dominoes:
        x, top = dm.body.position.x, dm.body.position.y + dm.size[1] / 2
        fx, fy = _flip(x, top)
        text = str(dm.number)
        box = d.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        tx, ty = fx - tw / 2, fy - th - 8
        d.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 4], fill=(255, 255, 255), outline=(0, 0, 0))
        d.text((tx, ty - box[1] + 1), text, fill=(0, 0, 0), font=font)
    return img
