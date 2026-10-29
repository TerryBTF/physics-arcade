"""Pool rendering: flat (one color per class, source of ground-truth masks) and cartoon."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sim import BALL_R, HEIGHT, POCKET_DRAW_R, POCKETS, TABLE_X0, TABLE_X1, TABLE_Y0, TABLE_Y1, WIDTH, Table

PALETTE: dict[str, tuple[int, int, int]] = {
    "felt": (50, 130, 80),
    "rail": (110, 70, 40),
    "pocket": (20, 20, 20),
    "cue": (240, 240, 230),
    "ball": (220, 60, 40),
}
DYNAMIC_CLASSES = ["cue", "ball"]
LUT_PATH = Path(__file__).with_name("cartoon_lut.npy")
SS = 3

# Cartoon colors of object balls 1..9: no greens (felt), no white (cue), no browns (rail).
BALL_COLORS = [
    (240, 200, 40), (40, 80, 200), (210, 40, 40), (120, 50, 160), (240, 130, 30),
    (130, 30, 40), (230, 110, 160), (90, 170, 230), (200, 170, 90),
]
RAIL = 22  # rail width in px (outside the playing surface)


def _flip(x: float, y: float, s: float = 1.0) -> tuple[float, float]:
    return x * s, (HEIGHT - y) * s


def _circle(d: ImageDraw.ImageDraw, x: float, y: float, r: float, fill, s: float = 1.0) -> None:
    cx, cy = _flip(x, y, s)
    d.ellipse([cx - r * s, cy - r * s, cx + r * s, cy + r * s], fill=fill)


def render(table: Table) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), PALETTE["rail"])
    d = ImageDraw.Draw(img)
    d.rectangle([*_flip(TABLE_X0, TABLE_Y1), *_flip(TABLE_X1, TABLE_Y0)], fill=PALETTE["felt"])
    for px, py in POCKETS:
        _circle(d, px, py, POCKET_DRAW_R, PALETTE["pocket"])
    for b in table.balls:
        if not b.potted:
            _circle(d, *b.body.position, BALL_R, PALETTE["cue" if b.number == 0 else "ball"])
    return img


@lru_cache(maxsize=1)
def _cartoon_background() -> Image.Image:
    s = SS
    img = Image.new("RGB", (WIDTH * s, HEIGHT * s), (34, 30, 40))  # room behind the table
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([*_flip(TABLE_X0 - RAIL, TABLE_Y1 + RAIL, s), *_flip(TABLE_X1 + RAIL, TABLE_Y0 - RAIL, s)], radius=14 * s, fill=(110, 70, 40))
    d.rounded_rectangle([*_flip(TABLE_X0 - RAIL + 4, TABLE_Y1 + RAIL - 4, s), *_flip(TABLE_X1 + RAIL - 4, TABLE_Y0 - RAIL + 4, s)], radius=11 * s, fill=(132, 86, 50))
    felt = Image.new("RGB", img.size, (0, 0, 0))
    fd = ImageDraw.Draw(felt)
    fd.rectangle([*_flip(TABLE_X0, TABLE_Y1, s), *_flip(TABLE_X1, TABLE_Y0, s)], fill=(46, 126, 78))
    rng = np.random.default_rng(3)
    for _ in range(900):
        x, y = rng.uniform(TABLE_X0 + 2, TABLE_X1 - 2), rng.uniform(TABLE_Y0 + 2, TABLE_Y1 - 2)
        r = rng.uniform(0.6, 1.4)
        _circle(fd, x, y, r, (40, 116, 70), s)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle([*_flip(TABLE_X0, TABLE_Y1, s), *_flip(TABLE_X1, TABLE_Y0, s)], fill=255)
    img.paste(felt, (0, 0), mask)
    # Diamond sights on the rails.
    for i in range(1, 8):
        x = TABLE_X0 + (TABLE_X1 - TABLE_X0) * i / 8
        for y in (TABLE_Y0 - RAIL / 2, TABLE_Y1 + RAIL / 2):
            cx, cy = _flip(x, y, s)
            d.polygon([(cx - 3 * s, cy), (cx, cy - 2 * s), (cx + 3 * s, cy), (cx, cy + 2 * s)], fill=(225, 205, 160))
    for px, py in POCKETS:
        _circle(d, px, py, POCKET_DRAW_R + 2, (70, 50, 30), s)
        _circle(d, px, py, POCKET_DRAW_R, (14, 12, 12), s)
    return img


def _shade(c: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(int(min(255, max(0, v * f))) for v in c)


def render_cartoon(table: Table) -> Image.Image:
    s = SS
    img = _cartoon_background().copy()
    d = ImageDraw.Draw(img)
    for b in table.balls:
        if b.potted:
            continue
        x, y = b.body.position
        _circle(d, x + 1.5, y - 1.5, BALL_R, (30, 90, 56), s)  # shadow on the felt
        if b.number == 0:
            base, dark, light = (245, 245, 235), (205, 205, 195), (255, 255, 250)
        else:
            base = BALL_COLORS[(b.number - 1) % len(BALL_COLORS)]
            dark, light = _shade(base, 0.72), _shade(base, 1.25)
        _circle(d, x, y, BALL_R, dark, s)
        _circle(d, x + 0.6, y + 0.6, BALL_R - 1.4, base, s)
        _circle(d, x + 2.5, y + 2.5, 2.0, light, s)
    return img.resize((WIDTH, HEIGHT), Image.LANCZOS)


def label_balls(img: Image.Image, table: Table) -> Image.Image:
    """Copy of `img` with each object ball's number written next to it."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=13)
    for b in table.object_balls():
        x, y = _flip(*b.body.position)
        text = str(b.number)
        box = d.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        tx, ty = x - tw / 2, y - BALL_R - th - 7
        d.rectangle([tx - 3, ty - 2, tx + tw + 3, ty + th + 4], fill=(255, 255, 255), outline=(0, 0, 0))
        d.text((tx, ty - box[1] + 1), text, fill=(0, 0, 0), font=font)
    return img
