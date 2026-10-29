"""Cartoon renderer: textured, game-like frames drawn from the same world state as
the flat renderer. All artwork is original and procedural.

Ground-truth class masks always come from the flat renderer. Each class keeps its
own hue family (outlines and textures are darker or lighter shades of the base
color), so a color lookup table fitted on rendered frames
(physics_arcade.core.segment.fit_cartoon_lut) can map cartoon pixels back to classes.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .sim import GROUND_Y, HEIGHT, LAUNCH_POS, WIDTH, World

SS = 3  # supersampling factor; frames are drawn at SS x size and downscaled for anti-aliasing
LUT_PATH = Path(__file__).with_name("cartoon_lut.npy")

COLORS = {
    "wood": {"base": (196, 136, 66), "grain": (166, 108, 46), "edge": (122, 76, 32)},
    "stone": {"base": (142, 142, 152), "speck": (112, 112, 124), "edge": (86, 86, 98)},
    "ice": {"base": (176, 226, 246), "shine": (222, 246, 255), "edge": (110, 180, 216)},
    "robot": {
        "base": (245, 205, 40),
        "edge": (200, 160, 0),
        "screen": (35, 45, 70),
        "led": (130, 250, 150),
        "bolt": (215, 175, 25),
        "antenna": (70, 70, 80),
        "tip": (235, 70, 60),
    },
    "ball": {"base": (30, 100, 110), "edge": (16, 62, 70), "shine": (96, 164, 172)},
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

    # Catapult: a wheeled wooden frame with a throwing arm whose bucket sits at the launch point.
    sx, sy = LAUNCH_POS
    beam, dark, wheel = (78, 52, 36), (52, 32, 26), (60, 45, 35)
    d.polygon([_px(sx - 26, GROUND_Y + 8), _px(sx + 16, GROUND_Y + 8), _px(sx + 12, GROUND_Y + 15), _px(sx - 22, GROUND_Y + 15)], fill=beam)
    d.polygon([_px(sx - 8, GROUND_Y + 15), _px(sx - 2, GROUND_Y + 15), _px(sx - 4, GROUND_Y + 48), _px(sx - 9, GROUND_Y + 48)], fill=beam)
    for wx in (sx - 18, sx + 8):
        x, y = _px(wx, GROUND_Y + 7)
        rr = 7 * SS
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=wheel)
        d.ellipse([x - rr / 3, y - rr / 3, x + rr / 3, y + rr / 3], fill=dark)
    d.line([_px(sx - 6, GROUND_Y + 44), _px(sx, sy - 6)], fill=beam, width=4 * SS)
    d.chord([*_px(sx - 9, sy + 3), *_px(sx + 9, sy - 11)], 0, 180, fill=dark)
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


def _robot(d: ImageDraw.ImageDraw, obj) -> None:
    """Square yellow robot: dark screen face with two LED eyes, side bolts, short antenna."""
    c = COLORS["robot"]
    h = obj.size[0] / 2

    def box(x0, y0, x1, y1, fill):
        d.polygon(_local(obj, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), fill=fill)

    def dot(x, y, r, fill):
        _circle(d, _px(*obj.body.local_to_world((x, y))), r * SS, fill)

    box(-h, -h, h, h, c["edge"])
    box(-h + 1.2, -h + 1.2, h - 1.2, h - 1.2, c["base"])
    box(-0.7 * h, -0.05 * h, 0.7 * h, 0.6 * h, c["screen"])
    for ex in (-0.32 * h, 0.32 * h):
        dot(ex, 0.28 * h, 0.14 * h, c["led"])
    for bx in (-0.72 * h, 0.72 * h):
        dot(bx, -0.55 * h, 0.1 * h, c["bolt"])
    box(-0.08 * h, h, 0.08 * h, h + 0.35 * h, c["antenna"])
    dot(0.0, h + 0.4 * h, 0.14 * h, c["tip"])


def _ball(d: ImageDraw.ImageDraw, obj) -> None:
    """Iron cannonball with a highlight that keeps its place as the ball spins."""
    c = COLORS["ball"]
    r = obj.size[0]
    cx, cy = obj.body.position
    _circle(d, _px(cx, cy), r * SS, c["edge"])
    _circle(d, _px(cx, cy), (r - 1.2) * SS, c["base"])
    _circle(d, _px(cx - 0.3 * r, cy + 0.3 * r), 0.3 * r * SS, c["shine"])


def render_cartoon(world: World) -> Image.Image:
    img = _background().copy()
    d = ImageDraw.Draw(img)
    for o in world.objects:
        if not o.alive:
            continue
        if o.kind == "robot":
            _robot(d, o)
        elif o.kind == "ball":
            _ball(d, o)
        else:
            _block(d, o)
    return img.resize((WIDTH, HEIGHT), Image.LANCZOS)
