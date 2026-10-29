"""Flat-color renderer. Every object class has exactly one color so frames can be
segmented back into class masks (physics_arcade.core.segment) for automatic evaluation."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .sim import GROUND_Y, HEIGHT, LAUNCH_POS, WIDTH, World

PALETTE: dict[str, tuple[int, int, int]] = {
    "sky": (150, 200, 240),
    "ground": (90, 140, 60),
    "catapult": (60, 40, 20),
    "ball": (30, 100, 110),
    "robot": (245, 205, 40),
    "wood": (190, 130, 60),
    "stone": (120, 120, 130),
    "ice": (200, 240, 255),
}
CLASSES = list(PALETTE)
DYNAMIC_CLASSES = ["ball", "robot", "wood", "stone", "ice"]


def _flip(x: float, y: float) -> tuple[float, float]:
    return x, HEIGHT - y


def render(world: World) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), PALETTE["sky"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, HEIGHT - GROUND_Y, WIDTH, HEIGHT], fill=PALETTE["ground"])
    sx, sy = LAUNCH_POS
    d.rectangle([*_flip(sx - 3, sy - 12), *_flip(sx + 3, GROUND_Y)], fill=PALETTE["catapult"])
    for o in world.objects:
        if not o.alive:
            continue
        color = PALETTE[o.kind]
        if o.kind == "ball":
            x, y = _flip(*o.body.position)
            r = o.size[0]
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)
        else:
            pts = [_flip(*o.body.local_to_world(v)) for v in o.shape.get_vertices()]
            d.polygon(pts, fill=color)
    return img


def label_robots(img: Image.Image, world: World) -> Image.Image:
    """Copy of `img` with each robot's id written above it (for the QA prompt)."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=14)
    for p in world.robots:
        if not p.alive:
            continue
        x, y = _flip(*p.body.position)
        r = p.size[1] / 2
        text = str(p.robot_id)
        box = d.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        tx, ty = x - tw / 2, y - r - th - 8
        d.rectangle([tx - 3, ty - 2, tx + tw + 3, ty + th + 5], fill=(255, 255, 255), outline=(0, 0, 0))
        d.text((tx, ty - box[1] + 1), text, fill=(0, 0, 0), font=font)
    return img
