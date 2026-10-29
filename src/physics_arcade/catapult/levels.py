"""Procedural level generation."""

from __future__ import annotations

import random

from .sim import GROUND_Y, World, add_block, add_robot, new_world

STRUCT_X_MIN, STRUCT_X_MAX = 320.0, 620.0
POST_W = 10.0
PLANK_H = 10.0


def _material(rng: random.Random) -> str:
    return rng.choices(["wood", "stone", "ice"], weights=[0.5, 0.25, 0.25])[0]


def _tower(world: World, rng: random.Random, x0: float, next_robot: list[int]) -> float:
    """Build a stack of post-and-plank storeys starting at x0. Returns right edge."""
    span = rng.uniform(36, 64)
    floors = rng.choices([1, 2, 3], weights=[0.35, 0.4, 0.25])[0]
    width = span + POST_W
    cx = x0 + width / 2
    base = GROUND_Y
    for floor in range(floors):
        h = rng.uniform(34, 60)
        mat = _material(rng)
        for px in (cx - span / 2, cx + span / 2):
            add_block(world, mat, px, base + h / 2, POST_W, h)
        r = rng.uniform(8, 11)
        if r * 2 < span - POST_W - 2 and rng.random() < 0.6:
            add_robot(world, _take(next_robot), cx + rng.uniform(-3, 3), base + r, r)
        base += h
        plank_w = width + rng.uniform(4, 16)
        add_block(world, _material(rng), cx, base + PLANK_H / 2, plank_w, PLANK_H)
        base += PLANK_H
    if rng.random() < 0.7:
        r = rng.uniform(8, 12)
        add_robot(world, _take(next_robot), cx + rng.uniform(-6, 6), base + r, r)
    return x0 + width + 8


def _pedestal(world: World, rng: random.Random, x0: float, next_robot: list[int]) -> float:
    w, h = rng.uniform(18, 30), rng.uniform(15, 45)
    add_block(world, _material(rng), x0 + w / 2, GROUND_Y + h / 2, w, h)
    r = rng.uniform(8, min(12, w / 2))
    add_robot(world, _take(next_robot), x0 + w / 2, GROUND_Y + h + r, r)
    return x0 + w


def _ground_robot(world: World, rng: random.Random, x0: float, next_robot: list[int]) -> float:
    r = rng.uniform(8, 12)
    add_robot(world, _take(next_robot), x0 + r, GROUND_Y + r, r)
    return x0 + 2 * r


def _take(counter: list[int]) -> int:
    counter[0] += 1
    return counter[0]


def generate_level(rng: random.Random) -> World:
    """Place towers, pedestals and loose robots left-to-right inside the structure zone."""
    world = new_world()
    next_robot = [0]
    x = STRUCT_X_MIN + rng.uniform(0, 40)
    builders = [_tower, _tower, _tower, _pedestal, _ground_robot]
    while True:
        builder = rng.choice(builders)
        # Rough width check so structures stay on screen.
        if x + 90 > STRUCT_X_MAX:
            break
        x = builder(world, rng, x, next_robot) + rng.uniform(6, 30)
    return world
