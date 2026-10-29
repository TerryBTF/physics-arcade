"""Pool: the cue ball has just been struck on a top-down table. Which numbered balls end up
in a pocket once everything stops?"""

from __future__ import annotations

import math

from ...core.game import Game
from .episode import CONTEXT_FRAMES, DIFFICULTIES, TOTAL_FRAMES, generate, replay
from .render import DYNAMIC_CLASSES, LUT_PATH, PALETTE
from .sim import BALL_R, FPS, HEIGHT, POCKET_R, POCKETS, WIDTH

PROMPT = """You are looking at a pool (billiards) table from directly above. There is no gravity in the picture: balls roll on the table and slow down by friction.

The white cue ball has just been struck. The first image shows the table at that moment with every numbered ball labeled by its number. The next {n} images are frames of the cue ball rolling at t = {times} seconds. The frames stop before the cue ball touches anything: your job is to predict what happens AFTER the last frame. Extrapolate the cue ball's straight path and work out which balls it hits, where they go, and what they hit in turn.

Rules:
- Balls bounce off the cushions and off each other (nearly elastic collisions) and gradually slow down.
- A ball that rolls into one of the six pockets (the dark holes at the corners and in the middle of the long sides) is potted and leaves the table.

Question: which numbered balls are potted by the time everything has stopped?"""


def _first_hit_baseline(task: dict) -> list[int]:
    """Straight-line geometry, no simulation: find the first ball on the cue's path and pot it
    iff the line from the contact point through its centre runs into a pocket."""
    info = task["info"]
    cx, cy = info["cue"]["x"], info["cue"]["y"]
    a = math.radians(info["strike"]["angle_deg"])
    ux, uy = math.cos(a), math.sin(a)
    best = None
    for k, b in info["targets"].items():
        rx, ry = b["x"] - cx, b["y"] - cy
        along = rx * ux + ry * uy
        perp = abs(rx * uy - ry * ux)
        if along > 0 and perp < 2 * BALL_R:
            t = along - math.sqrt((2 * BALL_R) ** 2 - perp**2)  # cue travel until contact
            if best is None or t < best[0]:
                best = (t, int(k), cx + t * ux, cy + t * uy)
    if best is None:
        return []
    _, k, hx, hy = best
    b = info["targets"][str(k)]
    dx, dy = b["x"] - hx, b["y"] - hy
    n = math.hypot(dx, dy)
    for px, py in POCKETS:
        rx, ry = px - b["x"], py - b["y"]
        if rx * dx + ry * dy > 0 and abs(rx * dy - ry * dx) / n < POCKET_R:
            return [k]
    return []


GAME = Game(
    name="pool",
    version="pool-v1",
    target="ball",
    event="potted",
    question="Which numbered balls end up in a pocket once everything has stopped?",
    palette=PALETTE,
    dynamic_classes=DYNAMIC_CLASSES,
    count_class="ball",
    fps=FPS,
    width=WIDTH,
    height=HEIGHT,
    context_frames=CONTEXT_FRAMES,
    total_frames=TOTAL_FRAMES,
    difficulties=DIFFICULTIES,
    generate=generate,
    replay=replay,
    # Standard deviations: relative strike speed, strike angle (deg), ball position (px, both axes).
    perturbations={
        "launch_tiny": {"speed": 0.0025, "angle": 0.1, "pos": 0.0},
        "launch": {"speed": 0.01, "angle": 0.5, "pos": 0.0},
        "scene": {"speed": 0.0, "angle": 0.0, "pos": 0.5},
    },
    label_perturbation="launch_tiny",
    vlm_prompt=lambda n, times: PROMPT.format(n=n, times=times),
    lut_path=LUT_PATH,
    # Not a vision baseline: reads exact positions and the strike from tasks.jsonl "info".
    outcome_baselines={"first_hit": _first_hit_baseline},
)
