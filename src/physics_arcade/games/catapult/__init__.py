"""Catapult: a cannonball is fired at towers of wood, stone and ice with robots on and
between them. Which robots survive once everything comes to rest?"""

from __future__ import annotations

import numpy as np

from ...core.game import Game
from .cartoon import LUT_PATH
from .episode import CONTEXT_FRAMES, TOTAL_FRAMES, generate, replay
from .levels import DIFFICULTIES
from .render import DYNAMIC_CLASSES, PALETTE
from .sim import BALL_RADIUS, FPS, HEIGHT, WIDTH, ballistic_y

PROMPT = """You are looking at a 2D physics game shown from the side. Gravity points down.

A cannonball was just fired from the catapult on the left. The first image shows the level at launch with every robot labeled by its id. The next {n} images are frames of the cannonball's flight at t = {times} seconds. The frames stop before the cannonball hits anything: your job is to predict what happens AFTER the last frame. Extrapolate the cannonball's trajectory (it keeps flying along its arc under gravity) and work out what it will hit and what follows.

Rules of the game:
- The cannonball flies on, hits whatever is in its path, and pushes, topples or breaks blocks.
- Blocks are wood (brown), stone (gray, heaviest and strongest) or ice (pale blue, light and brittle).
- Robots are small yellow boxes. A robot is destroyed when it is hit hard enough: by the cannonball, by a falling or sliding block, or by falling itself. Robots that are only nudged survive.
- The game is over when everything has come to rest.

Question: which robots are still alive when everything has come to rest, after the cannonball's impact and any collapse it causes?"""


def _ballistic_hits(task: dict) -> set[int]:
    """Robots that the unobstructed parabola passes through (ignores blocks and collapses)."""
    speed, angle = task["info"]["launch"]["speed"], task["info"]["launch"]["angle_deg"]
    hit = set()
    for rid, p in task["info"]["targets"].items():
        for dx in np.linspace(-p["r"], p["r"], 9):
            y = ballistic_y(p["x"] + dx, speed, angle)
            if y is not None and abs(y - p["y"]) <= p["r"] + BALL_RADIUS:
                hit.add(int(rid))
                break
    return hit


GAME = Game(
    name="catapult",
    version="catapult-v1",
    target="robot",
    event="alive",
    question="Which robots are still alive after the cannonball's impact, once everything has come to rest?",
    palette=PALETTE,
    dynamic_classes=DYNAMIC_CLASSES,
    count_class="robot",
    fps=FPS,
    width=WIDTH,
    height=HEIGHT,
    context_frames=CONTEXT_FRAMES,
    total_frames=TOTAL_FRAMES,
    difficulties=DIFFICULTIES,
    generate=generate,
    replay=replay,
    # Standard deviations: relative launch speed, launch angle (deg), horizontal object position (px).
    #   launch_tiny  at or below the precision a careful estimate from the context frames reaches
    #   launch       a realistic but imperfect visual estimate
    #   scene        objects shifted slightly, launch exact
    perturbations={
        "launch_tiny": {"speed": 0.0025, "angle": 0.1, "pos": 0.0},
        "launch": {"speed": 0.01, "angle": 0.5, "pos": 0.0},
        "scene": {"speed": 0.0, "angle": 0.0, "pos": 0.5},
    },
    label_perturbation="launch_tiny",
    vlm_prompt=lambda n, times: PROMPT.format(n=n, times=times),
    lut_path=LUT_PATH,
    # Not a vision baseline: reads the exact launch numbers from tasks.jsonl "info".
    outcome_baselines={"ballistic": lambda t: sorted(set(t["target_ids"]) - _ballistic_hits(t))},
)
