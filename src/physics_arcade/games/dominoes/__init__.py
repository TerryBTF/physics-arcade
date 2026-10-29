"""Dominoes: a ball is tossed at the first domino of a chain. Which dominoes fall?"""

from __future__ import annotations

from ...core.game import Game
from .episode import CONTEXT_FRAMES, DIFFICULTIES, TOTAL_FRAMES, generate, replay
from .render import DYNAMIC_CLASSES, LUT_PATH, PALETTE
from .sim import DOMINO_W, FPS, HEIGHT, WIDTH

PROMPT = """You are looking at a row of dominoes from the side. Gravity points down.

A small blue ball has just been tossed at the first domino. The first image shows the scene at that moment with every domino labeled by its number. The next {n} images are frames of the ball's flight at t = {times} seconds. The frames stop before the ball touches anything: your job is to predict what happens AFTER the last frame.

Rules:
- A falling domino tips over toward the next one and can knock it down if it reaches it.
- Dominoes are light (cream) or heavy (dark blue, ten times heavier). A light domino may not be able to topple a heavy one.
- Some dominoes may stand on a raised brick step.
- A domino counts as fallen if it ends up tilted clearly away from upright.

Question: which dominoes have fallen once everything has come to rest?"""


def _gap_rule(task: dict) -> list[int]:
    """No simulation, exact layout from tasks.jsonl "info": the chain continues while the next
    domino stands closer than the falling one's height (ignores mass and steps)."""
    t = task["info"]["targets"]
    ids = sorted(int(k) for k in t)
    fallen = [ids[0]]
    for a, b in zip(ids, ids[1:]):
        if t[str(b)]["x"] - t[str(a)]["x"] - DOMINO_W < t[str(a)]["h"]:
            fallen.append(b)
        else:
            break
    return fallen


GAME = Game(
    name="dominoes",
    version="dominoes-v1",
    target="domino",
    event="fallen",
    question="Which dominoes have fallen once everything has come to rest?",
    palette=PALETTE,
    dynamic_classes=DYNAMIC_CLASSES,
    count_class="light",
    fps=FPS,
    width=WIDTH,
    height=HEIGHT,
    context_frames=CONTEXT_FRAMES,
    total_frames=TOTAL_FRAMES,
    difficulties=DIFFICULTIES,
    generate=generate,
    replay=replay,
    # Standard deviations: relative toss speed, toss angle (deg), horizontal domino position (px).
    perturbations={
        "launch_tiny": {"speed": 0.0025, "angle": 0.1, "pos": 0.0},
        "launch": {"speed": 0.01, "angle": 0.5, "pos": 0.0},
        "scene": {"speed": 0.0, "angle": 0.0, "pos": 0.5},
    },
    label_perturbation="launch_tiny",
    vlm_prompt=lambda n, times: PROMPT.format(n=n, times=times),
    lut_path=LUT_PATH,
    outcome_baselines={"gap_rule": _gap_rule},
)
