"""Domino episodes: a random chain, a tossed ball, the recorded rollout, and perturbed replays."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from ...core.game import Episode
from .render import label_dominoes, render, render_cartoon
from .sim import (
    FLOOR_Y,
    FPS,
    GRAVITY,
    STEPS_PER_FRAME,
    WIDTH,
    Scene,
    add_domino,
    add_platform,
    max_speed,
    new_scene,
    step,
    toss_ball,
)

CONTEXT_FRAMES = 12  # 0.4 s of the ball in flight
TOTAL_FRAMES = 180  # 6 s
DIFFICULTIES = ("easy", "medium", "hard")
N_DOMINOES = {"easy": (4, 6), "medium": (6, 8), "hard": (8, 11)}
P_HEAVY = {"easy": 0.0, "medium": 0.25, "hard": 0.3}
BALL_START = (40.0, FLOOR_Y + 60.0)
SETTLE_SPEED = 5.0


@dataclass
class Setup:
    attempt: int
    scene: Scene
    vx: float
    vy: float


def _layout(rng: random.Random, difficulty: str) -> Scene | None:
    scene = new_scene()
    n = rng.randint(*N_DOMINOES[difficulty])
    if difficulty == "easy":
        h0 = rng.uniform(34, 44)
        heights = [h0 + rng.uniform(-2, 2) for _ in range(n)]
    else:
        heights = [rng.uniform(26, 50) for _ in range(n)]
    materials = ["heavy" if i and rng.random() < P_HEAVY[difficulty] else "light" for i in range(n)]
    # Centre-to-centre spacing relative to the height of the domino that has to reach the next one;
    # around one height the chain may or may not continue.
    xs = [rng.uniform(150, 190)]
    for i in range(1, n):
        ratio = rng.uniform(1.0, 1.6) if rng.random() < 0.25 else rng.uniform(0.4, 0.85)
        xs.append(xs[-1] + max(14.0, ratio * heights[i - 1]))
    if xs[-1] > WIDTH - 40:
        return None
    bases = [FLOOR_Y] * n
    if difficulty == "hard" and rng.random() < 0.7:
        k = rng.randint(2, n - 2)
        top = FLOOR_Y + rng.uniform(10, 28)
        mid = (xs[k - 1] + xs[k]) / 2
        if rng.random() < 0.5:  # step up: dominoes k.. stand on the platform
            add_platform(scene, mid, WIDTH - 15, top)
            bases[k:] = [top] * (n - k)
        else:  # step down: dominoes ..k-1 stand on the platform
            add_platform(scene, xs[0] - 20, mid, top)
            bases[:k] = [top] * k
    for i in range(n):
        add_domino(scene, i + 1, materials[i], xs[i], bases[i], heights[i])
    return scene


def _toss(rng: random.Random, scene: Scene) -> tuple[float, float]:
    """Velocity that brings the ball to the upper part of the first domino after 0.55-0.85 s."""
    d = scene.dominoes[0]
    tx = d.body.position.x - d.size[0] / 2 - 7.0
    ty = d.body.position.y - d.size[1] / 2 + d.size[1] * rng.uniform(0.6, 0.85)
    t = rng.uniform(0.55, 0.85)
    vx = (tx - BALL_START[0]) / t
    vy = (ty - BALL_START[1] + 0.5 * GRAVITY * t * t) / t
    return vx, vy


def setups(seed: int, difficulty: str, max_tries: int = 80):
    rng = random.Random(f"dominoes-{difficulty}-{seed}")
    for attempt in range(max_tries):
        scene = _layout(rng, difficulty)
        if scene is None:
            continue
        start = [tuple(d.body.position) for d in scene.dominoes]
        for _ in range(FPS * STEPS_PER_FRAME):  # 1 s settle: the chain must stand on its own
            step(scene)
        if any(math.dist(a, tuple(d.body.position)) > 1.0 or d.fallen() for a, d in zip(start, scene.dominoes, strict=True)):
            continue
        scene.time = 0.0
        yield Setup(attempt, scene, *_toss(rng, scene))


def setup_for(seed: int, difficulty: str, attempt: int) -> Setup:
    for s in setups(seed, difficulty):
        if s.attempt == attempt:
            return s
    raise ValueError(f"seed {seed} ({difficulty}) has no setup with attempt {attempt}")


def rollout(scene: Scene, vx: float, vy: float, style: str | None = "flat"):
    toss_ball(scene, *BALL_START, vx, vy)
    if style is None:
        for _ in range(TOTAL_FRAMES - 1):
            for _ in range(STEPS_PER_FRAME):
                step(scene)
        return [], [], None
    styled = style != "flat"
    first = render(scene)
    first_display = render_cartoon(scene) if styled else first
    labeled = label_dominoes(first_display, scene)
    frames, display = [np.asarray(first)], [np.asarray(first_display)]
    for _ in range(TOTAL_FRAMES - 1):
        for _ in range(STEPS_PER_FRAME):
            step(scene)
        frames.append(np.asarray(render(scene)))
        display.append(np.asarray(render_cartoon(scene)) if styled else frames[-1])
    return frames, display, labeled


def generate(seed: int, difficulty: str, style: str = "flat") -> Episode:
    for s in setups(seed, difficulty):
        sc = s.scene
        layout = {
            str(d.number): {"x": round(d.body.position.x, 1), "base": round(d.body.position.y - d.size[1] / 2, 1),
                            "h": round(d.size[1], 1), "material": d.material}
            for d in sc.dominoes
        }
        frames, display, labeled = rollout(sc, s.vx, s.vy, style)
        if sc.first_contact_time is None or sc.first_contact_time < CONTEXT_FRAMES / FPS or max_speed(sc) > SETTLE_SPEED:
            continue
        return Episode(
            seed=seed,
            difficulty=difficulty,
            frames=frames,
            display_frames=display,
            labeled_frame=labeled,
            target_ids=[d.number for d in sc.dominoes],
            recorded_ids=[d.number for d in sc.dominoes if d.fallen()],
            replay_key={"attempt": s.attempt},
            info={"toss": {"vx": round(s.vx, 2), "vy": round(s.vy, 2)}, "targets": layout,
                  "platforms": [[round(v, 1) for v in p] for p in sc.platforms]},
            extra={"toss": {"vx": s.vx, "vy": s.vy}, "first_contact_time": sc.first_contact_time},
        )
    raise RuntimeError(f"could not generate a valid {difficulty} domino episode for seed {seed}")


def replay(seed: int, difficulty: str, replay_key: dict, noise: dict, rng: random.Random) -> list[int]:
    """Re-run the accepted setup with noise on the toss (relative speed, angle in degrees) and
    on every domino's horizontal position (pixels); return the dominoes that fall."""
    s = setup_for(seed, difficulty, replay_key["attempt"])
    if noise.get("pos"):
        for d in s.scene.dominoes:
            x, y = d.body.position
            d.body.position = (x + rng.gauss(0, noise["pos"]), y)
            s.scene.space.reindex_shapes_for_body(d.body)
    speed = math.hypot(s.vx, s.vy) * (1 + rng.gauss(0, noise.get("speed", 0.0)))
    angle = math.atan2(s.vy, s.vx) + math.radians(rng.gauss(0, noise.get("angle", 0.0)))
    rollout(s.scene, speed * math.cos(angle), speed * math.sin(angle), None)
    return [d.number for d in s.scene.dominoes if d.fallen()]
