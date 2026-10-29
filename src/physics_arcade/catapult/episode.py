"""Generate one benchmark episode: a settled level, a launch, and the recorded rollout."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .cartoon import render_cartoon
from .levels import generate_level
from .render import label_robots, render
from .sim import (
    FPS,
    GRAVITY,
    HEIGHT,
    LAUNCH_POS,
    STEPS_PER_FRAME,
    World,
    ballistic_angles,
    launch_ball,
    max_speed,
    settle,
    step,
)

CONTEXT_FRAMES = 12  # 0.4 s of the ball in flight
TOTAL_FRAMES = 180  # 6 s
SETTLE_SPEED = 8.0  # px/s; episode must be at rest by the last frame


@dataclass
class Episode:
    seed: int
    attempt: int  # index of the accepted setup, for setup_for()
    speed: float
    angle: float
    frames: list[np.ndarray]  # flat style; the source of ground-truth class masks
    display_frames: list[np.ndarray]  # what the model sees (same as `frames` for the flat style)
    labeled_first_frame: Image.Image
    robots_initial: dict[int, tuple[float, float, float]]  # id -> (x, y, r)
    robots_alive: list[int]
    blocks_initial: int
    blocks_destroyed: int
    first_contact_time: float | None


def _snapshot(world: World) -> list[tuple[str, float, float, float]]:
    return [(o.kind, *o.body.position, o.body.angle) for o in world.objects]


def _stable(before, after, tol: float = 4.0) -> bool:
    return all(
        math.hypot(a[1] - b[1], a[2] - b[2]) < tol and abs(a[3] - b[3]) < 0.08 for a, b in zip(before, after)
    )


def _choose_launch(world: World, rng: random.Random) -> tuple[float, float] | None:
    robots = [o for o in world.objects if o.kind == "robot"]
    blocks = [o for o in world.objects if o.kind not in ("robot", "ball")]
    pool = robots if (rng.random() < 0.6 or not blocks) else blocks
    target = rng.choice(pool)
    tx, ty = target.body.position
    for _ in range(10):
        speed = rng.uniform(420, 640)
        angles = ballistic_angles(tx - LAUNCH_POS[0], ty - LAUNCH_POS[1], speed)
        if not angles:
            continue
        angle = angles[0] + rng.gauss(0, 2.5)
        vy = speed * math.sin(math.radians(angle))
        apex = LAUNCH_POS[1] + vy * vy / (2 * GRAVITY) if vy > 0 else LAUNCH_POS[1]
        if apex < HEIGHT - 15:
            return speed, angle
    return None


@dataclass
class Setup:
    """A settled level with a chosen launch, just before the ball is fired."""

    attempt: int
    world: World
    speed: float
    angle: float


def setups(seed: int, max_tries: int = 50):
    """Yield every candidate setup for `seed` in order. Random numbers are consumed
    only here (never during rollouts), so replaying the first k setups is exact."""
    rng = random.Random(seed)
    for attempt in range(max_tries):
        world = generate_level(rng)
        if not world.robots:
            continue
        before = _snapshot(world)
        settle(world, 2.0)
        if not _stable(before, _snapshot(world)) or any(not o.alive for o in world.objects):
            continue
        launch = _choose_launch(world, rng)
        if launch is None:
            continue
        yield Setup(attempt, world, *launch)


def setup_for(seed: int, attempt: int) -> Setup:
    """Rebuild the setup that `generate_episode(seed)` accepted (see Episode.attempt)."""
    for s in setups(seed):
        if s.attempt == attempt:
            return s
    raise ValueError(f"seed {seed} has no setup with attempt {attempt}")


def rollout(world: World, speed: float, angle: float, style: str | None = "flat"):
    """Fire the ball and simulate TOTAL_FRAMES frames. With style=None nothing is rendered.

    Returns (frames, display_frames, labeled_first_frame); the lists are empty when style is None.
    """
    world.time = 0.0
    world.damage_enabled = True
    launch_ball(world, speed, angle)
    if style is None:
        for _ in range(TOTAL_FRAMES - 1):
            for _ in range(STEPS_PER_FRAME):
                step(world)
        return [], [], None
    styled = style != "flat"
    first = render(world)
    first_display = render_cartoon(world) if styled else first
    labeled = label_robots(first_display, world)
    frames, display = [np.asarray(first)], [np.asarray(first_display)]
    for _ in range(TOTAL_FRAMES - 1):
        for _ in range(STEPS_PER_FRAME):
            step(world)
        frames.append(np.asarray(render(world)))
        display.append(np.asarray(render_cartoon(world)) if styled else frames[-1])
    return frames, display, labeled


def generate_episode(seed: int, style: str = "flat", max_tries: int = 50) -> Episode:
    """Deterministically generate an episode from a seed (retrying internally on rejects)."""
    for s in setups(seed, max_tries):
        world = s.world
        robots_initial = {p.robot_id: (*p.body.position, p.size[0] / 2) for p in world.robots}  # (x, y, half-size)
        blocks_initial = sum(o.kind not in ("robot", "ball") for o in world.objects)
        frames, display, labeled = rollout(world, s.speed, s.angle, style)

        # Ball must still be in free flight for the whole context window.
        ctx_t = CONTEXT_FRAMES / FPS
        if world.first_contact_time is not None and world.first_contact_time < ctx_t:
            continue
        if max_speed(world) > SETTLE_SPEED:
            continue
        return Episode(
            seed=seed,
            attempt=s.attempt,
            speed=s.speed,
            angle=s.angle,
            frames=frames,
            display_frames=display,
            labeled_first_frame=labeled,
            robots_initial=robots_initial,
            robots_alive=sorted(p.robot_id for p in world.robots if p.alive),
            blocks_initial=blocks_initial,
            blocks_destroyed=sum(not o.alive for o in world.objects if o.kind not in ("robot", "ball")),
            first_contact_time=world.first_contact_time,
        )
    raise RuntimeError(f"could not generate a valid episode for seed {seed}")
