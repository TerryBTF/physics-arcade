"""Generate one benchmark episode: a settled level, a launch, and the recorded rollout."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .levels import generate_level
from .render import label_pigs, render
from .sim import (
    FPS,
    GRAVITY,
    HEIGHT,
    SLING_POS,
    STEPS_PER_FRAME,
    World,
    ballistic_angles,
    launch_bird,
    max_speed,
    settle,
    step,
)

CONTEXT_FRAMES = 12  # 0.4 s of the bird in flight
TOTAL_FRAMES = 180  # 6 s
SETTLE_SPEED = 8.0  # px/s; episode must be at rest by the last frame


@dataclass
class Episode:
    seed: int
    speed: float
    angle: float
    frames: list[np.ndarray]
    labeled_first_frame: Image.Image
    pigs_initial: dict[int, tuple[float, float, float]]  # id -> (x, y, r)
    pigs_alive: list[int]
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
    pigs = [o for o in world.objects if o.kind == "pig"]
    blocks = [o for o in world.objects if o.kind not in ("pig", "bird")]
    pool = pigs if (rng.random() < 0.6 or not blocks) else blocks
    target = rng.choice(pool)
    tx, ty = target.body.position
    for _ in range(10):
        speed = rng.uniform(420, 640)
        angles = ballistic_angles(tx - SLING_POS[0], ty - SLING_POS[1], speed)
        if not angles:
            continue
        angle = angles[0] + rng.gauss(0, 2.5)
        vy = speed * math.sin(math.radians(angle))
        apex = SLING_POS[1] + vy * vy / (2 * GRAVITY) if vy > 0 else SLING_POS[1]
        if apex < HEIGHT - 15:
            return speed, angle
    return None


def generate_episode(seed: int, max_tries: int = 50) -> Episode:
    """Deterministically generate an episode from a seed (retrying internally on rejects)."""
    rng = random.Random(seed)
    for _ in range(max_tries):
        world = generate_level(rng)
        if not world.pigs:
            continue
        before = _snapshot(world)
        settle(world, 2.0)
        if not _stable(before, _snapshot(world)) or any(not o.alive for o in world.objects):
            continue
        launch = _choose_launch(world, rng)
        if launch is None:
            continue
        speed, angle = launch

        pigs_initial = {p.pig_id: (*p.body.position, p.size[0]) for p in world.pigs}
        blocks_initial = sum(o.kind not in ("pig", "bird") for o in world.objects)
        world.time = 0.0
        world.damage_enabled = True
        launch_bird(world, speed, angle)
        first = render(world)
        labeled = label_pigs(first, world)
        frames = [np.asarray(first)]
        for _ in range(TOTAL_FRAMES - 1):
            for _ in range(STEPS_PER_FRAME):
                step(world)
            frames.append(np.asarray(render(world)))

        # Bird must still be in free flight for the whole context window.
        ctx_t = CONTEXT_FRAMES / FPS
        if world.first_contact_time is not None and world.first_contact_time < ctx_t:
            continue
        if max_speed(world) > SETTLE_SPEED:
            continue
        return Episode(
            seed=seed,
            speed=speed,
            angle=angle,
            frames=frames,
            labeled_first_frame=labeled,
            pigs_initial=pigs_initial,
            pigs_alive=sorted(p.pig_id for p in world.pigs if p.alive),
            blocks_initial=blocks_initial,
            blocks_destroyed=sum(not o.alive for o in world.objects if o.kind not in ("pig", "bird")),
            first_contact_time=world.first_contact_time,
        )
    raise RuntimeError(f"could not generate a valid episode for seed {seed}")
