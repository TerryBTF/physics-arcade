"""Pool episodes: a random layout, a cue strike, the recorded rollout, and perturbed replays."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from ...core.game import Episode
from .render import label_balls, render, render_cartoon
from .sim import (
    BALL_E, BALL_R, CUSHION_E, FPS, POCKET_R, POCKETS, ROLL_DECEL, STEPS_PER_FRAME, STOP_SPEED, TABLE_X0, TABLE_X1, TABLE_Y0, TABLE_Y1,
    Table, add_ball, max_speed, new_table, step, strike,
)

CONTEXT_FRAMES = 12  # 0.4 s of the cue ball rolling, before it touches anything
TOTAL_FRAMES = 180  # 6 s
DIFFICULTIES = ("easy", "medium", "hard")
N_BALLS = {"easy": (1, 2), "medium": (3, 5), "hard": (6, 9)}
# Aim error (deg). Pool is very sensitive: at 300 px, 1 deg moves the contact point ~5 px,
# which turns the object ball by ~18 deg. Shots are either aimed true (tiny error) or
# clearly off, so most outcomes are not decided by a hair.
P_TRUE_AIM = 0.55
TRUE_AIM_SD = 0.05
MISS_OFFSET = (0.6, 1.5)
MAX_CUT_DEG = 60.0
MAX_STRIKE = 850.0  # px/s; with ROLL_DECEL every ball stops within the episode


@dataclass
class Setup:
    attempt: int
    table: Table
    speed: float
    angle: float


def _free(p: tuple[float, float], placed: list[tuple[float, float]], margin: float = 2 * BALL_R + 1.0) -> bool:
    x, y = p
    if not (TABLE_X0 + BALL_R + 4 < x < TABLE_X1 - BALL_R - 4 and TABLE_Y0 + BALL_R + 4 < y < TABLE_Y1 - BALL_R - 4):
        return False
    if any(math.hypot(x - px, y - py) < POCKET_R + BALL_R + 6 for px, py in POCKETS):
        return False
    return all(math.hypot(x - qx, y - qy) >= margin for qx, qy in placed)


def _random_spot(rng: random.Random, placed: list[tuple[float, float]]) -> tuple[float, float] | None:
    for _ in range(200):
        p = (rng.uniform(TABLE_X0, TABLE_X1), rng.uniform(TABLE_Y0, TABLE_Y1))
        if _free(p, placed):
            return p
    return None


def _layout(rng: random.Random, difficulty: str) -> list[tuple[float, float]] | None:
    """Positions of the object balls; hard tables start with a tight cluster."""
    n = rng.randint(*N_BALLS[difficulty])
    placed: list[tuple[float, float]] = []
    if difficulty == "hard":
        cx, cy = rng.uniform(TABLE_X0 + 120, TABLE_X1 - 120), rng.uniform(TABLE_Y0 + 70, TABLE_Y1 - 70)
        a = rng.uniform(0, 2 * math.pi)
        ux, uy, vx, vy = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
        d = 2 * BALL_R + 0.5
        for row in range(3):  # a triangle of up to 6 balls
            for k in range(row + 1):
                if len(placed) >= min(n, 6):
                    break
                along, across = row * d * math.sqrt(3) / 2, (k - row / 2) * d
                p = (cx + along * ux + across * vx, cy + along * uy + across * vy)
                if _free(p, placed, margin=2 * BALL_R + 0.4):
                    placed.append(p)
    while len(placed) < n:
        p = _random_spot(rng, placed)
        if p is None:
            return None
        placed.append(p)
    return placed


def _clear(a: tuple[float, float], b: tuple[float, float], others: list[tuple[float, float]]) -> bool:
    """No ball in `others` within 2R of the segment a-b (a ball rolling a->b would miss them)."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    n2 = dx * dx + dy * dy
    for ox, oy in others:
        t = max(0.0, min(1.0, ((ox - ax) * dx + (oy - ay) * dy) / n2))
        if math.hypot(ax + t * dx - ox, ay + t * dy - oy) < 2 * BALL_R + 1:
            return False
    return True


SHOT_WEIGHTS = {"easy": {"direct": 1.0}, "medium": {"direct": 0.6, "bank": 0.4}, "hard": {"direct": 0.4, "bank": 0.3, "combo": 0.3}}


def _ghost(ball: tuple[float, float], toward: tuple[float, float]) -> tuple[float, float]:
    """Where a ball must be at contact to send `ball` straight at `toward`."""
    dx, dy = ball[0] - toward[0], ball[1] - toward[1]
    d = math.hypot(dx, dy)
    return ball[0] + 2 * BALL_R * dx / d, ball[1] + 2 * BALL_R * dy / d


def _cut_ok(frm: tuple[float, float], ghost: tuple[float, float], ball: tuple[float, float], toward: tuple[float, float]) -> bool:
    ax, ay = ghost[0] - frm[0], ghost[1] - frm[1]
    bx, by = toward[0] - ball[0], toward[1] - ball[1]
    cos = (ax * bx + ay * by) / (math.hypot(ax, ay) * math.hypot(bx, by))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) <= MAX_CUT_DEG


def _bank_point(cue: tuple[float, float], ghost: tuple[float, float], side: int):
    """Mirror method: the cue ball's centre bounces off the line BALL_R inside a cushion. The
    bounce keeps the tangential speed and scales the normal speed by the restitution e, so the
    mirror image sits at (normal distance / e) behind the line."""
    e = CUSHION_E * BALL_E
    lines = [("y", TABLE_Y0 + BALL_R), ("y", TABLE_Y1 - BALL_R), ("x", TABLE_X0 + BALL_R), ("x", TABLE_X1 - BALL_R)]
    axis, c = lines[side]
    if axis == "y":
        m = (ghost[0], c - (ghost[1] - c) / e)
        t = (c - cue[1]) / (m[1] - cue[1]) if m[1] != cue[1] else -1
    else:
        m = (c - (ghost[0] - c) / e, ghost[1])
        t = (c - cue[0]) / (m[0] - cue[0]) if m[0] != cue[0] else -1
    if not 0 < t < 1:
        return None
    hit = (cue[0] + t * (m[0] - cue[0]), cue[1] + t * (m[1] - cue[1]))
    if any(math.hypot(hit[0] - px, hit[1] - py) < POCKET_R + 3 * BALL_R for px, py in POCKETS):
        return None  # would run into a pocket mouth instead of the cushion
    return hit, m


def _choose_shot(rng: random.Random, cue: tuple[float, float], balls: list[tuple[float, float]], difficulty: str):
    """Plan a shot that would pot a ball if struck perfectly:
    direct  cue -> object ball -> pocket
    bank    cue -> cushion -> object ball -> pocket (straight-line reasoning from the cue fails)
    combo   cue -> ball A -> ball B -> pocket (the first ball hit is not the one potted)
    Paths are clear of other balls; the strike speed covers the friction losses on the way."""
    kinds = SHOT_WEIGHTS[difficulty]
    for _ in range(80):
        kind = rng.choices(list(kinds), weights=list(kinds.values()))[0]
        pocket = rng.choice(POCKETS)
        target = rng.choice(balls)
        others = [b for b in balls if b != target]
        g = _ghost(target, pocket)
        if not _clear(target, pocket, others):
            continue
        roll = math.dist(target, pocket) - 2 * BALL_R  # distance the potted ball rolls
        v_need = math.sqrt(2 * ROLL_DECEL * roll + 120.0**2)
        if kind == "combo":
            if not others:
                continue
            a = rng.choice(others)
            rest = [b for b in others if b != a]
            if not (_cut_ok(a, g, target, pocket) and _clear(a, g, rest)):
                continue
            v_need = math.sqrt(v_need**2 / 0.9 + 2 * ROLL_DECEL * math.dist(a, g))
            aim_at, frm_ball, toward = _ghost(a, g), a, g
            rest_for_cue = [b for b in balls if b != a]
        else:
            aim_at, frm_ball, toward = g, target, pocket
            rest_for_cue = others
        if kind == "bank":
            bank = _bank_point(cue, aim_at, rng.randrange(4))
            if bank is None:
                continue
            hit, mirror = bank
            if not (_clear(cue, hit, balls) and _clear(hit, aim_at, rest_for_cue) and _cut_ok(hit, aim_at, frm_ball, toward)):
                continue
            path, aim_point, v_bank = math.dist(cue, hit) + math.dist(hit, aim_at), mirror, 0.8
        else:
            if not (_clear(cue, aim_at, rest_for_cue) and _cut_ok(cue, aim_at, frm_ball, toward)):
                continue
            path, aim_point, v_bank = math.dist(cue, aim_at), aim_at, 1.0
        if path < 230:  # cue must roll for the whole context window first
            continue
        cut_cos = 0.7  # conservative transfer for the speed estimate
        v_contact = v_need / (0.95 * cut_cos) / v_bank
        speed = math.sqrt(v_contact**2 + 2 * ROLL_DECEL * path) * rng.uniform(1.0, 1.4)
        if speed > MAX_STRIKE:
            continue
        if rng.random() < P_TRUE_AIM:
            error = rng.gauss(0, TRUE_AIM_SD)
        else:
            error = rng.choice((-1, 1)) * rng.uniform(*MISS_OFFSET)
        return speed, math.degrees(math.atan2(aim_point[1] - cue[1], aim_point[0] - cue[0])) + error
    return None


def setups(seed: int, difficulty: str, max_tries: int = 80):
    rng = random.Random(f"pool-{difficulty}-{seed}")
    for attempt in range(max_tries):
        balls = _layout(rng, difficulty)
        if balls is None:
            continue
        cue = _random_spot(rng, balls)
        if cue is None:
            continue
        shot = _choose_shot(rng, cue, balls, difficulty)
        if shot is None:
            continue
        table = new_table()
        add_ball(table, 0, *cue)
        for i, p in enumerate(balls, start=1):
            add_ball(table, i, *p)
        yield Setup(attempt, table, *shot)


def setup_for(seed: int, difficulty: str, attempt: int) -> Setup:
    for s in setups(seed, difficulty):
        if s.attempt == attempt:
            return s
    raise ValueError(f"seed {seed} ({difficulty}) has no setup with attempt {attempt}")


def rollout(table: Table, speed: float, angle: float, style: str | None = "flat"):
    strike(table, speed, angle)
    if style is None:
        for _ in range(TOTAL_FRAMES - 1):
            for _ in range(STEPS_PER_FRAME):
                step(table)
        return [], [], None
    styled = style != "flat"
    first = render(table)
    first_display = render_cartoon(table) if styled else first
    labeled = label_balls(first_display, table)
    frames, display = [np.asarray(first)], [np.asarray(first_display)]
    for _ in range(TOTAL_FRAMES - 1):
        for _ in range(STEPS_PER_FRAME):
            step(table)
        frames.append(np.asarray(render(table)))
        display.append(np.asarray(render_cartoon(table)) if styled else frames[-1])
    return frames, display, labeled


def generate(seed: int, difficulty: str, style: str = "flat") -> Episode:
    for s in setups(seed, difficulty):
        t = s.table
        start = {b.number: tuple(b.body.position) for b in t.balls}
        frames, display, labeled = rollout(t, s.speed, s.angle, style)
        if t.first_contact_time is None or t.first_contact_time < CONTEXT_FRAMES / FPS:
            continue
        if t.cue.potted or max_speed(t) > STOP_SPEED:
            continue  # scratches and still-rolling tables are rejected
        return Episode(
            seed=seed,
            difficulty=difficulty,
            frames=frames,
            display_frames=display,
            labeled_frame=labeled,
            target_ids=[b.number for b in t.object_balls()],
            recorded_ids=sorted(b.number for b in t.object_balls() if b.potted),
            replay_key={"attempt": s.attempt},
            info={
                "strike": {"speed": round(s.speed, 2), "angle_deg": round(s.angle, 2)},
                "cue": {"x": round(start[0][0], 1), "y": round(start[0][1], 1)},
                "targets": {str(k): {"x": round(x, 1), "y": round(y, 1), "r": BALL_R} for k, (x, y) in start.items() if k},
            },
            extra={"strike": {"speed": s.speed, "angle_deg": s.angle}, "first_contact_time": t.first_contact_time},
        )
    raise RuntimeError(f"could not generate a valid {difficulty} pool episode for seed {seed}")


def replay(seed: int, difficulty: str, replay_key: dict, noise: dict, rng: random.Random) -> list[int]:
    """Re-run the accepted setup with noise on the strike (relative speed, angle in degrees)
    and on every ball's position (pixels, both axes); return the potted object balls."""
    s = setup_for(seed, difficulty, replay_key["attempt"])
    if noise.get("pos"):
        for b in s.table.balls:
            x, y = b.body.position
            b.body.position = (x + rng.gauss(0, noise["pos"]), y + rng.gauss(0, noise["pos"]))
            s.table.space.reindex_shapes_for_body(b.body)
    speed = s.speed * (1 + rng.gauss(0, noise.get("speed", 0.0)))
    angle = s.angle + rng.gauss(0, noise.get("angle", 0.0))
    rollout(s.table, speed, angle, None)
    return sorted(b.number for b in s.table.object_balls() if b.potted)
