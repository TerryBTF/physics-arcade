"""Top-down pool table built on pymunk.

Coordinates are pixels, origin bottom-left, y up (the renderer flips them). There is no
gravity; rolling friction is a constant deceleration against each ball's motion, as on a
real table. A ball whose centre comes within POCKET_R of a pocket centre drops in and is
removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pymunk

WIDTH, HEIGHT = 640, 360
FPS = 30
STEPS_PER_FRAME = 8
DT = 1.0 / (FPS * STEPS_PER_FRAME)

# Playing surface (inside the cushions).
TABLE_X0, TABLE_Y0, TABLE_X1, TABLE_Y1 = 40.0, 40.0, 600.0, 320.0
BALL_R = 8.0
POCKET_R = 16.0  # capture radius around each pocket centre (~one ball diameter, as on real tables)
POCKET_DRAW_R = 18.0
BALL_E = 0.95
CUSHION_E = 0.8  # restitution of the cushions (ball elasticity also multiplies in)
ROLL_DECEL = 150.0  # px/s^2, rolling friction
STOP_SPEED = 4.0  # px/s; slower balls are stopped
POCKETS = [
    (TABLE_X0 + 2, TABLE_Y0 + 2), ((TABLE_X0 + TABLE_X1) / 2, TABLE_Y0 - 3), (TABLE_X1 - 2, TABLE_Y0 + 2),
    (TABLE_X0 + 2, TABLE_Y1 - 2), ((TABLE_X0 + TABLE_X1) / 2, TABLE_Y1 + 3), (TABLE_X1 - 2, TABLE_Y1 - 2),
]


@dataclass
class Ball:
    number: int  # 0 is the cue ball
    shape: pymunk.Circle
    potted: bool = False

    @property
    def body(self) -> pymunk.Body:
        return self.shape.body


@dataclass
class Table:
    space: pymunk.Space
    balls: list[Ball] = field(default_factory=list)
    time: float = 0.0
    first_contact_time: float | None = None

    @property
    def cue(self) -> Ball:
        return self.balls[0]

    def object_balls(self) -> list[Ball]:
        return self.balls[1:]


def new_table() -> Table:
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.iterations = 20
    table = Table(space=space)
    gap = POCKET_R + 3  # cushions stop short of the pockets so balls can drop in
    xm = (TABLE_X0 + TABLE_X1) / 2
    cushions = [
        ((TABLE_X0 + gap, TABLE_Y0), (xm - gap, TABLE_Y0)), ((xm + gap, TABLE_Y0), (TABLE_X1 - gap, TABLE_Y0)),
        ((TABLE_X0 + gap, TABLE_Y1), (xm - gap, TABLE_Y1)), ((xm + gap, TABLE_Y1), (TABLE_X1 - gap, TABLE_Y1)),
        ((TABLE_X0, TABLE_Y0 + gap), (TABLE_X0, TABLE_Y1 - gap)), ((TABLE_X1, TABLE_Y0 + gap), (TABLE_X1, TABLE_Y1 - gap)),
    ]
    for a, b in cushions:
        seg = pymunk.Segment(space.static_body, a, b, 1.0)
        seg.elasticity = CUSHION_E
        seg.friction = 0.0  # no spin pick-up: bounces keep the tangential speed
        space.add(seg)

    def begin(arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> None:
        a, b = arbiter.shapes
        if table.first_contact_time is None and table.cue.shape in (a, b) and isinstance(a, pymunk.Circle) and isinstance(b, pymunk.Circle):
            table.first_contact_time = table.time

    space.on_collision(None, None, begin=begin)
    return table


def add_ball(table: Table, number: int, x: float, y: float) -> Ball:
    body = pymunk.Body()
    body.position = (x, y)
    shape = pymunk.Circle(body, BALL_R)
    shape.density = 1.0
    shape.elasticity = BALL_E
    shape.friction = 0.05
    table.space.add(body, shape)
    ball = Ball(number, shape)
    table.balls.append(ball)
    return ball


def strike(table: Table, speed: float, angle_deg: float) -> None:
    a = math.radians(angle_deg)
    table.cue.body.velocity = (speed * math.cos(a), speed * math.sin(a))


def step(table: Table) -> None:
    table.space.step(DT)
    table.time += DT
    for b in table.balls:
        if b.potted:
            continue
        v = b.body.velocity
        speed = v.length
        if speed < STOP_SPEED:
            b.body.velocity = (0.0, 0.0)
        else:
            b.body.velocity = v * max(0.0, 1.0 - ROLL_DECEL * DT / speed)
        x, y = b.body.position
        if any(math.hypot(x - px, y - py) < POCKET_R for px, py in POCKETS) or not (
            TABLE_X0 - 30 < x < TABLE_X1 + 30 and TABLE_Y0 - 30 < y < TABLE_Y1 + 30
        ):
            b.potted = True
            table.space.remove(b.body, b.shape)


def max_speed(table: Table) -> float:
    return max((b.body.velocity.length for b in table.balls if not b.potted), default=0.0)
