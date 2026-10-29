"""Side-view domino chain built on pymunk. Pixels, origin bottom-left, y up."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pymunk

WIDTH, HEIGHT = 640, 360
FPS = 30
STEPS_PER_FRAME = 8
DT = 1.0 / (FPS * STEPS_PER_FRAME)
GRAVITY = 600.0
FLOOR_Y = 50.0
BALL_R = 7.0
DOMINO_W = 7.0
# density, friction, elasticity
MATERIALS = {"light": (0.6, 0.6, 0.05), "heavy": (6.0, 0.6, 0.05), "ball": (2.5, 0.5, 0.2)}
FALLEN_ANGLE = math.radians(30)  # a domino tilted further than this has fallen


@dataclass
class Domino:
    number: int
    material: str
    shape: pymunk.Poly
    size: tuple[float, float]

    @property
    def body(self) -> pymunk.Body:
        return self.shape.body

    def fallen(self) -> bool:
        a = (self.body.angle + math.pi) % (2 * math.pi) - math.pi
        return abs(a) > FALLEN_ANGLE or self.body.position.y < FLOOR_Y - 5


@dataclass
class Scene:
    space: pymunk.Space
    dominoes: list[Domino] = field(default_factory=list)
    platforms: list[tuple[float, float, float]] = field(default_factory=list)  # (x0, x1, top y)
    ball: pymunk.Shape | None = None
    time: float = 0.0
    first_contact_time: float | None = None


def new_scene() -> Scene:
    space = pymunk.Space()
    space.gravity = (0.0, -GRAVITY)
    space.iterations = 30
    floor = pymunk.Segment(space.static_body, (-1000, FLOOR_Y), (2000, FLOOR_Y), 0.0)
    floor.friction = 0.8
    space.add(floor)
    scene = Scene(space=space)

    def begin(arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> None:
        if scene.first_contact_time is None and scene.ball is not None and scene.ball in arbiter.shapes:
            other = arbiter.shapes[1] if arbiter.shapes[0] is scene.ball else arbiter.shapes[0]
            if other.body.body_type == pymunk.Body.DYNAMIC:
                scene.first_contact_time = scene.time

    space.on_collision(None, None, begin=begin)
    return scene


def add_platform(scene: Scene, x0: float, x1: float, top: float) -> None:
    box = pymunk.Poly(scene.space.static_body, [(x0, FLOOR_Y), (x1, FLOOR_Y), (x1, top), (x0, top)])
    box.friction = 0.8
    scene.space.add(box)
    scene.platforms.append((x0, x1, top))


def add_domino(scene: Scene, number: int, material: str, x: float, base: float, h: float) -> Domino:
    density, friction, elasticity = MATERIALS[material]
    body = pymunk.Body()
    body.position = (x, base + h / 2)
    shape = pymunk.Poly.create_box(body, (DOMINO_W, h))
    shape.density, shape.friction, shape.elasticity = density, friction, elasticity
    scene.space.add(body, shape)
    d = Domino(number, material, shape, (DOMINO_W, h))
    scene.dominoes.append(d)
    return d


def toss_ball(scene: Scene, x: float, y: float, vx: float, vy: float) -> None:
    density, friction, elasticity = MATERIALS["ball"]
    body = pymunk.Body()
    body.position = (x, y)
    body.velocity = (vx, vy)
    shape = pymunk.Circle(body, BALL_R)
    shape.density, shape.friction, shape.elasticity = density, friction, elasticity
    scene.space.add(body, shape)
    scene.ball = shape


def step(scene: Scene) -> None:
    scene.space.step(DT)
    scene.time += DT


def max_speed(scene: Scene) -> float:
    bodies = [d.body for d in scene.dominoes] + ([scene.ball.body] if scene.ball else [])
    return max((b.velocity.length for b in bodies if -50 < b.position.x < WIDTH + 50), default=0.0)
