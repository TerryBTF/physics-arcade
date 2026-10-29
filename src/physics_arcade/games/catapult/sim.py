"""2D catapult physics simulation built on pymunk.

Coordinates are in pixels with y pointing up; the renderer flips them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pymunk

WIDTH, HEIGHT = 640, 360
GROUND_Y = 40.0
GRAVITY = 500.0
LAUNCH_POS = (100.0, 110.0)
DT = 1.0 / 120.0
FPS = 30
STEPS_PER_FRAME = int(round(1.0 / (DT * FPS)))

BALL_RADIUS = 8.0

# density, friction, elasticity, health
MATERIALS = {
    "wood": (1.0, 0.7, 0.1, 160.0),
    "stone": (2.5, 0.8, 0.05, 450.0),
    "ice": (0.8, 0.3, 0.1, 60.0),
    # Robots are squares; density is scaled by pi/4 so a robot of half-size r has the
    # mass of the circular target of radius r used before.
    "robot": (0.8 * math.pi / 4, 0.6, 0.2, 40.0),
    "ball": (4.0, 0.6, 0.2, math.inf),
}
# Impulse units that remove one health point.
DAMAGE_SCALE = 400.0
# Impulses below this cause no damage (filters out resting contacts).
DAMAGE_THRESHOLD = 2500.0


@dataclass
class Obj:
    """One dynamic object. `kind` is a material name; `robot_id` is set for robots."""

    kind: str
    shape: pymunk.Shape
    health: float
    robot_id: int | None = None
    size: tuple[float, float] | None = None  # (w, h) for blocks, (r, r) for circles
    alive: bool = True

    @property
    def body(self) -> pymunk.Body:
        return self.shape.body


@dataclass
class World:
    space: pymunk.Space
    objects: list[Obj] = field(default_factory=list)
    damage_enabled: bool = False
    ball: Obj | None = None
    first_contact_time: float | None = None
    time: float = 0.0

    def by_shape(self, shape: pymunk.Shape) -> Obj | None:
        return self._index.get(id(shape))

    def __post_init__(self) -> None:
        self._index: dict[int, Obj] = {}

    def add(self, obj: Obj) -> Obj:
        self.space.add(obj.body, obj.shape)
        self.objects.append(obj)
        self._index[id(obj.shape)] = obj
        return obj

    def kill(self, obj: Obj) -> None:
        if not obj.alive:
            return
        obj.alive = False
        self.space.remove(obj.body, obj.shape)

    @property
    def robots(self) -> list[Obj]:
        return [o for o in self.objects if o.kind == "robot"]


def new_world() -> World:
    space = pymunk.Space()
    space.gravity = (0.0, -GRAVITY)
    space.iterations = 30
    space.sleep_time_threshold = 0.5
    ground = pymunk.Segment(space.static_body, (-2000, GROUND_Y), (4000, GROUND_Y), 0.0)
    ground.friction = 0.9
    ground.elasticity = 0.1
    space.add(ground)
    world = World(space=space)

    def post_solve(arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> None:
        if not arbiter.is_first_contact:
            return
        objs = [world.by_shape(s) for s in arbiter.shapes]
        if world.ball is not None and world.first_contact_time is None and any(o is world.ball for o in objs):
            world.first_contact_time = world.time
        if not world.damage_enabled:
            return
        impulse = arbiter.total_impulse.length
        if impulse < DAMAGE_THRESHOLD:
            return
        damage = (impulse - DAMAGE_THRESHOLD) / DAMAGE_SCALE
        for o in objs:
            if o is not None and o.alive:
                o.health -= damage

    space.on_collision(None, None, post_solve=post_solve)
    return world


def add_block(world: World, material: str, x: float, y: float, w: float, h: float, angle: float = 0.0) -> Obj:
    density, friction, elasticity, health = MATERIALS[material]
    body = pymunk.Body()
    body.position = (x, y)
    body.angle = angle
    shape = pymunk.Poly.create_box(body, (w, h), radius=0.0)
    shape.density = density
    shape.friction = friction
    shape.elasticity = elasticity
    return world.add(Obj(kind=material, shape=shape, health=health, size=(w, h)))


def add_robot(world: World, robot_id: int, x: float, y: float, r: float) -> Obj:
    """A square robot with half-size r (side 2r) centred at (x, y)."""
    density, friction, elasticity, health = MATERIALS["robot"]
    body = pymunk.Body()
    body.position = (x, y)
    shape = pymunk.Poly.create_box(body, (2 * r, 2 * r), radius=0.0)
    shape.density = density
    shape.friction = friction
    shape.elasticity = elasticity
    return world.add(Obj(kind="robot", shape=shape, health=health, robot_id=robot_id, size=(2 * r, 2 * r)))


def launch_ball(world: World, speed: float, angle_deg: float) -> Obj:
    density, friction, elasticity, health = MATERIALS["ball"]
    body = pymunk.Body()
    body.position = LAUNCH_POS
    a = math.radians(angle_deg)
    body.velocity = (speed * math.cos(a), speed * math.sin(a))
    shape = pymunk.Circle(body, BALL_RADIUS)
    shape.density = density
    shape.friction = friction
    shape.elasticity = elasticity
    world.ball = world.add(Obj(kind="ball", shape=shape, health=health, size=(BALL_RADIUS, BALL_RADIUS)))
    return world.ball


def step(world: World) -> None:
    world.space.step(DT)
    world.time += DT
    for o in world.objects:
        if not o.alive:
            continue
        x, y = o.body.position
        if o.health <= 0 or y < -100 or x < -200 or x > WIDTH + 200:
            world.kill(o)


def settle(world: World, seconds: float) -> None:
    for _ in range(int(seconds / DT)):
        step(world)


def max_speed(world: World) -> float:
    return max((o.body.velocity.length for o in world.objects if o.alive), default=0.0)


def ballistic_angles(dx: float, dy: float, speed: float) -> list[float]:
    """Launch angles (degrees) that make an unobstructed projectile pass through (dx, dy)."""
    v2 = speed * speed
    disc = v2 * v2 - GRAVITY * (GRAVITY * dx * dx + 2 * dy * v2)
    if disc < 0 or dx <= 0:
        return []
    root = math.sqrt(disc)
    return sorted(math.degrees(math.atan((v2 + s * root) / (GRAVITY * dx))) for s in (-1, 1))


def ballistic_y(x: float, speed: float, angle_deg: float) -> float | None:
    """Height of the unobstructed trajectory at horizontal position x."""
    a = math.radians(angle_deg)
    vx, vy = speed * math.cos(a), speed * math.sin(a)
    if vx <= 0:
        return None
    t = (x - LAUNCH_POS[0]) / vx
    if t < 0:
        return None
    return LAUNCH_POS[1] + vy * t - 0.5 * GRAVITY * t * t
