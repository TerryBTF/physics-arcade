"""Registry of games. Each game package exposes a `GAME` (physics_arcade.core.game.Game)."""

from __future__ import annotations

from importlib import import_module

from ..core.game import Game

NAMES = ("catapult", "pool", "dominoes")


def get_game(name: str) -> Game:
    if name not in NAMES:
        raise KeyError(f"unknown game {name!r}; available: {', '.join(NAMES)}")
    return import_module(f".{name}", __name__).GAME
