"""Shared fixtures: one tiny dataset per game (one episode per difficulty), built once per run."""

from __future__ import annotations

import pytest

from physics_arcade.core.dataset import write_dataset
from physics_arcade.games import NAMES, get_game


@pytest.fixture(scope="session", params=NAMES)
def tiny_dataset(request, tmp_path_factory):
    game = get_game(request.param)
    out = tmp_path_factory.mktemp(request.param) / "data"
    write_dataset(game, out, n_per_difficulty=1, style="flat", workers=3)
    return game, out
