"""Physics determinism: the property every label and fingerprint relies on."""

from __future__ import annotations

import random

import numpy as np
import pytest

from physics_arcade.core.segment import segment
from physics_arcade.games import NAMES, get_game


@pytest.mark.parametrize("name", NAMES)
def test_generation_is_deterministic(name):
    game = get_game(name)
    a, b = game.generate(3, "medium", "flat"), game.generate(3, "medium", "flat")
    assert a.recorded_ids == b.recorded_ids and a.target_ids == b.target_ids
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames, strict=True))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_noise_free_replay_reproduces_the_recording(name, difficulty):
    game = get_game(name)
    ep = game.generate(5, difficulty, "flat")
    replayed = game.replay(5, difficulty, ep.replay_key, {"speed": 0.0, "angle": 0.0, "pos": 0.0}, random.Random(0))
    assert sorted(replayed) == sorted(ep.recorded_ids)


@pytest.mark.parametrize("name", NAMES)
def test_episode_shape(name):
    game = get_game(name)
    ep = game.generate(1, "easy", "cartoon")
    assert len(ep.frames) == len(ep.display_frames) == game.total_frames
    assert ep.frames[0].shape == (game.height, game.width, 3)
    assert set(ep.recorded_ids) <= set(ep.target_ids) and ep.target_ids
    assert ep.labeled_frame.size == (game.width, game.height)


@pytest.mark.parametrize("name", NAMES)
def test_flat_frames_segment_exactly(name):
    """Every pixel of a flat frame is a palette color, so segmentation is lossless."""
    game = get_game(name)
    frame = game.generate(2, "easy", "flat").frames[-1]
    seg = segment(game, frame)
    palette = np.array(list(game.palette.values()), dtype=np.uint8)
    assert np.array_equal(palette[seg], frame)


@pytest.mark.parametrize("name", NAMES)
def test_cartoon_segmentation_is_accurate(name):
    game = get_game(name)
    ep = game.generate(4, "medium", "cartoon")
    acc = np.mean([(segment(game, d, "cartoon") == segment(game, f)).mean() for d, f in zip(ep.display_frames[::30], ep.frames[::30], strict=True)])
    assert acc > 0.97
