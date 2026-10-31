"""The contract between a game and the shared benchmark machinery.

Every game asks the same kind of question: for each labeled target object, does an
event happen by the end of the episode (a robot survives, a ball is potted, a domino
falls)? A game provides physics, rendering in two styles, a deterministic replay with
perturbations (used for probabilistic labels), and the text shown to models.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

STYLES = ("flat", "cartoon")


@dataclass
class Episode:
    seed: int
    difficulty: str
    frames: list[np.ndarray]  # flat style: one solid color per class; source of ground-truth masks
    display_frames: list[np.ndarray]  # what models see (identical to `frames` for the flat style)
    labeled_frame: Image.Image  # first displayed frame with target ids drawn
    target_ids: list[int]
    recorded_ids: list[int]  # targets for which the event happened in this (recorded) run
    replay_key: dict  # whatever Game.replay needs to rebuild the exact pre-launch scene
    info: dict = field(default_factory=dict)  # model-visible extras (tasks.jsonl "info")
    extra: dict = field(default_factory=dict)  # hidden statistics (answers "extra")


@dataclass
class Game:
    name: str
    version: str
    target: str  # noun for a target object, e.g. "robot"
    event: str  # JSON key for the per-target event in model answers, e.g. "alive"
    question: str  # one-line human-readable question
    palette: dict[str, tuple[int, int, int]]  # flat-style class colors (CLASSES order)
    dynamic_classes: list[str]  # classes scored in the rollout track
    count_class: str  # class whose blobs are counted in the final frame
    fps: int
    width: int
    height: int
    context_frames: int
    total_frames: int
    difficulties: tuple[str, ...]
    generate: Callable[[int, str, str], Episode]  # (seed, difficulty, style) -> Episode
    # (seed, difficulty, replay_key, noise, rng) -> ids for which the event happens
    replay: Callable[[int, str, dict, dict, random.Random], list[int]]
    perturbations: dict[str, dict]  # name -> noise passed to replay
    label_perturbation: str  # which perturbation defines the outcome labels
    vlm_prompt: Callable[[int, str, str], str]  # (n_frames, frame times, ids) -> prompt text
    lut_path: Path  # fitted color table for the cartoon style
    outcome_baselines: dict[str, Callable[[dict], list[int]]] = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        return list(self.palette)
