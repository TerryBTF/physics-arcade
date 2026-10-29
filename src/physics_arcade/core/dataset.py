"""Write a game's episodes to disk in the benchmark layout.

    <out>/
      manifest.json
      tasks.jsonl                    # one line per episode (model-visible)
      inputs/<id>/context_XX.png     # the context frames, lossless
      inputs/<id>/context.mp4        # same, as video
      inputs/<id>/labeled.png        # first frame with target ids drawn
      answers/<id>.json              # outcome labels (hidden from models)
      answers/<id>_masks.npz         # per-frame class masks for the future frames
      answers/<id>.mp4               # full ground-truth video, for viewing

Episodes are spread evenly over the game's difficulties; the seed of episode i of
difficulty d is seed_offset + 100_000 * d_index + i.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from .game import Game
from .segment import segment

LABEL_RUNS = 50
SEED_STRIDE = 100_000


def survival_probabilities(game: Game, seed: int, difficulty: str, replay_key: dict, target_ids: list[int],
                           perturbation: str, runs: int, rng_tag: str) -> dict[int, float]:
    """Fraction of perturbed replays in which the event happens, per target."""
    noise = game.perturbations[perturbation]
    rng = random.Random(f"{rng_tag}-{game.name}-{seed}")
    counts = dict.fromkeys(target_ids, 0)
    for _ in range(runs):
        for i in game.replay(seed, difficulty, replay_key, noise, rng):
            counts[i] += 1
    return {i: c / runs for i, c in counts.items()}


def episode_id(difficulty: str, i: int) -> str:
    return f"{difficulty}_{i:04d}"


def _write_one(args: tuple[str, int, str, str, str, str]) -> dict:
    from ..games import get_game

    game_name, seed, difficulty, eid, out, style = args
    game = get_game(game_name)
    out_dir = Path(out)
    ep = game.generate(seed, difficulty, style)
    n_ctx = game.context_frames

    inp = out_dir / "inputs" / eid
    inp.mkdir(parents=True, exist_ok=True)
    for i in range(n_ctx):
        Image.fromarray(ep.display_frames[i]).save(inp / f"context_{i:02d}.png")
    iio.imwrite(inp / "context.mp4", np.stack(ep.display_frames[:n_ctx]), fps=game.fps, codec="libx264", quality=9, macro_block_size=1)
    ep.labeled_frame.save(inp / "labeled.png")

    ans = out_dir / "answers"
    ans.mkdir(parents=True, exist_ok=True)
    masks = np.stack([segment(game, f) for f in ep.frames[n_ctx:]]).astype(np.uint8)
    np.savez_compressed(ans / f"{eid}_masks.npz", masks=masks)
    iio.imwrite(ans / f"{eid}.mp4", np.stack(ep.display_frames), fps=game.fps, codec="libx264", quality=9, macro_block_size=1)
    p = survival_probabilities(game, seed, difficulty, ep.replay_key, ep.target_ids, game.label_perturbation, LABEL_RUNS, "label")
    answer = {
        "id": eid,
        "game": game.name,
        "seed": seed,
        "difficulty": difficulty,
        "target_ids": ep.target_ids,
        # The single run shown in the video and used for the rollout masks.
        "recorded_ids": ep.recorded_ids,
        # Outcome-track labels: per target, fraction of perturbed replays in which the event happens.
        "p": {str(k): v for k, v in p.items()},
        "label": {"perturbation": game.label_perturbation, "noise": game.perturbations[game.label_perturbation], "runs": LABEL_RUNS},
        "replay_key": ep.replay_key,
        "extra": ep.extra,
    }
    (ans / f"{eid}.json").write_text(json.dumps(answer, indent=1))
    return {
        "id": eid,
        "seed": seed,
        "difficulty": difficulty,
        "context_frames": [f"inputs/{eid}/context_{i:02d}.png" for i in range(n_ctx)],
        "context_video": f"inputs/{eid}/context.mp4",
        "labeled_frame": f"inputs/{eid}/labeled.png",
        "target_ids": ep.target_ids,
        "question": game.question,
        "frames_to_predict": game.total_frames - n_ctx,
        "info": ep.info,
    }


def write_dataset(game: Game, out_dir: Path, n_per_difficulty: int, style: str = "flat", workers: int = 8,
                  seed_offset: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (game.name, seed_offset + SEED_STRIDE * d + i, diff, episode_id(diff, i), str(out_dir), style)
        for d, diff in enumerate(game.difficulties)
        for i in range(n_per_difficulty)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = list(pool.map(_write_one, jobs, chunksize=2))
    with open(out_dir / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    manifest = {
        "game": game.name,
        "version": game.version,
        "style": style,
        "difficulties": {d: n_per_difficulty for d in game.difficulties},
        "num_episodes": len(tasks),
        "target": game.target,
        "event": game.event,
        "question": game.question,
        "fps": game.fps,
        "width": game.width,
        "height": game.height,
        "context_frames": game.context_frames,
        "total_frames": game.total_frames,
        "coordinates": "pixels, origin bottom-left, y up",
        "palette": game.palette,
        "dynamic_classes": game.dynamic_classes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
