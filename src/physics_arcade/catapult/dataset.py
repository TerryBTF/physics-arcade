"""Write episodes to disk in the benchmark layout.

    <out>/
      manifest.json
      tasks.jsonl                    # one line per episode (model-visible)
      inputs/<id>/context_XX.png     # first CONTEXT_FRAMES frames, lossless
      inputs/<id>/context.mp4        # same, as video
      inputs/<id>/labeled.png        # first frame with robot ids drawn
      answers/<id>.json              # outcome ground truth (hidden from models)
      answers/<id>_masks.npz         # per-frame class masks for the future frames
      answers/<id>.mp4               # full ground-truth video, for viewing
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from .episode import CONTEXT_FRAMES, TOTAL_FRAMES, generate_episode
from .predictability import CONDITIONS, LABEL_CONDITION, LABEL_RUNS, robot_survival
from .render import CLASSES, PALETTE, segment
from .sim import FPS, HEIGHT, WIDTH

VERSION = "catapult-v0"
STYLES = ("flat", "cartoon")

QUESTION = (
    "A cannonball has just been fired from the catapult on the left. The images show the first "
    f"{CONTEXT_FRAMES} frames ({CONTEXT_FRAMES / FPS:.1f} s at {FPS} fps). Robots are labeled with ids in "
    "labeled.png. After everything comes to rest, which robots are still alive? Answer with a list of robot ids."
)


def episode_id(seed: int) -> str:
    return f"ep_{seed:05d}"


def _write_one(args: tuple[int, str, str]) -> dict:
    seed, out, style = args
    out_dir = Path(out)
    ep = generate_episode(seed, style)
    eid = episode_id(seed)
    inp = out_dir / "inputs" / eid
    inp.mkdir(parents=True, exist_ok=True)
    for i in range(CONTEXT_FRAMES):
        Image.fromarray(ep.display_frames[i]).save(inp / f"context_{i:02d}.png")
    iio.imwrite(inp / "context.mp4", np.stack(ep.display_frames[:CONTEXT_FRAMES]), fps=FPS, codec="libx264", quality=9, macro_block_size=1)
    ep.labeled_first_frame.save(inp / "labeled.png")

    ans = out_dir / "answers"
    ans.mkdir(parents=True, exist_ok=True)
    masks = np.stack([segment(f) for f in ep.frames[CONTEXT_FRAMES:]]).astype(np.uint8)
    np.savez_compressed(ans / f"{eid}_masks.npz", masks=masks)
    iio.imwrite(ans / f"{eid}.mp4", np.stack(ep.display_frames), fps=FPS, codec="libx264", quality=9, macro_block_size=1)
    answer = {
        "id": eid,
        "robots_alive": ep.robots_alive,
        "robot_ids": sorted(ep.robots_initial),
        "blocks_initial": ep.blocks_initial,
        "blocks_destroyed": ep.blocks_destroyed,
        "first_contact_time": ep.first_contact_time,
        "attempt": ep.attempt,
        "launch": {"speed": ep.speed, "angle_deg": ep.angle},
        # Outcome-track labels: survival probability of each robot over perturbed replays.
        # `robots_alive` above is the single recorded run shown in the video.
        "p_alive": {str(k): v for k, v in robot_survival(seed, ep.attempt, sorted(ep.robots_initial)).items()},
        "label": {"condition": LABEL_CONDITION, "runs": LABEL_RUNS, "noise": CONDITIONS[LABEL_CONDITION]},
    }
    (ans / f"{eid}.json").write_text(json.dumps(answer, indent=1))

    return {
        "id": eid,
        "seed": seed,
        "context_frames": [f"inputs/{eid}/context_{i:02d}.png" for i in range(CONTEXT_FRAMES)],
        "context_video": f"inputs/{eid}/context.mp4",
        "labeled_frame": f"inputs/{eid}/labeled.png",
        "robot_ids": sorted(ep.robots_initial),
        "robots": {str(k): {"x": round(x, 1), "y": round(y, 1), "r": round(r, 1)} for k, (x, y, r) in ep.robots_initial.items()},
        "launch": {"speed": round(ep.speed, 2), "angle_deg": round(ep.angle, 2)},
        "question": QUESTION,
        "frames_to_predict": TOTAL_FRAMES - CONTEXT_FRAMES,
    }


def write_dataset(out_dir: Path, n: int, seed_offset: int = 0, workers: int = 8, style: str = "flat") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(seed_offset + i, str(out_dir), style) for i in range(n)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = list(pool.map(_write_one, jobs, chunksize=4))
    with open(out_dir / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    manifest = {
        "version": VERSION,
        "style": style,
        "num_episodes": n,
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "context_frames": CONTEXT_FRAMES,
        "total_frames": TOTAL_FRAMES,
        "coordinates": "pixels, origin bottom-left, y up",
        "palette": {c: PALETTE[c] for c in CLASSES},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
