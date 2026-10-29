"""Reference baselines. They read only model-visible inputs (tasks.jsonl + inputs/)."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .evaluate import game_of


def tasks_of(data_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (data_dir / "tasks.jsonl").read_text().splitlines() if line.strip()]


GENERIC_OUTCOME = {
    "all": lambda t: t["target_ids"],  # the event happens to every target
    "none": lambda t: [],
}
ROLLOUT = ("copy_last", "oracle")


def outcome_baselines(data_dir: Path) -> dict:
    return {**GENERIC_OUTCOME, **game_of(data_dir).outcome_baselines}


def run_baseline(name: str, data_dir: Path, out: Path) -> None:
    tasks = tasks_of(data_dir)
    if name in ROLLOUT:
        out.mkdir(parents=True, exist_ok=True)
        for t in tasks:
            if name == "oracle":  # ground-truth video as prediction (sanity check for the scorer)
                (out / f"{t['id']}.mp4").write_bytes((data_dir / "answers" / f"{t['id']}.mp4").read_bytes())
            else:  # freeze the last context frame: "nothing happens"
                last = iio.imread(data_dir / t["context_frames"][-1])[..., :3]
                frames = np.repeat(last[None], t["frames_to_predict"], axis=0)
                iio.imwrite(out / f"{t['id']}.mp4", frames, fps=30, codec="libx264", quality=9, macro_block_size=1)
        return
    fn = outcome_baselines(data_dir)[name]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t in tasks:
            f.write(json.dumps({"id": t["id"], "ids": fn(t)}) + "\n")
