"""Reference baselines. They read only model-visible inputs (tasks.jsonl + inputs/)."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .sim import BALL_RADIUS, ballistic_y


def _tasks(data_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (data_dir / "tasks.jsonl").read_text().splitlines() if line.strip()]


def _ballistic_hits(task: dict) -> set[int]:
    """Robots that the unobstructed parabola passes through (ignores blocks and collapses)."""
    speed, angle = task["launch"]["speed"], task["launch"]["angle_deg"]
    hit = set()
    for pid, p in task["robots"].items():
        for dx in np.linspace(-p["r"], p["r"], 9):
            y = ballistic_y(p["x"] + dx, speed, angle)
            if y is not None and abs(y - p["y"]) <= p["r"] + BALL_RADIUS:
                hit.add(int(pid))
                break
    return hit


OUTCOME_BASELINES = {
    "all_alive": lambda t: t["robot_ids"],
    "none_alive": lambda t: [],
    "ballistic": lambda t: sorted(set(t["robot_ids"]) - _ballistic_hits(t)),
}


def run_outcome_baseline(name: str, data_dir: Path, out_path: Path) -> None:
    fn = OUTCOME_BASELINES[name]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in _tasks(data_dir):
            f.write(json.dumps({"id": t["id"], "robots_alive": fn(t)}) + "\n")


def _copy_last(data_dir: Path, task: dict) -> np.ndarray:
    last = iio.imread(data_dir / task["context_frames"][-1])[..., :3]
    return np.repeat(last[None], task["frames_to_predict"], axis=0)


def _no_ball(data_dir: Path, task: dict) -> np.ndarray:
    """Freeze the scene and remove the ball: "nothing happens"."""
    frames = _copy_last(data_dir, task)
    first = iio.imread(data_dir / task["context_frames"][0])[..., :3]
    ball = np.all(frames[0] == (220, 30, 30), axis=-1)
    frames[:, ball] = first[0, 0]  # top-left pixel is always sky
    return frames


ROLLOUT_BASELINES = {"copy_last": _copy_last, "no_ball": _no_ball}


def run_rollout_baseline(name: str, data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = ROLLOUT_BASELINES[name]
    for t in _tasks(data_dir):
        iio.imwrite(out_dir / f"{t['id']}.mp4", fn(data_dir, t), fps=30, codec="libx264", quality=9, macro_block_size=1)


def write_oracle_rollouts(data_dir: Path, out_dir: Path) -> None:
    """Copy the ground-truth videos as predictions (sanity check for the scorer)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in _tasks(data_dir):
        (out_dir / f"{t['id']}.mp4").write_bytes((data_dir / "answers" / f"{t['id']}.mp4").read_bytes())
