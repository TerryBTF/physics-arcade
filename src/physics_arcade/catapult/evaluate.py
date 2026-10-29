"""Scoring for the two tracks.

Outcome track: predictions.jsonl with {"id": ..., "robots_alive": [ids]} and, optionally,
"p_alive": {id: probability}. Collisions are chaotic, so labels are each robot's survival
probability over perturbed replays (answers/<id>.json "p_alive"). The headline metric
is balanced accuracy on stable robots, whose fate agrees in >= 90 % of replays; the
Brier score covers every robot and rewards calibrated probabilities.

Rollout track: a directory of <id>.mp4 (or .gif / .npy) videos. A video may contain
either only the future frames or the full clip including the context frames. Each
ground-truth frame is compared with the best-matching predicted frame within
+-TIME_TOLERANCE frames (per class), so models generating at a lower frame rate are
not penalized for resampling.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy import ndimage

from .render import CLASSES, DYNAMIC_CLASSES, segment

MIN_ROBOT_AREA = 40  # px; smaller robot-colored blobs are treated as noise
# Predicted class masks are opened with this structuring element to drop isolated
# pixels that video codecs create where two colors meet.
OPEN = np.ones((3, 3), dtype=bool)
STABLE = 0.9  # a robot is stable when its fate agrees in at least this fraction of replays
TIME_TOLERANCE = 1  # frames (+-33 ms at 30 fps)


def _load_answers(data_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((data_dir / "answers").glob("ep_*.json"))}


def _load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def _balanced_accuracy(pred: list[bool], label: list[bool]) -> float | None:
    pos = [p for p, y in zip(pred, label) if y]
    neg = [not p for p, y in zip(pred, label) if not y]
    if not pos or not neg:
        return None
    return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2


def eval_outcome(data_dir: Path, pred_path: Path) -> dict:
    answers = _load_answers(data_dir)
    preds = _load_jsonl(pred_path)
    missing = 0
    pred_b, label, stable, brier, brier_floor = [], [], [], [], []
    exact_stable_eps = n_stable_eps = 0
    count_err = 0.0
    for eid, ans in answers.items():
        pred = preds.get(eid)
        missing += pred is None
        pred_alive = set(pred["robots_alive"]) if pred else set()
        pred_p = {int(k): float(v) for k, v in (pred or {}).get("p_alive", {}).items()}
        p_alive = {int(k): v for k, v in ans["p_alive"].items()}
        ep_ok = True
        for pid in ans["robot_ids"]:
            p = p_alive[pid]
            b = pid in pred_alive
            q = pred_p.get(pid, float(b))
            pred_b.append(b)
            label.append(p > 0.5)
            stable.append(p >= STABLE or p <= 1 - STABLE)
            # Expected Brier score against an outcome drawn from the replay distribution.
            brier.append((q - p) ** 2 + p * (1 - p))
            brier_floor.append(p * (1 - p))
            if stable[-1] and b != label[-1]:
                ep_ok = False
        all_stable = all(p >= STABLE or p <= 1 - STABLE for p in p_alive.values())
        n_stable_eps += all_stable
        exact_stable_eps += all_stable and ep_ok
        count_err += abs(len(pred_alive & set(ans["robot_ids"])) - sum(p_alive.values()))
    n = len(answers)
    sp = [b for b, s in zip(pred_b, stable) if s]
    sl = [y for y, s in zip(label, stable) if s]
    return {
        "episodes": n,
        "missing_predictions": missing,
        "stable_robot_balanced_accuracy": _balanced_accuracy(sp, sl),
        "stable_robot_fraction": sum(stable) / max(len(stable), 1),
        "all_robot_balanced_accuracy": _balanced_accuracy(pred_b, label),
        "brier": float(np.mean(brier)),
        "brier_floor": float(np.mean(brier_floor)),
        "stable_episode_exact_match": exact_stable_eps / max(n_stable_eps, 1),
        "stable_episodes": n_stable_eps,
        "alive_count_mae": count_err / max(n, 1),
    }


def load_video(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    frames = iio.imread(path)
    if frames.ndim == 3:
        frames = frames[None]
    return frames[..., :3]


def _align(frames: np.ndarray, n_future: int, n_total: int) -> np.ndarray:
    """Drop context frames if present, then nearest-neighbour resample to n_future frames."""
    if len(frames) == n_total:
        frames = frames[n_total - n_future :]
    if len(frames) != n_future:
        idx = np.round(np.linspace(0, len(frames) - 1, n_future)).astype(int)
        frames = frames[idx]
    return frames


def _clean(pred: np.ndarray) -> np.ndarray:
    """Open each dynamic class's mask; removed pixels become 255 (no class)."""
    out = pred.copy()
    for name in DYNAMIC_CLASSES:
        c = CLASSES.index(name)
        m = pred == c
        out[m & ~ndimage.binary_opening(m, structure=OPEN)] = 255
    return out


def _iu(clean: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(len(DYNAMIC_CLASSES), 2) array of [intersection, union] pixel counts."""
    out = np.zeros((len(DYNAMIC_CLASSES), 2), dtype=np.int64)
    for i, name in enumerate(DYNAMIC_CLASSES):
        c = CLASSES.index(name)
        p, g = clean == c, gt == c
        out[i] = np.logical_and(p, g).sum(), np.logical_or(p, g).sum()
    return out


def _inter_union(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return _iu(_clean(pred), gt)


def _tolerant_iu(cleans: list[np.ndarray], gt: np.ndarray, t: int) -> np.ndarray:
    """Per class, the [inter, union] of the predicted frame within +-TIME_TOLERANCE of t
    that best matches gt (frame t)."""
    lo, hi = max(0, t - TIME_TOLERANCE), min(len(cleans) - 1, t + TIME_TOLERANCE)
    cands = np.stack([_iu(cleans[k], gt) for k in range(lo, hi + 1)])  # (K, C, 2)
    iou = np.where(cands[..., 1] > 0, cands[..., 0] / np.maximum(cands[..., 1], 1), 1.0)
    best = iou.argmax(0)
    return cands[best, np.arange(cands.shape[1])]


def _class_ious(iu: np.ndarray) -> dict[str, float]:
    """IoU per class, skipping classes absent from both prediction and ground truth."""
    return {name: iu[i, 0] / iu[i, 1] for i, name in enumerate(DYNAMIC_CLASSES) if iu[i, 1]}


def _mean(ious: dict[str, float]) -> float:
    return float(np.mean(list(ious.values()))) if ious else 1.0


def count_robots(mask: np.ndarray) -> int:
    labels, n = ndimage.label(mask == CLASSES.index("robot"))
    if n == 0:
        return 0
    areas = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    return int((areas >= MIN_ROBOT_AREA).sum())


def _score_episode(args: tuple[Path, Path, str, int, int, str]) -> dict | None:
    data_dir, pred_dir, eid, n_future, n_total, style = args
    cands = [pred_dir / f"{eid}{ext}" for ext in (".mp4", ".gif", ".npy")]
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        return None
    gt = np.load(data_dir / "answers" / f"{eid}_masks.npz")["masks"]
    frames = _align(load_video(path), n_future, n_total)
    if frames.shape[1:3] != gt.shape[1:3]:
        raise ValueError(f"{path}: frame size {frames.shape[1:3]} != {gt.shape[1:3]}")
    cleans = [_clean(segment(f, style)) for f in frames]
    # IoU accumulated over all future frames, so a frame where a class is tiny cannot dominate.
    iu = sum(_tolerant_iu(cleans, gt[t], t) for t in range(n_future))
    final_pred = segment(frames[-1], style)
    return {
        "ious": _class_ious(iu),
        "final_ious": _class_ious(_inter_union(final_pred, gt[-1])),
        "robots": count_robots(final_pred),
    }


def eval_rollout(data_dir: Path, pred_dir: Path, workers: int = 8) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    n_total, n_ctx = manifest["total_frames"], manifest["context_frames"]
    n_future = n_total - n_ctx
    answers = _load_answers(data_dir)
    style = manifest.get("style", "flat")
    jobs = [(data_dir, pred_dir, eid, n_future, n_total, style) for eid in answers]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_score_episode, jobs, chunksize=4))
    per_class: dict[str, list[float]] = {c: [] for c in DYNAMIC_CLASSES}
    miou_all, miou_final, count_exact, count_err, missing = [], [], 0, 0, 0
    for ans, r in zip(answers.values(), results):
        n_gt = len(ans["robots_alive"])
        if r is None:
            missing += 1
            miou_all.append(0.0)
            miou_final.append(0.0)
            count_err += n_gt
            continue
        for k, v in r["ious"].items():
            per_class[k].append(v)
        miou_all.append(_mean(r["ious"]))
        miou_final.append(_mean(r["final_ious"]))
        count_exact += r["robots"] == n_gt
        count_err += abs(r["robots"] - n_gt)
    n = len(answers)
    return {
        "episodes": n,
        "missing_predictions": missing,
        "miou_all_frames": float(np.mean(miou_all)),
        "miou_final_frame": float(np.mean(miou_final)),
        "iou_per_class": {k: float(np.mean(v)) for k, v in per_class.items() if v},
        "final_robot_count_accuracy": count_exact / max(n, 1),
        "final_robot_count_mae": count_err / max(n, 1),
    }
