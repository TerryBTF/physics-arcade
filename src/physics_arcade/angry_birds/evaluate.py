"""Scoring for the two tracks.

Outcome track: predictions.jsonl with {"id": ..., "pigs_alive": [ids]}.
Rollout track: a directory of <id>.mp4 (or .gif / .npy) videos. A video may contain
either only the future frames or the full clip including the context frames.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy import ndimage

from .render import CLASSES, DYNAMIC_CLASSES, segment

MIN_PIG_AREA = 40  # px; smaller pig-colored blobs are treated as noise
# Predicted class masks are opened with this structuring element to drop isolated
# pixels that video codecs create where two colors meet.
OPEN = np.ones((3, 3), dtype=bool)


def _load_answers(data_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((data_dir / "answers").glob("ep_*.json"))}


def _load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def eval_outcome(data_dir: Path, pred_path: Path) -> dict:
    answers = _load_answers(data_dir)
    preds = _load_jsonl(pred_path)
    tp = tn = fp = fn = exact = count_err = missing = 0
    for eid, ans in answers.items():
        pred = preds.get(eid)
        if pred is None:
            missing += 1
        pred_alive = set(pred["pigs_alive"]) if pred else set()
        gt_alive = set(ans["pigs_alive"])
        for pid in ans["pig_ids"]:
            p, g = pid in pred_alive, pid in gt_alive
            tp += p and g
            tn += (not p) and (not g)
            fp += p and not g
            fn += (not p) and g
        exact += pred_alive & set(ans["pig_ids"]) == gt_alive
        count_err += abs(len(pred_alive & set(ans["pig_ids"])) - len(gt_alive))
    n = len(answers)
    total = tp + tn + fp + fn
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "episodes": n,
        "missing_predictions": missing,
        "pig_accuracy": (tp + tn) / max(total, 1),
        "pig_balanced_accuracy": (tpr + tnr) / 2,
        "episode_exact_match": exact / max(n, 1),
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


def _inter_union(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(len(DYNAMIC_CLASSES), 2) array of [intersection, union] pixel counts."""
    out = np.zeros((len(DYNAMIC_CLASSES), 2), dtype=np.int64)
    for i, name in enumerate(DYNAMIC_CLASSES):
        c = CLASSES.index(name)
        p = ndimage.binary_opening(pred == c, structure=OPEN)
        g = gt == c
        out[i] = np.logical_and(p, g).sum(), np.logical_or(p, g).sum()
    return out


def _class_ious(iu: np.ndarray) -> dict[str, float]:
    """IoU per class, skipping classes absent from both prediction and ground truth."""
    return {name: iu[i, 0] / iu[i, 1] for i, name in enumerate(DYNAMIC_CLASSES) if iu[i, 1]}


def _mean(ious: dict[str, float]) -> float:
    return float(np.mean(list(ious.values()))) if ious else 1.0


def count_pigs(mask: np.ndarray) -> int:
    labels, n = ndimage.label(mask == CLASSES.index("pig"))
    if n == 0:
        return 0
    areas = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    return int((areas >= MIN_PIG_AREA).sum())


def _score_episode(args: tuple[Path, Path, str, int, int]) -> dict | None:
    data_dir, pred_dir, eid, n_future, n_total = args
    cands = [pred_dir / f"{eid}{ext}" for ext in (".mp4", ".gif", ".npy")]
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        return None
    gt = np.load(data_dir / "answers" / f"{eid}_masks.npz")["masks"]
    frames = _align(load_video(path), n_future, n_total)
    if frames.shape[1:3] != gt.shape[1:3]:
        raise ValueError(f"{path}: frame size {frames.shape[1:3]} != {gt.shape[1:3]}")
    # IoU accumulated over all future frames, so a frame where a class is tiny cannot dominate.
    iu = sum(_inter_union(segment(frames[t]), gt[t]) for t in range(n_future))
    final_pred = segment(frames[-1])
    return {
        "ious": _class_ious(iu),
        "final_ious": _class_ious(_inter_union(final_pred, gt[-1])),
        "pigs": count_pigs(final_pred),
    }


def eval_rollout(data_dir: Path, pred_dir: Path, workers: int = 8) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    n_total, n_ctx = manifest["total_frames"], manifest["context_frames"]
    n_future = n_total - n_ctx
    answers = _load_answers(data_dir)
    jobs = [(data_dir, pred_dir, eid, n_future, n_total) for eid in answers]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_score_episode, jobs, chunksize=4))
    per_class: dict[str, list[float]] = {c: [] for c in DYNAMIC_CLASSES}
    miou_all, miou_final, count_exact, count_err, missing = [], [], 0, 0, 0
    for ans, r in zip(answers.values(), results):
        n_gt = len(ans["pigs_alive"])
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
        count_exact += r["pigs"] == n_gt
        count_err += abs(r["pigs"] - n_gt)
    n = len(answers)
    return {
        "episodes": n,
        "missing_predictions": missing,
        "miou_all_frames": float(np.mean(miou_all)),
        "miou_final_frame": float(np.mean(miou_final)),
        "iou_per_class": {k: float(np.mean(v)) for k, v in per_class.items() if v},
        "final_pig_count_accuracy": count_exact / max(n, 1),
        "final_pig_count_mae": count_err / max(n, 1),
    }
