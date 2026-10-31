"""Scoring for the two tracks, for any game.

Outcome track: predictions.jsonl with {"id": ..., "ids": [targets for which the event
happens]} and, optionally, "p": {id: probability}. Collisions are chaotic, so labels are
per-target event probabilities over perturbed replays (answers/<id>.json "p"). The
headline metric is balanced accuracy on stable targets, whose fate agrees in >= 90 % of
replays; the Brier score covers every target and rewards calibrated probabilities.
Results are also broken down by difficulty.

Rollout track: a directory of <id>.mp4 (or .gif / .npy) videos with either only the
future frames or the full clip. Predicted frames are segmented into classes and compared
with the simulator's masks (mean IoU of the game's dynamic classes). Each ground-truth
frame is matched with the best predicted frame within +-TIME_TOLERANCE frames per class,
so lower frame rates are not penalized for resampling.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy import ndimage

from .game import Game
from .segment import segment

STABLE = 0.9  # a target is stable when its fate agrees in at least this fraction of replays
TIME_TOLERANCE = 1  # frames (+-33 ms at 30 fps)
MIN_BLOB_AREA = 40  # px; smaller blobs of the counted class are treated as noise
# Predicted class masks are opened with this structuring element to drop isolated
# pixels that video codecs create where two colors meet.
OPEN = np.ones((3, 3), dtype=bool)


def load_manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "manifest.json").read_text())


def game_of(data_dir: Path) -> Game:
    from ..games import get_game

    return get_game(load_manifest(data_dir)["game"])


def load_answers(data_dir: Path) -> dict[str, dict]:
    tasks = [json.loads(line)["id"] for line in (data_dir / "tasks.jsonl").read_text().splitlines() if line]
    return {eid: json.loads((data_dir / "answers" / f"{eid}.json").read_text()) for eid in tasks}


def load_jsonl(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


# ---------------------------------------------------------------------------
# Outcome track


def balanced_accuracy(pred: list[bool], label: list[bool]) -> float | None:
    pos = [p for p, y in zip(pred, label, strict=True) if y]
    neg = [not p for p, y in zip(pred, label, strict=True) if not y]
    if not pos or not neg:
        return None
    return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2


def _outcome_metrics(rows: list[dict]) -> dict:
    """rows: one per target with keys pred (bool), q (pred prob), p (label prob), eid."""
    stable = [r for r in rows if r["p"] >= STABLE or r["p"] <= 1 - STABLE]
    by_ep: dict[str, list[dict]] = {}
    for r in rows:
        by_ep.setdefault(r["eid"], []).append(r)
    stable_eps = [rs for rs in by_ep.values() if all(r["p"] >= STABLE or r["p"] <= 1 - STABLE for r in rs)]
    return {
        "stable_balanced_accuracy": balanced_accuracy([r["pred"] for r in stable], [r["p"] > 0.5 for r in stable]),
        "stable_fraction": len(stable) / max(len(rows), 1),
        "all_balanced_accuracy": balanced_accuracy([r["pred"] for r in rows], [r["p"] > 0.5 for r in rows]),
        # Expected Brier score against an outcome drawn from the replay distribution.
        "brier": float(np.mean([(r["q"] - r["p"]) ** 2 + r["p"] * (1 - r["p"]) for r in rows])),
        "brier_floor": float(np.mean([r["p"] * (1 - r["p"]) for r in rows])),
        "stable_episode_exact_match": float(np.mean([all(r["pred"] == (r["p"] > 0.5) for r in rs) for rs in stable_eps])) if stable_eps else None,
        "stable_episodes": len(stable_eps),
        "targets": len(rows),
    }


def eval_outcome(data_dir: Path, pred_path: Path) -> dict:
    answers = load_answers(data_dir)
    preds = load_jsonl(pred_path)
    rows, missing = [], 0
    for eid, ans in answers.items():
        pred = preds.get(eid)
        missing += pred is None
        ids = set(pred["ids"]) if pred else set()
        q = {int(k): float(v) for k, v in (pred or {}).get("p", {}).items()}
        for k, p in ans["p"].items():
            t = int(k)
            rows.append({"eid": eid, "difficulty": ans["difficulty"], "pred": t in ids, "q": q.get(t, float(t in ids)), "p": p})
    out = {"episodes": len(answers), "missing_predictions": missing, **_outcome_metrics(rows)}
    diffs = sorted({r["difficulty"] for r in rows}, key=[a["difficulty"] for a in answers.values()].index)
    out["by_difficulty"] = {d: _outcome_metrics([r for r in rows if r["difficulty"] == d]) for d in diffs}
    return out


# ---------------------------------------------------------------------------
# Rollout track


def load_video(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    frames = iio.imread(path)
    if frames.ndim == 3:
        frames = frames[None]
    return frames[..., :3]


def align(frames: np.ndarray, n_future: int, n_total: int) -> np.ndarray:
    """Drop context frames if present, then nearest-neighbour resample to n_future frames."""
    if len(frames) == n_total:
        frames = frames[n_total - n_future :]
    if len(frames) != n_future:
        idx = np.round(np.linspace(0, len(frames) - 1, n_future)).astype(int)
        frames = frames[idx]
    return frames


class RolloutScorer:
    def __init__(self, game: Game):
        self.game = game
        self.dyn = [game.classes.index(c) for c in game.dynamic_classes]
        self.count_id = game.classes.index(game.count_class)

    def clean(self, pred: np.ndarray) -> np.ndarray:
        """Open each dynamic class's mask; removed pixels become 255 (no class)."""
        out = pred.copy()
        for c in self.dyn:
            m = pred == c
            out[m & ~ndimage.binary_opening(m, structure=OPEN)] = 255
        return out

    def iu(self, clean: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """(len(dynamic_classes), 2) array of [intersection, union] pixel counts."""
        out = np.zeros((len(self.dyn), 2), dtype=np.int64)
        for i, c in enumerate(self.dyn):
            p, g = clean == c, gt == c
            out[i] = np.logical_and(p, g).sum(), np.logical_or(p, g).sum()
        return out

    def inter_union(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return self.iu(self.clean(pred), gt)

    def tolerant_iu(self, cleans: list[np.ndarray], gt: np.ndarray, t: int) -> np.ndarray:
        """Per class, [inter, union] of the predicted frame within +-TIME_TOLERANCE of t matching gt best."""
        lo, hi = max(0, t - TIME_TOLERANCE), min(len(cleans) - 1, t + TIME_TOLERANCE)
        cands = np.stack([self.iu(cleans[k], gt) for k in range(lo, hi + 1)])  # (K, C, 2)
        iou = np.where(cands[..., 1] > 0, cands[..., 0] / np.maximum(cands[..., 1], 1), 1.0)
        return cands[iou.argmax(0), np.arange(cands.shape[1])]

    def class_ious(self, iu: np.ndarray) -> dict[str, float]:
        """IoU per class, skipping classes absent from both prediction and ground truth."""
        return {name: iu[i, 0] / iu[i, 1] for i, name in enumerate(self.game.dynamic_classes) if iu[i, 1]}

    def count(self, mask: np.ndarray) -> int:
        labels, n = ndimage.label(mask == self.count_id)
        if n == 0:
            return 0
        areas = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        return int((areas >= MIN_BLOB_AREA).sum())


def mean_iou(ious: dict[str, float]) -> float:
    return float(np.mean(list(ious.values()))) if ious else 1.0


def _score_episode(args: tuple[Path, Path, str]) -> dict | None:
    data_dir, pred_dir, eid = args
    manifest = load_manifest(data_dir)
    game, style = game_of(data_dir), manifest["style"]
    scorer = RolloutScorer(game)
    n_total, n_future = manifest["total_frames"], manifest["total_frames"] - manifest["context_frames"]
    gt = np.load(data_dir / "answers" / f"{eid}_masks.npz")["masks"]
    gt_count = scorer.count(gt[-1])
    path = next((p for p in (pred_dir / f"{eid}{e}" for e in (".mp4", ".gif", ".npy")) if p.exists()), None)
    if path is None:
        return {"missing": True, "gt_count": gt_count}
    frames = align(load_video(path), n_future, n_total)
    if frames.shape[1:3] != gt.shape[1:3]:
        raise ValueError(f"{path}: frame size {frames.shape[1:3]} != {gt.shape[1:3]}")
    cleans = [scorer.clean(segment(game, f, style)) for f in frames]
    # IoU accumulated over all future frames, so a frame where a class is tiny cannot dominate.
    iu = sum(scorer.tolerant_iu(cleans, gt[t], t) for t in range(n_future))
    final_pred = segment(game, frames[-1], style)
    return {
        "missing": False,
        "ious": scorer.class_ious(iu),
        "final_ious": scorer.class_ious(scorer.inter_union(final_pred, gt[-1])),
        "count": scorer.count(final_pred),
        "gt_count": gt_count,
    }


def eval_rollout(data_dir: Path, pred_dir: Path, workers: int = 8) -> dict:
    game = game_of(data_dir)
    answers = load_answers(data_dir)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_score_episode, [(data_dir, pred_dir, e) for e in answers], chunksize=4))
    per_class: dict[str, list[float]] = {c: [] for c in game.dynamic_classes}
    miou_all, miou_final, count_exact, count_err, missing = [], [], 0, 0, 0
    for r in results:
        if r["missing"]:
            missing += 1
            miou_all.append(0.0)
            miou_final.append(0.0)
            count_err += r["gt_count"]
            continue
        for k, v in r["ious"].items():
            per_class[k].append(v)
        miou_all.append(mean_iou(r["ious"]))
        miou_final.append(mean_iou(r["final_ious"]))
        count_exact += r["count"] == r["gt_count"]
        count_err += abs(r["count"] - r["gt_count"])
    n = len(answers)
    return {
        "episodes": n,
        "missing_predictions": missing,
        "miou_all_frames": float(np.mean(miou_all)),
        "miou_final_frame": float(np.mean(miou_final)),
        "iou_per_class": {k: float(np.mean(v)) for k, v in per_class.items() if v},
        f"final_{game.count_class}_count_accuracy": count_exact / max(n, 1),
        f"final_{game.count_class}_count_mae": count_err / max(n, 1),
    }
