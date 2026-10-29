"""Side-by-side comparison videos of a predicted rollout against ground truth.

Layout (each panel 640x360):
    ground-truth video     | predicted video
    ground-truth classes   | predicted classes, errors in magenta
with a header showing the frame index, that frame's mIoU, and live robot counts.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evaluate import RolloutScorer, align, game_of, load_manifest, load_video, mean_iou
from .segment import segment

HEADER = 28
ERROR = np.array([255, 0, 255], dtype=np.uint8)


def _header(width: int, text: str) -> np.ndarray:
    img = Image.new("RGB", (width, HEADER), (24, 24, 28))
    ImageDraw.Draw(img).text((8, 6), text, fill=(235, 235, 235), font=ImageFont.load_default(size=15))
    return np.asarray(img)


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(panel)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=13)
    box = d.textbbox((6, 4), text, font=font)
    d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def compare_video(data_dir: Path, pred_path: Path, eid: str, out_path: Path, model_name: str = "prediction") -> dict:
    manifest = load_manifest(data_dir)
    game = game_of(data_dir)
    scorer = RolloutScorer(game)
    style = manifest["style"]
    n_total, n_ctx = manifest["total_frames"], manifest["context_frames"]
    colors = np.array([game.palette[c] for c in game.classes] + [[0, 0, 0]] * (256 - len(game.classes)), dtype=np.uint8)
    dyn = np.array(scorer.dyn)
    gt_video = load_video(data_dir / "answers" / f"{eid}.mp4")
    gt_masks = np.load(data_dir / "answers" / f"{eid}_masks.npz")["masks"]
    pred = align(load_video(pred_path), n_total - n_ctx, n_total)

    frames, mious = [], []
    for t in range(n_total):
        gt_rgb = gt_video[t]
        if t < n_ctx:
            pred_rgb = gt_rgb
            gt_seg = pred_seg = segment(game, gt_rgb, style)
            err = colors[pred_seg]
            status = f"{eid} | frame {t:3d}/{n_total} | CONTEXT (given to the model)"
        else:
            f = t - n_ctx
            pred_rgb = pred[f]
            gt_seg, pred_seg = gt_masks[f], segment(game, pred_rgb, style)
            err = colors[pred_seg]
            err[(pred_seg != gt_seg) & (np.isin(pred_seg, dyn) | np.isin(gt_seg, dyn))] = ERROR
            m = mean_iou(scorer.class_ious(scorer.inter_union(pred_seg, gt_seg)))
            mious.append(m)
            c = game.count_class
            status = f"{eid} | frame {t:3d}/{n_total} | mIoU {m:.2f} | {c}s  GT {scorer.count(gt_seg)}  pred {scorer.count(pred_seg)}"
        top = np.concatenate([_label(gt_rgb, "ground truth"), _label(pred_rgb, model_name)], axis=1)
        bottom = np.concatenate([_label(colors[gt_seg], "GT classes"), _label(err, "predicted classes  (magenta = wrong)")], axis=1)
        grid = np.concatenate([top, bottom], axis=0)
        frames.append(np.concatenate([_header(grid.shape[1], status), grid], axis=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames), fps=manifest["fps"], codec="libx264", quality=8, macro_block_size=1)
    return {"episode": eid, "miou_all_frames": float(np.mean(mious)) if mious else None, "out": str(out_path)}
