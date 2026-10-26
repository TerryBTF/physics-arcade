"""Side-by-side comparison videos of a predicted rollout against ground truth.

Layout (each panel 640x360):
    ground-truth video     | predicted video
    ground-truth classes   | predicted classes, errors in magenta
with a header showing the frame index, that frame's mIoU, and live pig counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evaluate import _align, _class_ious, _inter_union, _mean, count_pigs, load_video
from .render import CLASSES, DYNAMIC_CLASSES, PALETTE, segment

HEADER = 28
ERROR = np.array([255, 0, 255], dtype=np.uint8)
LUT = np.array([PALETTE[c] for c in CLASSES], dtype=np.uint8)
DYNAMIC_IDS = [CLASSES.index(c) for c in DYNAMIC_CLASSES]


def _colorize(mask: np.ndarray) -> np.ndarray:
    return LUT[mask]


def _error_view(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    img = _colorize(pred)
    wrong = (pred != gt) & (np.isin(pred, DYNAMIC_IDS) | np.isin(gt, DYNAMIC_IDS))
    img[wrong] = ERROR
    return img


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
    manifest = json.loads((data_dir / "manifest.json").read_text())
    n_total, n_ctx = manifest["total_frames"], manifest["context_frames"]
    n_future = n_total - n_ctx
    gt_video = load_video(data_dir / "answers" / f"{eid}.mp4")
    gt_masks = np.load(data_dir / "answers" / f"{eid}_masks.npz")["masks"]
    pred = _align(load_video(pred_path), n_future, n_total)
    fps = manifest.get("fps", 30)

    frames, mious = [], []
    for t in range(n_total):
        gt_rgb = gt_video[t]
        if t < n_ctx:
            pred_rgb = gt_rgb
            gt_seg = pred_seg = segment(gt_rgb)
            err = _colorize(pred_seg)
            status = f"{eid} | frame {t:3d}/{n_total} | CONTEXT (given to the model)"
        else:
            f = t - n_ctx
            pred_rgb = pred[f]
            gt_seg, pred_seg = gt_masks[f], segment(pred_rgb)
            err = _error_view(pred_seg, gt_seg)
            m = _mean(_class_ious(_inter_union(pred_seg, gt_seg)))
            mious.append(m)
            status = (
                f"{eid} | frame {t:3d}/{n_total} | mIoU {m:.2f} | "
                f"pigs  GT {count_pigs(gt_seg)}  pred {count_pigs(pred_seg)}"
            )
        top = np.concatenate([_label(gt_rgb, "ground truth"), _label(pred_rgb, model_name)], axis=1)
        bottom = np.concatenate(
            [_label(_colorize(gt_seg), "GT classes"), _label(err, "predicted classes  (magenta = wrong)")], axis=1
        )
        grid = np.concatenate([top, bottom], axis=0)
        frames.append(np.concatenate([_header(grid.shape[1], status), grid], axis=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    return {"episode": eid, "miou_all_frames": float(np.mean(mious)) if mious else None, "out": str(out_path)}
