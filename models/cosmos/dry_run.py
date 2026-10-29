"""Check the Cosmos I/O path without the model.

Builds a fake "Cosmos output" from the ground-truth video: the 93 frames Cosmos
would produce at 16 fps if it were perfect, upscaled to the generation
resolution. It then goes through the same conversion as run_cosmos.py
(timing.to_benchmark_frames). Scoring the result shows how much the resampling
alone costs; it should sit just below the oracle.

    uv run python models/cosmos/dry_run.py --data data/slingshot_v0 --out preds/cosmos_dryrun
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from timing import BENCH_FPS, cosmos_frame_times, to_benchmark_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=704)
    args = ap.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    n_total, n_ctx = manifest["total_frames"], manifest["context_frames"]
    gt_index = np.clip(np.round(cosmos_frame_times(n_ctx) * BENCH_FPS), 0, n_total - 1).astype(int)
    args.out.mkdir(parents=True, exist_ok=True)
    for line in (args.data / "tasks.jsonl").read_text().splitlines():
        eid = json.loads(line)["id"]
        gt = iio.imread(args.data / "answers" / f"{eid}.mp4")
        fake = [Image.fromarray(gt[i]).resize((args.width, args.height), Image.BICUBIC) for i in gt_index]
        future = to_benchmark_frames(fake, n_ctx, n_total - n_ctx)
        iio.imwrite(args.out / f"{eid}.mp4", future, fps=BENCH_FPS, codec="libx264", quality=9, macro_block_size=1)


if __name__ == "__main__":
    main()
