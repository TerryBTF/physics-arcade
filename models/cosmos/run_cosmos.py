"""Roll out benchmark episodes with Cosmos-Predict2-2B-Video2World (diffusers).

Runs in its own environment (models/cosmos/.venv) so the benchmark package stays
free of torch. Writes one 640x360, 30 fps mp4 per episode containing only the
future frames, which is the format `physics-arcade eval-rollout` expects.

Timing: the benchmark gives 12 context frames at 30 fps (t = 0 .. 11/30 s) and
scores 168 future frames (t = 12/30 .. 179/30 s). Cosmos works at 16 fps and
generates 93 frames, conditioning on the first 5. We pick the 5 context frames
closest to a 16 fps grid ending at the last context frame, then map every
generated frame back onto the 30 fps timeline by nearest neighbour. The 93-frame
window ends ~0.1 s before the episode does; the last two scored frames repeat
the final generated frame.

Usage (on robotpc):
    cd ~/Benchmark/models/cosmos
    .venv/bin/python run_cosmos.py --data ../../data/angry_birds_v0 \
        --out ../../preds/rollout_cosmos2b --limit 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from diffusers import Cosmos2VideoToWorldPipeline
from huggingface_hub import snapshot_download
from PIL import Image

MODEL_ID = "nvidia/Cosmos-Predict2-2B-Video2World"
BENCH_FPS, COSMOS_FPS = 30, 16
NUM_FRAMES, NUM_COND = 93, 5
OUT_W, OUT_H = 640, 360

PROMPT = (
    "A side-view 2D physics game rendered with flat colors and a fixed camera. A small red bird has been "
    "launched from a slingshot on the left and flies along a ballistic arc toward structures built from "
    "brown wood, gray stone and pale blue ice blocks, with round green pigs resting on and between them. "
    "The bird hits the structures; blocks are pushed, topple and fall under gravity, and pigs that are "
    "struck hard or crushed disappear. Everything settles to rest on the green ground. The sky is plain "
    "light blue, the camera never moves, there is no text, no shading and no new objects appear."
)
NEGATIVE_PROMPT = (
    "camera motion, zoom, panning, shaky footage, motion blur, 3D rendering, realistic textures, shading, "
    "lighting changes, new objects appearing, objects changing color, text, watermark, flickering, "
    "low frame rate, artifacting, distorted shapes."
)


def cond_indices(n_ctx: int) -> list[int]:
    last = n_ctx - 1
    return [int(np.clip(round(last - (NUM_COND - 1 - i) * BENCH_FPS / COSMOS_FPS), 0, last)) for i in range(NUM_COND)]


def future_to_cosmos_index(n_ctx: int, n_future: int) -> np.ndarray:
    """Index into the 93 generated frames for each scored future frame."""
    last = n_ctx - 1
    t = (np.arange(n_ctx, n_ctx + n_future) - last) / BENCH_FPS  # seconds after last context frame
    return np.clip(np.round(NUM_COND - 1 + t * COSMOS_FPS), 0, NUM_FRAMES - 1).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    n_total, n_ctx = manifest["total_frames"], manifest["context_frames"]
    n_future = n_total - n_ctx
    tasks = [json.loads(line) for line in (args.data / "tasks.jsonl").read_text().splitlines() if line]
    tasks = tasks[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    # The repo also holds six native-format .pt checkpoints (~24 GB) that diffusers does not use.
    path = snapshot_download(MODEL_ID, ignore_patterns=["*.pt"])
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()  # T5-XXL + DiT + VAE do not fit in 16 GB together

    cidx = cond_indices(n_ctx)
    fidx = future_to_cosmos_index(n_ctx, n_future)
    log = []
    for task in tasks:
        eid = task["id"]
        dst = args.out / f"{eid}.mp4"
        if dst.exists():
            continue
        cond = [Image.open(args.data / task["context_frames"][i]).convert("RGB") for i in cidx]
        t0 = time.time()
        frames = pipe(
            video=cond,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=args.height,
            width=args.width,
            num_frames=NUM_FRAMES,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            fps=COSMOS_FPS,
            generator=torch.Generator().manual_seed(args.seed),
            output_type="pil",
        ).frames[0]
        future = np.stack([np.asarray(frames[k].resize((OUT_W, OUT_H), Image.BICUBIC)) for k in fidx])
        iio.imwrite(dst, future, fps=BENCH_FPS, codec="libx264", quality=9, macro_block_size=1)
        dt = time.time() - t0
        log.append({"id": eid, "seconds": round(dt, 1), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        print(json.dumps(log[-1]), flush=True)
    (args.out / "run_log.jsonl").open("a").write("".join(json.dumps(r) + "\n" for r in log))


if __name__ == "__main__":
    main()
