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
    umask 022   # NLTK (used by the guardrail) refuses group-writable data directories
    .venv/bin/python run_cosmos.py --data ../../data/slingshot_v0 \
        --out ../../preds/rollout_cosmos2b --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from diffusers import Cosmos2VideoToWorldPipeline
from huggingface_hub import snapshot_download
from PIL import Image
from timing import BENCH_FPS, COSMOS_FPS, NUM_FRAMES, cond_indices, to_benchmark_frames

MODEL_ID = "nvidia/Cosmos-Predict2-2B-Video2World"
OUT_W, OUT_H = 640, 360

SCENE = (
    "A small red bird has been launched from a slingshot on the left and flies along a ballistic arc toward "
    "structures built from brown wood, gray stone and pale blue ice blocks, with round green pigs resting on "
    "and between them. The bird hits the structures; blocks are pushed, topple and fall under gravity, and pigs "
    "that are struck hard or crushed disappear. Everything settles to rest on the ground. The camera never moves "
    "and no new objects appear."
)
PROMPTS = {
    "flat": "A side-view 2D physics game rendered with flat solid colors, no shading and no text. " + SCENE
    + " The sky is plain light blue and the ground is plain green.",
    "cartoon": "A side-view 2D cartoon physics game with a blue sky, white clouds, hills and grass. " + SCENE
    + " The pigs have simple cartoon faces; wood has grain, stone is speckled and ice is glossy.",
}
NEGATIVE_PROMPT = (
    "camera motion, zoom, panning, shaky footage, motion blur, 3D rendering, realistic textures, shading, "
    "lighting changes, new objects appearing, objects changing color, text, watermark, flickering, "
    "low frame rate, artifacting, distorted shapes."
)


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
    # CosmosSafetyChecker.to() returns None, which breaks accelerate's offload hook. The
    # pipeline moves the safety checker to the GPU and back around each use on its own,
    # so it is left out of the automatic offload (it still runs on every prompt and video).
    pipe._exclude_from_cpu_offload = ["safety_checker"]
    pipe.enable_model_cpu_offload()  # T5-XXL + DiT + VAE do not fit in 16 GB together
    pipe.safety_checker.to("cpu")
    pipe.vae.enable_tiling()  # encoding 93 frames at 1280x704 in one piece does not fit in 16 GB

    cidx = cond_indices(n_ctx)
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
            prompt=PROMPTS[manifest.get("style", "flat")],
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
        future = to_benchmark_frames(frames, n_ctx, n_future, (OUT_W, OUT_H))
        iio.imwrite(dst, future, fps=BENCH_FPS, codec="libx264", quality=9, macro_block_size=1)
        dt = time.time() - t0
        log.append({"id": eid, "seconds": round(dt, 1), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        print(json.dumps(log[-1]), flush=True)
    (args.out / "run_log.jsonl").open("a").write("".join(json.dumps(r) + "\n" for r in log))


if __name__ == "__main__":
    main()
