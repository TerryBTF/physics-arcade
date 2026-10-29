"""Outcome-track predictions with a local Qwen-VL model (transformers).

Runs in models/qwen_vl/.venv (see setup.sh). Prompting and answer parsing live in
physics_arcade.vlm so every backend sees exactly the same task.

    cd ~/Benchmark/models/qwen_vl
    .venv/bin/python run_qwen_vl.py --data ../../data/slingshot_v0 --out ../../preds/qwen3vl_4b.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physics_arcade.vlm import run


def make_backend(model_id: str, max_new_tokens: int, seed: int = 0):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def backend(parts: list[dict]) -> str:
        content = [
            {"type": "image", "image": Image.open(p["path"]).convert("RGB")} if p["type"] == "image" else p
            for p in parts
        ]
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        # Qwen3-VL's recommended sampling (greedy decoding tends to loop); fixed seed for reproducibility.
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.8, top_k=20
            )
        return processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    backend = make_backend(args.model, args.max_new_tokens, args.seed)
    print(json.dumps(run(backend, args.data, args.out, args.limit, args.model)))


if __name__ == "__main__":
    main()
