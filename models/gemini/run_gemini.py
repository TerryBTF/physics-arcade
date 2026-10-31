"""Outcome-track predictions with Google Gemini (google-genai SDK).

Uses the shared prompt, answer parsing and resumable loop in physics_arcade.vlm, so the
model sees exactly what the local Qwen-VL runs saw. The API key is read from the
GEMINI_API_KEY environment variable; it is never written to disk by this script.

    cd models/gemini && ./setup.sh
    export GEMINI_API_KEY=...            # e.g. in ~/.bashrc, not in the repo
    .venv/bin/python run_gemini.py --list-models
    .venv/bin/python run_gemini.py --model <name> --data ../../data/pool_v1_lite \
        --out ../../preds/gemini_pool_lite.jsonl --limit 10

Token usage per call is appended to <out>.usage.jsonl so the cost of a run can be computed.
On the free tier, use --rpm to stay under the per-minute request limit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

from physics_arcade.vlm import run

MAX_WAIT = 300.0  # seconds between retries at most


def make_backend(client: genai.Client, model: str, usage_path: Path, rpm: float, thinking_level: str | None, seed: int):
    config = types.GenerateContentConfig(
        seed=seed,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None,
    )
    min_interval = 60.0 / rpm if rpm else 0.0
    last_call = [0.0]

    def backend(parts: list[dict]) -> str:
        contents = [
            types.Part.from_bytes(data=Path(p["path"]).read_bytes(), mime_type="image/png") if p["type"] == "image" else p["text"]
            for p in parts
        ]
        wait = 10.0
        while True:
            pause = last_call[0] + min_interval - time.time()
            if pause > 0:
                time.sleep(pause)
            last_call[0] = time.time()
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                break
            except errors.ClientError as e:
                if e.code != 429:
                    raise
                print(f"rate limited, retrying in {wait:.0f}s", flush=True)
            except errors.ServerError as e:
                print(f"server error {e.code}, retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, MAX_WAIT)
        u = response.usage_metadata
        with open(usage_path, "a") as f:
            f.write(json.dumps({
                "prompt_tokens": u.prompt_token_count if u else None,
                "output_tokens": u.candidates_token_count if u else None,
                "thought_tokens": u.thoughts_token_count if u else None,
            }) + "\n")
        return response.text or ""

    return backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true", help="print the models this API key can call, then exit")
    ap.add_argument("--model")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rpm", type=float, default=0.0, help="max requests per minute (free tier); 0 = no limit")
    ap.add_argument("--thinking-level", default=None, help="optional, e.g. low / high; default: the model's own default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = genai.Client()  # reads GEMINI_API_KEY
    if args.list_models:
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print(m.name)
        return
    if not (args.model and args.data and args.out):
        ap.error("--model, --data and --out are required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    usage = args.out.with_suffix(".usage.jsonl")
    backend = make_backend(client, args.model, usage, args.rpm, args.thinking_level, args.seed)
    print(json.dumps(run(backend, args.data, args.out, args.limit, args.model)))


if __name__ == "__main__":
    main()
