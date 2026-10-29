"""Model-agnostic harness for the outcome track with vision-language models.

A backend only has to turn a list of message parts into text:

    parts = [{"type": "image", "path": Path}, {"type": "text", "text": str}, ...]
    backend(parts) -> str

`run` builds the prompt for every episode, calls the backend, parses the final
JSON answer, and appends one line per episode to the predictions file (resuming
if it already exists). The model sees images and text only, never the numeric
launch parameters or pig coordinates in tasks.jsonl.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

Backend = Callable[[list[dict]], str]

# Context frames shown to the model, out of the 12 available (0.4 s at 30 fps).
FRAME_IDS = (0, 4, 8, 11)

PROMPT = """You are looking at a 2D physics game shown from the side. Gravity points down.

A bird was just launched from the slingshot on the left. The first image shows the level at launch with every pig labeled by its id. The next {n} images are frames of the bird's flight at t = {times} seconds; it has not hit anything yet.

Rules of the game:
- The bird flies on, hits whatever is in its path, and pushes, topples or breaks blocks.
- Blocks are wood (brown), stone (gray, heaviest and strongest) or ice (pale blue, light and brittle).
- A pig dies when it is hit hard enough: by the bird, by a falling or sliding block, or by falling itself. Pigs that are only nudged survive.
- The game is over when everything has come to rest.

Question: which pigs are still alive when everything has come to rest?

Think it through briefly. Then finish with one line containing only a JSON object of this form:
{{"alive": [<ids of pigs that survive>], "p_alive": {{"<id>": <probability that this pig survives>, ...}}}}
Use the pig ids from the first image: {ids}."""


def build_parts(data_dir: Path, task: dict) -> list[dict]:
    fps = 30
    times = ", ".join(f"{i / fps:.2f}" for i in FRAME_IDS)
    parts: list[dict] = [{"type": "image", "path": data_dir / task["labeled_frame"]}]
    parts += [{"type": "image", "path": data_dir / task["context_frames"][i]} for i in FRAME_IDS]
    ids = ", ".join(str(i) for i in task["pig_ids"])
    parts.append({"type": "text", "text": PROMPT.format(n=len(FRAME_IDS), times=times, ids=ids)})
    return parts


def parse_answer(text: str, pig_ids: list[int]) -> dict | None:
    """Extract {"pigs_alive": [...], "p_alive": {...}} from the last JSON object in `text`."""
    valid = set(pig_ids)
    for m in reversed(list(re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "alive" not in obj:
            continue
        try:
            alive = sorted({int(i) for i in obj["alive"]} & valid)
            p_alive = {}
            for k, v in (obj.get("p_alive") or {}).items():
                if int(k) in valid:
                    p_alive[str(int(k))] = min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            continue
        out = {"pigs_alive": alive}
        if p_alive:
            out["p_alive"] = p_alive
        return out
    return None


def run(backend: Backend, data_dir: Path, out_path: Path, limit: int | None = None, model_name: str = "") -> dict:
    tasks = [json.loads(line) for line in (data_dir / "tasks.jsonl").read_text().splitlines() if line]
    tasks = tasks[:limit]
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines() if line}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parsed = failed = 0
    with open(out_path, "a") as f:
        for task in tasks:
            if task["id"] in done:
                continue
            t0 = time.time()
            text = backend(build_parts(data_dir, task))
            answer = parse_answer(text, task["pig_ids"])
            row = {"id": task["id"], "model": model_name, "seconds": round(time.time() - t0, 1), "raw": text}
            if answer is None:
                failed += 1
                row["pigs_alive"] = []  # unparseable answers count as "none alive"
                row["parse_error"] = True
            else:
                parsed += 1
                row.update(answer)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps({k: row[k] for k in ("id", "seconds", "pigs_alive")} | {"parse_error": answer is None}), flush=True)
    return {"new": parsed + failed, "parse_failures": failed}
