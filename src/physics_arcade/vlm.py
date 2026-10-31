"""Model-agnostic harness for the outcome track with vision-language models.

A backend only has to turn a list of message parts into text:

    parts = [{"type": "image", "path": Path}, {"type": "text", "text": str}, ...]
    backend(parts) -> str

`run` builds the prompt for every episode (the game's description and question plus a
shared answer format), calls the backend, parses the final JSON answer, and appends one
line per episode to the predictions file (resuming if it already exists). The model sees
images and text only, never the numeric extras in tasks.jsonl "info".
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

from .core.game import Game

Backend = Callable[[list[dict]], str]

# Context frames shown to the model, spread over the context window.
N_FRAMES = 4

ANSWER_FORMAT = """

Think it through briefly. Then finish with one line containing only a JSON object of this form:
{{"{event}": [<ids>], "p_{event}": {{"<id>": <probability>, ...}}}}
- "{event}" lists the ids of the {target}s that are {event} once everything has come to rest.
- "p_{event}" gives, for every {target} id, your probability (0 to 1) that the {target} is {event}.
- The list must contain exactly the ids whose probability is above 0.5.
The {target} ids are those labeled in the first image: {ids}."""


def frame_ids(n_ctx: int) -> list[int]:
    return sorted({round(i * (n_ctx - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)})


def build_parts(data_dir: Path, task: dict, game: Game, fps: int) -> list[dict]:
    ids_shown = frame_ids(len(task["context_frames"]))
    times = ", ".join(f"{i / fps:.2f}" for i in ids_shown)
    parts: list[dict] = [{"type": "image", "path": data_dir / task["labeled_frame"]}]
    parts += [{"type": "image", "path": data_dir / task["context_frames"][i]} for i in ids_shown]
    ids = ", ".join(str(i) for i in task["target_ids"])
    text = game.vlm_prompt(len(ids_shown), times) + ANSWER_FORMAT.format(event=game.event, target=game.target, ids=ids)
    parts.append({"type": "text", "text": text})
    return parts


FOLLOW_UP = (
    "Your previous answer did not end with the required JSON line. Here is how it ended:\n\n{tail}\n\n"
    "Reply with only the JSON line now."
)


def parse_answer(text: str, target_ids: list[int], event: str) -> dict | None:
    """Extract {"ids": [...], "p": {...}} from the last JSON object in `text` with key `event`."""
    valid = set(target_ids)
    for m in reversed(list(re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text))):
        # Models sometimes write bare numeric keys ({1: 0.5}); quote them.
        candidate = re.sub(r"([{,]\s*)(\d+)(\s*:)", r'\1"\2"\3', m.group(0))
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or event not in obj:
            continue
        try:
            ids = sorted({int(i) for i in obj[event]} & valid)
            probs = {}
            for k, v in (obj.get(f"p_{event}") or {}).items():
                if int(k) in valid:
                    probs[str(int(k))] = min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            continue
        out = {"ids": ids}
        if probs:
            out["p"] = probs
        return out
    return None


def run(backend: Backend, data_dir: Path, out_path: Path, limit: int | None = None, model_name: str = "") -> dict:
    from .core.evaluate import game_of, load_manifest

    game, fps = game_of(data_dir), load_manifest(data_dir)["fps"]
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
            parts = build_parts(data_dir, task, game, fps)
            text = backend(parts)
            answer = parse_answer(text, task["target_ids"], game.event)
            row = {"id": task["id"], "model": model_name, "raw": text}
            if answer is None:
                # One retry that shows the model the end of its own answer and asks for the JSON only.
                retry = backend(parts + [{"type": "text", "text": FOLLOW_UP.format(tail=text[-1500:])}])
                answer = parse_answer(retry, task["target_ids"], game.event)
                row["raw_retry"] = retry
            row["seconds"] = round(time.time() - t0, 1)
            if answer is None:
                failed += 1
                row["ids"] = []  # unparseable answers count as "the event happens to no target"
                row["parse_error"] = True
            else:
                parsed += 1
                row.update(answer)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps({k: row[k] for k in ("id", "seconds", "ids")} | {"parse_error": answer is None}), flush=True)
    return {"new": parsed + failed, "parse_failures": failed}
