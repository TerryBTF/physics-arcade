"""Fingerprints that tell whether a locally generated dataset is the official one.

    labels   tasks.jsonl, answers/<id>.json (ids, probabilities) and the class masks:
             everything scoring depends on. Must match exactly.
    inputs   pixels of the context frames and the labeled first frame, i.e. what a model
             sees. Expected to match, but image libraries may round differently.

Videos are not fingerprinted: H.264 output depends on the ffmpeg build.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _update_json(h, obj) -> None:
    h.update(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def fingerprint(data_dir: Path) -> dict:
    tasks = [json.loads(line) for line in (data_dir / "tasks.jsonl").read_text().splitlines() if line]
    labels, inputs = hashlib.sha256(), hashlib.sha256()
    for t in tasks:
        eid = t["id"]
        _update_json(labels, t)
        ans = json.loads((data_dir / "answers" / f"{eid}.json").read_text())
        _update_json(labels, {k: ans[k] for k in ("id", "game", "seed", "difficulty", "target_ids", "recorded_ids", "p", "replay_key")})
        labels.update(np.load(data_dir / "answers" / f"{eid}_masks.npz")["masks"].tobytes())
        for rel in [*t["context_frames"], t["labeled_frame"]]:
            inputs.update(np.asarray(Image.open(data_dir / rel).convert("RGB")).tobytes())
    return {"episodes": len(tasks), "labels": labels.hexdigest(), "inputs": inputs.hexdigest()}
