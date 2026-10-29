"""How predictable is each episode's outcome?

Every episode is replayed many times from its exact pre-launch scene under each of the
game's perturbations; we record how often each target's fate matches the recorded run.
The per-target majority vote is also a reference predictor (a perfect simulator fed the
perturbed inputs).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .dataset import survival_probabilities
from .evaluate import game_of, load_answers


def _episode(args: tuple[Path, dict, int]) -> dict:
    data_dir, ans, runs = args
    game = game_of(data_dir)
    recorded = set(ans["recorded_ids"])
    out = {"id": ans["id"], "difficulty": ans["difficulty"], "conditions": {}}
    for name in game.perturbations:
        p = survival_probabilities(game, ans["seed"], ans["difficulty"], ans["replay_key"], ans["target_ids"], name, runs, "analysis")
        out["conditions"][name] = {
            "p": {str(k): v for k, v in p.items()},
            # per target: probability its fate matches the recorded run
            "agreement": {str(k): (v if k in recorded else 1 - v) for k, v in p.items()},
            "majority_ids": sorted(k for k, v in p.items() if v > 0.5),
        }
    return out


def analyze(data_dir: Path, out_path: Path, runs: int = 50, workers: int = 16) -> dict:
    answers = list(load_answers(data_dir).values())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_episode, [(data_dir, a, runs) for a in answers], chunksize=2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = {"runs_per_condition": runs, "overall": summarize(results)}
    for d in dict.fromkeys(r["difficulty"] for r in results):
        summary[d] = summarize([r for r in results if r["difficulty"] == d])
    return summary


def summarize(results: list[dict]) -> dict:
    out = {"episodes": len(results)}
    for name in results[0]["conditions"]:
        agree = [a for r in results for a in r["conditions"][name]["agreement"].values()]
        out[name] = {
            "targets_stable_ge_90pct": sum(a >= 0.9 for a in agree) / len(agree),
            "targets_ambiguous_10_90": sum(0.1 < a < 0.9 for a in agree) / len(agree),
        }
    return out


def write_majority_predictions(results_path: Path, condition: str, out_path: Path) -> None:
    """Outcome-track predictions from the perturbed-replay majority vote."""
    with open(results_path) as src, open(out_path, "w") as dst:
        for line in src:
            r = json.loads(line)
            c = r["conditions"][condition]
            dst.write(json.dumps({"id": r["id"], "ids": c["majority_ids"], "p": c["p"]}) + "\n")
