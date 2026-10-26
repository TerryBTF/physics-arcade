"""How predictable is each episode's outcome?

Every episode is re-simulated many times from its exact pre-launch scene with small
perturbations, and we measure how often the outcome matches the recorded one. An
episode whose outcome flips under perturbations far below what a model could
perceive from the context frames cannot be predicted by anyone.

The majority vote over perturbed runs is also a reference predictor: a perfect
physics engine fed launch parameters with realistic estimation error.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .episode import rollout, setup_for

# Perturbations, each a standard deviation:
#   launch_tiny  speed 0.25 %, angle 0.1 deg: at or below the precision a careful estimate
#                from 12 frames can reach; flips here mean the episode is chaotic.
#   launch       speed 1 %, angle 0.5 deg: a realistic but imperfect visual estimate.
#   scene        every block/pig shifted horizontally by 0.5 px, launch exact.
CONDITIONS = {
    "launch_tiny": {"speed": 0.0025, "angle": 0.1, "pos": 0.0},
    "launch": {"speed": 0.01, "angle": 0.5, "pos": 0.0},
    "scene": {"speed": 0.0, "angle": 0.0, "pos": 0.5},
}


def _run(seed: int, attempt: int, cond: dict, rng: random.Random) -> tuple[list[int], float | None]:
    s = setup_for(seed, attempt)
    if cond["pos"]:
        for o in s.world.objects:
            x, y = o.body.position
            o.body.position = (x + rng.gauss(0, cond["pos"]), y)
            s.world.space.reindex_shapes_for_body(o.body)
            o.body.activate()
    speed = s.speed * (1 + rng.gauss(0, cond["speed"]))
    angle = s.angle + rng.gauss(0, cond["angle"])
    rollout(s.world, speed, angle, None)
    return sorted(p.pig_id for p in s.world.pigs if p.alive), s.world.first_contact_time


def _episode(args: tuple[str, int, int, list[int], list[int], int]) -> dict:
    eid, seed, attempt, pig_ids, gt_alive, runs = args
    gt = set(gt_alive)
    out = {"id": eid, "n_pigs": len(pig_ids), "conditions": {}}
    for name, cond in CONDITIONS.items():
        rng = random.Random(f"{seed}-{name}")
        alive_counts = {p: 0 for p in pig_ids}
        same = 0
        for _ in range(runs):
            alive, _ = _run(seed, attempt, cond, rng)
            same += set(alive) == gt
            for p in alive:
                alive_counts[p] += 1
        p_alive = {p: c / runs for p, c in alive_counts.items()}
        out["conditions"][name] = {
            "p_same_outcome": same / runs,
            # per pig: probability its fate matches the recorded one
            "pig_agreement": {str(p): (p_alive[p] if p in gt else 1 - p_alive[p]) for p in pig_ids},
            "majority_alive": sorted(p for p in pig_ids if p_alive[p] > 0.5),
        }
    return out


def analyze(data_dir: Path, out_path: Path, runs: int = 50, workers: int = 16) -> dict:
    tasks = [json.loads(line) for line in (data_dir / "tasks.jsonl").read_text().splitlines() if line]
    jobs = []
    for t in tasks:
        ans = json.loads((data_dir / "answers" / f"{t['id']}.json").read_text())
        if "attempt" not in ans:
            raise ValueError(f"{data_dir} predates attempt tracking; regenerate it")
        jobs.append((t["id"], t["seed"], ans["attempt"], ans["pig_ids"], ans["pigs_alive"], runs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_episode, jobs, chunksize=2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return summarize(results)


def summarize(results: list[dict]) -> dict:
    summary = {"episodes": len(results), "runs_per_condition": None}
    for name in CONDITIONS:
        ps = [r["conditions"][name]["p_same_outcome"] for r in results]
        pig = [a for r in results for a in r["conditions"][name]["pig_agreement"].values()]
        summary[name] = {
            "mean_p_same_outcome": sum(ps) / len(ps),
            "episodes_fully_stable": sum(p == 1.0 for p in ps) / len(ps),
            "episodes_majority_matches": sum(p > 0.5 for p in ps) / len(ps),
            "episodes_coin_flip_or_worse": sum(p <= 0.5 for p in ps) / len(ps),
            "pigs_stable_ge_90pct": sum(a >= 0.9 for a in pig) / len(pig),
            "pigs_ambiguous_10_90": sum(0.1 < a < 0.9 for a in pig) / len(pig),
        }
    return summary


def write_majority_predictions(results_path: Path, condition: str, out_path: Path) -> None:
    """Outcome-track predictions from the perturbed-simulation majority vote."""
    with open(results_path) as src, open(out_path, "w") as dst:
        for line in src:
            r = json.loads(line)
            dst.write(json.dumps({"id": r["id"], "pigs_alive": r["conditions"][condition]["majority_alive"]}) + "\n")
