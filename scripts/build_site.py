"""Build the data and media for the GitHub Pages site in docs/.

Reads the cartoon datasets (data/<game>_v1_cartoon) and model answers (results/v1/),
then writes
    docs/data/leaderboard.json   scores per model, game, style and difficulty (+ 95 % CI)
    docs/data/challenge.json     human-challenge rounds with clickable targets
    docs/data/gallery.json       instructive model failures
    docs/media/episodes/...      the images and videos those reference

    uv run python scripts/build_site.py
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path

from physics_arcade.core.baselines import outcome_baselines, tasks_of
from physics_arcade.core.evaluate import STABLE, balanced_accuracy, eval_outcome, load_answers
from physics_arcade.core.predictability import analyze, write_majority_predictions
from physics_arcade.games import NAMES, get_game

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MEDIA = DOCS / "media" / "episodes"
STYLES = ("flat", "cartoon")
MODELS = {"qwen3vl_4b": "Qwen3-VL-4B-Instruct", "qwen3vl_8b": "Qwen3-VL-8B-Instruct (FP8)"}
REFERENCE_RULE = {"catapult": ("ballistic", "Ballistic path"), "pool": ("first_hit", "First-hit geometry"), "dominoes": ("gap_rule", "Gap rule")}
TITLES = {"catapult": "Catapult", "pool": "Pool", "dominoes": "Dominoes"}
ROUNDS = {"easy": 2, "medium": 3, "hard": 3}
BOOTSTRAP = 1000


def data_dir(game: str) -> Path:
    return ROOT / "data" / f"{game}_v1_cartoon"


def stable(p: float) -> bool:
    return p >= STABLE or p <= 1 - STABLE


def ci(answers: dict, pred_ids: dict[str, set[int]]) -> list[float]:
    """95 % bootstrap interval over episodes of the stable balanced accuracy."""
    rows = {e: [(int(k) in pred_ids.get(e, set()), p > 0.5) for k, p in a["p"].items() if stable(p)] for e, a in answers.items()}
    ids, rng, vals = list(rows), random.Random(0), []
    for _ in range(BOOTSTRAP):
        sample = [r for e in (rng.choice(ids) for _ in ids) for r in rows[e]]
        vals.append(balanced_accuracy([a for a, _ in sample], [b for _, b in sample]))
    vals.sort()
    return [round(vals[int(0.025 * BOOTSTRAP)], 3), round(vals[int(0.975 * BOOTSTRAP) - 1], 3)]


def score(game: str, path: Path) -> dict:
    d = data_dir(game)
    r = eval_outcome(d, path)
    preds = {json.loads(l)["id"]: set(json.loads(l)["ids"]) for l in path.read_text().splitlines() if l}
    return {
        "overall": round(r["stable_balanced_accuracy"], 3),
        "ci": ci(load_answers(d), preds),
        "brier": round(r["brier"], 3),
        "tiers": {t: round(v["stable_balanced_accuracy"], 3) for t, v in r["by_difficulty"].items()},
    }


def leaderboard(tmp: Path) -> dict:
    entries = []
    for key, name in MODELS.items():
        entries.append({"name": name, "kind": "vlm", "results": {
            g: {s: score(g, ROOT / "results" / "v1" / f"{key}_{g}_{s}.jsonl") for s in STYLES} for g in NAMES}})
    rule, sim, trivial = {}, {}, {}
    for g in NAMES:
        d = data_dir(g)
        bname, _ = REFERENCE_RULE[g]
        fn = outcome_baselines(d)[bname]
        rule_path = tmp / f"{g}_rule.jsonl"
        rule_path.write_text("".join(json.dumps({"id": t["id"], "ids": fn(t)}) + "\n" for t in tasks_of(d)))
        rule[g] = score(g, rule_path)
        analysis = tmp / f"{g}_predictability.jsonl"
        analyze(d, analysis, runs=50, workers=16)
        sim_path = tmp / f"{g}_sim.jsonl"
        write_majority_predictions(analysis, "launch", sim_path)
        sim[g] = score(g, sim_path)
        all_path = tmp / f"{g}_all.jsonl"
        all_path.write_text("".join(json.dumps({"id": t["id"], "ids": t["target_ids"]}) + "\n" for t in tasks_of(d)))
        trivial[g] = score(g, all_path)
    both = lambda per_game: {g: {s: v for s in STYLES} for g, v in per_game.items()}  # noqa: E731
    entries += [
        {"name": "Game-specific rule", "names": {g: REFERENCE_RULE[g][1] for g in NAMES}, "kind": "reference",
         "note": "Reads exact launch and positions instead of images; ignores most of the physics.", "results": both(rule)},
        {"name": "Simulator, launch noise 1 % / 0.5°", "kind": "reference",
         "note": "The real physics engine fed a slightly wrong launch: what a near-perfect visual estimate allows.", "results": both(sim)},
        {"name": "Always \"yes\"", "kind": "reference", "note": "Says the event happens to every target.", "results": both(trivial)},
    ]
    games = [{"name": g, "title": TITLES[g], "question": get_game(g).question, "target": get_game(g).target,
              "event": get_game(g).event} for g in NAMES]
    return {"games": games, "styles": list(STYLES), "difficulties": ["easy", "medium", "hard"], "entries": entries}


def screen_targets(game: str, task: dict) -> list[dict]:
    """Click areas in 640x360 image pixels (origin top-left)."""
    h_img = get_game(game).height
    out = []
    for k, t in sorted(task["info"]["targets"].items(), key=lambda kv: int(kv[0])):
        if game == "dominoes":
            cy = h_img - (t["base"] + t["h"] / 2)
            out.append({"id": int(k), "x": t["x"], "y": cy, "w": 18, "h": t["h"] + 6})
        else:
            side = 2 * t["r"] + 10
            out.append({"id": int(k), "x": t["x"], "y": h_img - t["y"], "w": side, "h": side})
    return out


def copy_media(game: str, eid: str) -> dict:
    d = data_dir(game)
    MEDIA.mkdir(parents=True, exist_ok=True)
    names = {"still": f"{game}_{eid}_labeled.png", "context": f"{game}_{eid}_context.mp4", "outcome": f"{game}_{eid}_outcome.mp4"}
    shutil.copyfile(d / "inputs" / eid / "labeled.png", MEDIA / names["still"])
    shutil.copyfile(d / "inputs" / eid / "context.mp4", MEDIA / names["context"])
    shutil.copyfile(d / "answers" / f"{eid}.mp4", MEDIA / names["outcome"])
    return {k: f"media/episodes/{v}" for k, v in names.items()}


def model_answers(game: str) -> dict[str, dict[str, dict]]:
    out = {}
    for key, name in MODELS.items():
        rows = (ROOT / "results" / "v1" / f"{key}_{game}_cartoon.jsonl").read_text().splitlines()
        out[name] = {r["id"]: r for r in (json.loads(l) for l in rows if l)}
    return out


def challenge() -> dict:
    rounds = []
    for g in NAMES:
        d = data_dir(g)
        answers, tasks = load_answers(d), {t["id"]: t for t in tasks_of(d)}
        models = model_answers(g)
        rng = random.Random(f"challenge-{g}")
        for tier, n in ROUNDS.items():
            pool = [e for e, a in answers.items() if a["difficulty"] == tier and all(stable(p) for p in a["p"].values())]
            for eid in sorted(rng.sample(pool, n)):
                a = answers[eid]
                rounds.append({
                    "game": g, "id": eid, "difficulty": tier, **copy_media(g, eid),
                    "targets": screen_targets(g, tasks[eid]),
                    "truth": sorted(int(k) for k, p in a["p"].items() if p > 0.5),
                    "models": {name: sorted(m[eid]["ids"]) for name, m in models.items()},
                })
    return {"rounds": rounds}


def excerpt(text: str, n: int = 420) -> str:
    """The model's reasoning before its JSON answer ("" if it answered with JSON only)."""
    body = text.split("{", 1)[0].replace("```json", "").replace("```", "").strip()
    return body if len(body) <= n else body[:n].rsplit(" ", 1)[0] + " …"


def gallery() -> dict:
    items = []
    for g in NAMES:
        d = data_dir(g)
        answers, models = load_answers(d), model_answers(g)
        for name, rows in models.items():
            best = None
            for eid, a in answers.items():
                pred = set(rows[eid]["ids"])
                wrong = [int(k) for k, p in a["p"].items() if stable(p) and (int(k) in pred) != (p > 0.5)]
                reasoning = excerpt(rows[eid]["raw"])
                # Prefer failures the model explained in words, then the most clear-cut targets missed.
                key = (bool(reasoning) and len(wrong) >= 2, len(wrong))
                if best is None or key > best[0]:
                    best = (key, eid, wrong, reasoning)
            (_, _), eid, wrong, reasoning = best
            a = answers[eid]
            items.append({
                "game": g, "model": name, "id": eid, "difficulty": a["difficulty"], **copy_media(g, eid),
                "truth": sorted(int(k) for k, p in a["p"].items() if p > 0.5),
                "answer": sorted(rows[eid]["ids"]), "wrong": sorted(wrong), "reasoning": reasoning,
            })
    return {"items": items}


def main() -> None:
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    if MEDIA.exists():
        shutil.rmtree(MEDIA)
    with tempfile.TemporaryDirectory() as tmp:
        board = leaderboard(Path(tmp))
    outputs = {"leaderboard": board, "challenge": challenge(), "gallery": gallery()}
    for name, obj in outputs.items():
        (DOCS / "data" / f"{name}.json").write_text(json.dumps(obj, indent=1))
    size = sum(f.stat().st_size for f in MEDIA.iterdir()) / 1e6
    print(f"wrote {len(outputs['challenge']['rounds'])} challenge rounds, {len(outputs['gallery']['items'])} gallery items, {size:.1f} MB of media")


if __name__ == "__main__":
    main()
