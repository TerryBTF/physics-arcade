"""Dataset layout, labels and both scoring tracks on tiny datasets."""

from __future__ import annotations

import json

from physics_arcade.core.baselines import run_baseline
from physics_arcade.core.evaluate import eval_outcome, eval_rollout, load_answers
from physics_arcade.core.fingerprint import fingerprint


def test_layout_and_labels(tiny_dataset):
    game, data = tiny_dataset
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["game"] == game.name and manifest["num_episodes"] == len(game.difficulties)
    for eid, ans in load_answers(data).items():
        assert set(ans["p"]) == {str(i) for i in ans["target_ids"]}
        assert all(0.0 <= v <= 1.0 for v in ans["p"].values())
        assert (data / "inputs" / eid / "labeled.png").exists()
        assert len(list((data / "inputs" / eid).glob("context_*.png"))) == game.context_frames


def test_outcome_scoring(tiny_dataset, tmp_path):
    _, data = tiny_dataset
    answers = load_answers(data)
    perfect = tmp_path / "perfect.jsonl"
    perfect.write_text("".join(json.dumps({"id": e, "ids": [int(k) for k, p in a["p"].items() if p > 0.5], "p": a["p"]}) + "\n" for e, a in answers.items()))
    r = eval_outcome(data, perfect)
    assert r["stable_balanced_accuracy"] in (1.0, None)  # None when a tiny set has only one class
    assert abs(r["brier"] - r["brier_floor"]) < 1e-9
    for name in ("all", "none"):
        out = tmp_path / f"{name}.jsonl"
        run_baseline(name, data, out)
        assert eval_outcome(data, out)["stable_balanced_accuracy"] in (0.5, None)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert eval_outcome(data, empty)["missing_predictions"] == len(answers)


def test_rollout_oracle_scores_high(tiny_dataset, tmp_path):
    _, data = tiny_dataset
    run_baseline("oracle", data, tmp_path / "oracle")
    assert eval_rollout(data, tmp_path / "oracle", workers=2)["miou_all_frames"] > 0.9


def test_fingerprint_is_stable(tiny_dataset):
    _, data = tiny_dataset
    assert fingerprint(data) == fingerprint(data)
