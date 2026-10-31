"""Answer parsing and prompt construction for vision-language models."""

from __future__ import annotations

import json

from physics_arcade.vlm import build_parts, parse_answer, run


def test_parse_answer_variants():
    ids = [1, 2, 3, 4]
    assert parse_answer('so... {"alive": [1, 3], "p_alive": {"1": 0.9, "3": 0.7}}', ids, "alive") == {"ids": [1, 3], "p": {"1": 0.9, "3": 0.7}}
    assert parse_answer('{"alive": [], "p_alive": {1: 0.0, 2: 0.3}}', ids, "alive")["p"] == {"1": 0.0, "2": 0.3}  # bare keys
    assert parse_answer('{"alive": [2, 9], "p_alive": {"2": 1.4}}', ids, "alive") == {"ids": [2], "p": {"2": 1.0}}  # clamp, drop unknown ids
    assert parse_answer('```json\n{"fallen": ["4"]}\n```', ids, "fallen") == {"ids": [4]}
    assert parse_answer('{"x": 1} then {"potted": [1]}', ids, "potted") == {"ids": [1]}
    assert parse_answer("no json here", ids, "alive") is None
    assert parse_answer('{"alive": [1]}', ids, "fallen") is None  # wrong key


def test_prompt_and_run_loop(tiny_dataset, tmp_path):
    game, data = tiny_dataset
    task = json.loads((data / "tasks.jsonl").read_text().splitlines()[0])
    parts = build_parts(data, task, game, game.fps)
    assert [p["type"] for p in parts[:-1]] == ["image"] * (len(parts) - 1) and parts[-1]["type"] == "text"
    assert f'"{game.event}"' in parts[-1]["text"] and all(p["path"].exists() for p in parts[:-1])

    calls = []

    def backend(parts):
        calls.append(parts)
        return "I think nothing happens." if len(calls) == 1 else json.dumps({game.event: []})

    out = tmp_path / "preds.jsonl"
    summary = run(backend, data, out, limit=2)
    assert summary == {"new": 2, "parse_failures": 0}  # first answer recovered by the JSON-only retry
    assert run(backend, data, out, limit=2) == {"new": 0, "parse_failures": 0}  # resumes
