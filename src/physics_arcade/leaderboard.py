"""Score several outcome-track prediction files and print a Markdown leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

from .catapult.evaluate import eval_outcome


def _parse_failures(path: Path) -> int:
    return sum(bool(json.loads(line).get("parse_error")) for line in path.read_text().splitlines() if line)


def outcome_table(data_dir: Path, entries: list[tuple[str, Path]]) -> str:
    rows = []
    for name, path in entries:
        r = eval_outcome(data_dir, path)
        rows.append((name, r, _parse_failures(path)))
    rows.sort(key=lambda x: -(x[1]["stable_robot_balanced_accuracy"] or 0))
    lines = [
        "| Model | Stable-robot bal. acc. ↑ | Brier ↓ | Stable-episode exact ↑ | Unparsed |",
        "|---|---|---|---|---|",
    ]
    for name, r, bad in rows:
        acc = r["stable_robot_balanced_accuracy"]
        lines.append(
            f"| {name} | {acc:.3f} | {r['brier']:.3f} | {r['stable_episode_exact_match']:.3f} | {bad + r['missing_predictions']} |"
        )
    lines.append(f"\nBrier floor (perfect probabilities): {rows[0][1]['brier_floor']:.3f}")
    return "\n".join(lines)
