"""Score several outcome-track prediction files and print a Markdown leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

from .core.evaluate import eval_outcome


def _parse_failures(path: Path) -> int:
    return sum(bool(json.loads(line).get("parse_error")) for line in path.read_text().splitlines() if line)


def _fmt(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def outcome_table(data_dir: Path, entries: list[tuple[str, Path]]) -> str:
    rows = [(name, eval_outcome(data_dir, path), _parse_failures(path)) for name, path in entries]
    rows.sort(key=lambda x: -(x[1]["stable_balanced_accuracy"] or 0))
    diffs = list(rows[0][1]["by_difficulty"])
    head = ["Model", "Stable bal. acc. ↑", *[f"{d} ↑" for d in diffs], "Brier ↓", "Unparsed"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r, bad in rows:
        per = [_fmt(r["by_difficulty"][d]["stable_balanced_accuracy"]) for d in diffs]
        cells = [name, _fmt(r["stable_balanced_accuracy"]), *per, _fmt(r["brier"]), str(bad + r["missing_predictions"])]
        lines.append("| " + " | ".join(cells) + " |")
    r0 = rows[0][1]
    lines.append(
        f"\nStable targets: {r0['stable_fraction']:.0%} of {r0['targets']}. "
        f"Brier floor (perfect probabilities): {r0['brier_floor']:.3f}."
    )
    return "\n".join(lines)


def model_row(root: Path, preds: Path, model: str, release: str = "v1") -> str:
    """One leaderboard row for a model's six prediction files
    (<preds>/<model>_<game>_<style>.jsonl, scored on <root>/<game>_<release>[_<style>])."""
    from .core.benchmark import RELEASES, dataset_name

    spec = RELEASES[release]
    cells = []
    for g in spec["games"]:
        scores = []
        for s in spec["styles"]:
            path = preds / f"{model}_{g}_{s}.jsonl"
            if not path.exists():
                scores.append("missing")
                continue
            r = eval_outcome(root / dataset_name(g, s, release), path)
            scores.append(_fmt(r["stable_balanced_accuracy"]))
        cells.append(" / ".join(scores))
    return f"| {model} | " + " | ".join(cells) + " |"
