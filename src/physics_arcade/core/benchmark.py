"""The official benchmark releases: which datasets, generated with which parameters.

`make` regenerates a release locally (deterministic, so nothing has to be downloaded) and
`verify` checks generated datasets against the fingerprints shipped with the package.

v1-lite is a fixed, stratified 150-episode subset of the v1 cartoon datasets (50 per game)
for models that are expensive to run; its episode ids ship in benchmarks/v1-lite.json.
"""

from __future__ import annotations

import json
import random
import shutil
from importlib import resources
from pathlib import Path

from ..games import NAMES, get_game
from .dataset import write_dataset
from .fingerprint import fingerprint
from .game import STYLES

RELEASES = {
    "v1": {"n_per_difficulty": 100, "seed_offset": 0, "games": NAMES, "styles": STYLES},
    "v1-lite": {"parent": "v1", "parent_style": "cartoon", "per_difficulty": {"easy": 17, "medium": 17, "hard": 16},
                "games": NAMES, "styles": ("lite",)},
}


def dataset_name(game: str, style: str, release: str = "v1") -> str:
    if release == "v1-lite":
        return f"{game}_v1_lite"
    return f"{game}_{release}" + ("" if style == "flat" else f"_{style}")


def lite_ids(game: str) -> list[str]:
    """The v1-lite episodes of a game: a seeded stratified sample of the v1 episode ids."""
    spec, parent = RELEASES["v1-lite"], RELEASES["v1"]
    rng = random.Random(f"v1-lite-{game}")
    ids = []
    for difficulty, k in spec["per_difficulty"].items():
        pool = [f"{difficulty}_{i:04d}" for i in range(parent["n_per_difficulty"])]
        ids += sorted(rng.sample(pool, k))
    return ids


def make_lite(root: Path, games: list[str] | None = None) -> list[Path]:
    """Copy the v1-lite episodes out of the v1 cartoon datasets (run make-benchmark first)."""
    spec = RELEASES["v1-lite"]
    shipped = expected("v1-lite")
    out = []
    for g in games or spec["games"]:
        src = root / dataset_name(g, spec["parent_style"], spec["parent"])
        dst = root / dataset_name(g, "lite", "v1-lite")
        ids = shipped[dst.name]["ids"] if dst.name in shipped else lite_ids(g)
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "inputs").mkdir(parents=True)
        (dst / "answers").mkdir()
        tasks = {t["id"]: t for t in map(json.loads, (src / "tasks.jsonl").read_text().splitlines())}
        (dst / "tasks.jsonl").write_text("".join(json.dumps(tasks[i]) + "\n" for i in ids))
        for i in ids:
            shutil.copytree(src / "inputs" / i, dst / "inputs" / i)
            for f in (src / "answers").glob(f"{i}[._]*"):
                shutil.copy(f, dst / "answers" / f.name)
        manifest = json.loads((src / "manifest.json").read_text())
        manifest.update({"version": f"{g}-v1-lite", "subset_of": src.name, "num_episodes": len(ids),
                         "difficulties": spec["per_difficulty"]})
        (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
        out.append(dst)
    return out


def expected(release: str = "v1") -> dict:
    return json.loads(resources.files("physics_arcade.benchmarks").joinpath(f"{release}.json").read_text())


def make(root: Path, release: str = "v1", workers: int = 8, games: list[str] | None = None) -> list[Path]:
    if release == "v1-lite":
        return make_lite(root, games)
    spec = RELEASES[release]
    out = []
    for g in games or spec["games"]:
        for s in spec["styles"]:
            d = root / dataset_name(g, s, release)
            write_dataset(get_game(g), d, spec["n_per_difficulty"], s, workers, spec["seed_offset"])
            out.append(d)
    return out


def verify(data_dir: Path, release: str = "v1") -> dict:
    name = data_dir.name
    want = expected(release).get(name)
    if want is None:
        raise KeyError(f"{name} is not a {release} dataset; expected one of {sorted(expected(release))}")
    got = fingerprint(data_dir)
    return {
        "dataset": name,
        "labels_match": got["labels"] == want["labels"] and got["episodes"] == want["episodes"],
        "inputs_match": got["inputs"] == want["inputs"],
        "episodes": got["episodes"],
    }
