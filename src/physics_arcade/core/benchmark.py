"""The official benchmark releases: which datasets, generated with which parameters.

`make` regenerates a release locally (deterministic, so nothing has to be downloaded) and
`verify` checks generated datasets against the fingerprints shipped with the package.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from ..games import NAMES, get_game
from .dataset import write_dataset
from .fingerprint import fingerprint
from .game import STYLES

RELEASES = {"v1": {"n_per_difficulty": 100, "seed_offset": 0, "games": NAMES, "styles": STYLES}}


def dataset_name(game: str, style: str, release: str = "v1") -> str:
    return f"{game}_{release}" + ("" if style == "flat" else f"_{style}")


def expected(release: str = "v1") -> dict:
    return json.loads(resources.files("physics_arcade.benchmarks").joinpath(f"{release}.json").read_text())


def make(root: Path, release: str = "v1", workers: int = 8, games: list[str] | None = None) -> list[Path]:
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
