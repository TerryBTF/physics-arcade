"""Render one cartoon-style GIF per game for the README (docs/media/<game>.gif).

Picks the first episode of the given difficulty in which the event happens to at least
`min_events` targets, so the clip shows something interesting.

    uv run python scripts/make_gifs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from physics_arcade.games import NAMES, get_game

OUT = Path(__file__).resolve().parents[1] / "docs" / "media"
CHOICE = {"catapult": ("hard", 2), "pool": ("hard", 1), "dominoes": ("hard", 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        game = get_game(name)
        difficulty, min_events = CHOICE[name]
        for seed in range(200):
            ep = game.generate(seed, difficulty, "cartoon")
            changed = len(ep.recorded_ids) if name != "catapult" else len(ep.target_ids) - len(ep.recorded_ids)
            if changed >= min_events:
                break
        frames = [Image.fromarray(f).resize((480, 270), Image.LANCZOS) for f in ep.display_frames[::2]]
        frames = [ep.labeled_frame.convert("RGB").resize((480, 270), Image.LANCZOS)] * 8 + frames + [frames[-1]] * 10
        # One palette for the whole clip, built from several frames so no object loses its color.
        sample = [frames[i] for i in np.linspace(0, len(frames) - 1, 6).astype(int)]
        strip = Image.fromarray(np.concatenate([np.asarray(f) for f in sample], axis=0))
        pal = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        path = OUT / f"{name}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // 15, loop=0, optimize=True)
        print(name, "seed", seed, "events", ep.recorded_ids, f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
