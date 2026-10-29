"""Command line entry point: `physics-arcade <command> ...`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.game import STYLES
from .games import NAMES, get_game


def main() -> None:
    ap = argparse.ArgumentParser(prog="physics-arcade")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a dataset for one game")
    g.add_argument("--game", choices=NAMES, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--n", type=int, default=100, help="episodes per difficulty")
    g.add_argument("--style", choices=STYLES, default="flat")
    g.add_argument("--seed-offset", type=int, default=0)
    g.add_argument("--workers", type=int, default=8)

    f = sub.add_parser("fit-cartoon-lut", help="refit the color table that segments a game's cartoon frames")
    f.add_argument("--game", choices=NAMES, required=True)
    f.add_argument("--workers", type=int, default=8)

    eo = sub.add_parser("eval-outcome", help="score outcome predictions (jsonl)")
    eo.add_argument("--data", type=Path, required=True)
    eo.add_argument("--pred", type=Path, required=True)

    er = sub.add_parser("eval-rollout", help="score predicted videos (directory)")
    er.add_argument("--data", type=Path, required=True)
    er.add_argument("--pred", type=Path, required=True)
    er.add_argument("--workers", type=int, default=8)

    v = sub.add_parser("visualize", help="side-by-side GT vs prediction video for one episode")
    v.add_argument("--data", type=Path, required=True)
    v.add_argument("--pred", type=Path, required=True, help="prediction directory")
    v.add_argument("--ep", required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--label", default="prediction")

    pr = sub.add_parser("predictability", help="replay episodes under each perturbation")
    pr.add_argument("--data", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True, help="per-episode results (jsonl)")
    pr.add_argument("--runs", type=int, default=50)
    pr.add_argument("--workers", type=int, default=16)

    lb = sub.add_parser("leaderboard", help="Markdown table for several outcome prediction files")
    lb.add_argument("--data", type=Path, required=True)
    lb.add_argument("entries", nargs="+", help="NAME=path/to/predictions.jsonl")

    b = sub.add_parser("baseline", help="write baseline predictions (all, none, game-specific, copy_last, oracle)")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "generate":
        from .core.dataset import write_dataset

        game = get_game(args.game)
        write_dataset(game, args.out, args.n, args.style, args.workers, args.seed_offset)
        print(f"wrote {args.n * len(game.difficulties)} {args.game} episodes to {args.out}")
    elif args.cmd == "fit-cartoon-lut":
        from .core.segment import fit_cartoon_lut

        print(json.dumps(fit_cartoon_lut(get_game(args.game), workers=args.workers), indent=2))
    elif args.cmd == "eval-outcome":
        from .core.evaluate import eval_outcome

        print(json.dumps(eval_outcome(args.data, args.pred), indent=2))
    elif args.cmd == "eval-rollout":
        from .core.evaluate import eval_rollout

        print(json.dumps(eval_rollout(args.data, args.pred, args.workers), indent=2))
    elif args.cmd == "visualize":
        from .core.visualize import compare_video

        pred = next(args.pred / f"{args.ep}{e}" for e in (".mp4", ".gif", ".npy") if (args.pred / f"{args.ep}{e}").exists())
        print(json.dumps(compare_video(args.data, pred, args.ep, args.out, args.label)))
    elif args.cmd == "predictability":
        from .core.predictability import analyze

        print(json.dumps(analyze(args.data, args.out, args.runs, args.workers), indent=2))
    elif args.cmd == "leaderboard":
        from .leaderboard import outcome_table

        entries = [(e.split("=", 1)[0], Path(e.split("=", 1)[1])) for e in args.entries]
        print(outcome_table(args.data, entries))
    elif args.cmd == "baseline":
        from .core.baselines import run_baseline

        run_baseline(args.name, args.data, args.out)
        print(f"wrote {args.name} predictions to {args.out}")
