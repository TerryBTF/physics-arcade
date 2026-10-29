"""Command line entry point: `physics-arcade <command> ...`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .slingshot import baselines, dataset, evaluate


def main() -> None:
    ap = argparse.ArgumentParser(prog="physics-arcade")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a Slingshot dataset")
    g.add_argument("--out", type=Path, default=Path("data/slingshot_v0"))
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed-offset", type=int, default=0)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--style", choices=dataset.STYLES, default="flat")

    sub.add_parser("fit-cartoon-lut", help="refit the color table used to segment cartoon-style frames")

    eo = sub.add_parser("eval-outcome", help="score pig-survival predictions (jsonl)")
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

    pr = sub.add_parser("predictability", help="re-simulate episodes under small perturbations")
    pr.add_argument("--data", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True, help="per-episode results (jsonl)")
    pr.add_argument("--runs", type=int, default=50)
    pr.add_argument("--workers", type=int, default=16)

    lb = sub.add_parser("leaderboard", help="Markdown table for several outcome prediction files")
    lb.add_argument("--data", type=Path, required=True)
    lb.add_argument("entries", nargs="+", help="NAME=path/to/predictions.jsonl")

    b = sub.add_parser("baseline", help="write baseline predictions")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--name", required=True, choices=[*baselines.OUTCOME_BASELINES, *baselines.ROLLOUT_BASELINES, "oracle"])
    b.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "generate":
        dataset.write_dataset(args.out, args.n, args.seed_offset, args.workers, args.style)
        print(f"wrote {args.n} episodes to {args.out}")
    elif args.cmd == "fit-cartoon-lut":
        from .slingshot.cartoon import build_lut

        print(json.dumps(build_lut(), indent=2))
    elif args.cmd == "eval-outcome":
        print(json.dumps(evaluate.eval_outcome(args.data, args.pred), indent=2))
    elif args.cmd == "eval-rollout":
        print(json.dumps(evaluate.eval_rollout(args.data, args.pred, args.workers), indent=2))
    elif args.cmd == "visualize":
        from .slingshot.visualize import compare_video

        pred = next(args.pred / f"{args.ep}{e}" for e in (".mp4", ".gif", ".npy") if (args.pred / f"{args.ep}{e}").exists())
        print(json.dumps(compare_video(args.data, pred, args.ep, args.out, args.label)))
    elif args.cmd == "predictability":
        from .slingshot import predictability

        summary = predictability.analyze(args.data, args.out, args.runs, args.workers)
        summary["runs_per_condition"] = args.runs
        print(json.dumps(summary, indent=2))
    elif args.cmd == "leaderboard":
        from .leaderboard import outcome_table

        entries = [(e.split("=", 1)[0], Path(e.split("=", 1)[1])) for e in args.entries]
        print(outcome_table(args.data, entries))
    elif args.cmd == "baseline":
        if args.name in baselines.OUTCOME_BASELINES:
            baselines.run_outcome_baseline(args.name, args.data, args.out)
        elif args.name == "oracle":
            baselines.write_oracle_rollouts(args.data, args.out)
        else:
            baselines.run_rollout_baseline(args.name, args.data, args.out)
        print(f"wrote {args.name} predictions to {args.out}")
