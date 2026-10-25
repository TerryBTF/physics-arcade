"""Command line entry point: `physics-arcade <command> ...`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .angry_birds import baselines, dataset, evaluate


def main() -> None:
    ap = argparse.ArgumentParser(prog="physics-arcade")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate an Angry Birds dataset")
    g.add_argument("--out", type=Path, default=Path("data/angry_birds_v0"))
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed-offset", type=int, default=0)
    g.add_argument("--workers", type=int, default=8)

    eo = sub.add_parser("eval-outcome", help="score pig-survival predictions (jsonl)")
    eo.add_argument("--data", type=Path, required=True)
    eo.add_argument("--pred", type=Path, required=True)

    er = sub.add_parser("eval-rollout", help="score predicted videos (directory)")
    er.add_argument("--data", type=Path, required=True)
    er.add_argument("--pred", type=Path, required=True)
    er.add_argument("--workers", type=int, default=8)

    b = sub.add_parser("baseline", help="write baseline predictions")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--name", required=True, choices=[*baselines.OUTCOME_BASELINES, *baselines.ROLLOUT_BASELINES, "oracle"])
    b.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "generate":
        dataset.write_dataset(args.out, args.n, args.seed_offset, args.workers)
        print(f"wrote {args.n} episodes to {args.out}")
    elif args.cmd == "eval-outcome":
        print(json.dumps(evaluate.eval_outcome(args.data, args.pred), indent=2))
    elif args.cmd == "eval-rollout":
        print(json.dumps(evaluate.eval_rollout(args.data, args.pred, args.workers), indent=2))
    elif args.cmd == "baseline":
        if args.name in baselines.OUTCOME_BASELINES:
            baselines.run_outcome_baseline(args.name, args.data, args.out)
        elif args.name == "oracle":
            baselines.write_oracle_rollouts(args.data, args.out)
        else:
            baselines.run_rollout_baseline(args.name, args.data, args.out)
        print(f"wrote {args.name} predictions to {args.out}")
