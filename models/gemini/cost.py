"""Cost of a run from its <out>.usage.jsonl (thinking tokens are billed as output).

    .venv/bin/python cost.py ../../preds/gemini_pool_lite.usage.jsonl --input-price 2 --output-price 12
Prices are USD per million tokens; take them from the official pricing page.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("usage", nargs="+")
    ap.add_argument("--input-price", type=float, required=True)
    ap.add_argument("--output-price", type=float, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for path in args.usage for line in open(path) if line.strip()]
    inp = sum(r["prompt_tokens"] or 0 for r in rows)
    out = sum((r["output_tokens"] or 0) + (r["thought_tokens"] or 0) for r in rows)
    usd = inp / 1e6 * args.input_price + out / 1e6 * args.output_price
    n = max(len(rows), 1)
    print(f"{len(rows)} calls | input {inp:,} tok ({inp / n:,.0f}/call) | output incl. thinking {out:,} tok ({out / n:,.0f}/call) | ${usd:.2f}")


if __name__ == "__main__":
    main()
