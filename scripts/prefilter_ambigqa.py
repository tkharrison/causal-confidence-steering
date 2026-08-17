#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ambigqa_prefilter import prefilter_candidates, write_filtered_candidates
from src.semantic_grading import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filtered-output", required=True)
    parser.add_argument("--split", default="train", choices=("train", "dev", "all"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
        rows = [row for row in rows if args.split == "all" or row["source_split"] == args.split]
        print(json.dumps({"available": len(rows), "maximum_calls": min(len(rows), args.limit or len(rows))}, indent=2))
        return
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    summary = prefilter_candidates(
        args.input,
        args.output,
        api_key=api_key,
        model=args.model,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        concurrency=args.concurrency,
    )
    summary["filtered_count_all_completed"] = write_filtered_candidates(
        args.input, args.output, args.filtered_output
    )
    summary["filtered_output"] = args.filtered_output
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
