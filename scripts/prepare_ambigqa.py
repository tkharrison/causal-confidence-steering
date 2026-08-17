#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ambigqa_dataset import convert_ambigqa_rows, load_prior_questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--exclude-records", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    excluded = load_prior_questions(args.exclude_records)
    all_candidates = []
    report = {
        "seed": args.seed,
        "excluded_question_count": len(excluded),
        "splits": {},
    }
    for split, path in (("train", args.train), ("dev", args.dev)):
        rows = json.loads(Path(path).read_text())
        candidates, counts = convert_ambigqa_rows(
            rows, split=split, seed=args.seed, excluded_questions=excluded
        )
        all_candidates.extend(candidates)
        report["splits"][split] = counts

    all_candidates.sort(key=lambda row: row["id"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_candidates),
        encoding="utf-8",
    )
    report["accepted_total"] = len(all_candidates)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
