#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.semantic_grading import (
    DEFAULT_MODEL,
    GradeConfig,
    grade_records,
    load_existing_grades,
    load_jsonl,
    write_review_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade TriviaQA answers semantically through OpenRouter; safely resumes JSONL output."
    )
    parser.add_argument("--input", required=True, help="Experiment records.jsonl")
    parser.add_argument("--output", required=True, help="Append-only semantic grades JSONL")
    parser.add_argument("--csv-output", help="Review CSV written after successful grading")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", help="Comma-separated record ids to grade, preserving input order")
    parser.add_argument("--confidence-class", help="Only grade records with this confidence_class")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if args.confidence_class:
        records = [
            row for row in records
            if row.get("confidence_class") == args.confidence_class
        ]
        if not records:
            parser.error(f"no records found for --confidence-class {args.confidence_class!r}")
    if args.ids:
        requested = {value.strip() for value in args.ids.split(",") if value.strip()}
        records = [row for row in records if str(row["id"]) in requested]
        found = {str(row["id"]) for row in records}
        missing = requested - found
        if missing:
            parser.error(f"unknown --ids: {','.join(sorted(missing))}")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        records = records[: args.limit]
    existing = load_existing_grades(args.output)
    exact_remaining = sum(
        bool(row.get("correct_exact_match"))
        for row in records
        if str(row["id"]) not in existing
    )
    dry_summary = {
        "input_records": len(records),
        "already_graded": len(set(existing) & {str(row["id"]) for row in records}),
        "exact_matches_remaining_free": exact_remaining,
        "maximum_api_calls_remaining": len(records) - len(existing) - exact_remaining,
        "model": args.model,
    }
    if args.dry_run:
        print(json.dumps(dry_summary, indent=2))
        return

    summary = grade_records(
        records,
        args.output,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        config=GradeConfig(model=args.model, concurrency=args.concurrency),
    )
    grades = load_existing_grades(args.output)
    if args.csv_output:
        write_review_csv(records, grades, args.csv_output)
        summary["csv_output"] = args.csv_output
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
