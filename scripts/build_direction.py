#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.confidence_direction import build_direction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grades", help="Complete semantic-grades JSONL keyed by record id")
    parser.add_argument("--correctness-field", default="semantic_correct")
    parser.add_argument("--n-per-class", type=int, default=25)
    parser.add_argument("--include-incorrect", action="store_true")
    args = parser.parse_args()
    result = build_direction(
        args.run_dir,
        args.output,
        grades_file=args.grades,
        correctness_field=args.correctness_field,
        n_per_class=args.n_per_class,
        correct_only=not args.include_incorrect,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
