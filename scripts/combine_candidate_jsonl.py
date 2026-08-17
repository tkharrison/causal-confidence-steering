#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ambigqa_dataset import normalize_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    seen_ids = set()
    seen_questions = set()
    for path in args.inputs:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question = normalize_text(row["question"])
            if row["id"] in seen_ids or question in seen_questions:
                continue
            seen_ids.add(row["id"])
            seen_questions.add(question)
            row["combined_from"] = path
            rows.append(row)
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"inputs": args.inputs, "written": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
