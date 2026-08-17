#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.introspection_protocol import MEASURE_NAMES, PROTOCOL_VERSION, measurement_prompt


def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(definite_path: str | Path, ambiguous_path: str | Path) -> dict[str, object]:
    definite = load(definite_path)
    ambiguous = load(ambiguous_path)
    rows = definite + ambiguous
    counts = Counter(row["condition"] for row in rows)
    assert counts == {
        "definite_correct": 100,
        "definite_false": 100,
        "ambiguous": 100,
    }
    assert len(rows) == len({row["stimulus_id"] for row in rows}) == 300
    assert all(len(row["options"]) == 4 for row in rows)
    assert all(len(set(row["options"].values())) == 4 for row in rows)
    assert all(row["replayed_answer"] for row in definite)
    assert all(row["replayed_answer"] is None for row in ambiguous)
    prompts = {
        measure: measurement_prompt("validation-item", measure, seed=20260816)[0]
        for measure in MEASURE_NAMES
    }
    assert prompts["confidence_manipulation_check"].endswith("**Confidence**:")
    return {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "total": len(rows),
        "condition_counts": dict(counts),
        "all_ids_unique": True,
        "all_options_four_and_unique": True,
        "measures": list(MEASURE_NAMES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definite", required=True)
    parser.add_argument("--ambiguous", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate(args.definite, args.ambiguous)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
