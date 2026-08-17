#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REJECTIONS = {
    "ambigqa_train_8329007430165668321": (
        "The question asks for all four aces, but each option supplies only one ace."
    ),
    "ambigqa_train_-3086525406949326143": (
        "Two country-specific readings collapse to the same generic President option."
    ),
    "ambigqa_dev_5876321266833815192": (
        "The four readings collapse to only two semantically distinct equinox dates."
    ),
    "ambigqa_dev_434290836324078234": (
        "At least one interpretation has multiple opening acts rather than answer aliases."
    ),
}

NOTE_OVERRIDES = {
    "ambigqa_dev_-2928173928559810904": (
        "No information is provided about the specific television series or whether the role is Luv or Kush."
    ),
    "ambigqa_train_2557408066370024925": (
        "No information is provided about the specific time period or educational setting."
    ),
}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    candidates = {row["id"]: row for row in load(args.candidates)}
    passes = [row for row in load(args.qualification) if row["recognition_pass_5_of_5"]]
    curated = []
    rejected = []
    for result in passes:
        item_id = result["id"]
        if item_id in REJECTIONS:
            rejected.append({"id": item_id, "question": result["question"], "reason": REJECTIONS[item_id]})
            continue
        row = dict(candidates[item_id])
        note = NOTE_OVERRIDES.get(item_id, row["context_note"])
        row["context_note"] = note
        row["display_question"] = f"{row['question']}\n\n{note}"
        if item_id == "ambigqa_train_-5746040628732928353":
            row["substantive_options"] = [
                "Achilles tendon" if answer == "Achilles tenden" else answer
                for answer in row["substantive_options"]
            ]
            for pair in row["interpretations"]:
                if pair["answer"] == "Achilles tenden":
                    pair["answer"] = "Achilles tendon"
        row["curation"] = {
            "source_qualification": args.qualification,
            "passed_recognition_checks": 10,
            "full_aliases": True,
            "context_note_manually_revised": item_id in NOTE_OVERRIDES,
        }
        curated.append(row)

    curated.sort(key=lambda row: row["id"])
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in curated),
        encoding="utf-8",
    )
    report = {
        "full_recognition_passes": len(passes),
        "manual_rejections": len(rejected),
        "curated_for_requalification": len(curated),
        "rejected": rejected,
        "note_overrides": NOTE_OVERRIDES,
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
