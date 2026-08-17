#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import random
import re


HIGH_CONFIDENCE = {"Likely", "Highly likely", "Very good chance", "Almost certain"}


def normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def collect_excluded_ids(direction_metadata: str, steering_glob: str) -> set[str]:
    metadata = json.loads(Path(direction_metadata).read_text())
    excluded = set(metadata.get("selected_low_ids", [])) | set(metadata.get("selected_high_ids", []))
    for path in glob.glob(steering_glob):
        if not path.endswith(".jsonl"):
            continue
        for row in load_jsonl(path):
            if row.get("id"):
                excluded.add(str(row["id"]))
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--grades", required=True)
    parser.add_argument("--direction-metadata", required=True)
    parser.add_argument("--steering-glob", default="work/*steering*.jsonl")
    parser.add_argument("--ambiguous", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    grades = {row["id"]: row for row in load_jsonl(args.grades)}
    excluded_ids = collect_excluded_ids(args.direction_metadata, args.steering_glob)
    excluded_questions = {normalize(row["question"]) for row in load_jsonl(args.ambiguous)}
    counts = {
        "input": 0, "semantic_incorrect": 0, "not_high_confidence": 0,
        "used_in_direction_or_validation": 0, "ambiguous_question_overlap": 0,
        "answer_not_concise": 0, "eligible": 0,
    }
    eligible = []
    for row in load_jsonl(args.records):
        counts["input"] += 1
        grade = grades.get(row["id"])
        if not grade or not grade.get("semantic_correct"):
            counts["semantic_incorrect"] += 1
            continue
        if row.get("confidence_class") not in HIGH_CONFIDENCE:
            counts["not_high_confidence"] += 1
            continue
        if row["id"] in excluded_ids:
            counts["used_in_direction_or_validation"] += 1
            continue
        if normalize(row["question"]) in excluded_questions:
            counts["ambiguous_question_overlap"] += 1
            continue
        answer = " ".join(str(row.get("answer_text", "")).split())
        if not (2 <= len(answer) <= 60 and 1 <= len(answer.split()) <= 8):
            counts["answer_not_concise"] += 1
            continue
        eligible.append({
            "id": row["id"],
            "source": "TriviaQA_full_4k_unused",
            "question": row["question"],
            "aliases": row.get("aliases", []),
            "qwen_correct_answer": answer,
            "baseline_confidence_class": row["confidence_class"],
            "baseline_confidence_midpoint": row["confidence_midpoint"],
            "semantic_grade": grade,
        })
    counts["eligible"] = len(eligible)
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[:args.limit]
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8"
    )
    report = {
        **counts, "excluded_id_count": len(excluded_ids), "selected": len(selected),
        "seed": args.seed, "output": args.output,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
