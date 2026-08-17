#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.semantic_grading import load_existing_grades, load_jsonl, write_review_csv


def require_grade_ids(
    name: str,
    grades: dict[str, dict[str, Any]],
    expected: set[str],
) -> None:
    missing = expected - set(grades)
    extra = set(grades) - expected
    if missing or extra:
        raise ValueError(
            f"{name} ids do not match target: missing={len(missing)}, extra={len(extra)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Majority-adjudicate one confidence class without overwriting primary grades."
    )
    parser.add_argument("--input", required=True, help="Complete experiment records JSONL")
    parser.add_argument("--primary", required=True, help="Complete primary grades JSONL")
    parser.add_argument("--second", required=True, help="Second-pass target-class grades JSONL")
    parser.add_argument("--tie-breaker", help="Third-pass grades for first/second disagreements")
    parser.add_argument("--confidence-class", required=True)
    parser.add_argument("--output", required=True, help="Complete merged adjudicated grades JSONL")
    parser.add_argument("--csv-output")
    parser.add_argument("--disagreement-ids-output")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    record_ids = {str(row["id"]) for row in records}
    target_records = [
        row for row in records if row.get("confidence_class") == args.confidence_class
    ]
    target_ids = {str(row["id"]) for row in target_records}
    if not target_ids:
        parser.error(f"no records found for confidence class {args.confidence_class!r}")

    primary = load_existing_grades(args.primary)
    require_grade_ids("primary", primary, record_ids)
    second = load_existing_grades(args.second)
    require_grade_ids("second", second, target_ids)
    disagreement_ids = sorted(
        row_id
        for row_id in target_ids
        if primary[row_id]["semantic_correct"] != second[row_id]["semantic_correct"]
    )

    if args.disagreement_ids_output:
        Path(args.disagreement_ids_output).write_text(
            "\n".join(disagreement_ids) + ("\n" if disagreement_ids else ""),
            encoding="utf-8",
        )

    tie_breaker: dict[str, dict[str, Any]] = {}
    if disagreement_ids:
        if not args.tie_breaker:
            print(json.dumps({
                "target_records": len(target_ids),
                "first_second_agreements": len(target_ids) - len(disagreement_ids),
                "first_second_disagreements": len(disagreement_ids),
                "disagreement_ids": disagreement_ids,
                "status": "tie_breaker_required",
            }, indent=2))
            raise SystemExit(2)
        tie_breaker = load_existing_grades(args.tie_breaker)
        require_grade_ids("tie_breaker", tie_breaker, set(disagreement_ids))

    merged: dict[str, dict[str, Any]] = {row_id: dict(grade) for row_id, grade in primary.items()}
    changed_from_primary = 0
    for row_id in target_ids:
        pass_grades = [primary[row_id], second[row_id]]
        if row_id in disagreement_ids:
            pass_grades.append(tie_breaker[row_id])
        votes = [bool(grade["semantic_correct"]) for grade in pass_grades]
        majority = sum(votes) > len(votes) / 2
        chosen = next(
            grade for grade in reversed(pass_grades)
            if bool(grade["semantic_correct"]) == majority
        )
        changed_from_primary += int(majority != bool(primary[row_id]["semantic_correct"]))
        merged[row_id] = {
            **primary[row_id],
            "semantic_correct": majority,
            "grade_source": "majority_adjudication" if len(votes) == 3 else "two_pass_agreement",
            "judge_model": chosen.get("judge_model"),
            "matched_alias": chosen.get("matched_alias"),
            "judge_reason": chosen.get("judge_reason"),
            "adjudication": {
                "confidence_class": args.confidence_class,
                "votes": votes,
                "pass_reasons": [
                    primary[row_id].get("judge_reason"),
                    second[row_id].get("judge_reason"),
                    *(
                        [tie_breaker[row_id].get("judge_reason")]
                        if row_id in disagreement_ids
                        else []
                    ),
                ],
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(merged[str(record["id"])], ensure_ascii=False, sort_keys=True) + "\n")
    if args.csv_output:
        write_review_csv(records, merged, args.csv_output)

    target_correct = sum(merged[row_id]["semantic_correct"] is True for row_id in target_ids)
    print(json.dumps({
        "total_records": len(records),
        "target_records": len(target_ids),
        "first_second_agreements": len(target_ids) - len(disagreement_ids),
        "first_second_disagreements": len(disagreement_ids),
        "changed_from_primary": changed_from_primary,
        "adjudicated_target_correct": target_correct,
        "output": str(output),
        "csv_output": args.csv_output,
    }, indent=2))


if __name__ == "__main__":
    main()
