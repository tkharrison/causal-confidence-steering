from __future__ import annotations

import json

from src.semantic_grading import (
    _parse_judgment,
    _request_payload,
    exact_grade,
    load_existing_grades,
    write_review_csv,
)


def sample_row() -> dict:
    return {
        "id": "tc_1",
        "question": "What is the capital of France?",
        "answer_text": "Paris",
        "aliases": ["Paris", "City of Paris"],
        "confidence_class": "Likely",
        "correct_exact_match": True,
    }


def test_exact_grade() -> None:
    grade = exact_grade(sample_row())
    assert grade is not None
    assert grade["semantic_correct"] is True
    assert grade["grade_source"] == "normalized_exact_match"


def test_prompt_excludes_confidence() -> None:
    payload = _request_payload(sample_row(), "test/model")
    user_content = payload["messages"][1]["content"]
    assert "Likely" not in user_content
    assert "confidence_class" not in user_content
    assert "list can be incomplete" in payload["messages"][0]["content"]
    assert payload["temperature"] == 0


def test_parse_judgment() -> None:
    parsed = _parse_judgment({
        "choices": [{"message": {"content": json.dumps({
            "correct": True,
            "matched_alias": "Paris",
            "reason": "Exact city.",
        })}}]
    })
    assert parsed["correct"] is True


def test_resume_and_csv(tmp_path) -> None:
    grade_path = tmp_path / "grades.jsonl"
    grade_path.write_text(json.dumps({"id": "tc_1", "semantic_correct": True}) + "\n")
    grades = load_existing_grades(grade_path)
    assert set(grades) == {"tc_1"}
    csv_path = tmp_path / "review.csv"
    write_review_csv([sample_row()], grades, csv_path)
    text = csv_path.read_text()
    assert "semantic_correct" in text
    assert "Paris" in text
