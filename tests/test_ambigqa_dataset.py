from __future__ import annotations

from src.ambigqa_dataset import (
    ABSTENTION_TEXT,
    canonical_answer,
    convert_ambigqa_rows,
    parse_choice,
    recognition_choice_prompt,
    recognition_variants,
)


def _row() -> dict:
    return {
        "id": "item-1",
        "question": "Which version is meant?",
        "annotations": [{
            "type": "multipleQAs",
            "qaPairs": [
                {"question": f"Meaning {index}?", "answer": [f"Answer {index}"]}
                for index in range(6)
            ],
        }],
    }


def test_conversion_is_deterministic_and_selects_four() -> None:
    first, counts = convert_ambigqa_rows([_row()], split="dev", seed=42)
    second, _ = convert_ambigqa_rows([_row()], split="dev", seed=42)
    assert first == second
    assert counts["accepted"] == 1
    assert len(first[0]["interpretations"]) == 4
    assert len(set(first[0]["substantive_options"])) == 4


def test_abstention_is_counterbalanced() -> None:
    candidates, _ = convert_ambigqa_rows([_row()], split="dev", seed=42)
    variants = recognition_variants(candidates[0])
    assert [variant["abstention_label"] for variant in variants] == list("ABCDE")
    assert all(list(variant["options"].values()).count(ABSTENTION_TEXT) == 1 for variant in variants)


def test_parse_choice() -> None:
    assert parse_choice("Choice: C\nBecause it is ambiguous.") == "C"
    assert parse_choice("**Choice**: E") == "E"
    assert parse_choice("No valid selection") is None


def test_recognition_choice_prompt_ends_at_choice_slot() -> None:
    candidates, _ = convert_ambigqa_rows([_row()], split="dev", seed=42)
    variant = recognition_variants(candidates[0])[0]
    prompt = recognition_choice_prompt(candidates[0]["question"], variant)
    assert prompt.endswith("Choice:")
    assert "explain" not in prompt.lower()


def test_canonical_answer_prefers_self_contained_alias() -> None:
    assert canonical_answer(["Irons", "Jeremy Irons", "Jeremy John Irons"]) == "Jeremy Irons"
    assert canonical_answer(["3", "three"]) == "3"
