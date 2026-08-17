from __future__ import annotations

import re
import unicodedata
from typing import Any


SAMPLE_ITEMS = (
    {"id": "sample-0", "question": "What is the capital of France?", "aliases": ["Paris"]},
    {"id": "sample-1", "question": "What chemical element has atomic number 79?", "aliases": ["gold", "Au"]},
)


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(answer: str, aliases: list[str]) -> bool | None:
    if not aliases:
        return None
    normalized = normalize_answer(answer)
    return any(normalized == normalize_answer(alias) for alias in aliases)


def load_triviaqa(limit: int, split: str, *, sample_data: bool = False) -> list[dict[str, Any]]:
    if sample_data:
        return [dict(item) for item in SAMPLE_ITEMS[:limit]]
    from datasets import load_dataset

    dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)
    rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for index, row in enumerate(dataset):
        question = str(row["question"]).strip()
        normalized_question = normalize_answer(question)
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        answer = row.get("answer", {})
        aliases = list(answer.get("aliases", [])) if isinstance(answer, dict) else []
        rows.append({
            "id": str(row.get("question_id", index)),
            "question": question,
            "aliases": aliases,
        })
        if len(rows) >= limit:
            break
    return rows
