from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import string
from typing import Any, Iterable, Sequence


CHOICE_LABELS = tuple("ABCDE")
ABSTENTION_TEXT = "The question does not provide enough information to choose among these answers."


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def canonical_answer(aliases: Sequence[str]) -> str | None:
    candidates = [" ".join(str(value).split()) for value in aliases if str(value).strip()]
    candidates = [value for value in candidates if len(value) <= 80 and len(value.split()) <= 12]
    if not candidates:
        return None
    # AmbigQA commonly lists a surname before a full-name alias. Prefer the
    # shortest multiword alias when one exists so options remain concise but
    # self-contained (for example, "Jeremy Irons" rather than "Irons").
    multiword = [value for value in candidates if len(value.split()) >= 2]
    pool = multiword or candidates
    return min(pool, key=lambda value: (len(value.split()), len(value), value.lower()))


def distinct_pairs(annotation: dict[str, Any]) -> list[dict[str, str]]:
    """Extract disambiguated QA pairs with distinct concise answer texts."""
    if annotation.get("type") != "multipleQAs":
        return []
    result: list[dict[str, str]] = []
    seen_answers: set[str] = set()
    for pair in annotation.get("qaPairs", []):
        answer = canonical_answer(pair.get("answer", []))
        question = " ".join(str(pair.get("question", "")).split())
        if not answer or not question:
            continue
        normalized = normalize_text(answer)
        if not normalized or normalized in seen_answers:
            continue
        seen_answers.add(normalized)
        result.append({"interpretation": question, "answer": answer})
    return result


def _stable_seed(item_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def select_four_pairs(
    annotations: Sequence[dict[str, Any]], *, item_id: str, seed: int
) -> tuple[list[dict[str, str]], int] | None:
    """Choose one richest annotation, then four pairs without cherry-picking."""
    candidates = [(distinct_pairs(annotation), index) for index, annotation in enumerate(annotations)]
    candidates = [(pairs, index) for pairs, index in candidates if len(pairs) >= 4]
    if not candidates:
        return None
    # Prefer the annotation preserving the most interpretations; annotation
    # index is a deterministic tie-breaker. Randomly subsample four from it.
    pairs, annotation_index = max(candidates, key=lambda value: (len(value[0]), -value[1]))
    rng = random.Random(_stable_seed(item_id, seed))
    pairs = list(pairs)
    rng.shuffle(pairs)
    return pairs[:4], annotation_index


def convert_ambigqa_rows(
    rows: Iterable[dict[str, Any]],
    *,
    split: str,
    seed: int = 42,
    excluded_questions: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded = {normalize_text(value) for value in excluded_questions}
    counts = {
        "input_rows": 0,
        "excluded_prior_question": 0,
        "lacks_four_distinct_concise_answers": 0,
        "accepted": 0,
    }
    converted: list[dict[str, Any]] = []
    for row in rows:
        counts["input_rows"] += 1
        question = " ".join(str(row.get("question", "")).split())
        if normalize_text(question) in excluded:
            counts["excluded_prior_question"] += 1
            continue
        selected = select_four_pairs(row.get("annotations", []), item_id=str(row["id"]), seed=seed)
        if selected is None:
            counts["lacks_four_distinct_concise_answers"] += 1
            continue
        pairs, annotation_index = selected
        converted.append({
            "id": f"ambigqa_{split}_{row['id']}",
            "source": "AmbigQA/AmbigNQ-light",
            "source_split": split,
            "source_id": str(row["id"]),
            "question_type": "ambiguous",
            "question": question,
            "selected_annotation_index": annotation_index,
            "selection_seed": seed,
            "interpretations": pairs,
            "substantive_options": [pair["answer"] for pair in pairs],
        })
        counts["accepted"] += 1
    return converted, counts


def recognition_variants(candidate: dict[str, Any], *, seed: int = 42) -> list[dict[str, Any]]:
    """Create five counterbalanced variants with abstention at every label once."""
    substantive = list(candidate["substantive_options"])
    variants = []
    for variant_index, abstention_label in enumerate(CHOICE_LABELS):
        rng = random.Random(_stable_seed(candidate["id"], seed + variant_index))
        shuffled = list(substantive)
        rng.shuffle(shuffled)
        options: dict[str, str] = {}
        substantive_iter = iter(shuffled)
        for label in CHOICE_LABELS:
            options[label] = ABSTENTION_TEXT if label == abstention_label else next(substantive_iter)
        variants.append({
            "variant_index": variant_index,
            "abstention_label": abstention_label,
            "options": options,
        })
    return variants


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{label}. {options[label]}" for label in CHOICE_LABELS if label in options)


def recognition_prompt(question: str, variant: dict[str, Any]) -> str:
    return (
        f"Question: {question}\n\n{format_options(variant['options'])}\n\n"
        "Choose the single best option. On the first line write exactly `Choice: X`, "
        "where X is one of A, B, C, D, or E. Then explain the reason for your choice "
        "in one to three sentences."
    )


def recognition_choice_prompt(question: str, variant: dict[str, Any]) -> str:
    """Prompt whose next token is scored over A-E for cheap first-pass screening."""
    return (
        f"Question: {question}\n\n{format_options(variant['options'])}\n\n"
        "Choose the single best option. Respond with exactly one letter: A, B, C, D, or E.\n\n"
        "Choice:"
    )


def forced_choice_prompt(question: str, options: dict[str, str]) -> str:
    labels = tuple(options)
    return (
        f"Question: {question}\n\n{format_options(options)}\n\n"
        f"You must select exactly one of {', '.join(labels)} even if the question is ambiguous. "
        "Do not explain, qualify, or abstain. End your response with `Choice: X`, "
        f"where X is one of {', '.join(labels)}.\n\nChoice:"
    )


def parse_choice(text: str, labels: Sequence[str] = CHOICE_LABELS) -> str | None:
    allowed = "".join(re.escape(label) for label in labels)
    patterns = (
        rf"(?im)^\s*(?:\*\*)?choice(?:\*\*)?\s*:\s*([{allowed}])\b",
        rf"(?im)^\s*([{allowed}])(?:[.)]|\b)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def load_prior_questions(records_files: Sequence[str | Path]) -> list[str]:
    questions: list[str] = []
    for path in records_files:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("question"):
                    questions.append(str(row["question"]))
    return questions
