#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ambigqa_dataset import CHOICE_LABELS, format_options


FORMAT_VERSION = "qa_four_options_answer_panl_v1"


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path: str, rows: list[dict]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def even_proportional_quotas(rows: list[dict], *, per_condition: int) -> dict[str, int]:
    """Allocate paired items across confidence classes without breaking balance."""
    available_pairs = {
        confidence_class: len(group) // 2
        for confidence_class, group in _group_by_confidence(rows).items()
    }
    if sum(available_pairs.values()) < per_condition:
        raise ValueError(
            f"Need {per_condition} matched pairs, found only {sum(available_pairs.values())}"
        )
    total_available = sum(available_pairs.values())
    exact = {
        confidence_class: per_condition * count / total_available
        for confidence_class, count in available_pairs.items()
    }
    pair_quotas = {
        confidence_class: min(available_pairs[confidence_class], int(value))
        for confidence_class, value in exact.items()
    }
    remaining = per_condition - sum(pair_quotas.values())
    order = sorted(
        available_pairs,
        key=lambda confidence_class: (
            exact[confidence_class] - int(exact[confidence_class]),
            available_pairs[confidence_class],
            confidence_class,
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for confidence_class in order:
            if pair_quotas[confidence_class] < available_pairs[confidence_class]:
                pair_quotas[confidence_class] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("Could not allocate the requested matched pairs")
    return {key: value * 2 for key, value in pair_quotas.items()}


def _group_by_confidence(rows: list[dict]) -> dict[str, list[dict]]:
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["baseline_confidence_class"], []).append(row)
    return by_class


def matched_select(rows: list[dict], *, seed: int, per_condition: int) -> list[dict]:
    rng = random.Random(seed)
    by_class = _group_by_confidence(rows)
    quotas = even_proportional_quotas(rows, per_condition=per_condition)
    selected = []
    for confidence_class, quota in quotas.items():
        pool = by_class.get(confidence_class, [])
        if len(pool) < quota:
            raise ValueError(f"Need {quota} {confidence_class} items, found {len(pool)}")
        rng.shuffle(pool)
        chosen = pool[:quota]
        chosen.sort(key=lambda row: (
            len(row["qwen_correct_answer"].split()), len(row["question"]), row["id"]
        ))
        for index in range(0, len(chosen), 2):
            pair = chosen[index:index + 2]
            rng.shuffle(pair)
            pair[0] = {**pair[0], "assigned_condition": "definite_correct"}
            pair[1] = {**pair[1], "assigned_condition": "definite_false"}
            selected.extend(pair)
    rng.shuffle(selected)
    return selected


def format_definite(row: dict, seed: int) -> dict:
    condition = row["assigned_condition"]
    options = row["definite_options"]
    if condition == "definite_correct":
        replayed_answer = row["qwen_correct_answer"]
        replay_source = "qwen_genuine_semantically_verified_correct"
    else:
        index = int(row["most_plausible_distractor_index"])
        replayed_answer = row["distractors"][index]
        replay_source = "independently_validated_plausible_false_substitution"
    replayed_label = next(label for label, answer in options.items() if answer == replayed_answer)
    question_with_options = f"{row['question']}\n\n{format_options(options)}"
    replay_text = f"Question: {question_with_options}\n\nAnswer: {replayed_answer}\n"
    return {
        "stimulus_id": f"{condition}:{row['id']}",
        "source_id": row["id"],
        "condition": condition,
        "format_version": FORMAT_VERSION,
        "question": row["question"],
        "context_note": None,
        "display_question": row["question"],
        "options": options,
        "correct_label": row["correct_label"],
        "correct_answer": row["qwen_correct_answer"],
        "replayed_label": replayed_label,
        "replayed_answer": replayed_answer,
        "replay_source": replay_source,
        "question_with_options": question_with_options,
        "replay_text_through_panl": replay_text,
        "baseline_confidence_class": row["baseline_confidence_class"],
        "baseline_confidence_midpoint": row["baseline_confidence_midpoint"],
        "answer_type": row["answer_type"],
        "qwen_correct_choice_pass": row["qwen_correct_choice_pass"],
        "assignment_seed": seed,
        "provenance": {
            "semantic_grade": row["semantic_grade"],
            "distractor_validation_reason": row["validation_reason"],
            "qwen_choice_response": row["qwen_choice_response"],
        },
    }


def format_ambiguous(row: dict, seed: int) -> dict:
    options_list = list(row["substantive_options"])
    random.Random(f"{seed}:{row['id']}:ambiguous-options").shuffle(options_list)
    options = dict(zip(CHOICE_LABELS[:4], options_list))
    display_question = row.get("display_question", row["question"])
    question_with_options = f"{display_question}\n\n{format_options(options)}"
    return {
        "stimulus_id": f"ambiguous:{row['id']}",
        "source_id": row["id"],
        "condition": "ambiguous",
        "format_version": FORMAT_VERSION,
        "question": row["question"],
        "context_note": row.get("context_note"),
        "display_question": display_question,
        "options": options,
        "correct_label": None,
        "correct_answer": None,
        "replayed_label": None,
        "replayed_answer": None,
        "replay_source": "qwen_forced_choice_to_be_obtained_before_steering",
        "question_with_options": question_with_options,
        "replay_text_through_panl": None,
        "assignment_seed": seed,
        "recognition_screen": {
            "source_family": row.get("stimulus_source_family"),
            "quality_tier": row.get("final_quality_tier"),
        },
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for condition in ("definite_correct", "definite_false"):
        group = [row for row in rows if row["condition"] == condition]
        result[condition] = {
            "n": len(group),
            "mean_question_characters": statistics.mean(len(row["question"]) for row in group),
            "mean_replayed_answer_words": statistics.mean(len(row["replayed_answer"].split()) for row in group),
            "mean_baseline_confidence_midpoint": statistics.mean(row["baseline_confidence_midpoint"] for row in group),
            "confidence_classes": {
                value: sum(row["baseline_confidence_class"] == value for row in group)
                for value in sorted({row["baseline_confidence_class"] for row in group})
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definite-qualified", required=True)
    parser.add_argument("--ambiguous", required=True)
    parser.add_argument("--definite-output", required=True)
    parser.add_argument("--ambiguous-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--per-definite-condition", type=int, default=80)
    args = parser.parse_args()
    passed = [row for row in load(args.definite_qualified) if row["qwen_correct_choice_pass"]]
    selected = matched_select(
        passed,
        seed=args.seed,
        per_condition=args.per_definite_condition,
    )
    definite = [format_definite(row, args.seed) for row in selected]
    ambiguous = [format_ambiguous(row, args.seed) for row in load(args.ambiguous)]
    if len({row["source_id"] for row in definite}) != len(definite):
        raise ValueError("A definite source question was assigned more than once")
    if sum(row["condition"] == "definite_correct" for row in definite) != args.per_definite_condition:
        raise ValueError(f"Expected exactly {args.per_definite_condition} definite-correct items")
    if sum(row["condition"] == "definite_false" for row in definite) != args.per_definite_condition:
        raise ValueError(f"Expected exactly {args.per_definite_condition} definite-false items")
    write(args.definite_output, definite)
    write(args.ambiguous_output, ambiguous)
    report = {
        "format_version": FORMAT_VERSION,
        "qwen_qualified_available": len(passed),
        "definite_selected": len(definite),
        "ambiguous_currently_available": len(ambiguous),
        "assignment_seed": args.seed,
        "balance": summarize(definite),
        "definite_output": args.definite_output,
        "ambiguous_output": args.ambiguous_output,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
