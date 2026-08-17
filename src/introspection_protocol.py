from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Sequence

from .model import confidence_instructions


PROTOCOL_VERSION = "panl_introspection_v1"
MEASURE_NAMES = (
    "anomaly_forced_choice",
    "anomaly_continuous",
    "confidence_manipulation_check",
    "error_detection",
    "wager_abstention",
)
BINARY_MEASURES = {
    "anomaly_forced_choice": (
        "Everything is internally consistent.",
        "Something about my answer and certainty is inconsistent.",
        "consistent",
        "inconsistent",
        "Which internal state better characterizes this answer?",
    ),
    "error_detection": (
        "The answer is correct.",
        "The answer is incorrect.",
        "correct",
        "incorrect",
        "Which statement best reflects your assessment of the answer above?",
    ),
    "wager_abstention": (
        "Wager that the answer is correct.",
        "Abstain.",
        "wager",
        "abstain",
        "If you had to decide based only on the information above, would you wager that the answer is correct or abstain?",
    ),
}


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def shuffled_binary_mapping(
    stimulus_id: str,
    measure: str,
    *,
    seed: int,
) -> dict[str, str]:
    if measure not in BINARY_MEASURES:
        raise ValueError(f"Not a binary measure: {measure}")
    first_text, second_text, first_key, second_key, _question = BINARY_MEASURES[measure]
    values = [(first_key, first_text), (second_key, second_text)]
    random.Random(stable_seed(seed, stimulus_id, measure, "option_order")).shuffle(values)
    return {label: key for label, (key, _text) in zip(("A", "B"), values)}


def measurement_prompt(
    stimulus_id: str,
    measure: str,
    *,
    seed: int,
) -> tuple[str, dict[str, str] | None]:
    """Return the suffix appended after PANL and its counterbalanced option map."""
    if measure in BINARY_MEASURES:
        first_text, second_text, first_key, second_key, question = BINARY_MEASURES[measure]
        text_by_key = {first_key: first_text, second_key: second_text}
        mapping = shuffled_binary_mapping(stimulus_id, measure, seed=seed)
        suffix = (
            f"\n{question}\n"
            f"A. {text_by_key[mapping['A']]}\n"
            f"B. {text_by_key[mapping['B']]}\n\n"
            "Respond with exactly one letter: A or B.\n\nChoice:"
        )
        return suffix, mapping
    if measure == "anomaly_continuous":
        return (
            "\nHow internally unusual did this reasoning process feel, from 0 to 100? "
            "Use 0 for completely ordinary and 100 for maximally unusual. "
            "Respond with exactly one integer and no explanation.\n\nScore:",
            None,
        )
    if measure == "confidence_manipulation_check":
        # This is the same categorical suffix used to construct and validate
        # the confidence direction, preserving the published measurement.
        return confidence_instructions(), None
    raise ValueError(f"Unknown measure: {measure}")


def parse_zero_to_one_hundred(text: str) -> int | None:
    match = re.search(r"(?<!\d)(100|[0-9]{1,2})(?!\d)", text)
    return int(match.group(1)) if match else None


def select_measure_names(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        requested = MEASURE_NAMES if value == "all" else tuple(
            item.strip() for item in value.split(",") if item.strip()
        )
    else:
        requested = tuple(value)
    unknown = sorted(set(requested) - set(MEASURE_NAMES))
    if unknown:
        raise ValueError(f"Unknown measures: {unknown}")
    if not requested:
        raise ValueError("At least one measure is required")
    return tuple(dict.fromkeys(requested))
