from __future__ import annotations

from src.config import ExperimentConfig
from src.dataset import exact_match, normalize_answer
from src.model import CONFIDENCE_CLASSES, confidence_instructions
from src.interventions import select_held_out_records


def test_confidence_contract() -> None:
    assert len(CONFIDENCE_CLASSES) == 10
    assert [item.name for item in CONFIDENCE_CLASSES][3:7] == [
        "Unlikely", "Less than even", "Better than even", "Likely"
    ]
    assert abs(CONFIDENCE_CLASSES[3].midpoint - 0.35) < 1e-12
    assert abs(CONFIDENCE_CLASSES[6].midpoint - 0.65) < 1e-12
    assert confidence_instructions().endswith("**Confidence**:")


def test_config_round_trip() -> None:
    config = ExperimentConfig(limit=2, positions=("panl",))
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_triviaqa_normalization() -> None:
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert exact_match("Paris", ["Paris", "City of Paris"])


def test_held_out_selection_is_deterministic_and_excludes_training_ids() -> None:
    records = [{"id": f"q{index}"} for index in range(20)]
    first = select_held_out_records(records, excluded_ids={"q1", "q2"}, limit=5, seed=42)
    second = select_held_out_records(records, excluded_ids={"q1", "q2"}, limit=5, seed=42)
    assert first == second
    assert not ({row["id"] for row in first} & {"q1", "q2"})
    next_five = select_held_out_records(
        records, excluded_ids={"q1", "q2"}, limit=5, seed=42, offset=5
    )
    assert not ({row["id"] for row in first} & {row["id"] for row in next_five})
