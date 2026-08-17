from __future__ import annotations

from src.introspection_protocol import (
    CONTROL_MEASURE_NAMES,
    MEASURE_NAMES,
    SUPPORTED_MEASURE_NAMES,
    measurement_prompt,
    parse_zero_to_one_hundred,
    select_condition_names,
    shuffled_binary_mapping,
)


def test_all_measure_prompts_are_isolated_suffixes() -> None:
    prompts = {
        name: measurement_prompt("item-1", name, seed=42)[0]
        for name in MEASURE_NAMES
    }
    assert len(prompts) == 5
    assert prompts["anomaly_forced_choice"].endswith("Choice:")
    assert prompts["error_detection"].endswith("Choice:")
    assert prompts["wager_abstention"].endswith("Choice:")
    assert prompts["anomaly_continuous"].endswith("Score:")
    assert prompts["confidence_manipulation_check"].endswith("**Confidence**:")


def test_binary_order_is_deterministic_and_counterbalanced() -> None:
    first = shuffled_binary_mapping("item-1", "anomaly_forced_choice", seed=42)
    second = shuffled_binary_mapping("item-1", "anomaly_forced_choice", seed=42)
    assert first == second
    assert set(first) == {"A", "B"}
    assert set(first.values()) == {"consistent", "inconsistent"}
    mappings = [
        shuffled_binary_mapping(f"item-{index}", "anomaly_forced_choice", seed=42)["A"]
        for index in range(1000)
    ]
    assert 400 < mappings.count("consistent") < 600


def test_polarity_controls_are_a_matched_yes_no_pair() -> None:
    assert CONTROL_MEASURE_NAMES == (
        "inconsistency_yes_no",
        "consistency_yes_no",
    )
    prompts = {
        name: measurement_prompt("item-1", name, seed=42)
        for name in CONTROL_MEASURE_NAMES
    }
    assert set(SUPPORTED_MEASURE_NAMES) == set(MEASURE_NAMES) | set(CONTROL_MEASURE_NAMES)
    assert "inconsistent?" in prompts["inconsistency_yes_no"][0]
    assert "consistent with each other?" in prompts["consistency_yes_no"][0]
    for _suffix, mapping in prompts.values():
        assert mapping is not None
        assert set(mapping.values()) == {"yes", "no"}


def test_condition_selection_is_explicit_and_ordered() -> None:
    assert select_condition_names("definite_correct") == ("definite_correct",)
    assert select_condition_names("definite_false,definite_correct") == (
        "definite_false",
        "definite_correct",
    )


def test_continuous_parser_accepts_only_bounded_integer() -> None:
    assert parse_zero_to_one_hundred("75") == 75
    assert parse_zero_to_one_hundred("Score: 100") == 100
    assert parse_zero_to_one_hundred("0\n") == 0
    assert parse_zero_to_one_hundred("101") is None
    assert parse_zero_to_one_hundred("no score") is None


def test_single_use_hook_edits_panl_once() -> None:
    try:
        import torch
    except ImportError:
        return

    from src.introspection_experiment import single_use_residual_intervention

    class IdentityBlock(torch.nn.Module):
        def forward(self, hidden):
            return hidden

    class Inner:
        def __init__(self):
            self.layers = torch.nn.ModuleList([IdentityBlock()])

    class FakeModel:
        def __init__(self):
            self.model = Inner()

    model = FakeModel()
    hidden = torch.zeros((2, 5, 3))
    direction = torch.tensor([1.0, 2.0, 3.0])
    with single_use_residual_intervention(
        model,
        layer_id=0,
        token_indices=[2, 3],
        direction=direction,
        alpha=2.0,
    ) as tracker:
        first = model.model.layers[0](hidden)
        second = model.model.layers[0](hidden)
    assert tracker["applications"] == 1
    assert torch.equal(first[0, 2], 2 * direction)
    assert torch.equal(first[1, 3], 2 * direction)
    assert torch.equal(second, hidden)
