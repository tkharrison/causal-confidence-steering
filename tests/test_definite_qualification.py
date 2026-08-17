from src.definite_qualification import _options, _prompt


def test_definite_options_are_deterministic_and_complete() -> None:
    row = {
        "id": "x", "qwen_correct_answer": "Jane Austen",
        "distractors": ["George Eliot", "Mary Shelley", "Charlotte Bronte"],
    }
    first, label = _options(row, 42)
    second, second_label = _options(row, 42)
    assert first == second
    assert label == second_label
    assert set(first.values()) == {row["qwen_correct_answer"], *row["distractors"]}
    assert first[label] == row["qwen_correct_answer"]
    assert _prompt("Who wrote it?", first).count("Choice: X") == 1
