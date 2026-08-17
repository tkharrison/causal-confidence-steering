from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def build_direction(
    run_dir: str | Path,
    output_file: str | Path,
    *,
    grades_file: str | Path | None = None,
    correctness_field: str = "semantic_correct",
    low_class: str = "Unlikely",
    high_class: str = "Likely",
    correct_only: bool = True,
    n_per_class: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    run_dir = Path(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines() if line.strip()]
    grades: dict[str, dict[str, Any]] | None = None
    if grades_file is not None:
        grade_rows = [
            json.loads(line)
            for line in Path(grades_file).read_text().splitlines()
            if line.strip()
        ]
        grades = {}
        for grade in grade_rows:
            row_id = str(grade.get("id", ""))
            if not row_id or row_id in grades:
                raise ValueError(f"Missing or duplicate grade id: {row_id!r}")
            if not isinstance(grade.get(correctness_field), bool):
                raise ValueError(f"Grade {row_id!r} lacks boolean {correctness_field!r}")
            grades[row_id] = grade
        missing = [str(record["id"]) for record in records if str(record["id"]) not in grades]
        if missing:
            raise ValueError(f"Semantic grades missing for {len(missing)} records; first: {missing[:5]}")
    groups: dict[str, list[dict[str, Any]]] = {low_class: [], high_class: []}
    for record in records:
        label = record.get("confidence_class")
        is_correct = (
            grades[str(record["id"])][correctness_field]
            if grades is not None
            else record.get("correct_exact_match") is True
        )
        if label in groups and (not correct_only or is_correct):
            groups[label].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    if len(groups[low_class]) < n_per_class or len(groups[high_class]) < n_per_class:
        raise ValueError(
            f"Insufficient correct trials for requested n_per_class={n_per_class}: "
            f"{low_class}={len(groups[low_class])}, {high_class}={len(groups[high_class])}"
        )
    n = n_per_class
    low_records, high_records = groups[low_class][:n], groups[high_class][:n]
    selected_low_ids = [str(row["id"]) for row in low_records]
    selected_high_ids = [str(row["id"]) for row in high_records]
    low_files = [load_file(str(run_dir / row["activation_file"])) for row in low_records]
    high_files = [load_file(str(run_dir / row["activation_file"])) for row in high_records]
    first = low_files[0]
    directions: dict[str, Any] = {}
    for key in sorted(first):
        low = torch.stack([tensors[key].float() for tensors in low_files])
        high = torch.stack([tensors[key].float() for tensors in high_files])
        raw = high.mean(0) - low.mean(0)
        mean_residual_norm = torch.cat([low, high]).norm(dim=-1).mean()
        directions[key] = raw / raw.norm().clamp_min(1e-12) * (0.03 * mean_residual_norm)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(directions, str(output_file), metadata={
        "low_class": low_class,
        "high_class": high_class,
        "n_per_class": str(n),
        "correct_only": str(correct_only),
        "correctness_field": correctness_field if grades is not None else "correct_exact_match",
        "grades_file": str(grades_file) if grades_file is not None else "",
        "seed": str(seed),
        "selected_low_ids": json.dumps(selected_low_ids),
        "selected_high_ids": json.dumps(selected_high_ids),
        "scale": "3_percent_mean_residual_norm",
    })
    summary = {
        "output_file": str(output_file),
        "low_available": len(groups[low_class]),
        "high_available": len(groups[high_class]),
        "n_per_class": n,
        "correctness_field": correctness_field if grades is not None else "correct_exact_match",
        "grades_file": str(grades_file) if grades_file is not None else None,
        "seed": seed,
        "selected_low_ids": selected_low_ids,
        "selected_high_ids": selected_high_ids,
    }
    output_file.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
