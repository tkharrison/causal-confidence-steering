#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path: str, rows: list[dict]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def diverse_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Round-robin across categories, randomizing within each category."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("category") or "uncategorized"].append(row)
    for group in groups.values():
        rng.shuffle(group)
    category_order = sorted(groups)
    rng.shuffle(category_order)
    selected: list[dict] = []
    while len(selected) < count:
        progressed = False
        for category in category_order:
            if groups[category]:
                selected.append(groups[category].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"Need {count} rows but ran out after {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--new-qualified", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    core = load(args.core)
    passing = [row for row in load(args.new_qualified) if row.get("recognition_pass")]
    needed = args.target - len(core)
    if needed < 0:
        raise ValueError(f"Core already has {len(core)} rows, above target {args.target}")
    selected_new = diverse_sample(passing, needed, args.seed)
    for row in selected_new:
        row["stimulus_source_family"] = "user_generated_qualified"
        row["final_quality_tier"] = "core"
        row["pool_selection_seed"] = args.seed
    combined = core + selected_new
    if len(combined) != args.target or len({row["id"] for row in combined}) != args.target:
        raise ValueError("Final pool size or ID uniqueness check failed")
    rng = random.Random(args.seed)
    rng.shuffle(combined)
    write(args.output, combined)

    selected_ids = {row["id"] for row in selected_new}
    report = {
        "target": args.target,
        "existing_core": len(core),
        "new_qwen_3_of_3_available": len(passing),
        "new_selected": len(selected_new),
        "new_held_in_reserve": len(passing) - len(selected_new),
        "selection_seed": args.seed,
        "selected_new_ids": sorted(selected_ids),
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "selected_new_ids"}, indent=2))


if __name__ == "__main__":
    main()
