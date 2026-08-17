#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SYNTHETIC_EXCLUSIONS = {
    "synthetic_ambiguity_f0b4b47e130b2c01": "The singular primary-ingredient answers omit rice and are not defensible.",
    "synthetic_ambiguity_d1e5ebea322a741f": "Near-duplicate of another country-to-language stimulus.",
    "synthetic_ambiguity_f7fd823c55bb510e": "The claimed first computers for Windows versions are not reliably defined.",
    "synthetic_ambiguity_24dd22563ed6fab7": "Art movement still does not identify a unique work or artist.",
    "synthetic_ambiguity_f3bb86d76cc424f2": "The interpretations restate research categories as tautological functions.",
    "synthetic_ambiguity_22014d28f133cd99": "The Republic of India language answer is materially oversimplified.",
    "synthetic_ambiguity_0b912233a4f43988": "Most influential is subjective rather than qualifier-resolved.",
    "synthetic_ambiguity_86f4bab81782183d": "Near-duplicate of another James Bond adaptation stimulus.",
}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path: str, rows: list[dict]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--core-output", required=True)
    parser.add_argument("--expanded-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    def passed(row: dict) -> bool:
        return bool(row.get("recognition_pass", row.get("recognition_pass_5_of_5")))

    real = [row for row in load(args.real) if passed(row)]
    synthetic = [row for row in load(args.synthetic) if passed(row)]
    for row in real:
        row["stimulus_source_family"] = "real_ambigqa"
        row["final_quality_tier"] = "core"
    for row in synthetic:
        row["stimulus_source_family"] = "synthetic"
        row["final_quality_tier"] = (
            "expanded_only" if row["id"] in SYNTHETIC_EXCLUSIONS else "core"
        )
    core_synthetic = [row for row in synthetic if row["id"] not in SYNTHETIC_EXCLUSIONS]
    core = sorted(real + core_synthetic, key=lambda row: row["id"])
    expanded = sorted(real + synthetic, key=lambda row: row["id"])
    write(args.core_output, core)
    write(args.expanded_output, expanded)
    report = {
        "real_core": len(real),
        "synthetic_qwen_3_of_3": len(synthetic),
        "synthetic_core_after_manual_audit": len(core_synthetic),
        "core_total": len(core),
        "expanded_total": len(expanded),
        "synthetic_exclusions": SYNTHETIC_EXCLUSIONS,
        "core_output": args.core_output,
        "expanded_output": args.expanded_output,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
