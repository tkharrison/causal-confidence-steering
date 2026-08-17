#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--notes-from", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    notes = {row["id"]: row for row in load(args.notes_from)}
    output = []
    for candidate in load(args.candidates):
        prior = notes[candidate["id"]]
        output.append({
            **candidate,
            "original_question": candidate["question"],
            "context_note": prior["context_note"],
            "display_question": f"{candidate['question']}\n\n{prior['context_note']}",
            "embellishment_model": prior.get("embellishment_model"),
            "embellishment_usage": prior.get("embellishment_usage"),
            "embellishment_reused_from": args.notes_from,
        })
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"output": args.output, "count": len(output)}, indent=2))


if __name__ == "__main__":
    main()
