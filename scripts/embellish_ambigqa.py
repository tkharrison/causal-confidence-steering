#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.semantic_grading import DEFAULT_MODEL, _post_openrouter


SYSTEM = """You create neutral context notes for controlled ambiguity experiments.
Given an underspecified question and four human-written disambiguations, write exactly one
short sentence beginning with 'No information is provided about'. Identify only the missing
qualifier that distinguishes the four readings. Do not use the words ambiguous, uncertainty,
answer, choice, determine, correct, or confidence. Do not favor or reveal any option."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ambiguity_context_note",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"context_note": {"type": "string"}},
            "required": ["context_note"],
            "additionalProperties": False,
        },
    },
}


def judge(row: dict, *, api_key: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({
                "question": row["question"],
                "disambiguated_interpretations": [pair["interpretation"] for pair in row["interpretations"]],
            }, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 80,
        "response_format": SCHEMA,
        "provider": {"sort": "price", "require_parameters": True},
    }
    last = None
    for attempt in range(5):
        try:
            response = _post_openrouter(payload, api_key=api_key, timeout_seconds=60)
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            note = " ".join(parsed["context_note"].split())
            if not note.startswith("No information is provided about"):
                raise ValueError(f"Invalid note prefix: {note}")
            return {
                **row,
                "original_question": row["question"],
                "context_note": note,
                "display_question": f"{row['question']}\n\n{note}",
                "embellishment_model": response.get("model", model),
                "embellishment_usage": response.get("usage"),
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, IndexError,
                TypeError, ValueError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < 4:
                time.sleep(min(8, 0.5 * (2**attempt)) + random.random() * 0.2)
    raise RuntimeError(f"Failed {row['id']}: {last}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(judge, row, api_key=key, model=args.model) for row in rows]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda row: row["id"])
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    cost = sum(float(row.get("embellishment_usage", {}).get("cost", 0) or 0) for row in results)
    print(json.dumps({"input": len(rows), "output": len(results), "cost": cost, "output_file": args.output}, indent=2))


if __name__ == "__main__":
    main()
