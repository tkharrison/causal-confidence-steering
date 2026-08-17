#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import argparse
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.semantic_grading import DEFAULT_MODEL, _post_openrouter


SYSTEM = """Generate multiple-choice distractors for a factual trivia experiment.
The supplied Qwen answer has already been semantically verified as correct. Produce exactly
three plausible but factually false alternatives. All three must have the same semantic type,
specificity, grammatical form, and approximately the same length as the correct answer. They
must not be aliases, spelling variants, subsets, supersets, or restatements of the correct answer.
Avoid jokes, obviously absurd choices, and answers from a different category. Return only the
requested JSON."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "trivia_distractors",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer_type": {"type": "string"},
                "distractors": {
                    "type": "array", "minItems": 3, "maxItems": 3,
                    "items": {"type": "string"},
                },
            },
            "required": ["answer_type", "distractors"],
            "additionalProperties": False,
        },
    },
}


def generate(row: dict, *, api_key: str, model: str, retries: int = 5) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({
                "question": row["question"],
                "verified_correct_answer": row["qwen_correct_answer"],
                "known_correct_aliases": row.get("aliases", []),
            }, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": 140,
        "response_format": SCHEMA,
        "provider": {"sort": "price", "require_parameters": True},
    }
    last = None
    for attempt in range(retries):
        try:
            response = _post_openrouter(payload, api_key=api_key, timeout_seconds=60)
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            if not isinstance(parsed.get("distractors"), list) or len(parsed["distractors"]) != 3:
                raise ValueError("Expected exactly three distractors")
            return {
                **row,
                "answer_type": parsed["answer_type"],
                "distractors": [" ".join(str(value).split()) for value in parsed["distractors"]],
                "distractor_generation_model": response.get("model", model),
                "distractor_generation_usage": response.get("usage"),
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            last = RuntimeError(f"HTTP {exc.code}: {body}")
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError,
                ValueError, json.JSONDecodeError) as exc:
            last = exc
        if attempt + 1 < retries:
            time.sleep(min(8, 0.5 * 2**attempt) + random.random() * 0.2)
    raise RuntimeError(f"Failed to generate distractors for {row['id']}: {last}")


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
    output = Path(args.output)
    completed = {}
    if output.exists():
        completed = {row["id"]: row for row in
                     (json.loads(line) for line in output.read_text().splitlines() if line.strip())}
    pending = [row for row in rows if row["id"] not in completed]
    lock = threading.Lock()

    def append(result: dict) -> None:
        with lock, output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        iterator = iter(pending)
        futures: dict[Future, str] = {}
        def submit() -> None:
            try:
                row = next(iterator)
            except StopIteration:
                return
            futures[executor.submit(generate, row, api_key=key, model=args.model)] = row["id"]
        for _ in range(min(args.concurrency, len(pending))):
            submit()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                append(future.result())
                submit()
    results = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    selected = {row["id"] for row in rows}
    results = [row for row in results if row["id"] in selected]
    cost = sum(float(row.get("distractor_generation_usage", {}).get("cost", 0) or 0) for row in results)
    print(json.dumps({"selected": len(rows), "completed": len(results), "cost": cost,
                      "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
