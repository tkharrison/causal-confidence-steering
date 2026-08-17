#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.semantic_grading import DEFAULT_MODEL, _post_openrouter


SYSTEM = """Validate multiple-choice factual-trivia stimuli. The candidate includes a question,
a previously adjudicated Qwen answer, known correct aliases, and three generated distractors.
Accept only if the question has one ordinary intended factual answer, the Qwen answer is defensible,
every distractor is factually false under that intended reading but reasonably tempting, and all four
answers share the same semantic type and specificity. Reject time-sensitive, genuinely ambiguous,
malformed, trick, subjective, or multiple-answer questions. Return only the requested JSON."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "trivia_distractor_validation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "suitable": {"type": "boolean"},
                "question_definite": {"type": "boolean"},
                "correct_answer_supported": {"type": "boolean"},
                "distractors_all_false": {"type": "boolean"},
                "distractors_all_plausible": {"type": "boolean"},
                "same_semantic_type": {"type": "boolean"},
                "most_plausible_index": {"type": "integer", "minimum": 0, "maximum": 2},
                "reason": {"type": "string"},
            },
            "required": ["suitable", "question_definite", "correct_answer_supported",
                         "distractors_all_false", "distractors_all_plausible",
                         "same_semantic_type", "most_plausible_index", "reason"],
            "additionalProperties": False,
        },
    },
}


def normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def mechanical_reason(row: dict) -> str | None:
    distractors = row.get("distractors", [])
    if len(distractors) != 3:
        return "Does not contain exactly three distractors."
    normalized = [normalize(str(value)) for value in distractors]
    if any(not value for value in normalized) or len(set(normalized)) != 3:
        return "Distractors are empty or duplicated."
    correct = {normalize(row["qwen_correct_answer"])} | {
        normalize(str(value)) for value in row.get("aliases", [])
    }
    if set(normalized) & correct:
        return "A distractor duplicates the verified answer or a known alias."
    if any(len(str(value)) > 80 or len(str(value).split()) > 10 for value in distractors):
        return "A distractor is not concise."
    return None


def validate(row: dict, *, api_key: str, model: str, retries: int = 5) -> dict:
    reason = mechanical_reason(row)
    if reason:
        return {**row, "distractor_validation_pass": False, "validation_source": "hard_filter",
                "validation_reason": reason, "distractor_validation_usage": {"cost": 0}}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({
                "question": row["question"], "verified_qwen_answer": row["qwen_correct_answer"],
                "known_correct_aliases": row.get("aliases", []), "distractors": row["distractors"],
            }, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 220,
        "response_format": SCHEMA,
        "provider": {"sort": "price", "require_parameters": True},
    }
    last = None
    for attempt in range(retries):
        try:
            response = _post_openrouter(payload, api_key=api_key, timeout_seconds=60)
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            required = ["question_definite", "correct_answer_supported", "distractors_all_false",
                        "distractors_all_plausible", "same_semantic_type"]
            passed = bool(parsed["suitable"]) and all(bool(parsed[field]) for field in required)
            return {
                **row, "distractor_validation_pass": passed, "validation_source": "llm_judge",
                "validation_model": response.get("model", model),
                "validation_reason": parsed["reason"],
                "most_plausible_distractor_index": int(parsed["most_plausible_index"]),
                "validation_dimensions": {field: bool(parsed[field]) for field in required},
                "distractor_validation_usage": response.get("usage"),
            }
        except urllib.error.HTTPError as exc:
            last = RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}")
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError,
                ValueError, json.JSONDecodeError) as exc:
            last = exc
        if attempt + 1 < retries:
            time.sleep(min(8, 0.5 * 2**attempt) + random.random() * 0.2)
    return {**row, "distractor_validation_pass": False, "validation_source": "judge_error",
            "validation_reason": str(last), "distractor_validation_usage": {"cost": 0}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filtered-output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    output = Path(args.output)
    existing = {row["id"]: row for row in
                (json.loads(line) for line in output.read_text().splitlines() if line.strip())} \
                if output.exists() else {}
    pending = [row for row in rows if row["id"] not in existing]
    lock = threading.Lock()
    def append(result: dict) -> None:
        with lock, output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        iterator = iter(pending); futures: dict[Future, str] = {}
        def submit() -> None:
            try: row = next(iterator)
            except StopIteration: return
            futures[executor.submit(validate, row, api_key=key, model=args.model)] = row["id"]
        for _ in range(min(args.concurrency, len(pending))): submit()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future); append(future.result()); submit()
    results_by_id = {row["id"]: row for row in
                     (json.loads(line) for line in output.read_text().splitlines() if line.strip())}
    results = [results_by_id[row["id"]] for row in rows]
    passed = [row for row in results if row["distractor_validation_pass"]]
    Path(args.filtered_output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in passed), encoding="utf-8"
    )
    cost = sum(float(row.get("distractor_validation_usage", {}).get("cost", 0) or 0) for row in results)
    print(json.dumps({"input": len(rows), "hard_rejected": sum(row["validation_source"] == "hard_filter" for row in results),
                      "passed": len(passed), "cost": cost, "filtered_output": args.filtered_output}, indent=2))


if __name__ == "__main__":
    main()
