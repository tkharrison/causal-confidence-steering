from __future__ import annotations

import csv
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.dataset import exact_match


DEFAULT_MODEL = "openai/gpt-4o-mini-2024-07-18"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are grading short answers to TriviaQA questions.
The accepted reference aliases are known-correct examples, but the list can be incomplete.
Independently use your factual knowledge and the question to judge the proposed answer; it
may be correct even when its wording is absent from the aliases (for example, a real name
versus a stage name). Accept harmless formatting differences, reordered names, minor
spelling variants, abbreviations, and extra specificity that does not change the answer.
Reject a different entity, a materially wrong date/number, a partial answer when the
question requires multiple parts, contradictions, or a list containing a wrong answer.
Judge correctness rather than string overlap. Do not infer or discuss answerer confidence.
Return only the requested JSON object. Keep the reason under 25 words."""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "triviaqa_correctness",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "correct": {"type": "boolean"},
                "matched_alias": {"type": ["string", "null"]},
                "reason": {"type": "string"},
            },
            "required": ["correct", "matched_alias", "reason"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class GradeConfig:
    model: str = DEFAULT_MODEL
    concurrency: int = 12
    max_retries: int = 5
    timeout_seconds: float = 60.0


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"Missing id at {path}:{line_number}")
            if row_id in seen:
                raise ValueError(f"Duplicate id {row_id!r} in {path}")
            for field in ("question", "answer_text", "aliases"):
                if field not in row:
                    raise ValueError(f"Missing {field!r} for id {row_id!r}")
            if not isinstance(row["aliases"], list) or not row["aliases"]:
                raise ValueError(f"aliases must be a non-empty list for id {row_id!r}")
            seen.add(row_id)
            rows.append(row)
    return rows


def load_existing_grades(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    grades: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"Missing id at {path}:{line_number}")
            if row_id in grades:
                raise ValueError(f"Duplicate grade id {row_id!r} in {path}")
            grades[row_id] = row
    return grades


def exact_grade(row: dict[str, Any]) -> dict[str, Any] | None:
    if not exact_match(str(row["answer_text"]), [str(alias) for alias in row["aliases"]]):
        return None
    return {
        "id": str(row["id"]),
        "semantic_correct": True,
        "grade_source": "normalized_exact_match",
        "judge_model": None,
        "matched_alias": next(
            alias
            for alias in row["aliases"]
            if exact_match(str(row["answer_text"]), [str(alias)])
        ),
        "judge_reason": "Normalized answer exactly matches an accepted alias.",
        "usage": None,
    }


def _request_payload(row: dict[str, Any], model: str) -> dict[str, Any]:
    item = {
        "question": str(row["question"]),
        "proposed_answer": str(row["answer_text"]),
        "accepted_aliases": [str(alias) for alias in row["aliases"]],
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(item, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 120,
        "response_format": RESPONSE_SCHEMA,
        "provider": {"sort": "price", "require_parameters": True},
    }


def _parse_judgment(response: dict[str, Any]) -> dict[str, Any]:
    try:
        choice = response["choices"][0]
        message = choice["message"]
        content = message["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed OpenRouter response: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("correct"), bool):
        raise ValueError("Judge output lacks a boolean 'correct' field")
    if parsed.get("matched_alias") is not None and not isinstance(parsed["matched_alias"], str):
        raise ValueError("Judge output has invalid 'matched_alias'")
    if not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip():
        raise ValueError("Judge output lacks a non-empty 'reason'")
    return parsed


def _post_openrouter(
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/qwen-panl-confidence",
            "X-OpenRouter-Title": "Qwen PANL TriviaQA correctness grader",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def judge_row(
    row: dict[str, Any],
    *,
    api_key: str,
    config: GradeConfig,
) -> dict[str, Any]:
    payload = _request_payload(row, config.model)
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            response = _post_openrouter(
                payload,
                api_key=api_key,
                timeout_seconds=config.timeout_seconds,
            )
            judgment = _parse_judgment(response)
            usage = response.get("usage")
            return {
                "id": str(row["id"]),
                "semantic_correct": judgment["correct"],
                "grade_source": "llm_judge",
                "judge_model": response.get("model", config.model),
                "matched_alias": judgment.get("matched_alias"),
                "judge_reason": judgment["reason"].strip(),
                "usage": usage if isinstance(usage, dict) else None,
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < config.max_retries:
            delay = min(16.0, 0.75 * (2 ** (attempt - 1))) + random.random() * 0.25
            time.sleep(delay)
    raise RuntimeError(f"Failed to grade {row['id']} after {config.max_retries} attempts: {last_error}")


def _write_grade(handle: Any, grade: dict[str, Any], lock: threading.Lock) -> None:
    encoded = json.dumps(grade, ensure_ascii=False, sort_keys=True)
    with lock:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def grade_records(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    api_key: str | None,
    config: GradeConfig = GradeConfig(),
    progress_every: int = 10,
) -> dict[str, Any]:
    if config.concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_grades(output_path)
    input_ids = {str(row["id"]) for row in records}
    unknown = set(existing) - input_ids
    if unknown:
        raise ValueError(f"Output contains ids absent from input: {sorted(unknown)[:5]}")

    pending = [row for row in records if str(row["id"]) not in existing]
    exact = [(row, exact_grade(row)) for row in pending]
    exact_grades = [grade for _, grade in exact if grade is not None]
    api_rows = [row for row, grade in exact if grade is None]
    if api_rows and not api_key:
        raise RuntimeError(
            f"OPENROUTER_API_KEY is not set; {len(api_rows)} semantic judgments remain"
        )

    lock = threading.Lock()
    completed_now = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for grade in exact_grades:
            assert grade is not None
            _write_grade(handle, grade, lock)
            completed_now += 1

        if api_rows:
            with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
                iterator = iter(api_rows)
                futures: dict[Future[dict[str, Any]], str] = {}

                def submit_next() -> bool:
                    try:
                        row = next(iterator)
                    except StopIteration:
                        return False
                    future = executor.submit(judge_row, row, api_key=api_key or "", config=config)
                    futures[future] = str(row["id"])
                    return True

                for _ in range(min(config.concurrency, len(api_rows))):
                    submit_next()
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        row_id = futures.pop(future)
                        try:
                            grade = future.result()
                        except Exception:
                            for other in futures:
                                other.cancel()
                            raise RuntimeError(f"Grading stopped at id {row_id}; completed rows are resumable")
                        _write_grade(handle, grade, lock)
                        completed_now += 1
                        total_complete = len(existing) + completed_now
                        if progress_every and total_complete % progress_every == 0:
                            print(f"graded {total_complete}/{len(records)}", flush=True)
                        submit_next()

    grades = load_existing_grades(output_path)
    return summarize_grades(records, grades, output_path=output_path)


def summarize_grades(
    records: list[dict[str, Any]],
    grades: dict[str, dict[str, Any]],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    total_cost = 0.0
    cost_known = True
    for grade in grades.values():
        if grade.get("grade_source") != "llm_judge":
            continue
        usage = grade.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
            total_cost += float(usage["cost"])
        else:
            cost_known = False
    correct = sum(grade.get("semantic_correct") is True for grade in grades.values())
    return {
        "input_records": len(records),
        "graded_records": len(grades),
        "remaining_records": len(records) - len(grades),
        "correct": correct,
        "incorrect": len(grades) - correct,
        "exact_match_grades": sum(
            grade.get("grade_source") == "normalized_exact_match" for grade in grades.values()
        ),
        "llm_judge_grades": sum(
            grade.get("grade_source") == "llm_judge" for grade in grades.values()
        ),
        "reported_api_cost_usd": round(total_cost, 8) if cost_known else None,
        "output_path": str(output_path) if output_path is not None else None,
    }


def write_review_csv(
    records: Iterable[dict[str, Any]],
    grades: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> None:
    fields = [
        "id",
        "question",
        "answer_text",
        "aliases",
        "confidence_class",
        "correct_exact_match",
        "semantic_correct",
        "grade_source",
        "judge_model",
        "matched_alias",
        "judge_reason",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            grade = grades.get(str(row["id"]), {})
            writer.writerow({
                "id": row["id"],
                "question": row["question"],
                "answer_text": row["answer_text"],
                "aliases": json.dumps(row["aliases"], ensure_ascii=False),
                "confidence_class": row.get("confidence_class"),
                "correct_exact_match": row.get("correct_exact_match"),
                "semantic_correct": grade.get("semantic_correct"),
                "grade_source": grade.get("grade_source"),
                "judge_model": grade.get("judge_model"),
                "matched_alias": grade.get("matched_alias"),
                "judge_reason": grade.get("judge_reason"),
            })
