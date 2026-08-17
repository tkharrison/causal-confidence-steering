from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any
import urllib.error

from .semantic_grading import DEFAULT_MODEL, OPENROUTER_URL, _post_openrouter
from .ambigqa_dataset import normalize_text


SYSTEM_PROMPT = """You are validating stimuli for a mechanistic-interpretability experiment.
An underspecified question, a neutral missing-qualifier note, and four disambiguated questions
with their answers are supplied. Accept an item only when the original wording naturally leaves
out a qualifier, each of the four readings is genuinely plausible, and the truth of the
answer depends on that missing qualifier. Reject broad list questions where several options
are simultaneously true, items with an obvious/default reading, mixed answer types, weak or
contrived readings, bad annotations, or a question for which selecting one option can still
be confidently correct without resolving the ambiguity. All four answers must be the same
semantic kind: reject mixtures such as a job title plus named people, a category plus examples,
or a date plus an event. Return only the requested JSON."""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ambigqa_stimulus_validation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "suitable": {"type": "boolean"},
                "ambiguity_type": {
                    "type": "string",
                    "enum": ["entity", "time", "location", "role", "version", "scope", "other"],
                },
                "readings_all_plausible": {"type": "boolean"},
                "answers_context_dependent": {"type": "boolean"},
                "no_obvious_default": {"type": "boolean"},
                "options_same_semantic_type": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": [
                "suitable", "ambiguity_type", "readings_all_plausible",
                "answers_context_dependent", "no_obvious_default", "reason",
                "options_same_semantic_type",
            ],
            "additionalProperties": False,
        },
    },
}


def _payload(row: dict[str, Any], model: str) -> dict[str, Any]:
    item = {
        "original_question": row["question"],
        "context_note": row.get("context_note"),
        "interpretations": row["interpretations"],
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(item, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 180,
        "response_format": RESPONSE_SCHEMA,
        "provider": {"sort": "price", "require_parameters": True},
    }


def _mechanical_rejection(row: dict[str, Any]) -> str | None:
    interpretations = row.get("interpretations", [])
    answers = [str(pair.get("answer", "")) for pair in interpretations]
    normalized = [normalize_text(answer) for answer in answers]
    if len(interpretations) != 4 or len(answers) != 4:
        return "The item does not contain exactly four interpretations."
    if any(not answer for answer in normalized) or len(set(normalized)) != 4:
        return "The four answers are not nonempty and semantically distinct strings."
    note = row.get("context_note")
    if note is not None and not str(note).startswith("No information is provided about"):
        return "The context note does not use the required neutral format."
    return None


def _judge(row: dict[str, Any], *, api_key: str, model: str, retries: int = 5) -> dict[str, Any]:
    mechanical_rejection = _mechanical_rejection(row)
    if mechanical_rejection:
        return {
            "id": row["id"], "suitable": False, "judge_suitable": False,
            "ambiguity_type": "other", "readings_all_plausible": False,
            "answers_context_dependent": False, "no_obvious_default": False,
            "options_same_semantic_type": False, "reason": mechanical_rejection,
            "judge_model": "local_mechanical_filter", "usage": {"cost": 0},
        }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = _post_openrouter(_payload(row, model), api_key=api_key, timeout_seconds=60)
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            if not isinstance(parsed, dict) or not isinstance(parsed.get("suitable"), bool):
                raise ValueError("Judge response lacks suitable boolean")
            required_bools = (
                "readings_all_plausible", "answers_context_dependent",
                "no_obvious_default", "options_same_semantic_type",
            )
            if not all(isinstance(parsed.get(field), bool) for field in required_bools):
                raise ValueError("Judge response lacks required validation booleans")
            parsed["judge_suitable"] = parsed["suitable"]
            parsed["suitable"] = parsed["suitable"] and all(
                parsed[field] for field in required_bools
            )
            return {
                "id": row["id"],
                **parsed,
                "judge_model": response.get("model", model),
                "usage": response.get("usage"),
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
            if attempt + 1 < retries:
                time.sleep(min(8, 0.5 * (2**attempt)) + random.random() * 0.2)
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError,
                TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(8, 0.5 * (2**attempt)) + random.random() * 0.2)
    return {
        "id": row["id"], "suitable": False, "judge_suitable": False,
        "ambiguity_type": row.get("ambiguity_type", "other"),
        "readings_all_plausible": False, "answers_context_dependent": False,
        "no_obvious_default": False, "options_same_semantic_type": False,
        "reason": f"Validation request failed and was conservatively rejected: {last_error}",
        "judge_model": model, "usage": {"cost": 0}, "judge_error": True,
    }


def prefilter_candidates(
    input_file: str | Path,
    output_file: str | Path,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
    concurrency: int = 12,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(input_file).read_text().splitlines() if line.strip()]
    if split != "all":
        rows = [row for row in rows if row["source_split"] == split]
    random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if output_file.exists():
        for line in output_file.read_text().splitlines():
            if line.strip():
                result = json.loads(line)
                existing[str(result["id"])] = result
    pending = [row for row in rows if row["id"] not in existing]
    lock = threading.Lock()

    def append(result: dict[str, Any]) -> None:
        with lock, output_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        iterator = iter(pending)
        futures: dict[Future[dict[str, Any]], str] = {}

        def submit() -> bool:
            try:
                row = next(iterator)
            except StopIteration:
                return False
            future = executor.submit(_judge, row, api_key=api_key, model=model)
            futures[future] = row["id"]
            return True

        for _ in range(min(concurrency, len(pending))):
            submit()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                append(future.result())
                submit()

    results = {
        row["id"]: row
        for row in (json.loads(line) for line in output_file.read_text().splitlines() if line.strip())
    }
    selected_results = [results[row["id"]] for row in rows]
    cost = sum(
        float(result.get("usage", {}).get("cost", 0) or 0)
        for result in selected_results
    )
    return {
        "input_selected": len(rows),
        "already_completed": len(rows) - len(pending),
        "completed_now": len(pending),
        "suitable": sum(result["suitable"] for result in selected_results),
        "estimated_api_cost": cost,
        "output_file": str(output_file),
        "judge_model": model,
    }


def write_filtered_candidates(
    candidates_file: str | Path,
    judgments_file: str | Path,
    output_file: str | Path,
) -> int:
    candidates = {
        row["id"]: row
        for row in (json.loads(line) for line in Path(candidates_file).read_text().splitlines() if line.strip())
    }
    judgments = [
        json.loads(line) for line in Path(judgments_file).read_text().splitlines() if line.strip()
    ]
    accepted = [candidates[row["id"]] | {"prefilter_judgment": row} for row in judgments if row["suitable"]]
    accepted.sort(key=lambda row: row["id"])
    Path(output_file).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted),
        encoding="utf-8",
    )
    return len(accepted)
