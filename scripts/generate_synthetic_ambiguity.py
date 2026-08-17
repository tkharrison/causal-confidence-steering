#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ambigqa_dataset import normalize_text
from src.semantic_grading import DEFAULT_MODEL, _post_openrouter


SYSTEM = """Create high-quality multiple-choice stimuli for an epistemic-uncertainty experiment.
Each item must contain a naturally underspecified question, one neutral sentence explicitly
naming the missing qualifier, and exactly four distinct interpretations. Each interpretation
must use a different value of the same missing qualifier and have one concise answer. All four
answers must be the same semantic type and each must be uniquely correct under its interpretation.
There must be no obvious or culturally dominant default reading. Do not make broad list questions,
do not make several options correct under one interpretation, and do not copy the example.
Avoid subjective superlatives and vague prompts containing main, common, popular, known for,
mentioned, referenced, primary ingredient, primary purpose, or typical. Prefer concrete structures
such as actors across adaptations, officeholders across periods, release dates across countries,
locations across seasons, or answers across explicitly named editions. Every disambiguated question
must literally state its qualifier value. Use stable factual topics rather than current events.
The context note must begin exactly with
'No information is provided about'. Return only the requested JSON."""


INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "qualifier_value": {"type": "string"},
        "disambiguated_question": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["qualifier_value", "disambiguated_question", "answer"],
    "additionalProperties": False,
}


def schema(batch_size: int) -> dict:
    item = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "missing_qualifier": {"type": "string"},
            "context_note": {"type": "string"},
            "ambiguity_type": {
                "type": "string",
                "enum": ["entity", "time", "location", "role", "version", "scope", "other"],
            },
            "interpretations": {
                "type": "array", "items": INTERPRETATION_SCHEMA,
                "minItems": 4, "maxItems": 4,
            },
        },
        "required": ["question", "missing_qualifier", "context_note", "ambiguity_type", "interpretations"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "synthetic_ambiguity_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array", "items": item,
                        "minItems": batch_size, "maxItems": batch_size,
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }


def example(row: dict) -> dict:
    return {
        "question": row["question"],
        "context_note": row["context_note"],
        "interpretations": [{
            "disambiguated_question": pair["interpretation"],
            "answer": pair["answer"],
        } for pair in row["interpretations"]],
    }


def generate_batch(index: int, *, seeds: list[dict], api_key: str, model: str,
                   batch_size: int, seed: int) -> dict:
    template = seeds[index % len(seeds)]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({
                "instruction": (
                    f"Generate {batch_size} new, topically diverse analogues of the single good "
                    "example. Preserve its missing-qualifier architecture, not its topic or wording."
                ),
                "good_example": example(template),
                "batch_index": index,
            }, ensure_ascii=False)},
        ],
        "temperature": 0.6,
        "max_tokens": 3000,
        "response_format": schema(batch_size),
        "provider": {"sort": "price", "require_parameters": True},
    }
    last = None
    for attempt in range(5):
        try:
            response = _post_openrouter(payload, api_key=api_key, timeout_seconds=120)
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            if len(parsed["items"]) != batch_size:
                raise ValueError("Incorrect generated batch size")
            return {"batch_index": index, "items": parsed["items"], "usage": response.get("usage"),
                    "model": response.get("model", model)}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, IndexError,
                TypeError, ValueError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < 4:
                time.sleep(min(8, 0.5 * (2 ** attempt)) + random.random() * 0.2)
    raise RuntimeError(f"Batch {index} failed: {last}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.count % args.batch_size:
        raise ValueError("--count must be divisible by --batch-size")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    seeds = [row for row in (json.loads(line) for line in Path(args.seeds).read_text().splitlines())
             if row.get("recognition_pass") or row.get("recognition_pass_5_of_5")]
    batches = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(generate_batch, index, seeds=seeds, api_key=api_key,
                                   model=args.model, batch_size=args.batch_size, seed=args.seed)
                   for index in range(args.count // args.batch_size)]
        for future in as_completed(futures):
            batches.append(future.result())
    batches.sort(key=lambda row: row["batch_index"])
    output = []
    seen = set()
    for batch in batches:
        for item_index, item in enumerate(batch["items"]):
            normalized = normalize_text(item["question"])
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            pairs = [{"interpretation": pair["disambiguated_question"], "answer": pair["answer"]}
                     for pair in item["interpretations"]]
            output.append({
                "id": f"synthetic_ambiguity_{digest}",
                "source": "synthetic_from_curated_ambigqa_templates",
                "source_split": "synthetic",
                "question_type": "ambiguous",
                "question": item["question"],
                "original_question": item["question"],
                "missing_qualifier": item["missing_qualifier"],
                "context_note": item["context_note"],
                "display_question": f"{item['question']}\n\n{item['context_note']}",
                "ambiguity_type": item["ambiguity_type"],
                "interpretations": pairs,
                "substantive_options": [pair["answer"] for pair in item["interpretations"]],
                "generation_batch": batch["batch_index"],
                "generation_model": batch["model"],
                "generation_usage": batch["usage"],
            })
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8"
    )
    cost = sum(float(batch.get("usage", {}).get("cost", 0) or 0) for batch in batches)
    print(json.dumps({"requested": args.count, "unique_written": len(output), "cost": cost,
                      "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
