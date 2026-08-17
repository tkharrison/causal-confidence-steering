from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Callable

from .ambigqa_dataset import CHOICE_LABELS, format_options, parse_choice


def _options(row: dict[str, Any], seed: int) -> tuple[dict[str, str], str]:
    values = [row["qwen_correct_answer"], *row["distractors"]]
    random.Random(f"{seed}:{row['id']}:definite-options").shuffle(values)
    options = dict(zip(CHOICE_LABELS[:4], values))
    correct_label = next(label for label, value in options.items()
                         if value == row["qwen_correct_answer"])
    return options, correct_label


def _prompt(question: str, options: dict[str, str]) -> str:
    return (
        f"Question: {question}\n\n{format_options(options)}\n\n"
        "Choose the single best answer. On the first line write exactly `Choice: X`, "
        "where X is A, B, C, or D. Then explain your choice in one or two sentences."
    )


def _generate_batch(model: Any, tokenizer: Any, prompts: list[str], *, device: str,
                    max_new_tokens: int = 80) -> list[str]:
    import torch

    rendered = [tokenizer.apply_chat_template(
        [{"role": "system", "content": "Follow the requested answer format exactly."},
         {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    ) for prompt in prompts]
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(rendered, padding=True, return_tensors="pt").to(device)
    tokenizer.padding_side = previous_padding
    with torch.inference_mode():
        generated = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            use_cache=True, pad_token_id=tokenizer.pad_token_id,
        )
    prefix_length = encoded["input_ids"].shape[1]
    return [tokenizer.decode(sequence[prefix_length:], skip_special_tokens=True).strip()
            for sequence in generated]


def qualify_definite_candidates(
    *, candidates_file: str | Path, output_file: str | Path, model_id: str,
    revision: str | None, cache_dir: str | None, device: str = "cuda",
    dtype: str = "bfloat16", limit: int = 231, seed: int = 20260816,
    batch_size: int = 8, checkpoint_callback: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    from .model import load_model_and_tokenizer, set_determinism

    set_determinism(seed)
    candidates = [json.loads(line) for line in Path(candidates_file).read_text().splitlines()
                  if line.strip()][:limit]
    output_file = Path(output_file)
    completed = {}
    if output_file.exists():
        completed = {row["id"]: row for row in
                     (json.loads(line) for line in output_file.read_text().splitlines() if line.strip())}
    pending = [row for row in candidates if row["id"] not in completed]
    model, tokenizer, _ = load_model_and_tokenizer(
        model_id, revision=revision, cache_dir=cache_dir, device=device, dtype=dtype
    )
    written = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prepared = []
        for row in batch:
            options, correct_label = _options(row, seed)
            prepared.append((row, options, correct_label, _prompt(row["question"], options)))
        responses = _generate_batch(model, tokenizer, [item[3] for item in prepared], device=device)
        with output_file.open("a", encoding="utf-8") as handle:
            for (row, options, correct_label, _), response in zip(prepared, responses):
                selected = parse_choice(response, labels=CHOICE_LABELS[:4])
                record = {
                    **row, "definite_qualification_seed": seed,
                    "definite_options": options, "correct_label": correct_label,
                    "qwen_choice_response": response, "qwen_selected_label": selected,
                    "qwen_selected_answer": options.get(selected) if selected else None,
                    "qwen_correct_choice_pass": selected == correct_label,
                    "qwen_qualification_batch_size": batch_size,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
        if checkpoint_callback is not None:
            checkpoint_callback()
    results_by_id = {row["id"]: row for row in
                     (json.loads(line) for line in output_file.read_text().splitlines() if line.strip())}
    results = [results_by_id[row["id"]] for row in candidates]
    return {
        "selected": len(candidates), "written_this_run": written,
        "qwen_correct_choice_pass": sum(row["qwen_correct_choice_pass"] for row in results),
        "output_file": str(output_file), "batch_size": batch_size,
    }
