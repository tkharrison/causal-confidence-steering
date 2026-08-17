from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Callable, Sequence

from .ambigqa_dataset import (
    CHOICE_LABELS,
    forced_choice_prompt,
    format_options,
    parse_choice,
    recognition_choice_prompt,
    recognition_prompt,
    recognition_variants,
)


def _model_text(model: Any, tokenizer: Any, prompt: str, *, device: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": "Follow the requested answer format exactly."},
        {"role": "user", "content": prompt},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(generated[0, ids.shape[1] :], skip_special_tokens=True).strip()


def _next_logits(model: Any, tokenizer: Any, prompt: str, *, device: str):
    import torch

    messages = [
        {"role": "system", "content": "Follow the requested answer format exactly."},
        {"role": "user", "content": prompt},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        return model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).logits[0, -1].detach().float().cpu()


def _single_token_labels(tokenizer: Any, labels: Sequence[str]) -> dict[str, int]:
    result = {}
    seen: set[int] = set()
    for label in labels:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Choice label {label!r} is not one token: {ids}")
        if ids[0] in seen:
            raise ValueError(f"Choice labels do not have unique token IDs: {label!r}")
        seen.add(ids[0])
        result[label] = ids[0]
    return result


def _substantive_options(candidate: dict[str, Any], *, seed: int) -> dict[str, str]:
    options = list(candidate["substantive_options"])
    random.Random(f"{seed}:{candidate['id']}:forced").shuffle(options)
    return dict(zip(CHOICE_LABELS[:4], options))


def _confidence_measure(
    model: Any,
    tokenizer: Any,
    *,
    question_with_options: str,
    answer_text: str,
    device: str,
) -> dict[str, Any]:
    import torch
    from .model import build_phase1_ids, class_first_token_ids

    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    phase1_ids, positions, replay_prefix = build_phase1_ids(
        tokenizer, question_with_options, answer_ids
    )
    tensor = torch.tensor([phase1_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(
            input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False
        ).logits[0, -1].detach().float().cpu()
    class_tokens = class_first_token_ids(tokenizer)
    items = sorted(class_tokens.items())
    class_logits = torch.tensor([float(logits[token_id]) for token_id, _item in items])
    probabilities = torch.softmax(class_logits, dim=0)
    restricted_index = int(class_logits.argmax())
    expected = sum(float(probabilities[index]) * item.midpoint for index, (_token, item) in enumerate(items))
    global_id = int(logits.argmax())
    global_class = class_tokens.get(global_id)
    return {
        "confidence_class": items[restricted_index][1].name,
        "confidence_expected_midpoint": expected,
        "confidence_global_argmax_is_class": global_class is not None,
        "confidence_global_argmax_class": global_class.name if global_class else None,
        "confidence_class_logits": {
            item.name: float(class_logits[index])
            for index, (_token, item) in enumerate(items)
        },
        "phase1_token_ids": phase1_ids,
        "replay_prefix_token_ids": replay_prefix,
        "positions": positions,
    }


def qualify_candidates(
    *,
    candidates_file: str | Path,
    output_file: str | Path,
    model_id: str,
    revision: str | None,
    cache_dir: str | None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    split: str | None = "train",
    limit: int = 20,
    seed: int = 42,
    checkpoint_callback: Callable[[], Any] | None = None,
    fast_screen: bool = False,
    recognition_variant_indices: Sequence[int] | None = None,
    explanation_variant_indices: Sequence[int] | None = None,
    run_downstream: bool = True,
) -> dict[str, Any]:
    from .model import load_model_and_tokenizer, set_determinism
    import torch

    set_determinism(seed)
    candidates = [
        json.loads(line) for line in Path(candidates_file).read_text().splitlines() if line.strip()
    ]
    if split and split != "all":
        candidates = [row for row in candidates if row["source_split"] == split]
    random.Random(seed).shuffle(candidates)
    candidates = candidates[:limit]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output_file.exists():
        completed = {
            str(json.loads(line)["id"])
            for line in output_file.read_text().splitlines()
            if line.strip()
        }

    model, tokenizer, _ = load_model_and_tokenizer(
        model_id, revision=revision, cache_dir=cache_dir, device=device, dtype=dtype
    )
    choice_tokens = _single_token_labels(tokenizer, CHOICE_LABELS)
    written = 0
    passed = 0
    for candidate in candidates:
        if candidate["id"] in completed:
            continue
        display_question = candidate.get("display_question", candidate["question"])
        all_variants = recognition_variants(candidate, seed=seed)
        direct_variants = (
            [all_variants[index] for index in recognition_variant_indices]
            if recognition_variant_indices is not None else all_variants
        )
        explanation_variants = (
            [all_variants[index] for index in explanation_variant_indices]
            if explanation_variant_indices is not None else all_variants
        )
        recognition = []
        for variant in direct_variants:
            if fast_screen:
                logits = _next_logits(
                    model,
                    tokenizer,
                    recognition_choice_prompt(display_question, variant),
                    device=device,
                )
                restricted = torch.tensor([
                    float(logits[choice_tokens[label]]) for label in CHOICE_LABELS
                ])
                choice_probabilities = torch.softmax(restricted, dim=0)
                choice = CHOICE_LABELS[int(restricted.argmax())]
                text = None
            else:
                text = _model_text(
                    model,
                    tokenizer,
                    recognition_prompt(display_question, variant),
                    device=device,
                    max_new_tokens=96,
                )
                choice = parse_choice(text)
            recognition.append({
                **variant,
                "response_text": text,
                "parsed_choice": choice,
                "selected_abstention": choice == variant["abstention_label"],
                **({
                    "choice_probabilities": {
                        label: float(choice_probabilities[index])
                        for index, label in enumerate(CHOICE_LABELS)
                    },
                    "choice_margin": float(
                        choice_probabilities.topk(2).values[0]
                        - choice_probabilities.topk(2).values[1]
                    ),
                } if fast_screen else {}),
            })
        recognition_count = sum(row["selected_abstention"] for row in recognition)
        recognition_pass = recognition_count == len(recognition)

        explanation_confirmation = []
        explanation_count = None
        if fast_screen and recognition_pass:
            for variant in explanation_variants:
                text = _model_text(
                    model,
                    tokenizer,
                    recognition_prompt(display_question, variant),
                    device=device,
                    max_new_tokens=96,
                )
                choice = parse_choice(text)
                explanation_confirmation.append({
                    **variant,
                    "response_text": text,
                    "parsed_choice": choice,
                    "selected_abstention": choice == variant["abstention_label"],
                })
            explanation_count = sum(
                row["selected_abstention"] for row in explanation_confirmation
            )
            recognition_pass = explanation_count == len(explanation_confirmation)

        if fast_screen and not recognition_pass:
            record = {
                **candidate,
                "qualification_seed": seed,
                "qualification_mode": "fast_screen_then_explanations",
                "recognition": recognition,
                "recognition_abstention_count": recognition_count,
                "recognition_variant_count": len(recognition),
                "recognition_explanation_confirmation": explanation_confirmation,
                "recognition_explanation_abstention_count": explanation_count,
                "recognition_explanation_variant_count": len(explanation_confirmation),
                "recognition_pass": False,
                "recognition_pass_5_of_5": False,
                "qualification_stopped_after_recognition": True,
            }
            with output_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if checkpoint_callback is not None:
                checkpoint_callback()
            continue

        if fast_screen and recognition_pass and not run_downstream:
            record = {
                **candidate,
                "qualification_seed": seed,
                "qualification_mode": "fast_screen_then_explanations_no_downstream",
                "recognition": recognition,
                "recognition_abstention_count": recognition_count,
                "recognition_variant_count": len(recognition),
                "recognition_explanation_confirmation": explanation_confirmation,
                "recognition_explanation_abstention_count": explanation_count,
                "recognition_explanation_variant_count": len(explanation_confirmation),
                "recognition_pass": True,
                "recognition_pass_5_of_5": (
                    len(recognition) == 5
                    and len(explanation_confirmation) == 5
                ),
                "qualification_stopped_after_recognition": False,
                "qualification_completed_without_downstream": True,
            }
            with output_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            passed += 1
            if checkpoint_callback is not None:
                checkpoint_callback()
            continue

        if not fast_screen and not run_downstream:
            record = {
                **candidate,
                "qualification_seed": seed,
                "qualification_mode": "generated_explanations_no_downstream",
                "recognition": recognition,
                "recognition_abstention_count": recognition_count,
                "recognition_variant_count": len(recognition),
                "recognition_explanation_confirmation": [],
                "recognition_explanation_abstention_count": recognition_count,
                "recognition_explanation_variant_count": len(recognition),
                "recognition_pass": recognition_pass,
                "recognition_pass_5_of_5": (
                    recognition_pass and len(recognition) == 5
                ),
                "qualification_stopped_after_recognition": not recognition_pass,
                "qualification_completed_without_downstream": recognition_pass,
            }
            with output_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            passed += int(recognition_pass)
            if checkpoint_callback is not None:
                checkpoint_callback()
            continue

        substantive_options = _substantive_options(candidate, seed=seed)
        forced_prompt = forced_choice_prompt(display_question, substantive_options)
        choice_logits = _next_logits(model, tokenizer, forced_prompt, device=device)
        restricted_logits = torch.tensor([
            float(choice_logits[choice_tokens[label]]) for label in CHOICE_LABELS[:4]
        ])
        probabilities = torch.softmax(restricted_logits, dim=0)
        selected_index = int(restricted_logits.argmax())
        selected_label = CHOICE_LABELS[selected_index]
        selected_answer = substantive_options[selected_label]
        global_argmax = int(choice_logits.argmax())
        question_with_options = (
            f"{display_question}\n\n{format_options(substantive_options)}"
        )
        confidence = _confidence_measure(
            model,
            tokenizer,
            question_with_options=question_with_options,
            answer_text=selected_answer,
            device=device,
        )
        record = {
            **candidate,
            "qualification_seed": seed,
            "recognition": recognition,
            "recognition_abstention_count": recognition_count,
            "recognition_variant_count": len(recognition),
            "recognition_explanation_confirmation": explanation_confirmation,
            "recognition_explanation_abstention_count": explanation_count,
            "recognition_explanation_variant_count": len(explanation_confirmation),
            "recognition_pass": recognition_pass,
            "recognition_pass_5_of_5": recognition_pass,
            "qualification_mode": (
                "fast_screen_then_explanations" if fast_screen else "generated_explanations"
            ),
            "qualification_stopped_after_recognition": False,
            "qualification_completed_without_downstream": False,
            "forced_options": substantive_options,
            "forced_selected_label": selected_label,
            "forced_selected_answer": selected_answer,
            "forced_choice_probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(CHOICE_LABELS[:4])
            },
            "forced_choice_margin": float(
                probabilities.topk(2).values[0] - probabilities.topk(2).values[1]
            ),
            "forced_global_argmax_token_id": global_argmax,
            "forced_global_argmax_is_choice": global_argmax in {
                choice_tokens[label] for label in CHOICE_LABELS[:4]
            },
            **confidence,
        }
        with output_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
        passed += int(recognition_pass)
        if checkpoint_callback is not None:
            checkpoint_callback()

    all_rows = [
        json.loads(line) for line in output_file.read_text().splitlines() if line.strip()
    ]
    return {
        "output_file": str(output_file),
        "selected_candidate_count": len(candidates),
        "written_this_run": written,
        "completed_total": len(all_rows),
        "recognition_pass_5_of_5_total": sum(
            bool(row.get("recognition_pass", row.get("recognition_pass_5_of_5")))
            for row in all_rows
        ),
    }
