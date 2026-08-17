from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Sequence

from .ambigqa_dataset import CHOICE_LABELS, forced_choice_prompt
from .introspection_protocol import (
    BINARY_MEASURES,
    PROTOCOL_VERSION,
    canonical_hash,
    measurement_prompt,
    parse_zero_to_one_hundred,
    select_measure_names,
    stable_seed,
)
from .model import (
    PROMPT_HEADER,
    answer_prefix_ids,
    class_first_token_ids,
    newline_token_ids,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
    return count


def _file_hash(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _single_token_labels(tokenizer: Any, labels: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for label in labels:
        ids = list(tokenizer.encode(" " + label, add_special_tokens=False))
        if len(ids) != 1:
            raise ValueError(f"Label {label!r} is not exactly one token: {ids}")
        if ids[0] in result.values():
            raise ValueError(f"Label {label!r} does not have a unique token ID")
        result[label] = ids[0]
    return result


def _left_pad(input_ids: Sequence[Sequence[int]], pad_token_id: int, device: str):
    import torch

    width = max(len(ids) for ids in input_ids)
    tensor = torch.full(
        (len(input_ids), width), pad_token_id, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(tensor)
    offsets: list[int] = []
    for row, ids in enumerate(input_ids):
        offset = width - len(ids)
        offsets.append(offset)
        tensor[row, offset:] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[row, offset:] = 1
    return tensor, mask, offsets


@contextmanager
def single_use_residual_intervention(
    model: Any,
    *,
    layer_id: int,
    token_indices: Sequence[int],
    direction: Any,
    alpha: float,
):
    """Apply the PANL edit on the first full-prefix forward and never on decode steps."""
    from .activations import decoder_layers

    tracker = {"hook_calls": 0, "applications": 0}

    def hook(_module: Any, _inputs: Any, output: Any):
        import torch

        tracker["hook_calls"] += 1
        if tracker["applications"]:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != len(token_indices):
            raise ValueError("One PANL index is required per batch row")
        if any(index < 0 or index >= hidden.shape[1] for index in token_indices):
            # During cached generation later calls contain only the newest token.
            return output
        modified = hidden.clone()
        indices = torch.as_tensor(token_indices, dtype=torch.long, device=hidden.device)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        modified[rows, indices, :] += alpha * direction.to(
            device=hidden.device, dtype=hidden.dtype
        )
        tracker["applications"] += 1
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    layers = decoder_layers(model)
    if layer_id < 0 or layer_id >= len(layers):
        raise ValueError(f"Layer {layer_id} is outside 0..{len(layers) - 1}")
    handle = layers[layer_id].register_forward_hook(hook)
    try:
        yield tracker
    finally:
        handle.remove()


def _chat_prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": "Follow the requested answer format exactly."},
        {"role": "user", "content": prompt},
    ]
    return list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    )


def _batch_next_logits(
    model: Any,
    input_ids: Sequence[Sequence[int]],
    *,
    tokenizer: Any,
    device: str,
    layer_id: int | None = None,
    panl_indices: Sequence[int] | None = None,
    direction: Any | None = None,
    alpha: float = 0.0,
):
    import torch

    tensor, mask, offsets = _left_pad(input_ids, tokenizer.pad_token_id, device)
    shifted = [offset + index for offset, index in zip(offsets, panl_indices or ())]
    intervention = nullcontext({"hook_calls": 0, "applications": 0})
    if alpha != 0.0:
        if layer_id is None or direction is None or panl_indices is None:
            raise ValueError("Nonzero alpha requires layer, direction, and PANL indices")
        intervention = single_use_residual_intervention(
            model,
            layer_id=layer_id,
            token_indices=shifted,
            direction=direction,
            alpha=alpha,
        )
    with torch.inference_mode(), intervention as tracker:
        logits = model(input_ids=tensor, attention_mask=mask, use_cache=False).logits[:, -1]
    if alpha != 0.0 and tracker["applications"] != 1:
        raise RuntimeError(f"PANL intervention applied {tracker['applications']} times")
    return logits.detach().float().cpu(), tracker


def _batch_generate(
    model: Any,
    input_ids: Sequence[Sequence[int]],
    *,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    layer_id: int | None = None,
    panl_indices: Sequence[int] | None = None,
    direction: Any | None = None,
    alpha: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import torch

    tensor, mask, offsets = _left_pad(input_ids, tokenizer.pad_token_id, device)
    shifted = [offset + index for offset, index in zip(offsets, panl_indices or ())]
    intervention = nullcontext({"hook_calls": 0, "applications": 0})
    if alpha != 0.0:
        if layer_id is None or direction is None or panl_indices is None:
            raise ValueError("Nonzero alpha requires layer, direction, and PANL indices")
        intervention = single_use_residual_intervention(
            model,
            layer_id=layer_id,
            token_indices=shifted,
            direction=direction,
            alpha=alpha,
        )
    with torch.inference_mode(), intervention as tracker:
        generated = model.generate(
            input_ids=tensor,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    if alpha != 0.0 and tracker["applications"] != 1:
        raise RuntimeError(f"PANL intervention applied {tracker['applications']} times")

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    rows: list[dict[str, Any]] = []
    prompt_width = tensor.shape[1]
    for row_index in range(tensor.shape[0]):
        token_ids = generated.sequences[row_index, prompt_width:].tolist()
        kept: list[int] = []
        logprobs: list[float] = []
        for step, token_id in enumerate(token_ids):
            if token_id in eos_ids or token_id == tokenizer.pad_token_id:
                break
            kept.append(token_id)
            if step < len(generated.scores):
                score = generated.scores[step][row_index].float()
                logprobs.append(float(torch.log_softmax(score, dim=-1)[token_id].cpu()))
        text = tokenizer.decode(kept, skip_special_tokens=True).strip()
        rows.append({
            "generated_text": text,
            "generated_token_ids": kept,
            "generated_token_logprobs": logprobs,
        })
    return rows, tracker


def _stimulus_fingerprint(row: dict[str, Any]) -> str:
    return canonical_hash({
        "stimulus_id": row["stimulus_id"],
        "condition": row["condition"],
        "question_with_options": row["question_with_options"],
        "options": row["options"],
        "replayed_answer": row.get("replayed_answer"),
    })


def _resolved_row(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    answer_label: str,
    answer_text: str,
    forced_choice: dict[str, Any] | None,
    model_revision: str | None,
    seed: int,
) -> dict[str, Any]:
    answer_ids = list(tokenizer.encode(answer_text, add_special_tokens=False))
    if not answer_ids:
        raise ValueError(f"Empty answer tokenization for {row['stimulus_id']}")
    prefix_ids = answer_prefix_ids(tokenizer, row["question_with_options"])
    panl_ids = newline_token_ids(tokenizer)
    through_panl = prefix_ids + answer_ids + panl_ids
    panl_index = len(through_panl) - 1
    if through_panl[-len(panl_ids):] != panl_ids:
        raise AssertionError("PANL token sequence was not preserved")
    return {
        **row,
        "stimulus_fingerprint": _stimulus_fingerprint(row),
        "resolution_seed": seed,
        "resolution_model_revision": model_revision,
        "resolved_answer_label": answer_label,
        "resolved_answer": answer_text,
        "forced_choice": forced_choice,
        "kumaran_prompt_header": PROMPT_HEADER,
        "replay_prefix_token_ids": prefix_ids,
        "resolved_answer_token_ids": answer_ids,
        "panl_token_ids": panl_ids,
        "replay_through_panl_token_ids": through_panl,
        "panl_index": panl_index,
        "replay_through_panl_text": tokenizer.decode(
            through_panl, skip_special_tokens=False
        ),
    }


def resolve_stimuli(
    *,
    model: Any,
    tokenizer: Any,
    stimuli: Sequence[dict[str, Any]],
    output_file: str | Path,
    device: str,
    model_revision: str | None,
    seed: int,
    batch_size: int,
    checkpoint_callback: Callable[[], Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve ambiguous forced choices once, then freeze exact replay token IDs."""
    output_file = Path(output_file)
    existing_rows = load_jsonl(output_file) if output_file.exists() else []
    existing = {row["stimulus_id"]: row for row in existing_rows}
    for row in stimuli:
        prior = existing.get(row["stimulus_id"])
        if prior and prior["stimulus_fingerprint"] != _stimulus_fingerprint(row):
            raise ValueError(f"Stimulus changed after resolution: {row['stimulus_id']}")

    definite_pending = [
        row for row in stimuli
        if row["condition"] != "ambiguous" and row["stimulus_id"] not in existing
    ]
    for row in definite_pending:
        resolved = _resolved_row(
            row,
            tokenizer=tokenizer,
            answer_label=row["replayed_label"],
            answer_text=row["replayed_answer"],
            forced_choice=None,
            model_revision=model_revision,
            seed=seed,
        )
        append_jsonl(output_file, [resolved])
        existing[resolved["stimulus_id"]] = resolved

    choice_tokens = _single_token_labels(tokenizer, CHOICE_LABELS[:4])
    ambiguous_pending = [
        row for row in stimuli
        if row["condition"] == "ambiguous" and row["stimulus_id"] not in existing
    ]
    import torch

    for batch in _chunks(ambiguous_pending, batch_size):
        prompt_ids = [
            _chat_prompt_ids(
                tokenizer,
                forced_choice_prompt(row["display_question"], row["options"]),
            )
            for row in batch
        ]
        logits, _tracker = _batch_next_logits(
            model, prompt_ids, tokenizer=tokenizer, device=device
        )
        records = []
        for row, row_logits in zip(batch, logits):
            restricted = torch.tensor([
                float(row_logits[choice_tokens[label]]) for label in CHOICE_LABELS[:4]
            ])
            probabilities = torch.softmax(restricted, dim=0)
            selected_label = CHOICE_LABELS[int(restricted.argmax())]
            selected_answer = row["options"][selected_label]
            global_argmax = int(row_logits.argmax())
            forced = {
                "prompt": forced_choice_prompt(row["display_question"], row["options"]),
                "selected_label": selected_label,
                "selected_answer": selected_answer,
                "label_token_ids": choice_tokens,
                "label_logits": {
                    label: float(restricted[index])
                    for index, label in enumerate(CHOICE_LABELS[:4])
                },
                "label_probabilities": {
                    label: float(probabilities[index])
                    for index, label in enumerate(CHOICE_LABELS[:4])
                },
                "global_argmax_token_id": global_argmax,
                "global_argmax_is_choice": global_argmax in set(choice_tokens.values()),
            }
            records.append(_resolved_row(
                row,
                tokenizer=tokenizer,
                answer_label=selected_label,
                answer_text=selected_answer,
                forced_choice=forced,
                model_revision=model_revision,
                seed=seed,
            ))
        append_jsonl(output_file, records)
        existing.update({row["stimulus_id"]: row for row in records})
        if checkpoint_callback is not None:
            checkpoint_callback()
    if checkpoint_callback is not None and definite_pending:
        checkpoint_callback()

    ordered = [existing[row["stimulus_id"]] for row in stimuli]
    if len(ordered) != len(stimuli):
        raise AssertionError("Not all stimuli were resolved")
    return ordered


def _candidate_result(
    logits: Any,
    *,
    token_ids: dict[str, int],
    semantic_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    import torch

    labels = list(token_ids)
    values = torch.tensor([float(logits[token_ids[label]]) for label in labels])
    probabilities = torch.softmax(values, dim=0)
    selected_label = labels[int(values.argmax())]
    global_argmax = int(logits.argmax())
    result = {
        "candidate_token_ids": token_ids,
        "candidate_logits": {
            label: float(values[index]) for index, label in enumerate(labels)
        },
        "candidate_probabilities": {
            label: float(probabilities[index]) for index, label in enumerate(labels)
        },
        "selected_label": selected_label,
        "global_argmax_token_id": global_argmax,
        "global_argmax_is_candidate": global_argmax in set(token_ids.values()),
    }
    if semantic_mapping is not None:
        result["label_to_response"] = semantic_mapping
        result["selected_response"] = semantic_mapping[selected_label]
        result["response_probabilities"] = {
            semantic_mapping[label]: float(probabilities[index])
            for index, label in enumerate(labels)
        }
    return result


def _common_result(
    resolved: dict[str, Any],
    *,
    run_signature: str,
    model_id: str,
    model_revision: str | None,
    layer_id: int,
    alpha: float,
    measure: str,
    suffix: str,
    suffix_ids: list[int],
    input_ids: list[int],
    intervention_tracker: dict[str, int],
) -> dict[str, Any]:
    return {
        "trial_id": f"{resolved['stimulus_id']}|alpha={alpha:g}|measure={measure}",
        "run_signature": run_signature,
        "protocol_version": PROTOCOL_VERSION,
        "stimulus_id": resolved["stimulus_id"],
        "source_id": resolved["source_id"],
        "condition": resolved["condition"],
        "model_id": model_id,
        "model_revision": model_revision,
        "layer": layer_id,
        "alpha": alpha,
        "measure": measure,
        "question": resolved["question"],
        "question_with_options": resolved["question_with_options"],
        "options": resolved["options"],
        "resolved_answer_label": resolved["resolved_answer_label"],
        "resolved_answer": resolved["resolved_answer"],
        "replay_source": resolved["replay_source"],
        "panl_index": resolved["panl_index"],
        "panl_token_ids": resolved["panl_token_ids"],
        "replay_through_panl_token_ids": resolved["replay_through_panl_token_ids"],
        "measurement_suffix": suffix,
        "measurement_suffix_token_ids": suffix_ids,
        "input_token_ids": input_ids,
        "intervention_applied": intervention_tracker["applications"] == 1,
        "intervention_application_count": intervention_tracker["applications"],
    }


def _write_manifest(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    signature = canonical_hash(payload)
    document = {**payload, "run_signature": signature}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            raise ValueError(f"Existing manifest does not match requested run: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return signature


def run_introspection_experiment(
    *,
    definite_file: str | Path,
    ambiguous_file: str | Path,
    direction_file: str | Path,
    resolved_file: str | Path,
    output_file: str | Path,
    manifest_file: str | Path,
    model_id: str,
    revision: str | None,
    cache_dir: str | None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    layer_id: int = 15,
    direction_position: str = "panl",
    alphas: Sequence[float] = (0.0, 5.0, 10.0, 15.0),
    measures: str | Sequence[str] = "all",
    limit: int | None = None,
    per_condition_limit: int | None = None,
    batch_size: int = 8,
    seed: int = 20260816,
    max_score_tokens: int = 4,
    checkpoint_callback: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run independent, resumable PANL-steered dependent-measure branches."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    from .model import load_model_and_tokenizer, set_determinism

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    measure_names = select_measure_names(measures)
    alpha_values = tuple(dict.fromkeys(float(value) for value in alphas))
    if not alpha_values:
        raise ValueError("At least one alpha is required")
    set_determinism(seed)

    definite = load_jsonl(definite_file)
    ambiguous = load_jsonl(ambiguous_file)
    stimuli = definite + ambiguous
    if len({row["stimulus_id"] for row in stimuli}) != len(stimuli):
        raise ValueError("Stimulus IDs must be unique across input files")
    if per_condition_limit is not None and per_condition_limit > 0:
        selected: list[dict[str, Any]] = []
        for condition in ("definite_correct", "definite_false", "ambiguous"):
            group = [row for row in stimuli if row["condition"] == condition]
            random.Random(stable_seed(seed, condition, "condition_sample")).shuffle(group)
            if len(group) < per_condition_limit:
                raise ValueError(
                    f"Need {per_condition_limit} {condition} items, found {len(group)}"
                )
            selected.extend(group[:per_condition_limit])
        stimuli = selected
    random.Random(seed).shuffle(stimuli)
    if limit is not None and limit > 0:
        stimuli = stimuli[:limit]
    if not stimuli:
        raise ValueError("No stimuli selected")

    direction_key = f"{direction_position}.layer_{layer_id:02d}"
    directions = load_file(str(direction_file))
    if direction_key not in directions:
        raise ValueError(f"Direction file lacks {direction_key!r}; found {sorted(directions)}")
    direction = directions[direction_key]
    with safe_open(str(direction_file), framework="pt", device="cpu") as handle:
        direction_metadata = dict(handle.metadata() or {})

    manifest_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": model_id,
        "model_revision": revision,
        "device_dtype": dtype,
        "direction_file_sha256": _file_hash(direction_file),
        "direction_key": direction_key,
        "direction_metadata": direction_metadata,
        "layer_id": layer_id,
        "alphas": list(alpha_values),
        "measures": list(measure_names),
        "measurement_prompt_hashes": {
            measure: canonical_hash([
                measurement_prompt(row["stimulus_id"], measure, seed=seed)
                for row in stimuli
            ])
            for measure in measure_names
        },
        "seed": seed,
        "max_score_tokens": max_score_tokens,
        "per_condition_limit": per_condition_limit,
        "selected_stimulus_ids": [row["stimulus_id"] for row in stimuli],
        "definite_file_sha256": _file_hash(definite_file),
        "ambiguous_file_sha256": _file_hash(ambiguous_file),
        "expected_rows": len(stimuli) * len(alpha_values) * len(measure_names),
    }
    run_signature = _write_manifest(manifest_file, manifest_payload)

    output_file = Path(output_file)
    completed: set[tuple[str, float, str]] = set()
    if output_file.exists():
        for row in load_jsonl(output_file):
            if row.get("run_signature") != run_signature:
                raise ValueError("Output file contains rows from a different run signature")
            completed.add((row["stimulus_id"], float(row["alpha"]), row["measure"]))

    model, tokenizer, _resolved_dtype = load_model_and_tokenizer(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype,
    )
    tokenizer.padding_side = "left"
    resolved = resolve_stimuli(
        model=model,
        tokenizer=tokenizer,
        stimuli=stimuli,
        output_file=resolved_file,
        device=device,
        model_revision=revision,
        seed=seed,
        batch_size=batch_size,
        checkpoint_callback=checkpoint_callback,
    )

    binary_tokens = _single_token_labels(tokenizer, ("A", "B"))
    confidence_items = sorted(class_first_token_ids(tokenizer).items())
    confidence_tokens = {item.name: token_id for token_id, item in confidence_items}
    rows_written = 0

    job_groups = [(alpha, measure) for alpha in alpha_values for measure in measure_names]
    random.Random(stable_seed(seed, "job_groups")).shuffle(job_groups)
    for alpha, measure in job_groups:
        pending = [
            row for row in resolved
            if (row["stimulus_id"], alpha, measure) not in completed
        ]
        random.Random(stable_seed(seed, alpha, measure)).shuffle(pending)
        for batch in _chunks(pending, batch_size):
            prompts: list[list[int]] = []
            suffixes: list[str] = []
            suffix_ids_list: list[list[int]] = []
            mappings: list[dict[str, str] | None] = []
            panl_indices = [int(row["panl_index"]) for row in batch]
            for row in batch:
                suffix, mapping = measurement_prompt(
                    row["stimulus_id"], measure, seed=seed
                )
                suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
                input_ids = row["replay_through_panl_token_ids"] + suffix_ids
                suffixes.append(suffix)
                suffix_ids_list.append(suffix_ids)
                mappings.append(mapping)
                prompts.append(input_ids)

            if measure == "anomaly_continuous":
                generated_rows, tracker = _batch_generate(
                    model,
                    prompts,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=max_score_tokens,
                    layer_id=layer_id,
                    panl_indices=panl_indices,
                    direction=direction,
                    alpha=alpha,
                )
                output_rows = []
                for row, input_ids, suffix, suffix_ids, generated_row in zip(
                    batch, prompts, suffixes, suffix_ids_list, generated_rows
                ):
                    common = _common_result(
                        row,
                        run_signature=run_signature,
                        model_id=model_id,
                        model_revision=revision,
                        layer_id=layer_id,
                        alpha=alpha,
                        measure=measure,
                        suffix=suffix,
                        suffix_ids=suffix_ids,
                        input_ids=input_ids,
                        intervention_tracker=tracker,
                    )
                    parsed = parse_zero_to_one_hundred(generated_row["generated_text"])
                    output_rows.append({
                        **common,
                        **generated_row,
                        "parsed_score": parsed,
                        "parse_valid": parsed is not None,
                    })
            else:
                logits, tracker = _batch_next_logits(
                    model,
                    prompts,
                    tokenizer=tokenizer,
                    device=device,
                    layer_id=layer_id,
                    panl_indices=panl_indices,
                    direction=direction,
                    alpha=alpha,
                )
                output_rows = []
                for row, input_ids, suffix, suffix_ids, mapping, row_logits in zip(
                    batch, prompts, suffixes, suffix_ids_list, mappings, logits
                ):
                    common = _common_result(
                        row,
                        run_signature=run_signature,
                        model_id=model_id,
                        model_revision=revision,
                        layer_id=layer_id,
                        alpha=alpha,
                        measure=measure,
                        suffix=suffix,
                        suffix_ids=suffix_ids,
                        input_ids=input_ids,
                        intervention_tracker=tracker,
                    )
                    if measure == "confidence_manipulation_check":
                        result = _candidate_result(row_logits, token_ids=confidence_tokens)
                        probability_by_class = result["candidate_probabilities"]
                        expected_midpoint = sum(
                            probability_by_class[item.name] * item.midpoint
                            for _token_id, item in confidence_items
                        )
                        result.update({
                            "selected_confidence_class": result.pop("selected_label"),
                            "confidence_expected_midpoint": expected_midpoint,
                        })
                    else:
                        result = _candidate_result(
                            row_logits,
                            token_ids=binary_tokens,
                            semantic_mapping=mapping,
                        )
                    output_rows.append({**common, **result})

            append_jsonl(output_file, output_rows)
            rows_written += len(output_rows)
            for row in output_rows:
                completed.add((row["stimulus_id"], float(row["alpha"]), row["measure"]))
            if checkpoint_callback is not None:
                checkpoint_callback()

    all_rows = load_jsonl(output_file)
    trial_ids = [row["trial_id"] for row in all_rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise RuntimeError("Output contains duplicate trial IDs")
    expected_rows = len(stimuli) * len(alpha_values) * len(measure_names)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(all_rows)}")
    counts: dict[str, int] = {}
    for row in all_rows:
        key = f"{row['condition']}|alpha={float(row['alpha']):g}|{row['measure']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_signature": run_signature,
        "stimuli": len(stimuli),
        "conditions": {
            condition: sum(row["condition"] == condition for row in resolved)
            for condition in sorted({row["condition"] for row in resolved})
        },
        "alphas": list(alpha_values),
        "measures": list(measure_names),
        "expected_rows": expected_rows,
        "completed_rows": len(all_rows),
        "rows_written_this_run": rows_written,
        "continuous_parse_failures": sum(
            row["measure"] == "anomaly_continuous" and not row.get("parse_valid", False)
            for row in all_rows
        ),
        "counts": counts,
        "resolved_file": str(resolved_file),
        "output_file": str(output_file),
        "manifest_file": str(manifest_file),
    }
