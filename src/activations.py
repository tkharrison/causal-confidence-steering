from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .config import ExperimentConfig
from .dataset import exact_match, load_triviaqa
from .model import (
    build_phase1_ids,
    answer_stop_sequences,
    class_first_token_ids,
    generate_answer,
    load_model_and_tokenizer,
    newline_token_ids,
    phase0_answer_prefix_ids,
    set_determinism,
)


def decoder_layers(model: Any):
    for path in (("model", "layers"), ("transformer", "h")):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
        if obj is not None:
            return obj
    raise TypeError("Could not locate decoder blocks")


def parse_layers(spec: str, count: int) -> list[int]:
    if spec == "all":
        return list(range(count))
    selected = sorted({int(value.strip()) for value in spec.split(",") if value.strip()})
    if not selected or selected[0] < 0 or selected[-1] >= count:
        raise ValueError(f"Layers must be within 0..{count - 1}")
    return selected


@contextmanager
def capture_residuals(model: Any, layer_ids: Iterable[int], positions: dict[str, int]):
    import torch

    captured: dict[str, Any] = {}
    handles = []
    for layer_id in layer_ids:
        def hook(_module: Any, _inputs: Any, output: Any, layer_id: int = layer_id) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            for position_name, token_index in positions.items():
                captured[f"{position_name}.layer_{layer_id:02d}"] = (
                    hidden[0, token_index].detach().to("cpu", dtype=torch.float32)
                )
        handles.append(decoder_layers(model)[layer_id].register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def confidence_forward(model: Any, input_ids: list[int], device: str, layer_ids: list[int], positions: dict[str, int]):
    import torch

    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode(), capture_residuals(model, layer_ids, positions) as captured:
        outputs = model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False)
    return outputs.logits[0, -1].detach().float().cpu(), captured


def logprob_summaries(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "min", "max", "variance", "first", "last")}
    import torch

    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "variance": float(tensor.var(unbiased=False)),
        "first": float(tensor[0]),
        "last": float(tensor[-1]),
    }


def run_extraction(
    config: ExperimentConfig,
    *,
    cache_dir: str | None = None,
    checkpoint_callback: Callable[[], Any] | None = None,
    checkpoint_every: int = 0,
) -> dict[str, Any]:
    import torch
    import transformers
    from safetensors.torch import save_file

    set_determinism(config.seed)
    output_dir = config.output_path
    activation_dir = output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, resolved_dtype = load_model_and_tokenizer(
        config.model_id,
        revision=config.model_revision,
        device=config.device,
        dtype=config.dtype,
        cache_dir=cache_dir,
    )
    layers = parse_layers(config.layers, len(decoder_layers(model)))
    class_tokens = class_first_token_ids(tokenizer)
    items = load_triviaqa(
        config.limit if config.sample_data else config.limit + config.max_skipped_items,
        config.split,
        sample_data=config.sample_data,
    )
    records_path = output_dir / "records.jsonl"
    errors_path = output_dir / "errors.jsonl"
    completed: set[str] = set()
    if records_path.exists():
        completed = {
            str(json.loads(line)["id"])
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    skipped: set[str] = set()
    if errors_path.exists():
        skipped = {
            str(json.loads(line)["id"])
            for line in errors_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    manifest = {
        **config.to_dict(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "resolved_dtype": str(resolved_dtype),
        "model_training": model.training,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "confidence_class_first_tokens": {str(key): value.name for key, value in class_tokens.items()},
        "deterministic_decoding": {"do_sample": False, "num_beams": 1},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    written = 0
    for item in items:
        if len(completed) + written >= config.limit:
            break
        item_id = str(item["id"])
        if item_id in completed or item_id in skipped:
            continue
        answer_prefix = phase0_answer_prefix_ids(tokenizer, item["question"])
        answer_ids, answer_logprobs = generate_answer(
            model,
            answer_prefix,
            tokenizer=tokenizer,
            device=config.device,
            max_new_tokens=config.answer_max_new_tokens,
            newline_sequences=answer_stop_sequences(tokenizer),
        )
        if not answer_ids:
            with errors_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "id": item_id,
                    "stage": "phase0_answer_generation",
                    "reason": "empty_answer",
                }) + "\n")
            skipped.add(item_id)
            print(json.dumps({"skipped_id": item_id, "reason": "empty_answer"}))
            continue
        phase1_ids, all_positions, replay_prefix = build_phase1_ids(tokenizer, item["question"], answer_ids)
        positions = {name: all_positions[name] for name in config.positions}
        logits, captured = confidence_forward(model, phase1_ids, config.device, layers, positions)
        global_argmax = int(logits.argmax())
        if global_argmax not in class_tokens:
            decoded = tokenizer.decode([global_argmax])
            raise RuntimeError(f"Global confidence argmax {global_argmax} ({decoded!r}) is not a valid class token")
        predicted = class_tokens[global_argmax]
        class_logits = {item.name: float(logits[token_id]) for token_id, item in class_tokens.items()}
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        activation_file = activation_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', item_id)}.safetensors"
        save_file(captured, str(activation_file))
        record = {
            "id": item_id,
            "question": item["question"],
            "aliases": item.get("aliases", []),
            "answer_text": answer_text,
            "answer_token_ids": answer_ids,
            "answer_prefix_token_ids": answer_prefix,
            "replay_prefix_token_ids": replay_prefix,
            "phase1_token_ids": phase1_ids,
            "positions": all_positions,
            "confidence_token_id": global_argmax,
            "confidence_class": predicted.name,
            "confidence_midpoint": predicted.midpoint,
            "confidence_class_logits": class_logits,
            "correct_exact_match": exact_match(answer_text, item.get("aliases", [])),
            "answer_token_logprobs": answer_logprobs,
            "answer_logprob_summaries": logprob_summaries(answer_logprobs),
            "activation_file": str(activation_file.relative_to(output_dir)),
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
        if checkpoint_callback is not None and checkpoint_every > 0 and written % checkpoint_every == 0:
            checkpoint_callback()
            print(json.dumps({"checkpoint_records_written": written}))
        print(json.dumps({"id": item_id, "answer": answer_text, "confidence": predicted.name}))
    completed_total = len(completed) + written
    if completed_total < config.limit:
        raise RuntimeError(
            f"Only produced {completed_total}/{config.limit} usable records after "
            f"skipping {len(skipped)} items"
        )
    return {
        "output_dir": str(output_dir),
        "items_total": config.limit,
        "items_written": written,
        "items_completed_total": completed_total,
        "items_skipped": len(skipped),
    }
