from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

from .activations import decoder_layers


def _direction_metadata(direction_file: str | Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(direction_file), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def direction_training_ids(direction_file: str | Path) -> set[str]:
    """Return every trial used to estimate the direction."""
    metadata = _direction_metadata(direction_file)
    selected: set[str] = set()
    for field in ("selected_low_ids", "selected_high_ids"):
        values = json.loads(metadata.get(field, "[]"))
        if not isinstance(values, list):
            raise ValueError(f"Direction metadata {field!r} must be a JSON list")
        selected.update(str(value) for value in values)
    return selected


def select_held_out_records(
    records: Sequence[dict[str, Any]],
    *,
    excluded_ids: Iterable[str],
    limit: int,
    seed: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Select a deterministic held-out sample without relying on file order."""
    excluded = set(excluded_ids)
    eligible = [row for row in records if str(row["id"]) not in excluded]
    if offset < 0:
        raise ValueError("offset cannot be negative")
    if len(eligible) < offset + limit:
        raise ValueError(
            f"Only {len(eligible)} held-out records are available for "
            f"offset={offset}, limit={limit}"
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[offset : offset + limit]


@contextmanager
def additive_residual_intervention(
    model: Any,
    *,
    layer_id: int,
    token_indices: Sequence[int] | Any,
    direction: Any,
    alpha: float,
):
    """Add alpha * direction to one post-block residual vector per batch row."""
    def hook(_module: Any, _inputs: Any, output: Any):
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        indices = torch.as_tensor(token_indices, dtype=torch.long, device=hidden.device)
        if indices.numel() != hidden.shape[0]:
            raise ValueError("One intervention token index is required per batch row")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        modified[rows, indices, :] += alpha * direction.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = decoder_layers(model)[layer_id].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def confidence_logits_batch(
    model: Any,
    input_ids: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: str,
    layer_id: int | None = None,
    token_indices: Sequence[int] | None = None,
    direction: Any | None = None,
    alpha: float = 0.0,
):
    """Run a right-padded batch and return logits at each unpadded final token."""
    import torch

    lengths = torch.tensor([len(ids) for ids in input_ids], dtype=torch.long, device=device)
    max_length = int(lengths.max())
    tensor = torch.full(
        (len(input_ids), max_length), pad_token_id, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(tensor)
    for row, ids in enumerate(input_ids):
        tensor[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[row, : len(ids)] = 1

    intervention = nullcontext()
    if layer_id is not None:
        intervention = additive_residual_intervention(
            model,
            layer_id=layer_id,
            token_indices=token_indices or (),
            direction=direction,
            alpha=alpha,
        )
    with torch.inference_mode(), intervention:
        logits = model(input_ids=tensor, attention_mask=mask, use_cache=False).logits
    rows = torch.arange(len(input_ids), device=device)
    return logits[rows, lengths - 1].detach().float().cpu()


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _result_row(
    *,
    record: dict[str, Any],
    logits: Any,
    class_token_items: Sequence[tuple[int, Any]],
    position: str,
    direction_position: str,
    layer_id: int,
    alpha: float,
    batch_clean_expected_midpoint: float | None = None,
    batch_clean_global_class: str | None = None,
) -> dict[str, Any]:
    import torch

    token_ids = [token_id for token_id, _item in class_token_items]
    class_logits = logits[token_ids]
    probabilities = torch.softmax(class_logits, dim=0)
    restricted_index = int(class_logits.argmax())
    restricted_class = class_token_items[restricted_index][1]
    global_argmax = int(logits.argmax())
    global_classes = {token_id: item for token_id, item in class_token_items}
    global_class = global_classes.get(global_argmax)
    expected_midpoint = sum(
        float(probabilities[index]) * item.midpoint
        for index, (_token_id, item) in enumerate(class_token_items)
    )
    clean_logits = torch.tensor(
        [float(record["confidence_class_logits"][item.name]) for _token_id, item in class_token_items]
    )
    clean_probabilities = torch.softmax(clean_logits, dim=0)
    stored_clean_expected_midpoint = sum(
        float(clean_probabilities[index]) * item.midpoint
        for index, (_token_id, item) in enumerate(class_token_items)
    )
    reference_expected_midpoint = (
        stored_clean_expected_midpoint
        if batch_clean_expected_midpoint is None
        else batch_clean_expected_midpoint
    )
    return {
        "id": str(record["id"]),
        "position": position,
        "direction_position": direction_position,
        "layer": layer_id,
        "alpha": alpha,
        "clean_class": record["confidence_class"],
        "clean_midpoint": record["confidence_midpoint"],
        "stored_clean_expected_midpoint": stored_clean_expected_midpoint,
        "batch_clean_expected_midpoint": reference_expected_midpoint,
        "batch_clean_global_class": batch_clean_global_class,
        "global_argmax_token_id": global_argmax,
        "global_argmax_is_class": global_class is not None,
        "global_argmax_class": global_class.name if global_class else None,
        "restricted_class": restricted_class.name,
        "restricted_midpoint": restricted_class.midpoint,
        "expected_midpoint": expected_midpoint,
        "delta_expected_midpoint": expected_midpoint - reference_expected_midpoint,
        "class_logits": {
            item.name: float(class_logits[index])
            for index, (_token_id, item) in enumerate(class_token_items)
        },
    }


def run_steering_sweep(
    *,
    run_dir: str | Path,
    direction_file: str | Path,
    output_file: str | Path,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    revision: str | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    cache_dir: str | None = None,
    position: str = "panl",
    direction_position: str | None = None,
    alphas: tuple[float, ...] = (-5.0, 0.0, 5.0),
    layers: Sequence[int] | None = None,
    limit: int = 32,
    batch_size: int = 8,
    seed: int = 42,
    sample_offset: int = 0,
    checkpoint_callback: Any | None = None,
) -> dict[str, Any]:
    """Run a resumable, held-out, batched causal steering sweep."""
    from safetensors.torch import load_file
    from .model import class_first_token_ids, load_model_and_tokenizer, set_determinism

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    set_determinism(seed)
    run_dir, output_file = Path(run_dir), Path(output_file)
    all_records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    training_ids = direction_training_ids(direction_file)
    records = select_held_out_records(
        all_records, excluded_ids=training_ids, limit=limit, seed=seed, offset=sample_offset
    )
    directions = load_file(str(direction_file))
    direction_position = direction_position or position
    direction_keys = {
        int(key.rsplit("_", 1)[1]): key
        for key in directions
        if key.startswith(direction_position + ".layer_")
    }
    available_layers = sorted(direction_keys)
    layer_ids = available_layers if layers is None else [int(layer) for layer in layers]
    missing_layers = sorted(set(layer_ids) - set(available_layers))
    if missing_layers:
        raise ValueError(
            f"Direction file lacks {direction_position} vectors for layers {missing_layers}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, int, float]] = set()
    if output_file.exists():
        for line in output_file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((str(row["id"]), int(row["layer"]), float(row["alpha"])))

    model, tokenizer, _ = load_model_and_tokenizer(
        model_id, revision=revision, device=device, dtype=dtype, cache_dir=cache_dir
    )
    class_token_items = sorted(class_first_token_ids(tokenizer).items())
    alpha_values = tuple(dict.fromkeys(float(alpha) for alpha in alphas))
    nonzero_alphas = tuple(alpha for alpha in alpha_values if alpha != 0.0)
    rows_written = 0

    for batch in _chunks(records, batch_size):
        input_ids = [record["phase1_token_ids"] for record in batch]
        token_indices = [int(record["positions"][position]) for record in batch]

        clean_logits = confidence_logits_batch(
            model, input_ids, pad_token_id=tokenizer.pad_token_id, device=device
        )
        clean_reference_rows = [
            _result_row(
                record=record, logits=logits, class_token_items=class_token_items,
                position=position, direction_position=direction_position,
                layer_id=-1, alpha=0.0,
            )
            for record, logits in zip(batch, clean_logits)
        ]
        clean_expected = [row["expected_midpoint"] for row in clean_reference_rows]
        clean_global_classes = [row["global_argmax_class"] for row in clean_reference_rows]

        if 0.0 in alpha_values:
            with output_file.open("a", encoding="utf-8") as handle:
                for layer_id in layer_ids:
                    for index, (record, logits) in enumerate(zip(batch, clean_logits)):
                        key = (str(record["id"]), layer_id, 0.0)
                        if key in completed:
                            continue
                        handle.write(json.dumps(_result_row(
                            record=record, logits=logits, class_token_items=class_token_items,
                            position=position, direction_position=direction_position,
                            layer_id=layer_id, alpha=0.0,
                            batch_clean_expected_midpoint=clean_expected[index],
                            batch_clean_global_class=clean_global_classes[index],
                        )) + "\n")
                        completed.add(key)
                        rows_written += 1

        for layer_id in layer_ids:
            direction = directions[direction_keys[layer_id]]
            for alpha in nonzero_alphas:
                pending = [index for index, record in enumerate(batch)
                           if (str(record["id"]), layer_id, alpha) not in completed]
                if not pending:
                    continue
                pending_records = [batch[index] for index in pending]
                logits_batch = confidence_logits_batch(
                    model,
                    [input_ids[index] for index in pending],
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                    layer_id=layer_id,
                    token_indices=[token_indices[index] for index in pending],
                    direction=direction,
                    alpha=alpha,
                )
                with output_file.open("a", encoding="utf-8") as handle:
                    for pending_index, (record, logits) in zip(pending, zip(pending_records, logits_batch)):
                        handle.write(json.dumps(_result_row(
                            record=record, logits=logits, class_token_items=class_token_items,
                            position=position, direction_position=direction_position,
                            layer_id=layer_id, alpha=alpha,
                            batch_clean_expected_midpoint=clean_expected[pending_index],
                            batch_clean_global_class=clean_global_classes[pending_index],
                        )) + "\n")
                        completed.add((str(record["id"]), layer_id, alpha))
                        rows_written += 1
        if checkpoint_callback is not None:
            checkpoint_callback()

    expected_rows = len(records) * len(layer_ids) * len(alpha_values)
    selected_ids = [str(record["id"]) for record in records]
    return {
        "output_file": str(output_file),
        "rows_expected": expected_rows,
        "rows_written_this_run": rows_written,
        "trials": len(records),
        "layers": layer_ids,
        "position": position,
        "direction_position": direction_position,
        "alphas": list(alpha_values),
        "batch_size": batch_size,
        "seed": seed,
        "sample_offset": sample_offset,
        "selected_ids": selected_ids,
        "direction_training_overlap": len(set(selected_ids) & training_ids),
    }


def summarize_steering(output_file: str | Path) -> list[dict[str, Any]]:
    """Aggregate paired expected-confidence shifts by layer and alpha."""
    rows = [json.loads(line) for line in Path(output_file).read_text().splitlines() if line.strip()]
    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["layer"]), float(row["alpha"])), []).append(row)
    summary = []
    for (layer, alpha), group in sorted(groups.items()):
        summary.append({
            "layer": layer,
            "alpha": alpha,
            "n": len(group),
            "mean_delta_expected_midpoint": sum(
                float(row["delta_expected_midpoint"]) for row in group
            ) / len(group),
            "global_class_valid_rate": sum(bool(row["global_argmax_is_class"]) for row in group) / len(group),
        })
    return summary
