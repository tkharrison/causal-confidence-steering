from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ConfidenceClass:
    name: str
    lower: float
    upper: float

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2


CONFIDENCE_CLASSES = (
    ConfidenceClass("No chance", 0.0, 0.1),
    ConfidenceClass("Really unlikely", 0.1, 0.2),
    ConfidenceClass("Chances are slight", 0.2, 0.3),
    ConfidenceClass("Unlikely", 0.3, 0.4),
    ConfidenceClass("Less than even", 0.4, 0.5),
    ConfidenceClass("Better than even", 0.5, 0.6),
    ConfidenceClass("Likely", 0.6, 0.7),
    ConfidenceClass("Very good chance", 0.7, 0.8),
    ConfidenceClass("Highly likely", 0.8, 0.9),
    ConfidenceClass("Almost certain", 0.9, 1.0),
)

PROMPT_HEADER = "You will be shown a question, together with an answer you previously provided before."


def confidence_instructions() -> str:
    rows = "\n".join(
        f'- "{item.name}" ({item.lower:.1f}-{item.upper:.1f})'
        for item in CONFIDENCE_CLASSES
    )
    return (
        "Classify your confidence into one of the following classes based on how likely "
        "the answer above is to be correct (NO REASONING OR EXPLANATION):\n\n"
        f"{rows}\n\n"
        "Each category reflects the probability that the answer is correct.\n\n"
        "At the very end of your output, format your confidence as\n"
        "**Confidence**: $CLASS\n"
        "where CLASS is one of the names (only the names without the probability ranges) "
        "of the classes above.\n\n"
        "**Confidence**:"
    )


def answer_prefix_text(question: str) -> str:
    # Literal Phase-1 prefix shown in the paper's Figure 8.
    return f"{PROMPT_HEADER}\n\nQuestion: {question}\n**Answer**: "


def phase0_answer_prefix_text(question: str) -> str:
    """Stable Qwen Phase-0 answer request.

    The paper does not publish the exact Qwen Phase-0 prompt. Keeping this
    request answer-only prevents the confidence rubric from being mistaken for
    the requested answer; Phase 1 remains the literal categorical experiment.
    """
    return f"Question: {question}"


def answer_prefix_ids(tokenizer: Any, question: str) -> list[int]:
    # The paper depicts the literal prompt and defines the confidence colon as
    # its final token. Raw tokenization preserves that geometry; a chat template
    # would insert role-control tokens after the colon.
    return list(tokenizer.encode(answer_prefix_text(question), add_special_tokens=True))


def phase0_answer_prefix_ids(tokenizer: Any, question: str) -> list[int]:
    messages = [
        {
            "role": "system",
            "content": "Answer the question directly and concisely. Give only the answer, with no explanation.",
        },
        {"role": "user", "content": phase0_answer_prefix_text(question)},
    ]
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    return ids + list(tokenizer.encode("Answer: ", add_special_tokens=False))


def newline_token_ids(tokenizer: Any) -> list[int]:
    ids = list(tokenizer.encode("\n", add_special_tokens=False))
    if not ids:
        raise RuntimeError("Tokenizer produced no token for the PANL newline")
    return ids


def answer_stop_sequences(tokenizer: Any) -> list[list[int]]:
    sequences: list[list[int]] = []
    for text in ("\n", "\n\n", "\r\n"):
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if ids and ids not in sequences:
            sequences.append(ids)
    return sequences


def confidence_suffix_ids(tokenizer: Any) -> list[int]:
    return list(tokenizer.encode(confidence_instructions(), add_special_tokens=False))


def class_first_token_ids(tokenizer: Any) -> dict[int, ConfidenceClass]:
    result: dict[int, ConfidenceClass] = {}
    for item in CONFIDENCE_CLASSES:
        ids = list(tokenizer.encode(" " + item.name, add_special_tokens=False))
        if not ids:
            raise RuntimeError(f"No tokenization for confidence class {item.name!r}")
        if ids[0] in result:
            raise RuntimeError(
                f"Confidence class first tokens are not unique: {item.name!r} and "
                f"{result[ids[0]].name!r}"
            )
        result[ids[0]] = item
    return result


def build_phase1_ids(
    tokenizer: Any,
    question: str,
    answer_ids: Sequence[int],
) -> tuple[list[int], dict[str, int], list[int]]:
    prefix = answer_prefix_ids(tokenizer, question)
    panl_ids = newline_token_ids(tokenizer)
    through_panl = prefix + list(answer_ids) + panl_ids
    suffix = confidence_suffix_ids(tokenizer)
    full_ids = through_panl + suffix
    positions = {
        "last_answer": len(prefix) + len(answer_ids) - 1,
        "panl": len(through_panl) - 1,
        "panl_plus_1": len(through_panl),
        "confidence_colon": len(full_ids) - 1,
    }
    if positions["last_answer"] < len(prefix):
        raise ValueError("Answer must contain at least one token")
    return full_ids, positions, prefix


def set_determinism(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_dtype(name: str, device: str):
    import torch

    if name == "auto":
        return torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_model_and_tokenizer(
    model_id: str,
    *,
    revision: str | None,
    device: str,
    dtype: str,
    cache_dir: str | None = None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    resolved_dtype = resolve_dtype(dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        dtype=resolved_dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, tokenizer, resolved_dtype


def generate_answer(
    model: Any,
    input_ids: Sequence[int],
    *,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    newline_sequences: Sequence[Sequence[int]],
) -> tuple[list[int], list[float]]:
    import torch

    tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=tensor,
            attention_mask=torch.ones_like(tensor),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=model.generation_config.pad_token_id,
        )
    generated = output.sequences[0, tensor.shape[1] :].tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    # The answer ends immediately before the first generated newline, matching
    # the paper's answer/PANL boundary. If no newline appears, EOS is used.
    newline_start = None
    for sequence in newline_sequences:
        sequence = list(sequence)
        for index in range(0, len(generated) - len(sequence) + 1):
            if generated[index : index + len(sequence)] == sequence:
                newline_start = index if newline_start is None else min(newline_start, index)
                break
    if newline_start is not None:
        generated = generated[:newline_start]
    # Tokenization is context-dependent: Qwen has tokens that combine spaces,
    # punctuation, and one or more newlines. Catch any such token directly.
    for index, token_id in enumerate(generated):
        if "\n" in tokenizer.decode([token_id], skip_special_tokens=False):
            generated = generated[:index]
            break
    for index, token_id in enumerate(generated):
        if token_id in eos_ids:
            generated = generated[:index]
            break
    logprobs: list[float] = []
    for step, token_id in enumerate(generated):
        if step >= len(output.scores):
            break
        logprobs.append(float(torch.log_softmax(output.scores[step][0].float(), dim=-1)[token_id].cpu()))
    return generated, logprobs
