from __future__ import annotations

import json
import os
from pathlib import Path

import modal


APP_NAME = "qwen-confidence-introspection"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
CACHE_PATH = "/cache/huggingface"
RESULTS_PATH = "/results"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("qwen-confidence-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("qwen-confidence-results", create_if_missing=True)
# Qwen2.5 is public, so the app works without a token. Once a named secret is
# created, set MODAL_USE_HF_SECRET=1 when invoking Modal to attach it without
# placing credentials in source control.
hf_secrets = (
    [modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])]
    if os.environ.get("MODAL_USE_HF_SECRET") == "1"
    else []
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.13.0",
        "transformers==4.57.6",
        "accelerate==1.14.0",
        "datasets==4.8.5",
        "safetensors==0.8.0",
        "scikit-learn==1.7.2",
        "numpy==2.5.2",
        "pytest==9.1.1",
        "huggingface-hub==0.36.2",
    )
    .env({
        "HF_HOME": CACHE_PATH,
        "HF_HUB_CACHE": f"{CACHE_PATH}/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_python_source("src")
)

mounts = {CACHE_PATH: cache_volume, RESULTS_PATH: results_volume}


@app.function(image=image, volumes=mounts, cpu=1, memory=2048, timeout=300)
def validate_setup() -> dict[str, object]:
    import platform
    import torch
    import transformers
    from src import (  # noqa: F401
        activations,
        confidence_direction,
        evaluation,
        interventions,
        introspection_experiment,
        introspection_protocol,
    )
    from src.config import ExperimentConfig
    from src.model import CONFIDENCE_CLASSES, confidence_instructions

    config = ExperimentConfig(limit=1, sample_data=True, device="cpu", dtype="float32")
    assert ExperimentConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    assert len(CONFIDENCE_CLASSES) == 10
    assert confidence_instructions().endswith("**Confidence**:")
    for measure in introspection_protocol.SUPPORTED_MEASURE_NAMES:
        suffix, _mapping = introspection_protocol.measurement_prompt(
            "validation-item", measure, seed=20260816
        )
        assert suffix
    target = Path(RESULTS_PATH) / "validation" / "cpu_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "cache_mount": Path(CACHE_PATH).is_dir(),
        "results_mount": Path(RESULTS_PATH).is_dir(),
        "experiment_modules_imported": True,
        "introspection_protocol_version": introspection_protocol.PROTOCOL_VERSION,
        "introspection_measures": list(introspection_protocol.MEASURE_NAMES),
        "control_measures": list(introspection_protocol.CONTROL_MEASURE_NAMES),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    results_volume.commit()
    assert json.loads(target.read_text())["status"] == "ok"
    return payload


@app.function(image=image, volumes=mounts, secrets=hf_secrets, cpu=2, memory=4096, timeout=600)
def validate_introspection_runner(
    definite_name: str = "definite-experiment-stimuli-100-v1.jsonl",
    ambiguous_name: str = "ambiguous-experiment-stimuli-100-v1.jsonl",
    direction_name: str = "qwen25-7b-panl-direction-v1.safetensors",
    layer: int = 15,
) -> dict[str, object]:
    """CPU-only validation using the real tokenizer, stimuli, and direction."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from src.introspection_experiment import _resolved_row, _single_token_labels
    from src.introspection_protocol import SUPPORTED_MEASURE_NAMES, measurement_prompt
    from src.model import class_first_token_ids, newline_token_ids

    definite_path = Path(RESULTS_PATH) / "datasets" / definite_name
    ambiguous_path = Path(RESULTS_PATH) / "datasets" / ambiguous_name
    direction_path = Path(RESULTS_PATH) / "directions" / direction_name
    for path in (definite_path, ambiguous_path, direction_path):
        if not path.exists():
            raise FileNotFoundError(path)
    definite = [json.loads(line) for line in definite_path.read_text().splitlines() if line.strip()]
    ambiguous = [json.loads(line) for line in ambiguous_path.read_text().splitlines() if line.strip()]
    if len(definite) != 200 or len(ambiguous) != 100:
        raise ValueError(f"Expected 200 definite and 100 ambiguous, found {len(definite)} and {len(ambiguous)}")

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_PATH,
        allow_patterns=["tokenizer*", "*.json", "*.txt"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    newline_ids = newline_token_ids(tokenizer)
    binary_tokens = _single_token_labels(tokenizer, ("A", "B"))
    confidence_tokens = class_first_token_ids(tokenizer)
    direction_key = f"panl.layer_{layer:02d}"
    with safe_open(str(direction_path), framework="pt", device="cpu") as handle:
        if direction_key not in handle.keys():
            raise ValueError(f"Missing {direction_key}")
        direction_shape = list(handle.get_tensor(direction_key).shape)
        direction_metadata = dict(handle.metadata() or {})

    sample = definite[0]
    resolved = _resolved_row(
        sample,
        tokenizer=tokenizer,
        answer_label=sample["replayed_label"],
        answer_text=sample["replayed_answer"],
        forced_choice=None,
        model_revision=MODEL_REVISION,
        seed=20260816,
    )
    if resolved["panl_token_ids"] != newline_ids:
        raise AssertionError("Resolved PANL does not match tokenizer newline")

    # Exercise the actual one-use hook contract without loading the 7B weights.
    from src.introspection_experiment import single_use_residual_intervention

    class IdentityBlock(torch.nn.Module):
        def forward(self, hidden):
            return hidden

    class Inner:
        def __init__(self):
            self.layers = torch.nn.ModuleList([IdentityBlock()])

    class FakeModel:
        def __init__(self):
            self.model = Inner()

    fake = FakeModel()
    hidden = torch.zeros((1, 4, direction_shape[0]), dtype=torch.float32)
    direction = torch.ones(direction_shape[0], dtype=torch.float32)
    with single_use_residual_intervention(
        fake, layer_id=0, token_indices=[2], direction=direction, alpha=5.0
    ) as tracker:
        first = fake.model.layers[0](hidden)
        second = fake.model.layers[0](hidden)
    if tracker["applications"] != 1 or not torch.all(first[0, 2] == 5) or torch.any(second != 0):
        raise AssertionError("Single-use intervention contract failed")

    payload = {
        "status": "ok",
        "model_revision": MODEL_REVISION,
        "definite_items": len(definite),
        "ambiguous_items": len(ambiguous),
        "newline_token_ids": newline_ids,
        "binary_label_token_ids": binary_tokens,
        "confidence_class_token_count": len(confidence_tokens),
        "direction_key": direction_key,
        "direction_shape": direction_shape,
        "direction_metadata": direction_metadata,
        "sample_stimulus_id": sample["stimulus_id"],
        "sample_panl_index": resolved["panl_index"],
        "sample_panl_is_final_replay_token": resolved["panl_index"] == len(resolved["replay_through_panl_token_ids"]) - 1,
        "single_use_hook_applications": tracker["applications"],
        "measures": list(SUPPORTED_MEASURE_NAMES),
        "measurement_prompts_nonempty": all(
            measurement_prompt(sample["stimulus_id"], measure, seed=20260816)[0]
            for measure in SUPPORTED_MEASURE_NAMES
        ),
    }
    target = Path(RESULTS_PATH) / "validation" / "introspection-runner-cpu-v1.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    results_volume.commit()
    cache_volume.commit()
    return payload


@app.function(image=image, volumes={CACHE_PATH: cache_volume}, secrets=hf_secrets, cpu=4, memory=16384, timeout=3600)
def download_model(model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model_id,
        revision=revision or None,
        cache_dir=CACHE_PATH,
    )
    cache_volume.commit()
    return {"model_id": model_id, "snapshot_path": path}


@app.function(image=image, gpu="L4", volumes=mounts, secrets=hf_secrets, timeout=1800)
def gpu_smoke_test() -> dict[str, object]:
    import torch
    from safetensors.torch import load_file
    from src.activations import run_extraction
    from src.config import ExperimentConfig

    output_dir = f"{RESULTS_PATH}/validation/modal-l4-smoke-v4"
    config = ExperimentConfig(
        limit=1,
        sample_data=True,
        output_dir=output_dir,
        positions=("panl",),
        device="cuda",
        dtype="bfloat16",
    )
    result = run_extraction(config, cache_dir=CACHE_PATH)
    records = [json.loads(line) for line in (Path(output_dir) / "records.jsonl").read_text().splitlines()]
    tensors = load_file(str(Path(output_dir) / records[0]["activation_file"]))
    key = sorted(tensors)[0]
    payload = {
        **result,
        "gpu": torch.cuda.get_device_name(0),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "activation_key": key,
        "activation_shape": list(tensors[key].shape),
        "answer_token_ids": records[0]["answer_token_ids"],
        "confidence_class": records[0]["confidence_class"],
    }
    (Path(output_dir) / "smoke_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    results_volume.commit()
    cache_volume.commit()
    return payload


@app.function(image=image, gpu="L4", volumes=mounts, secrets=hf_secrets, timeout=24 * 60 * 60)
def extract_activations(limit: int = 3000, output_name: str = "qwen25-7b-triviaqa-kumaran-v1") -> dict[str, object]:
    from src.activations import run_extraction
    from src.config import ExperimentConfig

    config = ExperimentConfig(limit=limit, output_dir=f"{RESULTS_PATH}/activations/{output_name}")
    result = run_extraction(
        config,
        cache_dir=CACHE_PATH,
        checkpoint_callback=results_volume.commit,
        checkpoint_every=50,
    )
    results_volume.commit()
    cache_volume.commit()
    return result


@app.function(image=image, volumes={RESULTS_PATH: results_volume}, cpu=4, memory=16384, timeout=3600)
def build_direction(
    run_name: str = "qwen25-7b-triviaqa-kumaran-v1",
    output_name: str = "qwen25-7b-panl-direction-v1.safetensors",
    grades_name: str = "qwen25-7b-triviaqa-kumaran-v1-4k-semantic-grades-adjudicated.jsonl",
) -> dict[str, object]:
    from src.confidence_direction import build_direction as build

    result = build(
        f"{RESULTS_PATH}/activations/{run_name}",
        f"{RESULTS_PATH}/directions/{output_name}",
        grades_file=f"{RESULTS_PATH}/grades/{grades_name}",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="L4", volumes=mounts, secrets=hf_secrets, timeout=24 * 60 * 60)
def run_interventions(
    run_name: str = "qwen25-7b-triviaqa-kumaran-v1",
    direction_name: str = "qwen25-7b-panl-direction-v1.safetensors",
    output_name: str = "qwen25-7b-panl-steering-screen-v1.jsonl",
    limit: int = 32,
    sample_offset: int = 0,
    batch_size: int = 8,
    layers: str = "all",
    alphas: str = "-5,0,5",
    position: str = "panl",
    direction_position: str = "panl",
) -> dict[str, object]:
    from src.interventions import run_steering_sweep, summarize_steering

    parsed_layers = None if layers == "all" else [int(value) for value in layers.split(",")]
    parsed_alphas = tuple(float(value) for value in alphas.split(","))
    output_file = f"{RESULTS_PATH}/interventions/{output_name}"

    result = run_steering_sweep(
        run_dir=f"{RESULTS_PATH}/activations/{run_name}",
        direction_file=f"{RESULTS_PATH}/directions/{direction_name}",
        output_file=output_file,
        cache_dir=CACHE_PATH,
        limit=limit,
        sample_offset=sample_offset,
        batch_size=batch_size,
        layers=parsed_layers,
        alphas=parsed_alphas,
        position=position,
        direction_position=direction_position,
        checkpoint_callback=results_volume.commit,
    )
    summary = summarize_steering(output_file)
    summary_file = str(Path(output_file).with_suffix(".summary.json"))
    Path(summary_file).write_text(json.dumps(summary, indent=2) + "\n")
    results_volume.commit()
    return {**result, "summary_file": summary_file, "summary": summary}


@app.function(image=image, gpu="L4", volumes=mounts, secrets=hf_secrets, timeout=4 * 60 * 60)
def qualify_ambigqa(
    candidates_name: str = "ambigqa-four-option-candidates-v1.jsonl",
    output_name: str = "ambigqa-qwen-qualification-pilot-v1.jsonl",
    split: str = "train",
    limit: int = 20,
    seed: int = 42,
    fast_screen: bool = False,
    recognition_indices: str = "all",
    explanation_indices: str = "all",
    run_downstream: bool = True,
) -> dict[str, object]:
    from src.ambiguous_qualification import qualify_candidates

    def parse_indices(value: str) -> list[int] | None:
        return None if value == "all" else [int(part) for part in value.split(",")]

    result = qualify_candidates(
        candidates_file=f"{RESULTS_PATH}/datasets/{candidates_name}",
        output_file=f"{RESULTS_PATH}/datasets/{output_name}",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_PATH,
        split=split,
        limit=limit,
        seed=seed,
        fast_screen=fast_screen,
        recognition_variant_indices=parse_indices(recognition_indices),
        explanation_variant_indices=parse_indices(explanation_indices),
        run_downstream=run_downstream,
        checkpoint_callback=results_volume.commit,
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="L4", volumes=mounts, secrets=hf_secrets, timeout=4 * 60 * 60)
def qualify_definite_options(
    candidates_name: str = "definite-distractors-validated-v1.jsonl",
    output_name: str = "definite-qwen-qualified-v1.jsonl",
    limit: int = 231,
    seed: int = 20260816,
    batch_size: int = 8,
) -> dict[str, object]:
    from src.definite_qualification import qualify_definite_candidates

    result = qualify_definite_candidates(
        candidates_file=f"{RESULTS_PATH}/datasets/{candidates_name}",
        output_file=f"{RESULTS_PATH}/datasets/{output_name}",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_PATH,
        limit=limit,
        seed=seed,
        batch_size=batch_size,
        checkpoint_callback=results_volume.commit,
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="L4", volumes=mounts, secrets=hf_secrets, timeout=24 * 60 * 60)
def run_introspection_experiment(
    definite_name: str = "definite-experiment-stimuli-100-v1.jsonl",
    ambiguous_name: str = "ambiguous-experiment-stimuli-100-v1.jsonl",
    direction_name: str = "qwen25-7b-panl-direction-v1.safetensors",
    run_name: str = "panl-introspection-100x3-v1",
    layer: int = 15,
    alphas: str = "0,5,10,15",
    measures: str = "all",
    conditions: str = "all",
    limit: int = 0,
    per_condition_limit: int = 0,
    batch_size: int = 8,
    seed: int = 20260816,
) -> dict[str, object]:
    """Run the preregistered branches; use a distinct run_name for every pilot."""
    from src.introspection_experiment import run_introspection_experiment as run

    run_dir = Path(RESULTS_PATH) / "experiments" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run(
        definite_file=Path(RESULTS_PATH) / "datasets" / definite_name,
        ambiguous_file=Path(RESULTS_PATH) / "datasets" / ambiguous_name,
        direction_file=Path(RESULTS_PATH) / "directions" / direction_name,
        resolved_file=run_dir / "resolved_stimuli.jsonl",
        output_file=run_dir / "measurements.jsonl",
        manifest_file=run_dir / "manifest.json",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_PATH,
        layer_id=layer,
        alphas=tuple(float(value) for value in alphas.split(",")),
        measures=measures,
        conditions=conditions,
        limit=None if limit <= 0 else limit,
        per_condition_limit=None if per_condition_limit <= 0 else per_condition_limit,
        batch_size=batch_size,
        seed=seed,
        checkpoint_callback=results_volume.commit,
    )
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    results_volume.commit()
    cache_volume.commit()
    return result
