from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: str | None = "a09a35458c702b33eeacc393d103063234e8bc28"
    dataset_name: str = "mandarjoshi/trivia_qa"
    dataset_config: str = "rc"
    split: str = "validation"
    limit: int = 3000
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    layers: str = "all"
    positions: tuple[str, ...] = ("panl", "panl_plus_1", "confidence_colon", "last_answer")
    answer_max_new_tokens: int = 48
    max_skipped_items: int = 100
    output_dir: str = "/results/activations/qwen25-7b-triviaqa-kumaran-v1"
    sample_data: bool = False

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        clean = dict(values)
        if "positions" in clean:
            clean["positions"] = tuple(clean["positions"])
        return cls(**clean)
