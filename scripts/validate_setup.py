#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.model import CONFIDENCE_CLASSES, confidence_instructions


def validate(output: Path | None = None) -> dict[str, object]:
    config = ExperimentConfig(limit=1, sample_data=True, device="cpu", dtype="float32")
    names = [item.name for item in CONFIDENCE_CLASSES]
    assert len(names) == 10 and len(set(names)) == 10
    assert names[3] == "Unlikely" and names[6] == "Likely"
    assert confidence_instructions().endswith("**Confidence**:")
    payload = {
        "status": "ok",
        "python": platform.python_version(),
        "config_round_trip": ExperimentConfig.from_dict(config.to_dict()).to_dict() == config.to_dict(),
        "confidence_classes": names,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output), indent=2))


if __name__ == "__main__":
    main()
