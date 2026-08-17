#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations import run_extraction
from src.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to an ExperimentConfig JSON file")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    config = ExperimentConfig.from_dict(json.loads(open(args.config, encoding="utf-8").read()))
    print(json.dumps(run_extraction(config, cache_dir=args.cache_dir), indent=2))


if __name__ == "__main__":
    main()
