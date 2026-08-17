#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interventions import run_steering_sweep, summarize_steering


def comma_separated_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def comma_separated_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--position", default="panl")
    parser.add_argument("--direction-position")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--layers", type=comma_separated_ints)
    parser.add_argument("--alphas", type=comma_separated_floats, default=(-5.0, 0.0, 5.0))
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    print(json.dumps(run_steering_sweep(
        run_dir=args.run_dir,
        direction_file=args.direction,
        output_file=args.output,
        limit=args.limit,
        batch_size=args.batch_size,
        position=args.position,
        direction_position=args.direction_position,
        sample_offset=args.sample_offset,
        layers=args.layers,
        alphas=args.alphas,
        cache_dir=args.cache_dir,
    ), indent=2))
    print(json.dumps(summarize_steering(args.output), indent=2))


if __name__ == "__main__":
    main()
