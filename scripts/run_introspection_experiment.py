#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.introspection_experiment import run_introspection_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definite", required=True)
    parser.add_argument("--ambiguous", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--resolved-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--alphas", default="0,5,10,15")
    parser.add_argument("--measures", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-condition-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    result = run_introspection_experiment(
        definite_file=args.definite,
        ambiguous_file=args.ambiguous,
        direction_file=args.direction,
        resolved_file=args.resolved_output,
        output_file=args.output,
        manifest_file=args.manifest,
        model_id=args.model_id,
        revision=args.revision or None,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        layer_id=args.layer,
        alphas=tuple(float(value) for value in args.alphas.split(",")),
        measures=args.measures,
        limit=None if args.limit <= 0 else args.limit,
        per_condition_limit=(
            None if args.per_condition_limit <= 0 else args.per_condition_limit
        ),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
