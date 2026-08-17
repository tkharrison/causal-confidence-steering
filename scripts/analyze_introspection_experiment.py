#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


SEMANTIC_TARGET = {
    "anomaly_forced_choice": "inconsistent",
    "error_detection": "incorrect",
    "wager_abstention": "abstain",
}


def load(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def scalar(row: dict[str, Any]) -> float | None:
    measure = row["measure"]
    if measure in SEMANTIC_TARGET:
        return float(row["response_probabilities"][SEMANTIC_TARGET[measure]])
    if measure == "confidence_manipulation_check":
        return float(row["confidence_expected_midpoint"])
    if measure == "anomaly_continuous":
        score = row.get("parsed_score")
        return None if score is None else float(score) / 100.0
    raise ValueError(measure)


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "se": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
    }


def analyze(rows: list[dict[str, Any]], control_alpha: float, steered_alpha: float) -> dict[str, Any]:
    trial_ids = [row["trial_id"] for row in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("Duplicate trial IDs")
    signatures = {row["run_signature"] for row in rows}
    if len(signatures) != 1:
        raise ValueError("Rows contain more than one run signature")

    by_cell: dict[tuple[str, float, str], list[float]] = {}
    paired: dict[tuple[str, str], dict[float, float]] = {}
    for row in rows:
        value = scalar(row)
        if value is None:
            continue
        alpha = float(row["alpha"])
        by_cell.setdefault((row["condition"], alpha, row["measure"]), []).append(value)
        paired.setdefault((row["stimulus_id"], row["measure"]), {})[alpha] = value

    cells = {
        f"{condition}|alpha={alpha:g}|{measure}": describe(values)
        for (condition, alpha, measure), values in sorted(by_cell.items())
    }
    paired_deltas: dict[tuple[str, str], list[float]] = {}
    condition_by_id = {row["stimulus_id"]: row["condition"] for row in rows}
    for (stimulus_id, measure), values in paired.items():
        if control_alpha in values and steered_alpha in values:
            key = (condition_by_id[stimulus_id], measure)
            paired_deltas.setdefault(key, []).append(
                values[steered_alpha] - values[control_alpha]
            )
    delta_summary = {
        f"{condition}|{measure}": describe(values)
        for (condition, measure), values in sorted(paired_deltas.items())
    }
    all_alphas = sorted({float(row["alpha"]) for row in rows})
    dose_slopes: dict[tuple[str, str], list[float]] = {}
    dose_monotonic: dict[tuple[str, str], list[bool]] = {}
    for (stimulus_id, measure), values in paired.items():
        available = [(alpha, values[alpha]) for alpha in all_alphas if alpha in values]
        if len(available) < 2:
            continue
        xs = [item[0] for item in available]
        ys = [item[1] for item in available]
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        denominator = sum((value - mean_x) ** 2 for value in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in available) / denominator
        key = (condition_by_id[stimulus_id], measure)
        dose_slopes.setdefault(key, []).append(slope)
        dose_monotonic.setdefault(key, []).append(
            all(ys[index + 1] >= ys[index] for index in range(len(ys) - 1))
        )
    dose_response = {}
    for key, values in sorted(dose_slopes.items()):
        condition, measure = key
        monotonic = dose_monotonic[key]
        dose_response[f"{condition}|{measure}"] = {
            **describe(values),
            "quantity": "within-item outcome change per alpha unit",
            "monotonic_items": sum(monotonic),
            "monotonic_rate": sum(monotonic) / len(monotonic),
        }
    dose_slope_contrasts: dict[str, dict[str, float | int]] = {}
    for measure in sorted({measure for _condition, measure in dose_slopes}):
        correct = dose_slopes.get(("definite_correct", measure), [])
        if not correct:
            continue
        for condition in ("definite_false", "ambiguous"):
            values = dose_slopes.get((condition, measure), [])
            if values:
                dose_slope_contrasts[f"{condition}_minus_definite_correct|{measure}"] = {
                    "n_condition": len(values),
                    "n_definite_correct": len(correct),
                    "difference_in_mean_dose_slope": statistics.mean(values) - statistics.mean(correct),
                }
    difference_from_correct: dict[str, dict[str, float | int | None]] = {}
    measures = sorted({measure for _condition, measure in paired_deltas})
    for measure in measures:
        baseline = paired_deltas.get(("definite_correct", measure), [])
        if not baseline:
            continue
        baseline_mean = statistics.mean(baseline)
        for condition in ("definite_false", "ambiguous"):
            values = paired_deltas.get((condition, measure), [])
            if values:
                difference_from_correct[f"{condition}_minus_definite_correct|{measure}"] = {
                    "n_condition": len(values),
                    "n_definite_correct": len(baseline),
                    "difference_in_mean_paired_delta": statistics.mean(values) - baseline_mean,
                }
    steered_condition_contrasts: dict[str, dict[str, float | int | None]] = {}
    for measure in measures:
        correct = by_cell.get(("definite_correct", steered_alpha, measure), [])
        if not correct:
            continue
        for condition in ("definite_false", "ambiguous"):
            values = by_cell.get((condition, steered_alpha, measure), [])
            if values:
                steered_condition_contrasts[f"{condition}_minus_definite_correct|{measure}"] = {
                    "alpha": steered_alpha,
                    "n_condition": len(values),
                    "n_definite_correct": len(correct),
                    "difference_in_mean_level": statistics.mean(values) - statistics.mean(correct),
                }
    return {
        "run_signature": next(iter(signatures)),
        "rows": len(rows),
        "control_alpha": control_alpha,
        "steered_alpha": steered_alpha,
        "all_alphas": all_alphas,
        "cell_summaries": cells,
        "paired_steering_deltas": delta_summary,
        "dose_response_slopes": dose_response,
        "dose_slope_contrasts": dose_slope_contrasts,
        "difference_from_definite_correct": difference_from_correct,
        "steered_condition_contrasts": steered_condition_contrasts,
        "interpretation": {
            "anomaly_forced_choice": "Probability assigned to inconsistent",
            "anomaly_continuous": "Parsed 0-100 score divided by 100",
            "confidence_manipulation_check": "Expected midpoint of ten confidence classes",
            "error_detection": "Probability assigned to incorrect",
            "wager_abstention": "Probability assigned to abstain",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--control-alpha", type=float, default=0.0)
    parser.add_argument("--steered-alpha", type=float, default=15.0)
    args = parser.parse_args()

    rows = load(args.input)
    result = analyze(rows, args.control_alpha, args.steered_alpha)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "trial_id", "stimulus_id", "condition", "alpha", "measure", "value"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "trial_id": row["trial_id"],
                "stimulus_id": row["stimulus_id"],
                "condition": row["condition"],
                "alpha": row["alpha"],
                "measure": row["measure"],
                "value": scalar(row),
            })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
