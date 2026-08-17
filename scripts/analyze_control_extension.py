from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable


NEGATIVE_ALPHAS = (-15.0, -10.0, -5.0)
PRIMARY_ALPHAS = (0.0, 5.0, 10.0, 15.0)
FULL_ALPHAS = NEGATIVE_ALPHAS + PRIMARY_ALPHAS
NEGATIVE_MEASURES = (
    "confidence_manipulation_check",
    "anomaly_forced_choice",
    "error_detection",
)
POLARITY_MEASURES = (
    "inconsistency_yes_no",
    "consistency_yes_no",
)
ALARM_RESPONSE = {
    "anomaly_forced_choice": "inconsistent",
    "error_detection": "incorrect",
    "inconsistency_yes_no": "yes",
    "consistency_yes_no": "no",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: Iterable[float]) -> float:
    return statistics.mean(values)


def ci_mean(values: list[float]) -> list[float]:
    center = mean(values)
    if len(values) < 2:
        return [center, center]
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [center - half_width, center + half_width]


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "ci95": ci_mean(values),
        "positive_items": sum(value > 0 for value in values),
        "negative_items": sum(value < 0 for value in values),
        "unchanged_items": sum(value == 0 for value in values),
    }


def summarize_difference(first: list[float], second: list[float]) -> dict[str, Any]:
    center = mean(first) - mean(second)
    standard_error = math.sqrt(
        statistics.variance(first) / len(first)
        + statistics.variance(second) / len(second)
    )
    return {
        "n_first": len(first),
        "n_second": len(second),
        "mean": center,
        "ci95": [center - 1.96 * standard_error, center + 1.96 * standard_error],
    }


def candidate_log_odds(row: dict[str, Any], response: str) -> float:
    target = next(
        label for label, semantic in row["label_to_response"].items()
        if semantic == response
    )
    other = "B" if target == "A" else "A"
    return float(row["candidate_logits"][target]) - float(row["candidate_logits"][other])


def candidate_probability(row: dict[str, Any], response: str) -> float:
    return float(row["response_probabilities"][response])


def confidence(row: dict[str, Any]) -> float:
    return float(row["confidence_expected_midpoint"])


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    conditions: set[str],
    alphas: set[float],
    measures: set[str],
    expected_rows: int,
) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    if {row["condition"] for row in rows} != conditions:
        raise ValueError("Unexpected conditions")
    if {float(row["alpha"]) for row in rows} != alphas:
        raise ValueError("Unexpected alpha values")
    if {row["measure"] for row in rows} != measures:
        raise ValueError("Unexpected measures")
    trial_ids = [row["trial_id"] for row in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("Duplicate trial IDs")
    for row in rows:
        alpha = float(row["alpha"])
        applications = int(row["intervention_application_count"])
        if applications != (0 if alpha == 0 else 1):
            raise ValueError(f"Wrong hook count for {row['trial_id']}")
        if row["measure"] != "confidence_manipulation_check" and not row.get(
            "global_argmax_is_candidate", False
        ):
            raise ValueError(f"Invalid candidate argmax for {row['trial_id']}")


def index_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, float], dict[str, Any]]:
    indexed: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in rows:
        key = (row["stimulus_id"], row["measure"], float(row["alpha"]))
        if key in indexed:
            raise ValueError(f"Duplicate row key: {key}")
        indexed[key] = row
    return indexed


def paired_changes(
    indexed: dict[tuple[str, str, float], dict[str, Any]],
    stimulus_ids: list[str],
    measure: str,
    first_alpha: float,
    second_alpha: float,
    scalar: Callable[[dict[str, Any]], float],
) -> list[float]:
    return [
        scalar(indexed[(stimulus_id, measure, second_alpha)])
        - scalar(indexed[(stimulus_id, measure, first_alpha)])
        for stimulus_id in stimulus_ids
    ]


def slope(points: list[tuple[float, float]]) -> float:
    xbar = mean(x for x, _ in points)
    ybar = mean(y for _, y in points)
    return sum((x - xbar) * (y - ybar) for x, y in points) / sum(
        (x - xbar) ** 2 for x, _ in points
    )


def analyze_negative_sweep(
    primary_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    primary = [
        row for row in primary_rows
        if row["condition"] == "definite_correct"
        and row["measure"] in NEGATIVE_MEASURES
        and float(row["alpha"]) in PRIMARY_ALPHAS
    ]
    validate_rows(
        negative_rows,
        conditions={"definite_correct"},
        alphas=set(NEGATIVE_ALPHAS),
        measures=set(NEGATIVE_MEASURES),
        expected_rows=900,
    )
    if len(primary) != 1200:
        raise ValueError(f"Expected 1,200 matching primary rows, found {len(primary)}")
    indexed = index_rows(primary + negative_rows)
    stimulus_ids = sorted({row["stimulus_id"] for row in negative_rows})
    if len(stimulus_ids) != 100:
        raise ValueError("Negative sweep must contain 100 unique stimuli")
    primary_ids = {row["stimulus_id"] for row in primary}
    if set(stimulus_ids) != primary_ids:
        raise ValueError("Negative and primary correct-item sets differ")

    result: dict[str, Any] = {"alphas": list(FULL_ALPHAS), "measures": {}}
    for measure in NEGATIVE_MEASURES:
        if measure == "confidence_manipulation_check":
            value = confidence
            expected_direction = "increasing"
        else:
            response = ALARM_RESPONSE[measure]
            value = lambda row, response=response: candidate_log_odds(row, response)
            expected_direction = "decreasing"
        cells = {
            f"alpha={alpha:g}": {
                "mean": mean(
                    value(indexed[(stimulus_id, measure, alpha)])
                    for stimulus_id in stimulus_ids
                )
            }
            for alpha in FULL_ALPHAS
        }
        negative_to_zero = paired_changes(
            indexed, stimulus_ids, measure, -15.0, 0.0, value
        )
        full_change = paired_changes(
            indexed, stimulus_ids, measure, -15.0, 15.0, value
        )
        slopes = [
            slope([
                (alpha, value(indexed[(stimulus_id, measure, alpha)]))
                for alpha in FULL_ALPHAS
            ])
            for stimulus_id in stimulus_ids
        ]
        measure_result: dict[str, Any] = {
            "scale": "expected confidence" if measure == "confidence_manipulation_check" else "exact alarm log-odds",
            "expected_direction": expected_direction,
            "cells": cells,
            "change_0_minus_negative15": summarize(negative_to_zero),
            "change_positive15_minus_negative15": summarize(full_change),
            "linear_slope_per_alpha": summarize(slopes),
        }
        if measure != "confidence_manipulation_check":
            response = ALARM_RESPONSE[measure]
            measure_result["alarm_probability_cells"] = {
                f"alpha={alpha:g}": {
                    "mean": mean(
                        candidate_probability(
                            indexed[(stimulus_id, measure, alpha)], response
                        )
                        for stimulus_id in stimulus_ids
                    )
                }
                for alpha in FULL_ALPHAS
            }
        result["measures"][measure] = measure_result
    return result


def analyze_primary_log_odds(primary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    measures = ("anomaly_forced_choice", "error_detection")
    conditions = ("definite_correct", "definite_false", "ambiguous")
    rows = [
        row for row in primary_rows
        if row["measure"] in measures and float(row["alpha"]) in (0.0, 15.0)
    ]
    validate_rows(
        rows,
        conditions=set(conditions),
        alphas={0.0, 15.0},
        measures=set(measures),
        expected_rows=1200,
    )
    result: dict[str, Any] = {"condition_changes": {}, "contrasts": {}}
    raw: dict[tuple[str, str], list[float]] = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        indexed = index_rows(condition_rows)
        stimulus_ids = sorted({row["stimulus_id"] for row in condition_rows})
        for measure in measures:
            response = ALARM_RESPONSE[measure]
            changes = paired_changes(
                indexed,
                stimulus_ids,
                measure,
                0.0,
                15.0,
                lambda row, response=response: candidate_log_odds(row, response),
            )
            raw[(condition, measure)] = changes
            result["condition_changes"][f"{condition}|{measure}"] = summarize(changes)
    for measure in measures:
        correct = raw[("definite_correct", measure)]
        for condition in ("definite_false", "ambiguous"):
            result["contrasts"][f"{condition}_minus_definite_correct|{measure}"] = (
                summarize_difference(raw[(condition, measure)], correct)
            )
    return result


def analyze_polarity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validate_rows(
        rows,
        conditions={"definite_false"},
        alphas={0.0, 15.0},
        measures=set(POLARITY_MEASURES),
        expected_rows=400,
    )
    indexed = index_rows(rows)
    stimulus_ids = sorted({row["stimulus_id"] for row in rows})
    if len(stimulus_ids) != 100:
        raise ValueError("Polarity control must contain 100 unique stimuli")

    measures: dict[str, Any] = {}
    yes_changes: dict[str, list[float]] = {}
    for measure in POLARITY_MEASURES:
        alarm = ALARM_RESPONSE[measure]
        alarm_probability_changes = paired_changes(
            indexed,
            stimulus_ids,
            measure,
            0.0,
            15.0,
            lambda row, alarm=alarm: candidate_probability(row, alarm),
        )
        alarm_log_odds_changes = paired_changes(
            indexed,
            stimulus_ids,
            measure,
            0.0,
            15.0,
            lambda row, alarm=alarm: candidate_log_odds(row, alarm),
        )
        yes_changes[measure] = paired_changes(
            indexed,
            stimulus_ids,
            measure,
            0.0,
            15.0,
            lambda row: candidate_log_odds(row, "yes"),
        )
        measures[measure] = {
            "alarm_response": alarm,
            "alarm_probability_change_15_minus_0": summarize(alarm_probability_changes),
            "alarm_log_odds_change_15_minus_0": summarize(alarm_log_odds_changes),
            "yes_log_odds_change_15_minus_0": summarize(yes_changes[measure]),
        }

    interaction = [
        consistency - inconsistency
        for consistency, inconsistency in zip(
            yes_changes["consistency_yes_no"],
            yes_changes["inconsistency_yes_no"],
        )
    ]
    return {
        "measures": measures,
        "polarity_interaction_consistency_minus_inconsistency_yes_log_odds": summarize(
            interaction
        ),
        "interpretive_key": {
            "semantic_confidence_prediction": "inconsistency yes-log-odds decrease; consistency yes-log-odds increase",
            "literal_no_bias_prediction": "yes-log-odds decrease for both wordings",
        },
    }


def fmt(value: float) -> str:
    return f"{value:+.3f}"


def fmt_ci(item: dict[str, Any]) -> str:
    return f"[{item['ci95'][0]:+.3f}, {item['ci95'][1]:+.3f}]"


def make_report(payload: dict[str, Any]) -> str:
    primary = payload["primary_log_odds"]
    negative = payload["negative_sweep"]
    polarity = payload["polarity_control"]
    lines = [
        "# Post-hoc confidence-direction control extension",
        "",
        "## Conservative reanalysis of the original primary outcomes",
        "",
        "Changes are exact item-level alarm log-odds at alpha 15 minus alpha 0.",
        "",
        "| Measure | correct change | false change | false - correct | ambiguous change | ambiguous - correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for measure in ("anomaly_forced_choice", "error_detection"):
        correct = primary["condition_changes"][f"definite_correct|{measure}"]
        false = primary["condition_changes"][f"definite_false|{measure}"]
        ambiguous = primary["condition_changes"][f"ambiguous|{measure}"]
        false_contrast = primary["contrasts"][
            f"definite_false_minus_definite_correct|{measure}"
        ]
        ambiguous_contrast = primary["contrasts"][
            f"ambiguous_minus_definite_correct|{measure}"
        ]
        lines.append(
            f"| {measure} | {fmt(correct['mean'])} {fmt_ci(correct)} | "
            f"{fmt(false['mean'])} {fmt_ci(false)} | "
            f"{fmt(false_contrast['mean'])} {fmt_ci(false_contrast)} | "
            f"{fmt(ambiguous['mean'])} {fmt_ci(ambiguous)} | "
            f"{fmt(ambiguous_contrast['mean'])} {fmt_ci(ambiguous_contrast)} |"
        )
    lines.extend([
        "",
        "## Negative-alpha sweep on definite-correct answers",
        "",
        "Cell values for confidence are expected categorical midpoints. Alarm values are exact candidate log-odds,",
        "calculated as the alarm-token logit minus the reassuring-token logit.",
        "",
        "| Measure | alpha -15 | -10 | -5 | 0 | 5 | 10 | 15 | slope per alpha | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for measure in NEGATIVE_MEASURES:
        item = negative["measures"][measure]
        values = [item["cells"][f"alpha={alpha:g}"]["mean"] for alpha in FULL_ALPHAS]
        slope_item = item["linear_slope_per_alpha"]
        lines.append(
            f"| {measure} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + f" | {fmt(slope_item['mean'])} | {fmt_ci(slope_item)} |"
        )
    lines.extend([
        "",
        "## Matched yes/no polarity control on definite-false answers",
        "",
        "| Probe | alarm response | alarm probability change | alarm log-odds change | yes log-odds change |",
        "|---|---|---:|---:|---:|",
    ])
    for measure in POLARITY_MEASURES:
        item = polarity["measures"][measure]
        probability = item["alarm_probability_change_15_minus_0"]
        alarm_odds = item["alarm_log_odds_change_15_minus_0"]
        yes_odds = item["yes_log_odds_change_15_minus_0"]
        lines.append(
            f"| {measure} | {item['alarm_response']} | {fmt(probability['mean'])} "
            f"{fmt_ci(probability)} | {fmt(alarm_odds['mean'])} {fmt_ci(alarm_odds)} | "
            f"{fmt(yes_odds['mean'])} {fmt_ci(yes_odds)} |"
        )
    interaction = polarity[
        "polarity_interaction_consistency_minus_inconsistency_yes_log_odds"
    ]
    lines.extend([
        "",
        "The critical literal-response-bias test is the consistency-minus-inconsistency interaction",
        f"in yes log-odds: {fmt(interaction['mean'])} {fmt_ci(interaction)}.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, required=True)
    parser.add_argument("--polarity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    primary_rows = load_jsonl(args.primary_dir / "measurements.jsonl")
    negative_rows = load_jsonl(args.negative_dir / "measurements.jsonl")
    polarity_rows = load_jsonl(args.polarity_dir / "measurements.jsonl")
    payload = {
        "analysis_version": "panl_control_extension_v1",
        "primary_log_odds": analyze_primary_log_odds(primary_rows),
        "negative_sweep": analyze_negative_sweep(primary_rows, negative_rows),
        "polarity_control": analyze_polarity(polarity_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "control_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (args.output_dir / "control_analysis_report.md").write_text(make_report(payload))
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
