#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CONDITIONS = ("definite_correct", "definite_false", "ambiguous")
MEASURES = (
    "confidence_manipulation_check",
    "anomaly_forced_choice",
    "anomaly_continuous",
    "error_detection",
    "wager_abstention",
)
TARGET = {
    "anomaly_forced_choice": "inconsistent",
    "error_detection": "incorrect",
    "wager_abstention": "abstain",
}
LABEL = {
    "confidence_manipulation_check": "Expected confidence",
    "anomaly_forced_choice": "P(inconsistent)",
    "anomaly_continuous": "Unusualness score",
    "error_detection": "P(incorrect)",
    "wager_abstention": "P(abstain)",
}
MIDPOINT = {
    "No chance": 0.05,
    "Really unlikely": 0.15,
    "Chances are slight": 0.25,
    "Unlikely": 0.35,
    "Less than even": 0.45,
    "Better than even": 0.55,
    "Likely": 0.65,
    "Very good chance": 0.75,
    "Highly likely": 0.85,
    "Almost certain": 0.95,
}


def mean(values: Iterable[float]) -> float:
    return statistics.mean(values)


def scalar(row: dict[str, Any]) -> float:
    measure = row["measure"]
    if measure in TARGET:
        return float(row["response_probabilities"][TARGET[measure]])
    if measure == "confidence_manipulation_check":
        return float(row["confidence_expected_midpoint"])
    if measure == "anomaly_continuous":
        return float(row["parsed_score"]) / 100.0
    raise ValueError(f"Unknown measure: {measure}")


def ci_mean(values: list[float]) -> list[float]:
    """Normal-approximation 95% CI over independent item-level quantities."""
    center = mean(values)
    if len(values) < 2:
        return [center, center]
    se = statistics.stdev(values) / math.sqrt(len(values))
    return [center - 1.96 * se, center + 1.96 * se]


def ci_difference(first: list[float], second: list[float]) -> list[float]:
    center = mean(first) - mean(second)
    se = math.sqrt(
        statistics.variance(first) / len(first)
        + statistics.variance(second) / len(second)
    )
    return [center - 1.96 * se, center + 1.96 * se]


def slope(points: dict[float, float], alphas: list[float]) -> float:
    xs = alphas
    ys = [points[alpha] for alpha in xs]
    xbar, ybar = mean(xs), mean(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum(
        (x - xbar) ** 2 for x in xs
    )


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt_signed(value: float) -> str:
    return f"{value:+.3f}"


def fmt_ci(interval: list[float]) -> str:
    return f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(
    rows: list[dict[str, Any]], manifest: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    expected = int(manifest["expected_rows"])
    trial_ids = [row["trial_id"] for row in rows]
    signatures = {row["run_signature"] for row in rows}
    cell_counts = Counter(
        (row["condition"], float(row["alpha"]), row["measure"]) for row in rows
    )
    expected_cells = {
        (condition, alpha, measure)
        for condition in CONDITIONS
        for alpha in map(float, manifest["alphas"])
        for measure in MEASURES
    }
    binary_rows = [row for row in rows if row["measure"] in TARGET]
    categorical_rows = [
        row for row in rows if row["measure"] == "confidence_manipulation_check"
    ]
    continuous_rows = [row for row in rows if row["measure"] == "anomaly_continuous"]

    mapping_consistent = True
    mappings: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in binary_rows:
        mappings[(row["stimulus_id"], row["measure"])].add(
            json.dumps(row["label_to_response"], sort_keys=True)
        )
    if any(len(values) != 1 for values in mappings.values()):
        mapping_consistent = False

    checks = {
        "row_count": len(rows) == expected == int(summary["completed_rows"]),
        "unique_trial_ids": len(trial_ids) == len(set(trial_ids)),
        "one_run_signature": len(signatures) == 1,
        "signature_matches_manifest": signatures == {manifest["run_signature"]},
        "all_expected_cells_present": set(cell_counts) == expected_cells,
        "one_hundred_rows_per_cell": set(cell_counts.values()) == {100},
        "panl_token_is_198": {tuple(row["panl_token_ids"]) for row in rows} == {(198,)},
        "alpha_zero_has_no_hook": all(
            row["intervention_application_count"] == 0 and not row["intervention_applied"]
            for row in rows
            if float(row["alpha"]) == 0.0
        ),
        "positive_alpha_has_one_hook": all(
            row["intervention_application_count"] == 1 and row["intervention_applied"]
            for row in rows
            if float(row["alpha"]) > 0.0
        ),
        "all_binary_and_confidence_argmaxes_valid": all(
            row["global_argmax_is_candidate"] for row in binary_rows + categorical_rows
        ),
        "all_continuous_scores_valid": all(
            row["parse_valid"] and 0 <= int(row["parsed_score"]) <= 100
            for row in continuous_rows
        ),
        "binary_probabilities_sum_to_one": all(
            abs(sum(row["response_probabilities"].values()) - 1.0) < 1e-6
            for row in binary_rows
        ),
        "response_mapping_fixed_within_item": mapping_consistent,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Integrity checks failed: {failed}")
    return {
        "checks": checks,
        "rows": len(rows),
        "cells": len(cell_counts),
        "run_signature": next(iter(signatures)),
        "continuous_parse_failures": 0,
    }


def analyze(rows: list[dict[str, Any]], alphas: list[float]) -> dict[str, Any]:
    by_item: dict[tuple[str, str, str], dict[float, float]] = defaultdict(dict)
    by_cell: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    hard_by_item: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    class_by_item: dict[tuple[str, str], dict[float, str]] = defaultdict(dict)
    target_label_by_item: dict[tuple[str, str, str], str] = {}

    for row in rows:
        condition = row["condition"]
        measure = row["measure"]
        alpha = float(row["alpha"])
        value = scalar(row)
        by_item[(condition, row["stimulus_id"], measure)][alpha] = value
        by_cell[(condition, measure, alpha)].append(value)
        if measure in TARGET:
            target_label_by_item[(condition, row["stimulus_id"], measure)] = next(
                label
                for label, semantic in row["label_to_response"].items()
                if semantic == TARGET[measure]
            )
        if measure == "confidence_manipulation_check":
            key = (condition, row["stimulus_id"])
            class_name = row["selected_confidence_class"]
            hard_by_item[key][alpha] = MIDPOINT[class_name]
            class_by_item[key][alpha] = class_name

    cells: dict[str, Any] = {}
    paired_deltas: dict[str, Any] = {}
    slopes: dict[str, Any] = {}
    raw_slopes: dict[tuple[str, str], list[float]] = defaultdict(list)
    raw_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)

    for condition in CONDITIONS:
        for measure in MEASURES:
            for alpha in alphas:
                values = by_cell[(condition, measure, alpha)]
                cells[f"{condition}|{measure}|alpha={alpha:g}"] = {
                    "n": len(values),
                    "mean": mean(values),
                    "sd": statistics.stdev(values),
                }
            item_points = [
                points
                for (item_condition, _stimulus_id, item_measure), points in by_item.items()
                if item_condition == condition and item_measure == measure
            ]
            deltas = [points[alphas[-1]] - points[alphas[0]] for points in item_points]
            item_slopes = [slope(points, alphas) for points in item_points]
            raw_deltas[(condition, measure)] = deltas
            raw_slopes[(condition, measure)] = item_slopes
            paired_deltas[f"{condition}|{measure}"] = {
                "n": len(deltas),
                "mean": mean(deltas),
                "ci95": ci_mean(deltas),
                "positive_items": sum(value > 0 for value in deltas),
                "negative_items": sum(value < 0 for value in deltas),
                "unchanged_items": sum(value == 0 for value in deltas),
            }
            slopes[f"{condition}|{measure}"] = {
                "n": len(item_slopes),
                "mean_per_alpha_unit": mean(item_slopes),
                "mean_per_five_alpha": 5 * mean(item_slopes),
                "ci95_per_alpha_unit": ci_mean(item_slopes),
                "ci95_per_five_alpha": [5 * value for value in ci_mean(item_slopes)],
            }

    contrasts: dict[str, Any] = {}
    for measure in MEASURES:
        for condition in ("definite_false", "ambiguous"):
            item_slopes = raw_slopes[(condition, measure)]
            control_slopes = raw_slopes[("definite_correct", measure)]
            delta = mean(item_slopes) - mean(control_slopes)
            interval = ci_difference(item_slopes, control_slopes)
            contrasts[f"{condition}_minus_definite_correct|{measure}"] = {
                "difference_per_alpha_unit": delta,
                "difference_per_five_alpha": 5 * delta,
                "ci95_per_alpha_unit": interval,
                "ci95_per_five_alpha": [5 * value for value in interval],
            }

    hard_confidence: dict[str, Any] = {}
    for condition in CONDITIONS:
        keys = [key for key in hard_by_item if key[0] == condition]
        for alpha in alphas:
            values = [hard_by_item[key][alpha] for key in keys]
            hard_confidence[f"{condition}|alpha={alpha:g}"] = {
                "n": len(values),
                "mean": mean(values),
            }
        deltas = [hard_by_item[key][alphas[-1]] - hard_by_item[key][alphas[0]] for key in keys]
        hard_confidence[f"{condition}|alpha={alphas[-1]:g}_minus_{alphas[0]:g}"] = {
            "n": len(deltas),
            "mean": mean(deltas),
            "ci95": ci_mean(deltas),
            "class_changed": sum(
                class_by_item[key][alphas[-1]] != class_by_item[key][alphas[0]]
                for key in keys
            ),
        }

    label_order_audit: dict[str, Any] = {}
    for condition in CONDITIONS:
        for measure in TARGET:
            key = (condition, measure)
            item_deltas = {label: [] for label in ("A", "B")}
            for (item_condition, stimulus_id, item_measure), points in by_item.items():
                if item_condition != condition or item_measure != measure:
                    continue
                label = target_label_by_item[(condition, stimulus_id, measure)]
                item_deltas[label].append(points[alphas[-1]] - points[alphas[0]])
            label_order_audit[f"{condition}|{measure}"] = {
                label: {
                    "n": len(values),
                    "mean_alpha15_minus_alpha0": mean(values),
                    "ci95": ci_mean(values),
                }
                for label, values in item_deltas.items()
            }

    return {
        "alphas": alphas,
        "cell_summaries": cells,
        "paired_alpha15_minus_alpha0": paired_deltas,
        "within_item_dose_slopes": slopes,
        "dose_slope_contrasts": contrasts,
        "kumaran_style_hard_confidence": hard_confidence,
        "binary_label_order_audit": label_order_audit,
    }


def make_report(integrity: dict[str, Any], result: dict[str, Any]) -> str:
    alphas = result["alphas"]
    cells = result["cell_summaries"]
    deltas = result["paired_alpha15_minus_alpha0"]
    slopes = result["within_item_dose_slopes"]
    contrasts = result["dose_slope_contrasts"]
    hard = result["kumaran_style_hard_confidence"]
    label_audit = result["binary_label_order_audit"]
    lines = [
        "# Qwen PANL confidence-introspection experiment: full results",
        "",
        "## Integrity",
        "",
        f"All integrity checks passed: {integrity['rows']:,} unique measurement rows, "
        f"{integrity['cells']} complete cells, 100 observations per cell, no parse failures, "
        "PANL token 198 throughout, and exactly one intervention application at every positive alpha.",
        "",
        "All outcomes below are oriented so larger values mean more confidence, more perceived "
        "inconsistency/unusualness, more error detection, or more abstention. Confidence intervals "
        "are unadjusted 95% normal-approximation intervals over independent item-level paired "
        "quantities; error detection and abstention are secondary outcomes.",
        "",
        "## Cell means",
        "",
    ]
    for measure in MEASURES:
        lines.extend([
            f"### {LABEL[measure]}",
            "",
            "| Condition | alpha 0 | alpha 5 | alpha 10 | alpha 15 | paired change 15-0 | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for condition in CONDITIONS:
            values = [
                cells[f"{condition}|{measure}|alpha={alpha:g}"]["mean"] for alpha in alphas
            ]
            delta = deltas[f"{condition}|{measure}"]
            lines.append(
                f"| {condition.replace('_', ' ')} | "
                + " | ".join(fmt(value) for value in values)
                + f" | {fmt_signed(delta['mean'])} | {fmt_ci(delta['ci95'])} |"
            )
        lines.append("")

    lines.extend([
        "## Preregistered dose-response contrasts",
        "",
        "The table reports how much more or less the outcome changed per +5 alpha in each "
        "epistemically problematic condition relative to definite-correct controls.",
        "",
        "| Outcome | Contrast | difference per +5 alpha | 95% CI |",
        "|---|---|---:|---:|",
    ])
    for measure in MEASURES:
        for condition in ("definite_false", "ambiguous"):
            item = contrasts[f"{condition}_minus_definite_correct|{measure}"]
            lines.append(
                f"| {LABEL[measure]} | {condition.replace('_', ' ')} - definite correct | "
                f"{fmt_signed(item['difference_per_five_alpha'])} | "
                f"{fmt_ci(item['ci95_per_five_alpha'])} |"
            )

    lines.extend([
        "",
        "## Kumaran-style hard confidence classes",
        "",
        "| Condition | hard midpoint alpha 0 | hard midpoint alpha 15 | paired change | 95% CI | class changed |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        base = hard[f"{condition}|alpha={alphas[0]:g}"]["mean"]
        steered = hard[f"{condition}|alpha={alphas[-1]:g}"]["mean"]
        delta = hard[f"{condition}|alpha={alphas[-1]:g}_minus_{alphas[0]:g}"]
        lines.append(
            f"| {condition.replace('_', ' ')} | {fmt(base)} | {fmt(steered)} | "
            f"{fmt_signed(delta['mean'])} | {fmt_ci(delta['ci95'])} | "
            f"{delta['class_changed']}/100 |"
        )

    lines.extend([
        "",
        "## Binary label-order audit",
        "",
        "Binary response order was fixed within item across alpha. The table stratifies the paired "
        "alpha-15-minus-alpha-0 change by whether the target response appeared as A or B.",
        "",
        "| Outcome | Condition | target=A | target=B |",
        "|---|---|---:|---:|",
    ])
    for measure in TARGET:
        for condition in CONDITIONS:
            audit = label_audit[f"{condition}|{measure}"]
            lines.append(
                f"| {LABEL[measure]} | {condition.replace('_', ' ')} | "
                f"{fmt_signed(audit['A']['mean_alpha15_minus_alpha0'])} (n={audit['A']['n']}) | "
                f"{fmt_signed(audit['B']['mean_alpha15_minus_alpha0'])} (n={audit['B']['n']}) |"
            )

    fc_false = contrasts["definite_false_minus_definite_correct|anomaly_forced_choice"]
    fc_amb = contrasts["ambiguous_minus_definite_correct|anomaly_forced_choice"]
    continuous_false = contrasts["definite_false_minus_definite_correct|anomaly_continuous"]
    continuous_amb = contrasts["ambiguous_minus_definite_correct|anomaly_continuous"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The confidence manipulation succeeded. It was strongest for definite-false answers, "
        "moderate for ambiguous answers, and small for already-high-confidence correct answers.",
        "",
        "The primary provenance-anomaly prediction was not supported. Increasing the high-confidence "
        "PANL direction reduced rather than increased forced-choice inconsistency reports. Relative "
        f"to definite-correct controls, the per-+5-alpha slope difference was "
        f"{fmt_signed(fc_false['difference_per_five_alpha'])} for definite-false items and "
        f"{fmt_signed(fc_amb['difference_per_five_alpha'])} for ambiguous items. Both confidence "
        "intervals excluded zero in the negative direction.",
        "",
        "The continuous unusualness measure converged for definite-false answers: its dose response "
        f"was {fmt_signed(continuous_false['difference_per_five_alpha'])} per +5 alpha relative to "
        "correct controls. The ambiguous-minus-correct continuous contrast was "
        f"{fmt_signed(continuous_amb['difference_per_five_alpha'])}, with a confidence interval "
        "that included zero; it therefore supplied no evidence for an ambiguity-specific anomaly increase.",
        "",
        "Error detection and abstention moved in the same direction: stronger injected confidence "
        "made the model less likely to call the answer incorrect and less likely to abstain. Yet at "
        "alpha 15, false and ambiguous answers still had higher absolute anomaly, error, and abstention "
        "scores than correct controls. The model retained sensitivity to question-answer fit, but the "
        "injected confidence suppressed that sensitivity instead of becoming an independently detected anomaly.",
        "",
        "The most direct interpretation is that Qwen reads the steered confidence state as evidence "
        "about answer quality; it does not show behavioral access to the intervention's provenance. "
        "A narrower mechanistic alternative remains: the PANL direction may directly bias several "
        "downstream metacognitive reports, not only a unitary subjective confidence state. Control-position "
        "and unrelated-direction interventions would distinguish these explanations.",
        "",
        "The primary forced-choice anomaly conclusion survived both A/B response orders: false and "
        "ambiguous anomaly reports decreased in both strata. The abstention effect was notably "
        "label-order sensitive, especially in the target=A stratum, so abstention should be treated "
        "as weaker secondary evidence rather than a result of the same strength as anomaly or error detection.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    rows = load_jsonl(run_dir / "measurements.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    integrity = validate(rows, manifest, summary)
    result = analyze(rows, [float(value) for value in manifest["alphas"]])
    payload = {"integrity": integrity, **result}
    (run_dir / "full_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    (run_dir / "full_analysis_report.md").write_text(make_report(integrity, result))
    print(json.dumps({
        "status": "ok",
        "rows": integrity["rows"],
        "report": str(run_dir / "full_analysis_report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
