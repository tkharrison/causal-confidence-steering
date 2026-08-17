#!/usr/bin/env python3
"""Reproduce the experimental schematic and primary dose-response figure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "definite_correct": "#2878B5",
    "definite_false": "#D64541",
    "ambiguous": "#E49D26",
}
LABELS = {
    "definite_correct": "Correct",
    "definite_false": "False",
    "ambiguous": "Ambiguous",
}


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def center_text(draw, box, text, selected_font, fill="#20252A", spacing=5):
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox(
        (0, 0), text, font=selected_font, spacing=spacing, align="center"
    )
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(
        ((x0 + x1 - width) / 2, (y0 + y1 - height) / 2),
        text,
        font=selected_font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def design_figure(path: Path) -> None:
    image = Image.new("RGB", (1944, 564), "white")
    draw = ImageDraw.Draw(image)
    title_font, box_font, small_font = font(25, True), font(27), font(22)
    center_text(
        draw,
        (0, 5, 1944, 70),
        "CAUSAL INTERVENTION AT THE POST-ANSWER NEWLINE",
        title_font,
        "#343A40",
    )
    boxes = [
        (20, 170, 398, 430, "Question + fixed answer\n(exact replay)", "#EEF4FA", "#2878B5"),
        (520, 170, 898, 430, "PANL residual\nlayer 15", "#F4F0FA", "#7656A8"),
        (1020, 170, 1388, 430, "h' = h + alpha v_conf\nalpha = 0, 5, 10, 15", "#FFF4E2", "#E49D26"),
        (1510, 170, 1924, 430, "Independent probes\n(no shared context)", "#FCEEEE", "#D64541"),
    ]
    for x0, y0, x1, y1, text, fill, edge in boxes:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=24, fill=fill, outline=edge, width=5)
        center_text(draw, (x0, y0, x1, y1), text, box_font)
    for start, end in ((405, 510), (905, 1010), (1395, 1500)):
        y = 300
        draw.line((start, y, end, y), fill="#555B61", width=5)
        draw.polygon([(end, y), (end - 24, y - 14), (end - 24, y + 14)], fill="#555B61")
    center_text(
        draw,
        (1450, 445, 1944, 555),
        "confidence | inconsistency | unusualness\nerror detection | abstention",
        small_font,
        "#444A50",
    )
    image.save(path, dpi=(240, 240))


def dose_response_figure(path: Path, analysis: dict) -> None:
    metrics = [
        ("confidence_manipulation_check", "Expected confidence"),
        ("anomaly_forced_choice", "P(inconsistent)"),
        ("error_detection", "P(answer incorrect)"),
    ]
    alphas = [0, 5, 10, 15]
    width, height = 2214, 702
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font, legend_font = (
        font(27, True),
        font(22),
        font(20),
        font(22),
    )
    panel_width, gap = 650, 65
    first_left, top, bottom = 85, 70, 545
    for panel, (metric, title) in enumerate(metrics):
        left = first_left + panel * (panel_width + gap)
        right = left + panel_width
        center_text(draw, (left, 5, right, 58), title, title_font, "#22272B")
        for value in (0, 0.25, 0.5, 0.75, 1):
            y = bottom - int(value * (bottom - top))
            draw.line((left, y, right, y), fill="#DADDE0", width=2)
            label = f"{value:g}"
            bounds = draw.textbbox((0, 0), label, font=tick_font)
            draw.text(
                (left - 12 - (bounds[2] - bounds[0]), y - (bounds[3] - bounds[1]) / 2),
                label,
                font=tick_font,
                fill="#454B50",
            )
        draw.line((left, top, left, bottom), fill="#444A50", width=3)
        draw.line((left, bottom, right, bottom), fill="#444A50", width=3)
        xs = [left + int((alpha / 15) * panel_width) for alpha in alphas]
        for x, alpha in zip(xs, alphas):
            draw.line((x, bottom, x, bottom + 8), fill="#444A50", width=2)
            label = str(alpha)
            bounds = draw.textbbox((0, 0), label, font=tick_font)
            draw.text(
                (x - (bounds[2] - bounds[0]) / 2, bottom + 13),
                label,
                font=tick_font,
                fill="#454B50",
            )
        for condition in ("definite_correct", "definite_false", "ambiguous"):
            points = []
            for x, alpha in zip(xs, alphas):
                cell = analysis["cell_summaries"][f"{condition}|{metric}|alpha={alpha}"]
                mean = float(cell["mean"])
                sem = float(cell["sd"]) / math.sqrt(int(cell["n"]))
                y = bottom - int(mean * (bottom - top))
                y0 = bottom - int(min(1, mean + sem) * (bottom - top))
                y1 = bottom - int(max(0, mean - sem) * (bottom - top))
                points.append((x, y))
                draw.line((x, y0, x, y1), fill=COLORS[condition], width=3)
                draw.line((x - 8, y0, x + 8, y0), fill=COLORS[condition], width=3)
                draw.line((x - 8, y1, x + 8, y1), fill=COLORS[condition], width=3)
            draw.line(points, fill=COLORS[condition], width=6, joint="curve")
            for x, y in points:
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=COLORS[condition], outline="white", width=2)
        center_text(
            draw,
            (left, bottom + 46, right, bottom + 95),
            "Steering strength alpha",
            label_font,
            "#33383C",
        )
    legend_y = 660
    entries = [(LABELS[key], COLORS[key]) for key in COLORS]
    total = sum(
        54 + draw.textbbox((0, 0), label, font=legend_font)[2] + 55
        for label, _ in entries
    )
    x = (width - total) / 2
    for label, color in entries:
        draw.line((x, legend_y, x + 42, legend_y), fill=color, width=7)
        draw.ellipse((x + 13, legend_y - 8, x + 29, legend_y + 8), fill=color)
        x += 54
        draw.text((x, legend_y - 14), label, font=legend_font, fill="#33383C")
        x += draw.textbbox((0, 0), label, font=legend_font)[2] + 55
    image.save(path, dpi=(260, 260))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        default="results/main/full_analysis.json",
        help="Full analysis JSON generated by analyze_full_introspection_results.py",
    )
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(Path(args.analysis).read_text())
    design_figure(output_dir / "figure-1-design.png")
    dose_response_figure(output_dir / "figure-2-dose-response.png", analysis)


if __name__ == "__main__":
    main()
