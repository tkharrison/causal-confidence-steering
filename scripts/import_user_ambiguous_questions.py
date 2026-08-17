#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXCLUSIONS = {
    3: "Exact overlap with the existing James Bond ambiguous stimulus.",
    51: "Exact overlap with the existing UK prime-minister ambiguous stimulus.",
}

# Replace transition years or repair a localized factual/wording defect while
# preserving the user's question architecture and intended four mappings.
CORRECTIONS = {
    36: {
        "resolutions": [
            {"condition": "on 31 December 1968", "answer": "Jim Hines"},
            {"condition": "on 31 August 1991", "answer": "Carl Lewis"},
            {"condition": "on 1 August 1996", "answer": "Donovan Bailey"},
            {"condition": "on 17 August 2009", "answer": "Usain Bolt"},
        ],
    },
    52: {"resolutions": [
        {"condition": "1985", "answer": "Francois Mitterrand"},
        {"condition": "2000", "answer": "Jacques Chirac"},
        {"condition": "2010", "answer": "Nicolas Sarkozy"},
        {"condition": "2015", "answer": "Francois Hollande"},
    ]},
    53: {"resolutions": [
        {"condition": "1972", "answer": "Willy Brandt"},
        {"condition": "1990", "answer": "Helmut Kohl"},
        {"condition": "2000", "answer": "Gerhard Schroder"},
        {"condition": "2010", "answer": "Angela Merkel"},
    ]},
    54: {"resolutions": [
        {"condition": "1975", "answer": "Kurt Waldheim"},
        {"condition": "1985", "answer": "Javier Perez de Cuellar"},
        {"condition": "2000", "answer": "Kofi Annan"},
        {"condition": "2010", "answer": "Ban Ki-moon"},
    ]},
    55: {"resolutions": [
        {"condition": "1960", "answer": "John XXIII"},
        {"condition": "1970", "answer": "Paul VI"},
        {"condition": "1980", "answer": "John Paul II"},
        {"condition": "2010", "answer": "Benedict XVI"},
    ]},
    56: {"resolutions": [
        {"condition": "1950", "answer": "Jawaharlal Nehru"},
        {"condition": "1970", "answer": "Indira Gandhi"},
        {"condition": "1987", "answer": "Rajiv Gandhi"},
        {"condition": "2006", "answer": "Manmohan Singh"},
    ]},
    57: {"resolutions": [
        {"condition": "1996", "answer": "Nelson Mandela"},
        {"condition": "2002", "answer": "Thabo Mbeki"},
        {"condition": "2010", "answer": "Jacob Zuma"},
        {"condition": "2019", "answer": "Cyril Ramaphosa"},
    ]},
    58: {"resolutions": [
        {"condition": "1960", "answer": "Earl Warren"},
        {"condition": "1975", "answer": "Warren Burger"},
        {"condition": "1990", "answer": "William Rehnquist"},
        {"condition": "2010", "answer": "John Roberts"},
    ]},
    59: {"resolutions": [
        {"condition": "1982", "answer": "Paul Volcker"},
        {"condition": "1990", "answer": "Alan Greenspan"},
        {"condition": "2010", "answer": "Ben Bernanke"},
        {"condition": "2016", "answer": "Janet Yellen"},
    ]},
    60: {"resolutions": [
        {"condition": "1975", "answer": "Henry Kissinger"},
        {"condition": "1999", "answer": "Madeleine Albright"},
        {"condition": "2007", "answer": "Condoleezza Rice"},
        {"condition": "2011", "answer": "Hillary Clinton"},
    ]},
    61: {"resolutions": [
        {"condition": "1840", "answer": "Victoria"},
        {"condition": "1905", "answer": "Edward VII"},
        {"condition": "1940", "answer": "George VI"},
        {"condition": "1960", "answer": "Elizabeth II"},
    ]},
    63: {"resolutions": [
        {"condition": "1520", "answer": "Francis I"},
        {"condition": "1600", "answer": "Henry IV"},
        {"condition": "1660", "answer": "Louis XIV"},
        {"condition": "1780", "answer": "Louis XVI"},
    ]},
    71: {
        "question": "What is the largest metropolitan area by population in the country?",
        "missing_information": "Which country.",
    },
    73: {"question": "Which river is commonly listed as the longest on the continent?"},
    76: {
        "question": "What was the busiest airport by passenger traffic in the country in 2019?",
        "missing_information": "Which country.",
    },
    77: {"missing_information": "Which region is being considered."},
    80: {
        "resolutions": [
            {"condition": "Israel", "answer": "the Knesset"},
            {"condition": "Japan", "answer": "the National Diet"},
            {"condition": "Germany", "answer": "the Bundestag and Bundesrat"},
            {"condition": "Norway", "answer": "the Storting"},
        ],
        "options": ["the Knesset", "the National Diet", "the Bundestag and Bundesrat", "the Storting"],
    },
    82: {"missing_information": "Which body and what is meant by its surface."},
    87: {
        "resolutions": [
            {"condition": "Gregorian common year", "answer": "365"},
            {"condition": "Gregorian leap year", "answer": "366"},
            {"condition": "Islamic lunar common year", "answer": "354"},
            {"condition": "average Julian year", "answer": "365.25"},
        ],
    },
    89: {
        "question": "Which NASA spacecraft or rover landed on Mars?",
        "missing_information": "Which landing date.",
        "resolutions": [
            {"condition": "20 July 1976", "answer": "Viking 1"},
            {"condition": "4 July 1997", "answer": "Mars Pathfinder"},
            {"condition": "4 January 2004", "answer": "Spirit"},
            {"condition": "6 August 2012", "answer": "Curiosity"},
        ],
        "options": ["Viking 1", "Mars Pathfinder", "Spirit", "Curiosity"],
    },
    90: {"missing_information": "Which country or jurisdiction."},
    92: {
        "missing_information": "Which country, under 2024 rules.",
    },
}


def context_note(fragment: str) -> str:
    text = " ".join(fragment.strip().rstrip(".").split())
    return f"No information is provided about {text[0].lower() + text[1:]}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text())
    output = []
    for original in source["questions"]:
        item_id = int(original["id"])
        if item_id in EXCLUSIONS:
            continue
        row = dict(original)
        row.update(CORRECTIONS.get(item_id, {}))
        note = context_note(row["missing_information"])
        output.append({
            "id": f"user_ambiguous_{item_id:03d}",
            "source": "user_authored_ambiguous_questions",
            "source_split": "user_generated",
            "source_id": item_id,
            "question_type": "ambiguous",
            "category": row["category"],
            "question": row["question"],
            "original_question": row["question"],
            "missing_qualifier": row["missing_information"],
            "context_note": note,
            "display_question": f"{row['question']}\n\n{note}",
            "interpretations": [
                {"interpretation": f"{row['question']} [Condition: {resolution['condition']}]",
                 "answer": resolution["answer"]}
                for resolution in row["resolutions"]
            ],
            "substantive_options": row["options"],
            "user_correct_answer": row.get("correct_answer"),
            "local_correction_applied": item_id in CORRECTIONS,
        })
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8"
    )
    report = {
        "input": len(source["questions"]), "excluded": len(EXCLUSIONS),
        "corrected": sum(row["local_correction_applied"] for row in output),
        "output": len(output), "exclusions": EXCLUSIONS,
        "corrected_source_ids": sorted(CORRECTIONS), "output_file": args.output,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
