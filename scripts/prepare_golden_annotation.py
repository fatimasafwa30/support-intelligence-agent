"""Prepare annotation-ready CSV from Golden Set candidates.

Annotation Guidelines:
- candidate_intent_hint is only a sampling aid and must NOT be treated as ground truth.
- gold_intent must be assigned manually by reviewing customer text and context.
- gold_risk and gold_action must be assigned manually according to intent policy.
- annotation_notes should explain ambiguous, multi-intent, or difficult decisions.
"""

import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "golden" / "golden_candidates.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "golden" / "golden_annotation.csv"
README_PATH = PROJECT_ROOT / "data" / "golden" / "README.md"

NEW_ANNOTATION_COLUMNS = (
    "gold_intent",
    "gold_risk",
    "gold_action",
    "annotation_notes",
)

README_CONTENT = """# Golden Set Annotation Guide

This directory contains candidates and annotations for the evaluation Golden Set.

## Files
- `golden_candidates.csv`: Sampling candidates generated deterministically from eligible conversations.
- `golden_annotation.csv`: Annotation template for human reviewers.

## Annotation Guidelines & Principles
1. **Sampling Hints vs. Ground Truth**:
   - `candidate_intent_hint` is only a heuristic sampling aid produced by exploratory regexes.
   - It must never be treated as ground truth or copied directly without human verification.
2. **Manual Intent Assignment (`gold_intent`)**:
   - `gold_intent` must be assigned manually based on careful review of the customer's text and conversation context.
   - Assign the single primary intent from the defined intent taxonomy.
3. **Manual Risk & Action (`gold_risk`, `gold_action`)**:
   - `gold_risk` (`low`, `medium`, `high`, `critical`) and `gold_action` (`respond_directly`, `escalate_human`, etc.) must be assigned manually in accordance with brand policy.
4. **Annotation Notes (`annotation_notes`)**:
   - Document the rationale for difficult, low-context, vague, or multi-intent messages.
   - Note secondary intents if present.
"""


def prepare_annotation_file() -> None:
    """Create a separate annotation-ready CSV with empty gold columns."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Source candidate CSV not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        input_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    input_count = len(rows)

    # Verify new columns do not already exist in input
    for col in NEW_ANNOTATION_COLUMNS:
        if col in input_fieldnames:
            raise ValueError(f"Column '{col}' already exists in input CSV.")

    output_fieldnames = input_fieldnames + list(NEW_ANNOTATION_COLUMNS)

    # Populate empty strings for the four annotation columns
    annotation_rows = []
    for row in rows:
        annotated_row = dict(row)
        for col in NEW_ANNOTATION_COLUMNS:
            annotated_row[col] = ""
        annotation_rows.append(annotated_row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(annotation_rows)

    # Write companion README note for annotators
    README_PATH.write_text(README_CONTENT, encoding="utf-8")

    # Immediate verification
    with OUTPUT_PATH.open("r", encoding="utf-8", newline="") as verify_file:
        verify_reader = csv.DictReader(verify_file)
        verify_fieldnames = list(verify_reader.fieldnames or [])
        verify_rows = list(verify_reader)

    output_count = len(verify_rows)

    # Check preserved columns
    for col in input_fieldnames:
        assert col in verify_fieldnames, f"Missing preserved column: {col}"

    # Check new columns
    for col in NEW_ANNOTATION_COLUMNS:
        assert col in verify_fieldnames, f"Missing annotation column: {col}"

    # Check empty values
    for row in verify_rows:
        for col in NEW_ANNOTATION_COLUMNS:
            assert row[col] == "", f"Column {col} is not empty: {row[col]!r}"

    print(f"Successfully created {OUTPUT_PATH.name}")
    print(f"Input rows: {input_count}")
    print(f"Output rows: {output_count}")
    print(f"Preserved columns: {len(input_fieldnames)} columns")
    print(f"New annotation columns: {list(NEW_ANNOTATION_COLUMNS)}")
    print(f"Annotator README written to: {README_PATH}")


if __name__ == "__main__":
    prepare_annotation_file()
