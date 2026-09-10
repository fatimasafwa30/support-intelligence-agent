"""Audit high-precision SilverLabeler against the Golden Set.

Runs the deterministic SilverLabeler against manually annotated Golden Set messages,
evaluates precision, recall, F1, coverage, and confusion matrix, and prints a comprehensive
audit report. Does NOT modify or train on any Golden Set data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys
from typing import Any

# Ensure project root is available on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intents.silver_labeler import CANONICAL_INTENTS, SilverLabeler

GOLDEN_ANNOTATION_PATH = PROJECT_ROOT / "data" / "golden" / "golden_annotation.csv"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "silver_labeler_audit.txt"


def calculate_metrics(
    gold_labels: list[str],
    pred_labels: list[str],
    categories: tuple[str, ...],
) -> dict[str, Any]:
    """Calculate comprehensive classification metrics and confusion matrix."""
    total = len(gold_labels)
    if total == 0:
        raise ValueError("Empty labels list provided to calculate_metrics.")

    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    overall_accuracy = correct / total

    # Coverage: proportion of examples predicted as a specific (non-other_unclear) intent
    covered_indices = [i for i, p in enumerate(pred_labels) if p != "other_unclear"]
    coverage = len(covered_indices) / total
    covered_correct = sum(1 for i in covered_indices if gold_labels[i] == pred_labels[i])
    covered_accuracy = covered_correct / len(covered_indices) if covered_indices else 0.0

    # Per-intent metrics & Confusion Matrix
    confusion: dict[str, dict[str, int]] = {
        g: {p: 0 for p in categories} for g in categories
    }
    for g, p in zip(gold_labels, pred_labels):
        if g in confusion and p in confusion[g]:
            confusion[g][p] += 1

    per_intent: dict[str, dict[str, float | int]] = {}
    macro_p_sum = 0.0
    macro_r_sum = 0.0
    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0

    for cat in categories:
        tp = confusion[cat][cat]
        fp = sum(confusion[other][cat] for other in categories if other != cat)
        fn = sum(confusion[cat][other] for other in categories if other != cat)
        gold_count = tp + fn
        pred_count = tp + fp

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / gold_count if gold_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_intent[cat] = {
            "gold_count": gold_count,
            "pred_count": pred_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        macro_p_sum += precision
        macro_r_sum += recall
        macro_f1_sum += f1
        weighted_f1_sum += f1 * (gold_count / total)

    n_cats = len(categories)
    macro_precision = macro_p_sum / n_cats
    macro_recall = macro_r_sum / n_cats
    macro_f1 = macro_f1_sum / n_cats

    return {
        "total_samples": total,
        "correct_predictions": correct,
        "overall_accuracy": overall_accuracy,
        "coverage": coverage,
        "covered_accuracy": covered_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1_sum,
        "per_intent": per_intent,
        "confusion_matrix": confusion,
    }


def format_report(metrics: dict[str, Any], categories: tuple[str, ...]) -> str:
    """Format audit results into a detailed Markdown report."""
    lines: list[str] = [
        "=" * 80,
        "SILVER LABELER AUDIT REPORT (GOLDEN SET BENCHMARK)",
        "=" * 80,
        "",
        "## Overall Performance",
        f"- Total Golden Set Samples:  {metrics['total_samples']:,}",
        f"- Correct Predictions:       {metrics['correct_predictions']:,}",
        f"- Overall Accuracy:          {metrics['overall_accuracy']:.2%}",
        f"- Specific Coverage:         {metrics['coverage']:.2%} (predicted != other_unclear)",
        f"- Covered Subset Accuracy:   {metrics['covered_accuracy']:.2%}",
        f"- Macro Precision:           {metrics['macro_precision']:.4f}",
        f"- Macro Recall:              {metrics['macro_recall']:.4f}",
        f"- Macro F1-Score:            {metrics['macro_f1']:.4f}",
        f"- Weighted F1-Score:         {metrics['weighted_f1']:.4f}",
        "",
        "## Per-Intent Performance Metrics",
        f"{'Intent':<22} | {'Gold':>5} | {'Pred':>5} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9}",
        "-" * 95,
    ]

    for cat in categories:
        data = metrics["per_intent"][cat]
        lines.append(
            f"{cat:<22} | {data['gold_count']:>5} | {data['pred_count']:>5} | "
            f"{data['tp']:>4} | {data['fp']:>4} | {data['fn']:>4} | "
            f"{data['precision']:>8.2%} | {data['recall']:>8.2%} | {data['f1']:>8.2%}"
        )

    lines.extend([
        "-" * 95,
        "",
        "## Confusion Matrix (Rows = Ground Truth, Columns = Predicted)",
    ])

    # Abbreviated header for readable matrix
    header_cols = [cat[:8] for cat in categories]
    matrix_header = f"{'Gold Intent':<22} | " + " ".join(f"{col:>8}" for col in header_cols)
    lines.append(matrix_header)
    lines.append("-" * len(matrix_header))

    confusion = metrics["confusion_matrix"]
    for gold_cat in categories:
        row_vals = [f"{confusion[gold_cat][pred_cat]:>8}" for pred_cat in categories]
        lines.append(f"{gold_cat:<22} | " + " ".join(row_vals))

    lines.extend([
        "-" * len(matrix_header),
        "",
        "## Top Misclassifications (Gold != Pred)",
    ])

    misclass_counts: Counter[tuple[str, str]] = Counter()
    for gold_cat in categories:
        for pred_cat in categories:
            if gold_cat != pred_cat and confusion[gold_cat][pred_cat] > 0:
                misclass_counts[(gold_cat, pred_cat)] = confusion[gold_cat][pred_cat]

    for (gold_cat, pred_cat), count in misclass_counts.most_common(12):
        lines.append(f"- Gold: '{gold_cat}' -> Pred: '{pred_cat}': {count} cases")

    lines.extend([
        "",
        "## Audit Integrity & Non-Contamination",
        "- The Golden Set was accessed read-only for evaluation purposes.",
        "- Zero modifications were made to data/golden/ files.",
        "- This audit serves as a baseline calibration for silver labeling.",
        "=" * 80,
    ])

    return "\n".join(lines)


def run_audit() -> dict[str, Any]:
    """Execute the full audit on the Golden Set."""
    if not GOLDEN_ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"Golden annotation file not found: {GOLDEN_ANNOTATION_PATH}")

    labeler = SilverLabeler()
    gold_labels: list[str] = []
    pred_labels: list[str] = []
    sample_records: list[dict[str, Any]] = []

    with GOLDEN_ANNOTATION_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            text = row.get("text", "")
            gold_intent = row.get("gold_intent", "").strip()
            if not gold_intent:
                raise ValueError(f"Missing gold_intent on row {row_idx} of {GOLDEN_ANNOTATION_PATH}")
            if gold_intent not in CANONICAL_INTENTS:
                raise ValueError(f"Unknown gold_intent '{gold_intent}' on row {row_idx}")

            pred_res = labeler.label(text)
            pred_intent = pred_res["intent"]

            gold_labels.append(gold_intent)
            pred_labels.append(pred_intent)
            sample_records.append({
                "id": row.get("id"),
                "conversation_id": row.get("conversation_id"),
                "text": text,
                "gold_intent": gold_intent,
                "pred_intent": pred_intent,
                "confidence": pred_res["confidence"],
                "reason": pred_res["reason"],
                "matched_rules": pred_res["matched_rules"],
            })

    metrics = calculate_metrics(gold_labels, pred_labels, CANONICAL_INTENTS)
    report_text = format_report(metrics, CANONICAL_INTENTS)

    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    print(report_text)
    return metrics


if __name__ == "__main__":
    run_audit()
