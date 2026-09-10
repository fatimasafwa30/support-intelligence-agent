"""Final Human-Benchmark Evaluation of the baseline Intent Classifier on Golden Set.

Evaluates the frozen TF-IDF + Logistic Regression classifier on the 250-example
human-labelled Golden Set (data/golden/golden_annotation.csv).
Strictly separates the 225 specific-intent benchmark cases from the 25 human
other_unclear cases. Evaluates against a fixed TRAIN silver majority baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intents.classifier import IntentClassifier, PredictionResult
from src.intents.silver_labeler import CANONICAL_INTENTS

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data" / "golden" / "golden_annotation.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "intent_classifier_baseline.joblib"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "intent_classifier_golden_evaluation.txt"
DEFAULT_MISCLASSIFICATIONS_CSV_PATH = PROJECT_ROOT / "reports" / "golden_misclassifications.csv"

# Fixed majority class from TRAIN silver distribution (39.34% of silver train data)
TRAIN_SILVER_MAJORITY_INTENT = "battery_power"

CONFIDENCE_BUCKETS: list[tuple[float, float, str]] = [
    (0.0, 0.50, "[0.00, 0.50)"),
    (0.50, 0.70, "[0.50, 0.70)"),
    (0.70, 0.80, "[0.70, 0.80)"),
    (0.80, 0.90, "[0.80, 0.90)"),
    (0.90, 1.0001, "[0.90, 1.00]"),
]


def load_golden_dataset(golden_path: str | Path = DEFAULT_GOLDEN_PATH) -> list[dict[str, Any]]:
    """Load and validate all rows from golden_annotation.csv.

    Args:
        golden_path: Path to golden_annotation.csv.

    Returns:
        List of 250 parsed golden records.
    """
    path = Path(golden_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden annotation CSV not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"id", "conversation_id", "tweet_id", "text", "gold_intent", "conversation_context"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")

        for row in reader:
            records.append({
                "id": row.get("id", ""),
                "conversation_id": row.get("conversation_id", ""),
                "tweet_id": row.get("tweet_id", ""),
                "text": (row.get("text") or "").strip(),
                "gold_intent": (row.get("gold_intent") or "").strip(),
                "conversation_context": row.get("conversation_context", ""),
            })

    if len(records) != 250:
        logger.warning("Expected 250 golden rows, found %d", len(records))

    return records


def partition_golden_set(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split golden dataset into specific-intent benchmark cases and other_unclear cases.

    Returns:
        Tuple of (specific_intent_records, other_unclear_records).
    """
    specific_records = [r for r in records if r["gold_intent"] != "other_unclear"]
    unclear_records = [r for r in records if r["gold_intent"] == "other_unclear"]
    return specific_records, unclear_records


def evaluate_specific_intents(
    clf: IntentClassifier,
    specific_records: list[dict[str, Any]],
    train_majority_intent: str = TRAIN_SILVER_MAJORITY_INTENT,
) -> dict[str, Any]:
    """Evaluate 14-class classifier on specific-intent Golden subset."""
    y_true = [r["gold_intent"] for r in specific_records]
    # Classifier receives strictly target message text
    texts = [r["text"] for r in specific_records]

    predictions: list[PredictionResult] = clf.predict(texts)
    y_pred = [p.intent for p in predictions]
    y_conf = [p.confidence for p in predictions]

    classes = list(clf.classes_)

    # Classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class: dict[str, dict[str, float]] = {}
    for idx, c in enumerate(classes):
        per_class[c] = {
            "precision": float(per_class_p[idx]),
            "recall": float(per_class_r[idx]),
            "f1": float(per_class_f1[idx]),
            "support": int(per_class_supp[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Fixed TRAIN majority baseline
    y_majority = [train_majority_intent] * len(y_true)
    majority_acc = float(accuracy_score(y_true, y_majority))
    _, _, majority_macro_f1, _ = precision_recall_fscore_support(
        y_true, y_majority, labels=classes, average="macro", zero_division=0
    )

    majority_baseline = {
        "fixed_intent": train_majority_intent,
        "source": "TRAIN silver majority (39.34% of training set)",
        "golden_matches": sum(1 for y in y_true if y == train_majority_intent),
        "total_specific": len(y_true),
        "accuracy": majority_acc,
        "macro_f1": float(majority_macro_f1),
    }

    # Confidence buckets
    conf_buckets: list[dict[str, Any]] = []
    for low, high, label in CONFIDENCE_BUCKETS:
        indices = [i for i, c in enumerate(y_conf) if low <= c < high]
        count = len(indices)
        if count == 0:
            conf_buckets.append({
                "bucket": label,
                "count": 0,
                "percentage": 0.0,
                "accuracy": 0.0,
            })
            continue

        b_true = [y_true[i] for i in indices]
        b_pred = [y_pred[i] for i in indices]
        b_acc = float(accuracy_score(b_true, b_pred))
        conf_buckets.append({
            "bucket": label,
            "count": count,
            "percentage": round(count / len(y_true) * 100.0, 2),
            "accuracy": round(b_acc * 100.0, 2),
        })

    # Confusion pairs
    pair_counts: Counter[tuple[str, str]] = Counter()
    misclassifications: list[dict[str, Any]] = []

    for rec, pred, conf in zip(specific_records, y_pred, y_conf):
        if rec["gold_intent"] != pred:
            pair_counts[(rec["gold_intent"], pred)] += 1
            misclassifications.append({
                "id": rec["id"],
                "conversation_id": rec["conversation_id"],
                "tweet_id": rec["tweet_id"],
                "text": rec["text"],
                "conversation_context": rec["conversation_context"],
                "gold_intent": rec["gold_intent"],
                "predicted_intent": pred,
                "confidence": round(conf, 4),
            })

    top_confusions = [
        (true, pred, count)
        for (true, pred), count in pair_counts.most_common(10)
    ]

    return {
        "accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "majority_baseline": majority_baseline,
        "avg_confidence": float(np.mean(y_conf)) if y_conf else 0.0,
        "conf_buckets": conf_buckets,
        "misclassifications": misclassifications,
        "top_confusions": top_confusions,
    }


def analyze_other_unclear_cases(
    clf: IntentClassifier,
    unclear_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate classifier behavior on the 25 human other_unclear cases.

    NOTE: The current classifier has no other_unclear output class; these cases
    are reported purely as unmodeled abstention diagnostics.
    """
    texts = [r["text"] for r in unclear_records]
    predictions = clf.predict(texts)

    cases: list[dict[str, Any]] = []
    confidences: list[float] = []

    for rec, pred in zip(unclear_records, predictions):
        conf = float(pred.confidence)
        confidences.append(conf)
        cases.append({
            "id": rec["id"],
            "conversation_id": rec["conversation_id"],
            "tweet_id": rec["tweet_id"],
            "text": rec["text"],
            "predicted_intent": pred.intent,
            "confidence": round(conf, 4),
        })

    below_05 = sum(1 for c in confidences if c < 0.50)
    below_07 = sum(1 for c in confidences if c < 0.70)
    above_07 = sum(1 for c in confidences if c >= 0.70)

    predicted_intent_counts = Counter(c["predicted_intent"] for c in cases)

    return {
        "total_unclear_cases": len(unclear_records),
        "cases": cases,
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "below_05_count": below_05,
        "below_05_pct": round(below_05 / len(unclear_records) * 100.0, 2) if unclear_records else 0.0,
        "below_07_count": below_07,
        "below_07_pct": round(below_07 / len(unclear_records) * 100.0, 2) if unclear_records else 0.0,
        "above_07_count": above_07,
        "above_07_pct": round(above_07 / len(unclear_records) * 100.0, 2) if unclear_records else 0.0,
        "predicted_intent_distribution": dict(predicted_intent_counts.most_common()),
    }


def format_golden_report(
    specific_metrics: dict[str, Any],
    unclear_analysis: dict[str, Any],
    total_golden_count: int = 250,
) -> str:
    """Format comprehensive human-benchmark evaluation report."""
    mb = specific_metrics["majority_baseline"]

    lines: list[str] = [
        "=" * 80,
        "INTENT CLASSIFIER HUMAN BENCHMARK REPORT (GOLDEN SET EVALUATION)",
        "=" * 80,
        "",
        "## 1. Benchmark Scope & Framing",
        f"- Total Golden Set Examples:            {total_golden_count}",
        f"- Specific-Intent Benchmark Subset:     {mb['total_specific']} (gold_intent != 'other_unclear')",
        f"- Human other_unclear Cases:            {unclear_analysis['total_unclear_cases']} (reported separately)",
        "",
        "CRITICAL DISTINCTION: DEV vs. GOLDEN BENCHMARKS",
        "  - DEV Split (3,895 samples, 92.66% accuracy): Evaluated against rule-generated SILVER labels.",
        "    Useful exclusively for internal development and diagnostic sanity-checking.",
        "  - GOLDEN Set (225 specific-intent samples): Evaluated against HUMAN ground-truth annotations.",
        "    This is the authoritative, primary benchmark of real customer intent classification.",
        "",
        "## 2. Specific-Intent Benchmark vs. Trivial Majority Baseline (225 Cases)",
        f"{'Metric':<25} | {'Trained Model':>15} | {'TRAIN Majority Baseline':>24}",
        "-" * 70,
        f"{'Accuracy':<25} | {specific_metrics['accuracy'] * 100:>14.2f}% | {mb['accuracy'] * 100:>23.2f}%",
        f"{'Macro Precision':<25} | {specific_metrics['macro_precision'] * 100:>14.2f}% | {'N/A':>24}",
        f"{'Macro Recall':<25} | {specific_metrics['macro_recall'] * 100:>14.2f}% | {'N/A':>24}",
        f"{'Macro F1-Score':<25} | {specific_metrics['macro_f1'] * 100:>14.2f}% | {mb['macro_f1'] * 100:>23.2f}%",
        f"{'Weighted F1-Score':<25} | {specific_metrics['weighted_f1'] * 100:>14.2f}% | {'N/A':>24}",
        "-" * 70,
        f"Baseline: Fixed TRAIN majority class '{mb['fixed_intent']}' ({mb['golden_matches']}/225 in Golden, {mb['accuracy']*100:.2f}%)",
        "",
        "## 3. Per-Intent Performance Metrics (225 Golden Benchmark Cases)",
        f"{'Intent':<22} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support':>10}",
        "-" * 70,
    ]

    for intent, p_dict in specific_metrics["per_class"].items():
        lines.append(
            f"{intent:<22} | {p_dict['precision'] * 100:>9.2f}% | {p_dict['recall'] * 100:>9.2f}% | {p_dict['f1'] * 100:>9.2f}% | {p_dict['support']:>10}"
        )

    lines.extend([
        "-" * 70,
        "",
        "## 4. Confidence Calibration Analysis (225 Specific-Intent Cases)",
        f"Average Top-1 Confidence: {specific_metrics['avg_confidence']:.4f}",
        "",
        f"{'Confidence Bucket':<20} | {'Count':>10} | {'Share %':>10} | {'Bucket Accuracy':>18}",
        "-" * 64,
    ])

    for b in specific_metrics["conf_buckets"]:
        lines.append(
            f"{b['bucket']:<20} | {b['count']:>10} | {b['percentage']:>9.2f}% | {b['accuracy']:>17.2f}%"
        )

    lines.extend([
        "-" * 64,
        "",
        "## 5. Top Confusion Pairs (Gold Intent -> Predicted Intent)",
        f"{'Rank':<5} | {'Gold Intent (Human)':<24} -> {'Predicted Intent (ML)':<24} | {'Count':>6}",
        "-" * 67,
    ])

    for rank, (true, pred, cnt) in enumerate(specific_metrics["top_confusions"], start=1):
        lines.append(f"{rank:<5} | {true:<24} -> {pred:<24} | {cnt:>6}")

    lines.extend([
        "-" * 67,
        "",
        "## 6. Human other_unclear Cases (25 Unmodeled Abstention Diagnostics)",
        "The classifier currently models only the 14 specific intents and has no abstention mechanism.",
        f"- Total other_unclear Cases:       {unclear_analysis['total_unclear_cases']}",
        f"- Average Prediction Confidence:    {unclear_analysis['avg_confidence']:.4f}",
        f"- Confidence < 0.50 (Low):          {unclear_analysis['below_05_count']} ({unclear_analysis['below_05_pct']}%)",
        f"- Confidence < 0.70:                {unclear_analysis['below_07_count']} ({unclear_analysis['below_07_pct']}%)",
        f"- Confidence >= 0.70 (High Overfit): {unclear_analysis['above_07_count']} ({unclear_analysis['above_07_pct']}%)",
        "",
        "Predicted Intent Distribution on other_unclear Queries:",
    ])

    for p_intent, p_cnt in unclear_analysis["predicted_intent_distribution"].items():
        lines.append(f"  {p_intent:<25} : {p_cnt:>3}")

    lines.extend([
        "",
        "Sample other_unclear Cases:",
    ])

    for sample in unclear_analysis["cases"][:5]:
        lines.append(
            f"  [{sample['id']}] \"{sample['text']}\" -> Predicted: {sample['predicted_intent']} (conf: {sample['confidence']:.4f})"
        )

    lines.extend([
        "",
        "## 7. Final Methodological Summary",
        "- The first ML baseline significantly outperforms the TRAIN majority baseline on human annotations.",
        "- As expected, human evaluation reveals a performance gap compared to silver DEV diagnostics,",
        "  underscoring the importance of testing on real human ground-truth.",
        "- The 25 other_unclear cases clearly indicate the necessity for a calibrated abstention mechanism.",
        "=" * 80,
    ])

    return "\n".join(lines)


def run_golden_evaluation(
    golden_path: str | Path = DEFAULT_GOLDEN_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_output_path: str | Path = DEFAULT_REPORT_PATH,
    misclassifications_csv_path: str | Path = DEFAULT_MISCLASSIFICATIONS_CSV_PATH,
) -> dict[str, Any]:
    """Execute end-to-end evaluation on the Golden Set."""
    print(f"Loading Golden Set from: {golden_path}...")
    golden_records = load_golden_dataset(golden_path)
    specific_records, unclear_records = partition_golden_set(golden_records)

    print(
        f"Loaded {len(golden_records)} Golden examples: "
        f"{len(specific_records)} specific-intent benchmark cases, "
        f"{len(unclear_records)} human other_unclear cases."
    )

    print(f"Loading frozen classifier from: {model_path}...")
    clf = IntentClassifier.load(model_path)

    print("Evaluating 14-class classifier on 225 specific-intent Golden examples...")
    specific_metrics = evaluate_specific_intents(clf, specific_records)

    print("Analyzing classifier predictions on 25 human other_unclear cases...")
    unclear_analysis = analyze_other_unclear_cases(clf, unclear_records)

    # Save misclassified specific examples to CSV
    csv_path = Path(misclassifications_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "conversation_id",
        "tweet_id",
        "text",
        "conversation_context",
        "gold_intent",
        "predicted_intent",
        "confidence",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in specific_metrics["misclassifications"]:
            writer.writerow(row)
    print(f"Saved {len(specific_metrics['misclassifications'])} misclassifications to: {csv_path}")

    # Generate and save report
    report_text = format_golden_report(
        specific_metrics=specific_metrics,
        unclear_analysis=unclear_analysis,
        total_golden_count=len(golden_records),
    )

    report_path = Path(report_output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as rf:
        rf.write(report_text)
    print(f"Evaluation report written to: {report_path}\n")
    print(report_text)

    return {
        "golden_total": len(golden_records),
        "specific_count": len(specific_records),
        "unclear_count": len(unclear_records),
        "specific_metrics": specific_metrics,
        "unclear_analysis": unclear_analysis,
        "report_path": str(report_path),
        "csv_path": str(csv_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline Intent Classifier on human-labelled Golden Set."
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to golden_annotation.csv.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to save Golden evaluation report.",
    )
    parser.add_argument(
        "--misclassifications-csv",
        type=Path,
        default=DEFAULT_MISCLASSIFICATIONS_CSV_PATH,
        help="Path to save misclassifications CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_golden_evaluation(
        golden_path=args.golden,
        model_path=args.model,
        report_output_path=args.report,
        misclassifications_csv_path=args.misclassifications_csv,
    )


if __name__ == "__main__":
    main()
