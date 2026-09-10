"""Train the baseline TF-IDF + Logistic Regression Intent Classifier.

Trains exclusively on data/processed/silver_train.jsonl and saves the serialized
model artifact to data/processed/models/intent_classifier_baseline.joblib.
Runs a smoke test on representative inbound customer support queries.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intents.classifier import IntentClassifier

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "silver_train.jsonl"
DEFAULT_MODEL_SAVE_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "intent_classifier_baseline.joblib"

SMOKE_TEST_EXAMPLES: list[str] = [
    "My iPhone 7 battery is dying within 2 hours, it went from 100% to 15% rapidly.",
    "Unable to install iOS 11 update on my phone, keep getting an error message.",
    "Wi-Fi keeps dropping and Bluetooth won't connect to my headphones.",
    "My iCloud photo library is not syncing and backup says failed.",
    "Can you cancel my Apple Music family subscription and refund the last charge?",
    "Forgot my Apple ID password and my account is locked out.",
    "The screen on my iPhone X is completely unresponsive and frozen.",
    "Can't download or install apps from the App Store, getting verification required.",
    "How do I turn off do not disturb while driving on iOS 11?",
    "I was charged twice on my credit card for an iTunes purchase.",
    "Need to schedule an appointment at the Genius Bar to fix broken screen.",
    "Where is my iPhone order, shipment tracking shows delayed delivery.",
    "Someone tried to hack into my account and change my two-factor security.",
    "Cannot activate my new iPhone, activation server cannot be reached.",
]


def load_silver_training_data(data_path: str | Path) -> tuple[list[str], list[str]]:
    """Load texts and canonical silver labels from silver_train.jsonl.

    Args:
        data_path: Path to silver_train.jsonl.

    Returns:
        Tuple of (texts, labels).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at: {path}. "
            "Please run scripts/build_silver_dataset.py first."
        )

    texts: list[str] = []
    labels: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = str(rec.get("text") or "").strip()
            intent = str(rec.get("silver_intent") or "").strip()

            if not text or not intent:
                continue

            texts.append(text)
            labels.append(intent)

    return texts, labels


def train_baseline_classifier(
    train_data_path: str | Path = DEFAULT_TRAIN_DATA_PATH,
    model_save_path: str | Path = DEFAULT_MODEL_SAVE_PATH,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    max_features: int | None = 15000,
) -> tuple[IntentClassifier, dict[str, Any]]:
    """Train and persist the baseline TF-IDF + Logistic Regression IntentClassifier.

    Returns:
        Tuple of (trained classifier, training summary dict).
    """
    texts, labels = load_silver_training_data(train_data_path)

    label_counts = Counter(labels)
    total_examples = len(texts)
    num_classes = len(label_counts)

    print(f"Loaded {total_examples:,} training examples across {num_classes} classes.")

    clf = IntentClassifier(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        C=1.0,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,
    )

    print("Fitting TF-IDF Vectorizer and Logistic Regression...")
    clf.fit(texts, labels)

    save_path = clf.save(model_save_path)
    print(f"Model saved successfully to: {save_path}")

    summary = {
        "training_examples": total_examples,
        "num_classes": num_classes,
        "class_distribution": dict(sorted(label_counts.items(), key=lambda x: -x[1])),
        "model_config": clf.get_config(),
        "model_path": str(save_path),
    }

    return clf, summary


def run_smoke_test(clf: IntentClassifier, examples: list[str] = SMOKE_TEST_EXAMPLES) -> None:
    """Run smoke test predictions on sample queries and display results."""
    print("\n" + "=" * 80)
    print("SMOKE TEST: BASELINE INTENT CLASSIFIER PREDICTIONS")
    print("=" * 80)

    for idx, text in enumerate(examples, start=1):
        res = clf.predict_one(text)
        top_probs = sorted(res.probabilities.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{c}: {p:.3f}" for c, p in top_probs)

        print(f"[{idx:02d}] Query: \"{text}\"")
        print(f"     Predicted:  {res.intent} (confidence: {res.confidence:.4f})")
        print(f"     Top-3:      {top_str}\n")

    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline TF-IDF + Logistic Regression Intent Classifier."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=DEFAULT_TRAIN_DATA_PATH,
        help="Path to silver training data JSONL.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_SAVE_PATH,
        help="Path to save serialized model.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        help="Class weighting strategy ('balanced' or 'none').",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_weight = None if args.class_weight.lower() == "none" else args.class_weight

    clf, summary = train_baseline_classifier(
        train_data_path=args.train_data,
        model_save_path=args.model_out,
        class_weight=class_weight,
        random_state=args.random_state,
    )

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE SUMMARY (MILESTONE 11)")
    print("=" * 80)
    print(f"- Training Examples: {summary['training_examples']:,}")
    print(f"- Number of Classes:  {summary['num_classes']}")
    print("\nClass Distribution:")
    for intent, count in summary["class_distribution"].items():
        pct = count / summary["training_examples"] * 100.0
        print(f"  {intent:<22} : {count:>6,} ({pct:>5.2f}%)")

    print("\nModel Configuration:")
    for k, v in summary["model_config"].items():
        if k != "classes":
            print(f"  {k:<16} : {v}")

    run_smoke_test(clf)


if __name__ == "__main__":
    main()
