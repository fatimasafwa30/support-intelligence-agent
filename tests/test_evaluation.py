"""Unit tests for intent classifier evaluation utilities (scripts/evaluate_intent_classifier.py).

Verifies:
1. Metric calculation functions (accuracy, macro/weighted F1, per-class metrics).
2. Majority baseline identification and performance computation.
3. Confidence bucket aggregation and edge cases.
4. Top confusion pair extraction and ranking.
5. Strict Golden Set isolation in DEV split loader.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_intent_classifier import (
    compute_classification_metrics,
    compute_confidence_buckets,
    compute_majority_baseline,
    extract_top_confusion_pairs,
    format_evaluation_report,
    load_dev_conversation_ids,
)


class TestEvaluationUtilities(unittest.TestCase):
    """Test suite for classifier evaluation utilities."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.classes = ["battery_power", "connectivity", "software_update"]

        # Synthetic test data
        # Class counts: battery: 4, connectivity: 3, software_update: 3 (Total: 10)
        self.y_true = [
            "battery_power", "battery_power", "battery_power", "battery_power",
            "connectivity", "connectivity", "connectivity",
            "software_update", "software_update", "software_update",
        ]
        # Predictions: 8 correct, 2 errors
        self.y_pred = [
            "battery_power", "battery_power", "battery_power", "software_update",  # 1 error: battery -> software_update
            "connectivity", "connectivity", "battery_power",                      # 1 error: connectivity -> battery
            "software_update", "software_update", "software_update",              # 3 correct
        ]
        self.confidences = [
            0.95, 0.92, 0.88, 0.55,
            0.91, 0.85, 0.45,
            0.98, 0.90, 0.75,
        ]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_compute_classification_metrics(self) -> None:
        """Verify accuracy and macro/weighted metrics match expected calculations."""
        metrics = compute_classification_metrics(self.y_true, self.y_pred, self.classes)

        # 8 out of 10 correct -> 0.80 accuracy
        self.assertAlmostEqual(metrics["accuracy"], 0.80, places=4)

        # Macro and weighted F1 should be within [0, 1]
        self.assertGreater(metrics["macro_f1"], 0.70)
        self.assertLessEqual(metrics["macro_f1"], 1.0)
        self.assertGreater(metrics["weighted_f1"], 0.70)

        # Check per-class structure
        for c in self.classes:
            self.assertIn(c, metrics["per_class"])
            self.assertIn("precision", metrics["per_class"][c])
            self.assertIn("recall", metrics["per_class"][c])
            self.assertIn("f1", metrics["per_class"][c])
            self.assertIn("support", metrics["per_class"][c])

        self.assertEqual(metrics["per_class"]["battery_power"]["support"], 4)
        self.assertEqual(metrics["per_class"]["connectivity"]["support"], 3)
        self.assertEqual(metrics["per_class"]["software_update"]["support"], 3)

    def test_compute_majority_baseline(self) -> None:
        """Verify majority baseline identifies the most frequent class and computes metrics."""
        baseline = compute_majority_baseline(self.y_true, self.classes)

        # Majority class is battery_power (4 / 10 = 0.40)
        self.assertEqual(baseline["majority_intent"], "battery_power")
        self.assertEqual(baseline["majority_count"], 4)
        self.assertAlmostEqual(baseline["accuracy"], 0.40, places=4)
        self.assertGreaterEqual(baseline["macro_f1"], 0.0)

    def test_compute_confidence_buckets(self) -> None:
        """Verify confidence buckets group samples accurately and calculate bucket accuracy."""
        buckets = compute_confidence_buckets(self.y_true, self.y_pred, self.confidences)

        # Total counts across all buckets must equal len(y_true)
        total_bucketed = sum(b["count"] for b in buckets)
        self.assertEqual(total_bucketed, len(self.y_true))

        # Check lowest bucket [0.00, 0.50): has 1 sample (0.45, which was an error: connectivity -> battery)
        b_low = next(b for b in buckets if "0.00, 0.50" in b["bucket"])
        self.assertEqual(b_low["count"], 1)
        self.assertEqual(b_low["accuracy"], 0.0)

        # Check highest bucket [0.90, 1.00]: has 0.95, 0.92, 0.91, 0.98, 0.90 (5 samples, all correct)
        b_high = next(b for b in buckets if "0.90, 1.00" in b["bucket"])
        self.assertEqual(b_high["count"], 5)
        self.assertEqual(b_high["accuracy"], 100.0)

    def test_extract_top_confusion_pairs(self) -> None:
        """Verify top confusion pairs identify and sort misclassifications properly."""
        confusions = extract_top_confusion_pairs(self.y_true, self.y_pred, top_n=5)

        # There are 2 errors:
        # (battery_power -> software_update, count=1)
        # (connectivity -> battery_power, count=1)
        self.assertEqual(len(confusions), 2)
        pairs = {(c[0], c[1]) for c in confusions}
        self.assertIn(("battery_power", "software_update"), pairs)
        self.assertIn(("connectivity", "battery_power"), pairs)

    def test_dev_split_golden_leakage_rejection(self) -> None:
        """Verify load_dev_conversation_ids raises ValueError if Golden Set overlaps DEV."""
        splits_file = self.root / "splits.json"
        golden_file = self.root / "golden.csv"

        # Manifest with dev conv apple_000099
        with splits_file.open("w", encoding="utf-8") as f:
            json.dump({"dev_conversation_ids": ["apple_000099", "apple_000100"]}, f)

        # Golden set deliberately containing apple_000099
        with golden_file.open("w", encoding="utf-8") as f:
            f.write("conversation_id,intent\napple_000099,battery_power\n")

        with self.assertRaises(ValueError) as ctx:
            load_dev_conversation_ids(
                splits_manifest_path=splits_file,
                golden_annotation_path=golden_file,
                golden_candidates_path=None,
            )
        self.assertIn("Golden Set leakage in DEV split", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
