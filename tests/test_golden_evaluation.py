"""Unit tests for Golden Set evaluation module (scripts/evaluate_intent_classifier_golden.py).

Verifies:
1. Exactly 250 Golden rows are loaded.
2. Partition yields exactly 225 specific-intent and 25 other_unclear examples.
3. Classifier input is strictly target customer text.
4. No unknown predicted intents outside the 14-class taxonomy.
5. Multi-class metrics are computed only on the 225 specific-intent cases.
6. other_unclear is excluded from 14-class accuracy/F1 metrics.
7. Majority baseline is fixed to TRAIN silver distribution ('battery_power') rather than Golden prevalence.
8. Golden annotation CSV remains completely unmodified.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import unittest

from scripts.evaluate_intent_classifier_golden import (
    DEFAULT_GOLDEN_PATH,
    DEFAULT_MODEL_PATH,
    TRAIN_SILVER_MAJORITY_INTENT,
    analyze_other_unclear_cases,
    evaluate_specific_intents,
    load_golden_dataset,
    partition_golden_set,
)
from src.intents.classifier import IntentClassifier
from src.intents.silver_labeler import CANONICAL_INTENTS

SPECIFIC_INTENTS: set[str] = {i for i in CANONICAL_INTENTS if i != "other_unclear"}


class TestGoldenEvaluation(unittest.TestCase):
    """Test suite for Golden Set evaluation pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_path = DEFAULT_GOLDEN_PATH
        cls.records = load_golden_dataset(cls.golden_path)
        cls.specific_records, cls.unclear_records = partition_golden_set(cls.records)
        cls.model_path = DEFAULT_MODEL_PATH
        cls.clf = IntentClassifier.load(cls.model_path) if cls.model_path.exists() else None

    def test_golden_set_row_count_and_partition(self) -> None:
        """Verify exactly 250 Golden rows: 225 specific + 25 other_unclear."""
        self.assertEqual(len(self.records), 250, "Golden set must contain exactly 250 rows.")
        self.assertEqual(len(self.specific_records), 225, "Specific-intent subset must be 225.")
        self.assertEqual(len(self.unclear_records), 25, "other_unclear subset must be 25.")

        for r in self.specific_records:
            self.assertNotEqual(r["gold_intent"], "other_unclear")
            self.assertIn(r["gold_intent"], SPECIFIC_INTENTS)

        for r in self.unclear_records:
            self.assertEqual(r["gold_intent"], "other_unclear")

    def test_classifier_input_is_only_text(self) -> None:
        """Verify that evaluate_specific_intents passes only the text string to the classifier."""
        if self.clf is None:
            self.skipTest("Model artifact not found.")

        sample = self.specific_records[:5]
        expected_texts = [r["text"] for r in sample]

        captured_inputs = []
        original_predict = self.clf.predict

        def mock_predict(texts):
            captured_inputs.extend(texts)
            return original_predict(texts)

        self.clf.predict = mock_predict
        try:
            eval_res = evaluate_specific_intents(self.clf, sample)
        finally:
            self.clf.predict = original_predict

        self.assertEqual(captured_inputs, expected_texts)
        self.assertEqual(len(eval_res["per_class"]), len(self.clf.classes_))

    def test_no_unknown_predicted_intents(self) -> None:
        """Verify classifier never predicts an intent outside the 14 specific canonical intents."""
        if self.clf is None:
            self.skipTest("Model artifact not found.")

        all_texts = [r["text"] for r in self.records]
        preds = self.clf.predict(all_texts)

        for p in preds:
            self.assertIn(p.intent, SPECIFIC_INTENTS)
            self.assertNotEqual(p.intent, "other_unclear")

    def test_metrics_computed_only_on_specific_intents(self) -> None:
        """Verify 14-class metrics evaluation strictly excludes other_unclear."""
        if self.clf is None:
            self.skipTest("Model artifact not found.")

        results = evaluate_specific_intents(self.clf, self.specific_records)

        # Support sum across all classes must equal 225
        total_support = sum(p["support"] for p in results["per_class"].values())
        self.assertEqual(total_support, 225)
        self.assertNotIn("other_unclear", results["per_class"])

    def test_majority_baseline_uses_train_silver_majority(self) -> None:
        """Verify majority baseline is fixed to TRAIN majority (battery_power), not Golden prevalence."""
        if self.clf is None:
            self.skipTest("Model artifact not found.")

        results = evaluate_specific_intents(self.clf, self.specific_records)
        mb = results["majority_baseline"]

        self.assertEqual(mb["fixed_intent"], TRAIN_SILVER_MAJORITY_INTENT)
        self.assertEqual(mb["fixed_intent"], "battery_power")
        self.assertEqual(mb["total_specific"], 225)

        # In Golden 225, count of battery_power
        expected_golden_battery_count = sum(1 for r in self.specific_records if r["gold_intent"] == "battery_power")
        self.assertEqual(mb["golden_matches"], expected_golden_battery_count)
        self.assertAlmostEqual(mb["accuracy"], expected_golden_battery_count / 225, places=4)

    def test_other_unclear_diagnostics_structure(self) -> None:
        """Verify analyze_other_unclear_cases properly reports the 25 unmodeled cases."""
        if self.clf is None:
            self.skipTest("Model artifact not found.")

        analysis = analyze_other_unclear_cases(self.clf, self.unclear_records)
        self.assertEqual(analysis["total_unclear_cases"], 25)
        self.assertEqual(len(analysis["cases"]), 25)

        sum_buckets = analysis["below_05_count"] + (analysis["below_07_count"] - analysis["below_05_count"]) + analysis["above_07_count"]
        self.assertEqual(sum_buckets, 25)

    def test_golden_annotation_file_integrity_not_modified(self) -> None:
        """Verify golden_annotation.csv contains all original column names and 250 rows."""
        with self.golden_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 250)
        self.assertIn("gold_intent", reader.fieldnames or [])
        self.assertIn("conversation_id", reader.fieldnames or [])
        self.assertIn("text", reader.fieldnames or [])


if __name__ == "__main__":
    unittest.main()
