"""Unit tests for IntentClassifier module (src/intents/classifier.py).

Verifies:
1. Model can train on small fixture dataset.
2. Predictions return valid taxonomy intents.
3. Class probabilities are in [0, 1] and sum approximately to 1.0.
4. Empty/whitespace and edge-case inputs are handled safely without crashing.
5. Model serialization and deserialization (save/load) preserves identical predictions.
6. Rejection of labels outside the configured 14 specific taxonomy intents (and other_unclear).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.intents.classifier import IntentClassifier, PredictionResult
from src.intents.silver_labeler import CANONICAL_INTENTS

SPECIFIC_INTENTS: list[str] = [i for i in CANONICAL_INTENTS if i != "other_unclear"]


class TestIntentClassifier(unittest.TestCase):
    """Test suite for TF-IDF + Logistic Regression IntentClassifier."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Small balanced synthetic dataset across 4 specific intents
        self.sample_texts = [
            # battery_power
            "My battery is draining so fast since this morning",
            "Phone shuts down at 30% battery life remaining",
            "Battery percentage drops 50% in one hour",
            # connectivity
            "Wi-Fi keeps dropping and disconnecting constantly",
            "Cannot connect to bluetooth speaker or headphones",
            "Bluetooth not pairing with my Apple Watch",
            # software_update
            "Cannot install iOS update getting error message",
            "Update failed while downloading iOS 11 software",
            "Trouble updating my iPhone to the latest software",
            # billing_payments
            "I was charged twice on my credit card for this",
            "Need a refund for unauthorized payment charge",
            "Payment method was declined when renewing subscription",
        ]
        self.sample_labels = [
            "battery_power",
            "battery_power",
            "battery_power",
            "connectivity",
            "connectivity",
            "connectivity",
            "software_update",
            "software_update",
            "software_update",
            "billing_payments",
            "billing_payments",
            "billing_payments",
        ]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_train_on_fixture_and_predict(self) -> None:
        """Verify model trains on fixture and produces sensible predictions."""
        clf = IntentClassifier(min_df=1, random_state=42)
        clf.fit(self.sample_texts, self.sample_labels)

        res = clf.predict_one("My battery dies within 30 minutes")
        self.assertIsInstance(res, PredictionResult)
        self.assertEqual(res.intent, "battery_power")
        self.assertIn(res.intent, SPECIFIC_INTENTS)
        self.assertGreater(res.confidence, 0.25)

    def test_prediction_returns_valid_taxonomy_intent(self) -> None:
        """Verify all predictions return valid canonical taxonomy intents."""
        clf = IntentClassifier(min_df=1, random_state=42)
        clf.fit(self.sample_texts, self.sample_labels)

        queries = [
            "Bluetooth pairing issues",
            "Unexpected credit card charge",
            "Cannot install the latest update",
            "Random text with no obvious keywords",
        ]
        for q in queries:
            res = clf.predict_one(q)
            self.assertIn(res.intent, SPECIFIC_INTENTS)
            self.assertNotEqual(res.intent, "other_unclear")

    def test_probabilities_are_valid_and_sum_to_one(self) -> None:
        """Verify probabilities are non-negative and sum to 1.0."""
        clf = IntentClassifier(min_df=1, random_state=42)
        clf.fit(self.sample_texts, self.sample_labels)

        res = clf.predict_one("Wi-Fi and bluetooth disconnected")
        probs = res.probabilities

        # Check all trained classes are represented
        self.assertEqual(set(probs.keys()), set(clf.classes_))

        # Check sum to 1.0 within float tolerance
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

        # Check each prob in [0, 1]
        for p in probs.values():
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

        # Top confidence matches max probability
        self.assertAlmostEqual(res.confidence, max(probs.values()), places=4)

    def test_empty_and_invalid_input_handled_safely(self) -> None:
        """Verify empty string, whitespace, and special characters do not crash the model."""
        clf = IntentClassifier(min_df=1, random_state=42)
        clf.fit(self.sample_texts, self.sample_labels)

        edge_cases = ["", "   ", "\n\t", "???!!!", "1234567890", "https://t.co/abc"]
        for ec in edge_cases:
            res = clf.predict_one(ec)
            self.assertIsInstance(res, PredictionResult)
            self.assertIn(res.intent, SPECIFIC_INTENTS)
            self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=3)

    def test_saved_and_loaded_model_gives_same_predictions(self) -> None:
        """Verify model persistence produces identical outputs after load."""
        clf = IntentClassifier(min_df=1, random_state=42)
        clf.fit(self.sample_texts, self.sample_labels)

        save_path = self.root / "test_model.joblib"
        clf.save(save_path)

        loaded_clf = IntentClassifier.load(save_path)

        test_queries = [
            "Battery completely flat",
            "Wi-Fi network error",
            "Refund needed for double charge",
            "Software upgrade failing",
        ]
        for q in test_queries:
            orig = clf.predict_one(q)
            loaded = loaded_clf.predict_one(q)

            self.assertEqual(orig.intent, loaded.intent)
            self.assertAlmostEqual(orig.confidence, loaded.confidence, places=5)
            for k in orig.probabilities:
                self.assertAlmostEqual(orig.probabilities[k], loaded.probabilities[k], places=5)

    def test_reject_other_unclear_or_invalid_labels(self) -> None:
        """Verify fit() rejects 'other_unclear' and unknown intent strings."""
        clf = IntentClassifier(min_df=1)

        # Rejects other_unclear
        with self.assertRaises(ValueError) as ctx1:
            clf.fit(
                ["Some message", "Another message"],
                ["battery_power", "other_unclear"],
            )
        self.assertIn("other_unclear", str(ctx1.exception))

        # Rejects unknown intent
        with self.assertRaises(ValueError) as ctx2:
            clf.fit(
                ["Some message", "Another message"],
                ["battery_power", "nonexistent_custom_intent"],
            )
        self.assertIn("canonical taxonomy", str(ctx2.exception))

    # ── Calibrated Confidence Abstention Regression Tests ─────────────────────

    def test_disabled_abstention_preserves_current_behavior(self) -> None:
        """Verify default/disabled abstention (None) always returns specific intents."""
        clf = IntentClassifier(min_df=1, random_state=42, abstention_threshold=None)
        clf.fit(self.sample_texts, self.sample_labels)

        queries = [
            "My battery is draining rapidly",
            "completely random unknown words xyz 123",
            "",
        ]
        for q in queries:
            res = clf.predict_one(q)
            self.assertIn(res.intent, SPECIFIC_INTENTS)
            self.assertNotEqual(res.intent, "other_unclear")
            self.assertFalse(res.is_abstained)

    def test_high_confidence_prediction_unchanged_with_abstention(self) -> None:
        """Verify high-confidence predictions retain specific intent when abstention is active."""
        clf = IntentClassifier(min_df=1, random_state=42, abstention_threshold=0.25)
        clf.fit(self.sample_texts, self.sample_labels)

        res = clf.predict_one("My battery percentage drops 50% in one hour")
        self.assertEqual(res.intent, "battery_power")
        self.assertFalse(res.is_abstained)
        self.assertGreaterEqual(res.confidence, 0.25)

    def test_low_confidence_prediction_becomes_other_unclear(self) -> None:
        """Verify low-confidence inputs return other_unclear when below threshold."""
        clf = IntentClassifier(min_df=1, random_state=42, abstention_threshold=0.50)
        clf.fit(self.sample_texts, self.sample_labels)

        # Ambiguous query that splits probabilities across classes
        res_unabstained = clf.predict_one("I need assistance with something weird today", abstention_threshold=None)
        # Check with threshold higher than confidence
        higher_threshold = res_unabstained.confidence + 0.05
        res = clf.predict_one("I need assistance with something weird today", abstention_threshold=higher_threshold)

        self.assertEqual(res.intent, "other_unclear")
        self.assertTrue(res.is_abstained)
        self.assertAlmostEqual(res.confidence, res_unabstained.confidence, places=4)
        # Probabilities dictionary preserved
        self.assertEqual(set(res.probabilities.keys()), set(clf.classes_))

    def test_threshold_boundary_behavior(self) -> None:
        """Verify exact boundary behavior: >= threshold retains intent, < threshold abstains."""
        clf = IntentClassifier(min_df=1, random_state=42)
        clf.fit(self.sample_texts, self.sample_labels)

        raw = clf.predict_one("Trouble updating my iPhone to the latest software")
        conf = raw.confidence

        # Exactly at or slightly below confidence -> retained
        retained = clf.predict_one(
            "Trouble updating my iPhone to the latest software",
            abstention_threshold=conf,
        )
        self.assertEqual(retained.intent, raw.intent)
        self.assertFalse(retained.is_abstained)

        # Strictly above confidence -> abstained to other_unclear
        abstained = clf.predict_one(
            "Trouble updating my iPhone to the latest software",
            abstention_threshold=conf + 0.001,
        )
        self.assertEqual(abstained.intent, "other_unclear")
        self.assertTrue(abstained.is_abstained)

    def test_saved_and_loaded_model_preserves_abstention_threshold(self) -> None:
        """Verify abstention_threshold is persisted and loaded correctly."""
        clf = IntentClassifier(min_df=1, random_state=42, abstention_threshold=0.25)
        clf.fit(self.sample_texts, self.sample_labels)

        save_path = self.root / "abstention_model.joblib"
        clf.save(save_path)

        loaded_clf = IntentClassifier.load(save_path)
        self.assertEqual(loaded_clf.abstention_threshold, 0.25)

        # Check prediction behavior matches
        p1 = clf.predict_one("some random text")
        p2 = loaded_clf.predict_one("some random text")
        self.assertEqual(p1.intent, p2.intent)
        self.assertEqual(p1.is_abstained, p2.is_abstained)


if __name__ == "__main__":
    unittest.main()
