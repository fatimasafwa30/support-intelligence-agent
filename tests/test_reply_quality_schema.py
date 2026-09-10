"""Unit tests for reply quality evaluation schema and validation models."""

import unittest

from src.evaluation.reply_quality_schema import (
    DIMENSION_CORRECTNESS,
    DIMENSION_GROUNDEDNESS,
    DIMENSION_HELPFULNESS,
    DIMENSION_RELEVANCE,
    DIMENSION_TONE,
    EVALUATOR_HUMAN,
    EVALUATOR_LLM_JUDGE,
    RUBRIC_DIMENSIONS,
    EvaluationUnit,
    ReplyQualityRating,
)


class TestReplyQualitySchema(unittest.TestCase):
    """Test suite for ReplyQualityRating and EvaluationUnit schemas."""

    def test_valid_human_rating_instantiation(self):
        """Verify successful creation of a valid human rating."""
        rating = ReplyQualityRating(
            example_id="golden_0001",
            evaluator_type=EVALUATOR_HUMAN,
            groundedness=5,
            correctness=4,
            relevance=5,
            helpfulness=4,
            tone=5,
            overall_score=4.6,
            notes="Excellent grounded answer with official link.",
        )
        self.assertEqual(rating.example_id, "golden_0001")
        self.assertEqual(rating.evaluator_type, "human")
        self.assertEqual(rating.groundedness, 5)
        self.assertEqual(rating.correctness, 4)
        self.assertEqual(rating.relevance, 5)
        self.assertEqual(rating.helpfulness, 4)
        self.assertEqual(rating.tone, 5)
        self.assertEqual(rating.overall_score, 4.6)
        self.assertEqual(rating.notes, "Excellent grounded answer with official link.")
        self.assertAlmostEqual(rating.compute_average_score(), 4.6, places=4)

    def test_valid_llm_judge_rating_without_optionals(self):
        """Verify rating works without optional overall_score and notes."""
        rating = ReplyQualityRating(
            example_id="golden_0002",
            evaluator_type=EVALUATOR_LLM_JUDGE,
            groundedness=3,
            correctness=3,
            relevance=4,
            helpfulness=3,
            tone=4,
        )
        self.assertIsNone(rating.overall_score)
        self.assertIsNone(rating.notes)
        self.assertAlmostEqual(rating.compute_average_score(), 3.4, places=4)

    def test_evaluator_type_validation(self):
        """Ensure invalid evaluator types raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ReplyQualityRating(
                example_id="golden_0003",
                evaluator_type="crowd_worker",
                groundedness=5,
                correctness=5,
                relevance=5,
                helpfulness=5,
                tone=5,
            )
        self.assertIn("evaluator_type must be one of", str(ctx.exception))

    def test_missing_or_empty_example_id(self):
        """Ensure missing or empty example_id raises ValueError."""
        with self.assertRaises(ValueError):
            ReplyQualityRating(
                example_id="",
                evaluator_type=EVALUATOR_HUMAN,
                groundedness=5,
                correctness=5,
                relevance=5,
                helpfulness=5,
                tone=5,
            )

        with self.assertRaises(ValueError):
            ReplyQualityRating(
                example_id="   ",
                evaluator_type=EVALUATOR_HUMAN,
                groundedness=5,
                correctness=5,
                relevance=5,
                helpfulness=5,
                tone=5,
            )

    def test_score_out_of_bounds_low(self):
        """Ensure score < 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ReplyQualityRating(
                example_id="golden_0004",
                evaluator_type=EVALUATOR_HUMAN,
                groundedness=0,
                correctness=4,
                relevance=4,
                helpfulness=4,
                tone=4,
            )
        self.assertIn("must be between 1 and 5", str(ctx.exception))

    def test_score_out_of_bounds_high(self):
        """Ensure score > 5 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ReplyQualityRating(
                example_id="golden_0005",
                evaluator_type=EVALUATOR_LLM_JUDGE,
                groundedness=5,
                correctness=6,
                relevance=5,
                helpfulness=5,
                tone=5,
            )
        self.assertIn("must be between 1 and 5", str(ctx.exception))

    def test_score_non_integer_rejected(self):
        """Ensure float scores for individual rubric dimensions raise TypeError."""
        with self.assertRaises(TypeError):
            ReplyQualityRating(
                example_id="golden_0006",
                evaluator_type=EVALUATOR_HUMAN,
                groundedness=4.5,  # Floats not permitted for 1-5 discrete scale
                correctness=5,
                relevance=5,
                helpfulness=5,
                tone=5,
            )

    def test_score_boolean_rejected(self):
        """Ensure booleans (subclass of int) are explicitly rejected."""
        with self.assertRaises(TypeError):
            ReplyQualityRating(
                example_id="golden_0007",
                evaluator_type=EVALUATOR_HUMAN,
                groundedness=True,
                correctness=5,
                relevance=5,
                helpfulness=5,
                tone=5,
            )

    def test_all_five_dimensions_validated(self):
        """Ensure every one of the 5 dimensions is checked against the [1, 5] constraint."""
        base_kwargs = {
            "example_id": "golden_0008",
            "evaluator_type": EVALUATOR_HUMAN,
            "groundedness": 3,
            "correctness": 3,
            "relevance": 3,
            "helpfulness": 3,
            "tone": 3,
        }
        for dim in RUBRIC_DIMENSIONS:
            invalid_kwargs = dict(base_kwargs)
            invalid_kwargs[dim] = 10
            with self.assertRaises(ValueError, msg=f"Failed to catch invalid score on {dim}"):
                ReplyQualityRating(**invalid_kwargs)

    def test_overall_score_range_validation(self):
        """Ensure optional overall_score must fall within 1.0 to 5.0."""
        with self.assertRaises(ValueError):
            ReplyQualityRating(
                example_id="golden_0009",
                evaluator_type=EVALUATOR_HUMAN,
                groundedness=4,
                correctness=4,
                relevance=4,
                helpfulness=4,
                tone=4,
                overall_score=0.9,
            )

        with self.assertRaises(ValueError):
            ReplyQualityRating(
                example_id="golden_0009",
                evaluator_type=EVALUATOR_HUMAN,
                groundedness=4,
                correctness=4,
                relevance=4,
                helpfulness=4,
                tone=4,
                overall_score=5.1,
            )

    def test_serialization_and_deserialization(self):
        """Test round-trip serialization through to_dict and from_dict."""
        orig = ReplyQualityRating(
            example_id="golden_0010",
            evaluator_type=EVALUATOR_LLM_JUDGE,
            groundedness=4,
            correctness=5,
            relevance=4,
            helpfulness=3,
            tone=5,
            overall_score=4.2,
            notes="Automated judge rating with explanation.",
        )
        d = orig.to_dict()
        self.assertEqual(d["example_id"], "golden_0010")
        self.assertEqual(d["average_score"], 4.2)
        self.assertEqual(d["evaluator_type"], "llm_judge")

        restored = ReplyQualityRating.from_dict(d)
        self.assertEqual(orig, restored)

    def test_from_dict_missing_field_raises_key_error(self):
        """Test from_dict raises KeyError when required field is missing."""
        incomplete = {
            "example_id": "golden_0011",
            "evaluator_type": "human",
            "groundedness": 5,
            # missing correctness, relevance, helpfulness, tone
        }
        with self.assertRaises(KeyError):
            ReplyQualityRating.from_dict(incomplete)

    def test_evaluation_unit_instantiation_and_serialization(self):
        """Test EvaluationUnit schema creation, serialization, and deserialization."""
        unit = EvaluationUnit(
            example_id="golden_0050",
            customer_query="@AppleSupport my phone battery dies instantly after update",
            predicted_intent="battery_power",
            retrieved_evidence=[
                {"evidence_id": "res_123", "similarity_score": 0.45, "past_brand_resolution": "Try battery tips"}
            ],
            generated_reply={"reply_text": "Here are battery optimization tips", "grounded": True},
            conversation_id="apple_001",
        )
        self.assertEqual(unit.example_id, "golden_0050")
        self.assertEqual(unit.predicted_intent, "battery_power")
        self.assertEqual(len(unit.retrieved_evidence), 1)

        d = unit.to_dict()
        self.assertEqual(d["conversation_id"], "apple_001")

        restored = EvaluationUnit.from_dict(d)
        self.assertEqual(unit.example_id, restored.example_id)
        self.assertEqual(unit.customer_query, restored.customer_query)
        self.assertEqual(unit.predicted_intent, restored.predicted_intent)

    def test_evaluation_unit_validation(self):
        """Test EvaluationUnit catches missing or empty required parameters."""
        with self.assertRaises(ValueError):
            EvaluationUnit(
                example_id="",
                customer_query="query",
                predicted_intent="intent",
            )
        with self.assertRaises(ValueError):
            EvaluationUnit(
                example_id="ex_1",
                customer_query="",
                predicted_intent="intent",
            )


if __name__ == "__main__":
    unittest.main()
