"""Unit tests for the LLM Judge reply-quality evaluation module.

All tests run completely hermetically offline without invoking any external LLM APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from src.evaluation.llm_judge import (
    BaseJudgeClient,
    GeminiJudgeClient,
    HumanAgreementSampler,
    JudgeConfig,
    JudgeEvaluationRecord,
    JudgePromptBuilder,
    JudgeResponseValidator,
    MockJudgeClient,
    ResumableJudgeRunner,
)
from src.evaluation.reply_quality_schema import (
    DIMENSION_CORRECTNESS,
    DIMENSION_GROUNDEDNESS,
    DIMENSION_HELPFULNESS,
    DIMENSION_RELEVANCE,
    DIMENSION_TONE,
    EVALUATOR_LLM_JUDGE,
    RUBRIC_DIMENSIONS,
    EvaluationUnit,
    ReplyQualityRating,
)


class TestLLMJudge(unittest.TestCase):
    """Hermetic unit tests for LLM Judge schema, prompt builder, validator, and runner."""

    def setUp(self) -> None:
        """Create sample evaluation units for testing."""
        self.sample_unit = EvaluationUnit(
            example_id="golden_0001",
            customer_query="@AppleSupport iPhone battery draining quickly after iOS 11.1 update",
            predicted_intent="battery_power",
            intent_confidence=0.9250,
            retrieved_evidence=[
                {
                    "evidence_id": "res_101",
                    "similarity_score": 0.4850,
                    "past_customer_problem": "Battery drain after update",
                    "past_brand_resolution": "Try checking Settings > Battery and restart your device: https://support.apple.com/battery",
                    "extracted_urls": ["https://support.apple.com/battery"],
                    "intent": "battery_power",
                }
            ],
            generated_reply={
                "reply_text": "We can help. Check Settings > Battery for usage tips: https://support.apple.com/battery",
                "grounded": True,
                "used_evidence_ids": ["res_101"],
            },
            verified_reply={
                "reply_text": "We can help. Check Settings > Battery for usage tips: https://support.apple.com/battery",
                "grounded": True,
            },
            conversation_id="apple_conv_001",
        )

    @staticmethod
    def gemini_response(text, finish_reason="STOP"):
        return Mock(status_code=200, json=Mock(return_value={
            "candidates": [{"finishReason": finish_reason,
                            "content": {"parts": [{"text": text}]}}],
            "usageMetadata": {"thoughtsTokenCount": 1000},
        }))

    @patch("src.evaluation.llm_judge.requests.post")
    def test_gemini_native_schema_and_valid_response(self, post):
        scores = dict(zip(RUBRIC_DIMENSIONS, (2, 4, 3, 4, 5)))
        post.return_value = self.gemini_response(json.dumps(scores))
        record = GeminiJudgeClient("offline-key").evaluate_unit(self.sample_unit)
        post.assert_called_once()
        config = post.call_args.kwargs["json"]["generationConfig"]
        self.assertEqual(config["temperature"], 0)
        self.assertEqual(config["maxOutputTokens"], 1024)
        self.assertEqual(config["thinkingConfig"], {"thinkingBudget": 0})
        self.assertEqual(config["responseMimeType"], "application/json")
        schema = config["responseJsonSchema"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], list(RUBRIC_DIMENSIONS))
        self.assertEqual(schema["propertyOrdering"], list(RUBRIC_DIMENSIONS))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"], {
            dim: {"type": "integer", "minimum": 1, "maximum": 5}
            for dim in RUBRIC_DIMENSIONS
        })
        self.assertEqual(record.status, "SUCCESS")
        self.assertEqual(record.average_score, 3.6)
        self.assertEqual(record.overall_score, 3.6)
        self.assertEqual(record.rationales, {})
        self.assertNotIn("rationales", JudgePromptBuilder.JSON_SCHEMA_INSTRUCTION)
        self.assertNotIn("overall_score", JudgePromptBuilder.JSON_SCHEMA_INSTRUCTION)

    @patch("src.evaluation.llm_judge.time.sleep")
    @patch("src.evaluation.llm_judge.requests.post")
    def test_gemini_invalid_responses_exhaust_bounded_retries(self, post, sleep):
        scores = dict.fromkeys(RUBRIC_DIMENSIONS, 4)
        cases = [(self.gemini_response('{"groundedness": 2,'), "Malformed JSON")]
        for dim in RUBRIC_DIMENSIONS:
            for bad_score in (0, 6, 4.5, True):
                cases.append((self.gemini_response(json.dumps({**scores, dim: bad_score})), dim))
            cases.append((self.gemini_response(json.dumps({
                key: value for key, value in scores.items() if key != dim
            })), "Missing mandatory dimension"))
        cases.append((self.gemini_response(json.dumps(scores), "MAX_TOKENS"), "MAX_TOKENS"))
        for response, error in cases:
            with self.subTest(error=error, response=response.json()):
                post.reset_mock()
                sleep.reset_mock()
                post.return_value = response
                record = GeminiJudgeClient("offline-key", JudgeConfig(
                    max_retries=3, retry_base_delay=0.25,
                )).evaluate_unit(self.sample_unit)
                self.assertEqual(record.status, "FAILED")
                self.assertIn(error, record.error_message)
                self.assertEqual(post.call_count, 3)
                self.assertEqual(sleep.call_args_list, [call(0.25), call(0.5), call(1.0)])

    @patch("src.evaluation.llm_judge.requests.post")
    def test_gemini_joins_text_parts_without_thoughts(self, post):
        post.return_value = Mock(status_code=200, json=Mock(return_value={
            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"thought": True, "text": "Private reasoning"},
                {"text": '{"groundedness": 2, "correctness": 4,'},
                {"text": '"relevance": 3, "helpfulness": 4, "tone": 5}'},
            ]}}],
        }))
        self.assertEqual(GeminiJudgeClient("offline-key").evaluate_unit(self.sample_unit).status, "SUCCESS")

    @patch("src.evaluation.llm_judge.requests.post")
    def test_actual_gemini_payload_excludes_other_metadata(self, post):
        data = self.sample_unit.to_dict()
        data.update({"gold_custom_label": "TOP_SECRET_GOLD", "gold_intent": "TOP_SECRET_INTENT"})
        data["conversation_id"] = "PRIVATE_CONVERSATION"
        data["retrieved_evidence"][0]["other_golden_label"] = "PRIVATE_EVIDENCE_LABEL"
        data["generated_reply"]["other_golden_label"] = "PRIVATE_REPLY_LABEL"
        data["verified_reply"]["verification_details"] = "PRIVATE_VERIFICATION"
        data["verified_reply"]["reply_text"] = "Distinct verified reply"
        post.return_value = self.gemini_response(json.dumps(dict.fromkeys(RUBRIC_DIMENSIONS, 4)))
        GeminiJudgeClient("offline-key").evaluate_unit(EvaluationUnit.from_dict(data))
        wire = json.dumps(post.call_args.kwargs["json"])
        for secret in ("TOP_SECRET", "PRIVATE_", "gold_intent", "gold_custom_label", "other_golden_label"):
            self.assertNotIn(secret, wire)
        for value in (self.sample_unit.example_id, self.sample_unit.customer_query,
                      self.sample_unit.predicted_intent, "0.9250", "res_101",
                      self.sample_unit.generated_reply["reply_text"], "Distinct verified reply"):
            self.assertIn(value, wire)

    @patch("src.evaluation.llm_judge.requests.post")
    def test_gemini_rejects_nested_golden_labels_before_request(self, post):
        for key in ("gold_intent", "gold_risk", "gold_action", "annotation_notes"):
            with self.subTest(key=key):
                data = self.sample_unit.to_dict()
                data["retrieved_evidence"][0][key] = "FORBIDDEN"
                with self.assertRaisesRegex(ValueError, "ANTI-LEAKAGE"):
                    GeminiJudgeClient("offline-key").evaluate_unit(EvaluationUnit.from_dict(data))
        post.assert_not_called()

    @patch("src.evaluation.llm_judge.time.sleep")
    @patch("src.evaluation.llm_judge.requests.post")
    def test_gemini_resume_retries_failed_records_and_preserves_bytes(self, post, sleep):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = JudgeConfig(
                output_records_path=Path(tmpdir) / "records.jsonl",
                output_results_path=Path(tmpdir) / "results.json",
                output_report_path=Path(tmpdir) / "report.md",
                max_retries=1, request_delay_seconds=0.75,
            )
            runner = ResumableJudgeRunner(GeminiJudgeClient("offline-key", config), config)
            post.return_value = self.gemini_response('{"groundedness": 2,')
            runner.run_evaluation([self.sample_unit])
            original = config.output_records_path.read_bytes()
            post.return_value = self.gemini_response(json.dumps(dict.fromkeys(RUBRIC_DIMENSIONS, 4)))
            runner.run_evaluation([self.sample_unit])
            updated = config.output_records_path.read_bytes()
            self.assertTrue(updated.startswith(original))
            self.assertEqual([json.loads(line)["status"] for line in updated.splitlines()], ["FAILED", "SUCCESS"])
            runner.run_evaluation([self.sample_unit])
            self.assertEqual(post.call_count, 2)
            self.assertEqual(config.output_records_path.read_bytes(), updated)
            self.assertEqual(sleep.call_args_list.count(call(0.75)), 2)

    def test_1_valid_structured_score_accepted(self):
        """Test that a well-formed JSON score is parsed and accepted."""
        valid_json = json.dumps({
            "groundedness": 5,
            "correctness": 4,
            "relevance": 5,
            "helpfulness": 4,
            "tone": 5,
            "overall_score": 4.6,
            "rationales": {
                "groundedness": "Matches retrieved evidence completely.",
                "correctness": "Accurate battery troubleshooting advice.",
                "relevance": "Directly addresses iOS 11 battery drain.",
                "helpfulness": "Actionable settings inspection step.",
                "tone": "Polite and brand-aligned.",
            },
        })
        record = JudgeResponseValidator.parse_and_validate(
            raw_text=valid_json,
            example_id="golden_0001",
            judge_model="gemini-2.5-flash",
            latency_ms=120.5,
        )
        self.assertEqual(record.example_id, "golden_0001")
        self.assertEqual(record.groundedness, 5)
        self.assertEqual(record.correctness, 4)
        self.assertEqual(record.relevance, 5)
        self.assertEqual(record.helpfulness, 4)
        self.assertEqual(record.tone, 5)
        self.assertEqual(record.overall_score, 4.6)
        self.assertAlmostEqual(record.average_score, 4.6, places=4)
        self.assertEqual(record.status, "SUCCESS")
        self.assertIsNone(record.error_message)

    def test_2_score_outside_1_to_5_rejected(self):
        """Test that scores < 1 or > 5 raise ValueError."""
        invalid_low = json.dumps({
            "groundedness": 0,  # Below 1
            "correctness": 4,
            "relevance": 5,
            "helpfulness": 4,
            "tone": 5,
        })
        with self.assertRaises(ValueError):
            JudgeResponseValidator.parse_and_validate(invalid_low, "ex_1", "gemini-2.5-flash")

        invalid_high = json.dumps({
            "groundedness": 5,
            "correctness": 6,  # Above 5
            "relevance": 5,
            "helpfulness": 4,
            "tone": 5,
        })
        with self.assertRaises(ValueError):
            JudgeResponseValidator.parse_and_validate(invalid_high, "ex_2", "gemini-2.5-flash")

    def test_3_float_and_boolean_scores_rejected(self):
        """Test that floating-point and boolean scores for dimensions raise TypeError."""
        float_score = json.dumps({
            "groundedness": 4.5,  # Float not allowed for discrete dimensions
            "correctness": 4,
            "relevance": 5,
            "helpfulness": 4,
            "tone": 5,
        })
        with self.assertRaises(TypeError):
            JudgeResponseValidator.parse_and_validate(float_score, "ex_3", "gemini-2.5-flash")

        bool_score = json.dumps({
            "groundedness": True,  # Bool rejected
            "correctness": 4,
            "relevance": 5,
            "helpfulness": 4,
            "tone": 5,
        })
        with self.assertRaises(TypeError):
            JudgeResponseValidator.parse_and_validate(bool_score, "ex_4", "gemini-2.5-flash")

    def test_4_missing_dimension_rejected(self):
        """Test that missing mandatory dimensions raise KeyError."""
        missing_tone = json.dumps({
            "groundedness": 5,
            "correctness": 4,
            "relevance": 5,
            "helpfulness": 4,
            # tone missing
        })
        with self.assertRaises(KeyError):
            JudgeResponseValidator.parse_and_validate(missing_tone, "ex_5", "gemini-2.5-flash")

    def test_5_golden_labels_cannot_enter_judge_payload(self):
        """Test that anti-leakage verification triggers if golden labels are present in payload."""
        leak_payload = {
            "example_id": "golden_0001",
            "customer_query": "Battery drain",
            "gold_intent": "battery_power",  # Forbidden
        }
        with self.assertRaises(ValueError) as ctx:
            JudgePromptBuilder.verify_anti_leakage(leak_payload)
        self.assertIn("ANTI-LEAKAGE SECURITY VIOLATION", str(ctx.exception))

        nested_leak = {
            "example_id": "golden_0002",
            "customer_query": "Restart loop",
            "generated_reply": {"reply_text": "Restart", "gold_action": "ESCALATE"},  # Forbidden nested
        }
        with self.assertRaises(ValueError) as ctx:
            JudgePromptBuilder.verify_anti_leakage(nested_leak)
        self.assertIn("ANTI-LEAKAGE SECURITY VIOLATION", str(ctx.exception))

    def test_6_judge_input_contains_required_fields(self):
        """Test that prompt builder formats all required fields cleanly without leakage."""
        prompt = JudgePromptBuilder.build_prompt(self.sample_unit)
        self.assertIn("system_instruction", prompt)
        self.assertIn("user_prompt", prompt)
        user_p = prompt["user_prompt"]
        self.assertIn(self.sample_unit.customer_query, user_p)
        self.assertIn("battery_power", user_p)
        self.assertIn("0.9250", user_p)
        self.assertIn("res_101", user_p)
        self.assertIn("https://support.apple.com/battery", user_p)
        self.assertNotIn("gold_intent", user_p)
        self.assertNotIn("gold_action", user_p)
        self.assertNotIn("gold_risk", user_p)

    def test_7_judge_output_can_be_serialized(self):
        """Test serialization and deserialization of JudgeEvaluationRecord and conversion to ReplyQualityRating."""
        record = JudgeEvaluationRecord(
            example_id="golden_0010",
            evaluator_type=EVALUATOR_LLM_JUDGE,
            judge_model="gemini-2.5-flash",
            groundedness=4,
            correctness=5,
            relevance=5,
            helpfulness=4,
            tone=5,
            overall_score=4.6,
            average_score=4.6,
            rationales={"groundedness": "Good link"},
            status="SUCCESS",
        )
        d = record.to_dict()
        restored = JudgeEvaluationRecord.from_dict(d)
        self.assertEqual(record, restored)

        rating_model = record.to_rating_schema()
        self.assertIsInstance(rating_model, ReplyQualityRating)
        self.assertEqual(rating_model.example_id, "golden_0010")
        self.assertEqual(rating_model.groundedness, 4)

    def test_8_resume_logic_skips_already_completed_examples(self):
        """Test that ResumableJudgeRunner skips already evaluated items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "records.jsonl"
            existing_rec = JudgeEvaluationRecord(
                example_id="golden_0001",
                evaluator_type=EVALUATOR_LLM_JUDGE,
                judge_model="mock-judge",
                groundedness=5,
                correctness=5,
                relevance=5,
                helpfulness=5,
                tone=5,
                overall_score=5.0,
                average_score=5.0,
                status="SUCCESS",
            )
            with jsonl_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(existing_rec.to_dict()) + "\n")

            config = JudgeConfig(
                output_records_path=jsonl_path,
                output_results_path=Path(tmpdir) / "results.json",
                output_report_path=Path(tmpdir) / "report.md",
                request_delay_seconds=0.0,
            )
            mock_client = MockJudgeClient()
            runner = ResumableJudgeRunner(client=mock_client, config=config)

            # Two units: golden_0001 (already present) and golden_0002 (new)
            unit_2 = EvaluationUnit(
                example_id="golden_0002",
                customer_query="Wifi connection drop",
                predicted_intent="connectivity",
                generated_reply={"reply_text": "Reset network settings", "grounded": True},
            )
            runner.run_evaluation([self.sample_unit, unit_2], total_golden_size=250)

            # Client should have been called only once for golden_0002
            self.assertEqual(mock_client.call_count, 1)

    def test_9_failed_request_does_not_corrupt_existing_results(self):
        """Test that a failed request writes FAILED status without overwriting or corrupting valid records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "records.jsonl"
            config = JudgeConfig(
                output_records_path=jsonl_path,
                output_results_path=Path(tmpdir) / "results.json",
                output_report_path=Path(tmpdir) / "report.md",
                request_delay_seconds=0.0,
            )
            mock_client = MockJudgeClient(simulate_failure_ids={"golden_fail_01"})
            runner = ResumableJudgeRunner(client=mock_client, config=config)

            fail_unit = EvaluationUnit(
                example_id="golden_fail_01",
                customer_query="Random bug",
                predicted_intent="other_unclear",
                generated_reply={"reply_text": "Clarify please", "grounded": False},
            )
            metrics = runner.run_evaluation([self.sample_unit, fail_unit], total_golden_size=250)
            pop = metrics["evaluation_population"]
            self.assertEqual(pop["successfully_judged_count"], 1)
            self.assertEqual(pop["failed_judging_count"], 1)

    def test_10_retry_count_is_bounded_and_gemini_key_required(self):
        """Test that Gemini client requires an API key and configures bounded retries."""
        with self.assertRaises(ValueError):
            GeminiJudgeClient(api_key="")

        cfg = JudgeConfig(max_retries=3, retry_base_delay=0.01)
        client = GeminiJudgeClient(api_key="mock_key_123", config=cfg)
        self.assertEqual(client.config.max_retries, 3)

    def test_11_programmatic_url_verification_remains_separate(self):
        """Test that programmatic URL citation status is distinct from semantic groundedness."""
        verified_unit = EvaluationUnit(
            example_id="golden_0020",
            customer_query="Payment query",
            predicted_intent="billing_payments",
            generated_reply={"reply_text": "Check itunes: http://apple.co/pay", "grounded": True},
            verified_reply={"reply_text": "Check itunes: http://apple.co/pay", "grounded": True},
        )
        mock_client = MockJudgeClient()
        rec = mock_client.evaluate_unit(verified_unit)
        self.assertEqual(rec.groundedness, 5)

    def test_12_generation_coverage_135_vs_115_short_circuits(self):
        """Test that aggregate metrics calculate coverage correctly over 135 generated vs 115 short-circuits."""
        successful_records = [
            JudgeEvaluationRecord(
                example_id=f"golden_{i:04d}",
                evaluator_type=EVALUATOR_LLM_JUDGE,
                judge_model="gemini-2.5-flash",
                groundedness=4,
                correctness=4,
                relevance=5,
                helpfulness=4,
                tone=5,
                overall_score=4.4,
                average_score=4.4,
            )
            for i in range(135)
        ]
        metrics = ResumableJudgeRunner.compute_aggregate_metrics(
            successful_records=successful_records,
            total_generated=135,
            total_golden=250,
            failed_count=0,
        )
        pop = metrics["evaluation_population"]
        self.assertEqual(pop["total_golden_examples"], 250)
        self.assertEqual(pop["generated_replies_evaluated"], 135)
        self.assertEqual(pop["non_generated_short_circuits"], 115)
        self.assertEqual(pop["reply_generation_coverage"], 0.540)
        self.assertEqual(pop["successfully_judged_count"], 135)
        self.assertEqual(pop["failed_judging_count"], 0)

    def test_13_deterministic_human_agreement_sampler(self):
        """Test that human agreement sampler selects 50 examples deterministically without judge scores."""
        units = [
            EvaluationUnit(
                example_id=f"candidate_{i:04d}",
                customer_query=f"Query {i}",
                predicted_intent="battery_power" if i % 2 == 0 else "software_update",
                intent_confidence=0.85 if i % 3 == 0 else (0.65 if i % 3 == 1 else 0.35),
                generated_reply={"reply_text": f"Reply {i}", "grounded": bool(i % 2 == 0)},
            )
            for i in range(135)
        ]
        sample_1 = HumanAgreementSampler.sample_human_subset(units, sample_size=50, random_seed=42)
        sample_2 = HumanAgreementSampler.sample_human_subset(units, sample_size=50, random_seed=42)

        self.assertEqual(len(sample_1), 50)
        self.assertEqual(sample_1, sample_2)  # Deterministic with seed
        # Ensure no golden labels in sample
        for item in sample_1:
            self.assertNotIn("gold_intent", item)
            self.assertNotIn("gold_action", item)
            self.assertNotIn("gold_risk", item)


if __name__ == "__main__":
    unittest.main()
