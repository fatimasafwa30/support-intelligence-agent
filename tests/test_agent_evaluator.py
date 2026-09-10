"""Unit tests for the deterministic AgentEvaluator harness (Milestone 18.1).

Verifies:
1. Strict anti-leakage: gold annotations NEVER enter AgentController.process_query().
2. Read-only immutability of the evaluator.
3. Deterministic reproducibility across runs (excluding volatile runtime latency/timestamps).
4. Primary 3-way operational action preservation & secondary binary mapping.
5. Dynamic metric calculation derived from loaded annotations without hardcoded constants.
6. Evidence URL citation verification isolation and framing.
7. Record serialization to dict and CSV row.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from src.agent.agent_state import (
    ACTION_ASK_CLARIFICATION,
    ACTION_AUTO_HANDLE,
    ACTION_ESCALATE,
    AgentState,
    SUFFICIENCY_BORDERLINE,
    SUFFICIENCY_SUFFICIENT,
)
from src.agent.controller import AgentController
from src.agent.reply_schemas import EvidenceItem, GroundedReply
from src.agent.risk_detector import RiskAssessment
from src.evaluation.agent_evaluator import AgentEvaluationRecord, AgentEvaluator


def _make_dummy_state(
    customer_message: str = "My phone won't charge",
    intent: str = "battery_power",
    confidence: float = 0.88,
    risk_level: str = "low",
    action: str = ACTION_AUTO_HANDLE,
    target_queue: str = "self_service",
    url_verified: bool = True,
    ungrounded_urls: list[str] | None = None,
) -> AgentState:
    """Helper creating a populated AgentState for testing evaluator logic."""
    state = AgentState(customer_message=customer_message, conversation_id="conv_100")
    state.predicted_intent = intent
    state.intent_confidence = confidence
    state.risk_assessment = RiskAssessment(
        risk_level=risk_level,
        risk_categories=[],
        matched_signals=[],
        rationale="Test risk assessment",
    )
    state.retrieval_attempts = 1
    state.retrieved_evidence = [
        EvidenceItem(
            evidence_id="ev_01",
            similarity_score=0.45,
            past_customer_problem="charging issue",
            past_brand_resolution="Check charging port",
            extracted_urls=["https://support.apple.com/kb/HT201569"],
            intent=intent,
        )
    ]
    state.top_similarity_score = 0.45
    state.evidence_sufficiency = SUFFICIENCY_SUFFICIENT
    state.generation_attempts = 1
    state.generated_reply = GroundedReply(
        reply_text="Clean your charging port: https://support.apple.com/kb/HT201569",
        grounded=True,
        used_evidence_ids=["ev_01"],
        used_urls=["https://support.apple.com/kb/HT201569"],
        action=action,
        rationale="Clear self-service guidance.",
        provider="mock",
        latency_ms=10.0,
    )
    state.verification_passed = url_verified
    state.verification_details = {
        "verified_urls": ["https://support.apple.com/kb/HT201569"],
        "ungrounded_urls": ungrounded_urls or [],
    }
    state.final_action = action
    state.target_queue = target_queue
    state.decision_reasons = ["high_confidence_match", "verified_grounded_reply"]
    state.record_step(
        step_name="RETRIEVE_ATTEMPT_1",
        action="EVALUATE_EVIDENCE",
        details={"top_similarity": 0.45, "sufficiency": SUFFICIENCY_SUFFICIENT},
    )
    state.complete()
    return state


class TestAgentEvaluator(unittest.TestCase):
    """Test suite for AgentEvaluator and anti-leakage protections."""

    def setUp(self):
        self.evaluator = AgentEvaluator()

    def test_strict_anti_leakage_spy(self):
        """Verify gold_intent, gold_risk, gold_action, and notes NEVER enter process_query()."""
        mock_controller = MagicMock(spec=AgentController)
        mock_controller.process_query.return_value = _make_dummy_state()

        sample_golden_row = {
            "id": "golden_candidate_0042",
            "conversation_id": "apple_conv_12345",
            "tweet_id": "999888777",
            "text": "@AppleSupport My iPhone 8 battery drops to 20% in an hour",
            "gold_intent": "battery_power",
            "gold_risk": "high",
            "gold_action": "ESCALATE",
            "annotation_notes": "Rapid unexpected drain requires diagnostic battery test.",
        }

        eval_record = self.evaluator.evaluate_record(sample_golden_row, mock_controller)

        # 1. Assert process_query was called exactly once
        self.assertEqual(mock_controller.process_query.call_count, 1)

        # 2. Inspect call arguments passed to process_query
        args, kwargs = mock_controller.process_query.call_args

        # Positional arguments check
        self.assertEqual(len(args), 0, "Expected all parameters to be passed by keyword")

        # Keyword arguments check: must strictly be customer_message and conversation_id
        allowed_kwargs = {"customer_message", "conversation_id"}
        actual_kwargs = set(kwargs.keys())
        self.assertEqual(
            actual_kwargs,
            allowed_kwargs,
            f"process_query received unexpected kwargs: {actual_kwargs - allowed_kwargs}",
        )
        self.assertEqual(kwargs["customer_message"], sample_golden_row["text"])
        self.assertEqual(kwargs["conversation_id"], sample_golden_row["conversation_id"])

        # 3. Explicit negative assertions: no gold fields passed
        for forbidden in ("gold_intent", "gold_risk", "gold_action", "annotation_notes"):
            self.assertNotIn(forbidden, kwargs)

        # 4. Assert ground-truth fields are attached only post-inference on the returned record
        self.assertEqual(eval_record.gold_intent, "battery_power")
        self.assertEqual(eval_record.gold_risk, "high")
        self.assertEqual(eval_record.gold_action, "ESCALATE")
        self.assertEqual(eval_record.annotation_notes, sample_golden_row["annotation_notes"])

    def test_read_only_evaluator_immutability(self):
        """Verify that evaluating a record leaves controller attributes and thresholds unchanged."""
        mock_controller = MagicMock(spec=AgentController)
        mock_controller.intent_confidence_threshold = 0.50
        mock_controller.retrieval_similarity_threshold = 0.35
        mock_controller.process_query.return_value = _make_dummy_state()

        sample_row = {
            "id": "golden_candidate_0001",
            "conversation_id": "apple_01",
            "text": "Help with my password reset",
            "gold_intent": "apple_id_account",
            "gold_risk": "medium",
            "gold_action": "AUTO_HANDLE",
        }

        self.evaluator.evaluate_record(sample_row, mock_controller)

        # Controller attributes must be identical
        self.assertEqual(mock_controller.intent_confidence_threshold, 0.50)
        self.assertEqual(mock_controller.retrieval_similarity_threshold, 0.35)

    def test_deterministic_reproducibility_excluding_latency(self):
        """Verify two evaluations of the same record produce identical predictions, excluding latency."""
        mock_controller = MagicMock(spec=AgentController)

        # Create two separate state objects simulating identical inference runs
        state1 = _make_dummy_state(intent="connectivity", confidence=0.74, action=ACTION_AUTO_HANDLE)
        state2 = _make_dummy_state(intent="connectivity", confidence=0.74, action=ACTION_AUTO_HANDLE)

        mock_controller.process_query.side_effect = [state1, state2]

        sample_row = {
            "id": "golden_candidate_0010",
            "conversation_id": "apple_10",
            "text": "Wi-Fi disconnecting repeatedly",
            "gold_intent": "connectivity",
            "gold_risk": "low",
            "gold_action": "AUTO_HANDLE",
        }

        rec1 = self.evaluator.evaluate_record(sample_row, mock_controller)
        rec2 = self.evaluator.evaluate_record(sample_row, mock_controller)

        # Assert identical deterministic operational outputs
        self.assertEqual(rec1.pred_intent, rec2.pred_intent)
        self.assertEqual(rec1.intent_confidence, rec2.intent_confidence)
        self.assertEqual(rec1.pred_risk_level, rec2.pred_risk_level)
        self.assertEqual(rec1.final_action, rec2.final_action)
        self.assertEqual(rec1.effective_binary_action, rec2.effective_binary_action)
        self.assertEqual(rec1.target_queue, rec2.target_queue)
        self.assertEqual(rec1.decision_reasons, rec2.decision_reasons)
        self.assertEqual(rec1.initial_top_similarity, rec2.initial_top_similarity)
        self.assertEqual(rec1.final_top_similarity, rec2.final_top_similarity)
        self.assertEqual(rec1.retrieval_attempts, rec2.retrieval_attempts)
        self.assertEqual(rec1.generation_attempts, rec2.generation_attempts)
        self.assertEqual(rec1.evidence_url_verification_passed, rec2.evidence_url_verification_passed)

        # Latency is runtime dependent and must NOT be required to be bitwise equal
        self.assertIsInstance(rec1.latency_ms, float)
        self.assertIsInstance(rec2.latency_ms, float)

    def test_primary_3way_and_secondary_binary_action_mapping(self):
        """Verify primary 3-way actions and secondary binary alignment mapping."""
        mock_controller = MagicMock(spec=AgentController)

        # Case 1: AUTO_HANDLE
        mock_controller.process_query.return_value = _make_dummy_state(action=ACTION_AUTO_HANDLE)
        row1 = {"id": "1", "conversation_id": "c1", "text": "q", "gold_intent": "i", "gold_risk": "l", "gold_action": "AUTO_HANDLE"}
        r1 = self.evaluator.evaluate_record(row1, mock_controller)
        self.assertEqual(r1.final_action, ACTION_AUTO_HANDLE)
        self.assertEqual(r1.effective_binary_action, ACTION_AUTO_HANDLE)

        # Case 2: ASK_CLARIFICATION
        mock_controller.process_query.return_value = _make_dummy_state(action=ACTION_ASK_CLARIFICATION)
        row2 = {"id": "2", "conversation_id": "c2", "text": "q", "gold_intent": "i", "gold_risk": "l", "gold_action": "ESCALATE"}
        r2 = self.evaluator.evaluate_record(row2, mock_controller)
        self.assertEqual(r2.final_action, ACTION_ASK_CLARIFICATION)
        self.assertEqual(r2.effective_binary_action, ACTION_ESCALATE)  # Deflected from auto

        # Case 3: ESCALATE
        mock_controller.process_query.return_value = _make_dummy_state(action=ACTION_ESCALATE)
        row3 = {"id": "3", "conversation_id": "c3", "text": "q", "gold_intent": "i", "gold_risk": "h", "gold_action": "ESCALATE"}
        r3 = self.evaluator.evaluate_record(row3, mock_controller)
        self.assertEqual(r3.final_action, ACTION_ESCALATE)
        self.assertEqual(r3.effective_binary_action, ACTION_ESCALATE)

    def test_dynamic_metric_derivation_from_annotations(self):
        """Verify that metric calculation derives totals from records without hardcoded constants."""
        # Create a synthetic dataset of 6 records with dynamic distributions:
        # - 2 Gold AUTO_HANDLE, 4 Gold ESCALATE
        # - 2 Gold critical risks
        records = [
            # 1: True Auto
            AgentEvaluationRecord(
                example_id="1", conversation_id="c1", tweet_id="", customer_message="m",
                gold_intent="icloud", gold_risk="low", gold_action="AUTO_HANDLE", annotation_notes="",
                pred_intent="icloud", intent_confidence=0.9, pred_risk_level="low", risk_categories=[],
                matched_risk_signals=[], retrieval_attempts=1, initial_top_similarity=0.4, final_top_similarity=0.4,
                initial_sufficiency=SUFFICIENCY_SUFFICIENT, final_sufficiency=SUFFICIENCY_SUFFICIENT,
                retrieval_retry_triggered=False, retrieval_retry_adopted=False, retrieved_evidence_ids=["e1"],
                generation_attempts=1, raw_reply_text="t", verified_reply_text="t", grounded_urls_count=1,
                ungrounded_urls_stripped=0, evidence_url_verification_passed=True, final_action=ACTION_AUTO_HANDLE,
                effective_binary_action=ACTION_AUTO_HANDLE, target_queue="self_service", decision_reasons=[],
                execution_steps=[], execution_steps_count=3, early_exit_step=None, latency_ms=10.0,
            ),
            # 2: False Escalate on Auto (clarification)
            AgentEvaluationRecord(
                example_id="2", conversation_id="c2", tweet_id="", customer_message="m",
                gold_intent="connectivity", gold_risk="low", gold_action="AUTO_HANDLE", annotation_notes="",
                pred_intent="connectivity", intent_confidence=0.4, pred_risk_level="low", risk_categories=[],
                matched_risk_signals=[], retrieval_attempts=1, initial_top_similarity=0.3, final_top_similarity=0.3,
                initial_sufficiency=SUFFICIENCY_BORDERLINE, final_sufficiency=SUFFICIENCY_BORDERLINE,
                retrieval_retry_triggered=False, retrieval_retry_adopted=False, retrieved_evidence_ids=["e2"],
                generation_attempts=0, raw_reply_text=None, verified_reply_text=None, grounded_urls_count=0,
                ungrounded_urls_stripped=0, evidence_url_verification_passed=False, final_action=ACTION_ASK_CLARIFICATION,
                effective_binary_action=ACTION_ESCALATE, target_queue="customer_clarification", decision_reasons=[],
                execution_steps=[], execution_steps_count=2, early_exit_step=None, latency_ms=12.0,
            ),
            # 3: True Escalate (Critical Safety)
            AgentEvaluationRecord(
                example_id="3", conversation_id="c3", tweet_id="", customer_message="phone smoking",
                gold_intent="device_hardware", gold_risk="critical", gold_action="ESCALATE", annotation_notes="",
                pred_intent="device_hardware", intent_confidence=0.9, pred_risk_level="critical", risk_categories=["SAFETY_HAZARD"],
                matched_risk_signals=["smoke"], retrieval_attempts=0, initial_top_similarity=0.0, final_top_similarity=0.0,
                initial_sufficiency="empty", final_sufficiency="empty",
                retrieval_retry_triggered=False, retrieval_retry_adopted=False, retrieved_evidence_ids=[],
                generation_attempts=0, raw_reply_text=None, verified_reply_text=None, grounded_urls_count=0,
                ungrounded_urls_stripped=0, evidence_url_verification_passed=False, final_action=ACTION_ESCALATE,
                effective_binary_action=ACTION_ESCALATE, target_queue="safety_team", decision_reasons=["early_critical_hazard_short_circuit"],
                execution_steps=[], execution_steps_count=2, early_exit_step="EARLY_CRITICAL_HAZARD", latency_ms=5.0,
            ),
            # 4: True Escalate (Critical Security)
            AgentEvaluationRecord(
                example_id="4", conversation_id="c4", tweet_id="", customer_message="hacked account",
                gold_intent="apple_id_account", gold_risk="critical", gold_action="ESCALATE", annotation_notes="",
                pred_intent="apple_id_account", intent_confidence=0.95, pred_risk_level="critical", risk_categories=["SECURITY_THEFT"],
                matched_risk_signals=["hacked"], retrieval_attempts=0, initial_top_similarity=0.0, final_top_similarity=0.0,
                initial_sufficiency="empty", final_sufficiency="empty",
                retrieval_retry_triggered=False, retrieval_retry_adopted=False, retrieved_evidence_ids=[],
                generation_attempts=0, raw_reply_text=None, verified_reply_text=None, grounded_urls_count=0,
                ungrounded_urls_stripped=0, evidence_url_verification_passed=False, final_action=ACTION_ESCALATE,
                effective_binary_action=ACTION_ESCALATE, target_queue="account_security", decision_reasons=["early_critical_hazard_short_circuit"],
                execution_steps=[], execution_steps_count=2, early_exit_step="EARLY_CRITICAL_HAZARD", latency_ms=5.0,
            ),
            # 5: True Escalate (General Support)
            AgentEvaluationRecord(
                example_id="5", conversation_id="c5", tweet_id="", customer_message="order issue",
                gold_intent="orders_delivery", gold_risk="medium", gold_action="ESCALATE", annotation_notes="",
                pred_intent="orders_delivery", intent_confidence=0.7, pred_risk_level="medium", risk_categories=[],
                matched_risk_signals=[], retrieval_attempts=2, initial_top_similarity=0.28, final_top_similarity=0.38,
                initial_sufficiency=SUFFICIENCY_BORDERLINE, final_sufficiency=SUFFICIENCY_SUFFICIENT,
                retrieval_retry_triggered=True, retrieval_retry_adopted=True, retrieved_evidence_ids=["e5"],
                generation_attempts=1, raw_reply_text="t", verified_reply_text="t", grounded_urls_count=1,
                ungrounded_urls_stripped=0, evidence_url_verification_passed=True, final_action=ACTION_ESCALATE,
                effective_binary_action=ACTION_ESCALATE, target_queue="general_support", decision_reasons=[],
                execution_steps=[], execution_steps_count=4, early_exit_step=None, latency_ms=18.0,
            ),
            # 6: Under-Escalation (False Auto on Escalate)
            AgentEvaluationRecord(
                example_id="6", conversation_id="c6", tweet_id="", customer_message="m",
                gold_intent="software_update", gold_risk="high", gold_action="ESCALATE", annotation_notes="",
                pred_intent="software_update", intent_confidence=0.85, pred_risk_level="low", risk_categories=[],
                matched_risk_signals=[], retrieval_attempts=1, initial_top_similarity=0.42, final_top_similarity=0.42,
                initial_sufficiency=SUFFICIENCY_SUFFICIENT, final_sufficiency=SUFFICIENCY_SUFFICIENT,
                retrieval_retry_triggered=False, retrieval_retry_adopted=False, retrieved_evidence_ids=["e6"],
                generation_attempts=1, raw_reply_text="t", verified_reply_text="t", grounded_urls_count=1,
                ungrounded_urls_stripped=0, evidence_url_verification_passed=True, final_action=ACTION_AUTO_HANDLE,
                effective_binary_action=ACTION_AUTO_HANDLE, target_queue="self_service", decision_reasons=[],
                execution_steps=[], execution_steps_count=3, early_exit_step=None, latency_ms=14.0,
            ),
        ]

        metrics = self.evaluator.compute_metrics(records)

        # Dynamic supports
        supports = metrics["gold_supports"]
        self.assertEqual(supports["gold_escalate_count"], 4)
        self.assertEqual(supports["gold_auto_count"], 2)
        self.assertEqual(supports["gold_critical_count"], 2)

        # Critical Safety: 2 of 2 criticals escalated = 100.0%
        safety = metrics["critical_safety_audit"]
        self.assertEqual(safety["critical_cases_total"], 2)
        self.assertEqual(safety["critical_cases_escalated"], 2)
        self.assertEqual(safety["critical_safety_recall"], 1.0)
        self.assertEqual(safety["missed_critical_count"], 0)

        # Secondary Binary Action:
        # TP (Esc->Esc) = 3 (cases 3, 4, 5)
        # FN (Esc->Auto) = 1 (case 6, under-escalation)
        # FP (Auto->Esc) = 1 (case 2)
        # TN (Auto->Auto) = 1 (case 1)
        # Total = 6, Correct = 4 / 6 = 66.67%
        s_bin = metrics["secondary_binary_action_metrics"]
        self.assertAlmostEqual(s_bin["action_accuracy"], 4 / 6, places=4)
        self.assertAlmostEqual(s_bin["under_escalation_rate"], 1 / 4, places=4)  # 1 FN / 4 Escalate
        self.assertAlmostEqual(s_bin["false_escalation_rate_on_auto"], 1 / 2, places=4)  # 1 FP / 2 Auto

        # Retrieval retry upgrade rate:
        # 2 cases were initially borderline (case 2, case 5). Case 5 upgraded to sufficient -> 1/2 = 50.0%
        ret_m = metrics["retrieval_and_retry_dynamics"]
        self.assertEqual(ret_m["borderline_cases_count"], 2)
        self.assertEqual(ret_m["borderline_upgraded_count"], 1)
        self.assertAlmostEqual(ret_m["borderline_upgrade_rate"], 0.5, places=4)

    def test_evidence_url_verification_metric_isolation(self):
        """Verify URL verification is correctly tracked as citation integrity, isolated from reply correctness."""
        # Record with a stripped hallucinated URL
        rec_stripped = AgentEvaluationRecord(
            example_id="1", conversation_id="c1", tweet_id="", customer_message="m",
            gold_intent="icloud", gold_risk="low", gold_action="AUTO_HANDLE", annotation_notes="",
            pred_intent="icloud", intent_confidence=0.9, pred_risk_level="low", risk_categories=[],
            matched_risk_signals=[], retrieval_attempts=1, initial_top_similarity=0.4, final_top_similarity=0.4,
            initial_sufficiency=SUFFICIENCY_SUFFICIENT, final_sufficiency=SUFFICIENCY_SUFFICIENT,
            retrieval_retry_triggered=False, retrieval_retry_adopted=False, retrieved_evidence_ids=["e1"],
            generation_attempts=1, raw_reply_text="text http://fake.com", verified_reply_text="text",
            grounded_urls_count=0, ungrounded_urls_stripped=1, evidence_url_verification_passed=False,
            final_action=ACTION_ASK_CLARIFICATION, effective_binary_action=ACTION_ESCALATE,
            target_queue="customer_clarification", decision_reasons=[], execution_steps=[],
            execution_steps_count=3, early_exit_step=None, latency_ms=10.0,
        )

        metrics = self.evaluator.compute_metrics([rec_stripped])
        url_m = metrics["evidence_url_citation_integrity"]
        self.assertEqual(url_m["generated_replies_count"], 1)
        self.assertEqual(url_m["url_verification_passed_count"], 0)
        self.assertEqual(url_m["evidence_url_verification_rate"], 0.0)
        self.assertEqual(url_m["hallucinated_urls_stripped_count"], 1)
        self.assertIn("clarification_note", url_m)

    def test_record_serialization_roundtrip(self):
        """Verify to_dict() and to_csv_row() serialize cleanly."""
        state = _make_dummy_state()
        mock_controller = MagicMock(spec=AgentController)
        mock_controller.process_query.return_value = state

        sample_row = {
            "id": "golden_candidate_0055",
            "conversation_id": "apple_55",
            "tweet_id": "55555",
            "text": "How do I update macOS?",
            "gold_intent": "software_update",
            "gold_risk": "low",
            "gold_action": "AUTO_HANDLE",
            "annotation_notes": "Routine update how-to.",
        }

        rec = self.evaluator.evaluate_record(sample_row, mock_controller)

        # 1. to_dict
        d = rec.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["example_id"], "golden_candidate_0055")
        self.assertEqual(d["gold_intent"], "software_update")
        self.assertEqual(d["final_action"], ACTION_AUTO_HANDLE)

        # 2. to_csv_row
        csv_row = rec.to_csv_row()
        self.assertIsInstance(csv_row, dict)
        self.assertEqual(csv_row["example_id"], "golden_candidate_0055")
        self.assertEqual(csv_row["primary_final_action"], ACTION_AUTO_HANDLE)
        self.assertEqual(csv_row["effective_binary_action"], ACTION_AUTO_HANDLE)

    def test_golden_set_loader_validates_required_columns(self):
        """Verify load_golden_set validates presence of required schema fields."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", encoding="utf-8") as tf:
            writer = csv.DictWriter(
                tf,
                fieldnames=["id", "conversation_id", "text", "gold_intent", "gold_risk", "gold_action"],
            )
            writer.writeheader()
            writer.writerow({
                "id": "g1",
                "conversation_id": "c1",
                "text": "query",
                "gold_intent": "battery_power",
                "gold_risk": "low",
                "gold_action": "AUTO_HANDLE",
            })
            temp_path = tf.name

        try:
            records = self.evaluator.load_golden_set(temp_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["id"], "g1")
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
