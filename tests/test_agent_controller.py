"""Deterministic unit tests for AgentController state transitions, retries, and bounded execution."""

import unittest
from unittest.mock import MagicMock

from src.agent.agent_state import (
    ACTION_ASK_CLARIFICATION,
    ACTION_AUTO_HANDLE,
    ACTION_ESCALATE,
    SUFFICIENCY_BORDERLINE,
    SUFFICIENCY_EMPTY,
    SUFFICIENCY_INSUFFICIENT,
    SUFFICIENCY_SUFFICIENT,
    AgentState,
)
from src.agent.controller import AgentController
from src.agent.escalation_policy import EscalationDecision, EscalationEngine
from src.agent.grounded_generator import BaseReplyGenerator, MockReplyGenerator
from src.agent.reply_schemas import EvidenceItem, GenerationRequest, GroundedReply
from src.agent.risk_detector import RiskAssessment, RiskDetector
from src.retrieval.historical_resolution import HistoricalResolution, RetrievalResult


def _make_dummy_retrieval_result(
    res_id: str,
    score: float,
    customer_text: str = "Past customer problem",
    brand_text: str = "Past brand answer with official link https://support.apple.com/kb/HT1234",
    intent: str = "how_to_information",
) -> RetrievalResult:
    """Helper to create a valid mock RetrievalResult."""
    return RetrievalResult(
        resolution=HistoricalResolution(
            resolution_id=res_id,
            conversation_id="conv_001",
            customer_tweet_id="101",
            brand_tweet_id="102",
            customer_text=customer_text,
            brand_text=brand_text,
            intent=intent,
        ),
        score=score,
        rank=1,
    )


class DummyClassifier:
    """Deterministic classifier stub for testing."""

    def __init__(self, intent: str, confidence: float) -> None:
        self.classes_ = [intent, "other_unclear"]
        self.confidence = confidence

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[self.confidence, 1.0 - self.confidence]]


class TestAgentController(unittest.TestCase):
    """Test suite verifying agentic orchestration, state transitions, and bounded loops."""

    def test_critical_safety_short_circuit_before_retrieval(self):
        """Zero-tolerance physical safety hazard must immediately escalate to safety_team without retrieving."""
        mock_retriever = MagicMock()
        mock_generator = MagicMock()

        controller = AgentController(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        hazard_message = "@AppleSupport my iPhone battery is swollen, smoking, and caught on fire!"
        state = controller.process_query(hazard_message)

        # Assert immediate short-circuit
        self.assertEqual(state.final_action, ACTION_ESCALATE)
        self.assertEqual(state.target_queue, "safety_team")
        self.assertIn("physical_safety_hazard_detected", state.decision_reasons)
        self.assertIn("early_critical_hazard_short_circuit", state.decision_reasons)

        # Confirm retriever and generator were NEVER called (0 attempts)
        self.assertEqual(state.retrieval_attempts, 0)
        self.assertEqual(state.generation_attempts, 0)
        mock_retriever.retrieve.assert_not_called()
        mock_generator.generate.assert_not_called()

        # Check trace
        steps = [t["step"] for t in state.trace]
        self.assertIn("CRITICAL_SHORT_CIRCUIT", steps)

    def test_critical_security_short_circuit_before_retrieval(self):
        """Critical security incident (hacked/scam) must immediately escalate to account_security."""
        mock_retriever = MagicMock()
        mock_generator = MagicMock()

        controller = AgentController(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        security_message = "@AppleSupport my Apple ID was hacked and someone is stealing all my money!"
        state = controller.process_query(security_message)

        self.assertEqual(state.final_action, ACTION_ESCALATE)
        self.assertEqual(state.target_queue, "account_security")
        self.assertEqual(state.retrieval_attempts, 0)
        self.assertEqual(state.generation_attempts, 0)
        mock_retriever.retrieve.assert_not_called()

    def test_pre_generation_physical_damage_short_circuit(self):
        """Physical hardware break must short-circuit before generating a grounded self-service reply."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_dummy_retrieval_result("res_01", 0.45, intent="device_hardware")
        ]
        mock_generator = MagicMock()

        controller = AgentController(
            classifier=DummyClassifier("device_hardware", 0.95),
            retriever=mock_retriever,
            generator=mock_generator,
        )

        damage_message = "@AppleSupport I dropped my phone on concrete and now I have a cracked screen and broken glass"
        state = controller.process_query(damage_message)

        # Should retrieve evidence to assess, but short-circuit before generation
        self.assertEqual(state.final_action, ACTION_ESCALATE)
        self.assertEqual(state.target_queue, "hardware_repair")
        self.assertIn("escalation_engine_pre_generation_short_circuit", state.decision_reasons)
        self.assertEqual(state.retrieval_attempts, 1)
        self.assertEqual(state.generation_attempts, 0)
        mock_generator.generate.assert_not_called()

    def test_clean_self_service_auto_handle(self):
        """High-confidence intent + strong retrieval evidence + grounded URL yields AUTO_HANDLE."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_dummy_retrieval_result(
                "res_100",
                score=0.48,
                brand_text="Turn off DND While Driving here: https://support.apple.com/kb/HT208090",
                intent="how_to_information",
            )
        ]

        controller = AgentController(
            classifier=DummyClassifier("how_to_information", 0.88),
            retriever=mock_retriever,
            generator=MockReplyGenerator(),
        )

        query = "How do I disable Do Not Disturb while driving on my iPhone?"
        state = controller.process_query(query)

        self.assertEqual(state.final_action, ACTION_AUTO_HANDLE)
        self.assertEqual(state.target_queue, "self_service")
        self.assertEqual(state.evidence_sufficiency, SUFFICIENCY_SUFFICIENT)
        self.assertEqual(state.retrieval_attempts, 1)
        self.assertEqual(state.generation_attempts, 1)
        self.assertTrue(state.verification_passed)
        self.assertIsNotNone(state.generated_reply)
        self.assertTrue(state.generated_reply.grounded)
        self.assertIn("https://support.apple.com/kb/HT208090", state.generated_reply.used_urls)

    def test_borderline_retrieval_triggers_retry_and_succeeds(self):
        """Borderline retrieval score (0.25 <= sim < 0.35) triggers query augmentation retry and succeeds."""
        mock_retriever = MagicMock()

        # Call 1: borderline similarity (0.28)
        call_1_res = [_make_dummy_retrieval_result("res_init", 0.28, intent="battery_power")]
        # Call 2: augmented query produces strong similarity (0.42)
        call_2_res = [_make_dummy_retrieval_result("res_better", 0.42, intent="battery_power")]

        mock_retriever.retrieve.side_effect = [call_1_res, call_2_res]

        controller = AgentController(
            classifier=DummyClassifier("battery_power", 0.75),
            retriever=mock_retriever,
            generator=MockReplyGenerator(),
            borderline_retrieval_threshold=0.25,
            retrieval_similarity_threshold=0.35,
        )

        query = "battery discharging rapidly after update"
        state = controller.process_query(query)

        # Assert that retrieval was retried exactly once (2 attempts total)
        self.assertEqual(state.retrieval_attempts, 2)
        self.assertEqual(mock_retriever.retrieve.call_count, 2)

        # Assert trace contains explicit retry decision
        trace_steps = [t["step"] for t in state.trace]
        self.assertIn("DECIDE_RETRIEVAL_RETRY", trace_steps)
        self.assertIn("RETRIEVE_ATTEMPT_2", trace_steps)

        # Assert improved score was adopted
        self.assertEqual(state.top_similarity_score, 0.42)
        self.assertEqual(state.evidence_sufficiency, SUFFICIENCY_SUFFICIENT)
        self.assertEqual(state.final_action, ACTION_AUTO_HANDLE)

    def test_borderline_retrieval_retry_fails_and_escalates(self):
        """Borderline retrieval retry does not improve score; gracefully terminates at 2 attempts and escalates."""
        mock_retriever = MagicMock()

        # Both attempts return weak/borderline scores
        call_1_res = [_make_dummy_retrieval_result("res_1", 0.26, intent="connectivity")]
        call_2_res = [_make_dummy_retrieval_result("res_2", 0.27, intent="connectivity")]
        mock_retriever.retrieve.side_effect = [call_1_res, call_2_res]

        controller = AgentController(
            classifier=DummyClassifier("connectivity", 0.70),
            retriever=mock_retriever,
            generator=MockReplyGenerator(),
            borderline_retrieval_threshold=0.25,
            retrieval_similarity_threshold=0.35,
        )

        query = "My Wi-Fi connection drops intermittently when moving between different rooms in my house"
        state = controller.process_query(query)

        # Must strictly terminate at 2 attempts
        self.assertEqual(state.retrieval_attempts, 2)
        self.assertLess(state.top_similarity_score, 0.35)
        # Because retrieval remains below 0.35 threshold, it must not auto-handle
        self.assertEqual(state.final_action, ACTION_ESCALATE)
        self.assertEqual(state.target_queue, "general_support")

    def test_strict_bounded_attempts_limit(self):
        """Verify that max_attempts=2 is strictly enforced and never runs a 3rd attempt."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_dummy_retrieval_result("res_1", 0.29, intent="battery_power")
        ]

        controller = AgentController(
            classifier=DummyClassifier("battery_power", 0.80),
            retriever=mock_retriever,
            generator=MockReplyGenerator(),
            max_attempts=2,
            borderline_retrieval_threshold=0.25,
            retrieval_similarity_threshold=0.35,
        )

        state = controller.process_query("battery issue")
        self.assertLessEqual(state.retrieval_attempts, 2)
        self.assertLessEqual(state.generation_attempts, 2)

    def test_ask_clarification_as_first_class_action(self):
        """Short or vague inquiry with low confidence triggers ASK_CLARIFICATION action."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_dummy_retrieval_result("res_generic", 0.15, intent="device_hardware")
        ]

        controller = AgentController(
            classifier=DummyClassifier("device_hardware", 0.20),
            retriever=mock_retriever,
            generator=MockReplyGenerator(),
        )

        vague_query = "Need help with phone please"
        state = controller.process_query(vague_query)

        self.assertEqual(state.final_action, ACTION_ASK_CLARIFICATION)
        self.assertEqual(state.target_queue, "general_support")
        self.assertIn("vague_inquiry_requested_clarification", state.decision_reasons)
        self.assertIsNotNone(state.generated_reply)
        self.assertEqual(state.generated_reply.action, "ASK_CLARIFICATION")

    def test_grounding_verification_strips_fake_urls(self):
        """Generator outputting a fake URL has the link stripped by grounding guard."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_dummy_retrieval_result(
                "res_verified",
                0.40,
                brand_text="See official guide at https://support.apple.com/kb/HT9999",
                intent="icloud",
            )
        ]

        # Generator that outputs an ungrounded URL
        class HallucinatingGenerator(BaseReplyGenerator):
            def generate(self, request: GenerationRequest) -> GroundedReply:
                return GroundedReply(
                    reply_text="Click here to fix your account: https://fake-apple-phishing.com/fix",
                    grounded=True,
                    used_evidence_ids=["res_verified"],
                    used_urls=["https://fake-apple-phishing.com/fix"],
                    action="AUTO_REPLY",
                    rationale="Hallucinated reply",
                    provider="mock_bad",
                    latency_ms=10.0,
                )

        controller = AgentController(
            classifier=DummyClassifier("icloud", 0.90),
            retriever=mock_retriever,
            generator=HallucinatingGenerator(),
        )

        query = "How do I upgrade iCloud storage?"
        state = controller.process_query(query)

        # Grounding guard stripped the ungrounded URL
        self.assertNotIn("https://fake-apple-phishing.com/fix", state.generated_reply.used_urls)
        self.assertTrue(state.verification_details["urls_modified"])

    def test_error_handling_graceful_escalation(self):
        """Unexpected crash inside a component is caught gracefully and triggers fallback escalation."""
        crashing_retriever = MagicMock()
        crashing_retriever.retrieve.side_effect = RuntimeError("Disk I/O failure loading embeddings")

        controller = AgentController(
            classifier=DummyClassifier("device_hardware", 0.60),
            retriever=crashing_retriever,
            generator=MockReplyGenerator(),
        )

        state = controller.process_query("my phone won't turn on")

        self.assertEqual(state.final_action, ACTION_ESCALATE)
        self.assertEqual(state.target_queue, "general_support")
        self.assertTrue(any("controller_runtime_error" in r for r in state.decision_reasons))
        steps = [t["step"] for t in state.trace]
        self.assertIn("ERROR_FALLBACK", steps)

    def test_trace_and_state_observability(self):
        """Assert state and execution trace are fully populated and serializable."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_dummy_retrieval_result("res_1", 0.38, intent="how_to_information")
        ]

        controller = AgentController(
            classifier=DummyClassifier("how_to_information", 0.85),
            retriever=mock_retriever,
            generator=MockReplyGenerator(),
        )

        state = controller.process_query("How to adjust screen brightness?")
        d = state.to_dict()

        self.assertIn("customer_message", d)
        self.assertIn("predicted_intent", d)
        self.assertIn("trace", d)
        self.assertGreater(len(d["trace"]), 3)
        self.assertGreater(d["total_latency_ms"], 0.0)

    def test_offline_default_initialization(self):
        """Regression: Verify AgentController instantiates with default offline mock generator without error."""
        controller = AgentController()
        self.assertIsInstance(controller.generator, MockReplyGenerator)
        self.assertIsInstance(controller.risk_detector, RiskDetector)
        self.assertIsInstance(controller.escalation_engine, EscalationEngine)


if __name__ == "__main__":
    unittest.main()
