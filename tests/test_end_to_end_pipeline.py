"""End-to-end audit and validation test suite for the full support intelligence agent pipeline.

Validates all 8 core lifecycle scenarios across the complete controller pipeline:
1. Clear low-risk support query (AUTO_HANDLE)
2. Ambiguous low-confidence query (Calibrated Abstention to other_unclear -> ASK_CLARIFICATION)
3. Query that triggers clarification (brief inquiry with uncertain intent/retrieval)
4. Query that escalates (complex repair/technical dispute requiring human booking)
5. Query with weak/no retrieval evidence (out-of-domain query)
6. Query with conflicting/hallucinated evidence (GroundingGuard URL stripping)
7. Risky/security-related query (pre-retrieval critical hazard short-circuit)
8. Normal query with high-confidence intent and good grounding (AUTO_HANDLE)

For every scenario, asserts:
- predicted intent
- classifier confidence
- abstention status
- retrieved evidence
- generated response
- grounding checks
- escalation/clarification reason
- final action
"""

import unittest

from src.agent.agent_state import (
    ACTION_ASK_CLARIFICATION,
    ACTION_AUTO_HANDLE,
    ACTION_ESCALATE,
    SUFFICIENCY_EMPTY,
    SUFFICIENCY_SUFFICIENT,
)
from src.agent.controller import AgentController
from src.agent.grounded_generator import BaseReplyGenerator
from src.agent.reply_schemas import GenerationRequest, GroundedReply


class TestEndToEndPipeline(unittest.TestCase):
    """Full end-to-end integration and policy tests for the agent pipeline."""

    @classmethod
    def setUpClass(cls):
        """Initialize real controller with frozen artifacts and calibrated abstention."""
        cls.controller = AgentController(abstention_threshold=0.25)

    def test_scenario_1_clear_low_risk_support_query(self):
        """Scenario 1: Clear low-risk support query achieves AUTO_HANDLE with verified link."""
        query = "How do I disable Do Not Disturb while driving on my iPhone?"
        state = self.controller.process_query(query)

        # 1. Predicted intent
        self.assertEqual(state.predicted_intent, "how_to_information")
        # 2. Classifier confidence
        self.assertIsNotNone(state.intent_confidence)
        self.assertGreaterEqual(state.intent_confidence, 0.50)
        # 3. Abstention status
        self.assertFalse(state.is_abstained)
        # 4. Retrieved evidence
        self.assertGreaterEqual(state.retrieval_attempts, 1)
        self.assertGreater(len(state.retrieved_evidence), 0)
        self.assertEqual(state.evidence_sufficiency, SUFFICIENCY_SUFFICIENT)
        self.assertGreaterEqual(state.top_similarity_score, 0.35)
        # 5. Generated response
        self.assertIsNotNone(state.generated_reply)
        self.assertEqual(state.generated_reply.action, "AUTO_REPLY")
        self.assertTrue(len(state.generated_reply.used_urls) > 0)
        # 6. Grounding checks
        self.assertTrue(state.verification_passed)
        self.assertFalse(state.verification_details.get("urls_modified", False))
        # 7. Escalation/clarification reason
        self.assertIn("verified_official_self_service_resolution_available", state.decision_reasons)
        self.assertIn("self_service", state.target_queue)
        # 8. Final action
        self.assertEqual(state.final_action, ACTION_AUTO_HANDLE)

    def test_scenario_2_ambiguous_low_confidence_query(self):
        """Scenario 2: Ambiguous low-confidence query triggers calibrated abstention to other_unclear."""
        query = "hello please help me"
        state = self.controller.process_query(query)

        # 1. Predicted intent
        self.assertEqual(state.predicted_intent, "other_unclear")
        # 2. Classifier confidence
        self.assertIsNotNone(state.intent_confidence)
        self.assertLess(state.intent_confidence, 0.25)
        # 3. Abstention status
        self.assertTrue(state.is_abstained)
        # 4. Retrieved evidence
        self.assertGreaterEqual(state.retrieval_attempts, 1)
        # 5. Generated response
        self.assertIsNotNone(state.generated_reply)
        # 6. Grounding checks
        self.assertIsNotNone(state.verification_details)
        # 7. Escalation/clarification reason
        self.assertIn("unclear_intent_requires_resolution", state.decision_reasons)
        self.assertIn("vague_inquiry_requested_clarification", state.decision_reasons)
        self.assertEqual(state.target_queue, "general_support")
        # 8. Final action
        self.assertEqual(state.final_action, ACTION_ASK_CLARIFICATION)

    def test_scenario_3_query_that_should_trigger_clarification(self):
        """Scenario 3: Short technical inquiry with uncertain confidence triggers ASK_CLARIFICATION."""
        query = "my phone is acting weird"
        state = self.controller.process_query(query)

        # 1. Predicted intent
        self.assertIn(state.predicted_intent, ["device_hardware", "other_unclear"])
        # 2. Classifier confidence
        self.assertIsNotNone(state.intent_confidence)
        self.assertLess(state.intent_confidence, 0.50)
        # 3. Abstention status
        self.assertIsInstance(state.is_abstained, bool)
        # 4. Retrieved evidence
        self.assertGreaterEqual(state.retrieval_attempts, 1)
        # 5. Generated response
        self.assertIsNotNone(state.generated_reply)
        self.assertEqual(state.generated_reply.action, "ASK_CLARIFICATION")
        # 6. Grounding checks
        self.assertIsNotNone(state.verification_details)
        # 7. Escalation/clarification reason
        self.assertIn("vague_inquiry_requested_clarification", state.decision_reasons)
        # 8. Final action
        self.assertEqual(state.final_action, ACTION_ASK_CLARIFICATION)
        self.assertEqual(state.target_queue, "general_support")

    def test_scenario_4_query_that_should_escalate(self):
        """Scenario 4: High-touch repair dispute inquiry mandates human booking escalation."""
        query = (
            "I sent my MacBook Pro in for repair 3 times already for kernel panics "
            "and logic board failure, case ID 849204, and the issue is still not resolved."
        )
        state = self.controller.process_query(query)

        # 1. Predicted intent
        self.assertEqual(state.predicted_intent, "repair_service")
        # 2. Classifier confidence
        self.assertIsNotNone(state.intent_confidence)
        # 3. Abstention status
        self.assertFalse(state.is_abstained)
        # 4. Retrieved evidence
        self.assertEqual(state.retrieval_attempts, 1)
        # 5. Generated response (pre-generation short-circuit ensures no false reply is sent)
        self.assertEqual(state.generation_attempts, 0)
        # 6. Grounding checks
        self.assertFalse(state.verification_passed)
        # 7. Escalation/clarification reason
        self.assertIn("repair_service_mandates_human_booking", state.decision_reasons)
        self.assertIn("escalation_engine_pre_generation_short_circuit", state.decision_reasons)
        # 8. Final action
        self.assertEqual(state.final_action, ACTION_ESCALATE)
        self.assertEqual(state.target_queue, "hardware_repair")

    def test_scenario_5_query_with_weak_or_no_retrieval_evidence(self):
        """Scenario 5: Out-of-domain inquiry with no matching evidence avoids auto-handling."""
        query = "xyzzy qwerty flimflam quantum superconductor"
        state = self.controller.process_query(query)

        # 1. Predicted intent
        self.assertEqual(state.predicted_intent, "other_unclear")
        # 2. Classifier confidence
        self.assertLess(state.intent_confidence, 0.25)
        # 3. Abstention status
        self.assertTrue(state.is_abstained)
        # 4. Retrieved evidence (empty / zero similarity)
        self.assertEqual(state.top_similarity_score, 0.0)
        self.assertEqual(state.evidence_sufficiency, SUFFICIENCY_EMPTY)
        # 5. Generated response
        self.assertIsNotNone(state.generated_reply)
        self.assertEqual(state.generated_reply.action, "ASK_CLARIFICATION")
        # 6. Grounding checks
        self.assertEqual(len(state.generated_reply.used_urls), 0)
        # 7. Escalation/clarification reason
        self.assertIn("retrieval_similarity_below_threshold (0.000 < 0.35)", state.decision_reasons)
        # 8. Final action
        self.assertEqual(state.final_action, ACTION_ASK_CLARIFICATION)

    def test_scenario_6_query_with_conflicting_or_hallucinated_evidence(self):
        """Scenario 6: Grounding guard strips hallucinated external URL and blocks auto-handling."""
        class HallucinatingGenerator(BaseReplyGenerator):
            def generate(self, request: GenerationRequest) -> GroundedReply:
                return GroundedReply(
                    reply_text="Click here to fix your issue: https://fake-apple-phishing.com/fix",
                    grounded=True,
                    used_evidence_ids=["ev_1"],
                    used_urls=["https://fake-apple-phishing.com/fix"],
                    action="AUTO_REPLY",
                    rationale="Hallucinated link",
                    provider="mock_bad",
                    latency_ms=5.0,
                )

        custom_controller = AgentController(
            generator=HallucinatingGenerator(),
            abstention_threshold=0.25,
        )

        query = "How do I update to iOS 11 software?"
        state = custom_controller.process_query(query)

        # 1. Predicted intent
        self.assertEqual(state.predicted_intent, "software_update")
        # 2. Classifier confidence
        self.assertGreaterEqual(state.intent_confidence, 0.50)
        # 3. Abstention status
        self.assertFalse(state.is_abstained)
        # 4. Retrieved evidence
        self.assertGreater(len(state.retrieved_evidence), 0)
        # 5. Generated response (hallucinated URL was stripped)
        self.assertNotIn("https://fake-apple-phishing.com/fix", state.generated_reply.used_urls)
        # 6. Grounding checks
        self.assertTrue(state.verification_details["urls_modified"])
        # 7. Escalation reason
        self.assertIn("lacks_verified_self_service_link_escalated_to_agent", state.decision_reasons)
        # 8. Final action
        self.assertEqual(state.final_action, ACTION_ESCALATE)

    def test_scenario_7_risky_security_related_query(self):
        """Scenario 7: Critical account takeover hazard immediately short-circuits to account_security."""
        query = "@AppleSupport my Apple ID was hacked, password changed, and someone made unauthorized charges on my credit card!"
        state = self.controller.process_query(query)

        # 1. Predicted intent
        self.assertEqual(state.predicted_intent, "security_privacy")
        # 2. Classifier confidence
        self.assertIsNotNone(state.intent_confidence)
        # 3. Abstention status
        self.assertFalse(state.is_abstained)
        # 4. Retrieved evidence (pre-retrieval short circuit guarantees 0 retrieval attempts)
        self.assertEqual(state.retrieval_attempts, 0)
        self.assertEqual(len(state.retrieved_evidence), 0)
        # 5. Generated response (short-circuit guarantees 0 generation attempts)
        self.assertEqual(state.generation_attempts, 0)
        self.assertIsNone(state.generated_reply)
        # 6. Grounding checks
        self.assertFalse(state.verification_passed)
        # 7. Escalation reason
        self.assertIn("critical_risk_protocol_triggered", state.decision_reasons)
        self.assertIn("critical_security_compromise_detected", state.decision_reasons)
        self.assertIn("early_critical_hazard_short_circuit", state.decision_reasons)
        # 8. Final action
        self.assertEqual(state.final_action, ACTION_ESCALATE)
        self.assertEqual(state.target_queue, "account_security")

    def test_scenario_8_normal_query_high_confidence_good_grounding(self):
        """Scenario 8: Normal battery support query with verified URL achieves clean AUTO_HANDLE."""
        query = "iPhone battery draining fast what can I do"
        state = self.controller.process_query(query)

        # 1. Predicted intent
        self.assertIn(state.predicted_intent, ["battery_power", "how_to_information"])
        # 2. Classifier confidence
        self.assertGreaterEqual(state.intent_confidence, 0.50)
        # 3. Abstention status
        self.assertFalse(state.is_abstained)
        # 4. Retrieved evidence
        self.assertGreaterEqual(state.retrieval_attempts, 1)
        self.assertEqual(state.evidence_sufficiency, SUFFICIENCY_SUFFICIENT)
        self.assertTrue(len(state.retrieved_evidence[0].extracted_urls) > 0)
        # 5. Generated response
        self.assertIsNotNone(state.generated_reply)
        self.assertEqual(state.generated_reply.action, "AUTO_REPLY")
        self.assertTrue(len(state.generated_reply.used_urls) > 0)
        # 6. Grounding checks
        self.assertTrue(state.verification_passed)
        self.assertFalse(state.verification_details.get("urls_modified", False))
        # 7. Escalation reason
        self.assertIn("verified_official_self_service_resolution_available", state.decision_reasons)
        # 8. Final action
        self.assertEqual(state.final_action, ACTION_AUTO_HANDLE)
        self.assertEqual(state.target_queue, "self_service")


if __name__ == "__main__":
    unittest.main()
