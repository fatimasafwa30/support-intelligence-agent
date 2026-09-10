"""Unit tests for Risk Detection and Confidence-Aware Escalation (Milestone 16)."""

from __future__ import annotations

from pathlib import Path
import unittest

from src.agent.escalation_policy import EscalationDecision, EscalationEngine
from src.agent.reply_schemas import EvidenceItem, GroundedReply
from src.agent.risk_detector import (
    CRITICAL_SAFETY,
    CRITICAL_SECURITY,
    FINANCIAL_DISPUTE,
    LEGAL_REGULATORY,
    PHYSICAL_DAMAGE,
    RiskAssessment,
    RiskDetector,
)


class TestRiskDetector(unittest.TestCase):
    """Tests for deterministic regex-based risk detection."""

    def setUp(self) -> None:
        self.detector = RiskDetector()

    def test_critical_safety_signals(self) -> None:
        queries = [
            "My iPhone battery is swelling and the screen popped out!",
            "Phone started smoking while plugged in to charger",
            "The charger caught on fire and burned my hand",
            "Battery exploded in my pocket",
            "Phone is sparking from the charging port",
        ]
        for q in queries:
            assessment = self.detector.assess_risk(q)
            self.assertEqual(assessment.risk_level, "critical", f"Failed for query: {q}")
            self.assertIn(CRITICAL_SAFETY, assessment.risk_categories, f"Missing CRITICAL_SAFETY for: {q}")

    def test_critical_security_signals(self) -> None:
        queries = [
            "Help my Apple ID was hacked and password changed",
            "There are unauthorized charges on my credit card from iTunes",
            "Someone stole my phone and is blackmailing me",
            "Account takeover on my iCloud",
        ]
        for q in queries:
            assessment = self.detector.assess_risk(q)
            self.assertEqual(assessment.risk_level, "critical", f"Failed for query: {q}")
            self.assertIn(CRITICAL_SECURITY, assessment.risk_categories, f"Missing CRITICAL_SECURITY for: {q}")

    def test_physical_damage_signals(self) -> None:
        queries = [
            "I dropped my phone and now have a cracked screen",
            "Dropped in the toilet, severe water damage",
            "Home button fell off my iPad",
            "Charging port is damaged and corroded",
        ]
        for q in queries:
            assessment = self.detector.assess_risk(q)
            self.assertEqual(assessment.risk_level, "high", f"Failed for query: {q}")
            self.assertIn(PHYSICAL_DAMAGE, assessment.risk_categories, f"Missing PHYSICAL_DAMAGE for: {q}")

    def test_financial_and_legal_disputes(self) -> None:
        q_fin = "I was charged twice for my Apple Music subscription"
        ass_fin = self.detector.assess_risk(q_fin)
        self.assertEqual(ass_fin.risk_level, "high")
        self.assertIn(FINANCIAL_DISPUTE, ass_fin.risk_categories)

        q_legal = "I am contacting my lawyer and taking legal action"
        ass_legal = self.detector.assess_risk(q_legal)
        self.assertEqual(ass_legal.risk_level, "high")
        self.assertIn(LEGAL_REGULATORY, ass_legal.risk_categories)

    def test_low_risk_how_to(self) -> None:
        q = "How do I turn off Do Not Disturb While Driving in settings?"
        ass = self.detector.assess_risk(q)
        self.assertEqual(ass.risk_level, "low")
        self.assertEqual(ass.risk_categories, [])


class TestEscalationEngine(unittest.TestCase):
    """Tests for confidence/risk-aware escalation policy."""

    def setUp(self) -> None:
        self.engine = EscalationEngine(
            intent_confidence_threshold=0.50,
            retrieval_similarity_threshold=0.35,
        )
        self.strong_evidence = [
            EvidenceItem(
                evidence_id="res_test_01",
                similarity_score=0.65,
                past_customer_problem="How to turn off driving mode",
                past_brand_resolution="Follow these steps: https://apple.co/dnd",
                extracted_urls=["https://apple.co/dnd"],
                intent="how_to_information",
            )
        ]
        self.grounded_reply = GroundedReply(
            reply_text="Go to settings: https://apple.co/dnd",
            grounded=True,
            used_evidence_ids=["res_test_01"],
            used_urls=["https://apple.co/dnd"],
            action="AUTO_REPLY",
            rationale="Grounded in official support URL.",
            provider="mock",
            latency_ms=5.0,
        )

    def test_zero_tolerance_critical_safety_escalation(self) -> None:
        """Physical safety hazard must ALWAYS escalate even if high similarity or confident intent."""
        decision = self.engine.decide(
            query="Battery is swelling and smoking!",
            predicted_intent="battery_power",
            intent_confidence=0.99,
            top_evidence=self.strong_evidence,
            grounded_reply=self.grounded_reply,
        )
        self.assertEqual(decision.action, "ESCALATE")
        self.assertTrue(decision.requires_human)
        self.assertEqual(decision.routing_target, "safety_team")
        self.assertEqual(decision.risk_level, "critical")
        self.assertIn("physical_safety_hazard_detected", decision.reasons)

    def test_physical_damage_requires_human_repair(self) -> None:
        decision = self.engine.decide(
            query="Dropped phone, cracked screen shattered",
            predicted_intent="device_hardware",
            intent_confidence=0.95,
            top_evidence=self.strong_evidence,
        )
        self.assertEqual(decision.action, "ESCALATE")
        self.assertTrue(decision.requires_human)
        self.assertEqual(decision.routing_target, "hardware_repair")

    def test_grounded_self_service_auto_handle(self) -> None:
        """High confidence + strong retrieval + official URL + no hazard -> AUTO_HANDLE."""
        decision = self.engine.decide(
            query="How do I turn off Do Not Disturb While Driving?",
            predicted_intent="how_to_information",
            intent_confidence=0.85,
            top_evidence=self.strong_evidence,
            grounded_reply=self.grounded_reply,
        )
        self.assertEqual(decision.action, "AUTO_HANDLE")
        self.assertFalse(decision.requires_human)
        self.assertEqual(decision.routing_target, "self_service")
        self.assertIn("verified_official_self_service_resolution_available", decision.reasons)

    def test_low_intent_confidence_gating(self) -> None:
        """Uncertain classifier (< 0.50) with vague text -> ASK_CLARIFICATION."""
        decision = self.engine.decide(
            query="Help with my phone",
            predicted_intent="device_hardware",
            intent_confidence=0.22,  # below 0.50
            top_evidence=[],
        )
        self.assertEqual(decision.action, "ASK_CLARIFICATION")
        self.assertFalse(decision.requires_human)

    def test_weak_retrieval_similarity_gating(self) -> None:
        """Weak retrieval (< 0.35) -> cannot auto-handle."""
        weak_evidence = [
            EvidenceItem(
                evidence_id="res_weak",
                similarity_score=0.18,  # below 0.35
                past_customer_problem="Something else",
                past_brand_resolution="Guidance: https://apple.co/guide",
                extracted_urls=["https://apple.co/guide"],
            )
        ]
        decision = self.engine.decide(
            query="My iPhone 6s is having unusual speaker crackle during video playback",
            predicted_intent="device_hardware",
            intent_confidence=0.75,
            top_evidence=weak_evidence,
        )
        self.assertEqual(decision.action, "ESCALATE")
        self.assertTrue(decision.requires_human)

    def test_custom_boundary_thresholds(self) -> None:
        """Passing custom thresholds alters gating behavior."""
        strict_engine = EscalationEngine(
            intent_confidence_threshold=0.95,
            retrieval_similarity_threshold=0.90,
        )
        # Even with 0.85 confidence and 0.65 similarity, strict engine blocks auto-handle
        decision = strict_engine.decide(
            query="How do I turn off Do Not Disturb While Driving?",
            predicted_intent="how_to_information",
            intent_confidence=0.85,
            top_evidence=self.strong_evidence,
            grounded_reply=self.grounded_reply,
        )
        self.assertNotEqual(decision.action, "AUTO_HANDLE")

    def test_golden_set_isolation(self) -> None:
        """Ensure policy engine has zero imports or dependencies on Golden Set files."""
        agent_dir = Path(__file__).resolve().parent.parent / "src" / "agent"
        for py_file in ("risk_detector.py", "escalation_policy.py"):
            content = (agent_dir / py_file).read_text(encoding="utf-8")
            self.assertNotIn("golden_annotation.csv", content)
            self.assertNotIn("golden_candidates.csv", content)


if __name__ == "__main__":
    unittest.main()
