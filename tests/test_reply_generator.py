"""Unit tests for Grounded Reply Generation module (Milestone 15)."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from src.agent.grounded_generator import (
    MockReplyGenerator,
    OpenAIReplyGenerator,
    create_reply_generator,
)
from src.agent.grounding_guard import verify_and_filter_reply
from src.agent.reply_schemas import (
    EvidenceItem,
    GenerationRequest,
    GroundedReply,
    extract_urls,
)
from src.retrieval.historical_resolution import HistoricalResolution, RetrievalResult


class TestReplySchemas(unittest.TestCase):
    """Tests for reply generation data structures and URL extraction."""

    def test_extract_urls(self) -> None:
        text = "Visit https://apple.co/battery-fix, or https://apple.co/support. Thanks!"
        urls = extract_urls(text)
        self.assertEqual(urls, ["https://apple.co/battery-fix", "https://apple.co/support"])

    def test_evidence_item_from_retrieval_result(self) -> None:
        res = HistoricalResolution(
            resolution_id="res_100",
            conversation_id="conv_100",
            customer_tweet_id="t1",
            brand_tweet_id="t2",
            customer_text="iPhone Wi-Fi dropping",
            brand_text="Reset network settings: https://apple.co/wifi-help",
            intent="connectivity",
        )
        ret_result = RetrievalResult(resolution=res, score=0.75, rank=1)
        ev = EvidenceItem.from_retrieval_result(ret_result)

        self.assertEqual(ev.evidence_id, "res_100")
        self.assertEqual(ev.similarity_score, 0.75)
        self.assertEqual(ev.extracted_urls, ["https://apple.co/wifi-help"])
        self.assertEqual(ev.intent, "connectivity")

        d = ev.to_dict()
        self.assertEqual(d["evidence_id"], "res_100")


class TestGroundingGuard(unittest.TestCase):
    """Tests for exact evidence ID and URL verification."""

    def setUp(self) -> None:
        self.evidence = [
            EvidenceItem(
                evidence_id="ev_001",
                similarity_score=0.65,
                past_customer_problem="Battery issue",
                past_brand_resolution="Tips: https://apple.co/battery",
                extracted_urls=["https://apple.co/battery"],
                intent="battery_power",
            )
        ]
        self.request = GenerationRequest(
            customer_query="My battery drains fast",
            intent="battery_power",
            intent_confidence=0.85,
            retrieved_evidence=self.evidence,
        )

    def test_valid_evidence_and_urls_pass(self) -> None:
        raw_reply = GroundedReply(
            reply_text="Check out tips here: https://apple.co/battery",
            grounded=True,
            used_evidence_ids=["ev_001"],
            used_urls=["https://apple.co/battery"],
            action="AUTO_REPLY",
            rationale="Legitimate grounded response.",
            provider="test",
            latency_ms=10.0,
        )
        verified = verify_and_filter_reply(raw_reply, self.request)
        self.assertTrue(verified.grounded)
        self.assertEqual(verified.used_evidence_ids, ["ev_001"])
        self.assertEqual(verified.used_urls, ["https://apple.co/battery"])
        self.assertEqual(verified.action, "AUTO_REPLY")

    def test_ungrounded_evidence_id_filtered(self) -> None:
        raw_reply = GroundedReply(
            reply_text="We can help you with this.",
            grounded=True,
            used_evidence_ids=["ev_001", "ev_fake_999"],
            used_urls=[],
            action="AUTO_REPLY",
            rationale="Test fake evidence ID.",
            provider="test",
            latency_ms=10.0,
        )
        verified = verify_and_filter_reply(raw_reply, self.request)
        self.assertEqual(verified.used_evidence_ids, ["ev_001"])
        self.assertNotIn("ev_fake_999", verified.used_evidence_ids)

    def test_ungrounded_url_stripped_and_flagged(self) -> None:
        raw_reply = GroundedReply(
            reply_text="Visit our unofficial guide at https://fakeappleblog.com/fix for help!",
            grounded=True,
            used_evidence_ids=[],
            used_urls=["https://fakeappleblog.com/fix"],
            action="AUTO_REPLY",
            rationale="Test fake URL.",
            provider="test",
            latency_ms=10.0,
        )
        verified = verify_and_filter_reply(raw_reply, self.request)
        # Fake URL must be stripped from text
        self.assertNotIn("https://fakeappleblog.com/fix", verified.reply_text)
        self.assertEqual(verified.used_urls, [])
        # Without valid evidence, should be demoted to clarification
        self.assertEqual(verified.action, "ASK_CLARIFICATION")
        self.assertFalse(verified.grounded)


class TestMockReplyGenerator(unittest.TestCase):
    """Tests for deterministic MockReplyGenerator."""

    def setUp(self) -> None:
        self.generator = MockReplyGenerator()

    def test_strong_evidence_with_url(self) -> None:
        evidence = [
            EvidenceItem(
                evidence_id="res_01",
                similarity_score=0.55,
                past_customer_problem="Battery issue",
                past_brand_resolution="See here: https://apple.co/battery",
                extracted_urls=["https://apple.co/battery"],
                intent="battery_power",
            )
        ]
        req = GenerationRequest(
            customer_query="iPhone battery dying",
            intent="battery_power",
            intent_confidence=0.88,
            retrieved_evidence=evidence,
            retrieval_status="strong",
        )
        reply = self.generator.generate(req)

        self.assertTrue(reply.grounded)
        self.assertEqual(reply.action, "AUTO_REPLY")
        self.assertIn("https://apple.co/battery", reply.reply_text)
        self.assertEqual(reply.used_evidence_ids, ["res_01"])
        self.assertEqual(reply.used_urls, ["https://apple.co/battery"])

    def test_strong_evidence_without_url(self) -> None:
        evidence = [
            EvidenceItem(
                evidence_id="res_02",
                similarity_score=0.45,
                past_customer_problem="Screen frozen",
                past_brand_resolution="@12345 Force restart your device by holding power and volume down.",
                extracted_urls=[],
                intent="device_hardware",
            )
        ]
        req = GenerationRequest(
            customer_query="Screen is stuck",
            intent="device_hardware",
            intent_confidence=0.82,
            retrieved_evidence=evidence,
            retrieval_status="strong",
        )
        reply = self.generator.generate(req)

        self.assertTrue(reply.grounded)
        self.assertEqual(reply.action, "AUTO_REPLY")
        self.assertNotIn("@12345", reply.reply_text)
        self.assertEqual(reply.used_evidence_ids, ["res_02"])
        self.assertEqual(reply.used_urls, [])

    def test_heuristic_weak_retrieval_triggers_clarification(self) -> None:
        evidence = [
            EvidenceItem(
                evidence_id="res_03",
                similarity_score=0.15,  # below heuristic 0.30 threshold
                past_customer_problem="Something else",
                past_brand_resolution="Help",
                extracted_urls=[],
            )
        ]
        req = GenerationRequest(
            customer_query="Vague problem",
            retrieved_evidence=evidence,
            retrieval_status="heuristic_weak",
        )
        reply = self.generator.generate(req)

        self.assertFalse(reply.grounded)
        self.assertEqual(reply.action, "ASK_CLARIFICATION")
        self.assertIn("device model", reply.reply_text.lower())
        self.assertEqual(reply.used_evidence_ids, [])
        self.assertEqual(reply.used_urls, [])

    def test_empty_evidence_triggers_clarification(self) -> None:
        req = GenerationRequest(
            customer_query="Unknown problem",
            retrieved_evidence=[],
            retrieval_status="empty",
        )
        reply = self.generator.generate(req)
        self.assertFalse(reply.grounded)
        self.assertEqual(reply.action, "ASK_CLARIFICATION")


class TestOpenAIReplyGeneratorMocked(unittest.TestCase):
    """Tests for OpenAIReplyGenerator with mocked network responses."""

    def test_successful_api_generation(self) -> None:
        evidence = [
            EvidenceItem(
                evidence_id="ev_live_01",
                similarity_score=0.60,
                past_customer_problem="AirPlay not working",
                past_brand_resolution="Steps here: https://apple.co/airplay",
                extracted_urls=["https://apple.co/airplay"],
                intent="connectivity",
            )
        ]
        req = GenerationRequest(
            customer_query="Can't AirPlay from my phone",
            intent="connectivity",
            intent_confidence=0.85,
            retrieved_evidence=evidence,
            retrieval_status="strong",
        )

        mock_response_json = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "reply_text": "We can help you resolve AirPlay issues. Follow these steps: https://apple.co/airplay .",
                            "used_evidence_ids": ["ev_live_01"],
                            "used_urls": ["https://apple.co/airplay"],
                            "action": "AUTO_REPLY",
                            "rationale": "Grounded in official Apple AirPlay troubleshooting guide.",
                        })
                    }
                }
            ]
        }

        gen = OpenAIReplyGenerator(api_key="sk-test-key-mock")

        with patch("httpx.Client.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response_json
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            reply = gen.generate(req)

            self.assertTrue(reply.grounded)
            self.assertEqual(reply.action, "AUTO_REPLY")
            self.assertEqual(reply.used_evidence_ids, ["ev_live_01"])
            self.assertEqual(reply.used_urls, ["https://apple.co/airplay"])
            self.assertIn("https://apple.co/airplay", reply.reply_text)

    def test_api_network_failure_safely_falls_back(self) -> None:
        gen = OpenAIReplyGenerator(api_key="sk-test-key-mock")
        req = GenerationRequest(
            customer_query="My phone is broken",
            retrieved_evidence=[
                EvidenceItem(
                    evidence_id="ev_01",
                    similarity_score=0.5,
                    past_customer_problem="Broken",
                    past_brand_resolution="Fix",
                )
            ],
            retrieval_status="strong",
        )

        with patch("httpx.Client.post", side_effect=Exception("Connection timed out")):
            reply = gen.generate(req)
            self.assertFalse(reply.grounded)
            self.assertEqual(reply.action, "ASK_CLARIFICATION")
            self.assertIn("Connection timed out", reply.rationale)


class TestFactoryAndIsolation(unittest.TestCase):
    """Tests for generator factory and Golden Set isolation."""

    def test_factory_defaults_to_mock_without_api_key(self) -> None:
        gen = create_reply_generator(provider=None, api_key=None)
        self.assertIsInstance(gen, MockReplyGenerator)

    def test_golden_set_isolation(self) -> None:
        """Verify that reply generation code has 0 imports or references to golden data files."""
        agent_dir = Path(__file__).resolve().parent.parent / "src" / "agent"
        for py_file in agent_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            self.assertNotIn("golden_annotation.csv", content)
            self.assertNotIn("golden_candidates.csv", content)


if __name__ == "__main__":
    unittest.main()
