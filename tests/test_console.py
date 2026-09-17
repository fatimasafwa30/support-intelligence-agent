"""Contract and real loopback HTTP tests; all inference is mocked here."""
import json
import threading
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jsonschema import ValidationError
from src.agent.agent_state import AgentState
from src.agent.reply_schemas import EvidenceItem, GroundedReply
from src.agent.risk_detector import RiskAssessment
from src.console.api import ConsoleService, RESPONSE_VALIDATOR, make_server, serialize_state, validate_request


def state():
    return AgentState(customer_message="Synthetic test query", predicted_intent="test_intent", intent_confidence=.75,
        final_action="AUTO_HANDLE", target_queue="self_service", decision_reasons=["test_policy_reason"],
        risk_assessment=RiskAssessment("low", [], [], "PRIVATE_RATIONALE"), retrieval_attempts=1,
        evidence_sufficiency="sufficient", top_similarity_score=.6,
        retrieved_evidence=[EvidenceItem("ev_test", .6, "Test problem", "Test answer", ["https://support.apple.com/test"])],
        generated_reply=GroundedReply("Test reply", True, ["ev_test"], [], "AUTO_REPLY", "PRIVATE_RATIONALE", "mock", 1),
        verification_passed=True, generation_attempts=1, total_latency_ms=2)


class TestConsole(unittest.TestCase):
    def test_request_validation(self):
        for data in ({}, None, [], {"message": " "}, {"message": 1}, {"message": "x"*4001}, {"message": "x", "gold_intent": "secret"}):
            with self.subTest(data=type(data).__name__), self.assertRaises(ValueError):
                validate_request(data)
        self.assertEqual(validate_request({"message": " test "}), {"message": "test"})

    def test_response_contract_all_actions_and_data(self):
        for action in ("AUTO_HANDLE", "ASK_CLARIFICATION", "ESCALATE"):
            value = state(); value.final_action = action
            data = serialize_state(value)
            RESPONSE_VALIDATOR.validate(data)
            self.assertEqual(data["final_action"], action)
            self.assertEqual(data["evidence"][0]["similarity"], .6)
            self.assertEqual(data["reply"]["used_evidence_ids"], ["ev_test"])

    def test_unknown_scores_and_unexecuted_checks_are_null(self):
        value = AgentState("query", final_action="ESCALATE")
        data = serialize_state(value)
        self.assertIsNone(data["intent_confidence"])
        self.assertIsNone(data["retrieval"]["top_similarity"])
        self.assertIsNone(data["verification_passed"])
        self.assertIsNone(data["reply"])
        self.assertEqual(data["evidence"], [])

    def test_private_fields_and_hidden_reasoning_are_excluded(self):
        value = state()
        value.trace = [{"details": {"chain_of_thought": "PRIVATE_TRACE", "GEMINI_API_KEY": "PRIVATE_KEY"}}]
        value.verification_details = {"annotation_notes": "PRIVATE_NOTES"}
        value.gold_intent = "PRIVATE_GOLD"
        value.judge_scores = {"groundedness": 5}
        wire = json.dumps(serialize_state(value))
        for token in ("PRIVATE_", "trace", "rationale", "gold_intent", "judge_scores", "annotation_notes"):
            self.assertNotIn(token, wire)

    def test_invalid_response_and_runtime_fallback_rejected(self):
        for score in (-1, 2, float("nan")):
            value = state(); value.intent_confidence = score
            with self.assertRaises((ValidationError, ValueError)):
                serialize_state(value)
        value = state(); value.final_action = "PENDING"
        with self.assertRaises(ValidationError):
            serialize_state(value)
        value = state(); value.decision_reasons = ["controller_runtime_error: SecretError"]
        with self.assertRaises(RuntimeError):
            serialize_state(value)

    def test_busy_service_does_not_start_another_inference(self):
        service = ConsoleService(); service.controller = Mock()
        with service.lock:
            with self.assertRaises(BlockingIOError):
                service.analyze("query")
        service.controller.process_query.assert_not_called()


class TestConsoleHTTP(unittest.TestCase):
    def setUp(self):
        self.service = Mock(controller=object())
        self.service.analyze.return_value = serialize_state(state())
        self.server = make_server(0, self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def post(self, data, headers=None):
        request = Request(self.url + "/api/analyze", data=json.dumps(data).encode(),
                          headers={"Content-Type": "application/json", **(headers or {})})
        return urlopen(request, timeout=3)

    def test_success_health_and_request_boundary(self):
        with self.post({"message": "query"}) as response:
            RESPONSE_VALIDATOR.validate(json.load(response))
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        self.service.analyze.assert_called_once_with("query")
        with urlopen(self.url + "/api/health") as response:
            self.assertTrue(json.load(response)["ready"])

    def test_error_body_never_exposes_exception(self):
        self.service.analyze.side_effect = RuntimeError("SECRET_KEY traceback")
        with self.assertRaises(HTTPError) as caught:
            self.post({"message": "query"})
        self.assertEqual(caught.exception.code, 503)
        self.assertNotIn("SECRET", caught.exception.read().decode())
        caught.exception.close()

    def test_rejects_cross_origin_and_extra_labels(self):
        for data, headers, code in (({"message": "query"}, {"Origin": "https://evil.example"}, 403),
                                    ({"message": "query", "gold_action": "secret"}, {}, 400)):
            with self.assertRaises(HTTPError) as caught:
                self.post(data, headers)
            self.assertEqual(caught.exception.code, code)
            caught.exception.close()
        self.service.analyze.assert_not_called()

    def test_no_filesystem_or_secret_routes(self):
        for route in ("/.env", "/../.env", "/reports/reply_quality_judge_records.jsonl"):
            with self.assertRaises(HTTPError) as caught:
                urlopen(self.url + route)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()

    def test_get_server_config_environment_precedence(self):
        import os
        from scripts.serve_console import get_server_config
        old_port = os.environ.pop("PORT", None)
        old_host = os.environ.pop("HOST", None)
        try:
            host, port = get_server_config()
            self.assertEqual(host, "127.0.0.1")
            self.assertEqual(port, 8765)

            os.environ["PORT"] = "10000"
            host, port = get_server_config()
            self.assertEqual(host, "0.0.0.0")
            self.assertEqual(port, 10000)

            os.environ["HOST"] = "0.0.0.0"
            host, port = get_server_config()
            self.assertEqual(host, "0.0.0.0")
            self.assertEqual(port, 10000)
        finally:
            if old_port is not None:
                os.environ["PORT"] = old_port
            else:
                os.environ.pop("PORT", None)
            if old_host is not None:
                os.environ["HOST"] = old_host
            else:
                os.environ.pop("HOST", None)

    def test_bound_to_all_interfaces_supports_cloud_deployment(self):
        server = make_server(0, self.service, host="0.0.0.0")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_port
            url = f"http://127.0.0.1:{port}"
            with urlopen(f"{url}/api/health") as response:
                self.assertEqual(response.status, 200)

            req = Request(
                f"{url}/api/analyze",
                data=json.dumps({"message": "query"}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Host": "support-intelligence.onrender.com",
                    "Origin": "https://support-intelligence.onrender.com",
                },
            )
            with urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)

            evil_req = Request(
                f"{url}/api/analyze",
                data=json.dumps({"message": "query"}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Host": "support-intelligence.onrender.com",
                    "Origin": "https://evil.attacker.com",
                },
            )
            with self.assertRaises(HTTPError) as caught:
                urlopen(evil_req, timeout=3)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
