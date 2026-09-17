"""Versioned, allowlisted console contract and localhost-only HTTP serving."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import TypedDict
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]


class AnalysisRequest(TypedDict):
    message: str


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string"}
TEXTS = {"type": "array", "items": TEXT}
SCORE = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
COUNT = {"type": "integer", "minimum": 0}
BOOL = {"type": "boolean"}
RESPONSE_SCHEMA = obj({
    "version": {"const": 1}, "customer_message": TEXT,
    "predicted_intent": {"type": ["string", "null"]}, "intent_confidence": SCORE,
    "final_action": {"enum": ["AUTO_HANDLE", "ASK_CLARIFICATION", "ESCALATE"]},
    "target_queue": TEXT, "decision_reasons": TEXTS,
    "risk": {"anyOf": [{"type": "null"}, obj({"level": TEXT, "categories": TEXTS})]},
    "retrieval": obj({"attempts": COUNT, "top_similarity": SCORE,
                      "sufficiency": {"enum": ["sufficient", "borderline", "insufficient", "empty", "unknown"]}}),
    "evidence": {"type": "array", "items": obj({"evidence_id": TEXT, "similarity": SCORE,
        "customer_problem": TEXT, "brand_resolution": TEXT, "urls": TEXTS})},
    "reply": {"anyOf": [{"type": "null"}, obj({"text": TEXT, "grounded": BOOL,
        "used_evidence_ids": TEXTS, "used_urls": TEXTS, "provider": TEXT})]},
    "verification_passed": {"type": ["boolean", "null"]}, "generation_attempts": COUNT,
    "latency_ms": {"type": "number", "minimum": 0},
})
RESPONSE_VALIDATOR = Draft202012Validator(RESPONSE_SCHEMA)
REQUEST_VALIDATOR = Draft202012Validator(obj({"message": {"type": "string", "minLength": 1, "maxLength": 4000}}))


def validate_request(data) -> AnalysisRequest:
    if not REQUEST_VALIDATOR.is_valid(data) or not data["message"].strip():
        raise ValueError("Enter a customer message between 1 and 4,000 characters.")
    return {"message": data["message"].strip()}


def serialize_state(state):
    # Runtime fallback is an execution error, not a policy-driven escalation.
    if any(reason.startswith("controller_runtime_error:") for reason in state.decision_reasons):
        raise RuntimeError("Controller execution failed")
    reply = state.generated_reply
    risk = state.risk_assessment
    result = {
        "version": 1, "customer_message": state.customer_message,
        "predicted_intent": state.predicted_intent, "intent_confidence": state.intent_confidence,
        "final_action": state.final_action, "target_queue": state.target_queue,
        "decision_reasons": list(state.decision_reasons),
        "risk": {"level": risk.risk_level, "categories": list(risk.risk_categories)} if risk else None,
        "retrieval": {"attempts": state.retrieval_attempts,
                      "top_similarity": state.top_similarity_score if state.retrieval_attempts else None,
                      "sufficiency": state.evidence_sufficiency},
        "evidence": [{"evidence_id": e.evidence_id, "similarity": e.similarity_score,
                      "customer_problem": e.past_customer_problem, "brand_resolution": e.past_brand_resolution,
                      "urls": list(e.extracted_urls)} for e in state.retrieved_evidence],
        "reply": {"text": reply.reply_text, "grounded": reply.grounded,
                  "used_evidence_ids": list(reply.used_evidence_ids), "used_urls": list(reply.used_urls),
                  "provider": reply.provider} if reply else None,
        "verification_passed": state.verification_passed if reply else None,
        "generation_attempts": state.generation_attempts, "latency_ms": state.total_latency_ms,
    }
    RESPONSE_VALIDATOR.validate(result)
    # Reject nonfinite floats as well as out-of-contract states.
    json.dumps(result, allow_nan=False)
    return result


class ConsoleService:
    def __init__(self):
        self.controller = None
        self.lock = threading.Lock()

    def initialize(self):
        from src.agent.controller import AgentController
        from src.agent.grounded_generator import MockReplyGenerator
        controller = AgentController(generator=MockReplyGenerator())
        if controller._ensure_classifier() is None or controller._ensure_retriever() is None:
            raise RuntimeError("Frozen artifacts unavailable")
        self.controller = controller

    def analyze(self, message):
        if self.controller is None:
            raise RuntimeError("Frozen artifacts unavailable")
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Analysis already in progress")
        try:
            return serialize_state(self.controller.process_query(customer_message=message))
        finally:
            self.lock.release()


def make_server(port=8765, service=None, static_root=None, host="127.0.0.1"):
    service = service or ConsoleService()
    if static_root is None:
        static_dist = ROOT / "frontend/dist"
        static_root = static_dist if (static_dist / "index.html").exists() else ROOT / "frontend"
    static_root = Path(static_root)
    assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
              "/view.js": ("view.js", "text/javascript"), "/styles.css": ("styles.css", "text/css")}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Never log customer payloads, URLs with secrets, or exceptions.

        def respond(self, status, payload, mime="application/json"):
            content = json.dumps(payload, allow_nan=False).encode() if mime == "application/json" else payload
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(content)

        def local_request(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            host_header = self.headers.get("Host", "")
            if self.server.server_address[0] == "127.0.0.1":
                if host_header not in {expected, f"localhost:{self.server.server_port}"}:
                    self.respond(403, {"error": "Local access only."})
                    return False
            elif not host_header:
                self.respond(400, {"error": "Host header required."})
                return False
            origin = self.headers.get("Origin")
            allowed_origins = {None, f"http://{host_header}", f"https://{host_header}"}
            host_name = host_header.split(":")[0]
            allowed_origins.add(f"http://{host_name}")
            allowed_origins.add(f"https://{host_name}")
            if origin not in allowed_origins:
                self.respond(403, {"error": "Same-origin access required."})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            route = urlsplit(self.path).path
            if route == "/api/health":
                self.respond(200, {"ready": service.controller is not None, "mode": "offline"})
            elif route == "/api/schema":
                self.respond(200, RESPONSE_SCHEMA)
            elif route in assets:
                filename, mime = assets[route]
                try:
                    self.respond(200, (static_root / filename).read_bytes(), mime)
                except OSError:
                    self.respond(503, {"error": "Frontend build unavailable. Run npm run build in frontend."})
            else:
                self.respond(404, {"error": "Not found."})

        def do_POST(self):
            if not self.local_request():
                return
            if self.path != "/api/analyze":
                self.respond(404, {"error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20000 or self.headers.get_content_type() != "application/json":
                    raise ValueError
                self.connection.settimeout(10)
                data = validate_request(json.loads(self.rfile.read(length)))
            except (ValueError, UnicodeError, OSError):
                self.respond(400, {"error": "Enter a customer message between 1 and 4,000 characters as JSON."})
                return
            try:
                result = service.analyze(data["message"])
                self.respond(200, result)
            except BlockingIOError:
                self.respond(409, {"error": "An analysis is in progress. Please try again shortly."})
            except Exception:
                self.respond(503, {"error": "Analysis unavailable. Check the local model artifacts and restart the server."})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
