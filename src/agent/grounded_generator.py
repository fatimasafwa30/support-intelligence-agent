"""Grounded reply generation engine for customer support inquiries.

Provides:
- BaseReplyGenerator abstract class
- MockReplyGenerator (deterministic offline default)
- OpenAIReplyGenerator (using standard chat completions via httpx)
- Factory function create_reply_generator
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.grounding_guard import verify_and_filter_reply
from src.agent.reply_schemas import EvidenceItem, GenerationRequest, GroundedReply

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert customer support specialist for Apple Support.
Your goal is to draft a helpful, polite, and concise reply (1-3 sentences) to an inbound customer tweet.

STRICT GROUNDING RULES:
1. Grounding: You must ONLY provide troubleshooting advice, steps, or facts that are explicitly supported by the provided Historical Evidence.
2. No Hallucinations: Do NOT invent troubleshooting steps, hardware diagnoses, or policies not present in the evidence.
3. URLs: If including a link, you must ONLY use exact URLs listed under "Extracted URLs" in the evidence. Never invent, alter, or construct new URLs.
4. Tone: Empathetic, professional, and clear.
5. Evidence Attribution: List the Evidence IDs you used to formulate your answer.
6. Weak / Incomplete Evidence: If the evidence does not contain a clear resolution for the customer's specific problem, do not guess. Instead, ask for the necessary details (device model, OS version) and invite them to continue via DM.

Respond in valid JSON with these keys:
{
  "reply_text": "string",
  "used_evidence_ids": ["string"],
  "used_urls": ["string"],
  "action": "AUTO_REPLY" | "ASK_CLARIFICATION",
  "rationale": "string"
}
"""

CLARIFICATION_TEMPLATE = (
    "We'd like to look into this with you. Could you let us know your exact device model "
    "and iOS version? You can also meet us in DM with additional details so we can help."
)


class BaseReplyGenerator(ABC):
    """Abstract base class for grounded reply generators."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GroundedReply:
        """Generate a grounded customer support reply from the given request."""
        raise NotImplementedError


class MockReplyGenerator(BaseReplyGenerator):
    """Deterministic offline reply generator requiring no external API or internet access.

    Serves as the default execution mode to guarantee that all automated tests and
    offline pipelines run with zero credentials and 100% reproducibility.
    """

    def __init__(self) -> None:
        self.provider_name = "mock"

    def generate(self, request: GenerationRequest) -> GroundedReply:
        """Generate a deterministic grounded reply using rule-based heuristics."""
        start_time = time.perf_counter()

        # Heuristic weak retrieval or empty evidence: strictly abstain with clarification
        if request.retrieval_status == "heuristic_weak" or not request.retrieved_evidence:
            latency_ms = (time.perf_counter() - start_time) * 1000
            raw_reply = GroundedReply(
                reply_text=CLARIFICATION_TEMPLATE,
                grounded=False,
                used_evidence_ids=[],
                used_urls=[],
                action="ASK_CLARIFICATION",
                rationale="Heuristic weak retrieval or empty evidence; requested customer clarification.",
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
            return verify_and_filter_reply(raw_reply, request)

        # Strong evidence available: inspect top evidence item
        top_ev = request.retrieved_evidence[0]

        # If top evidence contains an official support URL, ground the reply with that link
        if top_ev.extracted_urls:
            url = top_ev.extracted_urls[0]
            reply_text = (
                f"We'd be glad to help with this. You can find the steps to resolve this here: {url} . "
                f"Let us know if you need any further assistance!"
            )
            used_urls = [url]
            used_eids = [top_ev.evidence_id]
            action = "AUTO_REPLY"
            grounded = True
            rationale = "Grounded in top historical resolution containing verified support URL."
        else:
            # Strip initial '@username' handle from brand resolution
            clean_brand = re.sub(r"^@\w+\s*", "", top_ev.past_brand_resolution).strip()
            reply_text = f"We're here to help with your device. {clean_brand}"
            used_urls = []
            used_eids = [top_ev.evidence_id]
            action = "AUTO_REPLY"
            grounded = True
            rationale = "Grounded in top historical resolution steps."

        latency_ms = (time.perf_counter() - start_time) * 1000
        raw_reply = GroundedReply(
            reply_text=reply_text,
            grounded=grounded,
            used_evidence_ids=used_eids,
            used_urls=used_urls,
            action=action,
            rationale=rationale,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )
        return verify_and_filter_reply(raw_reply, request)


class OpenAIReplyGenerator(BaseReplyGenerator):
    """Grounded reply generator using OpenAI Chat Completions API via httpx."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 250,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider_name = f"openai:{model}"

    def _build_user_prompt(self, request: GenerationRequest) -> str:
        """Format the user prompt with query context and structured evidence."""
        lines = [
            f'Customer Inquiry:\n"{request.customer_query}"',
            f"\nPredicted Problem Category: {request.intent or 'unknown'}",
        ]
        if request.intent_confidence is not None:
            lines.append(f"Intent Confidence: {request.intent_confidence:.2f}")

        lines.append(f"Retrieval Status: {request.retrieval_status}")
        lines.append("\nHistorical Evidence from Verified Past Resolutions:")

        if not request.retrieved_evidence:
            lines.append("No relevant historical resolutions found.")
        else:
            for idx, ev in enumerate(request.retrieved_evidence, start=1):
                lines.append(
                    f"[Evidence {idx}] (ID: {ev.evidence_id}, Similarity: {ev.similarity_score:.3f}, Intent: {ev.intent or 'unassigned'})"
                )
                lines.append(f'  Past Customer: "{ev.past_customer_problem}"')
                lines.append(f'  Past Resolution: "{ev.past_brand_resolution}"')
                url_str = ", ".join(ev.extracted_urls) if ev.extracted_urls else "None"
                lines.append(f"  Extracted URLs: {url_str}")

        lines.append("\nRespond with your JSON output:")
        return "\n".join(lines)

    def generate(self, request: GenerationRequest) -> GroundedReply:
        """Send request to OpenAI Chat Completions and verify response."""
        start_time = time.perf_counter()

        # If retrieval is heuristic weak, abstain immediately without wasting API tokens
        if request.retrieval_status == "heuristic_weak" or not request.retrieved_evidence:
            latency_ms = (time.perf_counter() - start_time) * 1000
            raw_reply = GroundedReply(
                reply_text=CLARIFICATION_TEMPLATE,
                grounded=False,
                used_evidence_ids=[],
                used_urls=[],
                action="ASK_CLARIFICATION",
                rationale="Heuristic weak retrieval or empty evidence; requested customer clarification.",
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
            return verify_and_filter_reply(raw_reply, request)

        user_prompt = self._build_user_prompt(request)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            choice_content = data["choices"][0]["message"]["content"]
            parsed = json.loads(choice_content)

            reply_text = str(parsed.get("reply_text") or "").strip()
            used_eids = [str(x) for x in parsed.get("used_evidence_ids", [])]
            used_urls = [str(x) for x in parsed.get("used_urls", [])]
            action = str(parsed.get("action") or "AUTO_REPLY")
            rationale = str(parsed.get("rationale") or "Generated via OpenAI model.")

            latency_ms = (time.perf_counter() - start_time) * 1000
            raw_reply = GroundedReply(
                reply_text=reply_text,
                grounded=True,
                used_evidence_ids=used_eids,
                used_urls=used_urls,
                action=action,
                rationale=rationale,
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
            return verify_and_filter_reply(raw_reply, request)

        except Exception as err:
            logger.error("Error during OpenAI reply generation: %s; falling back to clarification", err)
            latency_ms = (time.perf_counter() - start_time) * 1000
            fallback_reply = GroundedReply(
                reply_text=CLARIFICATION_TEMPLATE,
                grounded=False,
                used_evidence_ids=[],
                used_urls=[],
                action="ASK_CLARIFICATION",
                rationale=f"API error fallback: {err}",
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
            return verify_and_filter_reply(fallback_reply, request)


def create_reply_generator(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> BaseReplyGenerator:
    """Factory function creating a reply generator.

    Defaults to MockReplyGenerator if no provider/API key is provided.
    """
    chosen_provider = (provider or os.environ.get("LLM_PROVIDER") or "").lower()
    chosen_key = api_key or os.environ.get("OPENAI_API_KEY")
    chosen_model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"

    if chosen_provider == "openai" or (chosen_provider != "mock" and chosen_key):
        if not chosen_key:
            logger.warning("OpenAI provider requested but no API key found. Falling back to MockReplyGenerator.")
            return MockReplyGenerator()
        return OpenAIReplyGenerator(api_key=chosen_key, model=chosen_model)

    return MockReplyGenerator()
