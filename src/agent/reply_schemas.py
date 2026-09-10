"""Data schemas for grounded customer support reply generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from src.retrieval.historical_resolution import RetrievalResult

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    """Extract all HTTP/HTTPS URLs from text, stripping trailing punctuation."""
    matches = URL_PATTERN.findall(text or "")
    clean_urls: list[str] = []
    for m in matches:
        cleaned = m.rstrip(".,;!?:)\"'>")
        if cleaned:
            clean_urls.append(cleaned)
    return clean_urls


@dataclass(frozen=True)
class EvidenceItem:
    """Historical resolution evidence item formatted for reply generation."""

    evidence_id: str
    similarity_score: float
    past_customer_problem: str
    past_brand_resolution: str
    extracted_urls: list[str] = field(default_factory=list)
    intent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence item to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "similarity_score": round(self.similarity_score, 4),
            "past_customer_problem": self.past_customer_problem,
            "past_brand_resolution": self.past_brand_resolution,
            "extracted_urls": list(self.extracted_urls),
            "intent": self.intent,
        }

    @classmethod
    def from_retrieval_result(cls, result: RetrievalResult) -> EvidenceItem:
        """Construct an EvidenceItem directly from a retrieval result."""
        res = result.resolution
        urls = extract_urls(res.brand_text)
        return cls(
            evidence_id=res.resolution_id,
            similarity_score=float(result.score),
            past_customer_problem=res.customer_text,
            past_brand_resolution=res.brand_text,
            extracted_urls=urls,
            intent=res.intent,
        )


@dataclass(frozen=True)
class GenerationRequest:
    """Request payload for grounded reply generation."""

    customer_query: str
    intent: str | None = None
    intent_confidence: float | None = None
    retrieved_evidence: list[EvidenceItem] = field(default_factory=list)
    retrieval_status: str = "strong"  # "strong", "heuristic_weak", "empty"
    conversation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "customer_query": self.customer_query,
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 4) if self.intent_confidence is not None else None,
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
            "retrieval_status": self.retrieval_status,
            "conversation_id": self.conversation_id,
        }


@dataclass(frozen=True)
class GroundedReply:
    """Structured response from grounded reply generation."""

    reply_text: str
    grounded: bool
    used_evidence_ids: list[str]
    used_urls: list[str]
    action: str  # "AUTO_REPLY", "ASK_CLARIFICATION"
    rationale: str
    provider: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert reply to dictionary."""
        return {
            "reply_text": self.reply_text,
            "grounded": self.grounded,
            "used_evidence_ids": list(self.used_evidence_ids),
            "used_urls": list(self.used_urls),
            "action": self.action,
            "rationale": self.rationale,
            "provider": self.provider,
            "latency_ms": round(self.latency_ms, 2),
        }
