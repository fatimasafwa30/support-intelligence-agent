"""Agent state and execution trace definitions for orchestrating customer support inquiries.

Maintains complete intermediate state across classification, risk detection,
retrieval, verification, and escalation decisions for transparency and auditability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any

from src.agent.reply_schemas import EvidenceItem, GroundedReply
from src.agent.risk_detector import RiskAssessment

# Action Constants
ACTION_AUTO_HANDLE = "AUTO_HANDLE"
ACTION_ASK_CLARIFICATION = "ASK_CLARIFICATION"
ACTION_ESCALATE = "ESCALATE"
ACTION_PENDING = "PENDING"

VALID_ACTIONS = {
    ACTION_AUTO_HANDLE,
    ACTION_ASK_CLARIFICATION,
    ACTION_ESCALATE,
    ACTION_PENDING,
}

# Evidence Sufficiency Constants
SUFFICIENCY_SUFFICIENT = "sufficient"
SUFFICIENCY_BORDERLINE = "borderline"
SUFFICIENCY_INSUFFICIENT = "insufficient"
SUFFICIENCY_EMPTY = "empty"
SUFFICIENCY_UNKNOWN = "unknown"


@dataclass
class AgentState:
    """State container tracking the complete decision and execution lifecycle of an inquiry."""

    # 1. Customer Input
    customer_message: str
    conversation_id: str | None = None

    # 2. Intent Understanding
    predicted_intent: str | None = None
    intent_confidence: float | None = None
    is_abstained: bool = False

    # 3. Evidence Retrieval
    retrieval_attempts: int = 0
    retrieved_evidence: list[EvidenceItem] = field(default_factory=list)
    top_similarity_score: float = 0.0
    evidence_sufficiency: str = SUFFICIENCY_UNKNOWN

    # 4. Risk Assessment
    risk_assessment: RiskAssessment | None = None

    # 5. Generation & Verification
    generation_attempts: int = 0
    generated_reply: GroundedReply | None = None
    verification_passed: bool = False
    verification_details: dict[str, Any] = field(default_factory=dict)

    # 6. Final Decision (delegated to EscalationEngine)
    final_action: str = ACTION_PENDING
    target_queue: str = "general_support"
    decision_reasons: list[str] = field(default_factory=list)

    # 7. Trace & Performance
    trace: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
    _start_time: float = field(default_factory=time.perf_counter, repr=False)

    def record_step(
        self,
        step_name: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append an auditable state transition step to the execution trace."""
        self.trace.append({
            "step": step_name,
            "action": action,
            "details": dict(details or {}),
            "timestamp_ms": round((time.perf_counter() - self._start_time) * 1000, 2),
        })

    def complete(self) -> None:
        """Finalize state and calculate total execution latency."""
        self.total_latency_ms = round((time.perf_counter() - self._start_time) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a JSON-serializable dictionary."""
        return {
            "customer_message": self.customer_message,
            "conversation_id": self.conversation_id,
            "predicted_intent": self.predicted_intent,
            "intent_confidence": round(self.intent_confidence, 4) if self.intent_confidence is not None else None,
            "is_abstained": self.is_abstained,
            "retrieval_attempts": self.retrieval_attempts,
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
            "top_similarity_score": round(self.top_similarity_score, 4),
            "evidence_sufficiency": self.evidence_sufficiency,
            "risk_assessment": self.risk_assessment.to_dict() if self.risk_assessment else None,
            "generation_attempts": self.generation_attempts,
            "generated_reply": self.generated_reply.to_dict() if self.generated_reply else None,
            "verification_passed": self.verification_passed,
            "verification_details": dict(self.verification_details),
            "final_action": self.final_action,
            "target_queue": self.target_queue,
            "decision_reasons": list(self.decision_reasons),
            "trace": list(self.trace),
            "total_latency_ms": self.total_latency_ms,
        }
