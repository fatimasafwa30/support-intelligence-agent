"""Data schemas and validation models for customer support reply-quality evaluation.

Defines the multi-dimensional evaluation rubric (groundedness, correctness, relevance,
helpfulness, tone) and structured evaluation units combining query, intent, evidence,
and reply.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

# Evaluator types
EVALUATOR_HUMAN = "human"
EVALUATOR_LLM_JUDGE = "llm_judge"
VALID_EVALUATOR_TYPES = {EVALUATOR_HUMAN, EVALUATOR_LLM_JUDGE}

# Rubric dimensions
DIMENSION_GROUNDEDNESS = "groundedness"
DIMENSION_CORRECTNESS = "correctness"
DIMENSION_RELEVANCE = "relevance"
DIMENSION_HELPFULNESS = "helpfulness"
DIMENSION_TONE = "tone"

RUBRIC_DIMENSIONS = (
    DIMENSION_GROUNDEDNESS,
    DIMENSION_CORRECTNESS,
    DIMENSION_RELEVANCE,
    DIMENSION_HELPFULNESS,
    DIMENSION_TONE,
)

SCORE_MIN = 1
SCORE_MAX = 5


def _validate_score(name: str, value: Any) -> int:
    """Validate that a score is an integer strictly between 1 and 5 inclusive."""
    # bool is a subclass of int in Python, so explicitly reject bool
    if isinstance(value, bool):
        raise TypeError(f"Dimension '{name}' score must be an integer, got bool: {value}")
    if not isinstance(value, int):
        raise TypeError(f"Dimension '{name}' score must be an integer, got {type(value).__name__}: {value}")
    if value < SCORE_MIN or value > SCORE_MAX:
        raise ValueError(
            f"Dimension '{name}' score must be between {SCORE_MIN} and {SCORE_MAX} inclusive, got {value}"
        )
    return value


def _validate_overall_score(value: Any) -> float | None:
    """Validate optional overall score (float or int between 1.0 and 5.0)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"overall_score must be a number, got bool: {value}")
    if not isinstance(value, (int, float)):
        raise TypeError(f"overall_score must be a float or int, got {type(value).__name__}: {value}")
    score_f = float(value)
    if score_f < float(SCORE_MIN) or score_f > float(SCORE_MAX):
        raise ValueError(
            f"overall_score must be between {SCORE_MIN}.0 and {SCORE_MAX}.0 inclusive, got {score_f}"
        )
    return score_f


@dataclass(frozen=True)
class ReplyQualityRating:
    """Standardized rating for an individual generated customer support reply.

    Evaluates quality across five core dimensions:
    - groundedness: strictly supported by retrieved evidence (1-5)
    - correctness: technically accurate and aligned with Apple procedures (1-5)
    - relevance: directly addresses the customer's problem and context (1-5)
    - helpfulness: actionable and moves customer toward resolution (1-5)
    - tone: empathetic, professional, and matching Apple voice (1-5)
    """

    example_id: str
    evaluator_type: str
    groundedness: int
    correctness: int
    relevance: int
    helpfulness: int
    tone: int
    overall_score: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate all required fields and score constraints."""
        if not self.example_id or not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("example_id must be a non-empty string.")

        if self.evaluator_type not in VALID_EVALUATOR_TYPES:
            raise ValueError(
                f"evaluator_type must be one of {sorted(VALID_EVALUATOR_TYPES)}, got '{self.evaluator_type}'"
            )

        # Validate 5 dimension scores
        object.__setattr__(self, "groundedness", _validate_score("groundedness", self.groundedness))
        object.__setattr__(self, "correctness", _validate_score("correctness", self.correctness))
        object.__setattr__(self, "relevance", _validate_score("relevance", self.relevance))
        object.__setattr__(self, "helpfulness", _validate_score("helpfulness", self.helpfulness))
        object.__setattr__(self, "tone", _validate_score("tone", self.tone))

        # Validate overall score if provided
        if self.overall_score is not None:
            object.__setattr__(
                self, "overall_score", _validate_overall_score(self.overall_score)
            )

        # Validate notes if provided
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError(f"notes must be a string if provided, got {type(self.notes).__name__}")

    def compute_average_score(self) -> float:
        """Calculate the unweighted arithmetic mean of the five rubric dimensions."""
        scores = [
            self.groundedness,
            self.correctness,
            self.relevance,
            self.helpfulness,
            self.tone,
        ]
        return round(sum(scores) / len(scores), 4)

    def to_dict(self) -> dict[str, Any]:
        """Convert rating to JSON-serializable dictionary."""
        return {
            "example_id": self.example_id,
            "evaluator_type": self.evaluator_type,
            "groundedness": self.groundedness,
            "correctness": self.correctness,
            "relevance": self.relevance,
            "helpfulness": self.helpfulness,
            "tone": self.tone,
            "overall_score": round(self.overall_score, 4) if self.overall_score is not None else None,
            "average_score": self.compute_average_score(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplyQualityRating:
        """Construct and validate a ReplyQualityRating instance from a dictionary."""
        required = [
            "example_id",
            "evaluator_type",
            "groundedness",
            "correctness",
            "relevance",
            "helpfulness",
            "tone",
        ]
        missing = [f for f in required if f not in data]
        if missing:
            raise KeyError(f"Missing required rating fields: {missing}")

        return cls(
            example_id=str(data["example_id"]),
            evaluator_type=str(data["evaluator_type"]),
            groundedness=data["groundedness"],
            correctness=data["correctness"],
            relevance=data["relevance"],
            helpfulness=data["helpfulness"],
            tone=data["tone"],
            overall_score=data.get("overall_score"),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class EvaluationUnit:
    """Full evaluation context containing all materials necessary to judge a reply.

    CRITICAL: A reply must NEVER be evaluated in isolation. Evaluation requires
    customer context, predicted intent, and retrieved evidence.
    """

    example_id: str
    customer_query: str
    predicted_intent: str
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    generated_reply: dict[str, Any] = field(default_factory=dict)
    conversation_id: str | None = None
    intent_confidence: float | None = None
    verified_reply: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate non-empty query and example ID."""
        if not self.example_id or not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("example_id must be a non-empty string.")
        if not self.customer_query or not isinstance(self.customer_query, str):
            raise ValueError("customer_query must be a non-empty string.")
        if not self.predicted_intent or not isinstance(self.predicted_intent, str):
            raise ValueError("predicted_intent must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation unit to dictionary."""
        return {
            "example_id": self.example_id,
            "customer_query": self.customer_query,
            "predicted_intent": self.predicted_intent,
            "retrieved_evidence": [dict(e) for e in self.retrieved_evidence],
            "generated_reply": dict(self.generated_reply),
            "conversation_id": self.conversation_id,
            "intent_confidence": self.intent_confidence,
            "verified_reply": dict(self.verified_reply) if self.verified_reply is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationUnit:
        """Construct EvaluationUnit from dictionary."""
        required = ["example_id", "customer_query", "predicted_intent"]
        missing = [f for f in required if f not in data]
        if missing:
            raise KeyError(f"Missing required evaluation unit fields: {missing}")

        return cls(
            example_id=str(data["example_id"]),
            customer_query=str(data["customer_query"]),
            predicted_intent=str(data["predicted_intent"]),
            retrieved_evidence=list(data.get("retrieved_evidence") or []),
            generated_reply=dict(data.get("generated_reply") or {}),
            conversation_id=str(data["conversation_id"]) if data.get("conversation_id") is not None else None,
            intent_confidence=float(data["intent_confidence"]) if data.get("intent_confidence") is not None else None,
            verified_reply=dict(data["verified_reply"]) if data.get("verified_reply") is not None else None,
        )
