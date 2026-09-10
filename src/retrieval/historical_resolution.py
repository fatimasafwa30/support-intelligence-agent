"""Historical resolution data structures for customer support retrieval.

Represents a historical support interaction consisting of an inbound customer
problem query paired with an outbound brand resolution response.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HistoricalResolution:
    """A historical customer-brand resolution pair extracted from past interactions."""

    resolution_id: str
    conversation_id: str
    customer_tweet_id: str
    brand_tweet_id: str
    customer_text: str
    brand_text: str
    intent: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the historical resolution to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoricalResolution:
        """Construct a HistoricalResolution instance from a dictionary."""
        return cls(
            resolution_id=str(data["resolution_id"]),
            conversation_id=str(data["conversation_id"]),
            customer_tweet_id=str(data["customer_tweet_id"]),
            brand_tweet_id=str(data["brand_tweet_id"]),
            customer_text=str(data.get("customer_text") or ""),
            brand_text=str(data.get("brand_text") or ""),
            intent=str(data["intent"]) if data.get("intent") is not None else None,
            created_at=str(data["created_at"]) if data.get("created_at") is not None else None,
        )


@dataclass(frozen=True)
class RetrievalResult:
    """A scored retrieval candidate result returned for a query."""

    resolution: HistoricalResolution
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the retrieval result to a dictionary."""
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "resolution": self.resolution.to_dict(),
        }
