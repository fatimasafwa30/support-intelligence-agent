"""Confidence and risk-aware escalation policy engine.

Decouples intent classification from operational routing decisions (AUTO_HANDLE vs ESCALATE).
Gated on model confidence, retrieval similarity, grounding verification, and safety risks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "escalation.yaml"

# Default engineering heuristic thresholds
DEFAULT_INTENT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD = 0.35


@dataclass(frozen=True)
class EscalationDecision:
    """Operational routing decision for customer support inquiry."""

    action: str  # "AUTO_HANDLE", "ESCALATE", "ASK_CLARIFICATION"
    risk_level: str
    risk_categories: list[str]
    requires_human: bool
    routing_target: str  # "self_service", "safety_team", "account_security", "hardware_repair", "billing_support", "general_support"
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to dictionary."""
        return asdict(self)


class EscalationEngine:
    """Policy engine deciding whether to auto-handle, escalate, or clarify."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        intent_confidence_threshold: float | None = None,
        retrieval_similarity_threshold: float | None = None,
    ) -> None:
        cfg_path = Path(config_path or DEFAULT_CONFIG_PATH)
        cfg = self._load_config(cfg_path)

        thresholds = cfg.get("thresholds", {})
        if intent_confidence_threshold is not None:
            self.intent_confidence_threshold = float(intent_confidence_threshold)
        else:
            self.intent_confidence_threshold = float(
                thresholds.get("intent_confidence_threshold", DEFAULT_INTENT_CONFIDENCE_THRESHOLD)
            )

        if retrieval_similarity_threshold is not None:
            self.retrieval_similarity_threshold = float(retrieval_similarity_threshold)
        else:
            self.retrieval_similarity_threshold = float(
                thresholds.get("retrieval_similarity_threshold", DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD)
            )
        self.risk_detector = RiskDetector(config_path=cfg_path)

    def _load_config(self, path: Path) -> dict[str, Any]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def decide(
        self,
        query: str,
        predicted_intent: str | None = None,
        intent_confidence: float | None = None,
        top_evidence: Sequence[EvidenceItem] | None = None,
        grounded_reply: GroundedReply | None = None,
        risk_assessment: RiskAssessment | None = None,
    ) -> EscalationDecision:
        """Evaluate inquiry context and make a deterministic routing decision.

        Args:
            query: Inbound customer message text.
            predicted_intent: Predicted canonical intent from ML classifier.
            intent_confidence: Model confidence score for predicted intent.
            top_evidence: Retrieved historical resolution evidence items.
            grounded_reply: Optional drafted reply from generation module.
            risk_assessment: Optional pre-computed risk assessment.

        Returns:
            Structured EscalationDecision.
        """
        evidence_list = list(top_evidence or [])
        risk = risk_assessment or self.risk_detector.assess_risk(query)
        reasons: list[str] = []

        # ── 1. Critical Safety & Security (Zero-Tolerance Immediate Escalation) ─
        if risk.risk_level == "critical":
            reasons.append("critical_risk_protocol_triggered")
            if CRITICAL_SAFETY in risk.risk_categories:
                reasons.append("physical_safety_hazard_detected")
                return EscalationDecision(
                    action="ESCALATE",
                    risk_level=risk.risk_level,
                    risk_categories=risk.risk_categories,
                    requires_human=True,
                    routing_target="safety_team",
                    reasons=reasons,
                )
            else:
                reasons.append("critical_security_compromise_detected")
                return EscalationDecision(
                    action="ESCALATE",
                    risk_level=risk.risk_level,
                    risk_categories=risk.risk_categories,
                    requires_human=True,
                    routing_target="account_security",
                    reasons=reasons,
                )

        # ── 2. Physical Hardware Damage Override ──────────────────────────────
        if PHYSICAL_DAMAGE in risk.risk_categories:
            reasons.append("physical_hardware_damage_requires_inspection")
            return EscalationDecision(
                action="ESCALATE",
                risk_level=risk.risk_level,
                risk_categories=risk.risk_categories,
                requires_human=True,
                routing_target="hardware_repair",
                reasons=reasons,
            )

        # ── 3. Financial Disputes & Legal Escalations ─────────────────────────
        if FINANCIAL_DISPUTE in risk.risk_categories:
            reasons.append("financial_transaction_dispute")
            return EscalationDecision(
                action="ESCALATE",
                risk_level=risk.risk_level,
                risk_categories=risk.risk_categories,
                requires_human=True,
                routing_target="billing_support",
                reasons=reasons,
            )

        if LEGAL_REGULATORY in risk.risk_categories:
            reasons.append("legal_regulatory_escalation")
            return EscalationDecision(
                action="ESCALATE",
                risk_level=risk.risk_level,
                risk_categories=risk.risk_categories,
                requires_human=True,
                routing_target="general_support",
                reasons=reasons,
            )

        # ── 4. High-Touch Support Boundaries (Repair Service & Account Recovery) 
        if predicted_intent == "repair_service":
            reasons.append("repair_service_mandates_human_booking")
            return EscalationDecision(
                action="ESCALATE",
                risk_level=risk.risk_level,
                risk_categories=risk.risk_categories,
                requires_human=True,
                routing_target="hardware_repair",
                reasons=reasons,
            )

        # ── 5. Confidence & Retrieval Gating (Model Uncertainty) ──────────────
        conf = intent_confidence if intent_confidence is not None else 0.0
        best_sim = evidence_list[0].similarity_score if evidence_list else 0.0

        is_unclear_intent = predicted_intent == "other_unclear"
        is_low_conf = (conf < self.intent_confidence_threshold) or is_unclear_intent
        is_weak_retrieval = (not evidence_list) or (best_sim < self.retrieval_similarity_threshold)

        if is_low_conf or is_weak_retrieval:
            if is_unclear_intent:
                reasons.append("unclear_intent_requires_resolution")
            if conf < self.intent_confidence_threshold:
                reasons.append(f"intent_confidence_below_threshold ({conf:.3f} < {self.intent_confidence_threshold})")
            if is_weak_retrieval:
                reasons.append(f"retrieval_similarity_below_threshold ({best_sim:.3f} < {self.retrieval_similarity_threshold})")

            # If the inquiry is brief or vague, prompt for clarification; otherwise route to human
            word_count = len(query.split())
            if word_count <= 8 or is_low_conf and is_weak_retrieval:
                reasons.append("vague_inquiry_requested_clarification")
                return EscalationDecision(
                    action="ASK_CLARIFICATION",
                    risk_level=risk.risk_level,
                    risk_categories=risk.risk_categories,
                    requires_human=False,
                    routing_target="general_support",
                    reasons=reasons,
                )
            else:
                reasons.append("uncertain_technical_inquiry_escalated")
                return EscalationDecision(
                    action="ESCALATE",
                    risk_level=risk.risk_level,
                    risk_categories=risk.risk_categories,
                    requires_human=True,
                    routing_target="general_support",
                    reasons=reasons,
                )

        # ── 6. Grounded Self-Service Resolution (AUTO_HANDLE) ─────────────────
        # Intent vs Action Independence: Even if risk is medium, if a proven
        # official self-service resolution exists and model confidence is high:
        top_ev = evidence_list[0]
        has_url = len(top_ev.extracted_urls) > 0
        has_grounded_reply = (grounded_reply is None) or (grounded_reply.grounded and grounded_reply.action == "AUTO_REPLY")

        if has_url and has_grounded_reply:
            reasons.append(f"high_confidence_intent ({conf:.3f} >= {self.intent_confidence_threshold})")
            reasons.append(f"strong_retrieval_evidence ({best_sim:.3f} >= {self.retrieval_similarity_threshold})")
            reasons.append("verified_official_self_service_resolution_available")
            return EscalationDecision(
                action="AUTO_HANDLE",
                risk_level=risk.risk_level,
                risk_categories=risk.risk_categories,
                requires_human=False,
                routing_target="self_service",
                reasons=reasons,
            )

        # ── 7. Default Fallback ───────────────────────────────────────────────
        reasons.append("lacks_verified_self_service_link_escalated_to_agent")
        return EscalationDecision(
            action="ESCALATE",
            risk_level=risk.risk_level,
            risk_categories=risk.risk_categories,
            requires_human=True,
            routing_target="general_support",
            reasons=reasons,
        )
