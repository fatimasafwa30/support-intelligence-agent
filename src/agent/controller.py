"""Agent Controller orchestrating customer support understanding, retrieval, risk, and escalation.

Implements a dynamic state machine that makes explicit agentic decisions (answering,
clarifying, retrying retrieval, or escalating) around frozen models and components.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable

import joblib
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent_state import (
    ACTION_ASK_CLARIFICATION,
    ACTION_AUTO_HANDLE,
    ACTION_ESCALATE,
    SUFFICIENCY_BORDERLINE,
    SUFFICIENCY_EMPTY,
    SUFFICIENCY_INSUFFICIENT,
    SUFFICIENCY_SUFFICIENT,
    AgentState,
)
from src.agent.escalation_policy import EscalationDecision, EscalationEngine
from src.agent.grounded_generator import BaseReplyGenerator, create_reply_generator
from src.agent.grounding_guard import verify_and_filter_reply
from src.agent.reply_schemas import EvidenceItem, GenerationRequest, GroundedReply
from src.agent.risk_detector import CRITICAL_SAFETY, CRITICAL_SECURITY, RiskAssessment, RiskDetector
from src.retrieval.tfidf_retriever import TFIDFRetriever

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "agent_controller.yaml"
DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "intent_classifier_baseline.joblib"
DEFAULT_RETRIEVER_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "tfidf_retriever.joblib"


class AgentController:
    """Dynamic controller orchestrating the customer support reasoning and action lifecycle."""

    def __init__(
        self,
        classifier: Any = None,
        retriever: TFIDFRetriever | None = None,
        generator: BaseReplyGenerator | None = None,
        risk_detector: RiskDetector | None = None,
        escalation_engine: EscalationEngine | None = None,
        config_path: str | Path | None = None,
        max_attempts: int | None = None,
        intent_confidence_threshold: float | None = None,
        retrieval_similarity_threshold: float | None = None,
        borderline_retrieval_threshold: float | None = None,
    ) -> None:
        """Initialize controller with injected or lazily loaded frozen components."""
        cfg_path = Path(config_path or DEFAULT_CONFIG_PATH)
        cfg = self._load_config(cfg_path)

        limits = cfg.get("limits", {})
        thresholds = cfg.get("thresholds", {})
        query_aug = cfg.get("query_augmentation", {})

        # Configuration & Thresholds (treated as heuristic engineering thresholds)
        self.max_attempts = int(max_attempts if max_attempts is not None else limits.get("max_attempts", 2))
        self.max_retrieval_results = int(limits.get("max_retrieval_results", 3))
        self.intent_confidence_threshold = float(
            intent_confidence_threshold if intent_confidence_threshold is not None
            else thresholds.get("intent_confidence_threshold", 0.50)
        )
        self.retrieval_similarity_threshold = float(
            retrieval_similarity_threshold if retrieval_similarity_threshold is not None
            else thresholds.get("retrieval_similarity_threshold", 0.35)
        )
        self.borderline_retrieval_threshold = float(
            borderline_retrieval_threshold if borderline_retrieval_threshold is not None
            else thresholds.get("borderline_retrieval_threshold", 0.25)
        )
        self.query_augmentation_enabled = bool(query_aug.get("enabled", True))
        self.min_confidence_for_augmentation = float(query_aug.get("min_confidence_for_augmentation", 0.50))

        # Component Initialization
        self.classifier = classifier
        self._classifier_loaded = classifier is not None

        self.retriever = retriever
        self._retriever_loaded = retriever is not None

        self.generator = generator or create_reply_generator(mock=True)
        self.risk_detector = risk_detector or RiskDetector()
        self.escalation_engine = escalation_engine or EscalationEngine(
            intent_confidence_threshold=self.intent_confidence_threshold,
            retrieval_similarity_threshold=self.retrieval_similarity_threshold,
        )

    def _load_config(self, path: Path) -> dict[str, Any]:
        """Load YAML configuration safely."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _ensure_classifier(self) -> Any:
        """Lazily load the frozen classifier artifact if not injected."""
        if not self._classifier_loaded:
            if DEFAULT_CLASSIFIER_PATH.exists():
                artifact = joblib.load(DEFAULT_CLASSIFIER_PATH)
                self.classifier = artifact["pipeline"]
            else:
                logger.warning("Classifier model artifact not found at %s", DEFAULT_CLASSIFIER_PATH)
                self.classifier = None
            self._classifier_loaded = True
        return self.classifier

    def _ensure_retriever(self) -> TFIDFRetriever | None:
        """Lazily load the frozen retriever artifact if not injected."""
        if not self._retriever_loaded:
            if DEFAULT_RETRIEVER_PATH.exists():
                self.retriever = TFIDFRetriever.load(DEFAULT_RETRIEVER_PATH)
            else:
                logger.warning("Retriever artifact not found at %s", DEFAULT_RETRIEVER_PATH)
                self.retriever = None
            self._retriever_loaded = True
        return self.retriever

    def process_query(
        self,
        customer_message: str,
        conversation_id: str | None = None,
    ) -> AgentState:
        """Execute dynamic agentic orchestration for an incoming customer inquiry.

        Follows state-driven reasoning:
        1. Understand / Classify intent
        2. Early critical-risk safety check (short-circuit before retrieval)
        3. Retrieve evidence with sufficiency assessment & bounded retry
        4. Pre-generation boundary evaluation (hardware damage / legal / billing)
        5. Generate grounded reply & verify exact URLs and evidence IDs
        6. Delegate final action to EscalationEngine (AUTO_HANDLE, ASK_CLARIFICATION, ESCALATE)
        """
        state = AgentState(
            customer_message=customer_message,
            conversation_id=conversation_id,
        )

        try:
            # ── Step 1: Understand Intent ──────────────────────────────────
            self._step_understand(state)

            # ── Step 2: Critical Risk Check (Pre-Retrieval Short-Circuit) ──
            if self._step_critical_risk_check(state):
                state.complete()
                return state

            # ── Step 3: Evidence Retrieval & Sufficiency (Bounded Loop) ────
            self._step_retrieve_evidence(state)

            # ── Step 4: Pre-Generation Boundary Check ──────────────────────
            if self._step_pre_generation_check(state):
                state.complete()
                return state

            # ── Step 5: Grounded Reply Generation & Verification ───────────
            self._step_generate_and_verify(state)

            # ── Step 6: Final Action Decision Delegation ───────────────────
            self._step_final_decision(state)

        except Exception as err:
            logger.exception("Unexpected error in AgentController.process_query: %s", err)
            state.record_step(
                step_name="ERROR_FALLBACK",
                action="ESCALATE",
                details={"error": str(err)},
            )
            state.final_action = ACTION_ESCALATE
            state.target_queue = "general_support"
            state.decision_reasons.append(f"controller_runtime_error: {type(err).__name__}")

        state.complete()
        return state

    # ──────────────────────────────────────────────────────────────────────────
    # Step Implementations
    # ──────────────────────────────────────────────────────────────────────────

    def _step_understand(self, state: AgentState) -> None:
        """Classify customer intent and confidence score."""
        clf = self._ensure_classifier()
        if clf is not None:
            try:
                raw_probs = clf.predict_proba([state.customer_message])[0]
                if hasattr(raw_probs, "argmax"):
                    best_idx = int(raw_probs.argmax())
                else:
                    best_idx = max(range(len(raw_probs)), key=lambda i: raw_probs[i])
                state.predicted_intent = str(clf.classes_[best_idx])
                state.intent_confidence = float(raw_probs[best_idx])
            except Exception as err:
                logger.warning("Classifier prediction failed: %s", err)
                state.predicted_intent = "other_unclear"
                state.intent_confidence = 0.0
        else:
            state.predicted_intent = "other_unclear"
            state.intent_confidence = 0.0

        state.record_step(
            step_name="UNDERSTAND",
            action="CLASSIFIED",
            details={
                "intent": state.predicted_intent,
                "confidence": round(state.intent_confidence, 4) if state.intent_confidence is not None else None,
            },
        )

    def _step_critical_risk_check(self, state: AgentState) -> bool:
        """Run critical-risk check BEFORE retrieval; short-circuit immediate hazards."""
        risk = self.risk_detector.assess_risk(state.customer_message)
        state.risk_assessment = risk

        state.record_step(
            step_name="ASSESS_RISK",
            action=f"RISK_{risk.risk_level.upper()}",
            details={
                "risk_level": risk.risk_level,
                "categories": list(risk.risk_categories),
                "matched_signals": list(risk.matched_signals),
            },
        )

        if risk.risk_level == "critical":
            decision = self.escalation_engine.decide(
                query=state.customer_message,
                predicted_intent=state.predicted_intent or "other_unclear",
                intent_confidence=state.intent_confidence,
                top_evidence=[],
                risk_assessment=risk,
            )
            state.final_action = decision.action
            state.target_queue = decision.routing_target
            state.decision_reasons = list(decision.reasons)
            state.decision_reasons.append("early_critical_hazard_short_circuit")

            state.record_step(
                step_name="CRITICAL_SHORT_CIRCUIT",
                action="ESCALATE",
                details={
                    "target_queue": state.target_queue,
                    "reasons": state.decision_reasons,
                },
            )
            return True

        return False

    def _step_retrieve_evidence(self, state: AgentState) -> None:
        """Retrieve resolution evidence with sufficiency evaluation and bounded retry."""
        retriever = self._ensure_retriever()
        if retriever is None:
            state.evidence_sufficiency = SUFFICIENCY_EMPTY
            state.record_step(
                step_name="RETRIEVE",
                action="SKIPPED",
                details={"reason": "retriever_unavailable"},
            )
            return

        # Attempt 1: Search using initial customer query
        state.retrieval_attempts += 1
        ret_results = retriever.retrieve(state.customer_message, top_k=self.max_retrieval_results)
        evidence_items = [EvidenceItem.from_retrieval_result(r) for r in ret_results]
        top_sim = float(ret_results[0].score) if ret_results else 0.0

        state.retrieved_evidence = evidence_items
        state.top_similarity_score = top_sim
        state.evidence_sufficiency = self._classify_sufficiency(top_sim, bool(evidence_items))

        state.record_step(
            step_name="RETRIEVE_ATTEMPT_1",
            action="EVALUATE_EVIDENCE",
            details={
                "attempt": state.retrieval_attempts,
                "top_similarity": round(top_sim, 4),
                "sufficiency": state.evidence_sufficiency,
                "candidates_count": len(evidence_items),
            },
        )

        # Explicit Agentic Decision: Bounded Retrieval Retry on Borderline Evidence
        can_retry = (
            state.retrieval_attempts < self.max_attempts
            and self.query_augmentation_enabled
            and state.evidence_sufficiency == SUFFICIENCY_BORDERLINE
            and (state.intent_confidence or 0.0) >= self.min_confidence_for_augmentation
            and state.predicted_intent
            and state.predicted_intent != "other_unclear"
        )

        if can_retry:
            augmented_query = f"{state.customer_message} {state.predicted_intent.replace('_', ' ')}"
            state.record_step(
                step_name="DECIDE_RETRIEVAL_RETRY",
                action="AUGMENT_QUERY_AND_RETRY",
                details={
                    "reason": "borderline_evidence_with_high_intent_confidence",
                    "augmented_query": augmented_query,
                    "attempt": state.retrieval_attempts + 1,
                },
            )

            # Attempt 2: Bounded retry with augmented query
            state.retrieval_attempts += 1
            retry_results = retriever.retrieve(augmented_query, top_k=self.max_retrieval_results)
            retry_evidence = [EvidenceItem.from_retrieval_result(r) for r in retry_results]
            retry_sim = float(retry_results[0].score) if retry_results else 0.0

            # Adopt retry results if they improve evidence quality
            if retry_sim > top_sim:
                state.retrieved_evidence = retry_evidence
                state.top_similarity_score = retry_sim
                state.evidence_sufficiency = self._classify_sufficiency(retry_sim, bool(retry_evidence))
                state.record_step(
                    step_name="RETRIEVE_ATTEMPT_2",
                    action="ADOPTED_RETRY_EVIDENCE",
                    details={
                        "old_sim": round(top_sim, 4),
                        "new_sim": round(retry_sim, 4),
                        "sufficiency": state.evidence_sufficiency,
                    },
                )
            else:
                state.record_step(
                    step_name="RETRIEVE_ATTEMPT_2",
                    action="RETAINED_ORIGINAL_EVIDENCE",
                    details={
                        "attempt_2_sim": round(retry_sim, 4),
                        "retained_sim": round(top_sim, 4),
                    },
                )

    def _classify_sufficiency(self, score: float, has_items: bool) -> str:
        """Classify retrieval score into operational sufficiency categories."""
        if not has_items:
            return SUFFICIENCY_EMPTY
        if score >= self.retrieval_similarity_threshold:
            return SUFFICIENCY_SUFFICIENT
        if score >= self.borderline_retrieval_threshold:
            return SUFFICIENCY_BORDERLINE
        return SUFFICIENCY_INSUFFICIENT

    def _step_pre_generation_check(self, state: AgentState) -> bool:
        """Consult EscalationEngine prior to reply generation; short-circuit if escalation is required."""
        preliminary_decision = self.escalation_engine.decide(
            query=state.customer_message,
            predicted_intent=state.predicted_intent or "other_unclear",
            intent_confidence=state.intent_confidence,
            top_evidence=state.retrieved_evidence,
            grounded_reply=None,
            risk_assessment=state.risk_assessment,
        )

        if preliminary_decision.action == ACTION_ESCALATE:
            state.final_action = preliminary_decision.action
            state.target_queue = preliminary_decision.routing_target
            state.decision_reasons = list(preliminary_decision.reasons)
            state.decision_reasons.append("escalation_engine_pre_generation_short_circuit")

            state.record_step(
                step_name="PRE_GENERATION_SHORT_CIRCUIT",
                action=ACTION_ESCALATE,
                details={
                    "target_queue": state.target_queue,
                    "reasons": state.decision_reasons,
                    "requires_human": preliminary_decision.requires_human,
                },
            )
            return True

        return False

    def _step_generate_and_verify(self, state: AgentState) -> None:
        """Generate grounded reply and verify URLs/evidence IDs with bounded revision."""
        ret_status = "strong" if state.evidence_sufficiency == SUFFICIENCY_SUFFICIENT else "heuristic_weak"

        # Attempt 1: Generate initial reply
        state.generation_attempts += 1
        gen_req = GenerationRequest(
            customer_query=state.customer_message,
            intent=state.predicted_intent,
            intent_confidence=state.intent_confidence,
            retrieved_evidence=state.retrieved_evidence,
            retrieval_status=ret_status,
            conversation_id=state.conversation_id,
        )
        draft_reply = self.generator.generate(gen_req)

        # Grounding Guard Verification
        verified_reply = verify_and_filter_reply(draft_reply, gen_req)
        urls_modified = set(draft_reply.used_urls) != set(verified_reply.used_urls)

        state.generated_reply = verified_reply
        state.verification_passed = verified_reply.grounded and not urls_modified
        state.verification_details = {
            "verified_evidence_ids": list(verified_reply.used_evidence_ids),
            "verified_urls": list(verified_reply.used_urls),
            "urls_modified": urls_modified,
            "action": verified_reply.action,
        }

        state.record_step(
            step_name="GENERATE_AND_VERIFY_1",
            action="VERIFIED",
            details={
                "attempt": state.generation_attempts,
                "grounded": verified_reply.grounded,
                "urls_modified": urls_modified,
                "reply_action": verified_reply.action,
            },
        )

        # Explicit Agentic Decision: Bounded Revision if Grounding Verification Failed
        can_retry_generation = (
            urls_modified
            and state.generation_attempts < self.max_attempts
            and ret_status == "strong"
        )

        if can_retry_generation:
            state.record_step(
                step_name="DECIDE_GENERATION_REVISION",
                action="REGENERATE_CONSERVATIVE_REPLY",
                details={"reason": "ungrounded_urls_stripped_in_attempt_1"},
            )

            # Attempt 2: Bounded generation revision
            state.generation_attempts += 1
            revised_reply = self.generator.generate(gen_req)
            reverified_reply = verify_and_filter_reply(revised_reply, gen_req)

            state.generated_reply = reverified_reply
            state.verification_passed = reverified_reply.grounded
            state.record_step(
                step_name="GENERATE_AND_VERIFY_2",
                action="REVERIFIED",
                details={
                    "attempt": state.generation_attempts,
                    "grounded": reverified_reply.grounded,
                    "reply_action": reverified_reply.action,
                },
            )

    def _step_final_decision(self, state: AgentState) -> None:
        """Delegate final operational action and routing target to EscalationEngine."""
        decision: EscalationDecision = self.escalation_engine.decide(
            query=state.customer_message,
            predicted_intent=state.predicted_intent or "other_unclear",
            intent_confidence=state.intent_confidence,
            top_evidence=state.retrieved_evidence,
            grounded_reply=state.generated_reply,
            risk_assessment=state.risk_assessment,
        )

        state.final_action = decision.action
        state.target_queue = decision.routing_target
        state.decision_reasons = list(decision.reasons)

        state.record_step(
            step_name="FINAL_DECISION",
            action=state.final_action,
            details={
                "target_queue": state.target_queue,
                "reasons": state.decision_reasons,
                "requires_human": decision.requires_human,
            },
        )
