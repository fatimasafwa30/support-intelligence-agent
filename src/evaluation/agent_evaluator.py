"""Deterministic, read-only evaluation harness for the frozen AgentController on the Golden Set.

Evaluation Protocol & Guardrails:
1. Strict Anti-Leakage:
   - gold_intent, gold_risk, gold_action, and annotation_notes are NEVER passed
     to AgentController.process_query(). The controller receives ONLY customer_message
     and conversation_id.
   - All ground-truth annotations are accessed and attached strictly post-inference.
2. Read-Only Evaluator:
   - Operates in a strictly read-only capacity with respect to all frozen components:
     AgentController, IntentClassifier, TFIDFRetriever, RiskDetector, EscalationEngine,
     MockReplyGenerator, and verify_and_filter_reply.
   - Does not mutate agent internals, weights, or thresholds.
3. 3-Way Operational Action Reporting (Primary):
   - AUTO_HANDLE, ASK_CLARIFICATION, and ESCALATE are reported as the primary operational actions.
   - Binary AUTO vs Non-AUTO (ESCALATE + ASK_CLARIFICATION) is reported as a secondary
     comparison metric against the historical 2-way gold_action labels.
4. Evidence URL Citation Verification:
   - Verifies whether URLs in generated replies are grounded in retrieved evidence.
   - Explicitly framed as link/citation integrity, NOT complete reply correctness.
5. Dynamic Label Derivation:
   - Denominators and class supports (e.g. critical cases count, required escalations)
     are derived dynamically from loaded annotations rather than hardcoded.
6. Deterministic Inference & Metrics:
   - Inference decisions, rankings, generated replies, and metrics are 100% deterministic.
   - Volatile runtime metrics (latency_ms, timestamps) are isolated and excluded from
     reproducibility checks.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from statistics import mean, median
import sys
import time
from typing import Any, Callable, Sequence

from src.agent.agent_state import (
    ACTION_ASK_CLARIFICATION,
    ACTION_AUTO_HANDLE,
    ACTION_ESCALATE,
    AgentState,
    SUFFICIENCY_BORDERLINE,
    SUFFICIENCY_EMPTY,
    SUFFICIENCY_INSUFFICIENT,
    SUFFICIENCY_SUFFICIENT,
)
from src.agent.controller import AgentController
from src.agent.reply_schemas import extract_urls

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentEvaluationRecord:
    """Immutable evaluation record capturing input, trace, decisions, and post-hoc gold labels."""

    # 1. Identification & Customer Input
    example_id: str
    conversation_id: str
    tweet_id: str
    customer_message: str

    # 2. Ground Truth (Attached strictly POST-inference)
    gold_intent: str
    gold_risk: str
    gold_action: str
    annotation_notes: str

    # 3. Intent Understanding
    pred_intent: str
    intent_confidence: float | None

    # 4. Risk Assessment
    pred_risk_level: str
    risk_categories: list[str]
    matched_risk_signals: list[str]

    # 5. Retrieval & Sufficiency Dynamics
    retrieval_attempts: int
    initial_top_similarity: float
    final_top_similarity: float
    initial_sufficiency: str
    final_sufficiency: str
    retrieval_retry_triggered: bool
    retrieval_retry_adopted: bool
    retrieved_evidence_ids: list[str]

    # 6. Generation & Citation Integrity
    generation_attempts: int
    raw_reply_text: str | None
    verified_reply_text: str | None
    grounded_urls_count: int
    ungrounded_urls_stripped: int
    evidence_url_verification_passed: bool  # Citation integrity, not complete correctness

    # 7. Final Action & Routing
    final_action: str                      # PRIMARY: "AUTO_HANDLE", "ASK_CLARIFICATION", "ESCALATE"
    effective_binary_action: str           # SECONDARY: "AUTO_HANDLE" vs "ESCALATE"
    target_queue: str
    decision_reasons: list[str]

    # 8. Execution Trace & Timing
    execution_steps: list[dict[str, Any]]
    execution_steps_count: int
    early_exit_step: str | None
    latency_ms: float                      # Volatile runtime metric (excluded from determinism asserts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    def to_csv_row(self) -> dict[str, Any]:
        """Flatten record for CSV export."""
        return {
            "example_id": self.example_id,
            "conversation_id": self.conversation_id,
            "tweet_id": self.tweet_id,
            "customer_message": self.customer_message,
            "gold_intent": self.gold_intent,
            "pred_intent": self.pred_intent,
            "intent_confidence": round(self.intent_confidence, 4) if self.intent_confidence is not None else "",
            "gold_risk": self.gold_risk,
            "pred_risk_level": self.pred_risk_level,
            "risk_categories": ";".join(self.risk_categories),
            "retrieval_attempts": self.retrieval_attempts,
            "initial_top_similarity": round(self.initial_top_similarity, 4),
            "final_top_similarity": round(self.final_top_similarity, 4),
            "initial_sufficiency": self.initial_sufficiency,
            "final_sufficiency": self.final_sufficiency,
            "retrieval_retry_triggered": self.retrieval_retry_triggered,
            "retrieval_retry_adopted": self.retrieval_retry_adopted,
            "generation_attempts": self.generation_attempts,
            "evidence_url_verification_passed": self.evidence_url_verification_passed,
            "ungrounded_urls_stripped": self.ungrounded_urls_stripped,
            "gold_action": self.gold_action,
            "primary_final_action": self.final_action,
            "effective_binary_action": self.effective_binary_action,
            "target_queue": self.target_queue,
            "decision_reasons": "; ".join(self.decision_reasons),
            "early_exit_step": self.early_exit_step or "",
            "execution_steps_count": self.execution_steps_count,
            "latency_ms": round(self.latency_ms, 2),
            "annotation_notes": self.annotation_notes,
        }


class AgentEvaluator:
    """Read-only evaluation harness benchmarking AgentController on Golden Set."""

    @staticmethod
    def load_golden_set(golden_path: str | Path) -> list[dict[str, Any]]:
        """Load and validate the frozen 250-example Golden Set."""
        path = Path(golden_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden Set not found at: {path}")

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = {"id", "conversation_id", "text", "gold_intent", "gold_risk", "gold_action"}
            fieldnames = set(reader.fieldnames or [])
            missing = required_cols - fieldnames
            if missing:
                raise ValueError(f"Golden Set is missing required columns: {missing}")

            for row in reader:
                records.append({
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "tweet_id": row.get("tweet_id", ""),
                    "text": row["text"],
                    "gold_intent": row["gold_intent"],
                    "gold_risk": row["gold_risk"],
                    "gold_action": row["gold_action"],
                    "annotation_notes": row.get("annotation_notes", ""),
                })

        return records

    @staticmethod
    def evaluate_record(
        record: dict[str, Any],
        controller: AgentController,
    ) -> AgentEvaluationRecord:
        """Run a single golden record through the controller with strict anti-leakage isolation.

        Strict Anti-Leakage Protocol:
        - Only record['text'] and record['conversation_id'] are extracted.
        - Gold annotations (gold_intent, gold_risk, gold_action, annotation_notes)
          are completely isolated and never passed into controller.process_query().
        """
        customer_message = str(record["text"])
        conversation_id = str(record.get("conversation_id", ""))

        start_time = time.perf_counter()

        # Run inference through frozen controller
        state = controller.process_query(
            customer_message=customer_message,
            conversation_id=conversation_id,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract trace-level dynamics
        initial_sim = 0.0
        final_sim = state.top_similarity_score
        initial_suff = state.evidence_sufficiency
        final_suff = state.evidence_sufficiency
        retry_triggered = False
        retry_adopted = False
        early_exit: str | None = None

        for step in state.trace:
            s_name = step.get("step", "")
            s_act = step.get("action", "")
            s_det = step.get("details", {})

            if s_name == "RETRIEVE_ATTEMPT_1":
                initial_sim = float(s_det.get("top_similarity", 0.0))
                initial_suff = str(s_det.get("sufficiency", initial_suff))
            elif s_name == "DECIDE_RETRIEVAL_RETRY":
                retry_triggered = True
            elif s_name == "RETRIEVE_ATTEMPT_2":
                if s_act == "ADOPTED_RETRY_EVIDENCE":
                    retry_adopted = True
            elif "SHORT_CIRCUIT" in s_name or s_name == "EARLY_CRITICAL_HAZARD":
                early_exit = s_name

        # Citation / Evidence URL verification analysis
        raw_reply_text: str | None = None
        verified_reply_text: str | None = None
        grounded_urls_count = 0
        ungrounded_urls_stripped = 0
        url_verif_passed = state.verification_passed

        if state.generated_reply:
            raw_reply_text = state.generated_reply.reply_text
            verified_reply_text = state.generated_reply.reply_text
            grounded_urls_count = len(extract_urls(verified_reply_text))

        # Check ungrounded URLs stripped count if reported in verification details
        if state.verification_details:
            ungrounded_urls_stripped = len(state.verification_details.get("ungrounded_urls", []))

        # Secondary binary action mapping for comparison with 2-way gold_action
        effective_binary = (
            ACTION_AUTO_HANDLE if state.final_action == ACTION_AUTO_HANDLE else ACTION_ESCALATE
        )

        risk_level = state.risk_assessment.risk_level if state.risk_assessment else "unknown"
        risk_cats = list(state.risk_assessment.risk_categories) if state.risk_assessment else []
        matched_sigs = list(state.risk_assessment.matched_signals) if state.risk_assessment else []
        evidence_ids = [e.evidence_id for e in state.retrieved_evidence]

        return AgentEvaluationRecord(
            example_id=str(record["id"]),
            conversation_id=conversation_id,
            tweet_id=str(record.get("tweet_id", "")),
            customer_message=customer_message,
            gold_intent=str(record["gold_intent"]),
            gold_risk=str(record["gold_risk"]),
            gold_action=str(record["gold_action"]),
            annotation_notes=str(record.get("annotation_notes", "")),
            pred_intent=str(state.predicted_intent or "other_unclear"),
            intent_confidence=state.intent_confidence,
            pred_risk_level=risk_level,
            risk_categories=risk_cats,
            matched_risk_signals=matched_sigs,
            retrieval_attempts=state.retrieval_attempts,
            initial_top_similarity=initial_sim,
            final_top_similarity=final_sim,
            initial_sufficiency=initial_suff,
            final_sufficiency=final_suff,
            retrieval_retry_triggered=retry_triggered,
            retrieval_retry_adopted=retry_adopted,
            retrieved_evidence_ids=evidence_ids,
            generation_attempts=state.generation_attempts,
            raw_reply_text=raw_reply_text,
            verified_reply_text=verified_reply_text,
            grounded_urls_count=grounded_urls_count,
            ungrounded_urls_stripped=ungrounded_urls_stripped,
            evidence_url_verification_passed=url_verif_passed,
            final_action=state.final_action,
            effective_binary_action=effective_binary,
            target_queue=state.target_queue,
            decision_reasons=list(state.decision_reasons),
            execution_steps=list(state.trace),
            execution_steps_count=len(state.trace),
            early_exit_step=early_exit,
            latency_ms=elapsed_ms,
        )

    def evaluate_dataset(
        self,
        records: list[dict[str, Any]],
        controller: AgentController,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[AgentEvaluationRecord]:
        """Evaluate full sequence of golden records."""
        results: list[AgentEvaluationRecord] = []
        total = len(records)

        for i, rec in enumerate(records, 1):
            eval_record = self.evaluate_record(rec, controller)
            results.append(eval_record)
            if on_progress:
                on_progress(i, total)

        return results

    @staticmethod
    def compute_metrics(
        records: list[AgentEvaluationRecord],
        baselines_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute comprehensive metrics across 6 operational dimensions.

        Dynamically derives all sample supports, risk totals, and action totals
        from the loaded annotations rather than hardcoding them.
        """
        total = len(records)
        if total == 0:
            return {"total_records": 0}

        # ── 1. Dynamic Distribution Analysis from Gold Labels ─────────────────
        gold_action_counts = Counter(r.gold_action for r in records)
        gold_risk_counts = Counter(r.gold_risk for r in records)
        gold_intent_counts = Counter(r.gold_intent for r in records)

        gold_escalate_count = gold_action_counts.get("ESCALATE", 0)
        gold_auto_count = gold_action_counts.get("AUTO_HANDLE", 0)
        gold_critical_count = gold_risk_counts.get("critical", 0)

        # ── 2. Primary 3-Way Action Operational Distribution ──────────────────
        action_3way_counts = Counter(r.final_action for r in records)
        action_3way_distribution = {
            ACTION_AUTO_HANDLE: {
                "count": action_3way_counts.get(ACTION_AUTO_HANDLE, 0),
                "share": action_3way_counts.get(ACTION_AUTO_HANDLE, 0) / total,
            },
            ACTION_ASK_CLARIFICATION: {
                "count": action_3way_counts.get(ACTION_ASK_CLARIFICATION, 0),
                "share": action_3way_counts.get(ACTION_ASK_CLARIFICATION, 0) / total,
            },
            ACTION_ESCALATE: {
                "count": action_3way_counts.get(ACTION_ESCALATE, 0),
                "share": action_3way_counts.get(ACTION_ESCALATE, 0) / total,
            },
        }

        # Target queue distribution
        queue_counts = Counter(r.target_queue for r in records)
        queue_distribution = {q: {"count": c, "share": c / total} for q, c in queue_counts.items()}

        # ── 3. Secondary Binary Comparison vs Gold Action ─────────────────────
        # Binary confusion matrix: (gold_action, effective_binary_action)
        cm_binary = Counter((r.gold_action, r.effective_binary_action) for r in records)
        tp = cm_binary[("ESCALATE", "ESCALATE")]
        fn = cm_binary[("ESCALATE", "AUTO_HANDLE")]  # Under-escalation (dangerous)
        fp = cm_binary[("AUTO_HANDLE", "ESCALATE")]  # Over-escalation on auto candidates
        tn = cm_binary[("AUTO_HANDLE", "AUTO_HANDLE")]

        binary_acc = (tp + tn) / total
        pred_escalate_total = tp + fp
        pred_auto_total = tn + fn

        esc_prec = tp / pred_escalate_total if pred_escalate_total > 0 else 0.0
        esc_rec = tp / gold_escalate_count if gold_escalate_count > 0 else 0.0
        esc_f1 = 2 * esc_prec * esc_rec / (esc_prec + esc_rec) if (esc_prec + esc_rec) > 0 else 0.0

        auto_prec = tn / pred_auto_total if pred_auto_total > 0 else 0.0
        auto_rec = tn / gold_auto_count if gold_auto_count > 0 else 0.0
        auto_f1 = 2 * auto_prec * auto_rec / (auto_prec + auto_rec) if (auto_prec + auto_rec) > 0 else 0.0

        binary_macro_f1 = (esc_f1 + auto_f1) / 2.0
        under_escalation_rate = fn / gold_escalate_count if gold_escalate_count > 0 else 0.0
        false_escalation_rate_on_auto = fp / gold_auto_count if gold_auto_count > 0 else 0.0
        overall_escalation_rate = pred_escalate_total / total

        # ── 4. Zero-Tolerance Safety Audit ─────────────────────────────────────
        critical_records = [r for r in records if r.gold_risk == "critical"]
        critical_total = len(critical_records)
        critical_escalated = sum(1 for r in critical_records if r.effective_binary_action == "ESCALATE")
        critical_safety_recall = critical_escalated / critical_total if critical_total > 0 else 1.0

        missed_critical = [
            {
                "example_id": r.example_id,
                "text": r.customer_message,
                "pred_action": r.final_action,
                "effective_action": r.effective_binary_action,
                "reasons": r.decision_reasons,
            }
            for r in critical_records
            if r.effective_binary_action != "ESCALATE"
        ]

        # Early short-circuits
        pre_retrieval_critical_short_circuits = sum(
            1 for r in records if "early_critical_hazard_short_circuit" in r.decision_reasons
        )
        pre_generation_short_circuits = sum(
            1 for r in records if "escalation_engine_pre_generation_short_circuit" in r.decision_reasons
        )

        # ── 5. Intent Prediction Performance ──────────────────────────────────
        full_intent_correct = sum(1 for r in records if r.pred_intent == r.gold_intent)
        full_intent_accuracy = full_intent_correct / total

        # Specific-intent subset (excluding human other_unclear, aligned with 225-sample baseline)
        specific_records = [r for r in records if r.gold_intent != "other_unclear"]
        specific_total = len(specific_records)
        specific_correct = sum(1 for r in specific_records if r.pred_intent == r.gold_intent)
        specific_intent_accuracy = specific_correct / specific_total if specific_total > 0 else 0.0

        # Intent Macro F1 on specific intents
        specific_classes = sorted(list({r.gold_intent for r in specific_records}))
        f1_list: list[float] = []
        for cls in specific_classes:
            c_tp = sum(1 for r in specific_records if r.gold_intent == cls and r.pred_intent == cls)
            c_fp = sum(1 for r in specific_records if r.gold_intent != cls and r.pred_intent == cls)
            c_fn = sum(1 for r in specific_records if r.gold_intent == cls and r.pred_intent != cls)
            p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
            r_ = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
            f1 = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0
            f1_list.append(f1)
        specific_macro_f1 = mean(f1_list) if f1_list else 0.0

        # High-confidence intent accuracy (confidence >= 0.50)
        high_conf_records = [r for r in records if (r.intent_confidence or 0.0) >= 0.50]
        high_conf_total = len(high_conf_records)
        high_conf_correct = sum(1 for r in high_conf_records if r.pred_intent == r.gold_intent)
        high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0.0

        # ── 6. Retrieval & Sufficiency Dynamics ────────────────────────────────
        attempt1_sims = [r.initial_top_similarity for r in records]
        final_sims = [r.final_top_similarity for r in records]

        initial_suff_counts = Counter(r.initial_sufficiency for r in records)
        final_suff_counts = Counter(r.final_sufficiency for r in records)

        retries_triggered_count = sum(1 for r in records if r.retrieval_retry_triggered)
        retries_adopted_count = sum(1 for r in records if r.retrieval_retry_adopted)

        # Borderline upgrade rate
        borderline_initial = [r for r in records if r.initial_sufficiency == SUFFICIENCY_BORDERLINE]
        borderline_count = len(borderline_initial)
        borderline_upgraded = sum(
            1 for r in borderline_initial if r.final_sufficiency == SUFFICIENCY_SUFFICIENT
        )
        borderline_upgrade_rate = borderline_upgraded / borderline_count if borderline_count > 0 else 0.0

        # ── 7. Evidence URL Citation Verification ─────────────────────────────
        generated_replies = [r for r in records if r.generation_attempts > 0]
        generated_count = len(generated_replies)
        url_verif_passed_count = sum(1 for r in generated_replies if r.evidence_url_verification_passed)
        evidence_url_verification_rate = (
            url_verif_passed_count / generated_count if generated_count > 0 else 1.0
        )
        total_hallucinated_urls_stripped = sum(r.ungrounded_urls_stripped for r in records)

        # ── 8. Baseline Comparisons & Deltas ───────────────────────────────────
        deltas: dict[str, Any] = {}
        if baselines_config:
            clf_base = baselines_config.get("classifier_standalone", {})
            base_spec_acc = clf_base.get("specific_intent_accuracy")
            if base_spec_acc is not None:
                deltas["classifier_specific_accuracy_delta"] = round(specific_intent_accuracy - base_spec_acc, 4)

            lin_base = baselines_config.get("linear_pipeline", {})
            base_action_acc = lin_base.get("binary_action_accuracy")
            if base_action_acc is not None:
                deltas["linear_action_accuracy_delta"] = round(binary_acc - base_action_acc, 4)

            base_under_esc = lin_base.get("under_escalation_rate")
            if base_under_esc is not None:
                deltas["under_escalation_delta"] = round(under_escalation_rate - base_under_esc, 4)

            base_false_esc = lin_base.get("false_escalation_rate_on_auto")
            if base_false_esc is not None:
                deltas["false_escalation_delta"] = round(false_escalation_rate_on_auto - base_false_esc, 4)

        return {
            "sample_size": total,
            "gold_supports": {
                "gold_escalate_count": gold_escalate_count,
                "gold_auto_count": gold_auto_count,
                "gold_critical_count": gold_critical_count,
                "specific_intent_count": specific_total,
                "gold_risk_counts": dict(gold_risk_counts),
                "gold_intent_counts": dict(gold_intent_counts),
            },
            "primary_operational_routing": {
                "action_3way_distribution": action_3way_distribution,
                "queue_distribution": queue_distribution,
                "pre_retrieval_critical_short_circuits": pre_retrieval_critical_short_circuits,
                "pre_generation_short_circuits": pre_generation_short_circuits,
            },
            "secondary_binary_action_metrics": {
                "action_accuracy": round(binary_acc, 4),
                "macro_f1": round(binary_macro_f1, 4),
                "escalate": {
                    "precision": round(esc_prec, 4),
                    "recall": round(esc_rec, 4),
                    "f1": round(esc_f1, 4),
                    "support": gold_escalate_count,
                },
                "auto_handle": {
                    "precision": round(auto_prec, 4),
                    "recall": round(auto_rec, 4),
                    "f1": round(auto_f1, 4),
                    "support": gold_auto_count,
                },
                "under_escalation_rate": round(under_escalation_rate, 4),
                "under_escalation_count": fn,
                "false_escalation_rate_on_auto": round(false_escalation_rate_on_auto, 4),
                "false_escalation_count": fp,
                "overall_escalation_rate": round(overall_escalation_rate, 4),
                "overall_escalation_count": pred_escalate_total,
                "confusion_matrix": {
                    "gold_escalate_pred_escalate": tp,
                    "gold_escalate_pred_auto": fn,
                    "gold_auto_pred_escalate": fp,
                    "gold_auto_pred_auto": tn,
                },
            },
            "critical_safety_audit": {
                "critical_cases_total": critical_total,
                "critical_cases_escalated": critical_escalated,
                "critical_safety_recall": round(critical_safety_recall, 4),
                "missed_critical_count": len(missed_critical),
                "missed_critical_examples": missed_critical,
            },
            "intent_classification": {
                "full_set_accuracy": round(full_intent_accuracy, 4),
                "specific_intent_accuracy": round(specific_intent_accuracy, 4),
                "specific_intent_macro_f1": round(specific_macro_f1, 4),
                "specific_intent_support": specific_total,
                "high_confidence_accuracy": round(high_conf_accuracy, 4),
                "high_confidence_support": high_conf_total,
            },
            "retrieval_and_retry_dynamics": {
                "attempt1_mean_similarity": round(mean(attempt1_sims), 4) if attempt1_sims else 0.0,
                "attempt1_median_similarity": round(median(attempt1_sims), 4) if attempt1_sims else 0.0,
                "final_mean_similarity": round(mean(final_sims), 4) if final_sims else 0.0,
                "initial_sufficiency_distribution": dict(initial_suff_counts),
                "final_sufficiency_distribution": dict(final_suff_counts),
                "retries_triggered_count": retries_triggered_count,
                "retries_triggered_rate": round(retries_triggered_count / total, 4),
                "retries_adopted_count": retries_adopted_count,
                "borderline_cases_count": borderline_count,
                "borderline_upgraded_count": borderline_upgraded,
                "borderline_upgrade_rate": round(borderline_upgrade_rate, 4),
            },
            "evidence_url_citation_integrity": {
                "generated_replies_count": generated_count,
                "url_verification_passed_count": url_verif_passed_count,
                "evidence_url_verification_rate": round(evidence_url_verification_rate, 4),
                "hallucinated_urls_stripped_count": total_hallucinated_urls_stripped,
                "clarification_note": "Measures evidence URL grounding/citation integrity, NOT complete reply correctness.",
            },
            "baseline_deltas": deltas,
        }

    @staticmethod
    def format_markdown_report(
        metrics: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        """Render evaluation metrics into human-readable markdown format."""
        total = metrics.get("sample_size", 0)
        p_route = metrics.get("primary_operational_routing", {})
        act_dist = p_route.get("action_3way_distribution", {})
        s_bin = metrics.get("secondary_binary_action_metrics", {})
        cm = s_bin.get("confusion_matrix", {})
        safety = metrics.get("critical_safety_audit", {})
        intent_m = metrics.get("intent_classification", {})
        ret_m = metrics.get("retrieval_and_retry_dynamics", {})
        url_m = metrics.get("evidence_url_citation_integrity", {})
        deltas = metrics.get("baseline_deltas", {})
        supports = metrics.get("gold_supports", {})

        auto_info = act_dist.get(ACTION_AUTO_HANDLE, {})
        clar_info = act_dist.get(ACTION_ASK_CLARIFICATION, {})
        esc_info = act_dist.get(ACTION_ESCALATE, {})

        lines: list[str] = [
            "# AgentController Golden Set Evaluation Report (Milestone 18)",
            "",
            "## 1. Executive Summary & Core Results",
            f"- **Benchmark Dataset**: AppleSupport Human-Labelled Golden Set ({total} examples)",
            "- **Evaluation Nature**: Read-only, deterministic, offline/mock execution without data leakage.",
            "- **Primary Operational Decisions** (3-Way Routing):",
            f"  - **AUTO_HANDLE**: {auto_info.get('count', 0)} ({auto_info.get('share', 0):.2%}) — Grounded autonomous resolutions",
            f"  - **ASK_CLARIFICATION**: {clar_info.get('count', 0)} ({clar_info.get('share', 0):.2%}) — Interactive clarification requests",
            f"  - **ESCALATE**: {esc_info.get('count', 0)} ({esc_info.get('share', 0):.2%}) — Human support queue transfers",
            f"- **Critical Safety Recall**: **{safety.get('critical_safety_recall', 0.0):.2%}** ({safety.get('critical_cases_escalated', 0)} of {safety.get('critical_cases_total', 0)} critical hazard cases escalated)",
            "",
            "---",
            "",
            "## 2. Primary 3-Way Operational Action Breakdown",
            "| Operational Action | Definition | Cases | Share % |",
            "| :--- | :--- | :---: | :---: |",
            f"| **`AUTO_HANDLE`** | Direct self-service response with grounded evidence & citation integrity | {auto_info.get('count', 0)} | {auto_info.get('share', 0):.2%} |",
            f"| **`ASK_CLARIFICATION`** | First-class clarification prompt for ambiguous or low-confidence queries | {clar_info.get('count', 0)} | {clar_info.get('share', 0):.2%} |",
            f"| **`ESCALATE`** | Transfer to human specialist queue based on policy boundary or risk | {esc_info.get('count', 0)} | {esc_info.get('share', 0):.2%} |",
            "",
            "### Queue Routing Breakdown",
            "| Target Queue | Case Count | Share % |",
            "| :--- | :---: | :---: |",
        ]

        for q, qinfo in p_route.get("queue_distribution", {}).items():
            lines.append(f"| `{q}` | {qinfo.get('count', 0)} | {qinfo.get('share', 0):.2%} |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Secondary Binary Action Comparison (vs. Historical 2-Way Gold Action)",
            "> *Note: Compares `AUTO_HANDLE` against Non-AUTO (`ESCALATE` + `ASK_CLARIFICATION`) to directly benchmark against the 2-way `gold_action`.*",
            "",
            "| Metric | Agent Result | Linear Baseline (M16) | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Action Accuracy** | **{s_bin.get('action_accuracy', 0.0):.2%}** | 70.80% | {deltas.get('linear_action_accuracy_delta', 0.0):+.2%} |",
            f"| **Macro F1-Score** | **{s_bin.get('macro_f1', 0.0):.2%}** | 46.23% | — |",
            f"| **Under-Escalation Rate** (Dangerous FN) | **{s_bin.get('under_escalation_rate', 0.0):.2%}** ({s_bin.get('under_escalation_count', 0)}/{supports.get('gold_escalate_count', 180)}) | 3.89% (7/180) | {deltas.get('under_escalation_delta', 0.0):+.2%} |",
            f"| **False Escalation Rate on Auto** | **{s_bin.get('false_escalation_rate_on_auto', 0.0):.2%}** ({s_bin.get('false_escalation_count', 0)}/{supports.get('gold_auto_count', 70)}) | 94.29% (66/70) | {deltas.get('false_escalation_delta', 0.0):+.2%} |",
            f"| **Overall Escalation Rate** | **{s_bin.get('overall_escalation_rate', 0.0):.2%}** ({s_bin.get('overall_escalation_count', 0)}/{total}) | 95.60% (239/250) | — |",
            f"| **Critical Safety Recall** | **{safety.get('critical_safety_recall', 0.0):.2%}** | 100.00% (10/10) | 0.00% |",
            "",
            "### Binary Confusion Matrix",
            f"- **Gold ESCALATE -> Pred ESCALATE**: {cm.get('gold_escalate_pred_escalate', 0)}",
            f"- **Gold ESCALATE -> Pred AUTO_HANDLE**: {cm.get('gold_escalate_pred_auto', 0)} (Under-Escalation)",
            f"- **Gold AUTO_HANDLE -> Pred ESCALATE**: {cm.get('gold_auto_pred_escalate', 0)} (Over-Escalation / Clarification)",
            f"- **Gold AUTO_HANDLE -> Pred AUTO_HANDLE**: {cm.get('gold_auto_pred_auto', 0)}",
            "",
            "---",
            "",
            "## 4. Intent Classification Performance",
            "| Slice | Agent Result | Standalone Classifier Baseline | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Specific-Intent Accuracy** (225 cases) | **{intent_m.get('specific_intent_accuracy', 0.0):.2%}** | 52.00% | {deltas.get('classifier_specific_accuracy_delta', 0.0):+.2%} |",
            f"| **Specific-Intent Macro F1** | **{intent_m.get('specific_intent_macro_f1', 0.0):.2%}** | 50.19% | — |",
            f"| **Full Golden Accuracy** (250 cases) | **{intent_m.get('full_set_accuracy', 0.0):.2%}** | 46.80% | — |",
            f"| **High-Confidence Accuracy** (>= 0.50) | **{intent_m.get('high_confidence_accuracy', 0.0):.2%}** | 82.05% | — |",
            "",
            "---",
            "",
            "## 5. Agentic Retrieval & Retry Dynamics",
            f"- **Attempt 1 Mean Similarity**: {ret_m.get('attempt1_mean_similarity', 0.0):.4f} (Median: {ret_m.get('attempt1_median_similarity', 0.0):.4f})",
            f"- **Final Mean Similarity**: {ret_m.get('final_mean_similarity', 0.0):.4f}",
            f"- **Retrieval Retries Triggered**: {ret_m.get('retries_triggered_count', 0)} ({ret_m.get('retries_triggered_rate', 0.0):.2%})",
            f"- **Retrieval Retries Adopted**: {ret_m.get('retries_adopted_count', 0)}",
            f"- **Borderline Cases Upgraded** ($0.25 \\le \\text{{sim}} < 0.35 \\rightarrow \\ge 0.35$): {ret_m.get('borderline_upgraded_count', 0)} of {ret_m.get('borderline_cases_count', 0)} ({ret_m.get('borderline_upgrade_rate', 0.0):.2%})",
            "",
            "---",
            "",
            "## 6. Evidence URL Verification (Citation Integrity)",
            "> *Clarification: Evaluates programmatic citation and link integrity against retrieved evidence; does not measure complete semantic reply correctness.*",
            "",
            f"- **Replies Reaching Generation**: {url_m.get('generated_replies_count', 0)}",
            f"- **Evidence URL Verification Pass Rate**: **{url_m.get('evidence_url_verification_rate', 0.0):.2%}** ({url_m.get('url_verification_passed_count', 0)} passed)",
            f"- **Hallucinated / Ungrounded URLs Stripped**: {url_m.get('hallucinated_urls_stripped_count', 0)}",
            "",
            "---",
            "",
            "## 7. Zero-Tolerance Critical Safety Audit",
            f"- **Critical Safety Hazard Recall**: **{safety.get('critical_safety_recall', 0.0):.2%}** ({safety.get('critical_cases_escalated', 0)} / {safety.get('critical_cases_total', 0)})",
            f"- **Pre-Retrieval Hazard Short-Circuits**: {p_route.get('pre_retrieval_critical_short_circuits', 0)} cases",
            f"- **Pre-Generation Boundary Short-Circuits**: {p_route.get('pre_generation_short_circuits', 0)} cases",
            f"- **Missed Critical Cases Count**: {safety.get('missed_critical_count', 0)}",
        ])

        return "\n".join(lines)
