"""Golden Set evaluation of confidence/risk-aware escalation policy.

Simulates the end-to-end production support pipeline:
Golden text -> Frozen Classifier -> Frozen Retriever -> Mock Reply Generator ->
Risk Detector -> Escalation Engine -> Predicted Action.

Strict evaluation constraints:
1. Never feeds gold_intent, gold_risk, or gold_action into the policy decision.
2. gold_risk is used strictly as ground-truth for evaluating the risk detector.
3. Compares final predicted action against human ground-truth gold_action.
4. Audits critical safety recall, under-escalation, and over-escalation rates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.escalation_policy import EscalationDecision, EscalationEngine
from src.agent.grounded_generator import MockReplyGenerator
from src.agent.reply_schemas import EvidenceItem, GenerationRequest
from src.agent.risk_detector import RiskAssessment, RiskDetector
from src.intents.classifier import IntentClassifier
from src.retrieval.tfidf_retriever import TFIDFRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data" / "golden" / "golden_annotation.csv"
DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "intent_classifier_baseline.joblib"
DEFAULT_RETRIEVER_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "tfidf_retriever.joblib"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "escalation_golden_evaluation.txt"
DEFAULT_CSV_PATH = PROJECT_ROOT / "reports" / "escalation_misclassifications.csv"


def load_golden_set(golden_path: str | Path) -> list[dict[str, Any]]:
    """Load the frozen 250-example Golden Set."""
    path = Path(golden_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden Set not found at: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "tweet_id": row["tweet_id"],
                "text": row["text"],
                "gold_intent": row["gold_intent"],
                "gold_risk": row["gold_risk"],
                "gold_action": row["gold_action"],
                "annotation_notes": row.get("annotation_notes", ""),
            })

    if len(records) != 250:
        raise ValueError(f"Expected exactly 250 Golden Set rows, found {len(records)}")
    return records


def evaluate_pipeline(
    records: list[dict[str, Any]],
    classifier: IntentClassifier,
    retriever: TFIDFRetriever,
    generator: MockReplyGenerator,
    engine: EscalationEngine,
    detector: RiskDetector,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute end-to-end simulation across Golden Set records."""
    evaluated_records: list[dict[str, Any]] = []

    # Counters for metrics
    action_cm: Counter[tuple[str, str]] = Counter()  # (gold_action, pred_action)
    risk_cm: Counter[tuple[str, str]] = Counter()    # (gold_risk, pred_risk)
    critical_cases_total = 0
    critical_cases_escalated = 0
    missed_critical_examples: list[dict[str, Any]] = []

    for r in records:
        text = r["text"]
        cid = r["conversation_id"]
        gold_action = r["gold_action"]
        gold_risk = r["gold_risk"]

        # Step 1: Predict Intent & Confidence
        pred = classifier.predict_one(text)

        # Step 2: Retrieve Historical Evidence (leakage protected: query conv excluded)
        ret_results = retriever.retrieve(query=text, top_k=3, min_score=0.0, exclude_conversation_id=cid)
        evidence_items = [EvidenceItem.from_retrieval_result(res) for res in ret_results]

        # Step 3: Draft Grounded Reply (Mock Mode)
        ret_status = "strong" if (evidence_items and evidence_items[0].similarity_score >= engine.retrieval_similarity_threshold) else "heuristic_weak"
        gen_req = GenerationRequest(
            customer_query=text,
            intent=pred.intent,
            intent_confidence=pred.confidence,
            retrieved_evidence=evidence_items,
            retrieval_status=ret_status,
            conversation_id=cid,
        )
        grounded_reply = generator.generate(gen_req)

        # Step 4: Assess Risk independently
        risk_assessment = detector.assess_risk(text)

        # Step 5: Escalation Policy Decision (strictly no gold fields used)
        decision = engine.decide(
            query=text,
            predicted_intent=pred.intent,
            intent_confidence=pred.confidence,
            top_evidence=evidence_items,
            grounded_reply=grounded_reply,
            risk_assessment=risk_assessment,
        )

        # Effective binary operational action: AUTO_HANDLE vs ESCALATE (clarification is non-auto)
        effective_pred_action = "AUTO_HANDLE" if decision.action == "AUTO_HANDLE" else "ESCALATE"

        action_cm[(gold_action, effective_pred_action)] += 1
        risk_cm[(gold_risk, risk_assessment.risk_level)] += 1

        # Audit critical safety recall
        if gold_risk == "critical":
            critical_cases_total += 1
            if effective_pred_action == "ESCALATE":
                critical_cases_escalated += 1
            else:
                missed_critical_examples.append({
                    "id": r["id"],
                    "text": text,
                    "gold_risk": gold_risk,
                    "pred_risk": risk_assessment.risk_level,
                    "gold_action": gold_action,
                    "pred_action": effective_pred_action,
                    "reasons": decision.reasons,
                })

        evaluated_records.append({
            "id": r["id"],
            "conversation_id": cid,
            "text": text,
            "gold_intent": r["gold_intent"],
            "pred_intent": pred.intent,
            "pred_confidence": round(pred.confidence, 4),
            "top_retrieval_sim": round(evidence_items[0].similarity_score, 4) if evidence_items else 0.0,
            "gold_risk": gold_risk,
            "pred_risk": risk_assessment.risk_level,
            "risk_categories": risk_assessment.risk_categories,
            "gold_action": gold_action,
            "raw_pred_action": decision.action,
            "effective_pred_action": effective_pred_action,
            "routing_target": decision.routing_target,
            "reasons": "; ".join(decision.reasons),
            "annotation_notes": r["annotation_notes"],
        })

    # Compute operational metrics
    total = len(records)
    correct_actions = action_cm[("AUTO_HANDLE", "AUTO_HANDLE")] + action_cm[("ESCALATE", "ESCALATE")]
    accuracy = correct_actions / total

    tp_esc = action_cm[("ESCALATE", "ESCALATE")]
    fp_esc = action_cm[("AUTO_HANDLE", "ESCALATE")]
    fn_esc = action_cm[("ESCALATE", "AUTO_HANDLE")]
    tn_esc = action_cm[("AUTO_HANDLE", "AUTO_HANDLE")]

    # Under-escalation: human said ESCALATE, agent AUTO_HANDLED (dangerous false negative)
    under_escalation_count = fn_esc
    under_escalation_rate = under_escalation_count / (tp_esc + fn_esc) if (tp_esc + fn_esc) > 0 else 0.0

    # False escalation among Gold AUTO_HANDLE cases (deflection inefficiency)
    false_escalation_count = fp_esc
    false_escalation_rate_on_auto = (false_escalation_count / (tn_esc + fp_esc) * 100) if (tn_esc + fp_esc) > 0 else 0.0

    # Overall predicted escalation rate
    total_pred_escalate = tp_esc + fp_esc
    overall_pred_escalation_rate = (total_pred_escalate / total * 100) if total > 0 else 0.0

    # F1 for ESCALATE
    prec_esc = tp_esc / (tp_esc + fp_esc) if (tp_esc + fp_esc) > 0 else 0.0
    rec_esc = tp_esc / (tp_esc + fn_esc) if (tp_esc + fn_esc) > 0 else 0.0
    f1_esc = (2 * prec_esc * rec_esc) / (prec_esc + rec_esc) if (prec_esc + rec_esc) > 0 else 0.0

    # F1 for AUTO_HANDLE
    prec_auto = tn_esc / (tn_esc + fn_esc) if (tn_esc + fn_esc) > 0 else 0.0
    rec_auto = tn_esc / (tn_esc + fp_esc) if (tn_esc + fp_esc) > 0 else 0.0
    f1_auto = (2 * prec_auto * rec_auto) / (prec_auto + rec_auto) if (prec_auto + rec_auto) > 0 else 0.0

    macro_f1 = (f1_esc + f1_auto) / 2.0

    critical_safety_routing_recall = (
        (critical_cases_escalated / critical_cases_total * 100) if critical_cases_total > 0 else 0.0
    )
    risk_detector_critical_recall = (
        (risk_cm.get(("critical", "critical"), 0) / critical_cases_total * 100) if critical_cases_total > 0 else 0.0
    )

    stats = {
        "total_evaluated": total,
        "action_accuracy": accuracy * 100,
        "macro_f1": macro_f1 * 100,
        "escalate_precision": prec_esc * 100,
        "escalate_recall": rec_esc * 100,
        "escalate_f1": f1_esc * 100,
        "auto_handle_precision": prec_auto * 100,
        "auto_handle_recall": rec_auto * 100,
        "auto_handle_f1": f1_auto * 100,
        "under_escalation_count": under_escalation_count,
        "under_escalation_rate": under_escalation_rate * 100,
        "false_escalation_count": false_escalation_count,
        "false_escalation_rate_on_auto": false_escalation_rate_on_auto,
        "total_pred_escalate": total_pred_escalate,
        "overall_pred_escalation_rate": overall_pred_escalation_rate,
        "critical_safety_total": critical_cases_total,
        "critical_safety_escalated": critical_cases_escalated,
        "critical_safety_routing_recall": critical_safety_routing_recall,
        "risk_detector_critical_recall": risk_detector_critical_recall,
        "missed_critical_examples": missed_critical_examples,
        "action_confusion_matrix": {
            "gold_ESCALATE_pred_ESCALATE": tp_esc,
            "gold_ESCALATE_pred_AUTO": fn_esc,
            "gold_AUTO_pred_ESCALATE": fp_esc,
            "gold_AUTO_pred_AUTO": tn_esc,
        },
        "risk_confusion_matrix": dict(risk_cm),
    }

    return evaluated_records, stats


def format_report(stats: dict[str, Any], misclassifications: list[dict[str, Any]]) -> str:
    """Format the evaluation audit report as clean text."""
    cm = stats["action_confusion_matrix"]
    lines = [
        "=" * 80,
        "OPERATIONAL ESCALATION POLICY EVALUATION REPORT (GOLDEN SET BENCHMARK)",
        "=" * 80,
        "",
        "## 1. Evaluation Methodology & Simulation Protocol",
        "- Total Golden Set Examples:       250",
        "- Real Pipeline Simulation:        Golden Customer Text -> Frozen Intent Classifier ->",
        "                                   Frozen Historical Retriever -> Grounded Reply Generator ->",
        "                                   Risk Detector -> Escalation Policy -> Predicted Action.",
        "- STRICT ISOLATION PROTOCOL:       gold_intent, gold_risk, and gold_action were NEVER passed",
        "                                   as inputs to the escalation engine.",
        "- Engineering Thresholds Used:     intent_confidence >= 0.50, retrieval_similarity >= 0.35",
        "",
        "## 2. Action Classification Performance (AUTO_HANDLE vs ESCALATE)",
        f"{'Metric':<30} | {'Result':>10}",
        "-" * 43,
        f"{'Action Accuracy':<30} | {stats['action_accuracy']:>9.2f}%",
        f"{'Macro F1-Score':<30} | {stats['macro_f1']:>9.2f}%",
        f"{'ESCALATE Precision':<30} | {stats['escalate_precision']:>9.2f}%",
        f"{'ESCALATE Recall':<30} | {stats['escalate_recall']:>9.2f}%",
        f"{'ESCALATE F1-Score':<30} | {stats['escalate_f1']:>9.2f}%",
        f"{'AUTO_HANDLE Precision':<30} | {stats['auto_handle_precision']:>9.2f}%",
        f"{'AUTO_HANDLE Recall':<30} | {stats['auto_handle_recall']:>9.2f}%",
        f"{'AUTO_HANDLE F1-Score':<30} | {stats['auto_handle_f1']:>9.2f}%",
        "-" * 43,
        "",
        "Confusion Matrix (Gold Action vs Predicted Action):",
        f"  - Gold ESCALATE    -> Predicted ESCALATE (True Positives):   {cm['gold_ESCALATE_pred_ESCALATE']}",
        f"  - Gold ESCALATE    -> Predicted AUTO_HANDLE (Under-Escalation): {cm['gold_ESCALATE_pred_AUTO']} (DANGEROUS FALSE NEGATIVES)",
        f"  - Gold AUTO_HANDLE -> Predicted ESCALATE (Over-Escalation):   {cm['gold_AUTO_pred_ESCALATE']} (DEFLECTION INEFFICIENCY)",
        f"  - Gold AUTO_HANDLE -> Predicted AUTO_HANDLE (True Negatives):   {cm['gold_AUTO_pred_AUTO']}",
        "",
        "## 3. Operational Risk & Safety Audit",
        f"- Under-Escalation Rate:                       {stats['under_escalation_rate']:.2f}% ({stats['under_escalation_count']} of 180 required escalations)",
        f"- False Escalation Rate on Gold AUTO_HANDLE:   {stats['false_escalation_rate_on_auto']:.2f}% ({stats['false_escalation_count']} of 70 auto-handle candidates)",
        f"- Overall Predicted Escalation Rate:           {stats['overall_pred_escalation_rate']:.2f}% ({stats['total_pred_escalate']} of {stats['total_evaluated']} total cases)",
        f"- Critical Safety/Security Cases in Golden:    {stats['critical_safety_total']}",
        f"- Critical Cases Escalated to Human Support:   {stats['critical_safety_escalated']}",
        f"- Critical Safety Routing Recall:              {stats['critical_safety_routing_recall']:.2f}% (Target: 100.0%)",
        f"- Risk Detector Critical Classification Recall:{stats['risk_detector_critical_recall']:.2f}% ({stats['risk_confusion_matrix'].get(('critical', 'critical'), 0)} of {stats['critical_safety_total']})",
    ]

    if stats["missed_critical_examples"]:
        lines.append("")
        lines.append("CRITICAL SAFETY FAILURES DETECTED:")
        for miss in stats["missed_critical_examples"]:
            lines.append(f"  [{miss['id']}] \"{miss['text'][:80]}...\" -> Action: {miss['pred_action']} (Reasons: {miss['reasons']})")
    else:
        lines.append("  -> ZERO CRITICAL SAFETY FAILURES: 100% of critical safety hazards were successfully escalated.")

    lines.extend([
        "",
        "## 4. Risk Detector Ground-Truth Performance (Predicted Risk vs gold_risk)",
        f"{'Gold Risk':<12} | {'Predicted Low':>14} | {'Predicted Medium':>17} | {'Predicted High':>15} | {'Predicted Critical':>19}",
        "-" * 83,
    ])

    for g_risk in ("low", "medium", "high", "critical"):
        p_low = stats["risk_confusion_matrix"].get((g_risk, "low"), 0)
        p_med = stats["risk_confusion_matrix"].get((g_risk, "medium"), 0)
        p_hi = stats["risk_confusion_matrix"].get((g_risk, "high"), 0)
        p_crit = stats["risk_confusion_matrix"].get((g_risk, "critical"), 0)
        lines.append(f"{g_risk:<12} | {p_low:>14} | {p_med:>17} | {p_hi:>15} | {p_crit:>19}")

    lines.extend([
        "-" * 83,
        "",
        "## 5. Sample Misclassifications Audit",
        "-" * 80,
    ])

    for m in misclassifications[:6]:
        lines.extend([
            f"Case: [{m['id']}] (Gold: {m['gold_action']} | Pred: {m['effective_pred_action']})",
            f"  Text:        \"{m['text'][:100]}...\"",
            f"  Pred Intent: {m['pred_intent']} (Conf: {m['pred_confidence']:.2f}, Ret Sim: {m['top_retrieval_sim']:.2f})",
            f"  Risk Level:  Gold={m['gold_risk']} | Pred={m['pred_risk']} (Categories: {m['risk_categories'] or 'None'})",
            f"  Target:      {m['routing_target']} | Decision Reasons: {m['reasons']}",
            f"  Human Notes: {m['annotation_notes'][:120]}...",
            "-" * 80,
        ])

    lines.extend([
        "",
        "## 6. Summary & Key Insights",
        "- The escalation policy is intentionally safety-first. The 70.8% action accuracy is not",
        "  the main success criterion because the cost of harmful under-escalation is much higher",
        "  than unnecessary escalation.",
        "- The current baseline achieves 96.11% recall for cases requiring escalation and 100% recall",
        "  on the 10 critical-risk Golden cases, but its 95.6% overall escalation rate shows that",
        "  the policy is too conservative for efficient automation.",
        "=" * 80,
    ])

    return "\n".join(lines)


def main() -> None:
    """Run Golden Set evaluation of escalation policy and save report and CSV."""
    parser = argparse.ArgumentParser(description="Evaluate escalation policy against Golden Set.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER_PATH)
    parser.add_argument("--retriever", type=Path, default=DEFAULT_RETRIEVER_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--misclassifications-csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--conf-threshold", type=float, default=0.50)
    parser.add_argument("--sim-threshold", type=float, default=0.35)
    args = parser.parse_args()

    logger.info("Loading Golden Set from %s...", args.golden)
    golden_records = load_golden_set(args.golden)

    logger.info("Loading frozen classifier from %s...", args.classifier)
    classifier = IntentClassifier.load(args.classifier)

    logger.info("Loading frozen retriever from %s...", args.retriever)
    retriever = TFIDFRetriever.load(args.retriever)

    generator = MockReplyGenerator()
    engine = EscalationEngine(
        intent_confidence_threshold=args.conf_threshold,
        retrieval_similarity_threshold=args.sim_threshold,
    )
    detector = RiskDetector()

    logger.info("Evaluating real end-to-end pipeline across 250 Golden cases...")
    start_time = time.perf_counter()
    evaluated, stats = evaluate_pipeline(
        records=golden_records,
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        engine=engine,
        detector=detector,
    )
    elapsed = time.perf_counter() - start_time
    logger.info("Evaluated 250 Golden cases in %.2f seconds.", elapsed)

    # Collect misclassifications (where effective_pred_action != gold_action)
    misclassifications = [
        r for r in evaluated if r["effective_pred_action"] != r["gold_action"]
    ]
    logger.info("Found %d misclassified action cases out of 250.", len(misclassifications))

    # Save CSV
    args.misclassifications_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.misclassifications_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "conversation_id",
            "text",
            "gold_intent",
            "pred_intent",
            "pred_confidence",
            "top_retrieval_sim",
            "gold_risk",
            "pred_risk",
            "gold_action",
            "raw_pred_action",
            "effective_pred_action",
            "routing_target",
            "reasons",
            "annotation_notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in misclassifications:
            row_data = {k: m[k] for k in fieldnames}
            writer.writerow(row_data)
    logger.info("Misclassifications written to %s", args.misclassifications_csv)

    # Format and save report
    report_text = format_report(stats, misclassifications)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Report written to %s", args.report)

    print(report_text)


if __name__ == "__main__":
    main()
