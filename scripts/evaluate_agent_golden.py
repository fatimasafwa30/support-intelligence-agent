"""CLI Runner for AgentController Golden Set Evaluation (Milestone 18).

Executes the complete production support agent pipeline against the 250-example Golden Set
in a read-only, deterministic offline mode without data leakage.

Usage:
    .venv/Scripts/python.exe scripts/evaluate_agent_golden.py --help
    .venv/Scripts/python.exe scripts/evaluate_agent_golden.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import yaml

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.controller import AgentController
from src.evaluation.agent_evaluator import AgentEvaluationRecord, AgentEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "agent_evaluation.yaml"
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data" / "golden" / "golden_annotation.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "agent_golden_evaluation.md"
DEFAULT_CSV_PATH = PROJECT_ROOT / "reports" / "agent_misclassifications.csv"
DEFAULT_JSON_PATH = PROJECT_ROOT / "reports" / "agent_evaluation_summary.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run deterministic Golden Set evaluation for the frozen AgentController.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to agent evaluation configuration YAML.",
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to 250-example Golden Set annotations CSV.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Destination path for markdown evaluation report.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Destination path for misclassifications CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Destination path for machine-readable summary JSON.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional subset size for testing/dry-run execution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging per example.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration YAML if present."""
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> int:
    """Execute evaluation workflow."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    evaluator = AgentEvaluator()

    logger.info("Loading Golden Set from %s", args.golden_path)
    golden_records = evaluator.load_golden_set(args.golden_path)
    logger.info("Loaded %d Golden Set records.", len(golden_records))

    if args.sample_size is not None and args.sample_size > 0:
        golden_records = golden_records[: args.sample_size]
        logger.info("Truncated evaluation to first %d records as requested.", len(golden_records))

    logger.info("Initializing AgentController (offline deterministic mode)...")
    controller = AgentController()

    logger.info("Executing evaluation pipeline over %d records...", len(golden_records))
    start_eval_time = time.perf_counter()

    eval_records: list[AgentEvaluationRecord] = []
    total = len(golden_records)
    for i, rec in enumerate(golden_records, 1):
        r = evaluator.evaluate_record(rec, controller)
        eval_records.append(r)
        if i % 25 == 0 or i == total:
            logger.info("Processed %d / %d examples (%.1f%%)...", i, total, (i / total) * 100)

    total_eval_seconds = time.perf_counter() - start_eval_time
    logger.info("Inference completed in %.2f seconds (%.2f ms/query).", total_eval_seconds, (total_eval_seconds / total) * 1000)

    logger.info("Computing operational and safety metrics...")
    baselines_cfg = config.get("baselines", {})
    metrics = evaluator.compute_metrics(eval_records, baselines_config=baselines_cfg)

    # Format and save markdown report
    report_md = evaluator.format_markdown_report(metrics, config=config)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Saved Markdown report to %s", args.output_report)

    # Save summary JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved summary JSON to %s", args.output_json)

    # Save misclassifications CSV (filtered to binary mismatches or missed criticals)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    misclassifications = [
        r.to_csv_row()
        for r in eval_records
        if r.effective_binary_action != r.gold_action or (r.gold_risk == "critical" and r.effective_binary_action != "ESCALATE")
    ]
    if misclassifications:
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(misclassifications[0].keys()))
            writer.writeheader()
            writer.writerows(misclassifications)
        logger.info("Saved %d misclassifications to %s", len(misclassifications), args.output_csv)
    else:
        logger.info("Zero misclassifications found; CSV not written.")

    # Emit console summary
    p_route = metrics.get("primary_operational_routing", {})
    act_dist = p_route.get("action_3way_distribution", {})
    s_bin = metrics.get("secondary_binary_action_metrics", {})
    safety = metrics.get("critical_safety_audit", {})
    intent_m = metrics.get("intent_classification", {})

    print("\n" + "=" * 78)
    print("AGENT CONTROLLER GOLDEN EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Total Evaluated Records:       {metrics.get('sample_size', 0)}")
    print("Primary 3-Way Action Routing:")
    for act, info in act_dist.items():
        print(f"  - {act:<18}: {info.get('count', 0):>4} ({info.get('share', 0):>6.2%})")
    print(f"Secondary Binary Accuracy:     {s_bin.get('action_accuracy', 0):.2%}")
    print(f"Under-Escalation Rate:         {s_bin.get('under_escalation_rate', 0):.2%} ({s_bin.get('under_escalation_count', 0)} cases)")
    print(f"False Escalation Rate on Auto: {s_bin.get('false_escalation_rate_on_auto', 0):.2%} ({s_bin.get('false_escalation_count', 0)} cases)")
    print(f"Critical Safety Recall:        {safety.get('critical_safety_recall', 0):.2%} ({safety.get('critical_cases_escalated', 0)} / {safety.get('critical_cases_total', 0)})")
    print(f"Specific-Intent Accuracy:      {intent_m.get('specific_intent_accuracy', 0):.2%} (vs baseline 52.00%)")
    print("=" * 78 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
