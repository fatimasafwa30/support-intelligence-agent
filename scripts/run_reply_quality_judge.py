"""CLI Runner for LLM-as-a-Judge Reply Quality Evaluation (Milestone 19.2).

Extracts the 135 generated replies from the deterministic AgentController Golden evaluation,
samples the stratified 50-example human audit subset, and evaluates reply quality across
the five dimensions (Groundedness, Correctness, Relevance, Helpfulness, Tone) using Gemini 2.5 Flash.

Usage:
    # Run full LLM judge with Gemini API
    .venv/Scripts/python.exe scripts/run_reply_quality_judge.py

    # Run in offline mock mode (for verification / testing)
    .venv/Scripts/python.exe scripts/run_reply_quality_judge.py --mock

    # Sample and export the 50-example human audit subset only
    .venv/Scripts/python.exe scripts/run_reply_quality_judge.py --select-human-sample-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

from dotenv import load_dotenv
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

# Load .env if present
load_dotenv(PROJECT_ROOT / ".env")

from src.agent.controller import AgentController
from src.evaluation.agent_evaluator import AgentEvaluator
from src.evaluation.llm_judge import (
    BaseJudgeClient,
    GeminiJudgeClient,
    HumanAgreementSampler,
    JudgeConfig,
    MockJudgeClient,
    ResumableJudgeRunner,
)
from src.evaluation.reply_quality_schema import EvaluationUnit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "reply_evaluation.yaml"
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data" / "golden" / "golden_annotation.csv"
DEFAULT_RECORDS_PATH = PROJECT_ROOT / "reports" / "reply_quality_judge_records.jsonl"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "reports" / "reply_quality_judge_results.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "reply_quality_judge_report.md"
DEFAULT_HUMAN_SAMPLE_PATH = PROJECT_ROOT / "reports" / "human_agreement_sample_50.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge reply-quality evaluation on generated customer support replies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to reply evaluation YAML configuration.",
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to human-annotated Golden Set CSV.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Judge provider ('gemini' or 'mock'). Overrides config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model identifier (e.g. 'gemini-2.5-flash'). Overrides config.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Execute offline using MockJudgeClient without calling external APIs.",
    )
    parser.add_argument(
        "--select-human-sample-only",
        action="store_true",
        help="Only extract generated units and save the 50-example human audit sample.",
    )
    parser.add_argument(
        "--output-records",
        type=Path,
        default=DEFAULT_RECORDS_PATH,
        help="Destination path for append-only JSONL judged records.",
    )
    parser.add_argument(
        "--output-results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Destination path for summary JSON results.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Destination path for markdown report.",
    )
    parser.add_argument(
        "--human-sample-path",
        type=Path,
        default=DEFAULT_HUMAN_SAMPLE_PATH,
        help="Destination path for the 50-example human audit subset JSON.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional limit on number of generated replies to evaluate.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Pause in seconds between Gemini API calls to respect rate limits.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def extract_generated_evaluation_units(
    golden_path: Path,
) -> tuple[list[EvaluationUnit], int, int]:
    """Deterministically run Golden evaluation to extract generated EvaluationUnits.

    Returns:
        (generated_units, total_golden_count, short_circuited_count)
    """
    evaluator = AgentEvaluator()
    golden_records = evaluator.load_golden_set(golden_path)
    total_golden = len(golden_records)

    logger.info("Initializing AgentController in offline mode to extract generated replies...")
    controller = AgentController()

    generated_units: list[EvaluationUnit] = []
    short_circuited_count = 0

    for rec in golden_records:
        # Run inference strictly without gold annotations
        state = controller.process_query(
            customer_message=str(rec["text"]),
            conversation_id=str(rec.get("conversation_id", "")),
        )

        if state.generated_reply and state.generated_reply.reply_text:
            ev_list = [e.to_dict() for e in state.retrieved_evidence]
            gen_rep = state.generated_reply.to_dict()
            verif_rep = {
                "reply_text": state.generated_reply.reply_text,
                "grounded": state.generated_reply.grounded,
                "verification_passed": state.verification_passed,
                "verification_details": state.verification_details,
            }

            unit = EvaluationUnit(
                example_id=str(rec["id"]),
                customer_query=str(rec["text"]),
                predicted_intent=str(state.predicted_intent or "unknown"),
                retrieved_evidence=ev_list,
                generated_reply=gen_rep,
                conversation_id=str(rec.get("conversation_id", "")),
                intent_confidence=state.intent_confidence,
                verified_reply=verif_rep,
            )
            generated_units.append(unit)
        else:
            short_circuited_count += 1

    return generated_units, total_golden, short_circuited_count


def main() -> int:
    """Execute LLM Judge workflow."""
    args = parse_args()
    config_dict = load_config(args.config)
    llm_cfg = config_dict.get("llm_judge", {})

    provider = args.provider or ("mock" if args.mock else llm_cfg.get("provider", "gemini"))
    model = args.model or llm_cfg.get("model", "gemini-2.5-flash")
    temperature = float(llm_cfg.get("temperature", 0.0))
    max_tokens = int(llm_cfg.get("max_output_tokens", 1024))
    retries = int(llm_cfg.get("max_retries", 5))
    delay = float(args.request_delay if args.request_delay is not None else llm_cfg.get("request_delay_seconds", 1.0))

    judge_config = JudgeConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        max_retries=retries,
        request_delay_seconds=delay,
        output_records_path=args.output_records,
        output_results_path=args.output_results,
        output_report_path=args.output_report,
        human_sample_path=args.human_sample_path,
    )

    logger.info("Extracting generated evaluation units from %s...", args.golden_path)
    generated_units, total_golden, short_circuits = extract_generated_evaluation_units(args.golden_path)
    coverage = len(generated_units) / total_golden if total_golden > 0 else 0.0

    logger.info(
        "Extraction complete: %d generated replies / %d Golden examples (%.1f%% coverage). %d short-circuits excluded.",
        len(generated_units),
        total_golden,
        coverage * 100,
        short_circuits,
    )

    # 1. Deterministically sample and export the 50-example human audit subset
    logger.info("Sampling 50-example stratified human audit subset...")
    human_sample = HumanAgreementSampler.sample_human_subset(
        generated_units=generated_units,
        sample_size=50,
        random_seed=42,
    )
    args.human_sample_path.parent.mkdir(parents=True, exist_ok=True)
    with args.human_sample_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_size": len(human_sample),
                "total_generated_population": len(generated_units),
                "sampling_strategy": "Stratified across predicted_intent, confidence tiers, and routing actions",
                "random_seed": 42,
                "note": "Selected independently of and prior to LLM judge scores to prevent selection bias.",
                "samples": human_sample,
            },
            f,
            indent=2,
        )
    logger.info("Saved 50-example human audit subset to %s", args.human_sample_path)

    if args.select_human_sample_only:
        logger.info("Human sample selection complete (--select-human-sample-only flag set). Exiting.")
        return 0

    # Truncate sample if requested
    eval_units = generated_units
    if args.sample_size is not None and args.sample_size > 0:
        eval_units = eval_units[: args.sample_size]
        logger.info("Truncating evaluation to first %d units.", len(eval_units))

    # Initialize client
    client: BaseJudgeClient
    if provider == "mock" or args.mock:
        logger.info("Using MockJudgeClient (offline deterministic mode)...")
        client = MockJudgeClient()
    else:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.error(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set GEMINI_API_KEY in your environment or in a local .env file. "
                "Alternatively, run with --mock for offline mock verification."
            )
            return 1
        logger.info("Initializing GeminiJudgeClient with model '%s'...", model)
        client = GeminiJudgeClient(api_key=api_key, config=judge_config)

    # Run resumable evaluation
    runner = ResumableJudgeRunner(client=client, config=judge_config)
    logger.info("Starting reply quality evaluation over %d units...", len(eval_units))
    metrics = runner.run_evaluation(eval_units, total_golden_size=total_golden)

    # Print summary
    pop = metrics["evaluation_population"]
    dims = metrics["dimension_scores"]
    overall = metrics["overall_quality"]

    print("\n" + "=" * 78)
    print("LLM-AS-A-JUDGE REPLY QUALITY EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Total Golden Benchmark:      {pop['total_golden_examples']}")
    print(f"Generated Replies Scored:    {pop['generated_replies_evaluated']} ({pop['reply_generation_coverage']:.1%} coverage)")
    print(f"Non-Generated Short-Circuits:{pop['non_generated_short_circuits']} (excluded from reply scoring)")
    print(f"Successfully Judged Cases:   {pop['successfully_judged_count']}")
    print(f"Failed Cases:                {pop['failed_judging_count']}")
    print(f"Composite Mean Quality:      {overall['composite_mean_score']:.2f} / 5.00")
    print("\nPer-Dimension Mean Scores (1-5):")
    for dim, info in dims.items():
        print(f"  - {dim.capitalize():<14}: {info['mean']:.2f} (Median: {info['median']})")
    print("=" * 78 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
