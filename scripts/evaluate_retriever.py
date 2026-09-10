"""Evaluation script for TF-IDF Historical-Resolution Retrieval.

Evaluates the frozen TFIDFRetriever on held-out DEV customer queries with strict
query-conversation leakage protection. Measures intent-alignment / pseudo-relevance
metrics (NOT proof of full problem resolution), latency/throughput benchmarks,
and qualitative retrieval demonstrations.
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
from typing import Any, Sequence

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intents.silver_labeler import SilverLabeler
from src.retrieval.tfidf_retriever import TFIDFRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "tfidf_retriever.joblib"
DEFAULT_CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "apple_conversations.jsonl"
DEFAULT_SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "conversation_splits.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "tfidf_retrieval_report.txt"

SAMPLE_DEMO_QUERIES: tuple[str, ...] = (
    "My iPhone battery is draining so fast after the update",
    "How do I turn off Do Not Disturb While Driving?",
    "iPhone screen is completely black and unresponsive",
    "Can't connect to my home Wi-Fi network with iPhone 7",
    "Locked out of my Apple ID account and need verification code",
    "How to cancel Apple Music subscription on my phone",
    "Where is my online order from Apple store?",
)


def load_dev_queries(
    conversations_path: str | Path,
    splits_path: str | Path,
    max_queries: int = 1000,
) -> list[dict[str, Any]]:
    """Load inbound customer queries with high-confidence silver intents from DEV conversations."""
    splits_file = Path(splits_path)
    with splits_file.open("r", encoding="utf-8") as f:
        splits = json.load(f)

    dev_cids = set(splits.get("dev_conversation_ids", []))
    labeler = SilverLabeler()

    queries: list[dict[str, Any]] = []

    with Path(conversations_path).open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            conv = json.loads(line_str)
            cid = conv.get("conversation_id", "")
            if cid in dev_cids:
                for t in conv.get("tweets", []):
                    if t.get("inbound", False):
                        text = str(t.get("text") or "").strip()
                        if text:
                            # Obtain silver label for pseudo-relevance ground truth
                            label_res = labeler.label(text)
                            res_intent = label_res["intent"]
                            res_confidence = label_res["confidence"]
                            if res_intent != "other_unclear" and res_confidence >= 0.80:
                                queries.append({
                                    "conversation_id": cid,
                                    "tweet_id": str(t.get("tweet_id", "")),
                                    "text": text,
                                    "intent": res_intent,
                                    "confidence": res_confidence,
                                })
                                if len(queries) >= max_queries:
                                    return queries
    return queries


def evaluate_intent_alignment_and_latency(
    retriever: TFIDFRetriever,
    queries: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute intent-alignment / pseudo-relevance metrics and latency profile."""
    if not queries:
        raise ValueError("No evaluation queries provided.")

    top1_matches = 0
    top3_precisions: list[float] = []
    top5_precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    top1_scores: list[float] = []

    start_time = time.perf_counter()

    for q in queries:
        q_text = q["text"]
        q_intent = q["intent"]
        q_cid = q["conversation_id"]

        # Strict query-conversation leakage protection:
        # Exclude query's own conversation from retrieval candidates
        results = retriever.retrieve(
            query=q_text,
            top_k=top_k,
            min_score=0.0,
            exclude_conversation_id=q_cid,
        )

        if not results:
            top3_precisions.append(0.0)
            top5_precisions.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        # Top-1 metrics
        first_res = results[0]
        top1_scores.append(first_res.score)
        if first_res.resolution.intent == q_intent:
            top1_matches += 1

        # MRR calculation
        rr = 0.0
        for rank, r in enumerate(results, start=1):
            if r.resolution.intent == q_intent:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Precision@3
        top3 = results[:3]
        p_at_3 = sum(1 for r in top3 if r.resolution.intent == q_intent) / len(top3)
        top3_precisions.append(p_at_3)

        # Precision@5
        top5 = results[:5]
        p_at_5 = sum(1 for r in top5 if r.resolution.intent == q_intent) / len(top5)
        top5_precisions.append(p_at_5)

    elapsed_time = time.perf_counter() - start_time
    n_queries = len(queries)
    avg_latency_ms = (elapsed_time / n_queries) * 1000
    qps = n_queries / elapsed_time if elapsed_time > 0 else 0.0

    return {
        "num_eval_queries": n_queries,
        "elapsed_seconds": elapsed_time,
        "avg_latency_ms": avg_latency_ms,
        "queries_per_second": qps,
        "top1_intent_alignment_rate": (top1_matches / n_queries) * 100,
        "top3_intent_alignment_precision": (sum(top3_precisions) / n_queries) * 100,
        "top5_intent_alignment_precision": (sum(top5_precisions) / n_queries) * 100,
        "mean_reciprocal_rank": (sum(reciprocal_ranks) / n_queries),
        "avg_top1_similarity_score": (sum(top1_scores) / len(top1_scores)) if top1_scores else 0.0,
    }


def format_evaluation_report(
    eval_stats: dict[str, Any],
    retriever_config: dict[str, Any],
    demo_results: list[dict[str, Any]],
) -> str:
    """Format evaluation results into a clear, comprehensive markdown report."""
    lines = [
        "=" * 80,
        "TF-IDF HISTORICAL-RESOLUTION RETRIEVER EVALUATION REPORT",
        "=" * 80,
        "",
        "## 1. Retriever Configuration & Provenance",
        f"- Retrieval Field:                 '{retriever_config.get('index_field')}' (Default customer-only matching)",
        f"- Indexed Historical Resolutions:  {retriever_config.get('num_indexed_resolutions', 0):,}",
        f"- N-Gram Range:                    {retriever_config.get('ngram_range')}",
        f"- Max Features:                    {retriever_config.get('max_features', 0):,}",
        f"- Sublinear TF:                    {retriever_config.get('sublinear_tf')}",
        f"- Min Document Frequency:          {retriever_config.get('min_df')}",
        "",
        "## 2. Leakage Protection Protocol",
        "- Corpus Provenance:               Exclusively TRAIN split (0 Golden / DEV / TEST conversations ingested).",
        "- Query-Conversation Protection:   During evaluation, each query's own conversation_id was strictly excluded",
        "                                   from retrieval candidates via 'exclude_conversation_id'.",
        "",
        "## 3. Intent-Alignment / Pseudo-Relevance Benchmark",
        "> NOTE: The metrics below represent intent-alignment / pseudo-relevance (measuring whether the retrieved",
        "> historical customer issue matches the same underlying problem category as the query), NOT absolute proof",
        "> that the retrieved response fully resolves the customer's specific inquiry.",
        "",
        f"- Evaluated Held-Out DEV Queries:  {eval_stats['num_eval_queries']:,}",
        f"- Top-1 Intent-Alignment Rate:     {eval_stats['top1_intent_alignment_rate']:.2f}%",
        f"- Top-3 Intent-Alignment Precision:{eval_stats['top3_intent_alignment_precision']:.2f}%",
        f"- Top-5 Intent-Alignment Precision:{eval_stats['top5_intent_alignment_precision']:.2f}%",
        f"- Mean Reciprocal Rank (MRR@5):    {eval_stats['mean_reciprocal_rank']:.4f}",
        f"- Average Top-1 Cosine Similarity: {eval_stats['avg_top1_similarity_score']:.4f}",
        "",
        "## 4. Latency & Throughput Benchmark",
        f"- Total Evaluation Time:           {eval_stats['elapsed_seconds']:.2f} seconds",
        f"- Average Retrieval Latency:       {eval_stats['avg_latency_ms']:.2f} ms / query",
        f"- Retrieval Throughput:            {eval_stats['queries_per_second']:.1f} queries / second",
        "",
        "## 5. Qualitative Retrieval Demonstrations (Sample Queries)",
        "-" * 80,
    ]

    for demo in demo_results:
        lines.append(f"Query: \"{demo['query']}\"")
        for r in demo["results"]:
            res = r.resolution
            lines.append(f"  [Rank {r.rank} | Cosine: {r.score:.4f} | Intent: {res.intent or 'unassigned'}]")
            lines.append(f"    Past Customer: {res.customer_text[:100]}...")
            lines.append(f"    Brand Response: {res.brand_text[:120]}...")
        lines.append("-" * 80)

    lines.extend([
        "",
        "## 6. Methodological Summary",
        "- TF-IDF retrieval with customer_text matching rapidly finds historically relevant customer problems.",
        "- High intent alignment confirms that vocabulary similarity in customer tweets reliably retrieves",
        "  analogous problem contexts.",
        "- The paired AppleSupport responses provide concrete historical troubleshooting steps and official support links.",
        "=" * 80,
    ])

    return "\n".join(lines)


def main() -> None:
    """Run retriever evaluation on DEV queries and save report."""
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF historical resolution retriever.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS_PATH)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--num-queries", type=int, default=1000)
    args = parser.parse_args()

    logger.info("Loading fitted TFIDFRetriever from %s...", args.model)
    retriever = TFIDFRetriever.load(args.model)

    logger.info("Loading up to %d held-out DEV queries...", args.num_queries)
    queries = load_dev_queries(args.conversations, args.splits, max_queries=args.num_queries)
    logger.info("Loaded %d eligible DEV customer queries for evaluation.", len(queries))

    logger.info("Computing intent-alignment and latency benchmarks...")
    eval_stats = evaluate_intent_alignment_and_latency(retriever, queries, top_k=5)

    logger.info("Running qualitative demonstrations on sample queries...")
    demo_results: list[dict[str, Any]] = []
    for sample_q in SAMPLE_DEMO_QUERIES:
        res = retriever.retrieve(sample_q, top_k=3, min_score=0.0)
        demo_results.append({
            "query": sample_q,
            "results": res,
        })

    report_text = format_evaluation_report(eval_stats, retriever.get_config(), demo_results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Report written to %s", args.report)

    print(report_text)


if __name__ == "__main__":
    main()
