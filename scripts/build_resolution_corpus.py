"""Historical resolution corpus builder and TF-IDF indexer for TRAIN split.

Extracts customer-to-brand resolution pairs exclusively from the TRAIN split of
AppleSupport conversations, computes verified corpus statistics, exports
data/processed/historical_resolutions.jsonl, fits the default customer_text TFIDFRetriever,
and saves the artifact to data/processed/models/tfidf_retriever.joblib.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
from pathlib import Path
import re
from statistics import mean, median
import sys
from typing import Any, Sequence

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.historical_resolution import HistoricalResolution
from src.retrieval.tfidf_retriever import TFIDFRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "apple_conversations.jsonl"
DEFAULT_SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "conversation_splits.json"
DEFAULT_SILVER_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "silver_train.jsonl"
DEFAULT_GOLDEN_ANNOTATION_PATH = PROJECT_ROOT / "data" / "golden" / "golden_annotation.csv"
DEFAULT_GOLDEN_CANDIDATES_PATH = PROJECT_ROOT / "data" / "golden" / "golden_candidates.csv"
DEFAULT_OUTPUT_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "historical_resolutions.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "historical_resolutions_corpus_report.txt"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "tfidf_retriever.joblib"


def load_forbidden_golden_ids(
    golden_annotation_path: str | Path,
    golden_candidates_path: str | Path | None = None,
) -> set[str]:
    """Load conversation IDs from Golden Set files without reading any labels."""
    forbidden_ids: set[str] = set()

    for p in (golden_annotation_path, golden_candidates_path):
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "conversation_id" not in (reader.fieldnames or []):
                raise ValueError(f"Missing 'conversation_id' column in {path}")
            for row in reader:
                cid = (row.get("conversation_id") or "").strip()
                if cid:
                    forbidden_ids.add(cid)

    return forbidden_ids


def load_verified_train_conversation_ids(
    splits_manifest_path: str | Path,
    forbidden_golden_ids: set[str],
) -> set[str]:
    """Load train conversation IDs from split manifest and verify zero leakage."""
    path = Path(splits_manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Splits manifest not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        splits_data = json.load(f)

    train_ids = set(splits_data.get("train_conversation_ids", []))
    dev_ids = set(splits_data.get("dev_conversation_ids", []))
    test_ids = set(splits_data.get("test_conversation_ids", []))
    golden_excluded = set(splits_data.get("golden_excluded_conversation_ids", []))

    if not train_ids:
        raise ValueError("No train_conversation_ids found in splits manifest.")

    # Strict isolation checks
    overlap_golden_manifest = train_ids & golden_excluded
    if overlap_golden_manifest:
        raise ValueError(
            f"CRITICAL: Found {len(overlap_golden_manifest)} train conversations overlapping with golden_excluded!"
        )

    overlap_golden_files = train_ids & forbidden_golden_ids
    if overlap_golden_files:
        raise ValueError(
            f"CRITICAL: Found {len(overlap_golden_files)} train conversations overlapping with Golden Set files!"
        )

    overlap_dev = train_ids & dev_ids
    if overlap_dev:
        raise ValueError(f"CRITICAL: Found {len(overlap_dev)} train conversations overlapping with dev!")

    overlap_test = train_ids & test_ids
    if overlap_test:
        raise ValueError(f"CRITICAL: Found {len(overlap_test)} train conversations overlapping with test!")

    logger.info(
        "Verified %d TRAIN conversations (0 overlap with %d Golden, %d DEV, %d TEST)",
        len(train_ids),
        len(golden_excluded),
        len(dev_ids),
        len(test_ids),
    )
    return train_ids


def load_silver_intents_lookup(silver_train_path: str | Path) -> dict[str, str]:
    """Load mapping of customer tweet_id -> silver_intent from silver_train.jsonl."""
    path = Path(silver_train_path)
    if not path.exists():
        logger.warning("Silver train dataset not found at %s; proceeding without silver intent tags.", path)
        return {}

    silver_lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            tid = str(record.get("tweet_id", ""))
            intent = record.get("silver_intent")
            if tid and intent:
                silver_lookup[tid] = str(intent)

    logger.info("Loaded %d high-confidence silver intent tags from %s", len(silver_lookup), path)
    return silver_lookup


def extract_pairs_from_conversation(
    conversation: dict[str, Any],
    silver_lookup: dict[str, str],
) -> list[HistoricalResolution]:
    """Extract customer -> brand resolution pairs from a single conversation."""
    cid = conversation.get("conversation_id", "")
    tweets = conversation.get("tweets", [])
    if not cid or not tweets:
        return []

    tweet_map: dict[str, dict[str, Any]] = {
        str(t.get("tweet_id")): t for t in tweets if t.get("tweet_id") is not None
    }

    pairs: list[HistoricalResolution] = []

    # Sort tweets chronologically if needed, or scan directly
    for t in tweets:
        # Outbound brand response
        if not t.get("inbound", False):
            resp_to_id = t.get("in_response_to_tweet_id")
            if resp_to_id and str(resp_to_id) in tweet_map:
                parent = tweet_map[str(resp_to_id)]
                # Ensure parent is inbound customer tweet
                if parent.get("inbound", True):
                    cust_tid = str(parent.get("tweet_id", ""))
                    brand_tid = str(t.get("tweet_id", ""))
                    cust_text = str(parent.get("text") or "")
                    brand_text = str(t.get("text") or "")
                    res_id = f"res_{cid}_{cust_tid}_{brand_tid}"
                    intent = silver_lookup.get(cust_tid)
                    created_at = t.get("created_at") or parent.get("created_at")

                    pairs.append(
                        HistoricalResolution(
                            resolution_id=res_id,
                            conversation_id=cid,
                            customer_tweet_id=cust_tid,
                            brand_tweet_id=brand_tid,
                            customer_text=cust_text,
                            brand_text=brand_text,
                            intent=intent,
                            created_at=str(created_at) if created_at else None,
                        )
                    )

    return pairs


def build_corpus_and_statistics(
    conversations_path: str | Path,
    train_ids: set[str],
    silver_lookup: dict[str, str],
) -> tuple[list[HistoricalResolution], dict[str, Any]]:
    """Scan TRAIN conversations and extract resolution pairs and detailed corpus stats."""
    path = Path(conversations_path)
    if not path.exists():
        raise FileNotFoundError(f"Conversations file not found: {path}")

    all_resolutions: list[HistoricalResolution] = []
    scanned_train_convs = 0

    logger.info("Scanning %s for TRAIN resolution pairs...", path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            conv = json.loads(line_str)
            cid = conv.get("conversation_id", "")
            if cid in train_ids:
                scanned_train_convs += 1
                pairs = extract_pairs_from_conversation(conv, silver_lookup)
                all_resolutions.extend(pairs)

    logger.info(
        "Extracted %d resolution pairs from %d TRAIN conversations",
        len(all_resolutions),
        scanned_train_convs,
    )

    # Compute detailed empirical statistics
    unique_cust_tweets = len({r.customer_tweet_id for r in all_resolutions})
    unique_brand_responses = len({r.brand_tweet_id for r in all_resolutions})
    pairs_with_silver = sum(1 for r in all_resolutions if r.intent is not None)
    pairs_without_silver = len(all_resolutions) - pairs_with_silver

    # Duplicate pairs: same normalized customer text and brand text
    pair_signatures: Counter[tuple[str, str]] = Counter()
    for r in all_resolutions:
        pair_signatures[(r.customer_text.strip().lower(), r.brand_text.strip().lower())] += 1
    duplicate_pairs = sum(count - 1 for count in pair_signatures.values() if count > 1)

    empty_responses = sum(1 for r in all_resolutions if not r.brand_text.strip())
    url_pattern = re.compile(r"https?://", re.IGNORECASE)
    url_containing_responses = sum(1 for r in all_resolutions if url_pattern.search(r.brand_text))

    # Length statistics
    cust_char_lens = [len(r.customer_text) for r in all_resolutions]
    cust_word_lens = [len(r.customer_text.split()) for r in all_resolutions]
    brand_char_lens = [len(r.brand_text) for r in all_resolutions]
    brand_word_lens = [len(r.brand_text.split()) for r in all_resolutions]

    intent_counts = Counter(r.intent for r in all_resolutions if r.intent is not None)

    stats = {
        "train_conversations_scanned": scanned_train_convs,
        "total_pairs": len(all_resolutions),
        "unique_customer_tweets": unique_cust_tweets,
        "unique_brand_responses": unique_brand_responses,
        "pairs_with_silver_intent": pairs_with_silver,
        "pairs_without_silver_intent": pairs_without_silver,
        "duplicate_pairs": duplicate_pairs,
        "empty_responses": empty_responses,
        "url_containing_responses": url_containing_responses,
        "url_percentage": (url_containing_responses / len(all_resolutions) * 100) if all_resolutions else 0.0,
        "customer_text_stats": {
            "char_min": min(cust_char_lens) if cust_char_lens else 0,
            "char_max": max(cust_char_lens) if cust_char_lens else 0,
            "char_mean": round(mean(cust_char_lens), 2) if cust_char_lens else 0.0,
            "char_median": round(median(cust_char_lens), 2) if cust_char_lens else 0.0,
            "word_min": min(cust_word_lens) if cust_word_lens else 0,
            "word_max": max(cust_word_lens) if cust_word_lens else 0,
            "word_mean": round(mean(cust_word_lens), 2) if cust_word_lens else 0.0,
            "word_median": round(median(cust_word_lens), 2) if cust_word_lens else 0.0,
        },
        "brand_text_stats": {
            "char_min": min(brand_char_lens) if brand_char_lens else 0,
            "char_max": max(brand_char_lens) if brand_char_lens else 0,
            "char_mean": round(mean(brand_char_lens), 2) if brand_char_lens else 0.0,
            "char_median": round(median(brand_char_lens), 2) if brand_char_lens else 0.0,
            "word_min": min(brand_word_lens) if brand_word_lens else 0,
            "word_max": max(brand_word_lens) if brand_word_lens else 0,
            "word_mean": round(mean(brand_word_lens), 2) if brand_word_lens else 0.0,
            "word_median": round(median(brand_word_lens), 2) if brand_word_lens else 0.0,
        },
        "silver_intent_breakdown": dict(intent_counts.most_common()),
    }

    return all_resolutions, stats


def format_report(stats: dict[str, Any]) -> str:
    """Format the corpus audit report as a clean, human-readable text document."""
    lines = [
        "=" * 80,
        "HISTORICAL RESOLUTION CORPUS AUDIT REPORT (TRAIN SPLIT)",
        "=" * 80,
        "",
        "## 1. Corpus Overview & Leakage-Free Provenance",
        f"- TRAIN Conversations Scanned:     {stats['train_conversations_scanned']:,}",
        f"- Total Customer->Brand Pairs:      {stats['total_pairs']:,}",
        f"- Unique Customer Inbound Tweets:   {stats['unique_customer_tweets']:,}",
        f"- Unique Brand Response Tweets:     {stats['unique_brand_responses']:,}",
        f"- Golden Conversations Ingested:    0 (Strictly Excluded)",
        f"- DEV Conversations Ingested:       0 (Strictly Excluded)",
        f"- TEST Conversations Ingested:      0 (Strictly Excluded)",
        "",
        "## 2. Intent Annotation & Content Integrity",
        f"- Pairs with Silver Intent Tag:     {stats['pairs_with_silver_intent']:,} ({stats['pairs_with_silver_intent']/stats['total_pairs']*100:.2f}%)",
        f"- Pairs without Silver Intent Tag:  {stats['pairs_without_silver_intent']:,} ({stats['pairs_without_silver_intent']/stats['total_pairs']*100:.2f}%)",
        f"- Duplicate Pairs (Text Match):     {stats['duplicate_pairs']:,} ({stats['duplicate_pairs']/stats['total_pairs']*100:.2f}%)",
        f"- Empty Brand Responses:            {stats['empty_responses']:,}",
        f"- URL-Containing Brand Responses:   {stats['url_containing_responses']:,} ({stats['url_percentage']:.2f}%)",
        "",
        "## 3. Text Length Statistics",
        "Customer Problem Text:",
        f"  - Characters:  Min={stats['customer_text_stats']['char_min']}, Max={stats['customer_text_stats']['char_max']}, Mean={stats['customer_text_stats']['char_mean']:.1f}, Median={stats['customer_text_stats']['char_median']:.1f}",
        f"  - Words:       Min={stats['customer_text_stats']['word_min']}, Max={stats['customer_text_stats']['word_max']}, Mean={stats['customer_text_stats']['word_mean']:.1f}, Median={stats['customer_text_stats']['word_median']:.1f}",
        "Brand Resolution Response Text:",
        f"  - Characters:  Min={stats['brand_text_stats']['char_min']}, Max={stats['brand_text_stats']['char_max']}, Mean={stats['brand_text_stats']['char_mean']:.1f}, Median={stats['brand_text_stats']['char_median']:.1f}",
        f"  - Words:       Min={stats['brand_text_stats']['word_min']}, Max={stats['brand_text_stats']['word_max']}, Mean={stats['brand_text_stats']['word_mean']:.1f}, Median={stats['brand_text_stats']['word_median']:.1f}",
        "",
        "## 4. Silver Intent Distribution in Historical Resolutions",
        f"{'Intent':<25} | {'Count':>8} | {'Share of Tagged %':>18}",
        "-" * 57,
    ]

    tagged_total = max(1, stats["pairs_with_silver_intent"])
    for intent, count in stats["silver_intent_breakdown"].items():
        share = count / tagged_total * 100
        lines.append(f"{intent:<25} | {count:>8,d} | {share:>17.2f}%")

    lines.extend([
        "-" * 57,
        "",
        "## 5. Indexing Strategy",
        "- Default Retrieval Field: 'customer_text' (compares incoming customer query to past customer problem statements)",
        "- Historical Evidence: Paired AppleSupport response returned as resolution evidence",
        "- Vectorizer: TF-IDF (1-2 ngrams, sublinear TF scaling, max_features=50,000, L2 normalized)",
        "=" * 80,
    ])

    return "\n".join(lines)


def main() -> None:
    """Build historical resolution corpus, index retriever, and save artifacts."""
    parser = argparse.ArgumentParser(description="Build historical resolution corpus for TRAIN split.")
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS_PATH)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--silver-train", type=Path, default=DEFAULT_SILVER_TRAIN_PATH)
    parser.add_argument("--golden-annotation", type=Path, default=DEFAULT_GOLDEN_ANNOTATION_PATH)
    parser.add_argument("--golden-candidates", type=Path, default=DEFAULT_GOLDEN_CANDIDATES_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--index-field", type=str, default="customer", choices=["customer", "both", "brand"])
    args = parser.parse_args()

    # Step 1: Isolation verification
    golden_ids = load_forbidden_golden_ids(args.golden_annotation, args.golden_candidates)
    train_ids = load_verified_train_conversation_ids(args.splits, golden_ids)

    # Step 2: Silver intent lookup
    silver_lookup = load_silver_intents_lookup(args.silver_train)

    # Step 3: Extract pairs and compute stats
    resolutions, stats = build_corpus_and_statistics(args.conversations, train_ids, silver_lookup)

    # Step 4: Write JSONL
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %d historical resolutions to %s...", len(resolutions), args.output_jsonl)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for res in resolutions:
            f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")

    # Step 5: Write report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_text = format_report(stats)
    with args.report.open("w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Report written to %s", args.report)

    # Step 6: Fit and save default TFIDFRetriever
    logger.info("Fitting TFIDFRetriever with index_field='%s'...", args.index_field)
    retriever = TFIDFRetriever(
        index_field=args.index_field,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
    )
    retriever.fit(resolutions)
    saved_model_path = retriever.save(args.model_out)
    logger.info("Retriever model artifact saved successfully to %s", saved_model_path)

    print(report_text)


if __name__ == "__main__":
    main()
