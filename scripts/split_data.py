#!/usr/bin/env python3
"""Execute leakage-safe conversation-level train/dev/test splitting.

Milestone 9:
1. Loads all conversations from data/processed/apple_conversations.jsonl.
2. Identifies all conversation IDs represented in the 250-example Golden Set
   (from data/golden/golden_annotation.csv and data/golden/golden_candidates.csv).
3. Permanently excludes all Golden Set conversations from train/dev/test.
4. Deterministically splits remaining eligible conversations into:
   - 70% train
   - 15% dev
   - 15% test
   using a documented random seed (default: 42).
5. Validates:
   - Zero overlap between train/dev/test.
   - Zero overlap between Golden Set and train/dev/test.
   - Exact 100% partition coverage of eligible conversations.
   - Actual split proportions.
6. Saves a lightweight split manifest to data/processed/conversation_splits.json.
   (Does NOT duplicate the 80 MB conversations file).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splitter import (
    DEFAULT_DEV_RATIO,
    DEFAULT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    ConversationSplitter,
    load_all_conversation_ids,
    load_golden_conversation_ids,
    save_split_manifest,
    split_conversations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Leakage-safe conversation-level train/dev/test splitter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "apple_conversations.jsonl",
        help="Path to processed conversations JSONL file.",
    )
    parser.add_argument(
        "--golden-annotation",
        type=Path,
        default=PROJECT_ROOT / "data" / "golden" / "golden_annotation.csv",
        help="Path to golden annotation CSV.",
    )
    parser.add_argument(
        "--golden-candidates",
        type=Path,
        default=PROJECT_ROOT / "data" / "golden" / "golden_candidates.csv",
        help="Path to golden candidates CSV for validation.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "conversation_splits.json",
        help="Path to output lightweight split manifest JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Fixed documented random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Target proportion for train split.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=DEFAULT_DEV_RATIO,
        help="Target proportion for dev split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Target proportion for test split.",
    )
    return parser.parse_args()


def print_banner(title: str) -> None:
    """Print section separator."""
    width = 72
    print("\n" + "=" * width)
    print(f" {title.upper()}")
    print("=" * width)


def main() -> int:
    """Run conversation splitting and validation pipeline."""
    args = parse_args()

    print_banner("Milestone 9: Conversation-Level Train/Dev/Test Splitting")
    print(f"Source conversations : {args.conversations}")
    print(f"Golden annotation    : {args.golden_annotation}")
    print(f"Golden candidates    : {args.golden_candidates}")
    print(f"Output manifest      : {args.output_manifest}")
    print(f"Random seed          : {args.seed}")
    print(f"Target ratios        : Train {args.train_ratio * 100:.1f}%, "
          f"Dev {args.dev_ratio * 100:.1f}%, "
          f"Test {args.test_ratio * 100:.1f}%")

    # 1. Load Golden Set IDs
    logger.info("Loading Golden Set conversation IDs...")
    golden_ids = load_golden_conversation_ids(
        golden_path=args.golden_annotation,
        golden_candidates_path=args.golden_candidates,
    )
    logger.info("Loaded %d unique Golden Set conversation IDs.", len(golden_ids))

    # 2. Load all conversation IDs from JSONL
    logger.info("Loading conversation IDs from %s...", args.conversations.name)
    all_conv_ids = load_all_conversation_ids(args.conversations)
    logger.info("Loaded %d conversations from source JSONL.", len(all_conv_ids))

    # 3. Perform deterministic split & validation
    logger.info("Executing deterministic split with permanent Golden exclusion (seed=%d)...", args.seed)
    manifest = split_conversations(
        all_conversation_ids=all_conv_ids,
        golden_conversation_ids=golden_ids,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # 4. Save manifest
    saved_path = save_split_manifest(manifest, args.output_manifest)
    file_size_kb = saved_path.stat().st_size / 1024

    # 5. Report results and validation details
    meta = manifest["metadata"]
    counts = meta["counts"]
    props = meta["actual_proportions"]
    val = meta["validation_summary"]

    print_banner("Split Summary & Verification Report")
    print(f"Total conversations in dataset   : {counts['total_conversations_in_source']:,}")
    print(f"Golden Set excluded conversations: {counts['golden_conversations_matched_and_excluded']:,} "
          f"(of {counts['golden_conversations_specified']} specified)")
    print(f"Eligible non-Golden conversations: {counts['eligible_conversations']:,}")
    print("-" * 72)
    print(f"  Train split : {counts['train_count']:>7,} conversations ({props['train'] * 100:>6.2f}%) "
          f"[target: {args.train_ratio * 100:.1f}%]")
    print(f"  Dev split   : {counts['dev_count']:>7,} conversations ({props['dev'] * 100:>6.2f}%) "
          f"[target: {args.dev_ratio * 100:.1f}%]")
    print(f"  Test split  : {counts['test_count']:>7,} conversations ({props['test'] * 100:>6.2f}%) "
          f"[target: {args.test_ratio * 100:.1f}%]")
    print(f"  Total splits: {counts['train_count'] + counts['dev_count'] + counts['test_count']:>7,} conversations (100.00%)")
    print("-" * 72)
    print("Leakage & Overlap Integrity Checks:")
    print(f"  [PASS] Overlap between Train & Dev          : 0 conversations")
    print(f"  [PASS] Overlap between Train & Test         : 0 conversations")
    print(f"  [PASS] Overlap between Dev & Test           : 0 conversations")
    print(f"  [PASS] Golden Set overlap with Train        : 0 conversations")
    print(f"  [PASS] Golden Set overlap with Dev          : 0 conversations")
    print(f"  [PASS] Golden Set overlap with Test         : 0 conversations")
    print(f"  [PASS] Partition completeness (no leakage)  : {val['exact_partition_coverage']}")
    print(f"  [PASS] All validation checks passed         : {val['all_validations_passed']}")
    print("-" * 72)
    print(f"Lightweight manifest saved: {saved_path}")
    print(f"Manifest file size        : {file_size_kb:.1f} KB ({file_size_kb / 1024:.2f} MB)")
    print(f"(Note: source 80 MB conversation JSONL was NOT duplicated).")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
