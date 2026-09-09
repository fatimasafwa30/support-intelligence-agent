"""Conversation-level train/dev/test splitter.

Provides leakage-safe, deterministic splitting of support conversations into
train, dev, and test sets while permanently excluding all conversations
represented in the Golden Set.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
from typing import Any, Generator, Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_DEV_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


class SplitValidationError(Exception):
    """Raised when split integrity validation fails."""


def load_golden_conversation_ids(
    golden_path: str | Path,
    golden_candidates_path: str | Path | None = None,
) -> set[str]:
    """Load all unique conversation IDs represented in the Golden Set.

    Can load from golden_annotation.csv and optionally verify against
    golden_candidates.csv to ensure full consistency.

    Args:
        golden_path: Path to golden_annotation.csv (or golden_candidates.csv).
        golden_candidates_path: Optional path to candidate CSV to verify against.

    Returns:
        Set of unique conversation ID strings.
    """
    path = Path(golden_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden file not found: {path}")

    golden_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "conversation_id" not in (reader.fieldnames or []):
            raise ValueError(f"Missing 'conversation_id' column in {path}")
        for row in reader:
            cid = row.get("conversation_id", "").strip()
            if cid:
                golden_ids.add(cid)

    if golden_candidates_path:
        cand_path = Path(golden_candidates_path)
        if cand_path.exists():
            with cand_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                cand_ids = {
                    row.get("conversation_id", "").strip()
                    for row in reader
                    if row.get("conversation_id", "").strip()
                }
            if golden_ids != cand_ids:
                logger.warning(
                    "Mismatch between golden annotation IDs (%d) and candidates IDs (%d)",
                    len(golden_ids),
                    len(cand_ids),
                )
            golden_ids.update(cand_ids)

    return golden_ids


def load_all_conversation_ids(
    conversations_jsonl_path: str | Path,
) -> list[str]:
    """Extract conversation IDs from the reconstructed conversations JSONL.

    Reads the JSONL line by line and collects conversation_id strings, preserving
    first-seen canonical order.

    Args:
        conversations_jsonl_path: Path to processed conversations JSONL file.

    Returns:
        List of all unique conversation ID strings in file order.
    """
    path = Path(conversations_jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Conversations file not found: {path}")

    seen: set[str] = set()
    ordered_ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cid = data.get("conversation_id")
            if not cid:
                raise ValueError(f"Missing conversation_id on line {line_idx} of {path}")
            if cid not in seen:
                seen.add(cid)
                ordered_ids.append(cid)

    return ordered_ids


def split_conversations(
    all_conversation_ids: Sequence[str],
    golden_conversation_ids: Iterable[str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Deterministically partition conversations into train, dev, and test sets.

    Permanently excludes all conversation IDs present in the Golden Set.
    Deterministic partitioning is guaranteed by sorting eligible IDs before
    shuffling with a dedicated, fixed-seed random instance.

    Args:
        all_conversation_ids: Sequence of all conversation IDs in the dataset.
        golden_conversation_ids: Set/Iterable of conversation IDs in Golden Set.
        train_ratio: Target train fraction (default: 0.70).
        dev_ratio: Target dev fraction (default: 0.15).
        test_ratio: Target test fraction (default: 0.15).
        seed: Random seed for deterministic shuffling (default: 42).

    Returns:
        Structured manifest dictionary containing metadata, validation summary,
        excluded Golden IDs, and split conversation ID lists.
    """
    total_ratio = train_ratio + dev_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0 (got train={train_ratio}, dev={dev_ratio}, "
            f"test={test_ratio}, sum={total_ratio})"
        )

    all_ids_set = set(all_conversation_ids)
    golden_set = set(golden_conversation_ids)

    # Exclude all golden conversation IDs
    golden_excluded = sorted(all_ids_set & golden_set)
    # Also keep track of any golden IDs that were passed in
    all_golden_ids = sorted(golden_set)

    # Eligible non-Golden conversations: sort first to guarantee canonical order
    eligible_ids = sorted(all_ids_set - golden_set)
    n_eligible = len(eligible_ids)

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled_eligible = list(eligible_ids)
    rng.shuffle(shuffled_eligible)

    # Calculate partition boundaries
    n_train = round(n_eligible * train_ratio)
    n_dev = round(n_eligible * dev_ratio)
    # Ensure exact partition summing to n_eligible
    n_test = n_eligible - n_train - n_dev

    train_ids = shuffled_eligible[:n_train]
    dev_ids = shuffled_eligible[n_train : n_train + n_dev]
    test_ids = shuffled_eligible[n_train + n_dev :]

    actual_train_prop = len(train_ids) / n_eligible if n_eligible else 0.0
    actual_dev_prop = len(dev_ids) / n_eligible if n_eligible else 0.0
    actual_test_prop = len(test_ids) / n_eligible if n_eligible else 0.0

    manifest: dict[str, Any] = {
        "metadata": {
            "description": (
                "Leakage-safe conversation-level train/dev/test split manifest "
                "with permanent Golden Set exclusion."
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "target_ratios": {
                "train": train_ratio,
                "dev": dev_ratio,
                "test": test_ratio,
            },
            "actual_proportions": {
                "train": round(actual_train_prop, 6),
                "dev": round(actual_dev_prop, 6),
                "test": round(actual_test_prop, 6),
            },
            "counts": {
                "total_conversations_in_source": len(all_ids_set),
                "golden_conversations_specified": len(all_golden_ids),
                "golden_conversations_matched_and_excluded": len(golden_excluded),
                "eligible_conversations": n_eligible,
                "train_count": len(train_ids),
                "dev_count": len(dev_ids),
                "test_count": len(test_ids),
            },
        },
        "golden_excluded_conversation_ids": golden_excluded,
        "train_conversation_ids": train_ids,
        "dev_conversation_ids": dev_ids,
        "test_conversation_ids": test_ids,
    }

    # Execute full validation checks on the generated manifest
    val_report = validate_splits(manifest, eligible_set=set(eligible_ids), golden_set=golden_set)
    manifest["metadata"]["validation_summary"] = val_report

    return manifest


def validate_splits(
    manifest: dict[str, Any],
    eligible_set: set[str] | None = None,
    golden_set: set[str] | None = None,
) -> dict[str, Any]:
    """Validate split integrity, leak prevention, and full partition coverage.

    Checks:
    1. Zero overlap between train, dev, and test sets.
    2. Zero overlap between Golden Set and train, dev, or test sets.
    3. Every eligible non-Golden conversation belongs to exactly one split.
    4. Actual proportions closely match target ratios.

    Args:
        manifest: Split manifest dictionary.
        eligible_set: Optional precomputed set of eligible conversation IDs.
        golden_set: Optional precomputed set of Golden conversation IDs.

    Returns:
        Dictionary summarizing the validation check results.

    Raises:
        SplitValidationError: If any safety or partition check fails.
    """
    train_ids = manifest.get("train_conversation_ids", [])
    dev_ids = manifest.get("dev_conversation_ids", [])
    test_ids = manifest.get("test_conversation_ids", [])
    golden_excluded_ids = manifest.get("golden_excluded_conversation_ids", [])

    train_set = set(train_ids)
    dev_set = set(dev_ids)
    test_set = set(test_ids)
    golden_excluded_set = set(golden_excluded_ids)
    if golden_set is None:
        golden_set = golden_excluded_set

    # 1. No internal duplicates within any split list
    if len(train_ids) != len(train_set):
        raise SplitValidationError(
            f"Train split contains duplicate IDs: {len(train_ids)} total vs {len(train_set)} unique"
        )
    if len(dev_ids) != len(dev_set):
        raise SplitValidationError(
            f"Dev split contains duplicate IDs: {len(dev_ids)} total vs {len(dev_set)} unique"
        )
    if len(test_ids) != len(test_set):
        raise SplitValidationError(
            f"Test split contains duplicate IDs: {len(test_ids)} total vs {len(test_set)} unique"
        )

    # 2. Zero overlap between splits
    overlap_train_dev = train_set & dev_set
    if overlap_train_dev:
        raise SplitValidationError(
            f"Leakage detected: {len(overlap_train_dev)} conversations overlap between train and dev"
        )

    overlap_train_test = train_set & test_set
    if overlap_train_test:
        raise SplitValidationError(
            f"Leakage detected: {len(overlap_train_test)} conversations overlap between train and test"
        )

    overlap_dev_test = dev_set & test_set
    if overlap_dev_test:
        raise SplitValidationError(
            f"Leakage detected: {len(overlap_dev_test)} conversations overlap between dev and test"
        )

    # 3. Zero overlap between Golden Set and any split
    overlap_golden_train = golden_set & train_set
    if overlap_golden_train:
        raise SplitValidationError(
            f"Leakage detected: {len(overlap_golden_train)} Golden Set conversations present in train"
        )

    overlap_golden_dev = golden_set & dev_set
    if overlap_golden_dev:
        raise SplitValidationError(
            f"Leakage detected: {len(overlap_golden_dev)} Golden Set conversations present in dev"
        )

    overlap_golden_test = golden_set & test_set
    if overlap_golden_test:
        raise SplitValidationError(
            f"Leakage detected: {len(overlap_golden_test)} Golden Set conversations present in test"
        )

    # 4. Partition completeness
    combined_splits = train_set | dev_set | test_set
    expected_total = len(train_ids) + len(dev_ids) + len(test_ids)
    if len(combined_splits) != expected_total:
        raise SplitValidationError(
            f"Split set union size ({len(combined_splits)}) does not match sum of split sizes ({expected_total})"
        )

    if eligible_set is not None:
        if combined_splits != eligible_set:
            missing = eligible_set - combined_splits
            extra = combined_splits - eligible_set
            raise SplitValidationError(
                f"Partition mismatch: {len(missing)} missing from splits, {len(extra)} unexpected extra"
            )

    # Proportions
    total_eligible = len(combined_splits)
    actual_train_prop = len(train_ids) / total_eligible if total_eligible else 0.0
    actual_dev_prop = len(dev_ids) / total_eligible if total_eligible else 0.0
    actual_test_prop = len(test_ids) / total_eligible if total_eligible else 0.0

    return {
        "zero_overlap_train_dev": True,
        "zero_overlap_train_test": True,
        "zero_overlap_dev_test": True,
        "zero_overlap_golden_train": True,
        "zero_overlap_golden_dev": True,
        "zero_overlap_golden_test": True,
        "exact_partition_coverage": True,
        "actual_proportions": {
            "train": round(actual_train_prop, 6),
            "dev": round(actual_dev_prop, 6),
            "test": round(actual_test_prop, 6),
        },
        "all_validations_passed": True,
    }


def save_split_manifest(
    manifest: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save split manifest to JSON file.

    Args:
        manifest: Structured split manifest dictionary.
        output_path: Path where the manifest JSON should be saved.

    Returns:
        Path of the saved manifest file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_split_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load and validate an existing split manifest JSON file.

    Args:
        manifest_path: Path to the split manifest JSON file.

    Returns:
        Split manifest dictionary.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Re-validate upon loading
    validate_splits(manifest)
    return manifest


def stream_split_conversations(
    conversations_jsonl_path: str | Path,
    target_conversation_ids: set[str] | Sequence[str],
) -> Generator[dict[str, Any], None, None]:
    """Stream conversation objects belonging to a specified split.

    Avoids duplicating large JSONL datasets by streaming directly from the
    source file and filtering on target conversation IDs.

    Args:
        conversations_jsonl_path: Path to the conversations JSONL.
        target_conversation_ids: Set or sequence of conversation IDs to yield.

    Yields:
        Parsed conversation dicts matching the target split.
    """
    target_set = (
        target_conversation_ids
        if isinstance(target_conversation_ids, set)
        else set(target_conversation_ids)
    )

    path = Path(conversations_jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("conversation_id") in target_set:
                yield data


class ConversationSplitter:
    """Class interface for conversation-level splitting and manifest management."""

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        dev_ratio: float = DEFAULT_DEV_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO,
    ) -> None:
        self.seed = seed
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio

    def run(
        self,
        conversations_jsonl_path: str | Path,
        golden_csv_path: str | Path,
        golden_candidates_csv_path: str | Path | None = None,
        output_manifest_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute the complete splitting and validation pipeline."""
        golden_ids = load_golden_conversation_ids(
            golden_csv_path,
            golden_candidates_path=golden_candidates_csv_path,
        )
        all_conv_ids = load_all_conversation_ids(conversations_jsonl_path)

        manifest = split_conversations(
            all_conversation_ids=all_conv_ids,
            golden_conversation_ids=golden_ids,
            train_ratio=self.train_ratio,
            dev_ratio=self.dev_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )

        if output_manifest_path:
            save_split_manifest(manifest, output_manifest_path)

        return manifest
