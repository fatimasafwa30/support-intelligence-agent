"""Tests for scripts/build_silver_dataset.py.

Verifies:
1. Strict Golden Set exclusion (zero leakage).
2. TRAIN-split-only message filtering.
3. Inbound-only filtering (ignoring agent/support responses).
4. Omission of empty or whitespace-only texts.
5. Correct profiling logic and candidate threshold trade-off calculation.
6. Deterministic dataset generation with valid schema and required fields.
7. Verification of audit report formatting (disclaimers, selection rationale).
8. Integrity verification of generated silver training dataset (if present on disk).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_silver_dataset import (
    CANDIDATE_THRESHOLDS,
    DEFAULT_GOLDEN_ANNOTATION_PATH,
    DEFAULT_GOLDEN_CANDIDATES_PATH,
    DEFAULT_OUTPUT_JSONL_PATH,
    DEFAULT_SPLITS_PATH,
    FROZEN_RETENTION_THRESHOLD,
    REQUIRED_RECORD_FIELDS,
    build_silver_dataset,
    format_dataset_report,
    format_profile_report,
    load_forbidden_golden_ids,
    load_train_conversation_ids,
    profile_train_inbound_messages,
    stream_eligible_inbound_messages,
)
from src.intents.silver_labeler import CANONICAL_INTENTS, SilverLabeler


class TestBuildSilverDataset(unittest.TestCase):
    """Unit tests for silver training data builder and profiler."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Synthetic Golden CSVs
        self.golden_csv = self.root / "golden.csv"
        with self.golden_csv.open("w", encoding="utf-8") as f:
            f.write("candidate_id,conversation_id,tweet_id,intent\n")
            f.write("cand_1,apple_000001,101,battery_power\n")
            f.write("cand_2,apple_000002,102,software_update\n")

        self.candidates_csv = self.root / "candidates.csv"
        with self.candidates_csv.open("w", encoding="utf-8") as f:
            f.write("candidate_id,conversation_id,tweet_id\n")
            f.write("cand_1,apple_000001,101\n")
            f.write("cand_3,apple_000003,103\n")

        # Synthetic Split Manifest
        self.splits_json = self.root / "splits.json"
        manifest_data = {
            "metadata": {"seed": 42},
            "train_conversation_ids": [
                "apple_000010",
                "apple_000020",
                "apple_000030",
            ],
            "dev_conversation_ids": ["apple_000040"],
            "test_conversation_ids": ["apple_000050"],
        }
        with self.splits_json.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        # Synthetic Conversations JSONL
        self.conversations_jsonl = self.root / "conversations.jsonl"
        convs = [
            # Train conv 1: Battery issue (high confidence specific)
            {
                "conversation_id": "apple_000010",
                "tweets": [
                    {"tweet_id": "t10_1", "inbound": True, "created_at": "2017-10-01", "text": "My iPhone battery is draining so fast since today."},
                    {"tweet_id": "t10_2", "inbound": False, "created_at": "2017-10-01", "text": "We can help with that. DM us."},
                ],
            },
            # Train conv 2: Two customer messages (one Wi-Fi connectivity, one vague greeting)
            {
                "conversation_id": "apple_000020",
                "tweets": [
                    {"tweet_id": "t20_1", "inbound": True, "created_at": "2017-10-02", "text": "Wi-Fi not working and drops connection constantly"},
                    {"tweet_id": "t20_2", "inbound": False, "created_at": "2017-10-02", "text": "Have you restarted your router?"},
                    {"tweet_id": "t20_3", "inbound": True, "created_at": "2017-10-02", "text": "hello"},  # vague -> other_unclear
                ],
            },
            # Train conv 3: Empty text message + short strong signal
            {
                "conversation_id": "apple_000030",
                "tweets": [
                    {"tweet_id": "t30_1", "inbound": True, "created_at": "2017-10-03", "text": "   "},  # whitespace only
                    {"tweet_id": "t30_2", "inbound": True, "created_at": "2017-10-03", "text": "iPhones won't activate!"},  # setup_activation (0.80)
                ],
            },
            # Dev conv: should be excluded
            {
                "conversation_id": "apple_000040",
                "tweets": [
                    {"tweet_id": "t40_1", "inbound": True, "created_at": "2017-10-04", "text": "My iPad screen is cracked"},
                ],
            },
            # Golden conv: should be strictly excluded
            {
                "conversation_id": "apple_000001",
                "tweets": [
                    {"tweet_id": "t1_1", "inbound": True, "created_at": "2017-10-05", "text": "My battery is dead"},
                ],
            },
        ]
        with self.conversations_jsonl.open("w", encoding="utf-8") as f:
            for c in convs:
                f.write(json.dumps(c) + "\n")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_golden_conversation_exclusion_loading(self) -> None:
        """Verify forbidden golden IDs load from both annotation and candidate files."""
        forbidden = load_forbidden_golden_ids(self.golden_csv, self.candidates_csv)
        self.assertEqual(forbidden, {"apple_000001", "apple_000002", "apple_000003"})

    def test_golden_leakage_in_train_raises_error(self) -> None:
        """Verify load_train_conversation_ids raises ValueError if golden set overlaps train."""
        forbidden = {"apple_000010"}  # Deliberate injection of train conv
        with self.assertRaises(ValueError) as ctx:
            load_train_conversation_ids(self.splits_json, forbidden_golden_ids=forbidden)
        self.assertIn("FATAL: Golden Set leakage detected", str(ctx.exception))

    def test_streaming_filters_inbound_and_train_only(self) -> None:
        """Verify streaming only yields non-empty inbound messages from TRAIN non-golden conversations."""
        forbidden = load_forbidden_golden_ids(self.golden_csv, self.candidates_csv)
        train_ids = load_train_conversation_ids(self.splits_json, forbidden)

        messages = list(
            stream_eligible_inbound_messages(
                conversations_path=self.conversations_jsonl,
                train_conversation_ids=train_ids,
                forbidden_golden_ids=forbidden,
            )
        )

        # Expected messages:
        # conv apple_000010: t10_1
        # conv apple_000020: t20_1, t20_3
        # conv apple_000030: t30_2 (t30_1 was whitespace-only)
        # Total = 4
        self.assertEqual(len(messages), 4)

        tweet_ids = [m["tweet_id"] for m in messages]
        self.assertEqual(tweet_ids, ["t10_1", "t20_1", "t20_3", "t30_2"])

        # Check that no empty text exists
        for m in messages:
            self.assertTrue(bool(m["text"].strip()))

        # Check all conversation_ids are in train_ids
        for m in messages:
            self.assertIn(m["conversation_id"], train_ids)
            self.assertNotIn(m["conversation_id"], forbidden)

    def test_profiling_logic_and_candidate_thresholds(self) -> None:
        """Verify profile_train_inbound_messages accurately calculates distributions and trade-offs."""
        stats = profile_train_inbound_messages(
            conversations_path=self.conversations_jsonl,
            splits_manifest_path=self.splits_json,
            golden_annotation_path=self.golden_csv,
            golden_candidates_path=self.candidates_csv,
        )

        self.assertEqual(stats["total_train_messages_considered"], 4)
        self.assertEqual(stats["specific_label_count"], 3)
        self.assertEqual(stats["other_unclear_count"], 1)

        # Threshold simulations
        th_dict = {t["threshold"]: t for t in stats["threshold_analysis"]}

        # At 0.70 & 0.80: all 3 specific are retained (battery, connectivity, setup_activation)
        self.assertEqual(th_dict[0.70]["retained_count"], 3)
        self.assertEqual(th_dict[0.70]["other_unclear_excluded"], 1)
        self.assertEqual(th_dict[0.80]["retained_count"], 3)

        # At 0.85: setup_activation (0.80) is dropped due to low confidence
        self.assertEqual(th_dict[0.85]["retained_count"], 2)
        self.assertEqual(th_dict[0.85]["low_confidence_excluded"], 1)
        self.assertEqual(th_dict[0.85]["other_unclear_excluded"], 1)

        # At 0.95: all (0.92 and below) are dropped
        self.assertEqual(th_dict[0.95]["retained_count"], 0)
        self.assertEqual(th_dict[0.95]["low_confidence_excluded"], 3)

        # Format profile report
        report_text = format_profile_report(stats)
        self.assertIn("SILVER LABELER CONFIDENCE & COVERAGE PROFILE", report_text)
        self.assertIn("Candidate Retention Threshold Trade-Off Analysis", report_text)

    def test_build_silver_dataset_schema_and_determinism(self) -> None:
        """Verify building dataset produces deterministic, valid JSONL records matching schema."""
        out1 = self.root / "silver_1.jsonl"
        out2 = self.root / "silver_2.jsonl"

        stats1 = build_silver_dataset(
            conversations_path=self.conversations_jsonl,
            splits_manifest_path=self.splits_json,
            golden_annotation_path=self.golden_csv,
            golden_candidates_path=self.candidates_csv,
            output_jsonl_path=out1,
            threshold=0.80,
        )

        stats2 = build_silver_dataset(
            conversations_path=self.conversations_jsonl,
            splits_manifest_path=self.splits_json,
            golden_annotation_path=self.golden_csv,
            golden_candidates_path=self.candidates_csv,
            output_jsonl_path=out2,
            threshold=0.80,
        )

        # Byte-for-byte determinism
        self.assertEqual(out1.read_bytes(), out2.read_bytes())
        self.assertEqual(stats1["retained_count"], 3)
        self.assertEqual(stats1["excluded_other_unclear"], 1)

        # Schema verification
        with out1.open("r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        self.assertEqual(len(lines), 3)
        for rec in lines:
            self.assertEqual(set(rec.keys()), set(REQUIRED_RECORD_FIELDS))
            self.assertIn(rec["silver_intent"], CANONICAL_INTENTS)
            self.assertNotEqual(rec["silver_intent"], "other_unclear")
            self.assertGreaterEqual(rec["silver_confidence"], 0.80)
            self.assertTrue(bool(rec["text"].strip()))

    def test_format_dataset_report_content(self) -> None:
        """Verify that dataset report includes all required methodological notes and sections."""
        dummy_stats = {
            "threshold": 0.80,
            "total_considered": 83402,
            "retained_count": 18159,
            "retention_percentage": 21.77,
            "excluded_other_unclear": 65243,
            "excluded_low_confidence": 0,
            "total_excluded": 65243,
            "intent_counts": {"battery_power": 7143, "software_update": 2826},
            "confidence_counts": {"0.92": 8133, "0.80": 1686},
            "output_path": "data/processed/silver_train.jsonl",
        }
        report = format_dataset_report(dummy_stats)

        # Required elements check
        self.assertIn("83,402", report)
        self.assertIn("18,159", report)
        self.assertIn("21.77%", report)
        self.assertIn("65,243", report)
        self.assertIn("0.80", report)
        self.assertIn("Threshold Selection Rationale", report)
        self.assertIn("NOT be interpreted as an 80% probability", report)
        self.assertIn("NOT calibrated statistical probabilities", report)
        self.assertIn("Verification & Integrity Checklist", report)

    def test_verify_real_generated_dataset_integrity_if_exists(self) -> None:
        """If data/processed/silver_train.jsonl has been generated, comprehensively verify its integrity."""
        real_output = DEFAULT_OUTPUT_JSONL_PATH
        if not real_output.exists():
            self.skipTest(f"{real_output} does not exist yet; skipping disk verification.")

        forbidden = load_forbidden_golden_ids(
            DEFAULT_GOLDEN_ANNOTATION_PATH,
            DEFAULT_GOLDEN_CANDIDATES_PATH,
        )
        train_ids = load_train_conversation_ids(
            DEFAULT_SPLITS_PATH,
            forbidden_golden_ids=forbidden,
        )

        record_count = 0
        with real_output.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                rec = json.loads(line)
                record_count += 1

                # 1. Required fields
                self.assertEqual(set(rec.keys()), set(REQUIRED_RECORD_FIELDS))

                # 2. Golden exclusion
                cid = rec["conversation_id"]
                self.assertNotIn(cid, forbidden, f"Golden ID leakage at line {line_idx}: {cid}")

                # 3. Train-only
                self.assertIn(cid, train_ids, f"Non-train ID at line {line_idx}: {cid}")

                # 4. No other_unclear
                intent = rec["silver_intent"]
                self.assertIn(intent, CANONICAL_INTENTS)
                self.assertNotEqual(intent, "other_unclear", f"other_unclear at line {line_idx}")

                # 5. Confidence >= 0.80
                conf = rec["silver_confidence"]
                self.assertGreaterEqual(conf, 0.80, f"Confidence < 0.80 at line {line_idx}: {conf}")

                # 6. Non-empty text
                text = rec["text"]
                self.assertTrue(bool(text and text.strip()), f"Empty text at line {line_idx}")

        self.assertEqual(record_count, 18159, "Mismatch in expected silver retained count.")


if __name__ == "__main__":
    unittest.main()
