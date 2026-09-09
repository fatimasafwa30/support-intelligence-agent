"""Unit tests for conversation-level train/dev/test splitter."""

import json
from pathlib import Path
import tempfile
import unittest

from src.data.splitter import (
    ConversationSplitter,
    SplitValidationError,
    load_all_conversation_ids,
    load_golden_conversation_ids,
    load_split_manifest,
    save_split_manifest,
    split_conversations,
    stream_split_conversations,
    validate_splits,
)


class TestConversationSplitter(unittest.TestCase):
    """Test suite for conversation splitter and leak prevention."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ratio_sum_validation(self) -> None:
        """Check that invalid split ratios raise ValueError."""
        with self.assertRaises(ValueError):
            split_conversations(
                all_conversation_ids=["c1", "c2", "c3"],
                golden_conversation_ids=set(),
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.2,  # sum = 1.1
            )

    def test_golden_exclusion_and_zero_overlap(self) -> None:
        """Verify that golden conversations are completely excluded and splits are disjoint."""
        all_ids = [f"conv_{i:04d}" for i in range(100)]
        golden_ids = {f"conv_{i:04d}" for i in range(10)}  # first 10 are golden

        manifest = split_conversations(
            all_conversation_ids=all_ids,
            golden_conversation_ids=golden_ids,
            train_ratio=0.70,
            dev_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        train = set(manifest["train_conversation_ids"])
        dev = set(manifest["dev_conversation_ids"])
        test = set(manifest["test_conversation_ids"])
        golden_excluded = set(manifest["golden_excluded_conversation_ids"])

        # Check exclusion
        self.assertEqual(golden_excluded, golden_ids)
        self.assertTrue(train.isdisjoint(golden_ids))
        self.assertTrue(dev.isdisjoint(golden_ids))
        self.assertTrue(test.isdisjoint(golden_ids))

        # Check mutual disjointness
        self.assertTrue(train.isdisjoint(dev))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(dev.isdisjoint(test))

        # Check total count and proportions
        eligible = set(all_ids) - golden_ids
        self.assertEqual(train | dev | test, eligible)
        self.assertEqual(len(train) + len(dev) + len(test), 90)
        self.assertEqual(len(train), 63)
        self.assertEqual(len(dev), 14)  # round(90 * 0.15) = round(13.5) = 14
        self.assertEqual(len(test), 13) # 90 - 63 - 14 = 13

    def test_determinism(self) -> None:
        """Verify identical splits with same seed and different with different seeds."""
        all_ids = [f"conv_{i:04d}" for i in range(200)]
        golden_ids = {f"conv_{i:04d}" for i in range(20)}

        run1 = split_conversations(all_ids, golden_ids, seed=42)
        run2 = split_conversations(all_ids, golden_ids, seed=42)
        run3 = split_conversations(all_ids, golden_ids, seed=999)

        self.assertEqual(run1["train_conversation_ids"], run2["train_conversation_ids"])
        self.assertEqual(run1["dev_conversation_ids"], run2["dev_conversation_ids"])
        self.assertEqual(run1["test_conversation_ids"], run2["test_conversation_ids"])

        self.assertNotEqual(run1["train_conversation_ids"], run3["train_conversation_ids"])

    def test_leakage_validation_failure(self) -> None:
        """Ensure SplitValidationError is raised when overlap occurs."""
        corrupt_manifest = {
            "train_conversation_ids": ["c1", "c2", "c3"],
            "dev_conversation_ids": ["c3", "c4"],  # c3 overlaps with train
            "test_conversation_ids": ["c5"],
            "golden_excluded_conversation_ids": ["g1"],
        }
        with self.assertRaises(SplitValidationError) as ctx:
            validate_splits(corrupt_manifest)
        self.assertIn("Leakage detected", str(ctx.exception))

    def test_golden_leakage_validation_failure(self) -> None:
        """Ensure SplitValidationError is raised when Golden Set leaks into train."""
        corrupt_manifest = {
            "train_conversation_ids": ["c1", "g1"],  # g1 is golden!
            "dev_conversation_ids": ["c2"],
            "test_conversation_ids": ["c3"],
            "golden_excluded_conversation_ids": ["g1"],
        }
        with self.assertRaises(SplitValidationError) as ctx:
            validate_splits(corrupt_manifest)
        self.assertIn("Golden Set conversations present in train", str(ctx.exception))

    def test_save_and_load_manifest(self) -> None:
        """Test serializing and reading back manifest file."""
        all_ids = [f"c_{i}" for i in range(50)]
        golden = {"c_0", "c_1"}
        manifest = split_conversations(all_ids, golden, seed=123)

        manifest_file = self.temp_path / "test_splits.json"
        save_split_manifest(manifest, manifest_file)

        self.assertTrue(manifest_file.exists())
        loaded = load_split_manifest(manifest_file)
        self.assertEqual(loaded["train_conversation_ids"], manifest["train_conversation_ids"])
        self.assertEqual(loaded["metadata"]["seed"], 123)

    def test_streaming_conversations(self) -> None:
        """Test streaming helper retrieves only requested split conversations."""
        jsonl_path = self.temp_path / "dummy_convs.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"conversation_id": f"conv_{i}", "text": f"hello {i}"}) + "\n")

        target_ids = {"conv_2", "conv_5", "conv_9"}
        streamed = list(stream_split_conversations(jsonl_path, target_ids))
        streamed_ids = {c["conversation_id"] for c in streamed}
        self.assertEqual(streamed_ids, target_ids)
        self.assertEqual(len(streamed), 3)


if __name__ == "__main__":
    unittest.main()
