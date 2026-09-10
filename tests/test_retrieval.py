"""Unit tests for Historical-Resolution Retrieval pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.retrieval.historical_resolution import HistoricalResolution, RetrievalResult
from src.retrieval.tfidf_retriever import TFIDFRetriever
from scripts.build_resolution_corpus import (
    extract_pairs_from_conversation,
    load_verified_train_conversation_ids,
)


class TestHistoricalResolutionDataModel(unittest.TestCase):
    """Tests for HistoricalResolution and RetrievalResult dataclasses."""

    def test_round_trip_dict(self) -> None:
        res = HistoricalResolution(
            resolution_id="res_001_10_20",
            conversation_id="conv_001",
            customer_tweet_id="10",
            brand_tweet_id="20",
            customer_text="iPhone 7 battery draining fast",
            brand_text="Follow these steps: https://apple.co/battery",
            intent="battery_power",
            created_at="2017-10-01",
        )
        d = res.to_dict()
        res_reconstructed = HistoricalResolution.from_dict(d)
        self.assertEqual(res, res_reconstructed)

    def test_retrieval_result_formatting(self) -> None:
        res = HistoricalResolution(
            resolution_id="res_001",
            conversation_id="conv_001",
            customer_tweet_id="10",
            brand_tweet_id="20",
            customer_text="Help",
            brand_text="On it",
        )
        result = RetrievalResult(resolution=res, score=0.85432, rank=1)
        d = result.to_dict()
        self.assertEqual(d["rank"], 1)
        self.assertEqual(d["score"], 0.8543)
        self.assertEqual(d["resolution"]["resolution_id"], "res_001")


class TestPairExtraction(unittest.TestCase):
    """Tests for customer-brand pair extraction logic."""

    def test_extract_pairs_valid(self) -> None:
        conv = {
            "conversation_id": "conv_123",
            "tweets": [
                {
                    "tweet_id": "1",
                    "inbound": True,
                    "text": "My iPad screen is black",
                    "created_at": "2017-11-01",
                },
                {
                    "tweet_id": "2",
                    "inbound": False,
                    "in_response_to_tweet_id": "1",
                    "text": "Try force restarting: https://apple.co/restart",
                    "created_at": "2017-11-01",
                },
                {
                    "tweet_id": "3",
                    "inbound": True,
                    "in_response_to_tweet_id": "2",
                    "text": "Still not working",
                    "created_at": "2017-11-01",
                },
                {
                    "tweet_id": "4",
                    "inbound": False,
                    "in_response_to_tweet_id": "3",
                    "text": "Please DM us: https://apple.co/dm",
                    "created_at": "2017-11-01",
                },
            ],
        }
        silver_lookup = {"1": "device_hardware"}
        pairs = extract_pairs_from_conversation(conv, silver_lookup)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].customer_tweet_id, "1")
        self.assertEqual(pairs[0].brand_tweet_id, "2")
        self.assertEqual(pairs[0].intent, "device_hardware")
        self.assertEqual(pairs[1].customer_tweet_id, "3")
        self.assertEqual(pairs[1].brand_tweet_id, "4")
        self.assertIsNone(pairs[1].intent)


class TestLeakageVerification(unittest.TestCase):
    """Tests for strict split isolation and leakage prevention."""

    def test_leakage_detected_golden(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_file = Path(tmpdir) / "splits.json"
            splits_data = {
                "train_conversation_ids": ["conv_1", "conv_2", "conv_leak"],
                "golden_excluded_conversation_ids": ["conv_leak"],
                "dev_conversation_ids": [],
                "test_conversation_ids": [],
            }
            with splits_file.open("w") as f:
                json.dump(splits_data, f)

            with self.assertRaises(ValueError) as ctx:
                load_verified_train_conversation_ids(splits_file, forbidden_golden_ids={"conv_leak"})
            self.assertIn("overlapping with golden_excluded", str(ctx.exception))


class TestTFIDFRetriever(unittest.TestCase):
    """Tests for TFIDFRetriever indexing, scoring, and leakage protection."""

    def setUp(self) -> None:
        self.resolutions = [
            HistoricalResolution(
                resolution_id="res_01",
                conversation_id="conv_101",
                customer_tweet_id="t1",
                brand_tweet_id="t2",
                customer_text="How to restore my iPhone from iCloud backup?",
                brand_text="Check out this guide: https://apple.co/icloud-restore",
                intent="icloud",
            ),
            HistoricalResolution(
                resolution_id="res_02",
                conversation_id="conv_102",
                customer_tweet_id="t3",
                brand_tweet_id="t4",
                customer_text="My iPhone battery drains within two hours after updating iOS",
                brand_text="Review battery optimization tips: https://apple.co/battery-tips",
                intent="battery_power",
            ),
            HistoricalResolution(
                resolution_id="res_03",
                conversation_id="conv_103",
                customer_tweet_id="t5",
                brand_tweet_id="t6",
                customer_text="Cannot connect to home Wi-Fi network with iPhone 8",
                brand_text="Reset network settings in Settings > General > Reset",
                intent="connectivity",
            ),
        ]
        self.retriever = TFIDFRetriever(index_field="customer", min_df=1)
        self.retriever.fit(self.resolutions)

    def test_default_index_field_is_customer(self) -> None:
        retriever = TFIDFRetriever()
        self.assertEqual(retriever.index_field, "customer")

    def test_retrieval_relevance_and_ranking(self) -> None:
        results = self.retriever.retrieve("battery draining fast on my phone", top_k=2)
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match.resolution.resolution_id, "res_02")
        self.assertEqual(top_match.resolution.intent, "battery_power")
        self.assertGreater(top_match.score, 0.0)
        self.assertLessEqual(top_match.score, 1.0)
        self.assertEqual(top_match.rank, 1)

    def test_query_conversation_leakage_protection(self) -> None:
        """Verify that exclude_conversation_id strictly prevents the query's conversation from being retrieved."""
        # Query exactly matches res_01 in conv_101
        query = "How to restore my iPhone from iCloud backup?"

        # Without exclusion, conv_101 is the #1 result
        unrestricted = self.retriever.retrieve(query, top_k=1)
        self.assertEqual(len(unrestricted), 1)
        self.assertEqual(unrestricted[0].resolution.conversation_id, "conv_101")

        # With query-conversation exclusion, conv_101 MUST NEVER be returned
        restricted = self.retriever.retrieve(
            query,
            top_k=5,
            exclude_conversation_id="conv_101",
        )
        returned_conv_ids = [r.resolution.conversation_id for r in restricted]
        self.assertNotIn("conv_101", returned_conv_ids)

    def test_intent_filtering(self) -> None:
        # Query with battery words, but filter specifically for connectivity
        results = self.retriever.retrieve(
            "iPhone network connectivity issue",
            top_k=5,
            filter_intent="connectivity",
        )
        for r in results:
            self.assertEqual(r.resolution.intent, "connectivity")

    def test_empty_and_out_of_vocabulary_queries(self) -> None:
        self.assertEqual(self.retriever.retrieve(""), [])
        self.assertEqual(self.retriever.retrieve("   "), [])
        self.assertEqual(self.retriever.retrieve("xyznonexistentterm12345"), [])

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "retriever.joblib"
            self.retriever.save(artifact_path)

            loaded = TFIDFRetriever.load(artifact_path)
            self.assertEqual(loaded.index_field, "customer")
            self.assertEqual(len(loaded.resolutions), 3)

            orig_res = self.retriever.retrieve("battery drain", top_k=2)
            loaded_res = loaded.retrieve("battery drain", top_k=2)

            self.assertEqual(len(orig_res), len(loaded_res))
            for r1, r2 in zip(orig_res, loaded_res):
                self.assertEqual(r1.resolution.resolution_id, r2.resolution.resolution_id)
                self.assertAlmostEqual(r1.score, r2.score, places=4)


if __name__ == "__main__":
    unittest.main()
