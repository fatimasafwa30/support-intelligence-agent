"""Unit and integration tests for the Golden failure analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_failures import main, run_failure_analysis


class TestFailureAnalysisPipeline(unittest.TestCase):
    """Test suite verifying failure analysis reproducibility, metrics, and report schemas."""

    @classmethod
    def setUpClass(cls):
        """Run analysis once for the test class to avoid redundant 6-second re-runs."""
        cls.data = run_failure_analysis()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_frozen_benchmark_invariants_match_authoritative_records(self):
        """Verify that all 10 frozen facts match the evaluated Golden Set deterministically."""
        bench = self.data["frozen_benchmark_verification"]

        # 1. Sample Size
        self.assertEqual(bench["sample_size"], 250)

        # 2. Operational 3-way routing actions
        self.assertEqual(bench["actions_3way"]["AUTO_HANDLE"], 16)
        self.assertEqual(bench["actions_3way"]["ASK_CLARIFICATION"], 126)
        self.assertEqual(bench["actions_3way"]["ESCALATE"], 108)

        # 3. Binary action accuracy (72.00%)
        self.assertEqual(bench["binary_accuracy"], 0.72)

        # 4. Under-escalation count and rate (8 / 180 = 4.44%)
        self.assertEqual(bench["under_escalation_count"], 8)
        self.assertAlmostEqual(bench["under_escalation_rate"], 0.0444, places=4)

        # 5. False escalation on Gold AUTO (62 / 70 = 88.57%)
        self.assertEqual(bench["false_escalation_count"], 62)
        self.assertAlmostEqual(bench["false_escalation_rate"], 0.8857, places=4)

        # 6. Critical safety recall (10 / 10 = 100%)
        self.assertEqual(bench["critical_safety_recall"], 1.0)
        self.assertEqual(bench["critical_cases_total"], 10)
        self.assertEqual(bench["critical_cases_escalated"], 10)

        # 7. Full intent accuracy (128 / 250 = 51.20%)
        self.assertEqual(bench["full_intent_accuracy"], 0.512)

        # 8. Specific-intent accuracy post-abstention (108 / 225 = 48.00%)
        self.assertEqual(bench["specific_intent_accuracy"], 0.48)

        # 9. Generated reply population (142 replies)
        self.assertEqual(bench["generated_replies_count"], 142)

        # 10. Pre-generation short-circuits (108 = 7 critical + 101 policy)
        self.assertEqual(bench["pre_generation_short_circuits"], 108)
        self.assertEqual(bench["pre_retrieval_critical_short_circuits"], 7)
        self.assertEqual(bench["pre_generation_policy_short_circuits"], 101)

    def test_exactly_five_evaluator_facing_failure_modes_present(self):
        """Verify that exactly 5 failure modes are synthesized with required structure."""
        fmodes = self.data["failure_modes"]
        self.assertEqual(len(fmodes), 5)

        expected_ids = {1, 2, 3, 4, 5}
        actual_ids = {fm["mode_id"] for fm in fmodes}
        self.assertEqual(actual_ids, expected_ids)

        for fm in fmodes:
            self.assertIn("name", fm)
            self.assertIn("impact_priority", fm)
            self.assertIn("measured_frequency", fm)
            self.assertIn("examples", fm)
            self.assertGreaterEqual(len(fm["examples"]), 1)
            self.assertLessEqual(len(fm["examples"]), 3)
            self.assertTrue(
                "root_cause_hypothesis" in fm or "measured_facts_vs_hypothesis" in fm,
                f"Mode {fm['mode_id']} missing hypothesis section",
            )
            self.assertIn("one_more_week_mitigation", fm)

    def test_all_failure_mode_examples_are_real_golden_ids(self):
        """Verify that zero synthetic or hallucinated IDs exist in the failure mode cards."""
        golden_path = Path("data/golden/golden_annotation.csv")
        import csv
        with golden_path.open("r", encoding="utf-8") as f:
            valid_ids = {row["id"] for row in csv.DictReader(f)}

        for fm in self.data["failure_modes"]:
            for ex in fm["examples"]:
                eid = ex["example_id"]
                self.assertIn(
                    eid,
                    valid_ids,
                    f"Example ID {eid} in mode {fm['mode_id']} is not in the real Golden Set!",
                )

    def test_conversational_context_audit_exact_counts(self):
        """Verify exact measured facts regarding multi-turn threads, prior context, and anaphora."""
        ctx = self.data["summaries"]["context_audit"]

        # 1. Total Golden examples
        self.assertEqual(ctx["total_golden"], 250)

        # 2. Total belonging to threads with >1 tweet/turn
        self.assertEqual(ctx["threads_with_gt_1_turn"], 250)

        # 3. Turns with and without preceding context
        self.assertEqual(ctx["evaluated_message_turn_gt_0"], 96)
        self.assertEqual(ctx["evaluated_message_turn_0"], 154)
        self.assertEqual(ctx["evaluated_message_turn_gt_0"] + ctx["evaluated_message_turn_0"], 250)

        # 4. Intent misclassifications & anaphora breakdown
        self.assertEqual(ctx["intent_errors_total"], 122)
        self.assertEqual(ctx["intent_errors_with_anaphora"], 66)
        self.assertEqual(ctx["intent_errors_with_anaphora_and_prior_context"], 29)
        self.assertEqual(ctx["intent_errors_with_anaphora_turn_0"], 37)
        self.assertEqual(
            ctx["intent_errors_with_anaphora_and_prior_context"] + ctx["intent_errors_with_anaphora_turn_0"],
            66,
        )

    def test_under_escalation_cases_exact_eight(self):
        """Verify that all 8 under-escalation cases are documented with complete information."""
        under_cases = self.data["summaries"]["under_escalation_examples"]
        self.assertEqual(len(under_cases), 8)

        under_ids = {c["example_id"] for c in under_cases}
        expected_under = {
            "golden_candidate_0035",
            "golden_candidate_0039",
            "golden_candidate_0062",
            "golden_candidate_0066",
            "golden_candidate_0132",
            "golden_candidate_0193",
            "golden_candidate_0208",
            "golden_candidate_0231",
        }
        self.assertEqual(under_ids, expected_under)

    def test_cli_runner_generates_valid_json_and_markdown_artifacts(self):
        """Verify CLI entrypoint generates non-empty, well-formed reports."""
        out_json = Path(self.temp_dir.name) / "test_out.json"
        out_md = Path(self.temp_dir.name) / "test_out.md"

        ret = main(["--output-json", str(out_json), "--output-md", str(out_md), "--quiet"])
        self.assertEqual(ret, 0)
        self.assertTrue(out_json.exists())
        self.assertTrue(out_md.exists())

        loaded_json = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertIn("frozen_benchmark_verification", loaded_json)
        self.assertIn("failure_modes", loaded_json)

        md_content = out_md.read_text(encoding="utf-8")
        self.assertIn("# Failure Analysis Report: Final-Candidate Golden Evaluation", md_content)
        self.assertIn("### Failure Mode 1:", md_content)
        self.assertIn("### Failure Mode 2:", md_content)
        self.assertIn("### Failure Mode 3:", md_content)
        self.assertIn("### Failure Mode 4:", md_content)
        self.assertIn("### Failure Mode 5:", md_content)
        self.assertIn("MEASURED FACTS", md_content)
        self.assertIn("ROOT-CAUSE HYPOTHESIS", md_content)


if __name__ == "__main__":
    unittest.main()
