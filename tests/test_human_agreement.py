"""Offline tests; synthetic input is confined to temporary files, never human ratings."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from scripts import rate_human_agreement as tool


class TestHumanAgreement(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ratings.jsonl"
        self.samples = [{"example_id": f"test_{i}"} for i in range(13)]
        self.ids = {s["example_id"] for s in self.samples}
        self.rubric = yaml.safe_load(tool.DEFAULT_RUBRIC.read_text(encoding="utf-8"))

    def record(self, example_id="test_0"):
        return {"example_id": example_id, **dict.fromkeys(tool.RUBRIC_DIMENSIONS, 3)}

    def context(self, sample):
        return {**sample, "customer_query": "Synthetic query", "predicted_intent": "test intent",
                "retrieved_evidence": [{"evidence_id": "test_evidence", "past_customer_problem": "Test problem",
                                        "past_brand_resolution": "Test resolution", "extracted_urls": []}],
                "generated_reply": {"reply_text": "Synthetic reply"},
                "verified_reply": {"reply_text": "Synthetic verified reply"}}

    def test_score_validation(self):
        for score in range(1, 6):
            self.assertEqual(tool.parse_score(str(score)), score)
        for value in ("0", "6", "text", "", " ", "3.0", None, True, 3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tool.parse_score(value)
        for dim in tool.RUBRIC_DIMENSIONS:
            for value in (0, 6, "3", None, True, 3.0):
                with self.subTest(dim=dim, value=value), self.assertRaises((ValueError, TypeError)):
                    tool.validate_rating({**self.record(), dim: value}, self.ids)
            incomplete = self.record()
            del incomplete[dim]
            with self.assertRaises(ValueError):
                tool.validate_rating(incomplete, self.ids)

    def test_resume_skips_twelve_completed_and_appends_only_remaining(self):
        for sample in self.samples[:12]:
            tool.append_rating(self.path, self.record(sample["example_id"]), self.ids)
        before = self.path.read_bytes()
        provider = Mock(side_effect=self.context)
        answers = iter(["1", "2", "3", "4", "5", ""])
        tool.rate_samples(self.samples, self.path, self.rubric, provider, lambda _: next(answers), lambda _: None)
        provider.assert_called_once_with(self.samples[12])
        self.assertTrue(self.path.read_bytes().startswith(before))
        self.assertEqual(len(tool.load_ratings(self.path, self.ids)), 13)

    def test_duplicate_prevention_and_existing_duplicates_rejected(self):
        tool.append_rating(self.path, self.record(), self.ids)
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            tool.append_rating(self.path, self.record(), self.ids)
        self.assertEqual(self.path.read_bytes(), before)
        self.path.write_bytes(before + before)
        with self.assertRaises(ValueError):
            tool.load_ratings(self.path, self.ids)

    def test_corrupt_partial_file_is_not_silently_overwritten(self):
        self.path.write_text('{"example_id":', encoding="utf-8")
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            tool.append_rating(self.path, self.record(), self.ids)
        self.assertEqual(self.path.read_bytes(), before)

    def test_append_to_record_without_newline(self):
        self.path.write_text(json.dumps(self.record()), encoding="utf-8")
        tool.append_rating(self.path, self.record("test_1"), self.ids)
        self.assertEqual(len(tool.load_ratings(self.path, self.ids)), 2)

    def test_exit_and_interrupt_preserve_completed_records(self):
        for stop in ("q", "quit", "exit", KeyboardInterrupt(), EOFError()):
            with self.subTest(stop=type(stop).__name__):
                self.path.unlink(missing_ok=True)
                answers = iter(["1", "2", "3", "4", "5", "", "2", stop])
                def read(_):
                    value = next(answers)
                    if isinstance(value, BaseException):
                        raise value
                    return value
                tool.rate_samples(self.samples[:2], self.path, self.rubric, self.context, read, lambda _: None)
                self.assertEqual(set(tool.load_ratings(self.path, self.ids)), {"test_0"})
                self.assertFalse(self.path.with_suffix(".jsonl.lock").exists())

    def test_invalid_input_reprompts_and_rubric_matches_yaml(self):
        answers = iter(["", "0", "6", "text", "1", "2", "3", "4", "5", "Synthetic test note"])
        output = []
        tool.rate_samples(self.samples[:1], self.path, self.rubric, self.context, lambda _: next(answers), output.append)
        saved = tool.load_ratings(self.path, self.ids)["test_0"]
        self.assertEqual(saved["groundedness"], 1)
        self.assertEqual(saved["human_note"], "Synthetic test note")
        rendered = "\n".join(output)
        for dim in tool.RUBRIC_DIMENSIONS:
            self.assertIn(self.rubric["dimensions"][dim]["description"], rendered)
            for definition in self.rubric["dimensions"][dim]["scoring_guidance"].values():
                self.assertIn(definition, rendered)

    def test_labels_and_judge_scores_never_render_or_store(self):
        context = self.context(self.samples[0])
        extras = {key: "SECRET_SENTINEL" for key in
                  ("gold_intent", "gold_risk", "gold_action", "annotation_notes", "gold_custom",
                   "judge_scores", "rationales", "overall_score", *tool.RUBRIC_DIMENSIONS)}
        context.update(extras)
        for nested in (context["generated_reply"], context["verified_reply"], context["retrieved_evidence"][0]):
            nested.update(extras)
        self.assertNotIn("SECRET_SENTINEL", tool.render_context(context, 1, 50))
        for key in extras.keys() - set(tool.RUBRIC_DIMENSIONS):
            with self.assertRaises(ValueError):
                tool.append_rating(self.path, {**self.record(), key: "SECRET_SENTINEL"}, self.ids)
        self.assertFalse(self.path.exists())

    def test_progress_does_not_load_models_or_prompt_or_write(self):
        with patch.object(tool, "DEFAULT_RATINGS", self.path), patch("builtins.input", side_effect=AssertionError), patch("builtins.print") as output:
            self.assertEqual(tool.main(["--progress"]), 0)
        output.assert_called_once_with("Completed: 0/50; remaining: 50.")
        self.assertFalse(self.path.exists())

    def test_session_lock_prevents_concurrent_writers(self):
        with tool.rating_lock(self.path):
            with self.assertRaises(ValueError):
                with tool.rating_lock(self.path):
                    self.fail("Concurrent session allowed")
        self.assertFalse(self.path.with_suffix(".jsonl.lock").exists())

    def test_fixed_sample_unchanged_and_duplicate_samples_rejected(self):
        before = tool.DEFAULT_SAMPLE.read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(),
                         "c897b4c270f5ce1683ba1131684a0566c04bb4ea17623144931b3e2dc35cc744")
        samples = tool.load_sample()
        self.assertEqual(len({s["example_id"] for s in samples}), 50)
        self.assertEqual([s["example_id"] for s in samples],
                         [s["example_id"] for s in json.loads(before)["samples"]])
        self.assertEqual(tool.DEFAULT_SAMPLE.read_bytes(), before)
        duplicate = json.loads(before)
        duplicate["samples"][1] = duplicate["samples"][0]
        path = Path(self.temp.name) / "bad_sample.json"
        path.write_text(json.dumps(duplicate), encoding="utf-8")
        with self.assertRaises(ValueError):
            tool.load_sample(path)

    @patch("requests.sessions.Session.request", side_effect=AssertionError("External request forbidden"))
    def test_all_fifty_contexts_reconstruct_offline_without_ratings(self, request):
        from src.agent.controller import AgentController
        from src.agent.grounded_generator import create_reply_generator
        controller = AgentController(generator=create_reply_generator(provider="mock"))
        samples = tool.load_sample()
        for index, sample in enumerate(samples, 1):
            context = tool.build_context(sample, controller)
            rendered = tool.render_context(context, index, 50)
            self.assertIn(sample["example_id"], rendered)
            self.assertIn("Verified reply:", rendered)
        request.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_skip_on_first_dimension_proceeds_to_next_and_omits_skipped(self):
        output = []
        answers = iter(["s", "1", "2", "3", "4", "5", "test 1 note"])
        tool.rate_samples(self.samples[:2], self.path, self.rubric, self.context, lambda _: next(answers), output.append)
        ratings = tool.load_ratings(self.path, self.ids)
        self.assertNotIn("test_0", ratings)
        self.assertIn("test_1", ratings)
        self.assertEqual(ratings["test_1"]["human_note"], "test 1 note")
        self.assertNotIn("test_0", self.path.read_text(encoding="utf-8"))
        rendered = "\n".join(output)
        self.assertIn("Skipped this example without saving.", rendered)
        self.assertIn("Completed: 1/2", rendered)

    def test_skip_after_some_dimensions_discards_partial_record(self):
        output = []
        answers = iter(["4", "5", "skip", "1", "1", "1", "1", "1", ""])
        tool.rate_samples(self.samples[:2], self.path, self.rubric, self.context, lambda _: next(answers), output.append)
        ratings = tool.load_ratings(self.path, self.ids)
        self.assertNotIn("test_0", ratings)
        self.assertIn("test_1", ratings)
        self.assertNotIn("test_0", self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(ratings), 1)

    def test_skipped_example_is_absent_from_ratings_jsonl(self):
        answers = iter(["s", "skip"])
        tool.rate_samples(self.samples[:2], self.path, self.rubric, self.context, lambda _: next(answers), lambda _: None)
        self.assertFalse(self.path.exists())
        self.assertEqual(tool.load_ratings(self.path, self.ids), {})

    def test_session_proceeds_to_next_example_across_skips(self):
        visited = []
        def track_context(sample):
            visited.append(sample["example_id"])
            return self.context(sample)

        answers = iter(["S", "2", "3", "  skip  ", "5", "4", "3", "2", "1", ""])
        tool.rate_samples(self.samples[:3], self.path, self.rubric, track_context, lambda _: next(answers), lambda _: None)
        self.assertEqual(visited, ["test_0", "test_1", "test_2"])
        ratings = tool.load_ratings(self.path, self.ids)
        self.assertEqual(list(ratings.keys()), ["test_2"])
        lines = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["example_id"], "test_2")

    def test_q_still_exits_without_saving_or_advancing(self):
        visited = []
        def track_context(sample):
            visited.append(sample["example_id"])
            return self.context(sample)

        answers = iter(["4", "q"])
        output = []
        tool.rate_samples(self.samples[:2], self.path, self.rubric, track_context, lambda _: next(answers), output.append)
        self.assertEqual(visited, ["test_0"])
        self.assertFalse(self.path.exists())
        rendered = "\n".join(output)
        self.assertIn("Stopped. Completed ratings are preserved; any unfinished example will be shown again.", rendered)

    def test_normal_complete_rating_still_saves_exactly_once(self):
        answers = iter(["5", "4", "3", "2", "1", "All dimensions valid"])
        output = []
        tool.rate_samples(self.samples[:1], self.path, self.rubric, self.context, lambda _: next(answers), output.append)
        ratings = tool.load_ratings(self.path, self.ids)
        self.assertEqual(len(ratings), 1)
        record = ratings["test_0"]
        self.assertEqual(record["groundedness"], 5)
        self.assertEqual(record["correctness"], 4)
        self.assertEqual(record["relevance"], 3)
        self.assertEqual(record["helpfulness"], 2)
        self.assertEqual(record["tone"], 1)
        self.assertEqual(record["human_note"], "All dimensions valid")

        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)

        tool.rate_samples(self.samples[:1], self.path, self.rubric, self.context, lambda _: self.fail("Should not prompt"), output.append)
        lines_after = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines_after), 1)

    def test_help_and_prompt_text_displays_skip_instruction(self):
        self.assertIn("s/skip = skip this example without saving", tool.__doc__)
        prompts = []
        answers = iter(["s"])
        def read(prompt):
            prompts.append(prompt)
            return next(answers)
        output = []
        tool.rate_samples(self.samples[:1], self.path, self.rubric, self.context, read, output.append)
        self.assertTrue(any("s/skip = skip this example without saving" in p for p in prompts))
        rendered = "\n".join(output)
        self.assertIn("s/skip = skip this example without saving", rendered)


if __name__ == "__main__":
    unittest.main()
