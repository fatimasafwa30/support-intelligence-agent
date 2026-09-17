"""Independent manual ratings for the fixed audit sample; no judge/API calls.

Run normally to start or resume; use --progress for read-only completion counts.
Enter q/quit/exit or press Ctrl+C to stop. s/skip = skip this example without saving.
Only complete examples are saved.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reply_quality_schema import RUBRIC_DIMENSIONS, _validate_score

DEFAULT_SAMPLE = PROJECT_ROOT / "reports/human_agreement_sample_50.json"
DEFAULT_RATINGS = PROJECT_ROOT / "reports/human_agreement_ratings.jsonl"
DEFAULT_RUBRIC = PROJECT_ROOT / "configs/reply_evaluation.yaml"


def load_sample(path=DEFAULT_SAMPLE):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = data["samples"]
    if data["sample_size"] != 50 or len(samples) != 50:
        raise ValueError("Expected the fixed 50-example sample.")
    ids = [item["example_id"] for item in samples]
    if len(set(ids)) != 50:
        raise ValueError("Duplicate sample IDs are not allowed.")
    for item in samples:
        for key in ("example_id", "customer_query", "predicted_intent"):
            require_text(item[key])
    return samples


def require_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected non-empty text in evaluation context.")
    return value


def parse_score(value):
    if not isinstance(value, str) or value.strip() not in ("1", "2", "3", "4", "5"):
        raise ValueError("Enter one integer from 1 through 5; no default is provided.")
    return int(value.strip())


def validate_rating(record, sample_ids):
    required = {"example_id", *RUBRIC_DIMENSIONS}
    if not isinstance(record, dict) or not required <= record.keys():
        raise ValueError("Incomplete human rating record.")
    if record.keys() - required - {"human_note"}:
        raise ValueError("Unexpected fields in human rating record.")
    if record["example_id"] not in sample_ids:
        raise ValueError("Rating ID is not in the fixed sample.")
    for dim in RUBRIC_DIMENSIONS:
        _validate_score(dim, record[dim])
    if "human_note" in record and not isinstance(record["human_note"], str):
        raise ValueError("Human note must be text.")


def load_ratings(path, sample_ids):
    records = {}
    if not Path(path).exists():
        return records
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            validate_rating(record, sample_ids)
            if record["example_id"] in records:
                raise ValueError("Duplicate rating ID.")
        except (ValueError, TypeError, KeyError) as exc:
            # Do not echo arbitrary stored data (including possible labels/scores).
            raise ValueError(f"Invalid or duplicate rating at line {number}; file left unchanged.") from exc
        records[record["example_id"]] = record
    return records


@contextmanager
def rating_lock(path):
    """One writing session at a time; progress mode does not create a lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError("Another rating session holds the lock; close it before resuming.") from None
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def append_rating(path, record, sample_ids):
    """Call while holding rating_lock. Revalidate before every append."""
    validate_rating(record, sample_ids)
    if record["example_id"] in load_ratings(path, sample_ids):
        raise ValueError("This example already has a completed human rating.")
    # A valid last record without a newline must not be merged with the next one.
    with Path(path).open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        prefix = b""
        if size:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                prefix = b"\n"
        handle.write(prefix + (json.dumps(record, ensure_ascii=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def build_context(sample, controller):
    """Reconstruct the same offline reply path without reading Golden annotations."""
    state = controller.process_query(
        customer_message=sample["customer_query"],
        conversation_id=sample.get("conversation_id", ""),
    )
    if state.predicted_intent != sample["predicted_intent"]:
        raise ValueError("Predicted intent differs from the fixed sample; check frozen artifacts.")
    if not state.generated_reply or not state.generated_reply.reply_text:
        raise ValueError("The sampled example no longer produces a reply; check frozen artifacts.")
    return {
        "example_id": sample["example_id"],
        "customer_query": sample["customer_query"],
        "predicted_intent": sample["predicted_intent"],
        "retrieved_evidence": [e.to_dict() for e in state.retrieved_evidence],
        "generated_reply": {"reply_text": state.generated_reply.reply_text},
        # The controller's generated_reply is already passed through its URL guard,
        # matching the existing judge runner's verified-reply construction.
        "verified_reply": {"reply_text": state.generated_reply.reply_text},
    }


def render_context(context, index, total):
    """Display only explicitly allowed text fields, never whole state dictionaries."""
    lines = [f"\nExample {index}/{total}"]
    for key, label in (("example_id", "Example ID"), ("customer_query", "Customer query"),
                       ("predicted_intent", "Predicted intent")):
        lines.append(f"{label}: {require_text(context[key])}")
    lines.append("\nRetrieved historical evidence:")
    for index, evidence in enumerate(context["retrieved_evidence"], 1):
        lines.append(f"Evidence {index}:")
        for key in ("evidence_id", "past_customer_problem", "past_brand_resolution"):
            lines.append(f"  {key}: {require_text(evidence[key])}")
        urls = evidence.get("extracted_urls", [])
        if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
            raise ValueError("Invalid evidence URLs.")
        lines.append("  URLs: " + (", ".join(urls) or "None"))
    if not context["retrieved_evidence"]:
        lines.append("No evidence retrieved.")
    for key, label in (("generated_reply", "Generated reply"), ("verified_reply", "Verified reply")):
        lines.append(f"\n{label}:\n{require_text(context[key]['reply_text'])}")
    return "\n".join(lines)


class SkipExample(Exception):
    """Raised when the user skips rating the current example."""


def manual_input(prompt, read, allow_skip=False):
    value = read(prompt)
    lowered = value.strip().lower()
    if lowered in {"q", "quit", "exit"}:
        raise EOFError
    if allow_skip and lowered in {"s", "skip"}:
        raise SkipExample
    return value


def rate_samples(samples, path, rubric, context_provider, read=input, write=print):
    sample_ids = {item["example_id"] for item in samples}
    with rating_lock(path):
        completed = load_ratings(path, sample_ids)
        write(f"Completed: {len(completed)}/{len(samples)}. Enter q or press Ctrl+C to exit. s/skip = skip this example without saving.")
        try:
            for index, sample in enumerate(samples, 1):
                if sample["example_id"] in completed:
                    continue
                write(render_context(context_provider(sample), index, len(samples)))
                record = {"example_id": sample["example_id"]}
                try:
                    for dim in RUBRIC_DIMENSIONS:
                        definition = rubric["dimensions"][dim]
                        write(f"\n{definition['name']}: {definition['description']}")
                        for score in range(1, 6):
                            write(f"  {score}: {definition['scoring_guidance'][score]}")
                        while True:
                            try:
                                record[dim] = parse_score(
                                    manual_input(
                                        f"Your {dim} rating (1-5; s/skip = skip this example without saving): ",
                                        read,
                                        allow_skip=True,
                                    )
                                )
                                break
                            except ValueError:
                                write("Enter an integer from 1 through 5.")
                    note = manual_input("Optional human note (Enter to skip): ", read).strip()
                    if note:
                        record["human_note"] = note
                    append_rating(path, record, sample_ids)
                    completed[record["example_id"]] = record
                    write(f"Saved. Completed: {len(completed)}/{len(samples)}.")
                except SkipExample:
                    write("Skipped this example without saving.")
        except (KeyboardInterrupt, EOFError):
            write("Stopped. Completed ratings are preserved; any unfinished example will be shown again.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", action="store_true", help="Show completion counts without inference or rating prompts.")
    args = parser.parse_args(argv)
    try:
        samples = load_sample()
        if args.progress:
            count = len(load_ratings(DEFAULT_RATINGS, {s["example_id"] for s in samples}))
            print(f"Completed: {count}/50; remaining: {50-count}.")
            return 0
        rubric = yaml.safe_load(DEFAULT_RUBRIC.read_text(encoding="utf-8"))
        # Lazy imports: --progress/--help never load models. Explicit offline provider.
        from src.agent.controller import AgentController
        from src.agent.grounded_generator import create_reply_generator
        controller = AgentController(generator=create_reply_generator(provider="mock"))
        rate_samples(samples, DEFAULT_RATINGS, rubric, lambda s: build_context(s, controller))
        return 0
    except (ValueError, KeyError, TypeError, OSError):
        print("Unable to continue: check the fixed sample, rubric, frozen artifacts, ratings file, and session lock. No existing ratings were overwritten.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
