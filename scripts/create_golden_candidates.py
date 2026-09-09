"""Create a deterministic, stratified pool for manual Golden Set annotation."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import re
import sys

import yaml

# Direct execution (python scripts/create_golden_candidates.py) puts scripts/ rather
# than the project root on sys.path. Expose the package parent only in that case.
if __package__ in {None, ""}:
    PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.discover_intents import COMPILED_PATTERNS, INTENT_RULES, normalize_for_analysis


RANDOM_SEED = 20260909
TARGET_CANDIDATES = 250
CONTEXT_CHAR_LIMIT = 2_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "apple_conversations.jsonl"
TAXONOMY_PATH = PROJECT_ROOT / "configs" / "intents.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "golden" / "golden_candidates.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "golden_sampling_report.txt"

# The exploratory script used two separate account rules. Both are sampling hints for
# the final apple_id_account taxonomy entry, never automatically assigned labels.
DISCOVERY_TO_TAXONOMY = {
    "apple_id_account_access": "apple_id_account",
    "password_reset": "apple_id_account",
    "icloud_sync_storage": "icloud",
    "software_update": "software_update",
    "battery_power_charging": "battery_power",
    "connectivity": "connectivity",
    "device_hardware": "device_hardware",
    "app_store_downloads": "app_store",
    "billing_payments": "billing_payments",
    "subscriptions_services": "subscriptions_media",
    "setup_activation": "setup_activation",
    "repair_replacement": "repair_service",
    "security_privacy": "security_privacy",
    "orders_shipping": "orders_delivery",
    "how_to_information": "how_to_information",
}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def conversation_length_bucket(length: int) -> str:
    """Group contexts so each sampling stratum includes short and longer exchanges."""
    if length <= 2:
        return "short_2_or_fewer"
    if length <= 4:
        return "medium_3_to_4"
    return "long_5_or_more"


def build_context(tweets: list[dict[str, object]]) -> str:
    """Create readable context while limiting one CSV cell to a practical size."""
    context_parts = []
    for tweet in tweets:
        direction = "customer" if tweet.get("inbound") is True else "AppleSupport"
        text = re.sub(r"\s+", " ", str(tweet.get("text") or "")).strip()
        context_parts.append(f"[{direction}] {text}")
    return " | ".join(context_parts)[:CONTEXT_CHAR_LIMIT]


def hint_intents(text: str) -> list[str]:
    """Map exploratory regex matches to final-taxonomy sampling hints."""
    normalized = normalize_for_analysis(text)
    matches = []
    for rule in INTENT_RULES:
        if any(pattern.search(normalized) for pattern in COMPILED_PATTERNS[rule.name]):
            matches.append(DISCOVERY_TO_TAXONOMY[rule.name])
    return sorted(set(matches))


def risk_hint(intent_hints: list[str], taxonomy: dict[str, object]) -> str:
    """Use the highest referenced taxonomy risk only as an annotation priority hint."""
    if not intent_hints:
        return "medium"
    risks = [str(taxonomy[intent]["risk_level"]) for intent in intent_hints]
    return max(risks, key=lambda risk: RISK_ORDER[risk])


def difficulty_hint(text: str, intent_hints: list[str]) -> str:
    """Flag likely multi-intent, vague, or low-context examples for reviewers."""
    normalized = normalize_for_analysis(text)
    word_count = len(re.findall(r"[a-z]+", normalized))
    if word_count <= 4 or len(normalized) <= 25:
        return "low_context"
    if len(intent_hints) >= 2:
        return "likely_multi_intent"
    if not intent_hints:
        return "ambiguous_vague"
    return "specific_candidate"


def select_stratified(
    pool: list[dict[str, object]],
    selected_ids: set[str],
    requested_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    """Sample without replacement while round-robining conversation-length buckets."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in pool:
        if candidate["tweet_id"] not in selected_ids:
            grouped[str(candidate["length_bucket"])].append(candidate)
    for candidates in grouped.values():
        rng.shuffle(candidates)

    chosen = []
    bucket_order = ("short_2_or_fewer", "medium_3_to_4", "long_5_or_more")
    while len(chosen) < requested_count:
        added_this_round = False
        for bucket in bucket_order:
            if len(chosen) >= requested_count or not grouped[bucket]:
                continue
            candidate = grouped[bucket].pop()
            selected_ids.add(str(candidate["tweet_id"]))
            chosen.append(candidate)
            added_this_round = True
        if not added_this_round:
            break
    return chosen


def main() -> None:
    """Create candidate examples for human annotation without assigning final labels."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Conversation file not found: {INPUT_PATH}")
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {TAXONOMY_PATH}")

    with TAXONOMY_PATH.open("r", encoding="utf-8") as taxonomy_file:
        taxonomy_document = yaml.safe_load(taxonomy_file)
    taxonomy = taxonomy_document["intents"]

    candidates = []
    seen_tweet_ids: set[str] = set()
    # Stream JSONL to avoid loading the whole reconstructed-conversation file at once.
    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            conversation = json.loads(line)
            tweets = conversation.get("tweets", [])
            context = build_context(tweets)
            conversation_length = len(tweets)
            for tweet in tweets:
                if tweet.get("inbound") is not True:
                    continue
                tweet_id = str(tweet.get("tweet_id") or "")
                if not tweet_id or tweet_id in seen_tweet_ids:
                    continue
                seen_tweet_ids.add(tweet_id)

                text = str(tweet.get("text") or "")
                intent_hints = hint_intents(text)
                hint_value = "other_unclear" if not intent_hints else " | ".join(intent_hints)
                candidates.append(
                    {
                        "conversation_id": str(conversation.get("conversation_id") or ""),
                        "tweet_id": tweet_id,
                        "text": text,
                        "candidate_intent_hint": hint_value,
                        "risk_hint": risk_hint(intent_hints, taxonomy),
                        "difficulty_hint": difficulty_hint(text, intent_hints),
                        "conversation_length": conversation_length,
                        "conversation_context": context,
                        "length_bucket": conversation_length_bucket(conversation_length),
                        "intent_hints": intent_hints,
                    }
                )

    hint_counts = Counter(
        candidate["candidate_intent_hint"] for candidate in candidates
    )
    non_other_hints = [hint for hint in hint_counts if hint != "other_unclear"]
    # Treat the five least common specific sampling hints as rare. This is computed
    # from the current input, not hard-coded from an earlier report.
    rare_hints = set(sorted(non_other_hints, key=lambda hint: (hint_counts[hint], hint))[:5])

    rng = random.Random(RANDOM_SEED)
    selected_ids: set[str] = set()
    selected = []

    # Quotas overlap intentionally; de-duplication means each selected message appears once.
    sampling_strata = (
        ("rare_intent_candidates", lambda item: item["candidate_intent_hint"] in rare_hints, 45),
        ("high_risk_topics", lambda item: item["risk_hint"] in {"high", "critical"}, 55),
        ("likely_multi_intent", lambda item: item["difficulty_hint"] == "likely_multi_intent", 35),
        ("low_context", lambda item: item["difficulty_hint"] == "low_context", 30),
        ("ambiguous_vague", lambda item: item["difficulty_hint"] == "ambiguous_vague", 30),
        ("common_easy", lambda item: item["difficulty_hint"] == "specific_candidate", 55),
    )
    selected_by_stratum = Counter()
    for stratum_name, condition, quota in sampling_strata:
        additions = select_stratified(
            [candidate for candidate in candidates if condition(candidate)],
            selected_ids,
            quota,
            rng,
        )
        selected.extend(additions)
        selected_by_stratum[stratum_name] = len(additions)

    # Fill any unfilled quota from all remaining examples, still balanced by conversation length.
    if len(selected) < TARGET_CANDIDATES:
        additions = select_stratified(candidates, selected_ids, TARGET_CANDIDATES - len(selected), rng)
        selected.extend(additions)
        selected_by_stratum["deterministic_fill"] = len(additions)

    selected = selected[:TARGET_CANDIDATES]
    for number, candidate in enumerate(selected, start=1):
        candidate["id"] = f"golden_candidate_{number:04d}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_fields = (
        "id", "conversation_id", "tweet_id", "text", "candidate_intent_hint",
        "risk_hint", "difficulty_hint", "conversation_length", "conversation_context",
        "length_bucket", "intent_hints",
    )
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(selected)

    selected_hint_counts = Counter(candidate["candidate_intent_hint"] for candidate in selected)
    selected_risk_counts = Counter(candidate["risk_hint"] for candidate in selected)
    selected_difficulty_counts = Counter(candidate["difficulty_hint"] for candidate in selected)
    report_lines = [
        "# GOLDEN SET CANDIDATE SAMPLING REPORT",
        "",
        f"Total eligible inbound messages: {len(candidates):,}",
        f"Candidates selected: {len(selected):,}",
        f"Fixed random seed: {RANDOM_SEED}",
        "",
        "## Sampling strategy",
        "- Uses existing deterministic discovery regexes only as non-ground-truth sampling hints.",
        "- Prioritizes five least-common observed hint groups, high/critical risk topics, likely multi-intent messages, low-context messages, ambiguous messages, and specific/common examples.",
        "- Samples within each stratum round-robin across short, medium, and long conversations.",
        "- Removes duplicate tweet IDs and fills any remaining quota deterministically.",
        "",
        "## Selected by sampling stratum",
        *[f"- {name}: {count}" for name, count in sorted(selected_by_stratum.items())],
        "",
        "## Counts by candidate intent hint",
        *[f"- {hint}: {count}" for hint, count in sorted(selected_hint_counts.items())],
        "",
        "## Counts by risk hint",
        *[f"- {risk}: {count}" for risk, count in sorted(selected_risk_counts.items())],
        "",
        "## Counts by difficulty hint",
        *[f"- {difficulty}: {count}" for difficulty, count in sorted(selected_difficulty_counts.items())],
        "",
        "## Limitations",
        "- Candidate intent and risk hints are sampling aids, not final labels or ground truth.",
        "- Regex hints can miss implicit issues and can produce false positives.",
        "- Conversation context is capped for CSV readability and may omit distant messages in long conversations.",
        "- Manual annotation must determine the final single primary intent.",
    ]
    report = "\n".join(report_lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Candidates saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
