"""Discover candidate AppleSupport intent themes with transparent text analysis."""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "apple_conversations.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "intent_discovery.txt"
EXAMPLES_PER_INTENT = 5
TOP_TERMS = 30

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "for", "from", "get", "got", "have", "help", "i", "if", "in", "is",
    "it", "just", "me", "my", "of", "on", "or", "please", "so", "that", "the",
    "this", "to", "was", "we", "with", "you", "your", "apple", "applesupport",
}


@dataclass(frozen=True)
class IntentRule:
    """A candidate taxonomy suggestion and its transparent matching patterns."""

    name: str
    description: str
    keywords: tuple[str, ...]
    patterns: tuple[str, ...]


# These are exploratory candidate suggestions, not ground-truth labels. A message may
# match multiple rules when it contains multiple customer-support problems.
INTENT_RULES = (
    IntentRule("apple_id_account_access", "Trouble signing in to or accessing an Apple ID/account.",
               ("apple id", "sign in", "account locked"),
               (r"\bapple\s*id\b", r"\bsign\s*in\b", r"\blog(?:in|ged)\b", r"\baccount\s*(?:locked|access)\b")),
    IntentRule("password_reset", "Password, verification, or account-recovery requests.",
               ("password", "reset", "verification code", "recovery"),
               (r"\bpassword\b", r"\breset\b", r"\bverification\b", r"\brecover(?:y)?\b")),
    IntentRule("icloud_sync_storage", "iCloud syncing, backup, storage, or photo/data access issues.",
               ("icloud", "backup", "storage", "sync"),
               (r"\bicloud\b", r"\bbackup\b", r"\bstorage\b", r"\bsync(?:ing)?\b")),
    IntentRule("billing_payments", "Charges, refunds, payment methods, or billing questions.",
               ("charged", "refund", "payment", "billing"),
               (r"\bcharg(?:e|ed|ing)\b", r"\brefund\b", r"\bpayment\b", r"\bbill(?:ing)?\b", r"\bcredit\s*card\b")),
    IntentRule("subscriptions_services", "Subscription or Apple media-service problems.",
               ("subscription", "apple music", "itunes", "cancel"),
               (r"\bsubscription\b", r"\bapple\s+music\b", r"\bitunes\b", r"\bcancel(?:led|lation)?\b")),
    IntentRule("app_store_downloads", "App Store, app download, install, or purchase problems.",
               ("app store", "download", "install", "app"),
               (r"\bapp\s*store\b", r"\bdownload(?:ing|ed)?\b", r"\binstall(?:ing|ed)?\b", r"\bapp\s+(?:won't|will not|not)\b")),
    IntentRule("software_update", "iOS/macOS update, upgrade, or software-version issues.",
               ("update", "ios", "upgrade", "software"),
               (r"\bupdate(?:d|ing)?\b", r"\bios\b", r"\bmacos\b", r"\bupgrade\b", r"\bsoftware\b")),
    IntentRule("setup_activation", "Device setup, activation, or initial configuration problems.",
               ("activate", "activation", "setup", "set up"),
               (r"\bactivat(?:e|ed|ing|ion)\b", r"\bset\s*up\b", r"\bconfigure\b", r"\bnew\s+(?:iphone|ipad|mac)\b")),
    IntentRule("battery_power_charging", "Battery life, charging, or power-on issues.",
               ("battery", "charging", "charger", "won't turn on"),
               (r"\bbattery\b", r"\bcharg(?:e|ing|er)\b", r"\bpower\b", r"\bturn\s+on\b")),
    IntentRule("connectivity", "Wi-Fi, Bluetooth, cellular, network, or connection issues.",
               ("wifi", "bluetooth", "cellular", "connection"),
               (r"\bwi[ -]?fi\b", r"\bbluetooth\b", r"\bcellular\b", r"\bnetwork\b", r"\bconnect(?:ion|ing)?\b")),
    IntentRule("device_hardware", "Device physical-function, display, camera, or audio problems.",
               ("screen", "camera", "speaker", "broken"),
               (r"\bscreen\b", r"\bcamera\b", r"\bspeaker\b", r"\bheadphones?\b", r"\bbroken\b")),
    IntentRule("repair_replacement", "Repair, replacement, warranty, or service appointment requests.",
               ("repair", "replace", "warranty", "genius bar"),
               (r"\brepair\b", r"\breplace(?:ment)?\b", r"\bwarranty\b", r"\bgenius\s+bar\b", r"\bservice\b")),
    IntentRule("security_privacy", "Security, privacy, lost-device, or unauthorized-access concerns.",
               ("security", "privacy", "hacked", "lost phone"),
               (r"\bsecurity\b", r"\bprivacy\b", r"\bhack(?:ed)?\b", r"\blost\s+(?:iphone|phone|device)\b", r"\bunauthori[sz]ed\b")),
    IntentRule("orders_shipping", "Order status, delivery, shipment, or tracking questions.",
               ("order", "shipping", "delivery", "tracking"),
               (r"\border\b", r"\bship(?:ping|ped)?\b", r"\bdeliver(?:y|ed)?\b", r"\btrack(?:ing)?\b")),
    IntentRule("how_to_information", "Requests for instructions or product information rather than a clear fault.",
               ("how do i", "how can i", "where can i", "can i"),
               (r"\bhow\s+(?:do|can)\s+i\b", r"\bwhere\s+can\s+i\b", r"\bcan\s+i\b", r"\bwhat\s+is\b")),
)

# Required by create_golden_candidates.py. Compiling once keeps matching deterministic.
COMPILED_PATTERNS = {
    rule.name: tuple(re.compile(pattern, re.IGNORECASE) for pattern in rule.patterns)
    for rule in INTENT_RULES
}


def normalize_for_analysis(text: str) -> str:
    """Normalize text in memory only; original JSONL tweet text is not changed."""
    normalized = URL_PATTERN.sub(" ", text.lower())
    normalized = MENTION_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def meaningful_tokens(text: str) -> list[str]:
    """Return simple meaningful tokens after explicit stopword removal."""
    return [token for token in TOKEN_PATTERN.findall(text) if token not in STOPWORDS]


def top_items(counter: Counter[str]) -> list[tuple[str, int]]:
    """Make frequency-table ties deterministic."""
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:TOP_TERMS]


def main() -> None:
    """Create an exploratory report from inbound AppleSupport customer messages."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Conversation file not found: {INPUT_PATH}")

    message_count = 0
    customer_authors: set[str] = set()
    character_lengths: list[int] = []
    word_lengths: list[int] = []
    word_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    trigram_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    rule_examples = {rule.name: [] for rule in INTENT_RULES}
    other_examples: list[str] = []
    other_count = 0

    # Stream JSONL and inspect only inbound messages. The source file is never written.
    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            for tweet in json.loads(line).get("tweets", []):
                if tweet.get("inbound") is not True:
                    continue
                original_text = str(tweet.get("text") or "")
                normalized_text = normalize_for_analysis(original_text)
                tokens = meaningful_tokens(normalized_text)
                message_count += 1
                character_lengths.append(len(original_text))
                word_lengths.append(len(TOKEN_PATTERN.findall(normalized_text)))
                if tweet.get("author_id") is not None:
                    customer_authors.add(str(tweet["author_id"]))
                word_counts.update(tokens)
                bigram_counts.update(" ".join(tokens[index:index + 2]) for index in range(len(tokens) - 1))
                trigram_counts.update(" ".join(tokens[index:index + 3]) for index in range(len(tokens) - 2))

                matched_names = []
                for rule in INTENT_RULES:
                    if any(pattern.search(normalized_text) for pattern in COMPILED_PATTERNS[rule.name]):
                        matched_names.append(rule.name)
                        rule_counts[rule.name] += 1
                        if len(rule_examples[rule.name]) < EXAMPLES_PER_INTENT:
                            rule_examples[rule.name].append(original_text)
                if not matched_names:
                    other_count += 1
                    if len(other_examples) < EXAMPLES_PER_INTENT:
                        other_examples.append(original_text)

    lines = [
        "# APPLE SUPPORT INTENT DISCOVERY", "",
        "This report contains candidate taxonomy suggestions, not ground-truth labels.",
        "Messages may match multiple candidate rules when they contain multiple problems.",
        "Text normalization is used only in memory for analysis; the source JSONL is unchanged.", "",
        "## Customer-message coverage",
        f"Customer messages analyzed: {message_count:,}",
        f"Unique customer authors: {len(customer_authors):,}", "",
        "## Message-length statistics",
        f"Character length — min: {min(character_lengths) if character_lengths else 0}, max: {max(character_lengths) if character_lengths else 0}, mean: {mean(character_lengths) if character_lengths else 0:.2f}, median: {median(character_lengths) if character_lengths else 0}",
        f"Word length — min: {min(word_lengths) if word_lengths else 0}, max: {max(word_lengths) if word_lengths else 0}, mean: {mean(word_lengths) if word_lengths else 0:.2f}, median: {median(word_lengths) if word_lengths else 0}", "",
        "## Most common meaningful words",
        *[f"- {term}: {count}" for term, count in top_items(word_counts)], "",
        "## Most common meaningful bigrams",
        *[f"- {phrase}: {count}" for phrase, count in top_items(bigram_counts)], "",
        "## Most common meaningful trigrams",
        *[f"- {phrase}: {count}" for phrase, count in top_items(trigram_counts)], "",
        "## Candidate taxonomy suggestions",
    ]
    for rule in INTENT_RULES:
        lines.extend([
            "", f"### {rule.name}", f"Description: {rule.description}",
            f"Approximate matching messages: {rule_counts[rule.name]:,}",
            f"Motivating keywords/phrases: {', '.join(rule.keywords)}", "Examples:",
            *([f"  - {example}" for example in rule_examples[rule.name]] or ["  - No matching examples found."]),
        ])
    lines.extend([
        "", "### other_unclear",
        "Description: Greetings, generic dissatisfaction, bare help requests, ambiguous text, or messages with no specific transparent rule match.",
        f"Approximate matching messages: {other_count:,}",
        "Motivating keywords/phrases: no confident specific match", "Examples:",
        *([f"  - {example}" for example in other_examples] or ["  - No matching examples found."]), "",
        "## Methodology and limitations",
        "- Counts use deterministic regular-expression rules and are exploratory estimates.",
        "- Common terms and n-grams help suggest recurring themes, but frequency is not a label.",
        "- A message can match multiple candidate categories; no single-label claim is made.",
        "- Incomplete text, sarcasm, implicit requests, and missing response context can reduce rule coverage.",
    ])

    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
