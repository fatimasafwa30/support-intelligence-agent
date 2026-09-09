"""Build temporally and participant-coherent AppleSupport conversations."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import median

import pandas as pd


CHUNK_SIZE = 50_000
MAX_GAP_DAYS = 7
BRAND = "AppleSupport"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "apple_conversations.jsonl"
REQUIRED_COLUMNS = (
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
)


def ids_from_cell(value: object) -> list[str]:
    """Return all IDs from a possibly multi-value relationship cell."""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def value_or_none(value: object) -> str | None:
    """Convert pandas missing values into JSON-friendly None values."""
    return None if pd.isna(value) else str(value)


def record_from_row(row: object) -> dict[str, object]:
    """Keep the required output fields for one tweet record."""
    inbound_value = value_or_none(row.inbound)
    return {
        "tweet_id": value_or_none(row.tweet_id),
        "author_id": value_or_none(row.author_id),
        "inbound": inbound_value.lower() == "true" if inbound_value else None,
        "created_at": value_or_none(row.created_at),
        "text": value_or_none(row.text),
        "in_response_to_tweet_id": value_or_none(row.in_response_to_tweet_id),
        "response_tweet_id": value_or_none(row.response_tweet_id),
    }


def timestamp_value(record: dict[str, object]) -> int | None:
    """Return a UTC timestamp value, or None for an invalid source timestamp."""
    timestamp = pd.to_datetime(record["created_at"], errors="coerce", utc=True)
    return None if pd.isna(timestamp) else int(timestamp.value)


def sorted_tweets(tweet_ids: set[str], records: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """Sort records by time, then by ID for deterministic output."""
    return sorted(
        (records[tweet_id] for tweet_id in tweet_ids),
        key=lambda record: (
            timestamp_value(record) if timestamp_value(record) is not None else 2**63 - 1,
            str(record["tweet_id"]),
        ),
    )


def find_components(graph: dict[str, set[str]]) -> list[set[str]]:
    """Find components in one customer's local AppleSupport interaction graph."""
    visited: set[str] = set()
    components: list[set[str]] = []
    for start_id in sorted(graph):
        if start_id in visited:
            continue
        component: set[str] = set()
        pending = [start_id]
        visited.add(start_id)
        while pending:
            current_id = pending.pop()
            component.add(current_id)
            for neighbor_id in graph[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    pending.append(neighbor_id)
        components.append(component)
    return components


def split_at_large_time_gaps(tweets: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """Split chronological tweets whenever consecutive messages exceed MAX_GAP_DAYS."""
    if not tweets:
        return []

    maximum_gap_ns = MAX_GAP_DAYS * 24 * 60 * 60 * 1_000_000_000
    segments = [[tweets[0]]]
    previous_timestamp = timestamp_value(tweets[0])
    for tweet in tweets[1:]:
        current_timestamp = timestamp_value(tweet)
        if (
            previous_timestamp is None
            or current_timestamp is None
            or current_timestamp - previous_timestamp > maximum_gap_ns
        ):
            segments.append([tweet])
        else:
            segments[-1].append(tweet)
        previous_timestamp = current_timestamp
    return segments


def read_chunks():
    """Yield CSV chunks while preserving large tweet IDs as strings."""
    return pd.read_csv(
        DATASET_PATH, usecols=REQUIRED_COLUMNS, dtype="string", chunksize=CHUNK_SIZE
    )


def main() -> None:
    """Build AppleSupport-centered conversations and save them as JSONL."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    columns = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()
    missing_columns = set(REQUIRED_COLUMNS) - set(columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    print(f"Building coherent {BRAND} conversations from: {DATASET_PATH}")
    print("Pass 1 of 2: collecting AppleSupport tweets and directed response links...")
    apple_tweet_ids: set[str] = set()
    records: dict[str, dict[str, object]] = {}
    directed_edges: set[tuple[str, str]] = set()
    referenced_customer_ids: set[str] = set()

    for chunk_number, chunk in enumerate(read_chunks(), start=1):
        apple_rows = chunk.loc[
            chunk["author_id"].eq(BRAND)
            & chunk["inbound"].fillna("").str.strip().str.lower().eq("false")
        ]
        for row in apple_rows.itertuples(index=False):
            apple_tweet_id = value_or_none(row.tweet_id)
            if apple_tweet_id is None:
                continue
            apple_tweet_ids.add(apple_tweet_id)
            records[apple_tweet_id] = record_from_row(row)

            # in_response_to means parent customer tweet -> Apple reply.
            for parent_id in ids_from_cell(row.in_response_to_tweet_id):
                directed_edges.add((parent_id, apple_tweet_id))
                referenced_customer_ids.add(parent_id)
            # response_tweet_id means Apple tweet -> child customer reply.
            for child_id in ids_from_cell(row.response_tweet_id):
                directed_edges.add((apple_tweet_id, child_id))
                referenced_customer_ids.add(child_id)
        print(f"  Processed chunk {chunk_number}")

    print("Pass 2 of 2: finding customer tweets directly linked to AppleSupport...")
    for chunk_number, chunk in enumerate(read_chunks(), start=1):
        inbound_rows = chunk.loc[chunk["inbound"].fillna("").str.strip().str.lower().eq("true")]
        for row in inbound_rows.itertuples(index=False):
            customer_tweet_id = value_or_none(row.tweet_id)
            if customer_tweet_id is None:
                continue
            parent_ids = ids_from_cell(row.in_response_to_tweet_id)
            child_ids = ids_from_cell(row.response_tweet_id)
            links_to_apple = any(tweet_id in apple_tweet_ids for tweet_id in parent_ids + child_ids)
            if customer_tweet_id not in referenced_customer_ids and not links_to_apple:
                continue

            records[customer_tweet_id] = record_from_row(row)
            # For an inbound tweet, these relationship fields point in opposite directions.
            for apple_tweet_id in parent_ids:
                if apple_tweet_id in apple_tweet_ids:
                    directed_edges.add((apple_tweet_id, customer_tweet_id))
            for apple_tweet_id in child_ids:
                if apple_tweet_id in apple_tweet_ids:
                    directed_edges.add((customer_tweet_id, apple_tweet_id))
        print(f"  Processed chunk {chunk_number}")

    # Unrestricted connected components failed because a weak or old link could join
    # unrelated customers. Retain only direct AppleSupport/customer response edges.
    maximum_gap_ns = MAX_GAP_DAYS * 24 * 60 * 60 * 1_000_000_000
    edges_by_customer: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for parent_id, child_id in sorted(directed_edges):
        if parent_id not in records or child_id not in records:
            continue
        parent_is_apple = parent_id in apple_tweet_ids
        child_is_apple = child_id in apple_tweet_ids
        if parent_is_apple == child_is_apple:
            continue

        customer_tweet_id = child_id if parent_is_apple else parent_id
        customer_record = records[customer_tweet_id]
        if customer_record["inbound"] is not True or customer_record["author_id"] is None:
            continue

        parent_timestamp = timestamp_value(records[parent_id])
        child_timestamp = timestamp_value(records[child_id])
        if (
            parent_timestamp is None
            or child_timestamp is None
            or child_timestamp < parent_timestamp
            or child_timestamp - parent_timestamp > maximum_gap_ns
        ):
            continue

        # Participant coherence: a graph belongs to one customer, plus AppleSupport.
        edges_by_customer[str(customer_record["author_id"])].add((parent_id, child_id))

    conversation_tweets: list[list[dict[str, object]]] = []
    for customer_author_id in sorted(edges_by_customer):
        customer_graph: dict[str, set[str]] = defaultdict(set)
        for parent_id, child_id in sorted(edges_by_customer[customer_author_id]):
            customer_graph[parent_id].add(child_id)
            customer_graph[child_id].add(parent_id)

        for component in find_components(customer_graph):
            tweets = sorted_tweets(component, records)
            # Temporal coherence is necessary: even a linked component is split if it
            # contains a gap above the configured threshold.
            for segment in split_at_large_time_gaps(tweets):
                has_apple = any(tweet["author_id"] == BRAND and tweet["inbound"] is False for tweet in segment)
                has_customer = any(tweet["inbound"] is True for tweet in segment)
                if has_apple and has_customer:
                    conversation_tweets.append(segment)

    # Links in TWCS may be missing or incomplete. This conservative approach can split
    # a real conversation, but avoids inventing unsupported joins across accounts or time.
    conversation_tweets.sort(
        key=lambda tweets: (
            timestamp_value(tweets[0]) if timestamp_value(tweets[0]) is not None else 2**63 - 1,
            str(tweets[0]["tweet_id"]),
        )
    )
    conversations = [
        {"conversation_id": f"apple_{number:06d}", "brand": BRAND, "tweets": tweets}
        for number, tweets in enumerate(conversation_tweets, start=1)
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for conversation in conversations:
            output_file.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    lengths = [len(conversation["tweets"]) for conversation in conversations]
    all_tweets = [tweet for conversation in conversations for tweet in conversation["tweets"]]
    inbound_count = sum(tweet["inbound"] is True for tweet in all_tweets)
    outbound_count = sum(tweet["inbound"] is False for tweet in all_tweets)
    customer_authors = {
        tweet["author_id"] for tweet in all_tweets
        if tweet["inbound"] is True and tweet["author_id"] is not None
    }
    spans_ns = [
        timestamp_value(conversation["tweets"][-1]) - timestamp_value(conversation["tweets"][0])
        for conversation in conversations
    ]
    customer_counts = [
        len({tweet["author_id"] for tweet in conversation["tweets"] if tweet["inbound"] is True and tweet["author_id"] is not None})
        for conversation in conversations
    ]
    day_ns = 24 * 60 * 60 * 1_000_000_000

    print("\n# APPLE SUPPORT CONVERSATION SUMMARY")
    print(f"Saved conversations to: {OUTPUT_PATH}")
    print(f"Total reconstructed conversations: {len(conversations):,}")
    print(f"Total tweets in conversations: {len(all_tweets):,}")
    print(f"Conversations with at least 2 tweets: {sum(length >= 2 for length in lengths):,}")
    print(f"Conversations with at least 3 tweets: {sum(length >= 3 for length in lengths):,}")
    print(f"Average tweets per conversation: {sum(lengths) / len(lengths) if lengths else 0:.2f}")
    print(f"Median tweets per conversation: {median(lengths) if lengths else 0}")
    print(f"Minimum conversation length: {min(lengths) if lengths else 0}")
    print(f"Maximum conversation length: {max(lengths) if lengths else 0}")
    print(f"Inbound tweets: {inbound_count:,}")
    print(f"Outbound tweets: {outbound_count:,}")
    print(f"Unique customer authors: {len(customer_authors):,}")

    print("\n# SANITY CHECKS")
    print(f"Conversations spanning > 1 day: {sum(span > day_ns for span in spans_ns):,}")
    print(f"Conversations spanning > 7 days: {sum(span > 7 * day_ns for span in spans_ns):,}")
    print(f"Conversations spanning > 30 days: {sum(span > 30 * day_ns for span in spans_ns):,}")
    print(f"Conversations with more than 3 customer authors: {sum(count > 3 for count in customer_counts):,}")
    maximum_span_ns = max(spans_ns) if spans_ns else 0
    print(f"Maximum time span: {pd.Timedelta(maximum_span_ns, unit='ns')}")

    print("\n# EXAMPLE CONVERSATIONS")
    for conversation in conversations[:10]:
        tweets = conversation["tweets"]
        participant_count = len({tweet["author_id"] for tweet in tweets if tweet["author_id"] is not None})
        time_span_ns = timestamp_value(tweets[-1]) - timestamp_value(tweets[0])
        print(f"\n{conversation['conversation_id']}")
        print(f"  Conversation length: {len(tweets)}")
        print(f"  Distinct participants: {participant_count}")
        print(f"  Time span: {pd.Timedelta(time_span_ns, unit='ns')}")
        for tweet in tweets:
            text = (tweet["text"] or "").replace("\n", " ")
            preview = text[:140] + ("..." if len(text) > 140 else "")
            direction = "inbound" if tweet["inbound"] else "outbound"
            print(f"  [{tweet['created_at']}] {direction} {tweet['author_id']}: {preview}")


if __name__ == "__main__":
    main()
