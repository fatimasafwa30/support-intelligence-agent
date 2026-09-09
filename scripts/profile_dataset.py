"""Profile the full Customer Support on Twitter CSV in manageable chunks."""

from collections import Counter
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 50_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
LINK_COLUMNS = ("in_response_to_tweet_id", "response_tweet_id")


def percentage(part: int, whole: int) -> float:
    """Return a percentage safely when the total may be zero."""
    return (part / whole * 100) if whole else 0.0


def main() -> None:
    """Read the dataset chunk by chunk and print aggregate statistics."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    # Reading zero rows gives the schema without loading the dataset contents.
    columns = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()

    total_rows = 0
    inbound_count = 0
    outbound_count = 0
    all_authors: set[str] = set()
    inbound_authors: set[str] = set()
    outbound_authors: set[str] = set()
    outbound_author_counts: Counter[str] = Counter()
    link_missing_counts = {column: 0 for column in LINK_COLUMNS}
    link_present_counts = {column: 0 for column in LINK_COLUMNS}
    available_link_columns = {column: column in columns for column in LINK_COLUMNS}
    earliest_created_at = None
    latest_created_at = None
    minimum_text_length = None
    maximum_text_length = None
    total_text_length = 0
    text_value_count = 0

    print(f"Profiling file: {DATASET_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows\n")

    for chunk_number, chunk in enumerate(
        pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE), start=1
    ):
        total_rows += len(chunk)
        print(f"Processed chunk {chunk_number}: {total_rows:,} rows so far")

        if "inbound" in chunk.columns:
            # Normalizing strings handles both Boolean and text CSV representations.
            inbound_values = chunk["inbound"].astype("string").str.strip().str.lower()
            is_inbound = inbound_values.eq("true")
            is_outbound = inbound_values.eq("false")
            inbound_count += int(is_inbound.sum())
            outbound_count += int(is_outbound.sum())
        else:
            is_inbound = pd.Series(False, index=chunk.index)
            is_outbound = pd.Series(False, index=chunk.index)

        if "author_id" in chunk.columns:
            authors = chunk["author_id"].dropna().astype(str)
            all_authors.update(authors)
            inbound_authors.update(chunk.loc[is_inbound, "author_id"].dropna().astype(str))
            outbound_author_ids = chunk.loc[is_outbound, "author_id"].dropna().astype(str)
            outbound_authors.update(outbound_author_ids)
            outbound_author_counts.update(outbound_author_ids)

        for column in LINK_COLUMNS:
            if column in chunk.columns:
                missing_count = int(chunk[column].isna().sum())
                link_missing_counts[column] += missing_count
                link_present_counts[column] += len(chunk) - missing_count

        if "created_at" in chunk.columns:
            timestamps = pd.to_datetime(chunk["created_at"], errors="coerce", utc=True).dropna()
            if not timestamps.empty:
                chunk_earliest = timestamps.min()
                chunk_latest = timestamps.max()
                earliest_created_at = (
                    chunk_earliest
                    if earliest_created_at is None or chunk_earliest < earliest_created_at
                    else earliest_created_at
                )
                latest_created_at = (
                    chunk_latest
                    if latest_created_at is None or chunk_latest > latest_created_at
                    else latest_created_at
                )

        if "text" in chunk.columns:
            # Only this chunk's non-missing text values are held while measuring length.
            text_lengths = chunk["text"].dropna().astype(str).str.len()
            if not text_lengths.empty:
                chunk_minimum = int(text_lengths.min())
                chunk_maximum = int(text_lengths.max())
                minimum_text_length = (
                    chunk_minimum
                    if minimum_text_length is None or chunk_minimum < minimum_text_length
                    else minimum_text_length
                )
                maximum_text_length = (
                    chunk_maximum
                    if maximum_text_length is None or chunk_maximum > maximum_text_length
                    else maximum_text_length
                )
                total_text_length += int(text_lengths.sum())
                text_value_count += len(text_lengths)

    print("\n# DATASET PROFILE")
    print("\nA. Basic statistics")
    print(f"Total rows/tweets: {total_rows:,}")
    print(f"Number of columns: {len(columns)}")
    print(f"Column names: {columns}")

    print("\nB. Message statistics")
    print(f"Inbound=True messages: {inbound_count:,}")
    print(f"Inbound=False messages: {outbound_count:,}")
    print(f"Inbound percentage: {percentage(inbound_count, total_rows):.2f}%")
    print(f"Outbound percentage: {percentage(outbound_count, total_rows):.2f}%")

    print("\nC. Author statistics")
    print(f"Unique author_id values: {len(all_authors):,}")
    print(f"Unique inbound authors: {len(inbound_authors):,}")
    print(f"Unique outbound authors: {len(outbound_authors):,}")

    print("\nD. Brand candidates: top 20 outbound author_id values")
    print(f"{'author_id':<30} {'outbound tweets':>16} {'% of outbound':>15}")
    for author_id, count in outbound_author_counts.most_common(20):
        print(f"{author_id:<30} {count:>16,} {percentage(count, outbound_count):>14.2f}%")

    print("\nE. Conversation-link statistics")
    for column in LINK_COLUMNS:
        label = column.replace("_", " ")
        if available_link_columns[column]:
            print(f"Rows with missing {label}: {link_missing_counts[column]:,}")
            print(f"Rows with non-missing {label}: {link_present_counts[column]:,}")
        else:
            print(f"Column not found: {column}")

    print("\nF. Time coverage")
    print(f"Earliest created_at: {earliest_created_at}")
    print(f"Latest created_at: {latest_created_at}")

    print("\nG. Text statistics")
    print("Text lengths are computed for non-missing text values using streaming aggregates.")
    print(f"Minimum text length: {minimum_text_length}")
    print(f"Maximum text length: {maximum_text_length}")
    print(
        f"Mean text length: "
        f"{(total_text_length / text_value_count) if text_value_count else 0.0:.2f}"
    )


if __name__ == "__main__":
    main()
