"""Compare candidate support brands using direct tweet-response relationships."""

from collections import defaultdict
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 50_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
CANDIDATE_BRANDS = (
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
)
REQUIRED_COLUMNS = (
    "tweet_id",
    "author_id",
    "inbound",
    "response_tweet_id",
    "in_response_to_tweet_id",
)


def ids_from_cell(value: object) -> list[str]:
    """Return one or more tweet IDs from a possibly missing CSV cell."""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def add_edge(graph: dict[str, set[str]], first_id: str, second_id: str) -> None:
    """Add an undirected direct-response link to one brand's small graph."""
    graph[first_id].add(second_id)
    graph[second_id].add(first_id)


def connected_components(graph: dict[str, set[str]]) -> int:
    """Count direct-link components without expanding to unrelated tweets."""
    visited: set[str] = set()
    component_count = 0

    for start_id in graph:
        if start_id in visited:
            continue

        component_count += 1
        pending = [start_id]
        visited.add(start_id)

        while pending:
            current_id = pending.pop()
            for neighbor_id in graph[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    pending.append(neighbor_id)

    return component_count


def main() -> None:
    """Profile direct customer interactions for each specified brand candidate."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    columns = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()
    missing_columns = set(REQUIRED_COLUMNS) - set(columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    brand_tweet_ids = {brand: set() for brand in CANDIDATE_BRANDS}
    outbound_message_counts = {brand: 0 for brand in CANDIDATE_BRANDS}

    print(f"Comparing candidate brands in: {DATASET_PATH}")
    print("Pass 1 of 2: collecting candidate outbound tweet IDs...")

    # String dtype preserves large tweet IDs exactly while reading one chunk at a time.
    csv_chunks = pd.read_csv(
        DATASET_PATH,
        usecols=REQUIRED_COLUMNS,
        dtype="string",
        chunksize=CHUNK_SIZE,
    )
    for chunk_number, chunk in enumerate(csv_chunks, start=1):
        inbound_values = chunk["inbound"].str.strip().str.lower()
        is_outbound = inbound_values.eq("false")

        for brand in CANDIDATE_BRANDS:
            brand_rows = chunk.loc[is_outbound & chunk["author_id"].eq(brand)]
            outbound_message_counts[brand] += len(brand_rows)
            brand_tweet_ids[brand].update(brand_rows["tweet_id"].dropna())

        print(f"  Processed chunk {chunk_number}")

    # This lookup lets the second pass find a candidate brand in constant time.
    brand_for_tweet_id = {
        tweet_id: brand
        for brand, tweet_ids in brand_tweet_ids.items()
        for tweet_id in tweet_ids
    }
    direct_customer_tweet_ids = {brand: set() for brand in CANDIDATE_BRANDS}
    customer_authors = {brand: set() for brand in CANDIDATE_BRANDS}
    direct_interaction_graphs = {
        brand: defaultdict(set) for brand in CANDIDATE_BRANDS
    }

    print("Pass 2 of 2: finding inbound tweets directly linked to candidate tweets...")
    csv_chunks = pd.read_csv(
        DATASET_PATH,
        usecols=REQUIRED_COLUMNS,
        dtype="string",
        chunksize=CHUNK_SIZE,
    )
    for chunk_number, chunk in enumerate(csv_chunks, start=1):
        inbound_rows = chunk.loc[chunk["inbound"].str.strip().str.lower().eq("true")]

        for row in inbound_rows.itertuples(index=False):
            customer_tweet_id = row.tweet_id
            if pd.isna(customer_tweet_id):
                continue

            # A direct link may be expressed in either response relationship column.
            linked_tweet_ids = ids_from_cell(row.in_response_to_tweet_id)
            linked_tweet_ids.extend(ids_from_cell(row.response_tweet_id))

            for linked_tweet_id in set(linked_tweet_ids):
                brand = brand_for_tweet_id.get(linked_tweet_id)
                if brand is None:
                    continue

                direct_customer_tweet_ids[brand].add(customer_tweet_id)
                if not pd.isna(row.author_id):
                    customer_authors[brand].add(row.author_id)
                add_edge(
                    direct_interaction_graphs[brand],
                    customer_tweet_id,
                    linked_tweet_id,
                )

        print(f"  Processed chunk {chunk_number}")

    print("\n# CANDIDATE BRAND COMPARISON")
    print(
        f"{'brand':<20} {'outbound':>10} {'direct inbound':>16} "
        f"{'customers':>10} {'interaction tweets':>20} {'est. conversations':>20}"
    )
    for brand in CANDIDATE_BRANDS:
        graph = direct_interaction_graphs[brand]
        print(
            f"{brand:<20} "
            f"{outbound_message_counts[brand]:>10,} "
            f"{len(direct_customer_tweet_ids[brand]):>16,} "
            f"{len(customer_authors[brand]):>10,} "
            f"{len(graph):>20,} "
            f"{connected_components(graph):>20,}"
        )

    print("\nHow interactions are identified:")
    print(
        "A customer tweet is counted only when either response column explicitly "
        "references an outbound tweet from one of the candidate brands."
    )
    print("Interaction tweets are only the nodes in those direct response links.")

    print("\nLimitations before brand selection:")
    print(
        "The conversation estimate counts connected components of direct links, not "
        "fully reconstructed conversations. Missing links, multi-part response fields, "
        "and longer reply chains can split or combine real conversations."
    )
    print(
        "Tweet volume alone is insufficient because it does not show customer reach, "
        "link quality, conversation completeness, or suitability for the later task."
    )
    print("No brand is selected by this script.")


if __name__ == "__main__":
    main()
