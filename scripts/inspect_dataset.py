"""Inspect a small sample of the Customer Support on Twitter dataset."""

from pathlib import Path

import pandas as pd


SAMPLE_ROWS = 20
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"


def main() -> None:
    """Load and display a small, read-only sample of the dataset."""
    print(f"Inspecting file: {DATASET_PATH}")

    # nrows keeps this inspection from loading the full CSV into memory.
    sample = pd.read_csv(DATASET_PATH, nrows=SAMPLE_ROWS)

    print("\nColumn names:")
    print(sample.columns.tolist())
    print(f"\nNumber of columns: {len(sample.columns)}")

    print("\nFirst 5 rows:")
    print(sample.head().to_string(index=False))

    print("\nData types:")
    print(sample.dtypes.to_string())

    print(f"\nMissing-value counts in the {len(sample)} sampled rows:")
    print(sample.isna().sum().to_string())

    if "inbound" in sample.columns:
        print(f"\n'inbound' value counts in the {len(sample)} sampled rows:")
        print(sample["inbound"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
