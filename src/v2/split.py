"""
split.py — Strictly temporal train/val/test split.

Train: 2024-01-01 to 2024-08-31
Val:   2024-09-01 to 2024-10-31
Test:  2024-11-01 to 2024-12-31

No shuffling.  No random sampling.  Date-based only.
"""
import pandas as pd

from .config import TRAIN_END, VAL_START, VAL_END, TEST_START


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "fl_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train/val/test by date.

    Args:
        df: DataFrame with a date column (string or datetime).
        date_col: Name of the date column.

    Returns:
        (train_df, val_df, test_df) — disjoint, date-contiguous subsets.
    """
    # Ensure date column is datetime
    dates = pd.to_datetime(df[date_col])

    train_mask = dates <= TRAIN_END
    val_mask   = (dates >= VAL_START) & (dates <= VAL_END)
    test_mask  = dates >= TEST_START

    train = df[train_mask].copy()
    val   = df[val_mask].copy()
    test  = df[test_mask].copy()

    # Sanity checks
    total = len(train) + len(val) + len(test)
    assert total == len(df), (
        f"Split produced {total} rows but input has {len(df)} — "
        "some rows fell outside all date ranges!"
    )

    # Report
    for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        date_range = f"{pd.to_datetime(split_df[date_col]).min().date()} -> "
        date_range += f"{pd.to_datetime(split_df[date_col]).max().date()}"
        print(f"[split] {name:>5s}: {len(split_df):>10,} rows  ({date_range})")

    return train, val, test
