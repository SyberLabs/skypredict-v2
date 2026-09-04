"""
data_loader.py — Load and filter the raw BTS 2024 CSV.

Handles:
  - Loading only the columns we need (LOAD_COLS)
  - Dtype optimization (category, int16/32 downcasting)
  - Basic filter: drop cancelled, diverted, missing ARR_DELAY
  - Cache the cleaned result to parquet for fast reloads
"""
import os
import numpy as np
import pandas as pd

from .config import (
    RAW_CSV, SAMPLE_CSV, CLEAN_PARQUET,
    DATA_PROC_DIR, LOAD_COLS, RANDOM_SEED,
)


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types and convert strings to categoricals for memory."""
    for col in ["op_unique_carrier", "origin", "dest"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in ["day_of_week", "month", "day_of_month"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in ["crs_dep_time", "crs_arr_time"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(np.int16)

    if "cancelled" in df.columns:
        df["cancelled"] = df["cancelled"].astype(np.int8)
    if "diverted" in df.columns:
        df["diverted"] = df["diverted"].astype(np.int8)

    return df


def load_raw(sample: bool = False) -> pd.DataFrame:
    """
    Load the raw BTS 2024 CSV, keeping only LOAD_COLS columns.

    Args:
        sample: If True, use the small sample CSV for fast testing.

    Returns:
        DataFrame with raw flight data, dtypes optimized.
    """
    path = SAMPLE_CSV if sample else RAW_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Download via: kaggle datasets download hrishitpatil/flight-data-2024\n"
            "Then unzip to data/raw/"
        )

    print(f"[data_loader] Loading {'sample' if sample else 'full'} dataset: {path}")

    # Check which requested columns actually exist
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols_present = [c for c in LOAD_COLS if c in header]
    cols_missing = [c for c in LOAD_COLS if c not in header]
    if cols_missing:
        print(f"[data_loader] WARNING: columns not in CSV (skipping): {cols_missing}")

    df = pd.read_csv(path, usecols=cols_present, low_memory=False)
    df = _optimize_dtypes(df)

    print(f"[data_loader] Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancelled/diverted flights and rows with missing arr_delay.
    Drop the 'cancelled' and 'diverted' columns after filtering — they are
    not features and must not leak into the model.
    """
    n0 = len(df)
    df = df[df["cancelled"] == 0].copy()
    df = df[df["diverted"] == 0]
    df = df[df["arr_delay"].notna()]

    # Drop filter-only columns so they can't accidentally become features
    df = df.drop(columns=["cancelled", "diverted"], errors="ignore")

    print(f"[data_loader] After basic filter: {len(df):,} rows "
          f"(dropped {n0 - len(df):,} cancelled/diverted/missing)")
    return df.reset_index(drop=True)


def load_clean(sample: bool = False, force_reload: bool = False) -> pd.DataFrame:
    """
    Load the cleaned dataset.  Uses parquet cache if available.

    Args:
        sample: Use sample CSV (never cached).
        force_reload: Force re-read from CSV even if parquet exists.

    Returns:
        Cleaned DataFrame (no cancelled, no diverted, no missing arr_delay).
    """
    if sample:
        df = load_raw(sample=True)
        return basic_filter(df)

    if os.path.exists(CLEAN_PARQUET) and not force_reload:
        print(f"[data_loader] Loading cached clean data: {CLEAN_PARQUET}")
        df = pd.read_parquet(CLEAN_PARQUET)
        print(f"[data_loader] Loaded {len(df):,} rows from cache")
        return df

    df = load_raw(sample=False)
    df = basic_filter(df)

    os.makedirs(DATA_PROC_DIR, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"[data_loader] Cached clean data -> {CLEAN_PARQUET}")
    return df


if __name__ == "__main__":
    df = load_clean(sample=True)
    print(df.dtypes)
    print(df.head(3))
