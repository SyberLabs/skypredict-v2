"""
targets.py: Create all target columns from arr_delay.

Targets:
  - delayed        : binary, 1 if arr_delay > 15 min
  - delay_bucket   : ordinal, 0=on_time (≤15), 1=minor (15–60),
                     2=major (60–180), 3=severe (>180)
  - arr_delay_capped : regression, arr_delay clipped to [-60, 300]
"""
import numpy as np
import pandas as pd

from .config import (
    DELAY_THRESHOLD_MIN,
    ORDINAL_BINS,
    ORDINAL_LABELS,
    ARR_DELAY_CAP_LOW,
    ARR_DELAY_CAP_HIGH,
)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary, ordinal, and regression target columns.

    Operates on the raw arr_delay column.  Does NOT modify arr_delay itself
    (the capped version is a separate column).
    """
    assert "arr_delay" in df.columns, "arr_delay column required to create targets"

    # Binary: delayed if arr_delay > 15
    df["delayed"] = (df["arr_delay"] > DELAY_THRESHOLD_MIN).astype(np.int8)

    # Ordinal: 4-bucket classification
    df["delay_bucket"] = pd.cut(
        df["arr_delay"],
        bins=ORDINAL_BINS,
        labels=list(range(len(ORDINAL_LABELS))),
        right=True,        # bins are (-inf, 15], (15, 60], (60, 180], (180, inf]
    ).astype(np.int8)

    # Regression: capped arr_delay
    df["arr_delay_capped"] = df["arr_delay"].clip(
        lower=ARR_DELAY_CAP_LOW, upper=ARR_DELAY_CAP_HIGH
    )

    # Report class distribution
    delayed_pct = df["delayed"].mean()
    print(f"[targets] Binary: {delayed_pct:.1%} delayed (>{DELAY_THRESHOLD_MIN} min)")
    print(f"[targets] Ordinal distribution:")
    for i, label in enumerate(ORDINAL_LABELS):
        count = (df["delay_bucket"] == i).sum()
        print(f"  {label:>8s}: {count:>10,} ({count/len(df):.1%})")
    print(f"[targets] Regression: arr_delay_capped in "
          f"[{ARR_DELAY_CAP_LOW}, {ARR_DELAY_CAP_HIGH}], "
          f"mean={df['arr_delay_capped'].mean():.1f} min")

    return df
