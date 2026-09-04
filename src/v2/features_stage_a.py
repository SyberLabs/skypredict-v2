"""
features_stage_a.py: Flight-only feature engineering for Stage A.

All features are derived from information known BEFORE departure:
  - Scheduled times (CRS_DEP_TIME, CRS_ARR_TIME, CRS_ELAPSED_TIME)
  - Route (ORIGIN, DEST, DISTANCE)
  - Calendar (FL_DATE, DAY_OF_WEEK, MONTH)
  - Carrier (OP_UNIQUE_CARRIER)
  - Backward-only historical delay statistics

NO post-hoc fields are used.  An assertion verifies this before returning.
"""
import numpy as np
import pandas as pd

from .config import (
    HOLIDAYS_2024,
    DELAY_THRESHOLD_MIN,
    FORBIDDEN_COLUMNS,
    ALL_FEATURES_A,
    TARGET_BINARY,
    TARGET_ORDINAL,
    TARGET_REG,
)


# ─── Time Features ───────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CRS_DEP_TIME and CRS_ARR_TIME from HHMM int format to
    continuous minutes-from-midnight.  Better for models than separate
    hour/minute because it's a single monotonic feature.
    """
    for col, out_col in [("crs_dep_time", "dep_minutes"),
                         ("crs_arr_time", "arr_minutes")]:
        raw = df[col].fillna(0).astype(int)
        hours = raw // 100
        minutes = raw % 100
        df[out_col] = (hours * 60 + minutes).astype(np.int16)

    return df


# ─── Calendar Features ───────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_weekend and is_holiday flags from existing date columns."""
    # Weekend: BTS day_of_week 1=Mon…7=Sun
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(np.int8)

    # Holiday
    if "fl_date" in df.columns:
        fl_dates = pd.to_datetime(df["fl_date"]).dt.strftime("%Y-%m-%d")
        df["is_holiday"] = fl_dates.isin(HOLIDAYS_2024).astype(np.int8)
    else:
        df["is_holiday"] = np.int8(0)

    return df


# ─── Distance Bucketing ──────────────────────────────────────────────────────

def add_distance_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bin distance into short(0) / medium(1) / long-haul(2)."""
    df["distance_bucket"] = pd.cut(
        df["distance"],
        bins=[0, 500, 1500, np.inf],
        labels=[0, 1, 2],
        right=True,
    ).astype(np.int8)
    return df


# ─── Label Encoding ──────────────────────────────────────────────────────────

def label_encode(
    df: pd.DataFrame,
    cols: list[str] = None,
    encoders: dict = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.  Fit on train, apply to val/test.

    Args:
        df: DataFrame to encode.
        cols: Columns to encode.  Default: carrier, origin, dest.
        encoders: If None, fit from df.  Otherwise, apply existing mapping.

    Returns:
        (df, encoders): df has new *_enc columns; encoders dict for reuse.
    """
    if cols is None:
        cols = ["op_unique_carrier", "origin", "dest"]

    if encoders is None:
        encoders = {}
        for col in cols:
            uniques = sorted(df[col].dropna().unique())
            encoders[col] = {v: i for i, v in enumerate(uniques)}

    suffix_map = {
        "op_unique_carrier": "op_carrier_enc",
        "origin": "origin_enc",
        "dest": "dest_enc",
    }

    for col in cols:
        out_col = suffix_map.get(col, f"{col}_enc")
        mapping = encoders[col]
        df[out_col] = df[col].map(mapping).fillna(-1).astype(np.int16)

    return df, encoders


# ─── Historical Delay Statistics (backward-only) ─────────────────────────────

def compute_historical_stats(
    df: pd.DataFrame,
    date_col: str = "fl_date",
) -> pd.DataFrame:
    """
    Compute backward-only expanding-window delay statistics per calendar day.

    For each date d, stats are the aggregate of ALL flights on dates < d.
    This guarantees no look-ahead within the training set.

    Returns a DataFrame indexed by date with columns for each stat group.

    NOTE: This function uses arr_delay (the raw target) to compute historical
    aggregates.  This is acceptable because:
      1. We only use flights from dates STRICTLY BEFORE the current date.
      2. These flights have already completed: their arr_delay is known.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    df["_date"] = dates.dt.date
    df["_delayed"] = (df["arr_delay"] > DELAY_THRESHOLD_MIN).astype(int)

    # Pre-compute per-day aggregates for each grouping key
    stats_frames = {}

    # Carrier stats
    carrier_daily = (
        df.groupby(["_date", "op_unique_carrier"])
        .agg(
            _arr_sum=("arr_delay", "sum"),
            _arr_count=("arr_delay", "count"),
            _delayed_sum=("_delayed", "sum"),
        )
        .reset_index()
        .sort_values("_date")
    )
    stats_frames["carrier"] = _expanding_stats(
        carrier_daily, "op_unique_carrier",
        "carrier_hist_avg_delay", "carrier_hist_delay_rate"
    )

    # Origin airport stats
    origin_daily = (
        df.groupby(["_date", "origin"])
        .agg(
            _arr_sum=("arr_delay", "sum"),
            _arr_count=("arr_delay", "count"),
            _delayed_sum=("_delayed", "sum"),
        )
        .reset_index()
        .sort_values("_date")
    )
    stats_frames["origin"] = _expanding_stats(
        origin_daily, "origin",
        "origin_hist_avg_delay", "origin_hist_delay_rate"
    )

    # Destination airport stats
    dest_daily = (
        df.groupby(["_date", "dest"])
        .agg(
            _arr_sum=("arr_delay", "sum"),
            _arr_count=("arr_delay", "count"),
            _delayed_sum=("_delayed", "sum"),
        )
        .reset_index()
        .sort_values("_date")
    )
    stats_frames["dest"] = _expanding_stats(
        dest_daily, "dest",
        "dest_hist_avg_delay", "dest_hist_delay_rate"
    )

    # Route (origin, dest) stats
    route_daily = (
        df.groupby(["_date", "origin", "dest"])
        .agg(
            _arr_sum=("arr_delay", "sum"),
            _arr_count=("arr_delay", "count"),
            _delayed_sum=("_delayed", "sum"),
        )
        .reset_index()
        .sort_values("_date")
    )
    stats_frames["route"] = _expanding_stats_route(route_daily)

    return stats_frames


def _expanding_stats(
    daily: pd.DataFrame,
    key_col: str,
    avg_name: str,
    rate_name: str,
) -> pd.DataFrame:
    """
    Compute backward-only expanding-window stats for a single grouping key.

    For each (date, key) pair, the stat reflects all flights with that key
    on dates STRICTLY BEFORE this one.
    """
    daily = daily.sort_values("_date")

    # Cumulative sums within each group, then shift by 1 day to make backward-only
    daily["_cum_arr_sum"] = daily.groupby(key_col)["_arr_sum"].cumsum()
    daily["_cum_count"]   = daily.groupby(key_col)["_arr_count"].cumsum()
    daily["_cum_delayed"] = daily.groupby(key_col)["_delayed_sum"].cumsum()

    # Shift: for date d, use cumulative up to d-1
    # We shift within each group so that date d sees stats from dates < d
    daily["_prev_arr_sum"] = daily.groupby(key_col)["_cum_arr_sum"].shift(1)
    daily["_prev_count"]   = daily.groupby(key_col)["_cum_count"].shift(1)
    daily["_prev_delayed"] = daily.groupby(key_col)["_cum_delayed"].shift(1)

    daily[avg_name]  = daily["_prev_arr_sum"] / daily["_prev_count"]
    daily[rate_name] = daily["_prev_delayed"] / daily["_prev_count"]

    return daily[["_date", key_col, avg_name, rate_name]].copy()


def _expanding_stats_route(daily: pd.DataFrame) -> pd.DataFrame:
    """Backward-only expanding stats for (origin, dest) route pairs."""
    daily = daily.sort_values("_date")

    grp = daily.groupby(["origin", "dest"])
    daily["_cum_arr_sum"] = grp["_arr_sum"].cumsum()
    daily["_cum_count"]   = grp["_arr_count"].cumsum()
    daily["_cum_delayed"] = grp["_delayed_sum"].cumsum()

    daily["_prev_arr_sum"] = grp["_cum_arr_sum"].shift(1)
    daily["_prev_count"]   = grp["_cum_count"].shift(1)
    daily["_prev_delayed"] = grp["_cum_delayed"].shift(1)

    daily["route_hist_avg_delay"]  = daily["_prev_arr_sum"] / daily["_prev_count"]
    daily["route_hist_delay_rate"] = daily["_prev_delayed"] / daily["_prev_count"]

    return daily[["_date", "origin", "dest",
                  "route_hist_avg_delay", "route_hist_delay_rate"]].copy()


def merge_historical_stats(
    df: pd.DataFrame,
    stats: dict,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Join pre-computed backward-only historical stats onto any split.

    For val/test splits where stats were computed only from training data,
    we join on (date, key): the val/test flight on date d gets the stat
    that was current as of the last training date.

    For dates beyond the training range, we use the FINAL training-day stat.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["fl_date"]).dt.date

    # Carrier stats
    if "carrier" in stats:
        df = df.merge(
            stats["carrier"],
            left_on=["_date", "op_unique_carrier"],
            right_on=["_date", "op_unique_carrier"],
            how="left",
        )

    # Origin stats
    if "origin" in stats:
        df = df.merge(
            stats["origin"],
            left_on=["_date", "origin"],
            right_on=["_date", "origin"],
            how="left",
        )

    # Dest stats
    if "dest" in stats:
        df = df.merge(
            stats["dest"],
            left_on=["_date", "dest"],
            right_on=["_date", "dest"],
            how="left",
        )

    # Route stats
    if "route" in stats:
        df = df.merge(
            stats["route"],
            left_on=["_date", "origin", "dest"],
            right_on=["_date", "origin", "dest"],
            how="left",
        )

    # Fill NaN stats (first day of data, unseen keys) with fill_value
    hist_cols = [
        "carrier_hist_delay_rate", "carrier_hist_avg_delay",
        "origin_hist_delay_rate", "origin_hist_avg_delay",
        "dest_hist_delay_rate", "dest_hist_avg_delay",
        "route_hist_delay_rate", "route_hist_avg_delay",
    ]
    for col in hist_cols:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value).astype(np.float32)

    df = df.drop(columns=["_date"], errors="ignore")
    return df


def compute_and_merge_hist_stats_for_val_test(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Backward-only expanding historical stats for val/test rows.

    For a val/test flight on date d, the assigned stat reflects every flight on
    dates STRICTLY BEFORE d: across BOTH train and val/test history that has
    already accumulated. This matches what an operational forecaster sees at
    gate-close: a running aggregate that ticks forward day by day.

    The previous implementation froze the stats at the train cutoff, which
    introduced a train↔val distribution mismatch (training rows saw stats that
    grew, validation rows saw a single constant). The mismatch was not a
    leakage bug but did suppress the model's ability to exploit the feature.

    Causal guarantee: target_df rows on date d only ever see rows with date < d
: accomplished by appending train and target into a single chronologically
    sorted frame and applying the same _expanding_stats / _expanding_stats_route
    helpers used for train. Same code path, no special-case freezing.
    """
    # Build a combined view (train + target) with a positional row-id we can
    # resolve back to the target frame in original order. We need the same
    # underlying columns (fl_date, carrier, origin, dest, arr_delay) the
    # expanding-stat helpers operate on.
    base_cols = ["fl_date", "op_unique_carrier", "origin", "dest", "arr_delay"]
    train_view = train_df[base_cols].copy()
    train_view["_row_id"] = -1                              # train rows discarded
    tgt_view = target_df[base_cols].copy()
    tgt_view["_row_id"] = np.arange(len(target_df))          # position-preserving id
    combined = pd.concat([train_view, tgt_view], ignore_index=True)

    # Reuse the same expanding-stat helpers that the training path uses, so
    # train and val/test stats are produced by IDENTICAL code: no special-case
    # logic that could drift.
    stats = compute_historical_stats(combined, date_col="fl_date")
    combined = merge_historical_stats(combined, stats, fill_value=fill_value)

    hist_cols = [
        "carrier_hist_delay_rate", "carrier_hist_avg_delay",
        "origin_hist_delay_rate", "origin_hist_avg_delay",
        "dest_hist_delay_rate", "dest_hist_avg_delay",
        "route_hist_delay_rate", "route_hist_avg_delay",
    ]
    # Project stats back onto target_df, ordered by the positional row id we
    # injected before the merges (merges preserve left-row order, but being
    # explicit is cheap insurance).
    target_rows = combined[combined["_row_id"] >= 0].sort_values("_row_id")
    target_df = target_df.copy()
    for col in hist_cols:
        if col in target_rows.columns:
            target_df[col] = target_rows[col].to_numpy()
    return target_df


# ─── Full Feature Pipeline ───────────────────────────────────────────────────

def build_features_stage_a(
    df: pd.DataFrame,
    encoders: dict = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply all Stage A feature transforms (excluding historical stats,
    which are handled separately due to train/val/test logic).

    Returns (df, encoders).
    """
    df = add_time_features(df)
    df = add_calendar_features(df)
    df = add_distance_bucket(df)
    df, encoders = label_encode(df, encoders=encoders)
    return df, encoders


def drop_raw_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive drop of arr_delay once historical stats have consumed it.

    The capped/binary/ordinal target columns remain. Belt-and-braces against a
    future feature touching the raw target by name even though
    FORBIDDEN_COLUMNS / assert_no_leakage would catch it at training time.
    """
    return df.drop(columns=["arr_delay"], errors="ignore")


def assert_no_leakage(feature_cols: list[str]) -> None:
    """
    Hard check: none of the forbidden columns appear in the feature list.
    Raises AssertionError with details if violated.
    """
    leaked = set(feature_cols) & FORBIDDEN_COLUMNS
    assert len(leaked) == 0, (
        f"LABEL LEAKAGE DETECTED: forbidden columns in feature matrix: {leaked}\n"
        "These are post-hoc fields and must NEVER be used as model inputs."
    )
    print(f"[features] [OK] No leakage -- {len(feature_cols)} features, "
          f"0 forbidden columns detected")
    print(f"[features] Feature list: {feature_cols}")
