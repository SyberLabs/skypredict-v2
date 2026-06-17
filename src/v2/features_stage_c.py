"""
features_stage_c.py -- Efficient and exact network pressure feature calculation.

We calculate the average delay and count of arrivals at the origin and destination
airports in the 3 hours strictly preceding the flight's scheduled departure.
To do this efficiently on 7M rows, we use a cumulative sum approach grouped by airport,
and resolve the window boundaries using fast pd.merge_asof lookups.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from .airport_metadata import get_airport_metadata
from .features_stage_b import _convert_local_to_utc

def build_network_pressure_features(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Computes exact 3-hour arrival delay pressure for both origin and destination airports.
    """
    print("[features_c] Converting scheduled departure times to UTC...")
    # Get departure UTC times
    dep_utc = _convert_local_to_utc(df, "crs_dep_time", "origin", meta)
    
    # We will build a helper DataFrame of all "arrivals" in the dataset
    # Wait, the arrival scheduled time is crs_arr_time. We want to align arrivals
    # by their actual arrival time? No, actual arrival time = crs_arr_time + arr_delay.
    # But wait, in a real system, do we know the actual arrival time of an inbound flight?
    # Yes, we know it when it lands. But to be safe and conservative, we can use the scheduled arrival time
    # as the timestamp, or scheduled arrival time + arr_delay. Let's use scheduled arrival time + arr_delay
    # (actual arrival time) to represent when the event actually became known.
    # To be extremely conservative, let's use scheduled arrival time. If scheduled arrival time is 1:00 PM,
    # and it arrived at 1:30 PM (delay 30), it was physically known at 1:30 PM.
    # So the "actual arrival time" is: crs_arr_time + arr_delay.
    # Let's compute actual arrival time for all flights.
    print("[features_c] Computing actual arrival times (UTC)...")
    arr_utc = _convert_local_to_utc(df, "crs_arr_time", "dest", meta)
    
    # Actual arrival time = scheduled arrival time (UTC) + arr_delay (minutes)
    actual_arr_utc = arr_utc + pd.to_timedelta(df["arr_delay"].fillna(0), unit="m")
    
    # Create the arrivals database from the entire dataset
    print("[features_c] Creating arrivals database...")
    arrivals = pd.DataFrame({
        "airport": df["dest"].astype(str),
        "arr_time": actual_arr_utc,
        "arr_delay": df["arr_delay"].fillna(0)
    }).dropna(subset=["arr_time"])
    
    # Sort arrivals by time for merge_asof and rolling calculations
    arrivals = arrivals.sort_values("arr_time")
    
    # Compute cumulative sum of delay and cumulative count of arrivals grouped by airport
    print("[features_c] Computing cumulative arrival statistics...")
    arrivals["cum_sum"] = arrivals.groupby("airport")["arr_delay"].cumsum()
    arrivals["cum_cnt"] = arrivals.groupby("airport").cumcount() + 1
    
    # Let's keep a reference to the index of arrivals to map back
    arrivals["arr_idx"] = np.arange(len(arrivals))
    
    # To perform exact 3-hour window query:
    # For a departure at time T:
    # 1. Find the last arrival before T (idx_end)
    # 2. Find the last arrival before T - 3 hours (idx_start)
    # 3. Sum = cum_sum[idx_end] - cum_sum[idx_start]
    # 4. Count = cum_cnt[idx_end] - cum_cnt[idx_start]
    # 5. Mean = Sum / Count
    
    # Build helper DataFrame for departure queries
    dep_queries = pd.DataFrame({
        "airport_org": df["origin"].astype(str),
        "airport_dst": df["dest"].astype(str),
        "dep_time": dep_utc,
        "dep_time_minus_3h": dep_utc - pd.Timedelta(hours=3),
        "orig_index": df.index
    }).dropna(subset=["dep_time"])
    
    # Helper to query pressure
    def query_pressure(dep_df: pd.DataFrame, airport_col: str, time_col: str, prefix: str) -> pd.DataFrame:
        print(f"[features_c] Querying pressure for {prefix}...")
        # We want to find the last arrival at airport_col before time_col
        # Sort queries by time
        query_sorted = dep_df.sort_values(time_col)
        
        # We use merge_asof to find the last arrival strictly before time_col
        merged = pd.merge_asof(
            query_sorted,
            arrivals[["airport", "arr_time", "cum_sum", "cum_cnt"]],
            left_on=time_col,
            right_on="arr_time",
            left_by=airport_col,
            right_by="airport",
            direction="backward",
            allow_exact_matches=False
        )
        
        # Fill missing values (no arrivals before this time) with 0
        merged["cum_sum"] = merged["cum_sum"].fillna(0.0)
        merged["cum_cnt"] = merged["cum_cnt"].fillna(0.0)
        
        # Restore original index order
        merged = merged.set_index("orig_index").sort_index()
        return merged[["cum_sum", "cum_cnt"]].rename(columns={"cum_sum": f"{prefix}_sum", "cum_cnt": f"{prefix}_cnt"})

    # Query end of window (T_dep)
    org_end = query_pressure(dep_queries, "airport_org", "dep_time", "org_end")
    # Query start of window (T_dep - 3h)
    org_start = query_pressure(dep_queries, "airport_org", "dep_time_minus_3h", "org_start")
    
    # Query destination end and start
    dst_end = query_pressure(dep_queries, "airport_dst", "dep_time", "dst_end")
    dst_start = query_pressure(dep_queries, "airport_dst", "dep_time_minus_3h", "dst_start")
    
    # Compute the final rolling windows
    print("[features_c] Computing final rolling metrics...")
    
    # Origin pressure
    df["origin_pressure_count"] = org_end["org_end_cnt"] - org_start["org_start_cnt"]
    org_delay_sum = org_end["org_end_sum"] - org_start["org_start_sum"]
    df["origin_pressure_3h"] = np.where(
        df["origin_pressure_count"] > 0,
        org_delay_sum / df["origin_pressure_count"],
        0.0
    )
    
    # Destination pressure
    df["dest_pressure_count"] = dst_end["dst_end_cnt"] - dst_start["dst_start_cnt"]
    dst_delay_sum = dst_end["dst_end_sum"] - dst_start["dst_start_sum"]
    df["dest_pressure_3h"] = np.where(
        df["dest_pressure_count"] > 0,
        dst_delay_sum / df["dest_pressure_count"],
        0.0
    )
    
    # Ensure types are correct
    df["origin_pressure_count"] = df["origin_pressure_count"].fillna(0).astype(np.int32)
    df["dest_pressure_count"] = df["dest_pressure_count"].fillna(0).astype(np.int32)
    df["origin_pressure_3h"] = df["origin_pressure_3h"].fillna(0.0).astype(np.float32)
    df["dest_pressure_3h"] = df["dest_pressure_3h"].fillna(0.0).astype(np.float32)

    # Defensive: arr_delay has been fully consumed (targets created, historical
    # stats merged, pressure aggregated). Drop it locally so downstream code
    # cannot accidentally pass it into the feature matrix even if the
    # FORBIDDEN_COLUMNS assertion is bypassed. The capped/binary/ordinal targets
    # remain on the dataframe under their own names.
    df = df.drop(columns=["arr_delay"], errors="ignore")

    return df
