"""
features_stage_b.py -- Weather feature engineering and joining.

Handles:
1. Timezone-aware UTC conversion of scheduled departure and arrival times.
2. Join of hourly weather via merge_asof.
3. Feature derivation (IFR conditions, visibility, etc).
"""
import numpy as np
import pandas as pd
from datetime import timedelta

from .airport_metadata import get_airport_metadata
from .features_stage_a import assert_no_leakage

def _convert_local_to_utc(df: pd.DataFrame, time_col: str, iata_col: str, meta: pd.DataFrame) -> pd.Series:
    """
    Converts a local time integer (e.g. 1430 for 14:30) and date into a UTC datetime.
    """
    # Create local datetime string without timezone
    date_str = df["fl_date"].astype(str)
    # Pad to 4 digits, e.g. 30 -> '0030'
    time_str = df[time_col].fillna(0).astype(int).astype(str).str.zfill(4)
    time_str = time_str.str.slice(0, 2) + ":" + time_str.str.slice(2, 4) + ":00"
    
    # Handle cases where hour is 24 (e.g., 2400)
    time_str = time_str.replace("24:00:00", "00:00:00")
    
    dt_local_unaware = pd.to_datetime(date_str + " " + time_str, errors="coerce")
    
    # We will build a temporary dataframe to do grouped timezone localization
    # Doing this per-airport is much faster than per-row
    temp_df = pd.DataFrame({
        "dt": dt_local_unaware,
        "iata": df[iata_col]
    })
    
    # Join the IANA timezone strings
    temp_df = temp_df.merge(meta[["timezone"]], left_on="iata", right_index=True, how="left")
    
    # We will compute UTC time grouped by timezone
    result = pd.Series(index=df.index, dtype='datetime64[ns, UTC]')
    
    for tz, group in temp_df.groupby("timezone"):
        try:
            # Localize to the specific timezone (handles DST), then convert to UTC
            localized = group["dt"].dt.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
            result.loc[group.index] = localized.dt.tz_convert("UTC")
        except Exception as e:
            print(f"Error converting timezone {tz}: {e}")
            
    return result

def prepare_weather_features(wx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the raw hourly meteostat data, derive visibility and boolean weather flags.
    """
    wx = wx_df.copy()
    wx["time"] = pd.to_datetime(wx["time"]).dt.tz_localize("UTC")
    
    # Derived visibility approximation from meteostat coco (condition code) 
    # if actual visibility isn't present.
    # We'll use coco codes: 1-4 clear/cloudy, 5-6 fog, 7-18 rain, 19-22 snow, 23-27 showers, 28-29 storm
    wx["visibility_km"] = 10.0 # Default good vis
    wx.loc[wx["coco"].isin([5, 6]), "visibility_km"] = 0.5 # Fog
    wx.loc[wx["coco"].isin([7, 8, 9, 14, 15, 16]), "visibility_km"] = 5.0 # Rain
    wx.loc[wx["coco"].isin([19, 20, 21, 22]), "visibility_km"] = 2.0 # Snow
    wx.loc[wx["coco"].isin([25, 26, 27, 28, 29]), "visibility_km"] = 1.0 # Heavy precip / storm
    
    wx["is_ifr"] = (wx["visibility_km"] < 4.8).fillna(False).astype(np.int8) # < 3 miles
    wx["is_low_vis"] = (wx["visibility_km"] < 1.6).fillna(False).astype(np.int8) # < 1 mile
    
    if "prcp" not in wx.columns:
        wx["prcp"] = 0.0
    if "wspd" not in wx.columns:
        wx["wspd"] = 0.0
    if "wpgt" not in wx.columns:
        wx["wpgt"] = 0.0
        
    wx["is_precip"] = (wx["prcp"] > 0).fillna(False).astype(np.int8)
    wx["is_high_wind"] = ((wx["wspd"].fillna(0) > 40) | (wx["wpgt"].fillna(0) > 60)).astype(np.int8)
    
    # Fill missing columns we need
    expected_cols = ["temp", "dwpt", "rhum", "wspd", "wpgt", "wdir", "prcp", "pres"]
    for c in expected_cols:
        if c not in wx.columns:
            wx[c] = 0.0
        wx[c] = pd.to_numeric(wx[c], errors='coerce').fillna(0.0)
            
    # Keep only what we need
    cols = ["iata", "time", "visibility_km", "is_ifr", "is_low_vis", "is_precip", "is_high_wind"] + expected_cols
    
    return wx[cols]

def merge_weather_features(df: pd.DataFrame, wx_df: pd.DataFrame, prefix: str, iata_col: str, time_col: str, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Converts flight time to UTC and joins the nearest weather record.
    """
    print(f"[features_b] Preparing {prefix} timezone conversions...")
    utc_times = _convert_local_to_utc(df, time_col, iata_col, meta)
    
    # Temporary copy for merging
    join_df = df[[iata_col]].copy()
    join_df["join_time"] = utc_times
    join_df["_index"] = join_df.index
    
    # Drop rows where we couldn't get a UTC time
    join_df = join_df.dropna(subset=["join_time"])
    join_df = join_df.sort_values("join_time")
    
    # Cast to string to match wx_df's iata column type
    join_df[iata_col] = join_df[iata_col].astype(str)
    join_df["join_time"] = join_df["join_time"].astype("datetime64[ns, UTC]")
    
    wx_sorted = wx_df.sort_values("time")
    wx_sorted["time"] = wx_sorted["time"].astype("datetime64[ns, UTC]")
    
    print(f"[features_b] Merging {prefix} weather data (asof, backward 1h)...")
    # CAUSAL: direction="backward" with allow_exact_matches=True returns the most
    # recent observation at or before the scheduled time. tolerance=1h matches the
    # hourly ISD cadence: if no obs in the last hour we'd rather have NaN (filled
    # later with median) than reach further. "nearest" would let us pull weather
    # from the FUTURE of scheduled time, which a forecaster at gate-close does not
    # have. Empirical impact is small (adjacent hourly METARs are autocorrelated)
    # but the previous formulation contradicted the project's no-leakage claim.
    merged = pd.merge_asof(
        join_df,
        wx_sorted,
        left_on="join_time",
        right_on="time",
        left_by=iata_col,
        right_by="iata",
        direction="backward",
        allow_exact_matches=True,
        tolerance=pd.Timedelta(hours=1),
    )
    
    # Set index back to original
    merged = merged.set_index("_index").sort_index()
    
    # Rename columns with prefix
    rename_dict = {
        "temp": f"{prefix}_temp",
        "dwpt": f"{prefix}_dwpt",
        "rhum": f"{prefix}_rhum",
        "wspd": f"{prefix}_wspd",
        "wpgt": f"{prefix}_wpgt",
        "wdir": f"{prefix}_wdir",
        "prcp": f"{prefix}_prcp",
        "pres": f"{prefix}_pres",
        "visibility_km": f"{prefix}_visibility",
        "is_ifr": f"{prefix}_is_ifr",
        "is_low_vis": f"{prefix}_is_low_vis",
        "is_precip": f"{prefix}_is_precip",
        "is_high_wind": f"{prefix}_is_high_wind",
    }
    
    merged = merged[list(rename_dict.keys())].rename(columns=rename_dict)
    
    # Join back to original df
    df = df.join(merged)
    
    # Fill NAs for missing weather with median/0
    for col in rename_dict.values():
        if "is_" in col or "prcp" in col:
            df[col] = df[col].fillna(0)
        else:
            # Fill other continuous with overall median
            median_val = wx_df[col.replace(f"{prefix}_", "")].median() if col.replace(f"{prefix}_", "") in wx_df.columns else 0
            df[col] = df[col].fillna(median_val)
            
    return df

def build_features_stage_b(df: pd.DataFrame, wx_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for Stage B feature engineering.
    Applies weather joining to both origin and destination.
    """
    wx_proc = prepare_weather_features(wx_df)
    
    df = merge_weather_features(df, wx_proc, "origin_wx", "origin", "crs_dep_time", meta)
    df = merge_weather_features(df, wx_proc, "dest_wx", "dest", "crs_arr_time", meta)
    
    return df
