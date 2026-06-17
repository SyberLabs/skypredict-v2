"""
weather_download.py -- Downloads and caches 2024 hourly weather for all airports.

Uses Meteostat bulk data wrapper to download hourly NOAA ISD data.
"""
import os
import pandas as pd
from datetime import datetime
from meteostat import hourly

from .config import DATA_PROC_DIR, CLEAN_PARQUET
from .airport_metadata import get_airport_metadata

WEATHER_CACHE = os.path.join(DATA_PROC_DIR, "hourly_weather_2024.parquet")

def download_weather_for_airports():
    if os.path.exists(WEATHER_CACHE):
        print(f"[weather_download] Weather already downloaded at {WEATHER_CACHE}")
        return pd.read_parquet(WEATHER_CACHE)
        
    print("[weather_download] Loading airport list from dataset...")
    df = pd.read_parquet(CLEAN_PARQUET)
    iata_codes = set(df['origin'].unique()) | set(df['dest'].unique())
    print(f"[weather_download] Found {len(iata_codes)} unique airports.")
    
    meta = get_airport_metadata()
    start = datetime(2023, 12, 31) # Slightly before 2024 for timezone overlap
    end = datetime(2025, 1, 1)     # Slightly after 2024
    
    frames = []
    missing_airports = []
    
    print("[weather_download] Downloading hourly weather (this takes a few minutes)...")
    for i, iata in enumerate(iata_codes):
        if i % 50 == 0:
            print(f"  ... processed {i}/{len(iata_codes)}")
            
        if iata not in meta.index:
            missing_airports.append(iata)
            continue
            
        row = meta.loc[iata]
        station_id = row["meteostat_id"]
        if pd.isna(station_id):
            missing_airports.append(iata)
            continue
            
        try:
            ts = hourly(str(station_id), start, end)
            wx_df = ts.fetch()
            if not wx_df.empty:
                wx_df = wx_df.reset_index()
                wx_df["iata"] = iata
                frames.append(wx_df)
            else:
                missing_airports.append(iata)
        except Exception as e:
            print(f"Error fetching {iata}: {e}")
            missing_airports.append(iata)
            
    if missing_airports:
        print(f"[weather_download] WARNING: Could not fetch weather for {len(missing_airports)} airports: {missing_airports[:10]}...")
        
    all_weather = pd.concat(frames, ignore_index=True)
    all_weather.to_parquet(WEATHER_CACHE, index=False)
    print(f"[weather_download] Saved {len(all_weather)} hourly weather records to {WEATHER_CACHE}")
    return all_weather

if __name__ == "__main__":
    wx = download_weather_for_airports()
    print("Sample:\n", wx.head())
