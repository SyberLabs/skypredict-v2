"""
airport_metadata.py: IATA to Lat/Lon and Timezone mapping.

Fetches airport data from OpenFlights to map the BTS 3-letter IATA codes
to their latitude, longitude, and IANA timezone strings.
"""
import os
import pandas as pd
import requests
import gzip
import json

from .config import DATA_PROC_DIR

AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
AIRPORTS_CACHE = os.path.join(DATA_PROC_DIR, "airports.csv")
METEOSTAT_STATIONS_URL = "https://bulk.meteostat.net/v2/stations/full.json.gz"
METEOSTAT_CACHE = os.path.join(DATA_PROC_DIR, "meteostat_stations.csv")

def get_airport_metadata() -> pd.DataFrame:
    """
    Returns a DataFrame with columns [iata, icao, lat, lon, timezone, meteostat_id].
    Index is IATA.
    """
    if os.path.exists(AIRPORTS_CACHE):
        df = pd.read_csv(AIRPORTS_CACHE)
    else:
        print("[airport_metadata] Downloading OpenFlights airport database...")
        os.makedirs(DATA_PROC_DIR, exist_ok=True)
        cols = [
            "airport_id", "name", "city", "country", "iata", "icao",
            "lat", "lon", "alt", "tz_offset", "dst", "timezone",
            "type", "source"
        ]
        df = pd.read_csv(AIRPORTS_URL, header=None, names=cols, na_values=["\\N"])
        df = df.dropna(subset=["iata", "icao", "lat", "lon", "timezone"])
        df = df[["iata", "icao", "lat", "lon", "timezone"]]
        df.to_csv(AIRPORTS_CACHE, index=False)
        
    if os.path.exists(METEOSTAT_CACHE):
        m_df = pd.read_csv(METEOSTAT_CACHE)
    else:
        print("[airport_metadata] Downloading Meteostat station inventory...")
        r = requests.get(METEOSTAT_STATIONS_URL)
        data = json.loads(gzip.decompress(r.content))
        m_df = pd.DataFrame(data)
        # Extract ICAO from identifiers dict
        m_df["icao"] = m_df["identifiers"].apply(lambda x: x.get("icao") if isinstance(x, dict) else None)
        m_df = m_df.dropna(subset=["icao"])
        m_df = m_df[["id", "icao", "timezone"]].rename(columns={"id": "meteostat_id"})
        m_df.to_csv(METEOSTAT_CACHE, index=False)
        
    # Join on ICAO
    df = df.merge(m_df[["icao", "meteostat_id"]], on="icao", how="left")
    
    # K-prefix fallback for missing ICAOs in OpenFlights that might just be K+IATA
    # Wait, we already have ICAO from OpenFlights. If meteostat doesn't have it, we get NaN.
    
    df = df.set_index("iata")
    return df

if __name__ == "__main__":
    df = get_airport_metadata()
    print(f"Loaded {len(df)} airports.")
    print("Sample:\n", df.head())
    if "LAX" in df.index:
        print("LAX details:\n", df.loc["LAX"])

