"""Cached artifact loaders for the SkyPredict dashboard.

This dashboard runs against the artifacts that already exist on disk after the
audit-fixed pipeline runs were completed:

  * results/v2/metrics.json: Stage A val metrics + baselines
  * results/v2/metrics_stage_b.json: Stage B val metrics
  * results/v2/metrics_stage_c.json: Stage C val metrics + baselines + ordinal
  * results/v2/final_test_metrics.json: held-out test, A/B/C side by side,
        including PR-AUC@val-prevalence and ordinal multiclass blocks
  * data/processed_v2/clean.parquet: 6.96M cleaned flights
  * data/processed_v2/airports.csv: IATA → lat/lon/timezone
  * data/processed_v2/hourly_weather_2024.parquet: NOAA hourly weather

We deliberately do NOT depend on val_predictions_*.parquet or stage_*.joblib
bundles: the teammate's dashboard required those and they don't exist on this
machine. Every chart this dashboard renders is either (a) a scalar/array read
from JSON, or (b) an aggregate computed from clean.parquet on the fly with
Streamlit's cache keeping the cost paid once per session.
"""
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ── Paths ───────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v2")
DATA_PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v2")

METRICS_PATHS = {
    "A": os.path.join(RESULTS_DIR, "metrics.json"),
    "B": os.path.join(RESULTS_DIR, "metrics_stage_b.json"),
    "C": os.path.join(RESULTS_DIR, "metrics_stage_c.json"),
}
FINAL_METRICS_PATH = os.path.join(RESULTS_DIR, "final_test_metrics.json")
CLEAN_PARQUET     = os.path.join(DATA_PROC_DIR, "clean.parquet")
AIRPORTS_CSV      = os.path.join(DATA_PROC_DIR, "airports.csv")
WEATHER_PARQUET   = os.path.join(DATA_PROC_DIR, "hourly_weather_2024.parquet")

# How final_test_metrics.json names each stage
FINAL_STAGE_KEY = {
    "A": "Stage A (Flight-Only)",
    "B": "Stage B (Flight + Weather)",
    "C": "Stage C (Flight + Weather + Network)",
}


# ── Cache plumbing ──────────────────────────────────────────────────────────
def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data(show_spinner=False)
def _load_json(path: str, _mt: float) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_clean(_mt: float) -> Optional[pd.DataFrame]:
    if not os.path.exists(CLEAN_PARQUET):
        return None
    df = pd.read_parquet(CLEAN_PARQUET)
    df["fl_date"] = pd.to_datetime(df["fl_date"])
    return df


@st.cache_data(show_spinner=False)
def _load_airports(_mt: float) -> Optional[pd.DataFrame]:
    if not os.path.exists(AIRPORTS_CSV):
        return None
    return pd.read_csv(AIRPORTS_CSV).set_index("iata")


# ── Public reader API ───────────────────────────────────────────────────────
def stage_metrics(stage: str) -> Optional[dict]:
    return _load_json(METRICS_PATHS[stage], _mtime(METRICS_PATHS[stage]))


def final_metrics() -> Optional[dict]:
    return _load_json(FINAL_METRICS_PATH, _mtime(FINAL_METRICS_PATH))


def clean_flights() -> Optional[pd.DataFrame]:
    return _load_clean(_mtime(CLEAN_PARQUET))


def airports() -> Optional[pd.DataFrame]:
    return _load_airports(_mtime(AIRPORTS_CSV))


# ── Derived aggregates (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def val_window_aggregate(_mt: float) -> Optional[pd.DataFrame]:
    """Per-airport flight count + delay rate restricted to the validation
    window (Sep-Oct 2024). Used by the Overview map; doesn't require the
    val-prediction parquets the teammate's dashboard expected: we just slice
    clean.parquet by date."""
    df = _load_clean(_mt)
    if df is None:
        return None
    val = df[(df["fl_date"] >= "2024-09-01") & (df["fl_date"] <= "2024-10-31")]
    val = val.assign(delayed=(val["arr_delay"] > 15).astype(np.int8))
    agg = (
        val.groupby(val["origin"].astype(str))
           .agg(flights=("delayed", "size"),
                delayed_rate=("delayed", "mean"),
                mean_delay=("arr_delay", "mean"))
           .reset_index()
           .rename(columns={"origin": "iata"})
    )
    return agg


def val_window_delays() -> Optional[np.ndarray]:
    """The validation-window arr_delay column, capped to [-60, 300] to match
    the regression target's framing. Used by the Overview delay histogram."""
    df = clean_flights()
    if df is None:
        return None
    val = df[(df["fl_date"] >= "2024-09-01") & (df["fl_date"] <= "2024-10-31")]
    return val["arr_delay"].clip(-60, 300).to_numpy()


@st.cache_data(show_spinner=False)
def arrivals_for_pressure_demo(_mt: float, airport: str,
                               window_start: str, window_end: str
                               ) -> Optional[pd.DataFrame]:
    """All arrivals at a specified airport between two dates, with the actual
    arrival time computed as crs_arr_time + arr_delay. Used by the Network
    Propagation page to *demonstrate* Stage C's 3-hour rolling-window feature
    without needing the model.

    Returns columns: actual_arr_minute (minutes from window_start), arr_delay,
    fl_date, origin, op_unique_carrier.
    """
    df = _load_clean(_mt)
    if df is None:
        return None
    win = df[(df["fl_date"] >= window_start) & (df["fl_date"] <= window_end)
             & (df["dest"].astype(str) == airport)].copy()
    if win.empty:
        return win
    # CRS arrival as datetime (no timezone correction needed: same-airport;
    # the dashboard's demo only needs *relative* timestamps for one airport).
    crs_h = (win["crs_arr_time"] // 100).clip(0, 23)
    crs_m = (win["crs_arr_time"] % 100).clip(0, 59)
    crs_dt = (pd.to_datetime(win["fl_date"])
              + pd.to_timedelta(crs_h, unit="h")
              + pd.to_timedelta(crs_m, unit="m"))
    # Actual arrival = scheduled + delay
    win["actual_arr_dt"] = crs_dt + pd.to_timedelta(win["arr_delay"].fillna(0), unit="m")
    return win[["fl_date", "op_unique_carrier", "origin", "arr_delay", "actual_arr_dt"]]


# ── Convenience accessors ───────────────────────────────────────────────────
def final_stage_binary(stage: str) -> Optional[dict]:
    fm = final_metrics()
    if fm is None:
        return None
    return fm["stages"].get(FINAL_STAGE_KEY[stage], {}).get("binary")


def final_stage_regression(stage: str) -> Optional[dict]:
    fm = final_metrics()
    if fm is None:
        return None
    return fm["stages"].get(FINAL_STAGE_KEY[stage], {}).get("regression")


def final_stage_ordinal(stage: str) -> Optional[dict]:
    fm = final_metrics()
    if fm is None:
        return None
    return fm["stages"].get(FINAL_STAGE_KEY[stage], {}).get("ordinal_multiclass")


def all_artifacts_present() -> tuple[bool, list[str]]:
    required = list(METRICS_PATHS.values()) + [FINAL_METRICS_PATH,
                                               CLEAN_PARQUET, AIRPORTS_CSV]
    missing = [p for p in required if not os.path.exists(p)]
    return len(missing) == 0, missing
