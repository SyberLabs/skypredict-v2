"""
config.py — Central configuration for SkyPredict v2 pipeline.

All paths, constants, allowed/forbidden feature lists, and target definitions
live here.  Import this module everywhere to keep magic numbers out of logic.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed_v2")
MODELS_DIR    = os.path.join(BASE_DIR, "models", "v2")
RESULTS_DIR   = os.path.join(BASE_DIR, "results", "v2")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")

RAW_CSV       = os.path.join(DATA_RAW_DIR, "flight_data_2024.csv")
SAMPLE_CSV    = os.path.join(DATA_RAW_DIR, "flight_data_2024_sample.csv")
CLEAN_PARQUET = os.path.join(DATA_PROC_DIR, "clean.parquet")
METRICS_JSON  = os.path.join(RESULTS_DIR, "metrics.json")

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Temporal Split Boundaries (inclusive) ────────────────────────────────────
# Train: Jan 1 – Aug 31, Val: Sep 1 – Oct 31, Test: Nov 1 – Dec 31
TRAIN_END = "2024-08-31"
VAL_START = "2024-09-01"
VAL_END   = "2024-10-31"
TEST_START = "2024-11-01"

# ─── Target Definitions ──────────────────────────────────────────────────────
DELAY_THRESHOLD_MIN = 15          # BTS convention: delayed if ARR_DELAY > 15

# Ordinal buckets: on-time, minor, major, severe
ORDINAL_BINS   = [-float("inf"), 15, 60, 180, float("inf")]
ORDINAL_LABELS = ["on_time", "minor", "major", "severe"]

# Regression cap — limits tail influence
ARR_DELAY_CAP_LOW  = -60
ARR_DELAY_CAP_HIGH = 300

# ─── Column Names (as they appear in the raw CSV) ────────────────────────────
# These are the ONLY raw columns we load.  Everything else is dropped at read.
LOAD_COLS = [
    "fl_date",
    "op_unique_carrier",
    "origin",
    "dest",
    "crs_dep_time",
    "crs_arr_time",
    "crs_elapsed_time",
    "distance",
    "arr_delay",
    "cancelled",
    "diverted",         # used ONLY for filtering, never as a feature
    "day_of_week",
    "month",
    "day_of_month",
]

# ─── FORBIDDEN Columns ───────────────────────────────────────────────────────
# These are POST-HOC fields that only exist after departure/arrival.
# They must NEVER appear in ANY feature matrix.  An assertion checks this
# before every model training run.
FORBIDDEN_COLUMNS = frozenset([
    "dep_delay", "dep_time", "taxi_out", "wheels_off",
    "wheels_on", "taxi_in", "arr_time",
    "actual_elapsed_time", "air_time",
    "carrier_delay", "weather_delay", "nas_delay",
    "security_delay", "late_aircraft_delay",
    "cancellation_code", "diverted",
    # Also block the target itself and its derivatives
    "arr_delay", "delayed", "delay_bucket",
])

# ─── Stage A Feature List ────────────────────────────────────────────────────
# Numeric features (no scaling needed for tree models; LR gets its own scaler)
NUMERIC_FEATURES_A = [
    "dep_minutes",          # CRS_DEP_TIME → minutes from midnight
    "arr_minutes",          # CRS_ARR_TIME → minutes from midnight
    "crs_elapsed_time",     # scheduled block time
    "distance",             # route distance in miles
    "day_of_month",
    # Backward-only historical stats
    "carrier_hist_delay_rate",
    "carrier_hist_avg_delay",
    "origin_hist_delay_rate",
    "dest_hist_delay_rate",
    "route_hist_delay_rate",
    "route_hist_avg_delay",
]

CATEGORICAL_FEATURES_A = [
    "op_carrier_enc",       # label-encoded carrier
    "origin_enc",           # label-encoded origin airport
    "dest_enc",             # label-encoded destination airport
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "distance_bucket",
]

ALL_FEATURES_A = NUMERIC_FEATURES_A + CATEGORICAL_FEATURES_A

# ─── Stage B Feature List ────────────────────────────────────────────────────
WEATHER_FEATURES = [
    "origin_wx_temp", "origin_wx_dwpt", "origin_wx_rhum", "origin_wx_wspd",
    "origin_wx_wpgt", "origin_wx_wdir", "origin_wx_prcp", "origin_wx_pres",
    "origin_wx_visibility", "origin_wx_is_ifr", "origin_wx_is_low_vis",
    "origin_wx_is_precip", "origin_wx_is_high_wind",
    "dest_wx_temp", "dest_wx_dwpt", "dest_wx_rhum", "dest_wx_wspd",
    "dest_wx_wpgt", "dest_wx_wdir", "dest_wx_prcp", "dest_wx_pres",
    "dest_wx_visibility", "dest_wx_is_ifr", "dest_wx_is_low_vis",
    "dest_wx_is_precip", "dest_wx_is_high_wind",
]

ALL_FEATURES_B = ALL_FEATURES_A + WEATHER_FEATURES

# ─── Stage C Feature List ────────────────────────────────────────────────────
NETWORK_FEATURES = [
    "origin_pressure_3h",
    "origin_pressure_count",
    "dest_pressure_3h",
    "dest_pressure_count",
]

ALL_FEATURES_C = ALL_FEATURES_B + NETWORK_FEATURES

# Target column names
TARGET_BINARY    = "delayed"          # 1 if arr_delay > 15
TARGET_ORDINAL   = "delay_bucket"     # 0=on_time, 1=minor, 2=major, 3=severe
TARGET_REG       = "arr_delay_capped" # arr_delay clipped to [CAP_LOW, CAP_HIGH]
N_ORDINAL_CLASSES = 4                  # len(ORDINAL_LABELS)

# ─── U.S. Federal Holidays 2024 ──────────────────────────────────────────────
HOLIDAYS_2024 = frozenset([
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # MLK Jr Day
    "2024-02-19",  # Presidents' Day
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-10-14",  # Columbus Day
    "2024-11-11",  # Veterans Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
])
