"""
train_stage_a.py: Train Stage A models (flight-only, no weather/network).

Models:
  - Logistic Regression (binary classification)
  - LightGBM Classifier (binary classification)
  - LightGBM Regressor  (regression on arr_delay_capped)

All models use ONLY features in ALL_FEATURES_A.
An assertion checks for leakage before training.
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, LGBMRegressor

from .config import (
    RANDOM_SEED,
    ALL_FEATURES_A,
    TARGET_BINARY,
    TARGET_ORDINAL,
    TARGET_REG,
    N_ORDINAL_CLASSES,
)
from .features_stage_a import assert_no_leakage


def _get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Extract and validate feature matrix from dataframe."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[feature_cols].fillna(0)


def train_stage_a(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str] = None,
) -> dict:
    """
    Train all Stage A models.

    Args:
        train_df: Training data with features and targets.
        val_df: Validation data for evaluation during training.
        feature_cols: Feature column names.  Default: ALL_FEATURES_A.

    Returns:
        Dict with trained models, scaler, and training metadata.
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES_A

    # ── LEAKAGE CHECK ─────────────────────────────────────────────────────
    assert_no_leakage(feature_cols)

    X_train = _get_feature_matrix(train_df, feature_cols)
    y_train_clf = train_df[TARGET_BINARY]
    y_train_reg = train_df[TARGET_REG]

    X_val = _get_feature_matrix(val_df, feature_cols)

    results = {}

    # ── 1. Logistic Regression ────────────────────────────────────────────
    print("\n[train] -- Logistic Regression (binary) --")
    t0 = time.time()

    # LR needs scaling; tree models don't
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
    )
    lr.fit(X_train_scaled, y_train_clf)
    elapsed = time.time() - t0
    print(f"[train]   Trained in {elapsed:.1f}s")

    results["logistic_regression"] = {
        "model": lr,
        "scaler": scaler,
        "train_time": round(elapsed, 1),
        "task": "classification",
    }

    # ── 2. LightGBM Classifier ────────────────────────────────────────────
    print("\n[train] -- LightGBM Classifier (binary) --")
    t0 = time.time()

    lgbm_clf = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_clf.fit(X_train, y_train_clf)
    elapsed = time.time() - t0
    print(f"[train]   Trained in {elapsed:.1f}s")

    results["lightgbm_clf"] = {
        "model": lgbm_clf,
        "scaler": None,
        "train_time": round(elapsed, 1),
        "task": "classification",
    }

    # ── 3. LightGBM Regressor ─────────────────────────────────────────────
    print("\n[train] -- LightGBM Regressor (regression) --")
    t0 = time.time()

    lgbm_reg = LGBMRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="mae",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_reg.fit(X_train, y_train_reg)
    elapsed = time.time() - t0
    print(f"[train]   Trained in {elapsed:.1f}s")

    results["lightgbm_reg"] = {
        "model": lgbm_reg,
        "scaler": None,
        "train_time": round(elapsed, 1),
        "task": "regression",
    }

    # ── 4. LightGBM Ordinal Classifier ────────────────────────────────────
    # Actual multiclass on delay_bucket {0=on_time, 1=minor, 2=major, 3=severe}.
    # Replaces the old "ordinal recall" that incorrectly reused the binary
    # classifier, that metric was misnamed. We now train a real 4-class model
    # and report adjacent-bucket accuracy / quadratic-weighted kappa downstream.
    print("\n[train] -- LightGBM Ordinal Classifier (4-class) --")
    t0 = time.time()
    y_train_ord = train_df[TARGET_ORDINAL].astype(int)
    lgbm_ord = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        num_class=N_ORDINAL_CLASSES,
        class_weight="balanced",     # buckets are extremely imbalanced (~80/13/5/2 %)
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_ord.fit(X_train, y_train_ord)
    elapsed = time.time() - t0
    print(f"[train]   Trained in {elapsed:.1f}s")
    results["lightgbm_ordinal"] = {
        "model": lgbm_ord,
        "scaler": None,
        "train_time": round(elapsed, 1),
        "task": "ordinal",
    }

    return results
