"""
baselines.py — Context baselines for SkyPredict v2.

These are simple predictors that set the floor for model performance:
  1. Predict-global-median: predict the training-set median arr_delay for all.
  2. Per-route historical mean: predict the mean arr_delay for each (origin, dest)
     route, computed from training data only.
"""
import numpy as np
import pandas as pd

from .config import DELAY_THRESHOLD_MIN, ORDINAL_BINS, ORDINAL_LABELS


def predict_global_median(
    train_target: pd.Series,
    n_test: int,
) -> dict:
    """
    Baseline 1: predict the training-set median for every flight.

    Returns a dict with predictions and summary stats.
    """
    median_val = train_target.median()
    preds = np.full(n_test, median_val)

    return {
        "name": "predict_median",
        "value": float(median_val),
        "predictions": preds,
    }


def predict_majority_class(
    train_binary: pd.Series,
    n_test: int,
) -> dict:
    """
    Classification floor: predict the majority class (almost always "on-time")
    for every flight, with a constant score equal to the training positive rate.

    This makes the no-skill PR-AUC (= positive base rate) and ROC-AUC (= 0.5)
    explicit in the metrics table, so a reader can see how far above floor each
    model actually sits.
    """
    majority = int(train_binary.mode().iloc[0])
    pos_rate = float(train_binary.mean())
    preds = np.full(n_test, majority, dtype=np.int8)
    # Constant score for every sample → ROC-AUC = 0.5, PR-AUC = base rate
    scores = np.full(n_test, pos_rate, dtype=np.float32)
    return {
        "name": "predict_majority_class",
        "majority_class": majority,
        "train_pos_rate": pos_rate,
        "predictions": preds,
        "scores": scores,
    }


def predict_route_mean(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "arr_delay_capped",
) -> dict:
    """
    Baseline 2: predict the per-route (origin, dest) mean delay from training.

    Unseen routes fall back to the global training mean.
    """
    global_mean = train_df[target_col].mean()

    route_means = (
        train_df.groupby(["origin", "dest"])[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: "_route_pred"})
    )

    merged = test_df[["origin", "dest"]].merge(
        route_means, on=["origin", "dest"], how="left"
    )
    preds = merged["_route_pred"].fillna(global_mean).values

    n_unseen = merged["_route_pred"].isna().sum()
    print(f"[baselines] Route-mean baseline: {n_unseen:,} unseen routes "
          f"({n_unseen/len(test_df):.1%}) fell back to global mean={global_mean:.1f}")

    return {
        "name": "route_historical_mean",
        "global_mean": float(global_mean),
        "n_unseen_routes": int(n_unseen),
        "predictions": preds,
    }
