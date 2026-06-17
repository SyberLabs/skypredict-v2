"""
run_stage_b.py -- Orchestrator for SkyPredict v2 Stage B.

Runs the full pipeline adding Weather features:
  1. Load and clean data
  2. Create targets
  3. Temporal split (train/val/test)
  4. Feature engineering (Stage A + Stage B Weather)
  5. Train models (LR, LightGBM clf, LightGBM reg)
  6. Evaluate on val set
  7. Compute context baselines
  8. Save all metrics to results/v2/metrics_stage_b.json
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.v2.config import (
    ALL_FEATURES_B,
    RESULTS_DIR,
    MODELS_DIR,
    TARGET_BINARY,
    TARGET_ORDINAL,
    TARGET_REG,
    RANDOM_SEED,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    DATA_PROC_DIR,
)
from src.v2.data_loader import load_clean
from src.v2.targets import add_targets
from src.v2.split import temporal_split
from src.v2.features_stage_a import (
    build_features_stage_a,
    compute_historical_stats,
    merge_historical_stats,
    compute_and_merge_hist_stats_for_val_test,
    assert_no_leakage,
)
from src.v2.features_stage_b import build_features_stage_b
from src.v2.airport_metadata import get_airport_metadata
from src.v2.baselines import predict_global_median, predict_route_mean
from src.v2.train_stage_a import train_stage_a
from src.v2.evaluate_v2 import (
    evaluate_binary_clf,
    evaluate_ordinal_recall,
    evaluate_regression,
    evaluate_baseline_regression,
)

METRICS_JSON_B = os.path.join(RESULTS_DIR, "metrics_stage_b.json")

def run(sample: bool = False):
    np.random.seed(RANDOM_SEED)
    t_start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("STAGE B -- Flight + Weather Baseline")
    print("=" * 70)

    # 1. Load Data
    df = load_clean(sample=sample, force_reload=False)

    # 2. Targets
    print("\n-- Creating targets --")
    df = add_targets(df)

    # 3. Split
    print("\n-- Temporal split --")
    train, val, test = temporal_split(df)
    del df

    # 4. Feature Engineering
    print("\n-- Feature engineering (Stage A: Basic + Historical) --")
    train, encoders = build_features_stage_a(train, encoders=None)
    val, _ = build_features_stage_a(val, encoders=encoders)
    test, _ = build_features_stage_a(test, encoders=encoders)

    hist_stats = compute_historical_stats(train)
    train = merge_historical_stats(train, hist_stats)
    val = compute_and_merge_hist_stats_for_val_test(train, val)
    test = compute_and_merge_hist_stats_for_val_test(train, test)

    print("\n-- Feature engineering (Stage B: Weather) --")
    meta = get_airport_metadata()
    wx_path = os.path.join(DATA_PROC_DIR, "hourly_weather_2024.parquet")
    if not os.path.exists(wx_path):
        raise FileNotFoundError(f"Weather data missing at {wx_path}. Run weather_download.py first.")
    wx_df = pd.read_parquet(wx_path)
    
    train = build_features_stage_b(train, wx_df, meta)
    val = build_features_stage_b(val, wx_df, meta)
    test = build_features_stage_b(test, wx_df, meta)
    
    # We can delete wx_df to save memory
    del wx_df

    # 5. Validate Features
    feature_cols = [c for c in ALL_FEATURES_B if c in train.columns]
    assert_no_leakage(feature_cols)

    missing = [c for c in ALL_FEATURES_B if c not in train.columns]
    if missing:
        print(f"[run] WARNING: expected features not in data: {missing}")

    # 6. Train Models
    print("\n-- Training Stage B models --")
    trained = train_stage_a(train, val, feature_cols)

    # 7. Evaluate
    print("\n-- Evaluating on VALIDATION set --")
    X_val = val[feature_cols].fillna(0)
    y_val_binary = val[TARGET_BINARY]
    y_val_ordinal = val[TARGET_ORDINAL]
    y_val_reg = val[TARGET_REG]

    metrics = {
        "stage": "B",
        "timestamp": datetime.now().isoformat(),
        "dataset": "BTS 2024 (Kaggle hrishitpatil/flight-data-2024)",
        "sample_mode": sample,
        "split": {
            "train": f"2024-01-01 to {TRAIN_END}",
            "val": f"{VAL_START} to {VAL_END}",
            "test": f"{TEST_START} to 2024-12-31",
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
        },
        "feature_columns": feature_cols,
        "seed": RANDOM_SEED,
        "models": {},
        "baselines": {},
    }

    lr_info = trained["logistic_regression"]
    metrics["models"]["logistic_regression"] = {
        "binary": evaluate_binary_clf(lr_info["model"], X_val, y_val_binary, "Logistic Regression", scaler=lr_info["scaler"]),
        "ordinal_recall": evaluate_ordinal_recall(lr_info["model"], X_val, y_val_ordinal, "Logistic Regression", scaler=lr_info["scaler"]),
        "train_time_s": lr_info["train_time"],
    }

    lgbm_clf_info = trained["lightgbm_clf"]
    metrics["models"]["lightgbm_clf"] = {
        "binary": evaluate_binary_clf(lgbm_clf_info["model"], X_val, y_val_binary, "LightGBM Classifier"),
        "ordinal_recall": evaluate_ordinal_recall(lgbm_clf_info["model"], X_val, y_val_ordinal, "LightGBM Classifier"),
        "train_time_s": lgbm_clf_info["train_time"],
    }

    lgbm_reg_info = trained["lightgbm_reg"]
    metrics["models"]["lightgbm_reg"] = {
        "regression": evaluate_regression(lgbm_reg_info["model"], X_val, y_val_reg, y_val_ordinal, "LightGBM Regressor"),
        "train_time_s": lgbm_reg_info["train_time"],
    }

    # 8. Baselines
    print("\n-- Context baselines --")
    median_result = predict_global_median(train[TARGET_REG], len(val))
    median_metrics = evaluate_baseline_regression(median_result["predictions"], y_val_reg, y_val_binary, "predict_median")
    median_metrics["predicted_value"] = median_result["value"]
    metrics["baselines"]["predict_median"] = median_metrics

    route_result = predict_route_mean(train, val, TARGET_REG)
    route_metrics = evaluate_baseline_regression(route_result["predictions"], y_val_reg, y_val_binary, "route_historical_mean")
    route_metrics["n_unseen_routes"] = route_result["n_unseen_routes"]
    metrics["baselines"]["route_historical_mean"] = route_metrics

    # 9. Save
    elapsed = time.time() - t_start
    metrics["total_time_s"] = round(elapsed, 1)

    with open(METRICS_JSON_B, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[run] [OK] Metrics saved -> {METRICS_JSON_B}")

    # 10. Summary
    print("\n" + "=" * 70)
    print("STAGE B RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'ROC':>7s} {'PR-AUC':>7s}")
    print("-" * 70)
    for mname in ["logistic_regression", "lightgbm_clf"]:
        m = metrics["models"][mname]["binary"]
        print(f"{m['model']:<25s} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['roc_auc']:>7.4f} {m['pr_auc']:>7.4f}")

    print(f"\n{'Model':<25s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}")
    print("-" * 50)
    reg = metrics["models"]["lightgbm_reg"]["regression"]
    print(f"{reg['model']:<25s} {reg['mae']:>8.3f} {reg['rmse']:>8.3f} {reg['r2']:>8.4f}")

    print(f"\nTotal pipeline time: {elapsed:.0f}s")
    print("=" * 70)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkyPredict v2 -- Stage B")
    parser.add_argument("--sample", action="store_true", help="Use 10K sample CSV for fast testing")
    args = parser.parse_args()
    run(sample=args.sample)
