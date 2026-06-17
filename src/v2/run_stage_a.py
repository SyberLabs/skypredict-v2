"""
run_stage_a.py — Orchestrator for SkyPredict v2 Stage A.

Runs the full pipeline:
  1. Load and clean data
  2. Create targets
  3. Temporal split (train/val/test)
  4. Feature engineering (with backward-only historical stats)
  5. Train models (LR, LightGBM clf, LightGBM reg)
  6. Evaluate on val set
  7. Compute context baselines
  8. Save all metrics to results/v2/metrics.json

Usage:
    py -m src.v2.run_stage_a [--sample]
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.v2.config import (
    ALL_FEATURES_A,
    RESULTS_DIR,
    MODELS_DIR,
    METRICS_JSON,
    TARGET_BINARY,
    TARGET_ORDINAL,
    TARGET_REG,
    RANDOM_SEED,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
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
    drop_raw_target,
)
from src.v2.baselines import (
    predict_global_median,
    predict_route_mean,
    predict_majority_class,
)
from src.v2.train_stage_a import train_stage_a
from src.v2.evaluate_v2 import (
    evaluate_binary_clf,
    evaluate_ordinal_recall,           # legacy — kept for back-compat reporting
    evaluate_ordinal_multiclass,       # the real ordinal metric
    evaluate_regression,
    evaluate_baseline_classification,
    evaluate_baseline_regression,
)


def run(sample: bool = False):
    """Execute the full Stage A pipeline."""
    np.random.seed(RANDOM_SEED)
    t_start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load and Clean ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE A -- Flight-Only Baseline (No Weather, No Network)")
    print("=" * 70)

    df = load_clean(sample=sample, force_reload=False)

    # ── 2. Create Targets ─────────────────────────────────────────────────
    print("\n-- Creating targets --")
    df = add_targets(df)

    # ── 3. Temporal Split ─────────────────────────────────────────────────
    print("\n-- Temporal split --")
    train, val, test = temporal_split(df)

    # Free memory — we don't need the full df anymore
    del df

    # ── 4. Feature Engineering ────────────────────────────────────────────
    print("\n-- Feature engineering (Stage A) --")

    # 4a. Basic features (time, calendar, distance, encoding)
    train, encoders = build_features_stage_a(train, encoders=None)
    val, _ = build_features_stage_a(val, encoders=encoders)
    test, _ = build_features_stage_a(test, encoders=encoders)

    # 4b. Historical stats — backward-only expanding window for train
    print("\n-- Computing backward-only historical stats (train) --")
    hist_stats = compute_historical_stats(train)
    train = merge_historical_stats(train, hist_stats)

    # 4c. Historical stats for val/test — backward-only EXPANDING (not frozen)
    # Each split's history includes all earlier splits, so val sees train, and
    # test sees train+val. This matches what an operational forecaster has at
    # gate-close. See [features_stage_a.py:compute_and_merge_hist_stats_for_val_test].
    print("-- Computing expanding historical stats for val/test --")
    val = compute_and_merge_hist_stats_for_val_test(train, val)
    test = compute_and_merge_hist_stats_for_val_test(
        pd.concat([train, val], ignore_index=True), test
    )

    # 4d. Defensive arr_delay drop — targets created, hist stats merged,
    # raw target no longer needed downstream. See features_stage_a.drop_raw_target.
    train = drop_raw_target(train)
    val   = drop_raw_target(val)
    test  = drop_raw_target(test)

    # ── 5. Validate Features ──────────────────────────────────────────────
    feature_cols = [c for c in ALL_FEATURES_A if c in train.columns]
    assert_no_leakage(feature_cols)

    # Report any missing features
    missing = [c for c in ALL_FEATURES_A if c not in train.columns]
    if missing:
        print(f"[run] WARNING: expected features not in data: {missing}")

    # ── 6. Train Models ───────────────────────────────────────────────────
    print("\n-- Training Stage A models --")
    trained = train_stage_a(train, val, feature_cols)

    # ── 7. Evaluate on Validation Set ─────────────────────────────────────
    print("\n-- Evaluating on VALIDATION set --")

    X_val = val[feature_cols].fillna(0)
    y_val_binary = val[TARGET_BINARY]
    y_val_ordinal = val[TARGET_ORDINAL]
    y_val_reg = val[TARGET_REG]

    metrics = {
        "stage": "A",
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

    # 7a. Logistic Regression — classification
    lr_info = trained["logistic_regression"]
    lr_binary = evaluate_binary_clf(
        lr_info["model"], X_val, y_val_binary, "Logistic Regression",
        scaler=lr_info["scaler"],
    )
    lr_ordinal = evaluate_ordinal_recall(
        lr_info["model"], X_val, y_val_ordinal, "Logistic Regression",
        scaler=lr_info["scaler"],
    )
    metrics["models"]["logistic_regression"] = {
        "binary": lr_binary,
        "ordinal_recall": lr_ordinal,
        "train_time_s": lr_info["train_time"],
    }

    # 7b. LightGBM Classifier
    lgbm_clf_info = trained["lightgbm_clf"]
    lgbm_binary = evaluate_binary_clf(
        lgbm_clf_info["model"], X_val, y_val_binary, "LightGBM Classifier",
    )
    lgbm_ordinal = evaluate_ordinal_recall(
        lgbm_clf_info["model"], X_val, y_val_ordinal, "LightGBM Classifier",
    )
    metrics["models"]["lightgbm_clf"] = {
        "binary": lgbm_binary,
        "ordinal_recall": lgbm_ordinal,
        "train_time_s": lgbm_clf_info["train_time"],
    }

    # 7b'. LightGBM Ordinal — actual 4-class delay_bucket classifier
    lgbm_ord_info = trained["lightgbm_ordinal"]
    lgbm_ord_metrics = evaluate_ordinal_multiclass(
        lgbm_ord_info["model"], X_val, y_val_ordinal, "LightGBM Ordinal",
    )
    metrics["models"]["lightgbm_ordinal"] = {
        "ordinal_multiclass": lgbm_ord_metrics,
        "train_time_s": lgbm_ord_info["train_time"],
    }

    # 7c. LightGBM Regressor
    lgbm_reg_info = trained["lightgbm_reg"]
    lgbm_reg_metrics = evaluate_regression(
        lgbm_reg_info["model"], X_val, y_val_reg, y_val_ordinal,
        "LightGBM Regressor",
    )
    metrics["models"]["lightgbm_reg"] = {
        "regression": lgbm_reg_metrics,
        "train_time_s": lgbm_reg_info["train_time"],
    }

    # ── 8. Context Baselines ──────────────────────────────────────────────
    print("\n-- Context baselines --")

    # Baseline 0: classification floor — predict majority class for every flight.
    # By construction ROC-AUC = 0.5 and PR-AUC = positive base rate; surfacing
    # these in the metrics table makes the lift of the real models concrete.
    majority_result = predict_majority_class(train[TARGET_BINARY], len(val))
    majority_metrics = evaluate_baseline_classification(
        majority_result["predictions"], majority_result["scores"],
        y_val_binary, "predict_majority_class",
    )
    majority_metrics["train_pos_rate"] = majority_result["train_pos_rate"]
    metrics["baselines"]["predict_majority_class"] = majority_metrics

    # Baseline 1: predict median
    median_result = predict_global_median(train[TARGET_REG], len(val))
    median_metrics = evaluate_baseline_regression(
        median_result["predictions"], y_val_reg, y_val_binary,
        "predict_median",
    )
    median_metrics["predicted_value"] = median_result["value"]
    metrics["baselines"]["predict_median"] = median_metrics

    # Baseline 2: per-route historical mean
    route_result = predict_route_mean(train, val, TARGET_REG)
    route_metrics = evaluate_baseline_regression(
        route_result["predictions"], y_val_reg, y_val_binary,
        "route_historical_mean",
    )
    route_metrics["n_unseen_routes"] = route_result["n_unseen_routes"]
    metrics["baselines"]["route_historical_mean"] = route_metrics

    # ── 9. Save Metrics ───────────────────────────────────────────────────
    elapsed = time.time() - t_start
    metrics["total_time_s"] = round(elapsed, 1)

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[run] [OK] Metrics saved -> {METRICS_JSON}")

    # ── 10. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE A RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} "
          f"{'F1':>7s} {'ROC':>7s} {'PR-AUC':>7s}")
    print("-" * 70)
    for mname in ["logistic_regression", "lightgbm_clf"]:
        m = metrics["models"][mname]["binary"]
        print(f"{m['model']:<25s} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['roc_auc']:>7.4f} "
              f"{m['pr_auc']:>7.4f}")

    print(f"\n{'Model':<25s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}")
    print("-" * 50)
    reg = metrics["models"]["lightgbm_reg"]["regression"]
    print(f"{reg['model']:<25s} {reg['mae']:>8.3f} {reg['rmse']:>8.3f} "
          f"{reg['r2']:>8.4f}")

    print(f"\n{'Baseline':<25s} {'MAE':>8s} {'Bin Acc':>8s}")
    print("-" * 42)
    for bname in ["predict_median", "route_historical_mean"]:
        b = metrics["baselines"][bname]
        print(f"{b['name']:<25s} {b['mae']:>8.3f} {b['binary_accuracy']:>8.4f}")

    print(f"\nTotal pipeline time: {elapsed:.0f}s")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkyPredict v2 - Stage A")
    parser.add_argument("--sample", action="store_true",
                        help="Use 10K sample CSV for fast testing")
    args = parser.parse_args()
    run(sample=args.sample)
