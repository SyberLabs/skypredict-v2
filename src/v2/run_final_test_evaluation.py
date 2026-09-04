"""
run_final_test_evaluation.py -- Final evaluation on the held-out test set (Nov-Dec 2024).

This script compares Stage A (Flight-only), Stage B (Flight + Weather), and 
Stage C (Flight + Weather + Network Pressure) directly on the held-out test set.
"""
import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.v2.config import (
    ALL_FEATURES_A,
    ALL_FEATURES_B,
    ALL_FEATURES_C,
    RESULTS_DIR,
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
)
from src.v2.features_stage_b import build_features_stage_b
from src.v2.features_stage_c import build_network_pressure_features
from src.v2.airport_metadata import get_airport_metadata
from src.v2.train_stage_a import train_stage_a
from src.v2.evaluate_v2 import (
    evaluate_binary_clf,
    evaluate_ordinal_multiclass,
    evaluate_regression,
    pr_auc_at_common_base_rate,
)

FINAL_METRICS_JSON = os.path.join(RESULTS_DIR, "final_test_metrics.json")

def main():
    np.random.seed(RANDOM_SEED)
    t0 = time.time()

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT TEST SET EVALUATION (NOVEMBER - DECEMBER 2024)")
    print("=" * 80)

    # 1. Load Data
    df = load_clean(sample=False, force_reload=False)
    df = add_targets(df)
    train, val, test = temporal_split(df)
    del df

    # 2. Build Stage A Features
    print("\n[Step 1/3] Engineering Stage A features...")
    train, encoders = build_features_stage_a(train, encoders=None)
    val, _ = build_features_stage_a(val, encoders=encoders)
    test, _ = build_features_stage_a(test, encoders=encoders)

    hist_stats = compute_historical_stats(train)
    train = merge_historical_stats(train, hist_stats)
    # Expanding (not frozen) historical stats for val/test — see Stage A notes.
    val = compute_and_merge_hist_stats_for_val_test(train, val)
    test = compute_and_merge_hist_stats_for_val_test(
        pd.concat([train, val], ignore_index=True), test
    )

    # 3. Build Stage B Features
    print("\n[Step 2/3] Engineering Stage B (Weather) features...")
    meta = get_airport_metadata()
    wx_path = os.path.join(DATA_PROC_DIR, "hourly_weather_2024.parquet")
    if not os.path.exists(wx_path):
        raise FileNotFoundError(f"Weather data missing at {wx_path}. Run weather_download.py first.")
    wx_df = pd.read_parquet(wx_path)
    
    train = build_features_stage_b(train, wx_df, meta)
    val = build_features_stage_b(val, wx_df, meta)
    test = build_features_stage_b(test, wx_df, meta)
    del wx_df

    # 4. Build Stage C Features
    print("\n[Step 3/3] Engineering Stage C (Network) features...")
    train["split_group"] = "train"
    train["orig_split_index"] = train.index
    val["split_group"] = "val"
    val["orig_split_index"] = val.index
    test["split_group"] = "test"
    test["orig_split_index"] = test.index
    
    combined = pd.concat([train, val, test], ignore_index=True)
    combined = build_network_pressure_features(combined, meta)
    
    train = combined[combined["split_group"] == "train"].set_index("orig_split_index").drop(columns=["split_group"])
    train.index.name = None
    val = combined[combined["split_group"] == "val"].set_index("orig_split_index").drop(columns=["split_group"])
    val.index.name = None
    test = combined[combined["split_group"] == "test"].set_index("orig_split_index").drop(columns=["split_group"])
    test.index.name = None
    del combined

    # Prep test set arrays
    X_test_all = test.fillna(0)
    y_test_binary = test[TARGET_BINARY]
    y_test_reg = test[TARGET_REG]
    y_test_ordinal = test[TARGET_ORDINAL]

    # Reference prevalence: PR-AUC is sensitive to positive class rate, so
    # before claiming "test > val generalization" we rescale PR-AUC on test to
    # the val positive rate. ROC-AUC is prevalence-invariant and shown as-is.
    val_pos_rate = float(val[TARGET_BINARY].mean())
    test_pos_rate = float(y_test_binary.mean())
    print(f"\n[final] val pos rate = {val_pos_rate:.4f}  |  "
          f"test pos rate = {test_pos_rate:.4f}  "
          f"(PR-AUC on test will be reported both at natural prevalence and rebalanced to val's {val_pos_rate:.4f})")

    final_results = {
        "timestamp": datetime.now().isoformat(),
        "test_rows": len(test),
        "val_pos_rate": round(val_pos_rate, 4),
        "test_pos_rate": round(test_pos_rate, 4),
        "stages": {}
    }

    stages_to_eval = [
        ("Stage A (Flight-Only)", ALL_FEATURES_A),
        ("Stage B (Flight + Weather)", ALL_FEATURES_B),
        ("Stage C (Flight + Weather + Network)", ALL_FEATURES_C),
    ]

    for name, feature_list in stages_to_eval:
        print("\n" + "-" * 50)
        print(f"Training and Evaluating: {name}")
        print("-" * 50)
        
        feature_cols = [c for c in feature_list if c in train.columns]
        
        # Train
        trained_models = train_stage_a(train, val, feature_cols)
        
        # Evaluate on Held-out Test Set
        X_test_scaled = X_test_all[feature_cols]
        
        # Classifier Evaluation
        lgbm_clf_info = trained_models["lightgbm_clf"]
        binary_metrics = evaluate_binary_clf(lgbm_clf_info["model"], X_test_scaled, y_test_binary, f"LGBM Clf - {name}")

        # PR-AUC at val's positive base rate — for a fair test↔val comparison
        scores_test = lgbm_clf_info["model"].predict_proba(X_test_scaled)[:, 1]
        pr_balanced = pr_auc_at_common_base_rate(
            y_test_binary, scores_test, target_pos_rate=val_pos_rate,
        )
        binary_metrics["pr_auc_at_val_prevalence"] = pr_balanced

        # Ordinal multiclass evaluation
        lgbm_ord_info = trained_models["lightgbm_ordinal"]
        ord_metrics = evaluate_ordinal_multiclass(
            lgbm_ord_info["model"], X_test_scaled, y_test_ordinal, f"LGBM Ord - {name}",
        )

        # Regressor Evaluation
        lgbm_reg_info = trained_models["lightgbm_reg"]
        reg_metrics = evaluate_regression(lgbm_reg_info["model"], X_test_scaled, y_test_reg, y_test_ordinal, f"LGBM Reg - {name}")

        final_results["stages"][name] = {
            "binary": binary_metrics,
            "ordinal_multiclass": ord_metrics,
            "regression": reg_metrics,
        }

    # Save metrics
    with open(FINAL_METRICS_JSON, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n[OK] Final test metrics saved -> {FINAL_METRICS_JSON}")

    # Print gorgeous side-by-side final table
    print("\n" + "=" * 80)
    print("FINAL HELD-OUT TEST SET RESULTS PROGRESSION")
    print("=" * 80)
    print(f"{'Stage/Model':<40s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'ROC':>7s} {'PR-AUC':>7s} {'PR@valpr':>9s}")
    print("-" * 92)
    for name in final_results["stages"]:
        b = final_results["stages"][name]["binary"]
        pr_bal = b.get("pr_auc_at_val_prevalence", {}).get("pr_auc_rebalanced", float("nan"))
        print(f"{name:<40s} {b['accuracy']:>7.4f} {b['precision']:>7.4f} {b['recall']:>7.4f} "
              f"{b['f1']:>7.4f} {b['roc_auc']:>7.4f} {b['pr_auc']:>7.4f} {pr_bal:>9.4f}")
    print(f"\n[note] PR-AUC@valpr = PR-AUC on test after downsampling negatives to "
          f"match val's positive rate ({val_pos_rate:.4f}). PR-AUC is prevalence-sensitive; "
          "ROC-AUC is not.")

    print("\n" + "-" * 60)
    print(f"{'Stage/Model':<40s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}")
    print("-" * 60)
    for name in final_results["stages"]:
        r = final_results["stages"][name]["regression"]
        print(f"{name:<40s} {r['mae']:>8.3f} {r['rmse']:>8.3f} {r['r2']:>8.4f}")
        
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"Final test evaluation completed in {elapsed:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
