"""
evaluate_v2.py — Full evaluation suite for SkyPredict v2.

Metrics:
  Classification (binary):
    accuracy, precision, recall, F1, ROC-AUC, PR-AUC

  Classification (ordinal per-bucket recall):
    recall for each of on_time / minor / major / severe

  Regression:
    MAE, RMSE, R², MAE stratified by true delay bucket

  Context baselines:
    MAE and binary accuracy for predict-median and route-mean
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
    cohen_kappa_score,
    recall_score as sklearn_recall,
)

from .config import (
    DELAY_THRESHOLD_MIN,
    ORDINAL_BINS,
    ORDINAL_LABELS,
    ALL_FEATURES_A,
    TARGET_BINARY,
    TARGET_ORDINAL,
    TARGET_REG,
)


def evaluate_binary_clf(
    model, X: pd.DataFrame, y_true: pd.Series, name: str,
    scaler=None,
) -> dict:
    """Evaluate a binary classifier. Returns metrics dict."""
    X_input = scaler.transform(X) if scaler is not None else X

    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:, 1]

    metrics = {
        "model": name,
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 4),
    }

    print(f"[eval] {name} -- Acc={metrics['accuracy']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
          f"PR-AUC={metrics['pr_auc']:.4f}")

    return metrics


def evaluate_ordinal_recall(
    model, X: pd.DataFrame, y_ordinal: pd.Series, name: str,
    scaler=None,
) -> dict:
    """
    Evaluate per-bucket recall for ordinal classification.
    Uses the binary classifier's probability to rank, then maps to buckets.
    """
    X_input = scaler.transform(X) if scaler is not None else X
    y_pred_binary = model.predict(X_input)

    per_bucket = {}
    for i, label in enumerate(ORDINAL_LABELS):
        mask = y_ordinal == i
        if mask.sum() == 0:
            per_bucket[label] = None
            continue
        # For ordinal: count how many in this true bucket are correctly
        # identified as delayed (for minor/major/severe) or on-time
        if i == 0:  # on_time
            correct = ((y_pred_binary == 0) & mask).sum()
        else:  # delayed buckets
            correct = ((y_pred_binary == 1) & mask).sum()
        per_bucket[label] = round(float(correct / mask.sum()), 4)

    print(f"[eval] {name} -- Ordinal recall: {per_bucket}")
    return per_bucket


def evaluate_ordinal_multiclass(
    model, X: pd.DataFrame, y_ordinal: pd.Series, name: str,
    scaler=None,
) -> dict:
    """
    Evaluate a true ordinal/multiclass classifier on delay_bucket.

    Reports:
      - per-class precision/recall (4 buckets)
      - overall accuracy and macro-F1
      - quadratic-weighted Cohen's kappa (penalizes farther-off predictions)
      - adjacent-bucket accuracy (|pred - true| <= 1)

    This is the metric set the project originally promised; the prior
    "ordinal recall" implementation reused the binary classifier and is
    deprecated.
    """
    X_input = scaler.transform(X) if scaler is not None else X
    y_pred = model.predict(X_input)
    y_true = y_ordinal.astype(int).to_numpy()
    y_pred = np.asarray(y_pred, dtype=int)

    per_bucket = {}
    for i, label in enumerate(ORDINAL_LABELS):
        mask = y_true == i
        if mask.sum() == 0:
            per_bucket[label] = {"precision": None, "recall": None, "support": 0}
            continue
        per_bucket[label] = {
            "precision": round(float(precision_score(
                y_true == i, y_pred == i, zero_division=0)), 4),
            "recall": round(float(recall_score(
                y_true == i, y_pred == i, zero_division=0)), 4),
            "support": int(mask.sum()),
        }

    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    adj_acc = float(np.mean(np.abs(y_pred - y_true) <= 1))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    metrics = {
        "model": name,
        "accuracy":              round(acc, 4),
        "macro_f1":              round(macro_f1, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "adjacent_bucket_accuracy": round(adj_acc, 4),
        "per_bucket":            per_bucket,
    }
    print(f"[eval] {name} -- Acc={acc:.4f}  macroF1={macro_f1:.4f}  "
          f"QWK={qwk:.4f}  AdjAcc={adj_acc:.4f}")
    for label, m in per_bucket.items():
        if m["precision"] is None:
            continue
        print(f"  {label:>8s}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"N={m['support']:,}")
    return metrics


def evaluate_regression(
    model, X: pd.DataFrame, y_true: pd.Series, y_ordinal: pd.Series,
    name: str, scaler=None,
) -> dict:
    """Evaluate regression model with stratified MAE by delay bucket."""
    X_input = scaler.transform(X) if scaler is not None else X
    y_pred = model.predict(X_input)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    # Per-bucket MAE
    mae_by_bucket = {}
    for i, label in enumerate(ORDINAL_LABELS):
        mask = y_ordinal == i
        if mask.sum() == 0:
            mae_by_bucket[label] = None
            continue
        bucket_mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        mae_by_bucket[label] = round(bucket_mae, 2)

    metrics = {
        "model": name,
        "mae":  round(mae, 3),
        "rmse": round(rmse, 3),
        "r2":   round(r2, 4),
        "mae_by_bucket": mae_by_bucket,
    }

    print(f"[eval] {name} -- MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
    print(f"[eval] {name} -- MAE by bucket: {mae_by_bucket}")

    return metrics


def pr_auc_at_common_base_rate(
    y_true: pd.Series,
    y_score: np.ndarray,
    target_pos_rate: float,
    seed: int = 42,
) -> dict:
    """
    Recompute PR-AUC after rebalancing the negatives so the positive rate
    matches `target_pos_rate`. Lets us compare PR-AUC across splits whose
    natural prevalence differs (e.g. Nov-Dec test ~30% delayed vs Sep-Oct val
    ~20% delayed); PR-AUC is prevalence-sensitive while ROC-AUC is not, so a
    raw test > val PR-AUC delta can be entirely a base-rate artifact.

    Returns the resampled PR-AUC, the natural PR-AUC, and the prevalence
    actually used so the reader can see the adjustment.

    Bidirectional: if natural rate > target we downsample POSITIVES; if natural
    rate < target we downsample NEGATIVES. Never upsamples either class. With
    Nov-Dec 2024 test (~17% delayed) vs Sep-Oct val (~13.7%), we down-sample
    positives — losing some signal in the rare class but giving an apples-to-
    apples PR-AUC comparison.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    n = len(y_true)
    nat_rate = n_pos / max(n, 1)
    nat_pr = float(average_precision_score(y_true, y_score)) if n_pos > 0 else float("nan")

    if n_pos == 0 or n_neg == 0 or abs(nat_rate - target_pos_rate) < 1e-6:
        return {
            "natural_pos_rate":    round(nat_rate, 4),
            "target_pos_rate":     round(target_pos_rate, 4),
            "rebalanced_pos_rate": round(nat_rate, 4),
            "pr_auc_natural":      round(nat_pr, 4),
            "pr_auc_rebalanced":   round(nat_pr, 4),
            "rebalanced": False,
        }

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if nat_rate < target_pos_rate:
        # Downsample negatives — keep all positives, sample k negatives
        # n_pos / (n_pos + k) = target → k = n_pos * (1-target) / target
        k = int(round(n_pos * (1.0 - target_pos_rate) / target_pos_rate))
        k = min(k, n_neg)
        sampled_neg = rng.choice(neg_idx, size=k, replace=False)
        keep = np.concatenate([pos_idx, sampled_neg])
    else:
        # Downsample positives — keep all negatives, sample k positives
        # k / (k + n_neg) = target → k = n_neg * target / (1-target)
        k = int(round(n_neg * target_pos_rate / (1.0 - target_pos_rate)))
        k = min(k, n_pos)
        sampled_pos = rng.choice(pos_idx, size=k, replace=False)
        keep = np.concatenate([neg_idx, sampled_pos])

    keep.sort()
    y_sub = y_true[keep]
    s_sub = y_score[keep]
    pr_sub = float(average_precision_score(y_sub, s_sub))
    return {
        "natural_pos_rate":    round(nat_rate, 4),
        "target_pos_rate":     round(target_pos_rate, 4),
        "rebalanced_pos_rate": round(float(y_sub.mean()), 4),
        "pr_auc_natural":      round(nat_pr, 4),
        "pr_auc_rebalanced":   round(pr_sub, 4),
        "rebalanced": True,
        "kept_rows": int(len(keep)),
    }


def evaluate_baseline_classification(
    predictions: np.ndarray,
    scores: np.ndarray,
    y_true: pd.Series,
    name: str,
) -> dict:
    """
    Evaluate a classification floor (e.g. predict-majority-class).

    With a constant score vector ROC-AUC = 0.5 and PR-AUC = positive rate;
    we still compute them so the table is self-consistent.
    """
    acc = float(accuracy_score(y_true, predictions))
    prec = float(precision_score(y_true, predictions, zero_division=0))
    rec  = float(recall_score(y_true, predictions, zero_division=0))
    f1   = float(f1_score(y_true, predictions, zero_division=0))
    # roc_auc_score requires at least one unique score for both classes; with a
    # constant score we manually report the no-skill values.
    if len(np.unique(scores)) <= 1:
        roc = 0.5
        pr  = float(y_true.mean())
    else:
        roc = float(roc_auc_score(y_true, scores))
        pr  = float(average_precision_score(y_true, scores))

    metrics = {
        "name": name,
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "roc_auc":   round(roc, 4),
        "pr_auc":    round(pr, 4),
    }
    print(f"[eval] Baseline '{name}' -- Acc={acc:.4f}  P={prec:.4f}  "
          f"R={rec:.4f}  F1={f1:.4f}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")
    return metrics


def evaluate_baseline_regression(
    predictions: np.ndarray,
    y_true: pd.Series,
    y_binary: pd.Series,
    name: str,
) -> dict:
    """Evaluate a context baseline (predict-median or route-mean)."""
    mae  = float(mean_absolute_error(y_true, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))

    # Binary accuracy: is the prediction's threshold correct?
    pred_binary = (predictions > DELAY_THRESHOLD_MIN).astype(int)
    acc = float(accuracy_score(y_binary, pred_binary))

    metrics = {
        "name": name,
        "mae":  round(mae, 3),
        "rmse": round(rmse, 3),
        "binary_accuracy": round(acc, 4),
    }

    print(f"[eval] Baseline '{name}' -- MAE={mae:.3f}  RMSE={rmse:.3f}  "
          f"Binary Acc={acc:.4f}")

    return metrics
