"""Centralised color palette so every chart speaks the same visual language.

Lifted from teammate's dashboard; the per-stage colors carry semantic weight
(A=blue baseline, B=green +weather, C=red +network) and are reused for the
delta arrows on the Stage Story page.
"""

STAGE_COLORS = {
    "A": "#1f77b4",
    "B": "#2ca02c",
    "C": "#d62728",
}

STAGE_LABELS = {
    "A": "Stage A · Flight only",
    "B": "Stage B · + Weather",
    "C": "Stage C · + Network",
}

STAGE_SHORT = {"A": "Stage A", "B": "Stage B", "C": "Stage C"}

MODEL_COLORS = {
    "logistic_regression": "#9467bd",
    "lightgbm_clf":        "#ff7f0e",
    "lightgbm_reg":        "#17becf",
    "lightgbm_ordinal":    "#8c564b",
    "baseline":            "#7f7f7f",
}

ACCENT = "#0f4c81"
MUTED  = "#6c757d"
GOOD   = "#2ca02c"
BAD    = "#d62728"
WARN   = "#f59e0b"
