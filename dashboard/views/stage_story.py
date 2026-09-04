"""Stage Story page — what each feature family buys us, A → B → C.

Compared to the teammate's version, this page does three things that matter
for the audit story:

  1. Shows PR-AUC at val-prevalence side by side with natural PR-AUC.
     The Nov-Dec test set has a higher delay base rate (17.3%) than the val
     set (13.75%); PR-AUC scales with prevalence, so reporting only the raw
     test number inflates the apparent lift.
  2. Draws the majority-class PR-AUC floor as a horizontal reference line on
     the PR-AUC chart so a reader can see "how far above no-skill" we sit.
  3. Surfaces the ordinal-multiclass metrics (QWK, adjacent-bucket accuracy,
     per-bucket precision/recall) which are present in
     final_test_metrics.json but invisible in the teammate's dashboard.

Everything here reads from JSON only — no model bundles required.
"""
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import loaders, theme

STAGE_KEYS = ["A", "B", "C"]


# ── Metric progression bars ─────────────────────────────────────────────────
def _metric_progression() -> None:
    final = loaders.final_metrics()
    if final is None:
        st.warning("Final test metrics not found.")
        return

    rows = []
    for stage in STAGE_KEYS:
        b = loaders.final_stage_binary(stage)
        r = loaders.final_stage_regression(stage)
        if b is None or r is None:
            continue
        pr_bal = b.get("pr_auc_at_val_prevalence", {}).get("pr_auc_rebalanced")
        rows.append({
            "stage_key": stage,
            "Stage": theme.STAGE_SHORT[stage],
            "ROC-AUC":            b["roc_auc"],
            "PR-AUC (natural)":   b["pr_auc"],
            "PR-AUC @ val prev.": pr_bal,
            "MAE":                r["mae"],
            "R²":                 r["r2"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No stage metrics present.")
        return
    colors = [theme.STAGE_COLORS[r["stage_key"]] for _, r in df.iterrows()]

    # Majority-class PR-AUC floor from Stage A's val baselines block.
    # PR-AUC under a random scorer = positive class rate.
    meta_a = loaders.stage_metrics("A")
    val_pos_rate = None
    if meta_a:
        val_pos_rate = (
            meta_a.get("baselines", {})
                  .get("predict_majority_class", {})
                  .get("pr_auc")
        )

    # 5-column metric grid: ROC, PR(natural), PR(@valprev), MAE, R²
    cols = st.columns(5)
    plan = [
        ("ROC-AUC",             True,  ":.3f", None),
        ("PR-AUC (natural)",    True,  ":.3f", val_pos_rate),
        ("PR-AUC @ val prev.",  True,  ":.3f", val_pos_rate),
        ("MAE",                 False, ":.2f", None),
        ("R²",                  True,  ":.3f", None),
    ]
    for (mname, higher_better, fmt, floor), col in zip(plan, cols):
        with col:
            fig = go.Figure(go.Bar(
                x=df["Stage"], y=df[mname],
                marker_color=colors,
                text=[f"{v:.3f}" if mname != "MAE" else f"{v:.2f}"
                      for v in df[mname]],
                textposition="outside",
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            ))
            if floor is not None and floor > 0:
                fig.add_hline(
                    y=floor, line_dash="dot", line_color="#999",
                    annotation_text=f"no-skill floor ({floor:.3f})",
                    annotation_position="bottom right",
                    annotation_font=dict(color="#666", size=10),
                )
            fig.update_layout(
                title=dict(text=mname, x=0.5, xanchor="center",
                           font=dict(size=14)),
                height=260,
                margin=dict(l=8, r=8, t=40, b=20),
                yaxis=dict(showgrid=True, gridcolor="#eee"),
                plot_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

            # Deltas vs Stage A (the baseline). Computed from the raw JSON
            # values, not the rounded display strings, so the arithmetic is
            # exact even when bars look identical to 3 decimals.
            base = df[df["stage_key"] == "A"][mname].iloc[0]
            chips = []
            for _, row in df.iterrows():
                if row["stage_key"] == "A":
                    continue
                d = row[mname] - base
                good = (d >= 0) == higher_better
                arrow = "🟢" if good else "🔴"
                sign = "+" if d >= 0 else ""
                chips.append(f"{arrow} {row['Stage']}: {sign}{d:.3f}")
            st.caption("vs A · " + " · ".join(chips))


# ── Lift narrative ──────────────────────────────────────────────────────────
def _lift_narrative() -> None:
    def _roc(stage): return (loaders.final_stage_binary(stage) or {}).get("roc_auc")
    def _mae(stage): return (loaders.final_stage_regression(stage) or {}).get("mae")
    def _prv(stage):
        b = loaders.final_stage_binary(stage) or {}
        return b.get("pr_auc_at_val_prevalence", {}).get("pr_auc_rebalanced")

    def _d(a, b, fmt=".3f"):
        if a is None or b is None: return "—"
        d = a - b
        return f"{d:+{fmt}}"

    a_to_b_roc = _d(_roc("B"), _roc("A"))
    b_to_c_roc = _d(_roc("C"), _roc("B"))
    a_to_b_mae = _d(_mae("B"), _mae("A"), ".2f")
    b_to_c_mae = _d(_mae("C"), _mae("B"), ".2f")
    a_to_b_pr  = _d(_prv("B"), _prv("A"))
    b_to_c_pr  = _d(_prv("C"), _prv("B"))

    callout_style = ("padding:0.75rem 1rem;border-radius:6px;color:#1f2937;"
                     "font-size:0.95rem;line-height:1.5;")
    code_style = ("background:rgba(0,0,0,0.06);padding:1px 6px;border-radius:4px;"
                  "font-family:Menlo,monospace;color:#111;")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div style="{callout_style}
                        border-left:4px solid {theme.STAGE_COLORS['B']};
                        background:#f6fff6;">
              <span style="font-weight:700;color:#15803d;">
                + Weather (A → B)
              </span><br>
              ROC-AUC: <span style="{code_style}">{a_to_b_roc}</span> ·
              PR-AUC@val: <span style="{code_style}">{a_to_b_pr}</span> ·
              MAE: <span style="{code_style}">{a_to_b_mae} min</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="{callout_style}
                        border-left:4px solid {theme.STAGE_COLORS['C']};
                        background:#fff6f6;">
              <span style="font-weight:700;color:#b91c1c;">
                + Network propagation (B → C)
              </span><br>
              ROC-AUC: <span style="{code_style}">{b_to_c_roc}</span> ·
              PR-AUC@val: <span style="{code_style}">{b_to_c_pr}</span> ·
              MAE: <span style="{code_style}">{b_to_c_mae} min</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # The headline insight
    if _roc("C") and _roc("A") and _prv("C") and _prv("A"):
        roc_lift = _roc("C") - _roc("A")
        pr_lift  = _prv("C") - _prv("A")
        st.markdown(
            f"""
            <div style="margin-top:0.6rem;padding:0.8rem 1rem;
                        background:#fff8e1;border-left:4px solid {theme.WARN};
                        border-radius:6px;color:#1f2937;font-size:0.95rem;">
              <strong style="color:#92400e;">Headline:</strong>
              Network arrival-pressure features deliver roughly twice the
              ROC-AUC and PR-AUC lift of hourly weather. Stage A → C is
              <span style="{code_style}">+{roc_lift:.3f}</span> ROC-AUC and
              <span style="{code_style}">+{pr_lift:.3f}</span> PR-AUC at val
              prevalence — supporting the project's network-propagation thesis.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Why the PR-AUC re-balance matters ───────────────────────────────────────
def _prevalence_note() -> None:
    final = loaders.final_metrics()
    if final is None:
        return
    val_pr = final.get("val_pos_rate")
    test_pr = final.get("test_pos_rate")
    if val_pr is None or test_pr is None:
        return
    st.markdown(
        f"""
        <div style="padding:0.8rem 1.05rem;background:#eef4ff;
                    border-left:4px solid {theme.ACCENT};
                    border-radius:6px;color:#1f2937;font-size:0.92rem;
                    line-height:1.55;">
          <strong style="color:{theme.ACCENT};">Why two PR-AUC columns?</strong>
          Validation prevalence is <strong>{val_pr:.1%}</strong> delayed
          (Sep–Oct), test prevalence is <strong>{test_pr:.1%}</strong>
          (Nov–Dec — winter holidays). PR-AUC is sensitive to positive base
          rate: reporting only natural-prevalence PR-AUC would mix the model's
          improvement with the test set's higher delay floor. The
          <em>PR-AUC @ val prev.</em> column downsamples test positives to
          match the val rate, isolating the model lift. ROC-AUC, which is
          prevalence-invariant, is reported as-is.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Test-set table (sourced 100% from JSON) ─────────────────────────────────
def _full_test_table() -> None:
    final = loaders.final_metrics()
    if final is None:
        return
    rows = []
    for stage in STAGE_KEYS:
        b = loaders.final_stage_binary(stage)
        r = loaders.final_stage_regression(stage)
        if b is None or r is None:
            continue
        pr_bal = b.get("pr_auc_at_val_prevalence", {}).get("pr_auc_rebalanced")
        rows.append({
            "Stage":          theme.STAGE_LABELS[stage],
            "Accuracy":       round(b["accuracy"], 4),
            "Precision":      round(b["precision"], 4),
            "Recall":         round(b["recall"], 4),
            "F1":             round(b["f1"], 4),
            "ROC-AUC":        round(b["roc_auc"], 4),
            "PR-AUC (nat.)":  round(b["pr_auc"], 4),
            "PR-AUC @ valpr": round(pr_bal, 4) if pr_bal is not None else None,
            "MAE (min)":      round(r["mae"], 2),
            "RMSE":           round(r["rmse"], 2),
            "R²":             round(r["r2"], 4),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Held-out test set (Nov–Dec 2024, 1,156,765 flights). "
               "Numbers come directly from `results/v2/final_test_metrics.json`.")


# ── Ordinal multiclass panel ────────────────────────────────────────────────
def _ordinal_panel() -> None:
    st.subheader("Ordinal · 4-class delay-bucket classifier")
    st.caption(
        "Real multiclass LightGBM (`objective='multiclass'`, `class_weight='balanced'`) "
        "on `delay_bucket` ∈ {on_time, minor, major, severe}. Replaces the "
        "pre-audit 'ordinal recall' that was actually a thresholded binary slice."
    )

    rows = []
    for stage in STAGE_KEYS:
        o = loaders.final_stage_ordinal(stage)
        if not o: continue
        rows.append({
            "Stage": theme.STAGE_SHORT[stage],
            "stage_key": stage,
            "Accuracy":         o["accuracy"],
            "Macro-F1":         o["macro_f1"],
            "QWK":              o["quadratic_weighted_kappa"],
            "Adj-bucket acc.":  o["adjacent_bucket_accuracy"],
        })
    if not rows:
        st.info("Ordinal metrics not in final_test_metrics.json.")
        return
    df = pd.DataFrame(rows)
    colors = [theme.STAGE_COLORS[r["stage_key"]] for _, r in df.iterrows()]

    cols = st.columns(4)
    plan = [
        ("Accuracy",         "Accuracy"),
        ("Macro-F1",         "Macro-F1"),
        ("QWK",              "Quad-weighted κ"),
        ("Adj-bucket acc.",  "Adj-bucket accuracy"),
    ]
    for (col_name, title), col in zip(plan, cols):
        with col:
            fig = go.Figure(go.Bar(
                x=df["Stage"], y=df[col_name],
                marker_color=colors,
                text=[f"{v:.3f}" for v in df[col_name]],
                textposition="outside",
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor="center",
                           font=dict(size=13)),
                height=230, margin=dict(l=8, r=8, t=40, b=20),
                yaxis=dict(showgrid=True, gridcolor="#eee"),
                plot_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

    # Stage C per-bucket detail
    o_c = loaders.final_stage_ordinal("C")
    if o_c and "per_bucket" in o_c:
        st.markdown("**Stage C per-bucket precision / recall on the test set**")
        pb_rows = []
        for label, vals in o_c["per_bucket"].items():
            pb_rows.append({
                "Bucket": label,
                "Precision": round(vals.get("precision") or 0, 4),
                "Recall":    round(vals.get("recall") or 0, 4),
                "Support":   f"{vals.get('support', 0):,}",
            })
        st.dataframe(pd.DataFrame(pb_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Severe-bucket precision stays low (0.04) — predicting *magnitude* "
            "before departure is genuinely hard, dominated by post-departure "
            "operational events the model can't see. Adjacent-bucket accuracy "
            "(0.846) shows the model lands within one bucket of the truth most "
            "of the time."
        )


# ── Public render ───────────────────────────────────────────────────────────
def render() -> None:
    st.title("Stage A → B → C")
    st.caption(
        "Each stage layers on a new feature family. All numbers below come "
        "from the **held-out test set (Nov–Dec 2024, 1.16 M flights)**, "
        "measured on LightGBM."
    )

    _metric_progression()
    st.write("")
    _lift_narrative()
    st.write("")
    _prevalence_note()
    st.divider()

    st.subheader("Full test-set table")
    _full_test_table()
    st.divider()

    _ordinal_panel()
