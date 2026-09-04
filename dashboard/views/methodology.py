"""Methodology & audit page.

Surfaces the causal guardrails the pipeline enforces and the six issues a
forensic audit caught and fixed. None of this is visible in the teammate's
dashboard, but it is genuinely the project's strongest credibility signal in
a CSEN 140 demo: showing your work was reviewed, what was wrong, and how
each issue was resolved with code anchors.

Sources
-------
- `results/v2/final_test_metrics.json` for the prevalence numbers
- `writeups/final_project_writeup.md` §3.5 for the audit narrative
- `src/v2/*.py` for the code anchors (line ranges are stable as of audit run)
"""
import streamlit as st

from dashboard import loaders, theme


# ── Causal guardrails ───────────────────────────────────────────────────────
def _guardrails() -> None:
    st.subheader("Causal guardrails (defense-in-depth)")
    code_style = ("background:rgba(0,0,0,0.06);padding:1px 6px;border-radius:4px;"
                  "font-family:Menlo,monospace;color:#111;")

    cols = st.columns(3, gap="medium")
    cards = [
        {
            "title": "1. Column whitelist",
            "body": (
                f"<span style='{code_style}'>LOAD_COLS</span> in "
                f"<span style='{code_style}'>src/v2/config.py:45-60</span> "
                "lists the 14 raw BTS columns we read from CSV. Post-hoc "
                "fields like <span style='" + code_style + "'>DEP_DELAY</span>, "
                f"<span style='{code_style}'>TAXI_OUT</span>, "
                f"<span style='{code_style}'>WHEELS_OFF</span> "
                "are never even loaded into memory."
            ),
            "color": theme.STAGE_COLORS["A"],
        },
        {
            "title": "2. Column blacklist + runtime assertion",
            "body": (
                f"<span style='{code_style}'>FORBIDDEN_COLUMNS</span> in "
                f"<span style='{code_style}'>config.py:66-75</span> names "
                "every banned field, including the target. "
                f"<span style='{code_style}'>assert_no_leakage(feature_cols)</span> "
                "runs <strong>before every training call</strong> and fails "
                "loudly if any forbidden name appears in the feature list."
            ),
            "color": theme.STAGE_COLORS["B"],
        },
        {
            "title": "3. Defensive drop",
            "body": (
                "Once <span style='" + code_style + "'>arr_delay</span> has "
                "been consumed by target construction, historical-stat "
                "aggregation, and arrival-pressure computation, it is dropped "
                f"from the dataframe entirely (<span style='{code_style}'>"
                "features_stage_a.py:402-413</span> and "
                f"<span style='{code_style}'>features_stage_c.py:134-140</span>) "
                "so a future feature can't accidentally include it."
            ),
            "color": theme.STAGE_COLORS["C"],
        },
    ]
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    border-top: 4px solid {card['color']};
                    background: white;
                    border-radius: 8px;
                    padding: 1rem 1.1rem;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
                    height: 100%;
                    font-size:0.95rem;
                    line-height:1.55;
                ">
                  <div style="font-size:1.0rem;font-weight:600;color:#222;
                              margin-bottom:0.45rem;">
                    {card['title']}
                  </div>
                  <div style="color:#333;">{card['body']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── The six audit fixes ─────────────────────────────────────────────────────
def _audit_table() -> None:
    st.subheader("The six audit fixes")
    st.caption(
        "A forensic audit reviewed every stage of the pipeline before the "
        "final numbers were locked. The table below is the punch-list of "
        "issues caught (none were strict target leakage) and what code "
        "changed to resolve each. The numbers in **Stage A → B → C** and "
        "**Network propagation** above reflect the post-audit pipeline."
    )

    fixes = [
        {
            "n": 1,
            "severity": "🔴 Causal integrity",
            "issue": "Stage B `merge_asof(direction=\"nearest\", tol=2h)` could pull weather from up to 2 h *after* scheduled departure/arrival: a forecaster at gate-close does not have that data.",
            "fix": "`direction=\"backward\", tolerance=pd.Timedelta(hours=1)`",
            "where": "`src/v2/features_stage_b.py:119-138`",
            "impact": "Stage B val ROC-AUC moved 0.6575 → 0.6531 (≈0: the leakage was contributing almost nothing). Honesty preserved at near-zero empirical cost.",
        },
        {
            "n": 2,
            "severity": "🟠 Honesty",
            "issue": "PR-AUC reported only at natural test prevalence (17.3%). Val prevalence is 13.7%; the gap mechanically inflates test PR-AUC.",
            "fix": "New `pr_auc_at_common_base_rate` helper rebalances test to val prevalence by downsampling.",
            "where": "`src/v2/evaluate_v2.py:190` + `run_final_test_evaluation.py:164`",
            "impact": "Stage C test PR-AUC: 0.403 natural → 0.349 at val prev. Lift is still strong but no longer headline-doubled.",
        },
        {
            "n": 3,
            "severity": "🟠 Honesty",
            "issue": "The pre-audit 'ordinal recall' was actually the binary classifier thresholded into buckets, not an ordinal model.",
            "fix": "Real 4-class LightGBM (`objective=\"multiclass\"`, `class_weight=\"balanced\"`) on `delay_bucket`; QWK + adjacent-accuracy reported.",
            "where": "`src/v2/train_stage_a.py:147-176` + `evaluate_v2.py:94-156`",
            "impact": "Stage C ordinal: QWK = 0.191, adj-bucket acc. = 0.846. Real metric replaces a fictitious one.",
        },
        {
            "n": 4,
            "severity": "🟡 Robustness",
            "issue": "No classification floor surfaced in any results table. PR-AUC 0.403 looks impressive without context; what's the no-skill baseline?",
            "fix": "`predict_majority_class` baseline returns ROC-AUC = 0.5 and PR-AUC = positive base rate.",
            "where": "`src/v2/baselines.py:15-37`",
            "impact": "PR-AUC floor at val prevalence = 0.137; Stage C sits ~0.21 above it.",
        },
        {
            "n": 5,
            "severity": "🟡 Defense-in-depth",
            "issue": "`arr_delay` remained on dataframes after consumption. The `assert_no_leakage` check caught it at training time, but a future feature could still touch it.",
            "fix": "Defensive drop of `arr_delay` immediately after stat / pressure computation.",
            "where": "`features_stage_a.py:402-413` and `features_stage_c.py:134-140`",
            "impact": "Belt-and-braces: the runtime assertion is no longer the only line of defense.",
        },
        {
            "n": 6,
            "severity": "🟡 Robustness",
            "issue": "Train historical stats were expanding (per-day) but val/test stats were frozen at the train cutoff. The model trained on a growing series and was scored against a constant: a distributional mismatch.",
            "fix": "Val/test stats now use the same expanding backward-only helper as train, combining train + earlier splits.",
            "where": "`src/v2/features_stage_a.py:262-310`",
            "impact": "Matches what an operational forecaster sees at gate-close: a stat that ticks forward day by day.",
        },
    ]

    for fix in fixes:
        st.markdown(
            f"""
            <div style="margin:0.5rem 0 0.85rem 0;
                        background:white;
                        border-left:4px solid #cbd5e1;
                        border-radius:6px;padding:0.85rem 1.1rem;
                        font-size:0.93rem;line-height:1.5;color:#1f2937;">
              <div style="display:flex;justify-content:space-between;
                          align-items:baseline;margin-bottom:0.35rem;">
                <div style="font-weight:600;color:#111;">
                  #{fix['n']} &nbsp; {fix['issue']}
                </div>
                <div style="font-size:0.8rem;color:#475569;
                            background:#f1f5f9;border-radius:4px;
                            padding:2px 8px;white-space:nowrap;">
                  {fix['severity']}
                </div>
              </div>
              <div style="margin-bottom:0.25rem;">
                <strong>Fix:</strong> {fix['fix']}
              </div>
              <div style="margin-bottom:0.25rem;color:#475569;">
                <strong>Where:</strong>
                <code style="background:#f1f5f9;padding:1px 6px;border-radius:4px;">
                  {fix['where']}
                </code>
              </div>
              <div style="color:#1f2937;">
                <strong>Impact:</strong> {fix['impact']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "None of the six issues constituted strict target leakage: the "
        "pipeline's ROC-AUC numbers were honestly earned at every stage. "
        "The fixes tightened causal correctness (#1), honesty of reporting "
        "(#2, #3, #4), and robustness (#5, #6)."
    )


# ── Test-set prevalence callout ─────────────────────────────────────────────
def _prevalence_recap() -> None:
    fm = loaders.final_metrics()
    if fm is None:
        return
    val_pr = fm.get("val_pos_rate")
    test_pr = fm.get("test_pos_rate")
    if val_pr is None or test_pr is None:
        return
    st.markdown(
        f"""
        <div style="margin-top:0.5rem;padding:0.85rem 1.1rem;
                    background:#eef4ff;border-left:4px solid {theme.ACCENT};
                    border-radius:6px;color:#1f2937;font-size:0.93rem;
                    line-height:1.55;">
          <strong style="color:{theme.ACCENT};">Reminder on prevalence.</strong>
          Validation set positive rate (Sep–Oct 2024):
          <strong>{val_pr:.2%}</strong>. Test set positive rate (Nov–Dec 2024):
          <strong>{test_pr:.2%}</strong>. The gap (+{(test_pr-val_pr)*100:.1f}
          pp) reflects winter-holiday traffic + weather. PR-AUC depends on
          this prevalence; ROC-AUC does not. The audit's rebalanced PR-AUC
          column on the Stage Story page strips out the base-rate
          contribution.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Public render ───────────────────────────────────────────────────────────
def render() -> None:
    st.title("Methodology & audit")
    st.caption(
        "How SkyPredict v2 enforces causal correctness, and the six issues "
        "a forensic audit caught and fixed before the final numbers were "
        "locked."
    )
    _guardrails()
    st.write("")
    _prevalence_recap()
    st.divider()
    _audit_table()
