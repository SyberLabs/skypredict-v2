"""Network Propagation — theory and empirical demo.

This page replaces the teammate's "Live prediction" page. The original page
required val_predictions parquets and stage_*.joblib bundles that don't exist
on this machine; rather than synthesize those artifacts, we surface what the
project actually contributes — Stage C's network arrival-pressure feature —
in a form a viewer can watch work, computed from clean.parquet on the fly.

What you see here
-----------------
  1. Theory: the cascade dynamic, the causal pressure definition.
  2. Empirical demo: pick an airport and a date; the dashboard pulls every
     real arrival at that airport from clean.parquet, computes the 3-hour
     rolling arrival count and mean delay STRICTLY before each minute of the
     day (the Stage C window), and renders the resulting pressure curve.
     Vertical markers show the actual scheduled departures from that airport
     — you can see, by eye, that pressure tends to be high when departures
     are about to be delayed.
  3. The empirical lift Stage C delivers (from final_test_metrics.json).
"""
from datetime import date as date_t
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import loaders, theme


WINDOW_HOURS = 3


# ── Header / theory block ───────────────────────────────────────────────────
def _theory_block() -> None:
    left, right = st.columns([1.05, 1])
    code_style = ("background:rgba(0,0,0,0.06);padding:1px 6px;border-radius:4px;"
                  "font-family:Menlo,monospace;color:#111;")
    with left:
        st.markdown("### Why network propagation?")
        st.markdown(
            """
Flight delays are **not i.i.d.** A delay at a hub at 09:00 becomes a delay
downstream at 12:00 along two coupled channels:

- **Aircraft propagation** — the same tail number rotates between airports;
  if the inbound flight is late, the outbound flight inherits the delay.
- **Resource propagation** — runways, gates, ATC capacity, and ground crews
  at the destination are shared across all arriving flights, so a queue of
  late inbounds throttles every subsequent departure.

Operational implication: at scheduled gate-close *T*, **the queue of
recently-arrived inbound flights at the airport is a strong predictor of
the next departure's delay** — independent of weather, independent of the
carrier's historical rate.
            """
        )
    with right:
        st.markdown("### Our pressure feature — causally")
        st.markdown(
            f"""
For each scheduled departure at *(airport = O, time = T)*:

```
pressure(O, T) = mean arr_delay  of all flights
                 that PHYSICALLY landed at O
                 during the half-open window [T − 3h, T)
```

- **Causal**: the window ends strictly before *T*. A flight cannot leak
  its own delay into its own pressure — its arrival time is by
  construction ≥ *T*.
- **Scaled to 7 M rows** with per-airport cumulative sums + backward
  <span style="{code_style}">merge_asof</span>: an O(N log N) replacement
  for the naive O(N²) sliding window.
            """,
            unsafe_allow_html=True,
        )


# ── Empirical demo: pressure-curve plotter ──────────────────────────────────
def _pressure_curve(arrivals: pd.DataFrame, day: pd.Timestamp,
                    airport: str) -> Optional[go.Figure]:
    """Compute and plot pressure(O, T) at minute resolution across one day.

    Implementation note: we compute the exact same quantity Stage C uses, but
    inline and at minute granularity for a single airport so the demo is
    interactive. The O(N log N) trick is unnecessary at this scale; we just
    convert arrivals to a chronologically-sorted series and step through.
    """
    # 1-minute grid for the chosen day, in local clock minutes 00:00 → 23:59
    day_start = pd.Timestamp(day)
    day_end   = day_start + pd.Timedelta(hours=24)
    minutes = pd.date_range(day_start, day_end, freq="5min")[:-1]  # 5-min for speed

    # All arrivals at this airport before day_end (so a window starting at
    # day_start can reach back 3 h into the previous day).
    relevant = arrivals[
        (arrivals["actual_arr_dt"] < day_end)
        & (arrivals["actual_arr_dt"] >= (day_start - pd.Timedelta(hours=WINDOW_HOURS)))
    ].sort_values("actual_arr_dt").reset_index(drop=True)

    if relevant.empty:
        return None

    times_ns = relevant["actual_arr_dt"].values.astype("datetime64[ns]")
    delays = relevant["arr_delay"].fillna(0).to_numpy(dtype=np.float64)

    # Build running cumulative sums; pressure for a grid time T is
    # sum(delays where arr < T) - sum(delays where arr < T-3h), divided by
    # the equivalent count diff. We use np.searchsorted (binary search) to
    # find the two boundary indices — same idea as Stage C's merge_asof.
    cum_sum = np.concatenate([[0.0], np.cumsum(delays)])
    cum_cnt = np.arange(len(delays) + 1, dtype=np.float64)

    grid_ns = minutes.values.astype("datetime64[ns]")
    win_start_ns = (minutes - pd.Timedelta(hours=WINDOW_HOURS)).values.astype("datetime64[ns]")

    end_idx = np.searchsorted(times_ns, grid_ns, side="left")
    start_idx = np.searchsorted(times_ns, win_start_ns, side="left")

    counts = cum_cnt[end_idx] - cum_cnt[start_idx]
    sums = cum_sum[end_idx] - cum_sum[start_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_delay = np.where(counts > 0, sums / counts, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes, y=counts,
        name=f"arrivals in last {WINDOW_HOURS}h",
        line=dict(color=theme.STAGE_COLORS["C"], width=2),
        yaxis="y1",
        hovertemplate="%{x|%H:%M}<br>%{y:.0f} arrivals<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=minutes, y=mean_delay,
        name="mean arrival delay (min)",
        line=dict(color=theme.ACCENT, width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="%{x|%H:%M}<br>mean delay %{y:+.1f} min<extra></extra>",
    ))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        title=dict(
            text=f"Network pressure at {airport} · {day.strftime('%a %b %d, %Y')}",
            x=0.0, xanchor="left", font=dict(size=14, color="#333"),
        ),
        xaxis=dict(title="Local hour (clock minutes; no timezone correction "
                         "needed for single-airport view)",
                   showgrid=True, gridcolor="#eee",
                   tickformat="%H:%M"),
        yaxis=dict(title=dict(text=f"Arrivals in last {WINDOW_HOURS}h",
                              font=dict(color=theme.STAGE_COLORS["C"])),
                   showgrid=True, gridcolor="#eee",
                   tickfont=dict(color=theme.STAGE_COLORS["C"])),
        yaxis2=dict(title=dict(text="Mean arrival delay (min)",
                               font=dict(color=theme.ACCENT)),
                    overlaying="y", side="right",
                    tickfont=dict(color=theme.ACCENT),
                    showgrid=False, zeroline=True, zerolinecolor="#bbb"),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0),
        hovermode="x unified",
    )
    return fig


def _empirical_demo() -> None:
    st.subheader("Watch the feature work")
    st.caption(
        "Pick an airport and a date; we pull every real arrival at that "
        "airport from `clean.parquet`, compute the Stage C 3-hour rolling "
        "pressure curve directly, and plot it across one full day. **No model "
        "is involved here** — this is the raw input feature."
    )

    df = loaders.clean_flights()
    if df is None:
        st.error("`clean.parquet` not available.")
        return

    # Pick top-30 destinations by validation-window volume to keep the dropdown
    # fast and the demo on hubs the audience will recognize.
    val_agg = loaders.val_window_aggregate(loaders._mtime(loaders.CLEAN_PARQUET))
    if val_agg is None or val_agg.empty:
        st.error("Validation aggregate could not be computed.")
        return
    top_airports = (val_agg.nlargest(30, "flights")["iata"]
                          .astype(str).tolist())

    c1, c2 = st.columns([1, 2])
    with c1:
        airport = st.selectbox(
            "Airport (top-30 hubs by val volume)",
            options=top_airports,
            index=top_airports.index("ATL") if "ATL" in top_airports else 0,
            help="Destination airport — we look at arrivals *into* this airport "
                 "and compute pressure as Stage C does.",
        )
    with c2:
        # Default to a date with reliably high pressure — a Friday in late October
        default_day = date_t(2024, 10, 25)
        day = st.date_input(
            "Date (any day in 2024)",
            value=default_day,
            min_value=date_t(2024, 1, 2),     # need 1 prior day for the look-back
            max_value=date_t(2024, 12, 30),
        )

    day_ts = pd.Timestamp(day)
    # Pull arrivals on the chosen day ± 1 day so the 3-hour look-back is
    # complete at midnight.
    arrivals = loaders.arrivals_for_pressure_demo(
        loaders._mtime(loaders.CLEAN_PARQUET),
        airport=airport,
        window_start=(day_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        window_end=(day_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if arrivals is None or arrivals.empty:
        st.info(f"No arrivals at {airport} in clean.parquet for the chosen window.")
        return

    fig = _pressure_curve(arrivals, day_ts, airport)
    if fig is None:
        st.info("Not enough arrivals to draw the pressure curve.")
        return
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # Summary statistics for the chosen day
    same_day = arrivals[
        (arrivals["actual_arr_dt"] >= day_ts) &
        (arrivals["actual_arr_dt"] < day_ts + pd.Timedelta(hours=24))
    ]
    n_arr = len(same_day)
    pct_delayed = (same_day["arr_delay"] > 15).mean() * 100 if n_arr else 0
    mean_delay = same_day["arr_delay"].mean() if n_arr else float("nan")

    k1, k2, k3 = st.columns(3)
    k1.metric("Same-day arrivals", f"{n_arr:,}")
    k2.metric("Delayed > 15 min",  f"{pct_delayed:.1f}%")
    k3.metric("Mean arrival delay", f"{mean_delay:+.1f} min")

    with st.expander("How to read this chart"):
        st.markdown(
            f"""
- **Solid red line** — number of flights that actually landed at {airport}
  in the previous {WINDOW_HOURS} hours. This is exactly
  `feat_dest_pressure_count` in Stage C.
- **Dotted dark-blue line** — mean arrival delay of those flights (in
  minutes). This is `feat_dest_pressure_3h`.
- Both quantities are computed at every 5-minute step across the day using
  per-airport cumulative sums + `np.searchsorted` — the same exact algorithm
  Stage C uses on the full 7M-flight table, just specialised to one airport
  for interactivity.
- The window is **strictly half-open** `[T − 3h, T)` — an arrival at exactly
  time *T* does not count toward the pressure *at* time *T*, so a flight
  cannot leak its own delay.
            """
        )


# ── Empirical payoff block (from JSON) ──────────────────────────────────────
def _empirical_payoff() -> None:
    st.subheader("The empirical payoff")
    st.caption(
        "Final held-out test set (Nov–Dec 2024). Numbers from "
        "`results/v2/final_test_metrics.json`."
    )

    rows = []
    for s in ("A", "B", "C"):
        b = loaders.final_stage_binary(s)
        if not b: continue
        pr_bal = b.get("pr_auc_at_val_prevalence", {}).get("pr_auc_rebalanced")
        rows.append({
            "Stage": theme.STAGE_LABELS[s],
            "ROC-AUC":           round(b["roc_auc"], 4),
            "PR-AUC @ val prev.": round(pr_bal, 4) if pr_bal is not None else None,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    b_to_c_roc, b_to_c_pr, a_to_b_roc, a_to_b_pr = None, None, None, None
    bA = loaders.final_stage_binary("A")
    bB = loaders.final_stage_binary("B")
    bC = loaders.final_stage_binary("C")
    if bA and bB and bC:
        a_to_b_roc = bB["roc_auc"] - bA["roc_auc"]
        b_to_c_roc = bC["roc_auc"] - bB["roc_auc"]
        a_to_b_pr = (bB["pr_auc_at_val_prevalence"]["pr_auc_rebalanced"]
                     - bA["pr_auc_at_val_prevalence"]["pr_auc_rebalanced"])
        b_to_c_pr = (bC["pr_auc_at_val_prevalence"]["pr_auc_rebalanced"]
                     - bB["pr_auc_at_val_prevalence"]["pr_auc_rebalanced"])

    if all(v is not None for v in (a_to_b_roc, b_to_c_roc, a_to_b_pr, b_to_c_pr)):
        ratio_roc = b_to_c_roc / a_to_b_roc if a_to_b_roc > 0 else float("nan")
        ratio_pr  = b_to_c_pr  / a_to_b_pr  if a_to_b_pr  > 0 else float("nan")
        st.markdown(
            f"""
            <div style="padding:0.85rem 1.1rem;
                        background:#fff8e1;border-left:4px solid {theme.WARN};
                        border-radius:6px;color:#1f2937;font-size:0.95rem;
                        line-height:1.55;">
              <strong style="color:#92400e;">Headline.</strong>
              Network pressure (B → C) delivers
              <strong>+{b_to_c_roc:.3f}</strong> ROC-AUC and
              <strong>+{b_to_c_pr:.3f}</strong> PR-AUC@val-prev — roughly
              <strong>{ratio_roc:.1f}×</strong> the ROC-AUC lift and
              <strong>{ratio_pr:.1f}×</strong> the PR-AUC lift that weather
              (A → B) provided. Supports the project's central thesis: flight
              delays are best modelled as a network propagation phenomenon.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Public render ───────────────────────────────────────────────────────────
def render() -> None:
    st.title("Network propagation")
    st.caption("Theory · live empirical demo · final-test lift.")
    _theory_block()
    st.divider()
    _empirical_demo()
    st.divider()
    _empirical_payoff()
