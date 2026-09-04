"""Overview page: the entry surface of the demo.

Tells a cold visitor three things in order:
  1. What SkyPredict v2 is and why network propagation matters (hero)
  2. What we actually fed the model (data shape + KPIs from final_test_metrics.json)
  3. What the pipeline progression buys us (three side-by-side stage cards)

Everything on this page is read from JSON + clean.parquet. No model joblibs
or val-prediction parquets are required: those don't exist on this machine.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import loaders, theme


# ── Hero ────────────────────────────────────────────────────────────────────
def _hero() -> None:
    st.markdown(
        f"""
        <div style="
            padding: 1.5rem 2rem;
            background: linear-gradient(135deg, {theme.ACCENT} 0%, #1d3557 100%);
            border-radius: 14px;
            color: white;
            margin-bottom: 1.25rem;
        ">
          <div style="font-size:0.8rem;letter-spacing:0.15em;opacity:0.7;
                      text-transform:uppercase;margin-bottom:0.4rem;">
            CSEN 140 · Data Mining and Machine Learning · Public portfolio artifact
          </div>
          <div style="font-size:2.6rem;font-weight:700;line-height:1.1;
                      margin-bottom:0.4rem;">
            SkyPredict v2 ✈️
          </div>
          <div style="font-size:1.15rem;opacity:0.9;max-width:780px;">
            Pre-departure flight delay prediction as a
            <strong>network propagation problem</strong>: schedule, weather,
            and rolling airport backlog layered stage by stage, with strict
            causal guarantees.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── KPI strip ───────────────────────────────────────────────────────────────
def _kpi_cards() -> None:
    final = loaders.final_metrics()
    meta_a = loaders.stage_metrics("A")
    if final is None or meta_a is None:
        st.warning("Run the pipeline first: metric artifacts are missing.")
        return

    split = meta_a["split"]
    total = split["train_rows"] + split["val_rows"] + split["test_rows"]

    best_stage, best = None, -1.0
    for s in ("A", "B", "C"):
        b = loaders.final_stage_binary(s)
        if b and b["roc_auc"] > best:
            best, best_stage = b["roc_auc"], s

    reg_c = loaders.final_stage_regression("C")
    mae_c = reg_c["mae"] if reg_c else None

    # Baseline reference for MAE delta: pull from Stage C val baselines block
    baseline_mae = None
    metrics_c = loaders.stage_metrics("C")
    if metrics_c:
        baseline_mae = metrics_c.get("baselines", {}).get("predict_median", {}).get("mae")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Flights modeled",
        f"{total/1_000_000:.2f} M",
        delta=f"{split['train_rows']:,} train · {split['val_rows']:,} val "
              f"· {split['test_rows']:,} test",
        delta_color="off",
    )
    c2.metric(
        "Best test ROC-AUC",
        f"{best:.3f}" if best > 0 else ": ",
        delta=f"Stage {best_stage}" if best_stage else None,
        delta_color="off",
    )
    pr_c = loaders.final_stage_binary("C")
    if pr_c:
        pr_bal = pr_c.get("pr_auc_at_val_prevalence", {}).get("pr_auc_rebalanced")
        c3.metric(
            "Stage C PR-AUC (test)",
            f"{pr_c['pr_auc']:.3f}",
            delta=(f"{pr_bal:.3f} at val prevalence" if pr_bal is not None
                   else None),
            delta_color="off",
        )
    delta_mae = (f"{baseline_mae - mae_c:+.2f} min vs median baseline"
                 if baseline_mae and mae_c else None)
    c4.metric(
        "Stage C MAE (test)",
        f"{mae_c:.2f} min" if mae_c else ": ",
        delta=delta_mae,
        delta_color="off",
    )


# ── Project blurb + temporal split ──────────────────────────────────────────
def _project_blurb() -> None:
    st.subheader("What this project does")
    st.markdown(
        """
A three-stage flight-delay model where each stage layers on a new feature
family, and we measure exactly what each one buys us:

- **Stage A: Flight only.** Schedule, route, calendar, and *backward-only*
  expanding historical delay rates per carrier / origin / dest / route.
- **Stage B: + Weather.** NOAA-ISD hourly observations joined via
  **timezone-aware, backward-only** `merge_asof` (1-hour tolerance) so no
  future weather can leak into the prediction.
- **Stage C: + Network propagation.** A vectorized 3-hour rolling
  arrival-delay *pressure* feature at both origin and destination: the
  project's main contribution.

Every feature is **causal**: a forecaster at scheduled gate-close has access
to all of it. A `FORBIDDEN_COLUMNS` runtime assertion blocks post-hoc fields
from entering the feature matrix before any model is trained.
        """
    )


def _temporal_split() -> None:
    st.subheader("Temporal split")
    meta = loaders.stage_metrics("A")
    counts = meta["split"] if meta else {}
    df = pd.DataFrame([
        {"Phase": "Train", "Start": "2024-01-01", "End": "2024-09-01",
         "n": counts.get("train_rows", 0)},
        {"Phase": "Val",   "Start": "2024-09-01", "End": "2024-11-01",
         "n": counts.get("val_rows", 0)},
        {"Phase": "Test",  "Start": "2024-11-01", "End": "2025-01-01",
         "n": counts.get("test_rows", 0)},
    ])
    df["Label"] = df.apply(lambda r: f"{r['n']:,} flights", axis=1)
    fig = px.timeline(
        df, x_start="Start", x_end="End", y="Phase",
        color="Phase",
        color_discrete_map={"Train": theme.STAGE_COLORS["A"],
                            "Val":   theme.STAGE_COLORS["B"],
                            "Test":  theme.STAGE_COLORS["C"]},
        text="Label",
        category_orders={"Phase": ["Test", "Val", "Train"]},
    )
    fig.update_traces(
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="white", size=13),
        hovertemplate="<b>%{y}</b><br>%{base|%b %d} → %{x|%b %d}<extra></extra>",
    )
    fig.update_layout(
        height=210, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
        xaxis=dict(range=["2024-01-01", "2024-12-31"], showgrid=True,
                   gridcolor="#eee", tickformat="%b", dtick="M1"),
        yaxis=dict(title=None),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption(
        "Strictly chronological. Stage A's historical delay rates expand "
        "day-by-day; every val/test row sees only history strictly earlier "
        "than its own date: no random k-fold, no temporal overlap."
    )


# ── Problem space ───────────────────────────────────────────────────────────
def _delay_distribution() -> None:
    """Histogram of validation-window delays computed live from clean.parquet."""
    y = loaders.val_window_delays()
    if y is None:
        st.info("clean.parquet not found.")
        return
    delayed_pct = float(np.mean(y > 15))
    median = float(np.median(y))
    p90 = float(np.percentile(y, 90))

    fig = go.Figure(go.Histogram(
        x=y, xbins=dict(start=-60, end=300, size=10),
        marker_color=theme.ACCENT,
        opacity=0.92,
        hovertemplate="%{x} min · %{y:,} flights<extra></extra>",
    ))
    fig.add_vline(
        x=15, line_dash="dash", line_color=theme.BAD,
        annotation_text="15-min delay threshold",
        annotation_position="top right",
        annotation_font=dict(color=theme.BAD, size=11),
    )
    fig.add_vline(
        x=0, line_dash="dot", line_color="#999",
        annotation_text="on time", annotation_position="top left",
        annotation_font=dict(color="#999", size=11),
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Arrival delay (minutes, capped to [-60, 300])",
                   showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Flights", showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption(
        f"Validation period (Sep–Oct 2024). **{delayed_pct:.1%}** of flights "
        f"cross the 15-min delay threshold · median {median:+.0f} min · "
        f"90th percentile {p90:+.0f} min. Long right tail = asymmetric loss."
    )


def _airport_map() -> None:
    """Validation-window origin map: marker size = volume, color = delay rate."""
    by_origin = loaders.val_window_aggregate(loaders._mtime(loaders.CLEAN_PARQUET))
    airports = loaders.airports()
    if by_origin is None or airports is None:
        return
    df = by_origin.merge(
        airports.reset_index()[["iata", "lat", "lon"]],
        on="iata", how="left",
    ).dropna(subset=["lat", "lon"])

    df["size"] = np.sqrt(df["flights"]).clip(lower=4, upper=32)
    # Only label the largest hubs: 348 overlapping IATA codes turn the map
    # into noise. customdata carries the full hover string independently.
    top25 = df.nlargest(25, "flights")["iata"].tolist()
    df["label"] = df["iata"].where(df["iata"].isin(top25), "")
    df["hover"] = (
        df["iata"] + "<br>" + df["flights"].astype(str) + " val departures"
        + "<br>" + df["delayed_rate"].apply(lambda x: f"{x:.0%} delayed")
        + "<br>" + df["mean_delay"].apply(lambda x: f"avg {x:+.1f} min")
    )

    fig = go.Figure(go.Scattergeo(
        lon=df["lon"], lat=df["lat"],
        text=df["label"],                # what's drawn on the map
        customdata=df["hover"],          # rich hover content
        hovertemplate="%{customdata}<extra></extra>",
        mode="markers+text",
        marker=dict(
            size=df["size"],
            color=df["delayed_rate"],
            colorscale=[(0.0, "#2ca02c"), (0.5, "#f4a261"), (1.0, "#d62728")],
            cmin=0.05, cmax=0.30,
            colorbar=dict(title=dict(text="Delay rate", side="right"),
                          tickformat=".0%", thickness=12, len=0.7),
            line=dict(width=0.6, color="white"),
            opacity=0.92,
        ),
        textfont=dict(size=9, color="#333"),
        textposition="top center",
    ))
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        geo=dict(scope="usa", projection=dict(type="albers usa"),
                 showland=True, landcolor="#f7f7f9",
                 subunitcolor="#dadada", countrycolor="#dadada"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption(
        "Origin airports in the validation window (Sep–Oct 2024). "
        "Marker area ∝ √(departures); color = on-time vs. delayed rate. "
        "Only the top-25 hubs by volume are labeled to avoid clutter."
    )


# ── Pipeline cards ──────────────────────────────────────────────────────────
def _pipeline_cards() -> None:
    cards = [
        {
            "stage": "A",
            "title": "Stage A · Flight only",
            "tagline": "Schedule, route, calendar, historical rates",
            "examples": "carrier_hist_delay_rate · route_hist_avg_delay · "
                        "dep_minutes · is_holiday",
            "n_features": 19,
        },
        {
            "stage": "B",
            "title": "Stage B · + Weather",
            "tagline": "Hourly NOAA-ISD obs joined backward-only (1h)",
            "examples": "origin_wx_visibility · dest_wx_is_ifr · "
                        "origin_wx_wspd · is_precip",
            "n_features": 45,
        },
        {
            "stage": "C",
            "title": "Stage C · + Network",
            "tagline": "3-hour rolling arrival-delay pressure at origin & dest",
            "examples": "origin_pressure_3h · dest_pressure_3h · "
                        "origin_pressure_count",
            "n_features": 49,
        },
    ]

    cols = st.columns(3, gap="medium")
    for col, card in zip(cols, cards):
        b = loaders.final_stage_binary(card["stage"])
        roc_str = f"{b['roc_auc']:.3f}" if b else ": "
        color = theme.STAGE_COLORS[card["stage"]]
        with col:
            st.markdown(
                f"""
                <div style="
                    border-top: 4px solid {color};
                    background: white;
                    border-radius: 8px;
                    padding: 1rem 1.1rem;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
                    height: 100%;
                ">
                  <div style="display:flex;align-items:baseline;
                              justify-content:space-between;margin-bottom:0.4rem;">
                    <div style="font-size:1.05rem;font-weight:600;color:#222;">
                      {card['title']}
                    </div>
                    <div style="font-size:0.75rem;color:#888;
                                background:#f4f4f6;border-radius:4px;
                                padding:2px 8px;">
                      {card['n_features']} features
                    </div>
                  </div>
                  <div style="color:#555;font-size:0.92rem;margin-bottom:0.55rem;">
                    {card['tagline']}
                  </div>
                  <div style="color:#888;font-size:0.78rem;
                              font-family:Menlo, monospace;line-height:1.5;
                              margin-bottom:0.7rem;">
                    {card['examples']}
                  </div>
                  <div style="border-top:1px solid #eee;padding-top:0.6rem;
                              display:flex;align-items:center;
                              justify-content:space-between;">
                    <span style="font-size:0.78rem;color:#777;">
                      Test ROC-AUC
                    </span>
                    <span style="font-size:1.1rem;font-weight:600;color:{color};">
                      {roc_str}
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Closing callout ─────────────────────────────────────────────────────────
def _closing_callout() -> None:
    st.markdown(
        f"""
        <div style="
            background: #fffbe6;
            border-left: 4px solid {theme.WARN};
            padding: 1rem 1.2rem;
            border-radius: 6px;
            margin-top: 0.5rem;
            color: #1f2937;
            font-size: 0.95rem;
            line-height: 1.55;
        ">
          <span style="color:#92400e;font-weight:700;">
            Why this framing is interesting.
          </span>
          <span>
            Flight delays don't exist in isolation: a storm in Atlanta at
            9 am ripples through Chicago at noon and Denver at 3 pm. Stage C's
            3-hour rolling arrival pressure captures that ripple with a
            vectorized cumulative-sum window join that runs on 7 M flights in
            seconds. The next page quantifies what each stage layers on; the
            <em>Network propagation</em> page lets you watch the feature work.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Public render ───────────────────────────────────────────────────────────
def render() -> None:
    _hero()
    _kpi_cards()
    st.write("")

    left, right = st.columns([1.1, 1])
    with left:
        _project_blurb()
    with right:
        _temporal_split()

    st.divider()
    st.subheader("The problem space")
    st.caption(
        "Validation period (Sep–Oct 2024). Modeled flights only: "
        "cancellations and diversions are dropped at load time."
    )
    cL, cR = st.columns([1, 1.05])
    with cL:
        _delay_distribution()
    with cR:
        _airport_map()

    st.divider()
    st.subheader("The pipeline · A → B → C")
    st.caption(
        "Each card shows what the stage layers on, example feature names, "
        "and its final-test ROC-AUC. Tap **Stage A → B → C** in the sidebar "
        "to see how the metrics actually move."
    )
    _pipeline_cards()
    st.write("")
    _closing_callout()
