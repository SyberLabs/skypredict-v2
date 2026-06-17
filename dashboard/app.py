"""SkyPredict v2 dashboard — Streamlit entrypoint.

Run from the project root:

    streamlit run dashboard/app.py

Pages
-----
1. Overview              — hero, KPIs, temporal split, delay distribution, map
2. Stage A → B → C       — final-test metric progression with the audit's
                            PR-AUC@val-prevalence column + majority-class floor
3. Network propagation   — the project's contribution: theory + a live demo
                            of the 3-hour arrival pressure feature computed
                            directly from clean.parquet (no model required)
4. Methodology & audit   — causal guardrails + the six audit fixes,
                            file:line linked

This dashboard intentionally avoids depending on val_prediction parquets or
joblib model bundles — it runs against the artifacts the audited pipeline
already produced on disk.
"""
import os
import sys

import plotly.io as pio
import streamlit as st

# Make `dashboard` importable when launched via `streamlit run dashboard/app.py`.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Plotly renders into a white card so its axis labels are readable regardless of
# Streamlit's light/dark theme.
pio.templates.default = "plotly_white"

from dashboard import loaders
from dashboard.views import overview, stage_story, network_propagation, methodology


def _page_config() -> None:
    st.set_page_config(
        page_title="SkyPredict v2",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _global_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        [data-testid="stMetricValue"] { font-size: 2rem; }
        /* White card around every Plotly chart so labels never sit on dark bg */
        div[data-testid="stPlotlyChart"] {
            background: white;
            border-radius: 8px;
            padding: 6px 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


PAGES = [
    "Overview",
    "Stage A → B → C",
    "Network propagation",
    "Methodology & audit",
]


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("### SkyPredict v2")
        st.caption("Flight delay prediction · CSEN 140 · Public portfolio artifact")
        page = st.radio(
            "Navigate",
            options=PAGES,
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("Artifact status")
        ok, missing = loaders.all_artifacts_present()
        if ok:
            st.success("All artifacts loaded.")
        else:
            st.warning(f"{len(missing)} artifact(s) missing.")
            with st.expander("Missing files"):
                for m in missing:
                    st.code(m, language="text")

        st.divider()
        st.caption("Mateo Robles · Applied ML / data systems project")
    return page


def main() -> None:
    _page_config()
    _global_styles()
    page = _sidebar()
    if page == "Overview":
        overview.render()
    elif page == "Stage A → B → C":
        stage_story.render()
    elif page == "Network propagation":
        network_propagation.render()
    elif page == "Methodology & audit":
        methodology.render()


if __name__ == "__main__":
    main()
