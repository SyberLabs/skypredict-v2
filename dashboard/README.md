# SkyPredict v2 - Dashboard

Streamlit dashboard for the SkyPredict v2 flight-delay pipeline. Runs against
the artifacts the audited pipeline has already written to disk - **no pipeline
re-runs required**.

## Run

From the project root (`d:/ml_hw/`):

```bash
streamlit run dashboard/app.py
```

The app opens at <http://localhost:8501>. The sidebar reports which artifacts
are loaded; if anything is missing it lists exactly which files.

## Pages

1. **Overview** - hero card, KPI strip (total flights, best test ROC-AUC,
   Stage C PR-AUC at both prevalences, MAE vs baseline), chronological-split
   timeline, validation-window delay distribution, U.S. airport map (top-25
   hubs labeled), and three per-stage cards with feature counts and final
   test ROC-AUC.
2. **Stage A → B → C** - five-bar metric progression (ROC-AUC, PR-AUC
   natural, PR-AUC at val prevalence, MAE, R²) with the majority-class floor
   drawn as a reference line, lift-narrative callouts, the full test-set
   table, and a dedicated panel for the 4-class ordinal LightGBM (QWK,
   adjacent-bucket accuracy, per-bucket P/R).
3. **Network propagation** - the project's contribution. Theory + a *live*
   demo: pick any of the top-30 hubs and any 2024 date, the dashboard pulls
   every real arrival at that airport from `clean.parquet`, computes the
   Stage C 3-hour rolling pressure feature at 5-minute resolution, and
   draws both the arrival count and mean-delay curves across the full day.
   No model is involved - this is the raw feature.
4. **Methodology & audit** - three causal-guardrail cards (whitelist,
   blacklist + assertion, defensive drop), prevalence recap, and the
   six audit fixes with file:line code anchors and per-fix impact notes.

## What this dashboard reads (and does NOT need)

**Required and present on disk:**

- `results/v2/metrics.json` · `metrics_stage_b.json` · `metrics_stage_c.json` ·
  `final_test_metrics.json` - JSON metric files from the audited runs.
- `data/processed_v2/clean.parquet` - 6.96 M cleaned flights.
- `data/processed_v2/airports.csv` - IATA → lat/lon/timezone for the map.

**Deliberately NOT required:**

- `val_predictions_stage_*.parquet` - the teammate's dashboard needed these
  to render confusion matrices and live per-flight scoring. We don't have
  them on this machine; we replaced live-prediction with the
  *Network propagation* page that demonstrates the feature directly.
- `stage_*.joblib` model bundles - same reason.
- `weather_plots/*.png` - the audited writeups already document weather
  behaviour; we surface it via the Stage Story page's lift narrative.

## File layout

```
dashboard/
├── app.py                              # Streamlit entrypoint + sidebar nav
├── loaders.py                          # Cached artifact readers
├── theme.py                            # Stage + model color palette
└── views/
    ├── __init__.py
    ├── overview.py                     # KPIs, map, delay distribution
    ├── stage_story.py                  # A→B→C progression + ordinal panel
    ├── network_propagation.py          # Theory + live pressure demo
    └── methodology.py                  # Causal guardrails + audit fixes
```

## Design notes

- Every chart is wrapped in a white card via `app.py` global CSS so axis
  text remains legible regardless of Streamlit's light/dark theme.
- All cached readers key on `os.path.getmtime(path)` - re-running the
  pipeline triggers a cache refresh on next page navigation.
- The network-propagation pressure curve is computed with the same
  algorithm Stage C uses on the full table (per-airport cumulative sums
  + `np.searchsorted` boundary lookups), just specialised to one airport
  and 5-minute resolution so the demo is interactive.
- The dashboard reports PR-AUC at **both** natural test prevalence
  (17.3%) and rebalanced to val prevalence (13.7%). This is the audit's
  most consequential change - without it, the natural test PR-AUC
  inflates the apparent lift by a base-rate artifact.
