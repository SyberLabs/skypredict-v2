# SkyPredict V2

Leakage-aware flight delay prediction under strict pre-departure observability constraints.

## Summary

SkyPredict V2 is an applied machine-learning pipeline for predicting U.S. domestic flight delays before departure. It uses 6.96M BTS 2024 flight records and progressively adds feature groups across three stages: flight metadata, backward-joined hourly weather, and 3-hour rolling airport network pressure. The project was built for CSEN 140 at Santa Clara University and emphasizes causal feature availability, audit fixes, staged ablation, and honest reporting of prevalence-sensitive metrics.

## What It Demonstrates

- End-to-end applied ML pipeline design over a multi-million-row dataset.
- Leakage-aware feature engineering with whitelist, blacklist, and runtime assertion checks.
- Staged evaluation that isolates the marginal lift from flight metadata, weather, and network pressure.
- Dashboard/reporting work that explains both headline results and methodological limitations.

## Architecture

The pipeline reads BTS flight records, applies strict column filtering, builds chronological train/validation/test splits, generates progressively richer feature sets, trains LightGBM models, evaluates multiple task heads, and publishes metrics to reports and a dashboard.

```mermaid
flowchart LR
  A[BTS 2024 Flights] --> B[Column Whitelist + Leakage Guards]
  B --> C[Chronological Split]
  C --> D[Stage A: Flight Metadata]
  D --> E[Stage B: Backward Weather Join]
  E --> F[Stage C: Network Pressure]
  F --> G[LightGBM Models]
  G --> H[Metrics JSON + Reports]
  H --> I[Streamlit Dashboard]
```

## Installation

```bash
pip install -r requirements.txt
```

The raw BTS dataset is not bundled. Place the 2024 BTS flight CSV at:

```text
data/raw/flight_data_2024.csv
```

Weather data is fetched programmatically:

```bash
python -m src.v2.weather_download
```

## Usage

Run the staged pipeline:

```bash
python -m src.v2.run_stage_a
python -m src.v2.run_stage_b
python -m src.v2.run_stage_c
python -m src.v2.run_final_test_evaluation
```

Run the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

## Results

Held-out evaluation uses a chronological Jan-Aug / Sep-Oct / Nov-Dec 2024 split. Stage C, which adds network pressure features, reached:

- ROC-AUC: 0.713
- PR-AUC at validation prevalence: 0.349
- Regression MAE: 19.23 minutes
- Held-out test window: November-December 2024, approximately 1.16M flights

The strongest defensible claim is that network arrival-pressure features provide the largest incremental lift over flight metadata and weather features in this leakage-aware setup.

## Known Limitations

- This is a course/applied ML project, not a deployed airline operations system.
- PR-AUC is prevalence-sensitive, so results are reported at both natural test prevalence and validation prevalence.
- The network pressure features are engineered rolling-window features, not a full temporal graph neural network.
- Raw BTS data is not bundled, so full reproduction requires downloading the source dataset.
- Some dashboard and reporting components assume the pipeline has already generated local JSON/parquet artifacts.

## Acknowledgements
The development of this project was supplemented by fellow students Yash Sanghvin and Xiyi Wang, during our time in Santa Clara University's CSEN 140 - Machine Learning course and was submitted as a final project. 

## Status

Public portfolio candidate. Applied ML / data systems artifact.

