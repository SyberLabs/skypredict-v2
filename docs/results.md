# SkyPredict V2 Results

This public summary records the headline evaluation from the audited SkyPredict V2 pipeline.

## Evaluation Setup

- Dataset: BTS 2024 U.S. domestic flights
- Total rows: approximately 6.96M
- Train window: January-August 2024
- Validation window: September-October 2024
- Held-out test window: November-December 2024, approximately 1.16M flights
- Constraint: features must be observable strictly before scheduled departure / gate-close

## Staged Model Progression

| Stage | Feature family | ROC-AUC | PR-AUC at validation prevalence |
|---|---|---:|---:|
| A | Flight metadata | 0.601 | 0.182 |
| B | + backward-joined hourly weather | 0.653 | 0.244 |
| C | + 3-hour rolling network pressure | 0.713 | 0.349 |

## Main Finding

Network arrival-pressure features supplied the largest incremental lift. In the audited held-out evaluation, Stage C improved ROC-AUC by approximately +0.060 over Stage B and +0.112 over Stage A.

## Audit Notes

The final metrics reflect post-audit corrections:

- Weather joins use backward-only temporal matching.
- PR-AUC is reported at both natural and validation prevalence to avoid base-rate confusion.
- Forbidden post-hoc flight columns are blocked by feature whitelists, blacklists, and runtime assertions.
- Historical stats use backward-only expanding windows.
- Arrival delay is dropped from train/validation/test frames after being consumed for target and pressure construction.

## Limitations

- This is an applied ML project and public portfolio artifact, not a deployed airline operations system.
- PR-AUC is prevalence-sensitive; the validation-prevalence figure is the cleaner comparison across splits.
- The network-pressure feature is an engineered rolling-window signal, not a full temporal graph neural network.

