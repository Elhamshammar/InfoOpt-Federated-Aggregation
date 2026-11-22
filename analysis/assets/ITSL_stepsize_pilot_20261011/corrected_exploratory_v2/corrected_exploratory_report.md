# Corrected exploratory analysis (recomputed from saved artifacts)

- run: `stepsize_diag_pilot_20261004_175620`
- corrected output dir: `results\stepsize_diag_pilot_20261004_175620_audit_corrected_v2` (originals preserved)
- Interpretation caveat: the 3 repetitions within a (cohort, seed, round) cell are NOT independent measurements; the per-seed and per-round breakdowns below are cluster-level summaries, not independent samples.
- No claim about the variability hypothesis is made here; the corrected comparison must be inspected first.

## Held-out MAE / MSE (overall)

| Predictor | fit_status | dev cells | held cells | MAE held | MSE held |
|-----------|-----------|-----------|------------|----------|----------|
| phase_only | fit | 27 | 54 | 0.009036155845108362 | 0.00031357235306608236 |
| phase_plus_variability | fit | 27 | 54 | 0.020016339820771462 | 0.0006961795674210221 |

## Held-out MAE / MSE (by training seed)

| Predictor | seed | MAE | MSE |
|-----------|-------|-----|-----|
| phase_only | 202 | 0.005331391008793088 | 0.00011710656393943341 |
| phase_only | 203 | 0.012740920681423637 | 0.0005100381421927313 |
| phase_plus_variability | 202 | 0.023525321281601255 | 0.0008248750426670116 |
| phase_plus_variability | 203 | 0.016507358359941673 | 0.0005674840921750323 |

## Held-out MAE / MSE (by checkpoint round)

| Predictor | round | MAE | MSE |
|-----------|-------|-----|-----|
| phase_only | 5 | 0.026909703836323344 | 0.0009400059330535117 |
| phase_only | 50 | 0.00019876369900174225 | 7.111261447353465e-07 |
| phase_only | 100 | 3.469446951953614e-18 | 1.2037062152420224e-35 |
| phase_plus_variability | 5 | 0.03911276273364392 | 0.0017480439287388208 |
| phase_plus_variability | 50 | 0.009576346751754687 | 0.00015742997935497816 |
| phase_plus_variability | 100 | 0.011359909976915787 | 0.00018306479416926736 |
