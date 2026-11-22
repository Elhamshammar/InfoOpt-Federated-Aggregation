# Correction Report — validation_selected_scaling per-example export defect

**Date:** 2026-10-08
**Affected artifact:** `results/stepsize_diag_pilot_20261004_175620/results/per_example_losses.csv` (rows with `policy=validation_selected_scaling` only)
**Status of originals:** untouched. All original results, frozen protocols, and verdicts are preserved; this directory contains only new, corrective material.

## 1. Defect and root cause

`stepsize_diag/pilot.py`: `run_pilot` copied the module side cache
`_VALIDATION_GAMMA_BY_SOURCE` into the retained mapping
`per_source_validation_gamma` and then called `.clear()` on the cache
(old line 1512). The per-example exporter `_per_example_one_source` later
read the cleared cache via

```python
val_g = _VALIDATION_GAMMA_BY_SOURCE.get(_validation_gamma, GAMMA_ONE)
```

so every exported `validation_selected_scaling` row silently used γ=1.

**Measured impact on the defective export:** all 450,000
`validation_selected_scaling` rows carry γ=1.000000 and are bit-identical
to the `fedavg` rows (verified directly, not inferred). The retained
mapping in `results/pilot_decision.json` shows the intended γ differs
from 1 for **77 of 90** sources.

**Scope of impact:** only the per-example CSV's
`validation_selected_scaling` rows. The grid/rule summaries,
`split_metric_keys`, decision inputs, and all verdicts were computed in
stage 5–7 with the correct per-source γ and are NOT affected. Other
per-example policies (fedavg, fixed global/phase, claude) did not use the
cache and are unaffected (verified by the consistency checker: 0
mismatches outside `validation_selected_scaling`).

## 2. Fix

`pilot_validation_gamma_fix.patch` (against the pre-patch file, which is
preserved at `prepatch/pilot_orig.py`):

- `_per_example_for_decisive_policies` and `_per_example_one_source` now
  take the retained per-source selected-γ mapping **explicitly**
  (`validation_gamma_by_source`), wired from the retained
  `per_source_validation_gamma` at the call site.
- The lookup is strict: a missing key raises `KeyError`. The silent
  γ=1 fallback is removed.
- The exporter no longer reads `_VALIDATION_GAMMA_BY_SOURCE` at all, so
  clearing that cache cannot affect exports.

## 3. Verification

The consistency checker was subsequently HARDENED (2026-10-08, second
revision) and both verification reports in this directory are from the
hardened version:

| Check | Result |
|---|---|
| `selftest_validation_gamma_export.py` (exporter regression): cache populated then cleared; exporter called with explicit mapping γ=0.5 | **PASS** — exported rows carry γ=0.500000, differ from FedAvg rows; missing key raises KeyError |
| `selftest_consistency_hardening.py` (checker regression suite) | **PASS** — 14 cases: clean export passes (with and without inventory); non-finite loss, unknown policy, undecidable cell, duplicate authoritative key, out-of-holdout example ID, duplicate example IDs, the 2026-10-08 defect pattern, empty file, missing/extra cell vs inventory, non-finite reference value, gamma_origin mismatch all FAIL |
| `scripts/verify_per_example_consistency.py` (hardened) on the ORIGINAL defective export | **FAIL (expected)** — exactly 154 cells mismatched (77 sources × 2 holdouts), all `validation_selected_scaling`; 900/900 cells checked including per-cell example-set verification against `partition.json` (2,250,000 rows) |
| Same checker on the regenerated trajectory rows, with `--expect-inventory` (18 cells / 45,000 rows required exactly) | **PASS** — 18/18 cells: policy identity, γ, sample counts, example sets, and mean losses all match the retained grid/rule summaries (loss atol 3e-4) |

Hardening details:
1. Non-finite values: any non-finite per-example loss/γ is fatal; any value
   used as an authoritative reference must be finite. Summary rows that
   deliberately record NaN for INVALID candidates (e.g. γ=2.0
   extrapolations with `candidate_valid=False`) are catalogued, never
   consumed; if one is ever selected as a reference, the cell fails.
2. Unknown policies and undecidable cells are failures.
3. Summaries are filtered explicitly to `buffer_policy=all_floating_state`;
   duplicate authoritative keys abort the run.
4. Example IDs must be unique per cell and the per-cell example set must
   equal the expected `partition.json` holdout index set.
5. Partial verification requires an explicit expected inventory
   (`expected_inventory_trajectory_only.json`: 18 cells, 45,000 rows);
   present cells must equal it exactly; empty files always fail; missing
   cells are always reported.
6. `selftest_consistency_hardening.py` regression-tests every false-PASS
   case above.

The regeneration script now also fails immediately on an invalid candidate
instead of warning and exporting it.

## 4. Regeneration

**Checkpoints are sufficient for only 9 of 90 sources.** Each saved
checkpoint contains the trajectory cohort's post-local client states,
global before/after states, client IDs, normalized weights, and sample
counts. With the retained γ mapping, the 9 trajectory-cohort sources
(3 seeds × 3 rounds; 7 of them with intended γ≠1) were regenerated by
evaluation only — no training — into this directory:

`per_example_losses_validation_selected_trajectory_only.csv`
(9 sources × 2 holdouts × 2,500 examples = 45,000 rows)

All 18 regenerated cells pass consistency verification against the
retained rule summaries.

**The remaining 81 sources cannot be regenerated without training.** Missing
states (never persisted to disk; computed in-memory only):

- `original_cohort_rep`: per (seed, round), post-local client state dicts
  for the 5 trajectory-cohort clients × 3 repetition seeds;
- `alt_cohort`: per (seed, round), post-local client state dicts for the
  2 alternative cohorts × 5 clients × 3 repetition seeds.

Recovering those rows requires deterministic retraining of the affected
clients, which was explicitly out of scope for this correction.

## 5. What was NOT done

- No retraining, no tuning, no changes to any frozen protocol or verdict.
- No reconstruction of sample-level losses from aggregate means (all
  regenerated values are real per-example evaluations).
- The original defective CSV was not modified, renamed, or deleted.
- The pilot was not re-run.

## 6. Residual action for a future run

Any future pilot run using the patched `pilot.py` will export correct
`validation_selected_scaling` rows for all sources. To fully supersede the
defective export for the 81 non-trajectory sources, re-run the pilot under
the frozen protocol with the patched code, or extend checkpointing to
persist per-(client, rep_seed) states.

## Contents of this directory

| File | Content |
|---|---|
| `pilot_validation_gamma_fix.patch` | unified diff of the exporter fix |
| `prepatch/pilot_orig.py` | pre-patch pilot.py (from the 2026-10-04 zip; defective line verified present) |
| `per_example_losses_validation_selected_trajectory_only.csv` | corrected rows, 9 trajectory sources (45,000 rows) |
| `expected_inventory_trajectory_only.json` | explicit expected inventory for partial verification (18 cells / 45,000 rows) |
| `regeneration_metadata.json` | checkpoint hashes, γ source, model provenance, missing-states list, runtime |
| `consistency_report_DEFECTIVE_EXPORT.json` | hardened checker output on the original export (154 mismatches) |
| `consistency_report_regenerated_trajectory.json` | hardened checker output on the regenerated rows (PASS) |
| `selftest_validation_gamma_export.py` | exporter regression check (PASS) |
| `selftest_consistency_hardening.py` | checker hardening regression suite (14 cases, PASS) |
| `verify_per_example_consistency.py` | the hardened checker (as run) |
| `regenerate_validation_selected_trajectory.py` | the regeneration script (as run; fails fast on invalid candidates) |
