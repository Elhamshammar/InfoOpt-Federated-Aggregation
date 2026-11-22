# Sensitivity / RA reanalysis package v2 — 2026-10-08

**Label: RETROSPECTIVE REANALYSIS.** No training, no tuning, no manuscript
edits. The primary reanalysis package
(`ITSL_retrospective_reanalysis_20261008.zip`, sha256 `85e0cc8a…`) is
preserved unchanged.

## v2 changes (reviewer feedback)

1. **Missing latest-repeat entries now abort.** A repeat-selection cell
   without a latest-repeat entry in the frozen sensitivity inventory is a
   fatal error with a clear reason. Genuine single-source cells are
   explicitly labelled "same run (single source)". Regression-tested:
   removing one inventory entry aborts with
   `FATAL: repeat-selection cell … has NO latest-repeat entry …`.
2. **Primary values come from the frozen primary per-seed output**
   (`reanalysis_20261008/outputs/per_seed_round100.csv`), integrity-checked
   against the primary package's SHA256SUMS, with per-cell run-identity
   cross-checks. Run directories are no longer reread by name without
   verification.
3. **Correction of a previous claim.** An earlier message stated that the
   verified recomputation "corrected earlier pooled means". That was wrong:
   the per-seed (one-run-per-seed) selection had already been applied
   before the primary package was built, and the verified recomputation's
   RA strata means are IDENTICAL to the primary package's, value for value
   (all nine strata compared). What v2 changes is provenance discipline —
   RA summaries are now computed from the frozen, hash-verified inventory
   with run identities and per-seed values exported — not the numbers.

## Contents

- `frozen_inputs/`: `sensitivity_source_inventory_FROZEN_20261008.csv`
  (54 records: 30 latest-repeat, 5 FedOpt sensitivity, 19 RA members) +
  `FREEZE_RECORD.json` (dated, hashed, retrospective — not
  preregistration).
- `outputs/`: `verification_log.json` (54/54 sources hash- and
  endpoint-verified before calculation), `sensitivity_latest_repeat.json`,
  `sensitivity_cifar_fedopt.json`, `ra_strata_summaries.json`.
- `verify_frozen_manifest.py` + `frozen_manifest_verification.json`: the
  standalone frozen-manifest verifier and its report (160/160 verified).
- `build_sensitivity_manifest.py`, `run_sensitivity_reanalysis.py`.
- `SHA256SUMS.txt` (all of the above; no self-entry).
