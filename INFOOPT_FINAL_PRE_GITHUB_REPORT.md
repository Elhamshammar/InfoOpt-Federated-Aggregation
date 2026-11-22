# InfoOpt Public Release — Final Pre-GitHub Report

**Date:** 2026-11-22  
**Repository:** `D:\InfoOpt Latex\Code\InfoOpt_Public_Release`

## Final preparation tasks completed

1. **License review inspected.** No open-source license has been applied to the
   authors' original code. No explicit copyright or license headers were found
   in `source_snapshots/` or `feddisco_extension/snapshot/`. No third-party
   code blocks requiring separate notices were identified; dependencies are
   listed only in `requirements.txt`.

2. **Final Astra-corrected LaTeX source searched.** A workspace-wide search for
   `*.tex`, `*.bib`, and `*.bst` outside the release directory found no
   manuscript source files. The `manuscript/` placeholder directory is retained
   with its README documenting the required files and how to run the full
   `check_numbers.py` verification when the source becomes available.

3. **README note added.** `README.md` now includes an "Important notes" section
   explaining that historical audit-only records may contain original
   machine-specific paths, that the LaTeX source is absent, and that the missing
   historical pilot patch is documented.

4. **Missing pilot patch confirmed.**
   `analysis/assets/ITSL_stepsize_pilot_20261011/per_example_correction/pilot_validation_gamma_fix.patch`
   is listed in the asset package's `SHA256SUMS.txt` but is absent. The omission
   is documented in `README.md`, `INFOOPT_REPRO_STAGE2C_FINAL_VALIDATION.md`,
   and the Stage-2B evidence-reconciliation report. It does not affect the
   retained numerical results of the principal controlled studies (Studies
   1–5, four-arm, RA-matched, FedDisco extension), which are verified by their
   own independent `SHA256SUMS.txt` files.

5. **README commands verified.** All quick-start commands point to existing
   scripts:
   - `analysis/scripts/make_tables.py`
   - `analysis/scripts/test_make_tables.py`
   - `analysis/scripts/plot_fourarm.py`
   - `analysis/scripts/check_numbers.py`
   The repository has no existing `.git` directory and a `.gitignore` is in
   place.

6. **Final SHA-256 manifest recalculated.**
   - `SHA256SUMS_Public_Release.txt`: 1,320 entries, self-verifies.
   - `SHA256SUMS_Public_Release.txt.sha256`: records the manifest's own hash.
   - Package size: ~47 MB; total files: 1,322.

## Security/privacy scan

- No passwords, API tokens, private keys, or `.env` files were found.
- `RELEASE_WARNINGS.txt` contains 159 files flagged for potential
  machine-specific paths; these are historical audit-only records and are not
  referenced by the README quick-start commands.

## Exact remaining author decisions

Before any GitHub upload, the authors must decide:

1. **License.** Replace the placeholder `LICENSE` file with the final approved
   open-source license text. Recommended choices: Apache-2.0 or MIT. Do not
   upload publicly without a license.

2. **LaTeX manuscript source.** Either:
   - locate and copy the final Astra-corrected `main_revised.tex`,
     `supplementary.tex`, `references.bib`, and `elsarticle-num.bst` (plus
     `original/` subfiles) into `manuscript/` and rerun
     `analysis/scripts/check_numbers.py`, or
   - explicitly accept the documented limitation that prose and cross-reference
     checks are skipped.

3. **Historical path/secret flags.** Review the 159 files listed in
   `RELEASE_WARNINGS.txt` and decide whether any should be removed, anonymized,
   or kept as audit-only evidence before a fully public release.

4. **Missing pilot patch.** Decide whether to search archives for
   `pilot_validation_gamma_fix.patch`, permanently remove its entry from
   `analysis/assets/ITSL_stepsize_pilot_20261011/SHA256SUMS.txt`, or leave the
   documented warning in place.

## Recommendation

The repository is ready for **private GitHub upload** once decision 1 (license)
is resolved. Decisions 2–4 can be addressed before any public release or DOI
assignment.
