# InfoOpt Public Release — Stage 2C Final Validation Report

## Date

2026-11-22

## Package

`D:\InfoOpt Latex\Code\InfoOpt_Public_Release`

## Changes made in Stage 2C

1. **`analysis/scripts/check_numbers.py`** — made the manuscript directory
   configurable via the second positional argument or the
   `INFOOPT_MANUSCRIPT_DIR` environment variable. When the LaTeX source
   files are absent, the script now skips prose-quote,
   superseded-value, cross-reference, and LaTeX-log checks instead of
   crashing.
2. **`manuscript/README.md`** — added a placeholder directory documenting
   the required LaTeX files and how to run the full
   `check_numbers.py` verification.
3. **`LICENSE`** — added a placeholder stating that the license is to be
   determined before public release.
4. **`feddisco_extension/tests/test_configs.py`** — added the missing
   `import numpy as np` to fix two test collection errors.
5. **`feddisco_extension/tests/test_reconstruction.py`** — corrected the
   import from `verify_stage3b` to `verify_stage4.recompute_weights` so
   the released package can run its reconstruction tests.
6. **`experiments/ra_matched_preparation/`** — removed truncated,
   unreferenced files; the canonical RA-matched code remains in
   `source_snapshots/ITSL_RA_matched_prep_20261010/`.
7. **`SHA256SUMS_Public_Release.txt`** — regenerated and self-verified.

## Tests executed

| Test | Command | Result |
|---|---|---|
| Table generation | `python analysis/scripts/make_tables.py` | PASS |
| Table regression tests (39 tests) | `python analysis/scripts/test_make_tables.py` | PASS (39/39) |
| Numerical cross-checks | `python analysis/scripts/check_numbers.py` | PASS with manuscript checks SKIPPED |
| Four-arm figure generation | `python analysis/scripts/plot_fourarm.py` | PASS |
| FedDisco extension tests | `python -m pytest feddisco_extension/tests/ -v` | PASS (28/28) |
| SHA-256 manifest self-check | `sha256sum -c SHA256SUMS_Public_Release.txt` | PASS |
| Python syntax check | `python -m py_compile` on every `.py` file | PASS |

### `check_numbers.py` output summary

- All checksum, independent-recomputation, and generated-table checks
  passed.
- Prose-quote, superseded-value, cross-reference, and LaTeX-log checks
  were skipped because the manuscript source files are not present.
- The script exits with status 0 and reports `ALL CHECKS PASSED`.

### FedDisco tests

All 28 tests passed, including:

- numerical agreement with the official FedDisco weighting
- participating-client size normalization
- KL normalization
- ReLU and fallback behavior
- deterministic permutation and discrepancy-multiset preservation
- unnormalized-weight-multiset counterexample
- client-ID/weight/update alignment
- matching partitions, schedules, and initial hashes across arms
- RNG isolation from local training
- finite weights and aggregation outputs
- reconstruction of weights from saved diagnostics
- Stage-4 seed override propagation
- verification-marker regression tests

## Checks blocked or not run

| Check | Status | Reason |
|---|---|---|
| Manuscript prose-quote scan | BLOCKED | `main_revised.tex` and `supplementary.tex` not found in workspace |
| Superseded-value scan | BLOCKED | requires original manuscript source (`original/`) |
| Cross-document reference check | BLOCKED | requires compiled `.aux` files and manuscript source |
| LaTeX log error check | BLOCKED | requires compiled `.log` files and manuscript source |
| 39 table-generation regression tests | PASS | |
| Full GPU retraining | NOT RUN | restriction: do not run GPU experiments |

## Privacy and security scan

- No passwords, API tokens, private keys, or `.env` files were found in
  the release package.
- Historical audit files under `analysis/assets/` and `experiments/`
  contain machine-specific paths from the original development
  environment. These files are not referenced by the README quick-start
  commands and are retained as audit-only evidence.
- One smoke-test YAML
  (`experiments/stage2b_corrected/smoke/smoke_ra_probeoff.yaml`) contains
  a hardcoded Windows data path. It is a historical, unreferenced test
  artifact and should not be executed without editing the path.

## Package integrity

- **Manifest:** `SHA256SUMS_Public_Release.txt`
- **Manifest hash file:** `SHA256SUMS_Public_Release.txt.sha256`
- **Size:** ~47 MB
- **Self-verification:** `sha256sum -c SHA256SUMS_Public_Release.txt` reports
  all files OK; the manifest's own SHA-256 is recorded in
  `SHA256SUMS_Public_Release.txt.sha256`.

## README quick-start verification

The README commands use relative paths and were verified:

```bash
pip install -r requirements.txt
python analysis/scripts/make_tables.py
python analysis/scripts/test_make_tables.py
python analysis/scripts/plot_fourarm.py
python analysis/scripts/check_numbers.py .
```

The last command runs successfully in manuscript-absent mode.

## Remaining issues before public release

1. Replace the `LICENSE` placeholder with the final approved license text.
2. Obtain and copy the frozen LaTeX manuscript source
   (`main_revised.tex`, `supplementary.tex`, `references.bib`,
   `elsarticle-num.bst`, and the `original/` subfiles) into
   `manuscript/` and rerun `check_numbers.py` to enable full prose and
   reference verification.
3. Decide whether to remove, anonymize, or document the historical
   hardcoded-path files before a fully public release.
