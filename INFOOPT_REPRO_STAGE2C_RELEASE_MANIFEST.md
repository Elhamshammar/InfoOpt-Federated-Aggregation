# InfoOpt Public Release — Stage 2C Release Manifest

## Package location

`D:\InfoOpt Latex\Code\InfoOpt_Public_Release`

## Package statistics

- **SHA-256 manifest:** `SHA256SUMS_Public_Release.txt`
- **Manifest hash file:** `SHA256SUMS_Public_Release.txt.sha256`
- **Total size:** ~47 MB

## Top-level layout

| Path | Contents | Reproducibility role |
|---|---|---|
| `README.md` | Quick-start and dataset-setup instructions | Primary entry point |
| `requirements.txt` | Python dependencies | Environment setup |
| `LICENSE` | License placeholder (TBD before public release) | Legal |
| `source_snapshots/` | Canonical training code for Studies 1–5, corrected Study 2B | Retraining |
| `configs/` | Frozen effective YAML configurations | Retraining |
| `experiments/` | Stage launch scripts and historical experiment directories | Retraining / audit |
| `analysis/` | Table/figure/analysis scripts and verified numerical results | Result regeneration |
| `feddisco_extension/` | FedDisco implementation, configs, 20 run directories, tests | Extension verification |
| `manuscript/` | Placeholder for LaTeX source (see `README.md`) | Prose verification |

## Key files added or changed in Stage 2C

| File | Change | Reason |
|---|---|---|
| `analysis/scripts/check_numbers.py` | Added configurable manuscript directory (`INFOOPT_MANUSCRIPT_DIR` env var or second CLI arg) and missing-manuscript safety | Allows CPU-only validation when LaTeX source is absent |
| `manuscript/README.md` | Added | Documents required LaTeX files and how to run `check_numbers.py` |
| `LICENSE` | Added | Placeholder pending final author/institutional approval |
| `feddisco_extension/tests/test_configs.py` | Added missing `import numpy as np` | Fixes two test collection failures |
| `feddisco_extension/tests/test_reconstruction.py` | Updated import from `verify_stage3b` to `verify_stage4.recompute_weights` | Fixes import error in released package |
| `experiments/ra_matched_preparation/` | Removed | Truncated, unreferenced files; canonical RA code is in `source_snapshots/ITSL_RA_matched_prep_20261010/` |
| `SHA256SUMS_Public_Release.txt` | Regenerated after changes | Self-verifies |

## What is included

- Executable training code and frozen configurations for Studies 1–5 and
  corrected Study 2B.
- FedDisco extension source, four pilot/final configs, 20 verified run
  directories, verification and analysis scripts.
- Per-seed round-100 accuracies and contrast summaries for all reported
  experiments.
- Table and figure generation scripts that regenerate all manuscript
  tables and main figures from retained results.
- CPU-only test suites (`test_make_tables.py`, FedDisco tests,
  `test_configs.py`, marker-regression tests).
- Verification reports and frozen evidence packages for the four-arm,
  RA-matched, and step-size-pilot experiments.

## What is excluded

- Raw CIFAR-100, EuroSAT, and PathMNIST datasets.
- Large model checkpoints (`.pth`, `.pt`, `.ckpt`, `.pkl`).
- Medical imaging arrays unrelated to the paper.
- Python virtual environments and build caches.
- Private credentials, tokens, and SSH keys.
- The LaTeX manuscript source files (not found in workspace; placeholder
  directory provided).

## Known remaining limitations

1. **Manuscript source absent.** `check_numbers.py` skips prose-quote,
   superseded-value, cross-reference, and LaTeX-log checks. Add the
   frozen LaTeX files to `manuscript/` to enable full verification.
2. **Missing patch file.** `pilot_validation_gamma_fix.patch` is listed
   in the step-size pilot `SHA256SUMS.txt` but is absent. The release
   logs this and continues; the missing patch does not affect the
   retained numerical results.
3. **Historical hardcoded paths.** Several files under `experiments/`
   and `analysis/assets/` contain machine-specific paths from the
   original development environment. These are historical audit-only
   files and are not referenced by the README quick-start commands.
4. **License placeholder.** `LICENSE` must be replaced before public
   release.
