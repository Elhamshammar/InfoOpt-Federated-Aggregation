# InfoOpt Reproducibility Stage 2B — Decision

**Date:** 2026-11-22  
**Public release staging directory:** `D:\InfoOpt Latex\Code\InfoOpt_Public_Release`  
**Total size:** 47 MB  
**Total files:** 1,310  
**SHA-256 manifest:** `SHA256SUMS_Public_Release.txt` (regenerate after any final edits)  
**Decision:** **GO with documented qualifications**

The public release staging directory `InfoOpt_Public_Release/` is scientifically reproducible for the manuscript's reported findings, subject to two small corrective actions before final publication.

---

## 1. What passed

- **Evidence reconciliation.** The manuscript's 515-run accounting (480 InfoOpt + 15 corrected Study 2B + 20 FedDisco) now maps cleanly to archived evidence directories on Windows.
- **FedDisco evidence transfer.** All 20 Stage 4 run directories transferred successfully. The server verification report shows zero failures: 100 rounds per run, matching partitions/init hashes/schedules, zero FedDisco weight-reconstruction mismatches, and expected shuffled-arm behavior.
- **Table generation.** `make_tables.py` regenerates all manuscript tables (3–6, A.1–A.6) from retained CSV/JSON inputs.
- **Regression tests.** `test_make_tables.py` passes 39/39 checks, including checksum/byte-identity verification of the four-arm, RA-matched, and step-size packages.
- **FedDisco CPU tests.** `test_feddisco.py` and `test_configs.py` pass.
- **Path sanitization.** Executable code, configs, launch scripts, and resolution records no longer contain hardcoded server paths. Remaining hardcoded paths are confined to frozen `analysis/assets/` audit/provenance records.
- **Repository packaging.** `InfoOpt_Public_Release/` contains canonical code, frozen configs, per-seed results, analysis scripts, verification reports, and the FedDisco extension evidence in 47 MB.

---

## 2. What remains

| # | Issue | Severity | Required action |
|---|-------|----------|-----------------|
| 1 | `check_numbers.py` cannot run without the frozen LaTeX manuscript (`main_revised.tex`, `supplementary.tex`). | Low-medium | Either include the LaTeX source in the package or document that `check_numbers.py` is a manuscript-build step, not part of the data-only reproducibility path. |
| 2 | `pilot_validation_gamma_fix.patch` is missing from the step-size pilot asset package (listed in SHA256SUMS.txt but absent). | Low | Public-release `make_tables.py` warns and skips it. Locate the patch file if possible, or permanently adjust the SHA256SUMS manifest / verification logic and document the omission. |
| 3 | CIFAR-100 dataset not included; Windows re-verification of FedDisco weights requires local download. | Low | Documented in README. Users can download via torchvision. |
| 4 | `experiments/ra_matched_preparation/` contains truncated filenames from the GitHub-release candidate. | Low | Canonical code is in `source_snapshots/ITSL_RA_matched_prep_20261010/`. The `experiments/` tree can be dropped or kept as a historical candidate. |
| 5 | License file is missing. | Medium | Add a `LICENSE` file after confirming ownership and third-party code permissions before any public release. |

---

## 3. Smallest corrective actions

1. **Add a `LICENSE` file** to `InfoOpt_Public_Release/`.
2. **Decide on the LaTeX manuscript files:** either add `main_revised.tex`, `supplementary.tex`, `references.bib`, and the required Elsevier class files to the package, or update `README.md` to state that `check_numbers.py` is a manuscript-build verification step requiring those files.
3. **Resolve the missing patch file:** search the original server or archives for `pilot_validation_gamma_fix.patch`; if it cannot be found, edit `analysis/assets/ITSL_stepsize_pilot_20261011/SHA256SUMS.txt` in the public release to remove the missing entry and document why.
4. **Drop or fix the truncated `experiments/ra_matched_preparation/` files** (optional but tidy).

---

## 4. Reproducibility classification

| Component | Classification |
|-----------|----------------|
| Manuscript tables and figures | **Reproducible from released results.** |
| FedDisco Stage 4 statistics and verification | **Reproducible from released results** (20 run directories included). |
| Original 480 + 15 corrected InfoOpt runs | **Reproducible by retraining** with source snapshots + datasets + GPU. |
| FedDyn/SCAFFOLD audit narrative | **Historical audit-only evidence.** |
| Step-size pilot patch file | **Partially reproducible** — one listed file is missing. |

---

## 5. Final decision

**GO** for Stage 2 packaging, with the four corrective actions above completed before any public release or GitHub upload. The core reproducibility claim is supported: the reported tables, figures, FedDisco extension, and training code are all present and verified on Windows.
