# InfoOpt Reproducibility Stage 2B — Repository Manifest

**Date:** 2026-11-22  
**Public release staging directory:** `D:\InfoOpt Latex\Code\InfoOpt_Public_Release`  
**Total size:** 47 MB  
**Total files:** 1,305

---

## 1. Repository structure

```
InfoOpt_Public_Release/
├── README.md                       # User-facing quick-start guide
├── requirements.txt                # Python dependencies
├── RELEASE_REPORT.txt              # GitHub-release candidate summary
├── RELEASE_WARNINGS.txt            # Files flagged for path/secret review
├── SOURCE_MANIFEST_SHA256.csv      # SHA-256 manifest of GitHub-release candidate files
├── SHA256SUMS_Public_Release.txt   # SHA-256 manifest of the full Stage 2 public release
├── INFOOPT_REPRO_STAGE2_*.md       # Stage 2 deliverables
├── configs/                        # Frozen effective YAML configurations
│   ├── four_arm/
│   ├── stage1/
│   ├── stage2_ra_matched/
│   ├── stage2b_corrected/
│   ├── stage3/
│   └── stage4/
├── source_snapshots/               # Canonical executable training code per study
│   ├── ITSL_four_arm_prep_20261009_v8/
│   ├── ITSL_RA_matched_prep_20261010/
│   ├── ITSL_stage1_prep_20261011/
│   ├── ITSL_stage3_prep_20261012_v2/
│   ├── ITSL_stage4_prep_20261014_v2/
│   └── ITSL_Stage2B_Corrected_RA_20261022/
├── experiments/                    # GitHub-release candidate experiment helpers
│   ├── four_arm/
│   ├── ra_matched_preparation/
│   ├── stage1/
│   ├── stage2_ra_matched/
│   ├── stage2b_corrected/
│   ├── stage3/
│   └── stage4/
├── analysis/                       # Results, figures, and analysis scripts
│   ├── data/                       # Study 3 primary summary and contrasts
│   ├── data_fourarm/               # Study 1 four-arm results
│   ├── data_ra/                    # Study 2 RA-InfoOpt results
│   ├── data_stepsize/              # Step-size pilot results (Table A.6)
│   ├── assets/                     # Frozen evidence packages
│   ├── scripts/                    # make_tables.py, check_numbers.py, etc.
│   ├── Figures/                    # Rendered figure PDFs/PNGs
│   └── docs/                       # Claim-evidence map and check reports
└── feddisco_extension/             # FedDisco Stage 4 extension
    ├── snapshot/                   # Training code
    ├── configs/                    # Four Stage 4 YAML configs
    ├── tests/                      # CPU validation tests
    ├── runs/                       # 20 verified run directories
    ├── verify_stage4.py            # Independent verifier
    ├── analyze_stage4.py           # Statistical analysis script
    ├── STAGE4_VERIFICATION.txt     # Server verification report
    └── STAGE4_ANALYSIS.txt         # Server statistical analysis report
```

---

## 2. Included items

### 2.1 Executable training code

- `source_snapshots/*/snapshot/` — canonical Python modules for `train.py`, `methods/`, `models/`, `utils/`, `data/`.
- Covers Studies 1–5, corrected Study 2B, and the FedDisco extension.

### 2.2 Frozen configurations

- `configs/*/configs_effective/*.yaml` — effective configs with relative dataset roots (`./datasets`).
- `configs/*/configs_effective/RESOLUTION.json` — resolution records (sanitized to relative paths).
- `feddisco_extension/configs/*.yaml` — Stage 4 FedDisco configs.

### 2.3 RA-InfoOpt implementations

- `source_snapshots/ITSL_RA_matched_prep_20261010/snapshot/methods/ra_infoopt.py`
- `source_snapshots/ITSL_Stage2B_Corrected_RA_20261022/snapshot/methods/ra_infoopt.py`

### 2.4 FedDisco implementation and tests

- `feddisco_extension/snapshot/methods/feddisco.py`
- `feddisco_extension/tests/test_feddisco.py`
- `feddisco_extension/tests/test_configs.py`
- `feddisco_extension/tests/test_stage4_seed_override.py`
- `feddisco_extension/tests/test_stage4_verifier_synthetic.py`
- `feddisco_extension/tests/test_stage4_verification_markers.py`

### 2.5 Numerical results

- `analysis/data/primary_summary.csv` — Table 3 source.
- `analysis/data/paired_contrasts.json` — Table 4 / Fig. 4 source.
- `analysis/data/sensitivity_latest_repeat.json` — Table A.2 source.
- `analysis/data/ra_strata_summaries.json` — Table A.3 source.
- `analysis/data_fourarm/*` — Table 5 / Fig. 5 / Table A.4 source.
- `analysis/data_ra/*` — Table 6 / Fig. 6 / Table A.5 source.
- `analysis/data_stepsize/*` — Table A.6 source.

### 2.6 Analysis and figure scripts

- `analysis/scripts/make_tables.py` — generates all LaTeX tables.
- `analysis/scripts/test_make_tables.py` — 39 regression tests.
- `analysis/scripts/check_numbers.py` — numerical/log cross-checks.
- `analysis/scripts/plot_fourarm.py` — renders Fig. 5.
- `analysis/scripts/span_scan.py` — position-aware token scanner.

### 2.7 Verification scripts and reports

- `feddisco_extension/verify_stage4.py` — independent FedDisco verifier.
- `feddisco_extension/STAGE4_VERIFICATION.txt` — zero-failure verification report.
- `feddisco_extension/STAGE4_ANALYSIS.txt` — statistical analysis report.
- Per-study `verify_run.py` / `verify_frozen_integrity.py` under `source_snapshots/`.

### 2.8 Environment specification

- `requirements.txt` — pinned minimum versions for NumPy, SciPy, PyTorch, torchvision, tqdm, PyYAML, Matplotlib, scikit-learn, Pandas.

### 2.9 Documentation

- `README.md` — quick-start, dataset setup, reproducibility notes.
- `INFOOPT_REPRO_STAGE2_EVIDENCE_RECONCILIATION.md`
- `INFOOPT_REPRO_STAGE2_REPOSITORY_MANIFEST.md`
- `INFOOPT_REPRO_STAGE2_TEST_REPORT.md`
- `INFOOPT_REPRO_STAGE2_DECISION.md`

---

## 3. Excluded items

- Raw CIFAR-100, EuroSAT, and PathMNIST datasets.
- Large model checkpoints (`final_model.pt`).
- Medical imaging arrays under `server_evidence/ITSL_unified/data/`.
- Python virtual environments, `__pycache__`, `.git` metadata.
- Original LaTeX manuscript files (`main_revised.tex`, `supplementary.tex`) — the manuscript is frozen and not part of the code package.

---

## 4. Sanitization applied

### 4.1 Paths replaced with relative/configurable forms

- `dataset.root` in effective YAML configs: `/home/whu/disk1/...` → `./datasets`.
- `resolve_dataset_roots.py`: removed hardcoded server fallback; defaults to script directory or `DATASETS_ROOT`/`EXP_ROOT` env vars.
- Launch scripts (`launch_stage*.sh`, `launch_four_arm_gpu1.sh`, `aunch_ra_gpu1.sh`, `SERVER_STAGE4_RUN.sh`): default `PKG`, `EXP_ROOT`, `PY`, `STAGE3_DIR`, `STAGE4_DIR` changed to relative paths or `python`.
- `build_run_manifest.py`, `analyze_stage*.py`, `verify_frozen_integrity.py`, `smoke_resnet_runtime.py`: default `EXP_ROOT`/`RUNS_DIR` changed to relative paths.
- `RESOLUTION.json` files: absolute dataset roots replaced with `./datasets`.

### 4.2 Historical audit records retained unchanged

The following files still contain original server paths because they are frozen provenance/audit records:

- `analysis/assets/ITSL_FedDyn_SCAFFOLD_audit_final_20261010/*`
- `analysis/assets/ITSL_RA_matched_20261010/PROTOCOL.md`
- `analysis/assets/ITSL_RA_matched_20261010/source_manifest.json`
- `analysis/assets/ITSL_four_arm_20261010/PROTOCOL_BEFORE_LAUNCH.md`
- `analysis/assets/ITSL_reanalysis_sensitivity_v2/verify_frozen_manifest.py`
- `analysis/assets/ITSL_stepsize_pilot_20261011/pilot/config.json`
- `analysis/assets/ITSL_stepsize_pilot_20261011/pilot/provenance.json`
- `analysis/assets/ITSL_stepsize_pilot_20261011/pilot/results/verification.json`
- `analysis/assets/ITSL_stepsize_pilot_20261011/pilot_documents/*.json` and `*.md`

These are documented as historical audit-only evidence.

---

## 5. Known issues

| Issue | Location | Impact | Mitigation |
|-------|----------|--------|------------|
| Missing `pilot_validation_gamma_fix.patch` | `analysis/assets/ITSL_stepsize_pilot_20261011/per_example_correction/` | `make_tables.py` would crash on checksum verification. | Public-release `make_tables.py` warns and skips the missing file; table content is unaffected. |
| `check_numbers.py` requires LaTeX manuscript | `analysis/scripts/check_numbers.py` | Cannot run full numerical/log cross-checks without `main_revised.tex` and `supplementary.tex`. | Manuscript files are frozen and not distributed with code; table/figure generation does not depend on them. |
| CIFAR-100 not included | `./datasets` | FedDisco verifier and retraining require local CIFAR-100 download. | Documented in README; torchvision can download CIFAR-100 automatically. |
| `experiments/ra_matched_preparation/` has truncated filenames | `experiments/ra_matched_preparation/` | Some helper scripts have names like `aunch_ra_gpu1.sh` (missing initial `l`). | Canonical code is in `source_snapshots/ITSL_RA_matched_prep_20261010/`; `experiments/` directory is retained as historical candidate. |

---

## 6. Reproducibility classification

| Component | Status |
|-----------|--------|
| Tables 3–6, A.1–A.6 | Reproducible from released results (CSV/JSON + `make_tables.py`). |
| Figures | Reproducible from released results and frozen asset packages. |
| Four-arm and RA-InfoOpt verification reports | Reproducible from released results. |
| FedDisco Stage 4 statistics | Reproducible from included 20 run directories + `analyze_stage4.py`. |
| FedDisco weight reconstruction | Reproducible from included 20 run directories + CIFAR-100 labels + `verify_stage4.py`. |
| Original 480 + 15 corrected InfoOpt runs | Reproducible by retraining with source snapshots + datasets + GPU environment. |
| FedDyn/SCAFFOLD historical audit | Audit-only evidence; no retraining possible. |
| Step-size pilot patch file | Partially reproducible; one listed patch file is missing. |
