# InfoOpt Reproducibility Stage 2B — Evidence Reconciliation

**Date:** 2026-11-22  
**Scope:** Reconcile the manuscript's run accounting with the actual archived evidence after the FedDisco GPU results were transferred from Linux to Windows.

---

## 1. Manuscript run accounting

The manuscript reports the following run counts:

| Study | Runs | Description |
|-------|------|-------------|
| Study 1 | 60 | Four-arm AIB confound experiment (FedAvg/InfoOpt-Full × AIB off/on). |
| Study 2 | 60 | Matched RA-InfoOpt experiment (FedAvg, InfoOpt-Full, RA probe off, RA probe on). |
| Study 3 | 90 | Stage 1 signal/control experiment (six arms across three datasets). |
| Study 4 | 180 | Stage 3 heterogeneity experiment (six arms × two alphas × three datasets). |
| Study 5 | 90 | Stage 4 backbone experiment (ResNet-18, six arms, three datasets). |
| Study 2B | 15 | Corrected RA-InfoOpt probe-off runs. |
| FedDisco extension | 20 | Five-seed, four-arm CIFAR-100 FedDisco pilot. |
| **Total** | **515** | |

---

## 2. Evidence-to-study mapping

Each study maps to a specific archived run directory tree under `InfoOpt_Code_For_Journal/server_evidence/`.

| Study | Evidence directory | Run count | Verification |
|-------|-------------------|-----------|--------------|
| Study 1 | `server_evidence/ITSL_four_arm_20261009/runs/` | 60 | `analysis/data_fourarm/fourarm_verification_report.json` |
| Study 2 | `server_evidence/ITSL_RA_matched_20261010/runs/` | 60 | `analysis/data_ra/ra_verification_report.json` |
| Study 3 | `server_evidence/ITSL_stage1_20261011/runs/` | 90 | `server_evidence/ITSL_stage1_20261011/verification/` |
| Study 4 | `server_evidence/ITSL_stage3_20261012/runs/` | 180 | `server_evidence/ITSL_stage3_20261012/verification/` |
| Study 5 | `server_evidence/ITSL_stage4_20261014/runs/` | 90 | `server_evidence/ITSL_stage4_20261014/verification/` |
| Study 2B | `server_evidence/ITSL_Stage2B_Corrected_RA_20261022/runs/` | 15 | `source_snapshots/ITSL_Stage2B_Corrected_RA_20261022/verify_run.py` |
| FedDisco | `InfoOpt_FedDisco_Evidence/Extracted/runs/` | 20 | `InfoOpt_FedDisco_Evidence/Extracted/STAGE4_VERIFICATION.txt` |

Run counts were obtained by counting immediate subdirectories of each `runs/` folder. The Stage 1 inventory's count of 315 was incomplete because it omitted Studies 3, 4, and 5 (Stage 1, Stage 3, and Stage 4 directories). The corrected total is 495 InfoOpt runs + 20 FedDisco runs = 515 runs.

---

## 3. Study 3 (Stage 1) arm breakdown

`ITSL_stage1_20261011/runs/` contains 90 runs:

- 3 datasets × 5 seeds × 6 arms = 90.
- Arms: `fedavg`, `infoopt`, `shuffled_infoopt`, `uniform`, `damping_050`, `damping_075`.
- All use Dirichlet alpha = 0.3 and 100 rounds.

---

## 4. Study 4 (Stage 3) arm breakdown

`ITSL_stage3_20261012/runs/` contains 180 runs:

- 3 datasets × 2 alphas (0.1, 1.0) × 5 seeds × 6 arms = 180.
- Arms: `fedavg`, `infoopt`, `shuffled_infoopt`, `uniform`, `damping_050`, `damping_075`.

---

## 5. Study 5 (Stage 4) arm breakdown

`ITSL_stage4_20261014/runs/` contains 90 runs:

- 3 datasets × 5 seeds × 6 arms = 90.
- Arms: `fedavg`, `infoopt`, `shuffled_infoopt`, `uniform`, `damping_050`, `damping_075`.
- All use ResNet-18 and Dirichlet alpha = 0.3.

---

## 6. Study 2B corrected run breakdown

`ITSL_Stage2B_Corrected_RA_20261022/runs/` contains 15 runs:

- 3 datasets × 5 seeds × 1 arm (`RA_InfoOpt` probe off) = 15.
- These are fresh corrected runs, not the original Study 2 RA-InfoOpt runs.

---

## 7. FedDisco extension run breakdown

`InfoOpt_FedDisco_Evidence/Extracted/runs/` contains 20 runs:

- 5 seeds (100–104) × 4 arms = 20.
- Arms: `fedavg`, `uniform`, `feddisco`, `feddisco_shuffled`.
- All use CIFAR-100, SmallCNN without AIB, Dirichlet alpha = 0.3, 100 rounds.

Verified on the Linux server and independently inspected after transfer:

- All 20 runs have exactly 100 completed rounds.
- Cross-arm partition, initialization hash, and participation-schedule checks pass for every seed.
- FedDisco weight reconstruction has zero mismatches across all 2,000 round records.
- Shuffled-arm discrepancy multisets match the original arm; nonidentity permutation frequency is 98–100 %.
- No fallback events occurred.

---

## 8. Derived result packages vs. raw runs

| Derived package | Source runs | Role |
|-----------------|-------------|------|
| `analysis/data_fourarm/` | Study 1 (60 runs) | Per-seed round-100 CSV and contrasts. |
| `analysis/data_ra/` | Study 2 (60 runs) | Per-seed round-100 CSV and contrasts. |
| `analysis/data/` | Study 3 (90 runs) + sensitivity v2 + historical strata | Primary summary and paired contrasts. |
| Supplement D.4 damping summaries | Study 4 (180 runs) | Damping γ = 0.5 / 0.75 descriptive results. |
| Study 5 results | Study 5 (90 runs) | ResNet-18 backbone results (not in main tables). |
| `feddisco_extension/runs/` | FedDisco (20 runs) | Full run directories included in public release. |

---

## 9. Discrepancies and corrections from Stage 1 inventory

- **Stage 1 inventory undercount:** Reported 315 counted directories, equated incorrectly with 495 runs. The correct count is 495 InfoOpt run directories (Studies 1–5 + Study 2B) plus 20 FedDisco runs = 515 total.
- **Study numbering:** The Stage 1 inventory did not map manuscript study numbers to evidence directories. This reconciliation establishes that mapping.
- **FedDisco evidence location:** Stage 1 reported the evidence as inaccessible. It has now been transferred and verified.

---

## 10. Unresolved data artifact

The frozen step-size pilot asset package (`analysis/assets/ITSL_stepsize_pilot_20261011/`) lists `per_example_correction/pilot_validation_gamma_fix.patch` in its `SHA256SUMS.txt` manifest, but the file is missing from both `InfoOpt_Code_For_Journal/final_analysis/assets/` and `InfoOpt_GitHub_Release/analysis/assets/`. `make_tables.py` was modified in the public release to warn and continue when this file is absent. The table content is unaffected because the patch is not a data input to Table A.6.
