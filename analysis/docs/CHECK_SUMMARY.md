# Compilation and check report (revision 8)

Revisions 4–7 are preserved unchanged at `/home/claude/InfoOpt_revised`,
`/home/claude/InfoOpt_revision5`, `/home/claude/InfoOpt_revision6`, and
`/home/claude/InfoOpt_revision7`. This report covers
`/home/claude/InfoOpt_revision8`, which integrates a new, recovered
CIFAR-100 saved-checkpoint step-size pilot on top of the revision-7
baseline (see `docs/CHANGELOG.md`, "Revision 8"). Every revision-7
historical/retrospective number, table and figure, the revision-5 four-arm
experiment, and the revision-7 matched RA-InfoOpt experiment are carried
over unchanged (confirmed byte-identical, see "Preservation" below); no
training, tuning, repeat selection, new statistical analysis beyond
independent verification, or replacement of earlier frozen results was
performed.

## Builds
| Document | Class | Result | Pages | Undefined refs / cites | Errors |
|---|---|---|---|---|---|
| `supplementary.tex` | `article` (real build) | OK | 17 | 0 / 0 | 0 |
| `main_revised.tex` | `cas-dc`, **validation build only** (see below) | OK | 24 | 0 / 0 | 0 |

**Notes:**
- **Validation build.** `cas-dc.cls` and `cas-common.sty` were not supplied; the supplied `elsarticle.cls` is a different class. The main PDF was therefore compiled in a scratch copy using a stand-in for `cas-dc` only. Both documents use the **official `elsarticle-num.bst`** you supplied. The project itself contains no stand-in class.
- **BibTeX.** Main: 34 entries, the same keys as the original. Supplement: 21 entries, the Table E.1 citations, which were undefined in the original supplement.
  - There are 2 warnings per document, both "empty pages" for the ICLR entries `acar2021federated` and `reddi2021adaptive`, which have no page numbers. The bibliography is unchanged.
- **Rendered PDFs.** No "??", no "Review:" flags and no placeholder text remain.
- **Overfull boxes.** Main: same 2 pre-existing boxes, unchanged from revision 6 (equation at line 928, 23 pt; Table 1, 22 pt). Supplement: same 6 pre-existing boxes, unchanged from revision 6, all under 11 pt.
- **New section build check.** The new main §5.5 (Matched RA-InfoOpt Experiment) and supplement §A.8 introduce no new overfull boxes. The page count grew (main 22→24, supplement 13→15) because of the added text and two new tables; no historical table, figure, or frozen numerical value changed. No figure was added or modified in that revision.
- **Cross-document reference fix.** Two hard-coded cross-document references in the generated tables (`tables/tableA5_ra_probe.tex`'s reference to the main-text table and `tables/table6_ra_matched.tex`'s reference to Supplementary Table A.5) initially used `\ref` to a label defined only in the *other* document's separate LaTeX build (undefined on first build, since main and supplement have independent label namespaces). Fixed by using plain text ("main Table~6", "main Table~5", "Supplementary Table~A.5") in `scripts/make_tables.py`, consistent with how every other main↔supplement cross-reference in this project is written. Two analogous fixes were made directly in `main_revised.tex`'s new §5.5 prose (`Supplementary Section~A.8`, `Supplementary Table~A.5`, both plain text).
- **(new, revision 8) New section build check.** The new main §6.2 discussion paragraph and supplement §A.9 (Saved-checkpoint step-size pilot, Table A.6) introduce no new overfull boxes in the final build. One intermediate overfull box (an unbreakable `\texttt{validation_selected_scaling}` token in the export-defect paragraph, 15.7pt) and one earlier intermediate overfull box inside the new table's footnote minipage (a `\texttt{...gate_failure_reason...}` identifier, 53.0pt) were each found and removed during drafting by rewording the surrounding prose; the supplement's final overfull-box count (6) is identical to revision 7's, confirming no regression. The page count grew (supplement 15→17) because of the added subsection and table; no historical table, figure, or frozen numerical value changed. No figure was added or modified in this revision.
- **(new, revision 8) Cross-document reference fix.** Three hard-coded cross-document references initially used `\ref` to a label defined only in the other document's separate build: the new main §6.2 paragraph's reference into the supplement, and the new supplement §A.9's reference to main §6.2 (both in two places, including the Record-of-Revisions audit-table row). Fixed by using plain text ("Supplementary Section~A.9, Table~A.6", "main Section~6.2") in `main_revised.tex` and `supplementary.tex`, consistent with every other main↔supplement cross-reference in this project.

## Data integrity
- **Figure package.** SHA256SUMS: 14/14 OK. The package is stored read-only in `assets/` and is unchanged.
- **Copies.** The four package files in `data/` are byte-identical to the package source data. So are three of the four verified `Figures/*.pdf`.
- **Re-rendered figure.** `figure_repeat_sensitivity.pdf` was re-rendered to fix the label placement only; its plotted values are asserted equal to the source.
- **Sensitivity v2 package.**
  - The archive SHA-256 is `ba5f20e430b6eafb551b4ed6c95240c4797b11141fa3ccc984fa0fa482469a96`, matching the hash recorded in the figure package README.
  - Its internal `SHA256SUMS.txt` passes 11/11. It is stored read-only in `assets/`.
  - `outputs/sensitivity_cifar_fedopt.json` has SHA-256 `e3d77c51…40e`.
  - The package's RA-strata and latest-repeat outputs are byte-identical to `data/`.
  - The package's own logs report 54/54 sources and 160/160 manifest entries verified. These were not re-run here, because they need the raw run directories.
- **Originals.** `original/` holds the untouched originals; `references.bib` is identical to the original bibliography.
- **Four-arm package (new, revision 5).**
  - Supplied separately from the manuscript archive, as `ITSL_four_arm_analysis_20261010.zip`.
  - All 8 files pass its own `SHA256SUMS.txt` (`ANALYSIS_REPORT.md`, `PROTOCOL_BEFORE_LAUNCH.md`, `aib_interactions.json`, `analyze_fourarm.py`, `per_seed_round100.csv`, `primary_contrasts.json`, `primary_summary.csv`, `verification_report.json`).
  - Stored read-only and unchanged in `assets/ITSL_four_arm_20261010/`.
  - Working copies in the new `data_fourarm/` directory (prefixed `fourarm_`, kept separate from the historical `data/` files of the same base names) are verified byte-identical to the frozen package by `scripts/make_tables.py::verify_fourarm_package()` on every build, and a tampered working copy is rejected (regression-tested in `test_make_tables.py`).
  - **Raw-results collection (reported by the package, not re-run here).** `verification_report.json` reports 406/406 raw result files hash-verified against `COLLECTION_MANIFEST.json`; 60/60 unique dataset-seed-arm runs verified (execution-manifest evidence hashes + server verification reports); partitions and cohort schedules confirmed equal across all four arms of each dataset-seed. The archive SHA-256 `0a56c0080cb8da42566be1ca4a976937d50f1f273b10eb06d4eb76d4b660f385` reported there is the hash of this raw-results collection, **not** of the analysis package (see next bullet); the raw run directories were not supplied, so this collection-level check was taken as the package's own claim, not independently re-run. Checkpoints were not independently reloaded (also a recorded limitation of the package).
  - **Analysis package (independently re-verified here).** All 8 checksummed files of the supplied `ITSL_four_arm_analysis_20261010.zip` pass its own `SHA256SUMS.txt`, confirmed by this session.
  - **Independent recomputation.** Every mean, SD, 95% CI, raw $p$, and Holm $p$ in `primary_contrasts.json` and `aib_interactions.json` was recomputed from `per_seed_round100.csv` (paired-$t$; Holm–Bonferroni, $m=6$ for the primary family and $m=3$ for the secondary interaction analysis) in `scripts/check_numbers.py`, independently of the supplied JSON, and matches to $10^{-9}$.
- **FedDyn/SCAFFOLD repair-campaign audit package (new, revision 6).**
  - Supplied separately, as `ITSL_FedDyn_SCAFFOLD_audit_final_20261010.zip`.
  - Top-level `SHA256SUMS.txt`: 148/148 entries pass, independently re-verified in this session (covers `campaign_source_map.json`, `campaign_summary.json`, `earlier_vs_campaign.json`, `run_inventory.json`, `equation_check_results.json`, every `evidence/run_metadata/*/{config_used.json,metrics.csv,summary.json}`, every `evidence/runs/_logs/*.log`, `evidence/train.py`, `evidence/utils/feddyn.py`, `evidence/methods/client.py`, `evidence/scripts/launch_phase2_{feddyn,scaffold}.sh`, `phase2_provenance.txt`, and the `synthetic_checks/` subtree).
  - `synthetic_checks/SHA256SUMS.txt`: 9/9 entries pass, independently re-verified in this session.
  - `phase2_provenance.txt` hash `a239b9eb4bc436e6d38c7e0fe122bbdd1433ae2fdefb4f072ac30aa4a2d09e61` independently recomputed and confirmed byte-identical to the separately packaged `synthetic_checks/phase2_provenance.txt` copy referenced by `FINAL_AUDIT_DECISION.md` and `synthetic_checks/README.md`.
  - `equation_check_results.json`: 17/17 CPU-only synthetic equation checks report `PASS`, as supplied; not independently re-executed in this session (no PyTorch in this environment), consistent with the task's "no new training/tuning/statistical analysis" constraint. Script hash and 5 snapshot-file hashes in the JSON match the checksum-verified package.
  - `campaign_source_map.json`: 30 run records (15 SCAFFOLD + 15 FedDyn), each with `historical_code_hash: "NOT RECOVERED"`; 10 CIFAR-100 records have `early_stop_enabled_saved: true`.
  - `campaign_summary.json`: six descriptive means ($n=5$ each), each explicitly labelled by the package itself as "campaign-linked descriptive audit only; no source-version certification". These six values, to two decimal places, numerically equal the corresponding FedDyn/SCAFFOLD cell of the manuscript's original, since-withdrawn table (cross-checked directly against `original/main_elsevier_ORIGINAL.tex`'s Table~\ref{tab:canonical} in this session).
  - Stored read-only in `assets/ITSL_FedDyn_SCAFFOLD_audit_final_20261010/`.
  - No new training, tuning, or statistical analysis was performed on this package; its own supplied numbers are reported as historical descriptive evidence only, independently checksum-verified, not recomputed.
- **Matched RA-InfoOpt package (new, revision 7).**
  - Supplied separately, as `ITSL_RA_matched_analysis_20261010.zip`.
  - All 10 checksummed files pass its own `SHA256SUMS.txt` (`ANALYSIS_REPORT.md`, `NUMERICAL_CHECKS.json`, `PROTOCOL.md`, `analyze_ra_matched.py`, `per_seed_round100.csv`, `primary_contrasts.json`, `primary_summary.csv`, `probe_contrasts.json`, `source_manifest.json`, `verification_report.json`), independently re-verified in this session.
  - Stored read-only and unchanged in `assets/ITSL_RA_matched_20261010/`.
  - Working copies in the new `data_ra/` directory (prefixed `ra_`, kept separate from the historical `data/` files and the revision-5 `data_fourarm/` files) are verified byte-identical to the frozen package by `scripts/make_tables.py::verify_ra_package()` on every build, and a tampered working copy is rejected (regression-tested in `test_make_tables.py`).
  - **Verified counts.** `per_seed_round100.csv`: 60 unique (dataset, seed, arm) records (3 × 5 × 4), confirmed by this session with no duplicates. `primary_summary.csv`: 12 rows (4 arms × 3 datasets). `primary_contrasts.json`: 6 primary comparisons. `probe_contrasts.json`: 3 secondary (probe) comparisons.
  - **Raw-results collection (reported by the package, not re-run here).** `verification_report.json` reports 537 collection-file hashes checked and all 60 dataset-seed-arm cells verified (execution-manifest evidence + server verification reports); exact rounds 1–100, finite/bounded accuracy, no early stopping, and shared partition/cohort-schedule agreement across all four arms of each cell are confirmed; no failed attempts recorded. Checkpoints remained on the server: checkpoint strict-loading and tensor-finiteness checks rely on these collected server reports and were not independently repeated here, since checkpoints were not themselves supplied to this analysis workspace.
  - **Analysis package (independently re-verified here).** All 10 checksummed files pass `SHA256SUMS.txt`, confirmed by this session.
  - **Independent recomputation.** Every mean, SD, 95% CI, raw $p$, and Holm $p$ in `primary_contrasts.json` and `probe_contrasts.json` was recomputed from `per_seed_round100.csv` (paired-$t$; Holm–Bonferroni, $m=6$ for the primary family and $m=3$ for the secondary probe family), independently of the supplied JSON and of the supplied `analyze_ra_matched.py` script, and matches to $10^{-6}$. No discrepancy was found.
  - **Stale manifest label.** `source_manifest.json`'s top-level `"experiment"` field reads `"four_arm_20261009"`, a leftover label from an earlier naming scheme (while its `run_dir` paths correctly reference `ITSL_RA_matched_20261010`). Preserved unaltered in the frozen manifest; documented in-text as a metadata issue, not used for run selection, and not mistaken for the distinct revision-5 four-arm package.
- **Saved-checkpoint CIFAR-100 step-size pilot package (new, revision 8).**
  - Supplied separately, first as `50527208-ITSL_coverage_audit_20261011.md` (the coverage audit alone, insufficient per its own "do not treat the research-history document as a substitute for raw authoritative summaries" instruction), then as the full evidence archive `f76202ff-ITSL_stepsize_writing_evidence_20261011.zip`.
  - All 59 files pass its own top-level `SHA256SUMS.txt`, independently re-verified in this session both on first extraction and again after copying into `assets/ITSL_stepsize_pilot_20261011/`.
  - Stored read-only and unchanged in `assets/ITSL_stepsize_pilot_20261011/`.
  - Working copies in the new `data_stepsize/` directory (prefixed `stepsize_`, kept separate from every other `data_*/` directory) are verified byte-identical to the frozen package by `scripts/make_tables.py::verify_stepsize_package()` on every build, and a tampered working copy is rejected (regression-tested in `test_make_tables.py`).
  - **Authoritative sources read and cross-checked before any prose or table was written** (per the task's "machine-check numbers from authoritative summaries" instruction): `README.md`, `pilot_documents/PROTOCOL.md` (v3), `pilot_documents/GO_NO_GO.md` (v3), `pilot_documents/PRE_EXECUTION_AMENDMENT.md`, `pilot_documents/PILOT_CLOSURE.md`, `pilot/results/pilot_decision.json`, `corrected_exploratory_v2/corrected_exploratory_report.md` + its two JSON files, `buffer_policy_v2/report.md` + `cohort_summary.csv`, `per_example_correction/CORRECTION_REPORT.md`. Every number in `PILOT_CLOSURE.md` and the coverage audit was independently confirmed against these raw sources; no discrepancy was found.
  - **Critical sourcing distinction confirmed.** `pilot_decision.json`'s own `exploratory_variability_signal` block holds the stale, pre-correction snapshot (`fit_status="no_cells"`, 0 cells used for both predictors) — the documented defect from `PILOT_CLOSURE.md` §2. The manuscript's exploratory-predictor numbers are sourced from `corrected_exploratory_v2/exploratory_signal_held_out_corrected.json` instead, and `load_stepsize_exploratory()` raises if `fit_status != "fit"`, so this cannot silently regress.
  - **Frozen-verdict guard.** `load_stepsize_decision()` asserts the four top-level verdict strings (`adaptive_headroom=INSUFFICIENT`, `claude_comparator=NO-GO`, `exploratory_variability_signal=EXPLORATORY_PROBE`, `overall_recommendation=ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT`) against the frozen `GO_NO_GO.md` v3 vocabulary before extracting any per-seed $G$-value; a changed verdict string raises rather than silently flowing into the table (regression-tested).
  - **Independent recomputation.** Every buffer-policy cohort-mean comparison (freeze-at-$\gamma=1$, validation-selected vs. both $\gamma=1$ references, validation-selected vs. both dev-selected fixed scales) was independently recomputed in `scripts/make_tables.py::load_stepsize_buffer()` from the raw `buffer_policy_v2/cohort_summary.csv` (180 rows), not transcribed from `report.md`, and matches `report.md`'s own reported values to $10^{-9}$.
  - **Export defect.** `per_example_correction/CORRECTION_REPORT.md`'s root-cause, measured-impact (450,000 defective rows; 77 of 90 sources, 154 of 900 checked cells mismatched), and partial-regeneration (9 of 90 sources, 45,000 of ~2.25M rows) figures were read in full and quoted verbatim in the new supplement subsection; the original defective CSV is not part of the supplied evidence and no sample-level statistic is computed from the defective or partial-corrected rows.
  - **Amendment byte-identity.** `PRE_EXECUTION_AMENDMENT.md` is documented, per the supplied `README.md`'s own instruction, as a later change record whose bytes do not match the historical frozen hash; it is not certified as the original pre-execution payload. This is disclosed in the new supplement subsection and does not affect any reported number, which is corroborated independently via `PILOT_CLOSURE.md` and the checksum-verified `pilot_decision.json`.

## Preservation of earlier evidence (revision 7; re-confirmed unchanged in revision 8)
- `data/primary_summary.csv`, `data/paired_contrasts.json`, `data/sensitivity_latest_repeat.json`, and `data/ra_strata_summaries.json` are confirmed byte-identical (SHA-256) to their revision-6 copies.
- `data_fourarm/`'s five working-copy files are confirmed byte-identical to their revision-6 copies (via `sha256sum`, independently of the in-script `verify_fourarm_package()` check, which continues to pass against the frozen `assets/ITSL_four_arm_20261010/` package).
- `data_ra/`'s working-copy files continue to verify byte-identical to the frozen `assets/ITSL_RA_matched_20261010/` package via `verify_ra_package()`.
- Historical `primary_summary.csv` row count (36 = 3 datasets × 12 arms), the revision-5 four-arm record counts (12 primary rows, 6 contrasts, 3 interactions), and the revision-7 RA-matched record counts (12 primary rows, 6 contrasts, 3 probe contrasts, 60 unique per-seed cells) are unchanged.
- No historical table, figure, or supplement subsection was edited beyond the additions listed in `docs/CHANGELOG.md`.

## Table generator (`scripts/test_make_tables.py`): 39/39 pass
- CIFAR-100 FedOpt primary status is "excluded".
- An injected sensitivity row cannot overwrite the excluded primary cell.
- When present, such a row appears only in the footnote.
- When absent, it is reported as absent, not invented.
- A rule–data conflict raises an error.
- 36 unique primary keys.
- The CIFAR-100 FedOpt sensitivity value (44.72 ± 2.15, n = 5) appears only in the footnote, and the primary cell stays excluded.
- The value is read from the original frozen output (44.724 ± 2.1531, n = 5).
- A tampered copy of that output is rejected on checksum.
- Duplicate sensitivity-only records raise an error.
- **(new)** Four-arm package checksums verified (frozen package + `data_fourarm/` byte-identical copies).
- **(new)** Four-arm primary summary has 12 rows (4 arms × 3 datasets).
- **(new)** Four-arm primary family has six within-AIB InfoOpt-Full − FedAvg contrasts.
- **(new)** Four-arm interaction family has three aggregation × AIB interactions.
- **(new)** A tampered four-arm working copy is rejected (byte-identity check).
- **(new)** Only the CIFAR-100 interaction excludes zero at 95% CI.
- Revision 6 added no new tables, so no new table-generator tests were added for the audit itself; 16 of the above rerun unchanged.
- **(new)** 3 regression tests for `scripts/span_scan.py`'s position-aware superseded-value exceptions (added in the revision-6 corrections pass): a campaign token confined to its descriptive-audit passage is accepted; the same token leaked into unrelated primary-result prose outside that passage is caught; the in-passage occurrence is still counted correctly alongside the stray one.
- **(new, revision 7)** RA-matched package checksums verified (frozen package + `data_ra/` byte-identical copies).
- **(new, revision 7)** RA-matched primary summary has 12 rows (4 arms × 3 datasets).
- **(new, revision 7)** RA-matched primary family has six RA(probe off)−FedAvg/InfoOpt-Full contrasts.
- **(new, revision 7)** RA-matched secondary family has three probe-on/off contrasts.
- **(new, revision 7)** RA-matched per-seed data has 60 unique (dataset, seed, arm) cells.
- **(new, revision 7)** A tampered RA-matched working copy is rejected (byte-identity check).
- **(new, revision 7)** Exactly CIFAR-100 RA−FedAvg and EuroSAT RA−InfoOpt-Full exclude zero (individual CIs); no RA-matched primary contrast survives Holm correction; all three RA-matched probe intervals include zero.
- **(new, revision 8)** Step-size pilot package checksums verified (frozen package + `data_stepsize/` byte-identical copies).
- **(new, revision 8)** Headroom and claude-comparator $G$-values loaded for both held-out seeds × both fixed controls; the tested shrinkage rule is worse than both fixed controls on every evaluated cell.
- **(new, revision 8)** A changed `adaptive_headroom` verdict string is rejected (frozen-vocabulary guard against `GO_NO_GO.md` v3).
- **(new, revision 8)** A tampered step-size working copy is rejected (byte-identity check).
- **(new, revision 8)** A stale/defective `fit_status="no_cells"` exploratory file is rejected by the corrected-exploratory loader.
- **(new, revision 8)** Buffer-policy cohort-mean cell counts (18 held-out / 9 dev, trajectory cohort excluded) and the freeze-at-$\gamma=1$ held-out loss-improvement sign, independently recomputed from `cohort_summary.csv`.

## Numerical checks (`scripts/check_numbers.py`): 367/367 pass (with the build directory; includes cross-document reference and LaTeX-log checks)
Full output: `check_numbers_output.txt`.
1. **Primary family recomputed from mean, sd_diff and n.** Raw p, CI bounds and Holm (m = 6) match the verified JSON to 1e-9.
   - **(new) Four-arm primary and interaction families recomputed from `per_seed_round100.csv`** (mean, SD, paired-t CI, raw p for all 6 primary + 3 interaction contrasts; Holm m=6 and m=3 respectively): all match the supplied JSON to 1e-9.
2. **Tables against outputs:**
   - Table 3: all 32 reported cells and the 3 withheld cells.
   - The excluded CIFAR-100 FedOpt cell; its sensitivity value appears in the footnote only and is quoted consistently in the main text and supplement.
   - Tables 4 and A.1: Δ, CI, raw p, Holm p and d_z; three "not estimable" rows each.
   - Table A.2: all 9 sensitivity records.
   - Table A.3: all 9 strata. Means and SDs were also recomputed in `make_tables.py`.
   - **(new) Table 5(a):** all 12 four-arm summary cells. **Table 5(b):** all six contrasts (Δ, CI, p_raw, p_Holm). **Table A.4:** all three interactions.
3. **Prose numbers.** 23 numbers quoted in the main text and 9 in the supplement match the outputs for the retrospective reanalysis. These include the sensitivity contrasts, 0.164, 165/135, and the damping summaries.
   - **(new)** 20 additional main-text tokens for the four-arm experiment (6 contrast means + 6 contrast CIs + 3 interaction means + 3 interaction CIs + 2 interaction p-values, pp-rounded) match the independently recomputed values.
4. **Superseded values.** None of the superseded historical values remains as a result. The exceptions are deliberate:
   - 87.92 and 0.048 appear in the supplement's audit text describing withdrawn claims.
   - 44.72 appears as the verified CIFAR-100 FedOpt sensitivity-only value. It numerically equals the old printed FedOpt mean but is not presented as a primary result.
   - (The new four-arm summary means — 43.10–86.09% range — do not coincide with any superseded historical token.)
   - **(new, revision 6)** 29.10, 42.28, 74.84, 85.10, 85.58, and 87.92 appear in main §4.5 (six tokens) and supplement §A.2 (five tokens; 87.92 was already an intentional exception) as the FedDyn/SCAFFOLD repair-campaign descriptive means from `campaign_summary.json`. These are historical descriptive audit evidence, not primary results, and are explicitly disclosed in-text as numerically reproducing the withdrawn table row; `check_numbers.py`'s `intentional` exception set and per-token rationale were extended accordingly (see `CHANGELOG.md`, "Revision 6").
   - **(new, revision 7)** The RA-matched summary means (43.35–86.39% range) and contrast values do not coincide with any superseded historical token; no new span exception was needed.
   - **(new, revision 8)** All headroom/claude-comparator $G$-values, corrected exploratory MAE/MSE values, and buffer-policy cohort-mean values are small signed decimals (e.g. $+0.005737$, $-0.019722$, $0.009036$, $-0.0629$) that do not coincide with any superseded historical token; no new span exception was needed.
5. **Cross-document references.** All hard-coded references resolve to the intended labels:
   - Main → supplement: Tables A.1–A.6, Figs. A.1–A.2, Sections A.4, A.7, A.8, A.9, D and G, Table E.1.
   - Supplement → main: §4.5, §5.4, §5.5, §6.2, Tables 3–6, Figs. 1–5, §3.4, §4.3.
   - **(new)** `sec:fourarm` = 5.4, `tab:fourarm` = Table 5, `fig:fourarm` = Fig. 5 (main `.aux`); `app:fourarm` = A.7, `app:tab:fourarm_interaction` = Table A.4 (supplement `.aux`).
   - **(new, revision 6)** Main §4.5's reference to the new supplement audit paragraph uses a plain-text "Supplementary Section~A.2" (not `\ref`), consistent with how every other main→supplement cross-reference in this project is written, since main and supplement are separate LaTeX builds with independent label namespaces.
   - **(new, revision 7)** `sec:ra_matched` = 5.5, `tab:ra_matched` = Table 6 (main `.aux`); `app:ra_matched` = A.8, `app:tab:ra_probe` = Table A.5 (supplement `.aux`). Main §5.5's two references into the supplement, and the generated `tableA5_ra_probe.tex`/`table6_ra_matched.tex`'s two references into the other document, all use plain text rather than `\ref`, for the same cross-build reason as above.
   - **(new, revision 8)** `subsec:discussion_diagnostics` = 6.2 (main `.aux`); `app:stepsize_pilot` = A.9, `app:tab:stepsize_pilot` = A.6 (supplement `.aux`). Main §6.2's reference into the supplement, and the new supplement §A.9's two references into main §6.2 (prose paragraph and the Record-of-Revisions audit-table row), all use plain text rather than `\ref`, for the same cross-build reason as above.
6. **Logs.** No undefined references or citations, no multiply-defined labels, no errors, in either document.
7. **(new, revision 6) FedDyn/SCAFFOLD audit package checksums.** Top-level `SHA256SUMS.txt` (148 entries) and `synthetic_checks/SHA256SUMS.txt` (9 entries) both independently re-verified, all pass. `phase2_provenance.txt`'s SHA-256 independently recomputed and confirmed byte-identical to its separately packaged copy, matching the hash asserted in `FINAL_AUDIT_DECISION.md` and `synthetic_checks/README.md`. (This check is a standalone `sha256sum -c` verification during this session, not part of `check_numbers.py`'s count.)
8. **(new, revision 7) Preservation checks.** `data/` historical files and `data_fourarm/` working copies confirmed byte-identical to revision 6 (SHA-256); historical and four-arm record counts unchanged (36/12/6/3).
9. **(new, revision 8) Step-size pilot section "1e".** Package checksums (59/59); frozen verdict strings against `GO_NO_GO.md` v3; every headroom and claude-comparator $G$-value against `pilot_decision.json` (8 values); the two corrected exploratory predictors' held MAE/MSE against `exploratory_signal_held_out_corrected.json` (not the stale `pilot_decision.json` block); the five buffer-policy cohort-mean comparisons independently recomputed from `cohort_summary.csv` (10 values, held + dev). Extends section 2 with Table A.6, section 3 with 20+ prose-quote checks (verdicts, MAE/MSE, buffer values, export-defect counts, scope statements, amendment byte-identity disclosure), and section 5 with the three new cross-document labels above.

## Figures and captions (checked against the rendered figures)
| Figure | Checked |
|---|---|
| Fig. 3 (accuracy) | 12 rows; "Withheld" for RA-InfoOpt; "Excluded: strict E3" for CIFAR-100 FedOpt; orange squares and dagger for FedDyn/SCAFFOLD; AIB labels; n = 5 note — all reflected in the caption |
| Fig. 4 (contrasts) | 3 estimates with Holm annotations; 3 "Not estimable" rows; zero line — reflected |
| Fig. 5 (four-arm) | Panel (a): six primary contrasts, grouped by dataset and AIB setting, with Holm p annotations. Panel (b): three interaction contrasts, CIFAR-100 marked as the only 95% CI excluding zero. Rendered by `scripts/plot_fourarm.py`, which asserts every plotted value equals `data_fourarm/` source — reflected in the caption. Unchanged in revision 7 |
| Fig. A.1 (repeat sensitivity) | Latest minus earliest; six points; −0.037 and −0.164 — reflected. Plotted values asserted equal to the source |
| Fig. A.2 (RA strata) | Hash-prefix labels; n and seeds; no bar for n = 1; separate y-scales — reflected |
| Figs. 1–2 | Fig. 2's "predefined" and "fixed configurations" labels are caveated in its caption |

**No new figure was added in revision 7.** Table 6 (main, `tables/table6_ra_matched.tex`) and Table A.5 (supplement, `tables/tableA5_ra_probe.tex`) carry the new matched RA-InfoOpt evidence in tabular form only; both are auto-generated by `scripts/make_tables.py` and checked against `data_ra/` in `scripts/check_numbers.py` §2.

**No new figure was added in revision 8.** Table A.6 (supplement, `tables/tableA6_stepsize_pilot.tex`) carries the new saved-checkpoint step-size pilot evidence in tabular form only; it is auto-generated by `scripts/make_tables.py` and checked against `data_stepsize/` in `scripts/check_numbers.py` §1e/§2.

## Review notes resolved
| Note | Resolution |
|---|---|
| Repeat-sensitivity definition | Confirmed from S: latest minus earliest |
| Table 3 placeholder | Generated from P |
| Fig. 3 caption | Rewritten from the package README and checked |
| Fig. 4 caption | Rewritten and checked |
| Secondary family (historical 21-comparison external-baseline family) | Decided: not reported; no paired outputs supplied. (Distinct from the new three-test aggregation×AIB secondary analysis of revision 5, which **is** reported — Table A.4.) |
| Supplement captions (×2) | Rewritten and checked |
| NO-GO verdict | Replaced by "consistent superiority not established" |
| Architecture confound (FedAvg vs. InfoOpt-Full) | **Resolved in revision 5** by the four-arm controlled experiment (Section 5.4); does not extend to RA-InfoOpt, FedDyn, or SCAFFOLD |
| FedDyn/SCAFFOLD implementation provenance | **Partially addressed in revision 6**: 30 repair-campaign runs now identified by exact path/log provenance and the recovered implementation passes 17/17 synthetic equation checks (Section 4.5); historical source-code identity remains unresolved, and the CIFAR-100 early-stop eligibility conflict is disclosed, not amended |
| RA-InfoOpt configuration heterogeneity (historical) | **Partially addressed in revision 7**: a new matched experiment (Section 5.5) fixes a single configuration across all five seeds, but none of its six primary contrasts survives Holm correction; it does not recover, certify, or substitute for the withheld historical five-seed estimate (Section 4.6), which remains unresolved |
| Whether tuned fixed-step controls are needed to evaluate adaptive step-size headroom | **Addressed (new) in revision 8**: a bounded saved-checkpoint CIFAR-100 pilot (Section 6.2, supplement §A.9) measures that validation-selected per-source scaling does not meet the frozen headroom gate against development-tuned global and phase-specific fixed scales; this is a measured negative for this one pilot only, not a general claim about step-size control |

## Revision 4 corrections (carried over, unchanged)
| Correction | Applied |
|---|---|
| CIFAR-100 FedOpt sensitivity | 44.72 ± 2.15 (5), footnote only; primary cell excluded. Read from the original checksum-verified output |
| RA e9827b note | States: identical terminal accuracy, distinct directories and different trajectories; not a duplicate |
| Timing sentence | The required sentence is used verbatim in §4.5 and supplement A.6, and paraphrased in §6.3 |
