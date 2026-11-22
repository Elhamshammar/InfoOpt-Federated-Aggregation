# Change log

## Revision 8 corrections — 2026-10-11 (wording only, no numerical changes)

Three wording corrections to the revision-8 text below, no new experiments
and no numerical changes:

1. **Selection-split description corrected (supplement §A.9).** The scope
   paragraph said development seed 201 was used for "all scale selection,"
   which incorrectly implied validation-selected scaling's per-source gamma
   was also chosen on the dev seed. The frozen protocol allows each
   held-out source's own selection split to choose its own gamma. Now
   reads: "Development seed 201 selected the global and phase-specific
   fixed scales and fitted the exploratory predictors before held-out
   processing. Validation-selected scaling chose gamma separately for each
   source, including sources from seeds 202 and 203, using its selection
   split; evaluation used the separate evaluation split."
2. **Table A.6 footnote corrected.** "The shrinkage rule fails the strict
   per-round screen" incorrectly described the gate as per-round; the
   `claude_comparator` gate is a mean-over-checkpoints criterion
   (`mean_G >= +0.001`), with the per-round comparison reported separately.
   Now reads: "The shrinkage rule fails the frozen mean-improvement screen
   against both controls on both held-out seeds and performs worse at
   every evaluated checkpoint."
3. **Limitations §"Provenance" paragraph qualified (main text).** "All
   reported results come from a retrospective reanalysis" incorrectly
   swept in the four-arm and matched RA-InfoOpt experiments, which are
   prospective. Now reads: "The historical comparison results come from a
   retrospective reanalysis... The subsequent AIB and matched RA
   experiments followed experimental and analysis plans frozen before
   launch." The step-size-pilot discussion paragraph's "it does not bear
   on either" is replaced with "it is analyzed separately and does not
   change either experiment's estimates," for the same reason (the pilot
   is independent, not simply inert with respect to those experiments).

**Checks.** `scripts/make_tables.py`'s only change is the Table A.6
footnote wording (no value changed); `test_make_tables.py`: 39/39 pass
(unchanged). `check_numbers.py`: all checks pass (unchanged pass/fail
behavior; no number changed).

**Rebuild.** Main: validation build, 0 errors, 24 pages (unchanged), same
2 pre-existing overfull boxes (unchanged). Supplement: 0 errors, 17 pages
(unchanged), same 6 pre-existing overfull boxes (unchanged), no undefined
references or citations in either document.

---

## Revision 8 — 2026-10-11 (saved-checkpoint CIFAR-100 step-size pilot)

Revision 7 (with its two post-delivery wording corrections) is preserved
unchanged at `/home/claude/InfoOpt_revision7`. Every frozen table,
contrast, and numerical result from revision 7 is unchanged in revision 8;
this revision adds one new, independent piece of evidence and corrects no
existing claim.

**Evidence added.** A previously closed, bounded CIFAR-100 pilot
(`stepsize_diag_pilot_20261004_175620`) evaluating per-source step-size
scaling as one-step interpolations from saved FedAvg checkpoints (rounds
5/50/100), recovered together with its coverage audit, frozen protocol
and decision documents (`PROTOCOL.md`, `GO_NO_GO.md` v3), retained
grid/rule/decision summaries (`pilot_decision.json`), a corrected (v2)
exploratory-predictor recomputation, a revised buffer-policy analysis, and
an export-defect correction report. The full evidence package (59 files)
was checksum-verified against its own `SHA256SUMS.txt` before any number
below was transcribed; every number quoted in the manuscript or table was
additionally cross-checked against, or independently recomputed from, the
authoritative machine-readable source (not the coverage-audit summary
document alone).

1. **New supplementary subsection, §A.9 "Saved-checkpoint step-size
   pilot."** States the pilot's exact scope (CIFAR-100, SmallCNN with AIB
   disabled, one partition at seed 42, development seed 201, held-out
   seeds 202/203, checkpoints at rounds 5/50/100, one-step evaluations —
   not end-to-end adaptive-controller training, clustered repetitions —
   not independent training seeds), a compact source-generated table
   (new Table~A.6) comparing validation-selected scaling and the tested
   update-based shrinkage rule against the development-selected global
   and phase-specific fixed scales, and four distinguished findings: (i)
   validation-selected scaling fails the frozen headroom gate against
   both tuned fixed controls on at least one held-out seed; (ii) the
   tested raw-shrinkage rule is worse than both fixed controls on every
   evaluated checkpoint of both held-out seeds; (iii) the corrected
   phase-plus-variability exploratory predictor does not improve
   held-out prediction over phase alone; (iv) freezing checkpoint buffers
   at unit scaling worsens held-out loss in this setting (a buffer-policy
   sensitivity result, not a normalization-recalibration claim). No new
   significance test was added; repetitions within a (cohort, seed,
   checkpoint) cell are not treated as independent samples; the pilot
   does not support a universal claim about step-size control.
2. **Export-defect disclosed.** The original large per-example CSV's
   `validation_selected_scaling` rows were silently defaulted to
   $\gamma=1$ after an internal per-source gamma cache was cleared
   post-use (77 of 90 sources affected, 154 of 900 checked cells); the
   retained grid/rule/decision summaries used for every finding above
   were computed upstream of this defect and are unaffected. The original
   defective CSV is not supplied. A corrected trajectory-only export
   recovers 9 of 90 sources (45,000 of ~2.25M rows); the remaining 81
   sources cannot be regenerated without retraining. This is disclosed as
   a partial, not complete, replacement; no sample-level interval or
   bootstrap result is computed from either the defective or the partial
   corrected rows.
3. **New main-text discussion paragraph (§6.2 "Diagnostic Evidence").**
   States that, in this bounded pilot, improvement over unit scaling did
   not meet the gate for incremental benefit beyond development-tuned
   fixed scales, and connects this to the importance of tuned fixed-step
   controls when evaluating adaptive aggregation. Explicitly kept separate
   from, and states it does not bear on, the four-arm AIB experiment
   (§5.4) and the matched RA-InfoOpt experiment (§5.5).
4. **Supplement §D.4 (Dynamic- and Fixed-Damping Diagnostics) wording
   locally updated.** The existing sentence disclosing
   that the historical oracle round-dependent damping analysis's outputs
   were not supplied and its conclusions are not restored is unchanged;
   one sentence is added distinguishing it from the newly supplied
   step-size pilot (§A.9), which evaluates one-step saved-checkpoint
   scalings under its own frozen protocol and does not restore or
   substitute for the withheld oracle-damping conclusions.
5. **Provenance/amendment status distinguished.** `PRE_EXECUTION_
   AMENDMENT.md` is documented in §A.9 as a later change record whose
   bytes do not match the historical frozen hash, not the certified
   original pre-execution payload; the protocol/decision-logic content it
   describes is corroborated independently via `PILOT_CLOSURE.md` and the
   checksum-verified `pilot_decision.json`.
6. **Audit table and interpretive-scope paragraph updated** (supplement
   §G, §F) to record the new subsection/table and note that its measured
   negative outcomes are scoped to this one bounded pilot.

**No training, tuning, repeat selection, or change to any frozen analysis**
was performed; every existing table, contrast, and numerical result from
revision 7 is byte-for-byte unchanged.

**Checks.** `scripts/make_tables.py` extended with `verify_stepsize_
package()`, `load_stepsize_decision()` (asserts the four frozen GO_NO_GO.md
v3 verdict strings before extracting any G-value), `load_stepsize_
exploratory()` (reads the corrected v2 file, not `pilot_decision.json`'s
own stale `exploratory_variability_signal` block), `load_stepsize_buffer()`
(independently recomputes the five buffer-policy cohort-mean comparisons
from the raw `cohort_summary.csv`, not transcribed from `report.md`), and
`tableA6_stepsize_pilot()`. `test_make_tables.py`: 11 new checks (checksum
verification, per-seed-per-control G-value loading, frozen-verdict guard
against a tampered verdict string, byte-identity rejection of a tampered
working copy, rejection of a stale `fit_status="no_cells"` exploratory
file, and the independently recomputed buffer-policy cell counts/signs)
— 39/39 pass (28 carried over from revision 7 unchanged + 11 new).
`check_numbers.py`: new section "1e"
(checksums, frozen verdicts, every headroom/claude-comparator G-value,
corrected exploratory MAE/MSE, independently recomputed buffer-policy
means) plus extended sections 2 (Table A.6), 3 (prose quotes), and 5
(new cross-document labels `app:stepsize_pilot`=A.9, `app:tab:
stepsize_pilot`=A.6, main `subsec:discussion_diagnostics`=6.2) — all
checks pass.

**Rebuild.** Main: validation build, 0 errors, 24 pages (unchanged page
count), same 2 pre-existing overfull boxes (unchanged). Supplement: 0
errors, 17 pages (+2 for the new subsection and table), same 6
pre-existing overfull boxes (no new overfull boxes introduced), no
undefined references or citations in either document after the final
build.

---

## Revision 7 corrections — 2026-10-11 (narrow cleanup, no numerical changes)

All data, frozen evidence files, and every table/figure value from
"Revision 7" below are unchanged. This pass corrects wording only.

1. **"Preparation for retrospective recomputation" → "a prospective
   experimental and analysis plan."** Main §5.5 and supplement §A.8 both
   described the matched RA-InfoOpt experiment's frozen protocol with
   language borrowed from the historical *retrospective* reanalysis
   ("preparation for retrospective recomputation"). This new experiment is
   prospective (60 fresh runs under a plan frozen before launch), not a
   retrospective recomputation of existing runs, so both occurrences now
   read "a prospective experimental and analysis plan, not an externally
   registered preregistration."
2. **Duplicated cross-reference fixed.** Main §5.5 read "(Supplementary
   Supplementary Section~A.8)" — a line-wrap duplication introduced when
   the plain-text cross-reference was written. Now reads "(Supplementary
   Section~A.8)".
3. **Holm-adjustment wording corrected (supplement §A.8).** "None of the
   six 95% confidence intervals survives Holm adjustment" conflated two
   distinct quantities: the reported intervals are individual, unadjusted
   95% confidence intervals, while Holm adjustment is applied to the
   hypothesis tests' $p$-values, not to the intervals themselves. Now
   reads "None of the six primary hypothesis tests rejects after Holm
   adjustment ($m=6$); the reported intervals are individual, unadjusted
   95% confidence intervals, not simultaneous ones." The parallel sentence
   for the secondary (probe) family was corrected the same way.
4. **Primary/secondary family membership clarified (supplement §A.8).**
   The "Primary family" and "Secondary family" paragraphs now state
   explicitly that all six primary contrasts use RA-InfoOpt with
   counterfactual probes *disabled* (probe off) against FedAvg and
   InfoOpt-Full, and that the probe-on/off comparison is a separate,
   distinct secondary family of three that does not appear in the primary
   family.
5. **Scope statement corrected (main §5.5 `nonclaim`).** "No training,
   tuning, or retrospective selection of the better-performing RA-InfoOpt
   arm was performed" incorrectly implied the matched experiment itself
   involved no training; it involved 60 fresh training runs, reported
   earlier in the same section. Now reads "No additional training or
   tuning was performed during manuscript integration (the experiment
   itself involved 60 fresh training runs, reported above), and no
   retrospective selection of the better-performing RA-InfoOpt arm was
   performed."
6. **Probe condition reporting clarified (main §5.5 `nonclaim`).** "both
   probe conditions are reported for all six primary and all three
   secondary comparisons" incorrectly implied the primary family mixes
   probe conditions. Now reads "all six primary comparisons use probe-off
   RA, and the three secondary comparisons report probe-on minus
   probe-off, regardless of outcome," matching the primary/secondary
   family membership already clarified in supplement §A.8 (item 4
   above).

**Checks.** No table content changed (the four corrected phrases do not
appear in any auto-generated table), so `scripts/make_tables.py`'s output
is byte-identical; `test_make_tables.py`: 28/28 pass (unchanged).
`check_numbers.py`: all checks pass (unchanged pass/fail behavior; this
pass changed no numbers).

**Rebuild.** Main: validation build, 0 errors, 24 pages (unchanged), same
2 pre-existing overfull boxes (unchanged). Supplement: 0 errors, 15 pages
(unchanged), same 6 overfull boxes (unchanged), no undefined references or
citations in either document.

---

## Revision 7 — 2026-10-10/11 (matched RA-InfoOpt experiment)

Revision 6 is preserved unchanged at `/home/claude/InfoOpt_revision6`. This
revision adds a **new, fully controlled experiment** on top of the
revision-6 baseline; it does not alter any historical/retrospective number,
table, or figure, and does not alter the revision-5 four-arm experiment.
No training, tuning, new statistical analysis beyond independent
verification, retrospective selection of the better-performing RA-InfoOpt
arm, or replacement of earlier frozen results was performed.

1. **New input package.** `ITSL_RA_matched_analysis_20261010.zip`
   (`PROTOCOL.md`, `ANALYSIS_REPORT.md`, `per_seed_round100.csv`,
   `primary_summary.csv`, `primary_contrasts.json`, `probe_contrasts.json`,
   `verification_report.json`, `NUMERICAL_CHECKS.json`,
   `source_manifest.json`, `analyze_ra_matched.py`, `SHA256SUMS.txt`),
   supplied separately from the revision-6 manuscript archive.
   - All 10 checksummed files pass `SHA256SUMS.txt`.
   - Stored read-only and unchanged in `assets/ITSL_RA_matched_20261010/`.
   - Working copies live in the new `data_ra/` directory, each prefixed
     `ra_` and verified byte-identical to the frozen package by
     `scripts/make_tables.py::verify_ra_package()` on every build; a
     tampered working copy is rejected (regression-tested).
2. **Verified counts.** `per_seed_round100.csv`: 60 unique
   (dataset, seed, arm) records (3 datasets $\times$ 5 seeds $\times$ 4
   arms). `primary_summary.csv`: 12 rows (4 arms $\times$ 3 datasets).
   `primary_contrasts.json`: 6 primary comparisons (RA-InfoOpt with probes
   disabled minus FedAvg and minus InfoOpt-Full, across 3 datasets).
   `probe_contrasts.json`: 3 prespecified secondary comparisons
   (probe-on minus probe-off, across 3 datasets). Every mean, SD, 95% CI,
   raw $p$ and Holm $p$ in the supplied package was **independently
   recomputed from the per-seed CSV** (paired-$t$, Holm–Bonferroni) in
   `scripts/check_numbers.py` and `scripts/test_make_tables.py` and
   matches to $10^{-6}$; no discrepancy was found.
3. **New manuscript content.**
   - Main: new §5.5 "Matched RA-InfoOpt Experiment", new Table 6
     (`tables/table6_ra_matched.tex`, auto-generated). Abstract, §4.6
     (provenance, RA-InfoOpt configuration strata paragraph), §5.8
     (evaluation outcome), §6.1/6.3/6.4 (discussion, limitations, future
     directions), and §7 (conclusion) updated to cross-reference the new
     experiment without recovering, certifying, or substituting for the
     withheld historical RA-InfoOpt estimate.
   - Supplement: new §A.8 "Matched RA-InfoOpt Experiment" with new Table
     A.5 (`tables/tableA5_ra_probe.tex`), the full protocol and
     verification record, and an explicit scope paragraph distinguishing
     this new evidence from the unresolved historical RA-InfoOpt
     configuration-heterogeneity issue (§A.5) and the unresolved
     FedDyn/SCAFFOLD implementation-provenance flag (§A.6).
4. **Findings reported without overstatement.** None of the six primary
   contrasts survives Holm correction (smallest Holm-adjusted
   $p\approx0.189$). Two individual, unadjusted 95% CIs exclude zero
   (CIFAR-100 RA-InfoOpt$-$FedAvg: $+0.208$ pp; EuroSAT
   RA-InfoOpt$-$InfoOpt-Full: $+6.681$ pp), reported alongside their
   Holm-adjusted $p$-values rather than as significant or consistent
   results. None of the three probe-on/off comparisons survives Holm
   correction; all three individual intervals include zero. The
   probe-on/off comparison is described as instrumentation sensitivity
   (including an unisolated probe-model RNG initialization effect), not a
   comparison between two distinct aggregation mechanisms, and the probe
   subset is noted to overlap the training data and not to have been used
   to select configurations.
5. **Stale manifest label documented, not altered.** `source_manifest.json`
   carries a stale top-level experiment label, `four_arm_20261009`, left
   over from an earlier naming scheme. It is preserved unaltered in the
   frozen manifest and is documented in both the main text and supplement
   as a metadata issue, not used for run selection, and explicitly
   distinguished from the separately frozen revision-5 four-arm
   experiment.
6. **Checkpoint-verification scope disclosed.** Checkpoint strict-loading
   and tensor-finiteness checks for this experiment are stated to rely on
   the collected server verification reports, since model checkpoints
   were not themselves supplied to this analysis workspace.
7. **Docs updated.** `CLAIM_EVIDENCE_MAP.md` (new evidence source and
   claims for the matched RA-InfoOpt experiment), `UNRESOLVED_ISSUES.md`
   (RA-InfoOpt and FedDyn/SCAFFOLD items annotated to distinguish this new
   evidence from what remains unresolved), `CHANGELOG.md` (this entry),
   `CHECK_SUMMARY.md`, and `README.md` updated consistently.

**Checks.** `test_make_tables.py`: 28/28 pass (19 carried over + 9 new for
the RA-matched package loader/checksum/byte-identity/cell-uniqueness/
significance checks). `check_numbers.py`: extended with new sections 1c
(preservation of earlier evidence) and 1d (RA-matched independent
recomputation), new Table 6/A.5 content checks, new prose-quote checks,
and two new cross-document reference labels (`sec:ra_matched`=5.5,
`tab:ra_matched`=6 in main; `app:ra_matched`=A.8, `app:tab:ra_probe`=A.5 in
supplement); all checks pass, including with the build directory (no
undefined references/citations, no multiply-defined labels, no LaTeX
errors).

**Rebuild.** Main: validation build (stand-in `cas-dc`, as in revisions
4–6), 0 errors, 24 pages (was 22), same 2 pre-existing overfull boxes
(unchanged). Supplement: 0 errors, 15 pages (was 13), same 6 overfull
boxes (unchanged, all under 11pt), no undefined citations. The added text
and two new tables are responsible for the page-count growth in both
documents; no historical table, figure, or frozen numerical value changed,
and the revision-5 four-arm evidence is confirmed byte-identical to
revision 6.

---

## Revision 6 corrections — 2026-10-10 (narrow corrections, no numerical changes)

All frozen tables, contrasts, the four-arm analysis, and every figure's
plotted content from "Revision 6" below are unchanged. This pass corrects
one causal overstatement, one imprecise figure-identity claim, and tightens
a verification check; no new training, tuning, statistical analysis, or
change to any frozen input was performed.

1. **Causal "because" removed for the CIFAR-100 early-stop conflict.** Main
   §4.5 and supplement §A.2 no longer assert that the ten CIFAR-100 repair
   runs completed 100 rounds *because* the recovered early-stop controller
   gates only InfoOpt/RA-InfoOpt — that reasoning assumes the recovered
   controller code is what historically executed, which is exactly the
   unverified premise. Both now read: "All ten runs completed 100 rounds
   without recorded early stopping. In the recovered implementation, the
   controller applies only to InfoOpt/RA; exact historical
   executable-source identity remains unverified."
2. **Figure byte-identity claim corrected.** `docs/CHANGELOG.md`'s "Revision
   6" entry and `README.md` previously said "every figure" or "every figure
   value" is byte-for-byte unchanged from revision 5. `Figures/figure_fourarm.pdf`
   was in fact rebuilt by `scripts/plot_fourarm.py` while rerunning the
   check suite, and `cmp -l` shows it differs from revision 5's copy in
   exactly 5 bytes, all inside the PDF's `/CreationDate` metadata field.
   The companion `figure_fourarm.png` the same script writes is confirmed
   byte-identical to revision 5's, and no plotted value, label, or layout
   changed. `README.md`, `CHANGELOG.md`, and `CHECK_SUMMARY.md` now state
   this precisely instead of claiming blanket byte-identity.
3. **Superseded-value scanner made position-aware for the six repair-campaign
   tokens.** The previous fix whitelisted each of 29.10, 42.28, 74.84,
   85.10, 85.58, and 87.92 for an entire file once it appeared anywhere in
   that file's designated descriptive-audit passage. That whole-file
   exception would equally have let the same number reappear, unnoticed, in
   unrelated primary-result prose elsewhere in the same file. A new module,
   `scripts/span_scan.py`, confines each exception to the character span of
   its specific descriptive-audit passage (the new revision-6 paragraph in
   each document, plus the supplement's two pre-existing legitimate spots
   for 87.92: the "earlier version ... withdrawn claims" sentence and the
   Record of Revisions table row); an occurrence of any of these six tokens
   outside all of its file's allowed spans now fails the check instead of
   passing silently. A regression test in `test_make_tables.py` constructs
   a synthetic document with the same token both inside its passage and
   leaked into unrelated "primary results" prose, and confirms the leaked
   occurrence is caught while the legitimate one is not (3 new assertions).

**Checks.** `test_make_tables.py`: 19/19 pass (16 carried over + 3 new for
the span-aware scanner). `check_numbers.py`: 228/228 pass (unchanged count;
the scan's internal exception mechanism changed, not its pass/fail
behavior on the actual documents).

**Rebuild.** Main: validation build, 0 errors, 22 pages (unchanged), same 2
pre-existing overfull boxes (unchanged). Supplement: 0 errors, 13 pages
(unchanged), same 6 overfull boxes (unchanged), no undefined citations.

---

## Revision 6 — 2026-10-10 (FedDyn/SCAFFOLD repair-campaign audit)

Revision 5 is preserved unchanged at `/home/claude/InfoOpt_revision5`. This
revision incorporates a separately supplied audit package
(`ITSL_FedDyn_SCAFFOLD_audit_final_20261010.zip`) on the FedDyn/SCAFFOLD
implementation-provenance question. No training, tuning, new statistical
analysis, or change to any frozen input was performed. All frozen primary
tables, contrasts, the four-arm analysis, figures, and numerical values from
"Revision 5" below are unchanged.

1. **Repair-campaign run identification.** Main §4.5 and supplement §A.2 now
   report that 30 FedDyn/SCAFFOLD repair-campaign runs (15 per method, five
   seeds $\times$ three datasets) have been identified through exact
   saved-directory paths and matching per-round logs — independently
   re-verified in this session against the audit package's own
   `SHA256SUMS.txt` manifests (148/148 top-level entries, 9/9
   `synthetic_checks/` entries, both pass).
2. **17 CPU synthetic equation checks.** The audit's 17 passing CPU-only
   synthetic equation checks against the recovered SCAFFOLD/FedDyn
   implementation are reported as validating the *tested recovered
   implementation* (its own update rules on tiny synthetic inputs), not as
   certifying historical executed code or full experimental correctness. The
   check script's hash and the five code-snapshot hashes match the supplied
   package, independently confirmed in this session.
3. **No inference from poor earlier results alone.** Main §4.5 and
   supplement §A.2 no longer let poor historical performance alone imply a
   broken implementation. An older backed-up FedDyn implementation
   (`train.py.bak.lines`) has materially different update logic from the
   currently recovered `utils/feddyn.py`, but no recovered source hash
   conclusively links either version to any particular historical run;
   SCAFFOLD's earlier poor performance remains unexplained in either
   direction.
4. **Launch-time provenance record, precisely scoped.** The audit's
   launch-time provenance record (hash `a239b9eb...09e61`, independently
   re-verified byte-identical to its recovered copy in this session) is
   described as supporting repaired-code presence through file
   modification timestamps, file sizes, and targeted code-integrity checks
   — explicitly not a source hash or Git commit, so it cannot bind the
   launched code to one specific historical version.
5. **Frozen material untouched.** Table 3's FedDyn/SCAFFOLD cells, Tables
   4/5/A.1–A.4, and the four-arm analysis are byte-for-byte unchanged from
   revision 5. Six of the seven figures are byte-for-byte unchanged;
   `Figures/figure_fourarm.pdf` was rebuilt by `scripts/plot_fourarm.py`
   while rerunning the check suite and differs from revision 5's copy in
   exactly 5 bytes, all inside the PDF's `/CreationDate` metadata field
   (confirmed by `cmp -l`); its rendered content is otherwise unchanged,
   and the companion `Figures/figure_fourarm.png` the same script writes is
   byte-identical to revision 5's. No plotted value, label, or layout
   changed; no new performance table was added.
6. **CIFAR-100 early-stop eligibility conflict disclosed.** Main §4.5 and
   supplement §A.2 now disclose that 10 of the 30 repair-campaign runs (the
   5 CIFAR-100 SCAFFOLD and 5 CIFAR-100 FedDyn seeds) have
   `early_stop.enabled: true` in their saved configuration, conflicting with
   the strict configured-early-stop eligibility rule used for the frozen
   retrospective rule; these runs nonetheless completed all 100 rounds
   because the recovered early-stop controller code gates only InfoOpt and
   RA-InfoOpt. No eligibility amendment has been adopted; the conflict is
   disclosed, not resolved.
7. **Repair campaign labelled historical descriptive evidence; no new
   table.** The six repair-campaign descriptive means (reported in
   supplement §A.2, referenced in main §4.5) are explicitly labelled
   historical descriptive evidence, not primary results — not substituted
   into any frozen table, not pooled across implementation eras, and not
   used to rank or to claim superiority/inferiority for either method. A
   noteworthy (disclosed, unproven) finding: to two decimal places, all six
   of these means numerically reproduce the corresponding cell of the
   manuscript's original, since-withdrawn FedDyn/SCAFFOLD table row.
8. **Docs updated.** `CLAIM_EVIDENCE_MAP.md` (new evidence source **AU** and
   claim 23), `UNRESOLVED_ISSUES.md` (new item 0.1; item 2's FedDyn/SCAFFOLD
   sub-point annotated "partially addressed"), `CHANGELOG.md` (this entry),
   `CHECK_SUMMARY.md`, and `README.md` updated consistently, distinguishing
   resolved run identification from unresolved historical source-code
   identity throughout.

**Checks.** `scripts/check_numbers.py`'s superseded-value scan was extended
with six new `intentional` entries (the campaign means above, which
coincidentally match old withdrawn table values) and accurate per-token
rationale text; no check's pass/fail logic otherwise changed.
`make_tables.py` / `test_make_tables.py` (16/16) rerun unchanged (no new
tables). `check_numbers.py`: 228/228 pass.

**Rebuild.** Supplement: 0 errors, 13 pages (was 12), 6 overfull boxes (was
5; one new box, 6.3 pt, from the added §A.2 text — all boxes remain under 11
pt). Main: validation build, 0 errors, 22 pages (was 21), 2 overfull boxes
(both pre-existing, unchanged from revision 5). The added text is
responsible for the page-count growth in both documents; no table, figure,
or frozen numerical value changed.

---

## Revision 5 cleanup — 2026-10-10 (narrow corrections, no numerical changes)

All six primary contrasts, all three interactions, and every table/figure
value from "Revision 5" below are unchanged. This pass corrects wording,
labeling, and layout only.

1. **Archive-hash misattribution fixed.** Supplement §A.7 (and
   `CHECK_SUMMARY.md`) previously called `0a56c0080cb8da4…` "the analysis
   package's archive SHA-256." It is actually the hash of the **raw
   results collection** (406 files), as reported by the supplied
   package's own `verification_report.json` — not independently re-run
   here, since the raw run directories were not supplied. The text now
   distinguishes that collection-level check (reported by the package)
   from the **analysis package's** own integrity, which this session did
   independently re-verify against its `SHA256SUMS.txt` (8/8 pass). Main
   §5.4 text adjusted to match.
2. **"Frozen pre-registered protocol" → "protocol and analysis plan frozen
   before launch."** Supplement §A.7 intro no longer implies external
   registration; matches `PROTOCOL_BEFORE_LAUNCH.md`'s own framing.
3. **"Exploratory" removed from the interaction's labeling.** The
   aggregation×AIB interaction is now described consistently as a
   "prespecified secondary analysis" in Table A.4's caption
   (`scripts/make_tables.py`), `CLAIM_EVIDENCE_MAP.md`, and
   `UNRESOLVED_ISSUES.md`. (The unrelated historical "exploratory"
   analyses — the α-sweep, ablation, etc. — are untouched; that usage was
   already correct and predates revision 5.)
4. **CIFAR-100 interaction interpretation simplified.** Main §5.4, the
   Evaluation Outcome (§5.6), and the Conclusion now lead with "Enabling
   AIB reduces InfoOpt-Full's relative advantage over FedAvg by 0.72
   percentage points under this protocol," then keep the existing CI,
   Holm $p$, and within-setting-uncertainty caveat, instead of the
   original double-difference phrasing ("AIB's effect on InfoOpt-Full
   relative to FedAvg differs from AIB's effect on FedAvg relative to
   itself").
5. **Table 5(b) column count fixed.** `scripts/make_tables.py` declared
   `\begin{tabular}{llcccccc}` (8 columns) and `\multicolumn{8}{l}{...}`
   for a table with 7 columns (Dataset, AIB, $n$, $\Delta$, 95% CI,
   $p_\mathrm{raw}$, $p_\mathrm{Holm}$). Corrected to `{llccccc}` and
   `\multicolumn{7}`.
6. **Figure 5 re-cropped.** `scripts/plot_fourarm.py`'s in-image footer
   (a long single-line note on significance marker, Holm family sizes,
   partition identity, and scope) forced `bbox_inches="tight"` to a
   ~1208$\times$266 pt canvas with excessive right-hand whitespace. The
   footer is removed from the image; its content now lives in the
   `fig:fourarm` LaTeX caption in `main_revised.tex`. Regenerated figure
   is 631$\times$245 pt, tightly cropped, labels unchanged and readable.
7. **`CHECK_SUMMARY.md` "Secondary family" entry qualified.** The Review
   Notes row "Secondary family | Decided: not reported" referred only to
   the historical 21-comparison external-baseline family (§5.5); it now
   says so explicitly and cross-references the new three-test
   aggregation×AIB secondary analysis (Table A.4), which **is** reported.

**Rebuild.** `make_tables.py` / `test_make_tables.py` (16/16) /
`plot_fourarm.py` rerun; both documents rebuilt. Supplement: 0 errors, 12
pages, 5 overfull boxes (all pre-existing, same as revision 4 baseline —
two drafts of the §A.7 rewrite introduced and then removed extra overfull
boxes before this count). Main: validation build, 0 errors, 21 pages, 2
overfull boxes (both pre-existing). `check_numbers.py`: 228/228 pass
(unchanged from "Revision 5" below — this pass changed no numbers).

---

## Revision 5 — 2026-10-10 (four-arm controlled experiment)

Revision 4 is preserved unchanged at `/home/claude/InfoOpt_revised`. This
revision adds a **new, fully controlled experiment** on top of the
revision-4 baseline; it does not alter any revision-4 historical/retrospective
number, table, or figure.

- **New input package.** `ITSL_four_arm_analysis_20261010.zip`
  (`ANALYSIS_REPORT.md`, `PROTOCOL_BEFORE_LAUNCH.md`, `primary_summary.csv`,
  `primary_contrasts.json`, `aib_interactions.json`, `per_seed_round100.csv`,
  `verification_report.json`, `analyze_fourarm.py`, `SHA256SUMS.txt`), supplied
  separately from the revision-4 manuscript archive.
  - All 8 checksummed files pass `SHA256SUMS.txt`.
  - Stored read-only and unchanged in `assets/ITSL_four_arm_20261010/`.
  - Working copies live in the new `data_fourarm/` directory, each prefixed
    `fourarm_` and verified byte-identical to the frozen package by
    `scripts/make_tables.py::verify_fourarm_package()` on every build; a
    tampered working copy is rejected (regression-tested).
- **Verified counts.**
  - `fourarm_primary_summary.csv`: **12 rows** (4 arms $\times$ 3 datasets): `fedavg_aiboff`, `fedavg_aibon`, `infoopt_aiboff`, `infoopt_aibon`.
  - `fourarm_primary_contrasts.json`: **6** within-AIB InfoOpt-Full$-$FedAvg contrasts (AIB off/on $\times$ 3 datasets).
  - `fourarm_aib_interactions.json`: **3** aggregation$\times$AIB interaction contrasts (1 per dataset).
  - `fourarm_per_seed_round100.csv`: 60 rows (4 arms $\times$ 3 datasets $\times$ 5 seeds), matching `verification_report.json` (406/406 collection files, 60/60 runs verified).
  - Every mean, SD, 95% CI, raw $p$ and Holm $p$ in the supplied package was **independently recomputed from the per-seed CSV** (paired-$t$, Holm–Bonferroni) in `scripts/check_numbers.py` and matches to $10^{-9}$.
- **New manuscript content.**
  - Main: new §5.4 "Four-Arm Controlled Experiment: Isolating Aggregation from AIB", new Table 5 (`tables/table5_fourarm.tex`, auto-generated), new Fig. 5 (`Figures/figure_fourarm.pdf`, rendered by new `scripts/plot_fourarm.py`, self-checked against source). Abstract, §4.5 (provenance), §5.6/5.7 (evaluation outcome), §6.1/6.3/6.4 (discussion, limitations, future directions) and §7 (conclusion) updated to cross-reference the new experiment without revising the qualitative outcome of the retrospective reanalysis.
  - Supplement: new §A.7 "Four-Arm Controlled Experiment" with new Table A.4 (`tables/tableA4_fourarm_interaction.tex`), the full verification record, and an explicit statement that this subsection (unlike the rest of Appendix A) reports new training, not a retrospective reanalysis. Interpretive-scope (§F) and revision-record (§G) tables updated; figure-disposition table records `figure_fourarm.pdf`.
- **Result.** All six primary within-AIB contrasts have 95% CIs including zero (Holm $p\ge0.564$); the four-arm experiment does not establish a consistent InfoOpt-Full advantage over FedAvg, consistent with revision 4's retrospective finding. The CIFAR-100 aggregation$\times$AIB interaction is Holm-significant ($-0.720$ pp, Holm $p=0.0202$, $m=3$); the other two interactions are not. The experiment resolves the AIB architecture confound for FedAvg vs. InfoOpt-Full only; it does not resolve RA-InfoOpt configuration heterogeneity or FedDyn/SCAFFOLD provenance.
- **Builds.** Supplement: real build, 0 errors, 12 pages. Main: validation build (stand-in `cas-dc`, as in revision 4), 0 errors, 21 pages (was 19 in revision 4). No undefined references/citations, no multiply-defined labels. Overfull boxes: main 2 (both pre-existing, content unchanged), supplement 5 (all pre-existing).
- **Tests.** `test_make_tables.py`: 16/16 pass (11 carried over + 5 new for the four-arm loader/checksum/byte-identity/interaction-significance checks). `check_numbers.py`: 228/228 pass (138 carried over + 90 new: per-seed-recomputed means/SDs/contrasts/interactions/Holm, table contents, quoted prose, and new cross-document reference labels).

---

## Revision 4 — 2026-10-09 (original FedOpt sensitivity output)

No training, tuning, new run selection or manuscript number changes.

- **Package preserved.**
  - `ITSL_reanalysis_sensitivity_20261008_v2.zip` is stored unchanged in `assets/`.
  - Its archive SHA-256 (`ba5f20e4…a96`) matches the "Sensitivity v2 package" hash recorded in the frozen figure package README.
  - The extracted contents sit read-only in `assets/ITSL_reanalysis_sensitivity_v2/`; all 11 entries pass the package's own `SHA256SUMS.txt`.
  - The package's `ra_strata_summaries.json` and `sensitivity_latest_repeat.json` are byte-identical to the files already in `data/`.
- **Transcribed value removed.** `data/sensitivity_cifar_fedopt.json` (transcribed in revision 3) is deleted.
- **Importer rewritten.** `load_excluded_cell_sensitivity()` in `make_tables.py` now reads the original `outputs/sensitivity_cifar_fedopt.json` in its own schema: `label`, `n`, `mean`, `sd`, and `runs[seed, run, round100_acc_pct]`. It:
  - verifies the output file and the frozen inventory against the package checksums
  - takes the cell identity (cifar100 / fedopt / `cifar_fedopt_sensitivity_only`) from the frozen inventory, matched by run name, and checks the seeds
  - requires the label to declare "SENSITIVITY ONLY"
  - recomputes the mean and SD from the five per-seed values
- **Value unchanged.** The original output gives 44.724 ± 2.1531 %, n = 5, identical to the transcribed value. Table 3 and the manuscript text therefore keep 44.72 ± 2.15 (5), and the primary cell stays excluded.
- **Tests.** `test_make_tables.py` now has 11 tests, including reading the original file and rejecting a tampered copy. `check_numbers.py` loads the value through the same verified importer.
- **Format postponed.** The official `cas-dc` build is postponed at the author's request; see `UNRESOLVED_ISSUES.md`.

---

## Revision 3 — 2026-10-08 (author corrections)

Made without training, tuning or new run selection.

1. **CIFAR-100 FedOpt sensitivity.**
   - Added `data/sensitivity_cifar_fedopt.json` with the value 44.724 ± 2.153 %, n = 5, role `sensitivity_only`.
   - The value is **transcribed from the author's message**. The cited source, `outputs/sensitivity_cifar_fedopt.json` in `ITSL_reanalysis_sensitivity_20261008_v2.zip`, was not attached to this session; the file's `provenance` field says so.
   - `make_tables.py` loads it into a third container (`ExcludedCellSensitivity`). That container can only reach the Table 3 footnote, and it raises an error if a value targets a non-excluded cell or is duplicated.
   - The primary cell stays "excluded".
   - Text updated in: §4.5 (early stopping), §5.1, §6.3 sensitivity scope, and supplement A.2.
   - New tests: the real value appears only in the footnote; duplicate records raise.
2. **RA stratum e9827b.** The "unexplained observation" note is replaced. The seed-103 and seed-104 runs reach the same terminal accuracy (43.20 %), but they have distinct run directories and different trajectories (author-verified), so this is not a duplicated run.
3. **Retrospective timing.** "frozen before the reported summaries were computed" is replaced in §4.5, §6.3 and supplement A.6 by: *The protocol and source manifest were frozen for retrospective recomputation after exploratory provenance inspection; this was not historical preregistration.*
4. **Bibliography style.** The supplied official `elsarticle-num.bst` was added at the project root. Both documents now use it; the supplement previously used `unsrtnat`.
5. **Repeat-sensitivity figure.**
   - `Figures/figure_repeat_sensitivity.pdf` was re-rendered by `scripts/replot_repeat_sensitivity.py`, using the same frozen source data, marks, axes, title and footnote.
   - Only the value labels moved: they are now centred above their points, so the PathMNIST "−0.164 pp" label no longer overlaps the axis.
   - The script asserts that the six plotted values equal the source.
   - The frozen package in `assets/` is unchanged (SHA256SUMS still 14/14 OK). Supplement Table G.2 records the re-render.

---

## Revision 2 — 2026-10-08 (completion with verified outputs)

**Inputs:**
- `primary_summary.csv`
- `paired_contrasts.json`
- the verified figure package `ITSL_verified_figures_20261008.zip` (SHA256SUMS verified, 14/14 OK)
- `figure1_framework.pdf` and `figure2_evaluation_framework.pdf`
- the original `Supplementary_Material.tex`

No training, tuning, new run selection or protocol change.

### Project and preservation
- The original supplement was added to `original/` (read-only).
- The figure package is stored unchanged and read-only in `assets/ITSL_verified_figures/`.
- `data/` and the four new `Figures/*.pdf` are byte-identical copies of the package files.
- The validation shim was **removed from the project**. The main manuscript uses only `\documentclass{cas-dc}`; see `UNRESOLVED_ISSUES.md` about the template files.
- The obsolete placeholder `data/primary_contrasts.csv` was removed. Table 4 is now generated from `paired_contrasts.json`.

### `scripts/make_tables.py` (rewritten)
- **Separation fix.** Primary records (`PrimaryRecord`, from `primary_summary.csv`) and sensitivity records (`SensitivityRecord`, from `sensitivity_latest_repeat.json`) are kept in separate containers and never merged.
- **Cell status rules.** A primary cell's status comes only from the primary record plus explicit rules:
  - RA-InfoOpt → withheld
  - CIFAR-100 FedOpt → excluded
  - FedDyn and SCAFFOLD → flagged
- **Fail loudly on conflicts.** A rule–data conflict raises an error, e.g. an excluded cell that has a primary value. Duplicate primary keys also raise.
- **Sensitivity values.** For an excluded cell, a sensitivity value is shown only in the footnote. When none exists, the footnote says so.
- **Regression test.** `scripts/test_make_tables.py` checks that an injected sensitivity row cannot overwrite the excluded primary cell.
- **New outputs:**
  - Table 3 from verified summaries, as mean ± sample SD with n.
  - Table 4 with a d_z column, computed from the verified mean and SD of the paired differences.
  - Supplementary Tables A.1–A.3: primary family, repeat sensitivity, RA strata. Each is validated against its source; strata means and SDs are recomputed.

### Main manuscript
- **Table 3.** Populated with the verified summaries. RA-InfoOpt is "withheld" and CIFAR-100 FedOpt is "excluded". The footnote states that no FedOpt sensitivity value exists in the supplied outputs. FedDyn and SCAFFOLD carry a dagger; there is no ranking and no bold maximum.
- **Figures.** All four verified figures are inserted, plus the two schematics. Placeholder macros are removed.
  - Captions are rewritten from the package README and checked against figure content: the "Excluded: strict E3" label, the orange squares and dagger, the Holm annotation, the non-estimable rows, and the separate y-scales and seed coverage of the strata panels.
  - Fig. 4 is now full width for legibility.
- **Wording.** "without new training, tuning, or run selection" is replaced everywhere by "without new training or tuning; runs selected under the frozen retrospective rule". This covers the abstract, introduction, §4.5 and the conclusion.
  - A new §4.5 paragraph defines the rule: strict configured-early-stop eligibility, then the earliest eligible run per seed within a configuration stratum. It states that the rule was adopted for the reanalysis and is not a historical preregistration.
- **Verdict.**
  - The NO-GO verdict is replaced by **"consistent superiority not established"**. The Fig. 2 categories are qualitative and state no operational decision rule. Several framework dimensions rely on unverified analyses, and the framework's integrity conditions (fixed configurations) are not met for RA-InfoOpt.
  - §5.6 is renamed "Evaluation Outcome". RA-InfoOpt is "not assessable" and GRASP "not assessed".
- **Fig. 2 caption and §3.5 text.** The framework is described as part of the original design, not a formal preregistration. The caption notes that the figure's "predefined" and "fixed configurations" labels describe intent.
- **Sensitivity (§5.3, §4.5, limitations).** Repeat sensitivity is now defined from the outputs as latest minus earliest. The sensitivity contrasts are reported:
  - EuroSAT −4.770 [−10.231, +0.690], p = 0.072
  - PathMNIST −0.072 [−2.977, +2.832], p = 0.948
  - The PathMNIST sign change is disclosed.
- **RA strata (§4.5).** Three strata per dataset (seeds 100–102, 103–104, and an extra seed-100 run); no stratum covers all five seeds.
- **Early stopping.** The false statement that the CIFAR-100 FedOpt value "is reported separately as a sensitivity result" is corrected: no such value exists.
- **Review notes.** All six main-text `[Review: …]` notes are resolved individually, as listed in `CHECK_SUMMARY.md`.
- A `\FloatBarrier` before the Discussion keeps Tables and Figures 3–4 within Results.

### Supplement (rebuilt from the original)
- **Restored verbatim** (verified by string match):
  - the original preamble (plus `xurl`)
  - the GRASP full derivation (D.2.1)
  - the solver settings and protocol (D.2.2)
  - the damping definition (D.4)
  - Table E.1
- **Edits within restored blocks:**
  - "diagnostics reported below" → "diagnostics of the earlier supplement", because those results are no longer reported.
  - D.2.2 heading: "and Empirical Summary" → "and Protocol".
  - Table E.1: three header column widths increased, to fix overlapping header text that was present in the original.
  - Table E.1: an AIB note added after the table.
- **Empirical claims audited individually** (Table G.1):
  - A.1, A.2, A.3 and B.1: superseded.
  - B.2, B.3, C.1, D.1, the GRASP empirical summary, D.3 and D.5: not restored, for lack of supplied output.
  - D.4 damping values: restated from verified Table 3, with the oracle claims not restored.
- **New content:**
  - A.4 and A.5: sensitivity and strata, with Tables A.2–A.3 and Figs. A.1–A.2.
  - A.6: provenance summary.
  - Table D.1, rewritten as a diagnostic-status table.
  - Section G: claim audit and figure disposition.
- **Bibliography.** Added `\bibliography{references}` with `unsrtnat`. The original supplement cited 21 keys in Table E.1 without a bibliography, so every citation there was undefined.

---

## Revision 1 — 2026-10-08 (initial revision, before outputs were supplied)

Revision date: 2026-10-08. Basis: `original/main_elsevier_ORIGINAL.tex` (SHA-256 `b312db09…`), `original/References_ORIGINAL.bib` (SHA-256 `5dc2e3ec…`), and the instruction note. No training, tuning, new run selection or protocol change.

**Inputs not supplied to this revision:**
- retrospective reanalysis outputs
- repair documents
- claim-to-evidence map
- original supplementary source
- all figure PDFs, both the four new ones and the original `Figures/` folder

Every edit below uses only the values and disclosures stated in the instruction note.

## Front matter
- **Abstract** — rewritten:
  - States the retrospective scope.
  - Reports the three InfoOpt-Full − FedAvg contrasts, with all CIs including zero and Holm p ≥ 0.346 (m = 6).
  - Withholds RA-InfoOpt.
  - Adds the interpretation caveat (uncertainty ≠ equivalence or ineffectiveness).
  - Discloses the AIB mismatch, the partial pairing evidence and the unrecovered provenance.
  - Removed: "no strategy consistently outperforms", the RA-InfoOpt "highest mean on PathMNIST", the "strongest method changes" claim, and the diagnostic conclusions.
- **Introduction** — the results paragraph is replaced with the central conclusion and the provenance and AIB caveats. GRASP and the other diagnostics are now stated as not validated by the reanalysis.
- **Contributions:**
  - Bullet 2 (RA-InfoOpt): its evaluation is now stated as withheld.
  - Bullet 3: rewritten as a provenance-audited retrospective reanalysis. The removed claim is that results "show that neither richer evidence nor sophisticated rules guarantee improved performance".

## Methodology (definitions and equations unchanged)
- **§3.1** — the sentence saying the comparison "isolates changes in the aggregation rule" is qualified: by design it would, but in practice AIB is present in the adaptive arms only.
- **§3.3 (RA coefficients)** — "held unchanged … throughout the canonical confirmatory evaluation" becomes "intended to be held unchanged". It now notes that the recovered RA runs fall into configuration strata that differ across seeds.
- **§3.3 (closing sentence)** — RA's benefit is now stated as not assessable from the recovered runs.
- **§3.4 GRASP** — "180-cell canonical comparison" becomes "the method comparison or the retrospective reanalysis".
- **§3.5 Backbone and AIB:**
  - Removed the false "shared backbone containing AIB / part of the common architecture" statement.
  - Added that AIB is present only in InfoOpt-Full and RA-InfoOpt, and that the other ten arms lack it.
  - The AIB equation is unchanged.
- **§3.5 hyperparameters, and the Table 1 caption** — values are relabelled as *intended* frozen settings, with the RA strata caveat. All values are unchanged.
- **§3.5 evaluation framework** — "use fixed configurations, matched seeds" becomes "designed to use…", with a reference to §4.5.
- **Figures 1–2** — wrapped in a placeholder-safe include, because the files were not supplied.

## Experimental setup
- **§4.1:**
  - Architecture-mismatch disclosure added.
  - Partition and seed matching restated as protocol intent, with verification deferred to §4.5.
  - "No early stopping" replaced: historical early-stopping configuration is not uniformly recoverable, and retrospective eligibility follows the strict configured-early-stop rule.
- **§4.2 and Table 2:**
  - Notes that no external baseline uses AIB.
  - FedDyn and SCAFFOLD carry the § flag (audit observation; excluded from rankings).
- **§4.3 diagnostics** — now states they are outside the reanalysis and are not used to support claims. The descriptions are kept.
- **§4.4:**
  - Retrospective summaries are mean ± sample SD with the actual n per cell, so n is no longer assumed to be 5.
  - The Holm family is preserved at m = 6, with the not-estimable RA contrasts counted.
  - The "five paired observations" wording is generalised.
- **New §4.5, "Retrospective Reanalysis and Provenance Constraints"**, covering:
  - historical vs retrospective tables (code and ledger unrecovered; four printed means unexplained; value matches are not certification)
  - early-stopping eligibility
  - pairing evidence (165 / 135 / 15; partition membership assumed)
  - the AIB mismatch
  - the RA strata
  - the FedDyn/SCAFFOLD flag (divergence does not prove broken code or version drift)
  - repeat-sensitivity scope (≈0.164 pp, six FedAvg/InfoOpt-Full summaries only)
- **Removed:** the "Experimental integrity" paragraph (180 cells, one valid execution per cell). The reanalysis contradicts it.

## Results (rewritten)
- **§5.1** — Table 3 is now "Retrospective reanalysis", generated by `scripts/make_tables.py`:
  - Mean ± sample SD with n per cell.
  - RA-InfoOpt withheld.
  - CIFAR-100 FedOpt excluded from the primary summary, with a sensitivity footnote.
  - FedDyn and SCAFFOLD flagged with a dagger.
  - No bold maximum.
  - **Placeholder** until `data/retrospective_summary.csv` is supplied.
- **§5.1 text** — the per-dataset "best method" statements are withdrawn and no ranking is made.
- **§5.2** — Table 4 is regenerated from `data/primary_contrasts.csv`:
  - Three InfoOpt-Full contrasts using the supplied values.
  - Three RA contrasts marked "not estimable".
  - The d_z column is removed, because d_z was not among the supplied values. A derived d_z appears only in Supplementary Table S1, labelled as derived.
- **§5.3 (new)** — sensitivity analyses: repeat-selection, CIFAR-100 FedOpt, RA strata.
- **§5.4** — external-baseline and damping results are withdrawn pending verified outputs. Removed:
  - FedOpt leads on CIFAR-100
  - FedDyn's 87.92 % on EuroSAT and its raw p = 0.048
  - SCAFFOLD as an outlier
- **§5.5** — the convergence, α-sweep, ablation and diagnostic subsections are replaced by a statement that these analyses are not re-verified, with the reason for omitting each.
- **§5.6 verdicts:**
  - InfoOpt-Full: NO-GO limited to the claim "general replacement for FedAvg", scoped to the arm as implemented (with AIB) and explicitly not a finding of ineffectiveness.
  - RA-InfoOpt: CONDITIONAL GO withdrawn; no verdict assigned.
  - GRASP: no verdict assigned.

## Discussion, limitations and conclusion
- **§6.1** — rewritten around uncertainty. The regime-dependence claims that rested on the historical rankings, diagnostics and ablations were removed.
- **§6.2** — diagnostic conclusions withdrawn.
- **§6.3 Limitations** — reorganised to cover: provenance, AIB, pairing, configuration strata, early stopping, FedDyn/SCAFFOLD, sensitivity scope, power and scope, tuning asymmetry (retained), and unverified analyses.
- **§6.4 Future directions** — now leads with an architecture-matched, provenance-logged comparison.
- **§7 Conclusion** — rewritten to the central conclusion. The removed claim is that no tested quantity "provided a universal criterion".
- **Supplementary Material paragraph** — updated.
- **Data availability, CRediT, funding and AI declaration** — unchanged in substance.

## Figures
| Original | Action |
|---|---|
| `fig2_main_comparison.pdf` | Omitted. It conflicts with retrospective Table 3 and is replaced by `figure_accuracy.pdf` (main, Fig. 3). |
| `fig3_gain_relative_fedavg.pdf` | Omitted. It conflicts with Table 4 and is replaced by `figure_paired_contrasts.pdf` (main, Fig. 4). |
| `fig6_convergence_trajectories.pdf` | Omitted. Its source data and selection rules are unavailable, and it contains a withheld RA series. |
| `fig4_alpha_sensitivity.pdf` | Omitted. Its source data and selection rules are unavailable. |
| `figure1_framework.pdf`, `figure2_evaluation_framework.pdf` | Retained, since they are conceptual. The files must be restored. |
| New: `figure_repeat_sensitivity.pdf`, `figure_ra_strata.pdf` | Supplement, Figs. S1–S2. |

All new captions are drafted from the stated caveats, because the supplied captions were not available. Each carries a visible review flag.

## Supplement (new skeleton; original not supplied)
- **A** — statistical details:
  - Table S1, with the derived t and d_z clearly marked as derived.
  - Holm arithmetic shown for m = 6 against m = 3.
  - A gap note for the secondary family.
  - The provenance list.
  - Figs. S1–S2.
- **B–E** — explicit source-gap boxes. The GRASP derivation and Table E.1 must be restored from the original supplement; they are not reconstructed, so that the mathematical definitions are not altered.
- **Table S2** — the disposition record for every figure.

## References
- `references.bib` is byte-identical to the original.
- The set of cited keys is identical to the original (34 keys).
- No citation was moved off its supporting claim, and none was added.
