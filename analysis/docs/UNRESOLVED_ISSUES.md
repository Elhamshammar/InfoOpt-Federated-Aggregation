# Unresolved issues (revision 8)

Revisions 4–7 are preserved unchanged at `/home/claude/InfoOpt_revised`,
`/home/claude/InfoOpt_revision5`, `/home/claude/InfoOpt_revision6`, and
`/home/claude/InfoOpt_revision7`. This file tracks
`/home/claude/InfoOpt_revision8`. All revision-7 issues below carry over
except where marked resolved; the new saved-checkpoint step-size pilot
(Section 6.2; supplement §A.9) adds a new, independent, bounded piece of
evidence (item 0.3 below) and resolves nothing listed below — it is a
separate mechanism investigation on one dataset, one partition, and does
not bear on RA-InfoOpt, FedDyn/SCAFFOLD, or the AIB architecture mismatch.

0. **Four-arm experiment: scope of what is resolved (new in revision 5).**
   - The four-arm controlled experiment isolates the AIB architecture
     confound **only for FedAvg versus InfoOpt-Full**, on three datasets,
     five seeds, at round 100. It directly answers revision 4's own Future
     Directions call for an architecture-matched, partition-verified
     comparison, for this one pair of arms.
   - It does **not** cover RA-InfoOpt (not part of the four-arm grid), so
     RA-InfoOpt's configuration-stratum problem (item 2 below) is unchanged.
   - It does **not** cover FedDyn or SCAFFOLD, so their implementation-
     provenance flag (item 2 below) is unchanged.
   - Enabling AIB reduces InfoOpt-Full's relative advantage over FedAvg by
     0.72 percentage points on CIFAR-100 under this protocol ($[-1.108,
     -0.332]$, Holm $p=0.0202$, $m=3$). This is a prespecified secondary
     analysis (its own Holm family, separate from the six-contrast primary
     family) and does not by itself establish that InfoOpt-Full is worse
     than FedAvg within either AIB setting on CIFAR-100 — both of those
     primary contrasts individually have intervals including zero. A
     larger seed count would narrow all of these intervals; whether the
     CIFAR-100 interaction persists at a larger $n$ is not assessed here.
   - Checkpoints from the four-arm runs remained on the server; their hashes
     and successful load verification are recorded, but checkpoint loading
     itself was not independently repeated in this session.

0.1. **FedDyn/SCAFFOLD repair-campaign audit: scope of what is resolved
   (new in revision 6).**
   - **Resolved: run identification.** 30 FedDyn/SCAFFOLD repair-campaign
     runs (15 per method, five seeds $\times$ three datasets) are
     identified by exact saved-directory path and a matching per-round
     log, independently re-verified against the supplied package's
     SHA-256 manifests (148/148 and 9/9 entries) in this session.
   - **Resolved: the tested recovered implementation passes its own
     equation checks.** 17/17 CPU-only synthetic equation checks pass
     against the recovered SCAFFOLD/FedDyn implementation; the check
     script's and snapshot hashes match the supplied package,
     independently re-verified in this session. This validates the
     tested implementation's own update rules on tiny synthetic CPU
     inputs; it does **not** certify that this is the code historically
     executed for any specific run, and it does not cover GPU, batch-norm,
     or full-training-orchestration behaviour.
   - **Not resolved: historical source-code identity.** No recovered
     source hash binds either the older backed-up FedDyn implementation
     or the current recovered implementation to any particular historical
     run. The launch-time provenance record (independently re-verified,
     hash `a239b9eb...09e61`) supports repaired-code presence at launch
     time through timestamps, file sizes, and targeted integrity checks,
     but contains no source hash and no Git commit. SCAFFOLD's own
     earlier poor performance remains unexplained by this material.
   - **Not resolved, disclosed as an open conflict: CIFAR-100 early-stop
     eligibility.** Ten of the 30 repair-campaign runs (5 SCAFFOLD + 5
     FedDyn, CIFAR-100) have `early_stop.enabled: true` in their saved
     configuration, conflicting with the strict configured-early-stop
     eligibility rule (item 9 below / Section 4.5); all ten nonetheless
     completed 100 rounds without recorded early stopping. In the
     recovered implementation, the controller applies only to InfoOpt and
     RA-InfoOpt; exact historical executable-source identity for these ten
     runs remains unverified. No eligibility amendment has been adopted to
     admit these runs under the frozen rule.
   - **Noted, not explained:** to two decimal places, all six
     repair-campaign descriptive means numerically reproduce the original,
     since-withdrawn FedDyn/SCAFFOLD table row. This is consistent with,
     but does not prove, that these runs produced that original printout;
     no source hash makes that link.
   - **Scope.** This audit does not substitute into any frozen table
     (Table 3/4/5/A.1–A.4), is not pooled across implementation eras, and
     is not used to rank FedDyn or SCAFFOLD or to support a superiority or
     inferiority claim for either method. It does not extend to
     RA-InfoOpt or to the four-arm experiment (item 0 above).

0.2. **Matched RA-InfoOpt experiment: scope of what is resolved (new in
   revision 7).**
   - **Resolved: a single, consistently configured RA-InfoOpt comparison is
     now possible.** The historical RA-InfoOpt runs fall into three
     configuration strata per dataset (item 2's RA-InfoOpt sub-point;
     Section 4.6), so no combined five-seed estimate could be formed from
     them. The new matched experiment fixes a single RA-InfoOpt
     configuration across all five seeds (100–104), on verified shared
     partitions and cohort schedules, with FedAvg and InfoOpt-Full run
     under the same matched settings for comparison.
   - **Not resolved: no advantage is established, and the historical
     estimate is not recovered or certified.** None of the six primary
     contrasts (RA-InfoOpt with probes disabled minus FedAvg and minus
     InfoOpt-Full, per dataset) survives Holm correction across this family
     of six (smallest Holm-adjusted $p\approx0.189$). Two individual,
     unadjusted 95% intervals exclude zero — CIFAR-100 RA$-$FedAvg
     ($+0.208$ pp) and EuroSAT RA$-$InfoOpt-Full ($+6.681$ pp) — but neither
     is a significant result at the family level, and this is not a
     consistent or multiplicity-adjusted RA-InfoOpt advantage. This new,
     prospective evidence does not recover, certify, or substitute for the
     historical, configuration-heterogeneous five-seed estimate, which
     remains withheld.
   - **Not resolved: instrumentation-sensitivity scope of the probe
     comparison.** A separate, prespecified secondary family of three
     probe-on$-$probe-off contrasts (one per dataset, Holm $m=3$) shows no
     interval excluding zero and none surviving Holm correction. This
     measures instrumentation sensitivity to enabling the counterfactual
     probe evaluation, including an unisolated probe-model initialization
     effect on the random-number-generator stream in the existing
     implementation; it is not a comparison between two distinct
     aggregation mechanisms and does not isolate a pure probe-evaluation
     effect. The probe subset overlaps the training data and was not used
     to select configurations or endpoints.
   - **Not resolved: FedDyn/SCAFFOLD.** This experiment does not include
     FedDyn or SCAFFOLD; their implementation-provenance flag (item 0.1
     above) is unchanged.
   - **Checkpoint-verification scope.** Checkpoints from these 60 runs
     remained on the server; checkpoint strict-loading and tensor-finiteness
     checks rely on the collected server verification reports and were not
     independently repeated in this session, since checkpoints were not
     themselves supplied to this analysis workspace.
   - **Metadata issue, disclosed, not corrected.** The raw execution
     manifest (`source_manifest.json`) carries a stale top-level experiment
     label, `four_arm_20261009`, left over from an earlier naming scheme.
     It is preserved unaltered in the frozen manifest, documented in-text as
     a metadata issue, and was not used for run selection; it must not be
     read as identifying this experiment with the distinct, separately
     frozen revision-5 four-arm package (item 0 above).

0.3. **Saved-checkpoint CIFAR-100 step-size pilot: scope of what is
   resolved (new in revision 8).**
   - **Resolved: nothing outside this pilot.** This is a new, independent
     mechanism investigation (one dataset, one partition at seed 42, one
     development seed, two held-out seeds, one-step saved-checkpoint
     evaluations). It does not touch, and is not pooled with, the
     retrospective reanalysis, the four-arm experiment (item 0), the
     matched RA-InfoOpt experiment (item 0.2), or the FedDyn/SCAFFOLD
     audit (item 0.1).
   - **Not resolved: whether adaptive step-size control ever offers
     headroom.** Validation-selected per-source scaling fails the frozen
     headroom gate against both development-tuned fixed controls (global
     and phase-specific) on at least one held-out seed; this is a measured
     negative for this one bounded pilot, not a universal claim about
     step-size control.
   - **Not resolved: the tested shrinkage rule's status.** Claude's raw
     shrinkage rule was worse than both fixed controls on every checkpoint
     of both held-out seeds; failure of this one comparator does not by
     itself establish that no adaptive step-size rule could help.
   - **Not resolved (exploratory only): update-statistics signal.** The
     corrected phase-plus-variability predictor does not improve held-out
     prediction over phase-only indicators in this small pilot (27
     development cells); this is specific to this feature set and sample
     size, not a general finding about whether any update-statistics
     signal carries information.
   - **Not resolved, disclosed: export defect and partial correction.** The
     original per-example CSV's validation-selected-scaling rows were
     silently defaulted to $\gamma=1$ for 77 of 90 sources after an
     internal cache was cleared post-use. The retained grid/rule/decision
     summaries behind every finding above are unaffected (computed
     upstream of the defect). Only 9 of the 90 affected sources could be
     regenerated from saved checkpoints without retraining; the remaining
     81 cannot be recovered. No sample-level interval or bootstrap result
     is computed from either the defective rows or the partial corrected
     subset, and none is planned from this material.
   - **Not resolved, disclosed: amendment byte-identity.** The supplied
     `PRE_EXECUTION_AMENDMENT.md` is a later change record whose bytes do
     not match the historical frozen hash; it is not certified as the
     original pre-execution payload. The protocol and decision-logic
     content it documents is corroborated independently via
     `PILOT_CLOSURE.md` and the checksum-verified `pilot_decision.json`,
     so this byte-identity gap does not by itself cast doubt on the
     reported findings, but it is disclosed rather than silently resolved.
   - **Frozen-buffer grid not measured.** The buffer-policy analysis
     evaluates freezing checkpoint buffers only at $\gamma=1$; a
     frozen-buffer scale grid was never launched, so parameter-scaling and
     buffer-scaling effects remain incompletely separated in this pilot.

1. **Official `cas-dc` build — postponed at the author's request.**
   - The supplied template files are `elsarticle.cls` (v1.21, 2010), `elsarticle-num.bst` and `SDlogo-3p.eps`.
   - The manuscript uses the CAS class (`\documentclass[a4paper,fleqn]{cas-dc}`), which needs `cas-dc.cls` and `cas-common.sty`; `elsarticle.cls` is a different class.
   - The class was not changed. `elsarticle.cls` and the logo are therefore not used and are not included in the project.
   - The official `elsarticle-num.bst` **is** now used by both documents.
   - The delivered main PDF is still a validation build: correct bibliography style, stand-in document class.
   - Final build: compile on Overleaf or locally with the Elsevier CAS template (`cas-dc.cls` and `cas-common.sty` next to `main_revised.tex`).
   - Then inspect the final PDFs for:
     - layout and float placement in two columns
     - equations (especially the heterogeneity normalisation)
     - figure legibility at column and page width
     - the reference list
   - Finally, rerun `scripts/check_numbers.py` on the build's `.aux` and `.log` files.

2. **Open scientific limitations.** The passing checks confirm that the manuscript reports the evidence consistently. They do not close these gaps:
   - **RA-InfoOpt:** configuration strata differ across seeds in the historical runs. Combined five-seed RA-InfoOpt comparisons from that historical selection remain withheld. **Partially addressed in revision 7** (item 0.2 above): a new matched experiment supplies a single, consistently configured five-seed comparison, but none of its six primary contrasts survives Holm correction, so it does not establish an advantage, and it does not recover or certify the withheld historical estimate.
   - **FedDyn and SCAFFOLD:** implementation provenance is unresolved. Their values are flagged audit observations and are excluded from rankings. **Partially addressed in revision 6** (item 0.1 above): 30 repair-campaign runs are now identified by exact path and log provenance, and the recovered implementation passes its own synthetic equation checks, but which historical source code produced the originally printed values remains unresolved.
   - **Aggregation effect not isolated (retrospective reanalysis only):** the AIB mismatch means no contrast in Tables 3–4 isolates the aggregation rule, and partition membership across matched seeds is assumed, not verified, for that material. **Partially resolved in revision 5** for FedAvg versus InfoOpt-Full specifically: the new four-arm experiment (Section 5.4) isolates the aggregation rule from AIB for that pair, with verified-identical partitions, and reaches the same qualitative outcome (see item 0 above). This does not isolate the aggregation effect for RA-InfoOpt, which remains unresolved.

3. **Secondary family (21 external-baseline comparisons).** No paired retrospective outputs exist in the supplied material, so these comparisons are not reported.

4. **Unverified diagnostics.** Convergence, the α-sweep, the ablation, GRASP empirical results, temporal risk, oracle damping and drift remain withdrawn until their own outputs are verified. The new saved-checkpoint step-size pilot (item 0.3) does not restore any of these; it is a separate, newly supplied, bounded piece of evidence under its own frozen protocol, not a recovery of the withheld oracle round-dependent damping analysis in particular.

5. **BibTeX warnings (2).** `elsarticle-num` reports empty `pages` for `acar2021federated` and `reddi2021adaptive`. Both are ICLR papers, which have no page numbers. The bibliography was left unchanged, as requested.

6. **Cosmetic overfull boxes.** All are in content unchanged from the originals:
   - Main: the heterogeneity-normalisation equation and Table 1.
   - Supplement: one D.2.2 line and four Table E.1 rows, each under 3 pt.

   They may differ under `cas-dc`.

7. **AI-use declaration.** It is unchanged; decide whether it should mention this or the prior revision.

_Resolved in revision 4:_ the CIFAR-100 FedOpt sensitivity value is now read from the original, checksum-verified package output, not a transcription.

_Resolved in revision 5 (partial):_ the AIB architecture confound between FedAvg and InfoOpt-Full specifically is now resolved by the four-arm controlled experiment (item 0); RA-InfoOpt, FedDyn, and SCAFFOLD remain unresolved as before.

_Resolved in revision 6 (partial):_ FedDyn/SCAFFOLD repair-campaign run identification (which 30 runs exist and where) is now resolved (item 0.1); which historical source code produced the originally printed FedDyn/SCAFFOLD values remains unresolved, and the CIFAR-100 early-stop eligibility conflict is disclosed but not amended.

_Resolved in revision 7 (partial):_ a single, consistently configured RA-InfoOpt comparison across all five seeds is now possible via the new matched experiment (item 0.2); none of its six primary contrasts survives Holm correction, so no RA-InfoOpt advantage is established, and the withheld historical, configuration-heterogeneous five-seed estimate is not recovered, certified, or substituted for. FedDyn and SCAFFOLD remain unresolved as before.

_Revision 8:_ no earlier unresolved item is resolved. A new, independent, bounded saved-checkpoint step-size pilot is added (item 0.3): it measures, on one dataset and one partition, that validation-selected step-size scaling does not meet the frozen headroom gate against tuned fixed controls and that the tested shrinkage rule is worse than those controls; both are measured negatives for this pilot only. An associated export defect (partial correction only, 9 of 90 affected sources regenerated) and the `PRE_EXECUTION_AMENDMENT.md` byte-identity gap are disclosed, not resolved. RA-InfoOpt, FedDyn/SCAFFOLD, and the AIB architecture mismatch for RA-InfoOpt remain unresolved as before.
