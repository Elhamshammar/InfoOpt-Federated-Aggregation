# IT-SL research-coverage audit — 11 October 2026

## Decision

Revision 7 covers the main classification evidence and both new controlled experiments. The most useful remaining addition is a compact supplementary section on the recovered CIFAR-100 step-size pilot, with one discussion paragraph linking it to the need for tuned fixed-step comparators. Add its export-correction disclosure at the same time. No further training is needed for this addition.

This is a coverage review of supplied records and manuscript source, not a fresh execution of all historical diagnostics or an independent LaTeX rebuild. Neither manuscript nor frozen inputs were modified.

## Material examined

- `revision7_corrected_review/InfoOpt_revision7/main_revised.tex`, `supplementary.tex`, and `docs/CLAIM_EVIDENCE_MAP.md`.
- `outputs/ITSL_Research_Documentation_2026-10-07-8-6.md`: research sequence, evidence ledger, Windows pilot recovery, targeted recovery and correction-archive audit.
- `local_recovery/FL_StepSize_Diagnostic/PILOT_CLOSURE.md`.
- `correction_audit/per_example_correction_20261008/CORRECTION_REPORT.md`.

The research documentation contains earlier historical comparison values. Those values are superseded by the frozen retrospective reanalysis and subsequent controlled experiments; do not import them into current results tables.

## Coverage and disposition

| Research strand | Revision 7 coverage | Recommended disposition |
|---|---|---|
| Historical classification reanalysis, repeated-run selection, architecture and code limitations | Main results/provenance and supplement A | Retain; do not overwrite frozen tables with historical summaries. |
| FedAvg/InfoOpt-Full × AIB off/on, 60 fresh runs | Main §5.4, Table 5/Fig. 5, supplement A.7 | Already covered. |
| Matched RA experiment, 60 fresh runs | Main §5.5, Table 6, supplement A.8 | Already covered; fix the remaining family-scope sentence described below. |
| FedDyn/SCAFFOLD repair campaign and synthetic equation checks | Main provenance/discussion, supplement A.2 | Already covered as historical descriptive evidence, not certified source identity or a replacement ranking. |
| Tuned fixed damping | Supplement dynamic/fixed-damping section | Covered descriptively. The one-step pilot supplies distinct evidence about incremental adaptive headroom. |
| Validation-selected step-size headroom and tested update-statistic shrinkage rule | No corresponding pilot section found | **Add supplementary subsection and small table; brief main discussion pointer.** |
| Phase-only versus variability prediction and frozen-buffer policy | No corresponding results found | Include briefly within the pilot subsection, clearly exploratory/policy-specific. Avoid a new headline or a separate figure. |
| Per-example selected-gamma export defect and partial correction | Not described | **Add essential provenance limitation alongside the pilot.** |
| Cohort-prior classifier correction (C2): CIFAR confirmation and EuroSAT gate failure | Not found | Optional short supplementary research ledger entry. Include both scopes if included; do not cherry-pick the replication failure or claim a new mechanism. It is not necessary for the aggregation paper's central comparison. |
| Local certificates/LCB and phase-confounded contribution correlations | Not reported as outcomes | Keep in research record unless the original observation stream and exact target/split are packaged for independent checking. Research-history summaries alone are insufficient to restore numerical results. |
| GRASP geometry, TRACE/temporal risk and drift | Definitions retained, empirical outcomes withheld | Keep withholding outcomes unless a dedicated source audit establishes the corresponding runs, valid diagnostics and prospective protocol. Recovered segmentation JSONLs are not automatically the classification GRASP campaign. |
| SAVI, block freezing, coalition normalization, trajectory/rollback oracles | Not included | Keep in separate segmentation/research record. Limited seeds, retrospective oracles, failed checks and inconsistent endpoints do not support a general classification intervention. |
| Dice/IoU empty-class conventions and best-validation versus test endpoints | Not classification results | Relevant to the separate segmentation manuscript and its evaluation repair. Do not import clinical performance claims here. |
| C1 recalibration, C3 head-row aggregation, H4/H4-WR mechanisms | No completed distinct mechanism established | No new empirical claims. Proposed or known-method ideas are not completed experiments. |

## Proposed pilot addition: exact scope and numbers

Use CIFAR-100, SmallCNN with AIB off, one partition seed 42, development training seed 201, held-out training seeds 202/203, saved checkpoints 5/50/100. These are **one-step evaluations at saved states**, not end-to-end controller training or independent partition replications. Repetitions are clustered; report seed-level summaries, not inflated n or new p-values.

Define G = evaluation loss of fixed control minus evaluation loss of selected action; positive is improvement. Development-selected controls are global gamma 0.5 and phase gamma 0.75/0.5/0.5.

| Action versus control | Seed 202 mean G | Seed 203 mean G | Interpretation |
|---|---:|---:|---|
| Validation-selected scaling vs global fixed | +0.005737 | +0.022656 | Frozen headroom gate not met across both seeds. |
| Validation-selected scaling vs phase fixed | +0.002900 | +0.003649 | Frozen headroom gate not met. |
| Tested raw shrinkage vs global fixed | −0.019722 | −0.035515 | Worse at each of the three checkpoints on both seeds. |
| Tested raw shrinkage vs phase fixed | −0.022559 | −0.054522 | Same bounded negative result. |

The headroom gate requires mean G at least 0.01 and positive improvement at least two of three checkpoints on each held-out seed against each control, with worst-phase deterioration at most 0.02. Against global fixed, seed 202 fails the mean requirement and seed 203 fails the positive-phase requirement. Against phase fixed, both fail the mean requirement. Improvement over gamma 1 does not establish incremental benefit beyond tuned fixed controls.

Optional brief pilot details: corrected exploratory held-out MAE is 0.009036 for phase-only and 0.020016 for phase-plus-variability (27 development/54 held-out cells, clustered repetitions; rank-deficient larger design). Frozen checkpoint buffers at gamma 1 worsen held-out mean loss by 0.0629 relative to full-state FedAvg; this does not test normalization recalibration or the unmeasured frozen-buffer scaling grid.

## Essential export limitation

The original 450,000 validation-selected per-example rows silently used gamma 1 after a cache was cleared; intended gamma differed for 77/90 sources. The hardened checker found 154 mismatched cells. Retained grid/rule summaries and decision inputs used correct gamma and remain the basis of the headroom decision. Do not compute selected-policy sample-level confidence intervals or bootstrap results from that original export.

Only nine trajectory sources were regenerated (45,000 rows/18 cells, verified). The other 81 sources lack persisted post-local states in the supplied recovery material. The corrected subset is not a full replacement. Describe the historical amendment as a later change record where its hash differs from the recorded freeze; do not call current bytes the exact original frozen amendment.

## Two manuscript wording fixes

1. Replace the remaining §5.5 sentence about both probe conditions being reported for all primary comparisons with: “All six primary comparisons use probe-off RA; the three secondary comparisons report probe-on minus probe-off, regardless of outcome.”
2. The supplement's diagnostic introduction currently says supporting outputs were not supplied “for this revision.” Retain this limitation for the named historical GRASP/TRACE/drift outcomes, but distinguish the newly supplied step-size pilot. Do not globally restore those outcomes merely because other diagnostic artifacts now exist.

## Prompt for the writing AI

Work from corrected revision 7; preserve it and every frozen analysis package. Create a new revision containing only the following evidence-grounded integration. Do not train, tune, select repeats, change frozen tables, or introduce new inferential tests.

First inspect the supplied CIFAR step-size pilot closure, retained decision/grid/rule summaries, corrected exploratory outputs and correction report. Cross-check every imported number against the authoritative saved summaries; if a required source is absent or disagrees, stop that numerical addition and name the gap. Do not treat the research-history document as a substitute for raw authoritative summaries.

Add one supplementary subsection titled “Saved-checkpoint step-size pilot” with a compact seed-level table using the scope/numbers in this coverage audit. Separate the frozen headroom gate, tested raw-shrinkage failure, exploratory predictor comparison and buffer-policy result. Include the export defect and partial-recovery limitation. Add one short main-discussion paragraph: in this bounded pilot, gains over unit scaling did not meet the gate beyond development-tuned fixed controls; this motivates fixed-step controls rather than a universal claim that adaptation fails.

Fix the remaining primary/secondary probe-family sentence exactly as specified. Update diagnostic-evidence wording locally, the claim-to-evidence map, changelog, unresolved issues and validation summary. Keep segmentation probes, C1/C3 proposals and unsupported GRASP/TRACE outcomes outside the numerical results. C2 can remain research-only; do not add it without both confirmation and replication evidence.

Regenerate only the new supplementary table from supplied machine-readable sources. Extend numerical checks to cover its values, scope and exclusion of the defective selected-policy CSV; rebuild both documents and report actual tests/build status. The official CAS build remains pending until the real class/style files are supplied. Deliver the revised project, change list and any unresolved evidence gaps.

## Files to supply for this next integration

- Corrected revision-7 project ZIP.
- This coverage audit.
- `FL_StepSize_Diagnostic/PILOT_CLOSURE.md`, pilot `PROTOCOL.md`, `GO_NO_GO.md`, and `PRE_EXECUTION_AMENDMENT.md` (the latter labelled historical change record, not certified original freeze).
- From `results/stepsize_diag_pilot_20261004_175620/results/`: `pilot_decision.json`, saved grid/aggregate and rule evaluation CSV/JSON files used by the closure. Include partition/config/provenance metadata defining the seed and split scope.
- Corrected-v2 exploratory output files and `analysis_buffer_policy_v2` report if importing those optional numbers.
- The corrected per-example recovery package/report, including consistency reports and explicit trajectory inventory. The huge defective CSV is unnecessary for manuscript integration and must not be used for selected-policy inference.

No checkpoints, datasets or further server training are required for this manuscript addition. This audit identifies coverage, not journal readiness; official-class compilation and final scientific/editorial review remain separate tasks.
