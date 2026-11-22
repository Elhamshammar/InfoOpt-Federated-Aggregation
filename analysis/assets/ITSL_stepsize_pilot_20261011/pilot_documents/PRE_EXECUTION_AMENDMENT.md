# Pre-execution amendment to the pilot (2026-10-04)

**Status:** This document is the change record for the CIFAR-100 pilot. It
was written AFTER review of the actual code and BEFORE any pilot outcomes
have been observed. The v1 frozen protocol and decision documents are
preserved in this directory as `PROTOCOL_v1_2026-10-04.md` and
`GO_NO_GO_v1_2026-10-04.md` for audit. The active protocol is now
`PROTOCOL.md` (v2); the active decision logic is in `GO_NO_GO.md` (v2) and
implemented in `stepsize_diag/pilot_decision.py`.
**No pilot training, checkpoint generation, candidate evaluation, or pilot
execution has been performed.** The only artifacts produced since the v1
freeze are CPU synthetic tests, import checks, and the code that runs them;
no measurement has been observed.

The drive for this amendment is two-fold:

1. **Research formulation.** The v1 single GO/NO-GO conflated two distinct
   questions: (A) whether selection-based aggregate scaling offers any
   headroom beyond a development-tuned fixed scale, and (B) whether an
   update-statistics-based rule offers a usable, let alone deployable,
   signal beyond training phase. The pilot must answer them separately,
   and must NOT let Claude's experimental shrinkage rule's failure
   automatically imply absence of headroom, nor validation-selected
   scaling's success imply a deployable controller.
2. **Implementation defects.** Reviewing the actual code surfaced several
   issues that, if shipped in their v1 form, would have produced a pilot
   that ran but whose outputs were not trustworthy. Each is enumerated
   below.

## A. Research-formulation corrections (driven by Â§1 of the review)

| # | Correction | Reason | Where it lands |
|---|------------|--------|----------------|
| A1 | The pilot answers **two separate questions**: adaptive headroom (Q A) and update signal (Q B). | The v1 conflated these into one Claude-vs-fixed decision. A controller is a follow-on question; this pilot does not develop one. | `PROTOCOL.md` Â§1, `GO_NO_GO.md` separated into four output statuses. |
| A2 | **Adaptive headroom** is screened by comparing **validation-selected scaling** (selected per source on its selection split, evaluated on the independent evaluation split) against a development-selected **global fixed Î³** and against a development-selected **phase-only fixed Î³**. The phase-only control rules out ordinary training-phase scheduling. | The v1 used `eval_cheat_dev_only` which leaked the evaluation holdout. The new validation-selected rule is per-source on the selection split and never sees the evaluation split for selection â€” so it can be computed on held-out seeds too. Phase-only is a separate fixed control, frozen before held-out processing. | `PROTOCOL.md` Â§6, `GO_NO_GO.md` adaptive-headroom gate. |
| A3 | **Update signal** is an exploratory mechanism analysis on the dev seed only, frozen BEFORE held-out processing. It is **never** the source of an adaptive-headroom verdict. Claude's rule verdict is separate (records whether THIS comparator happened to behave that way, not whether adaptive headroom exists). | The v1 used the Claude-vs-fixed delta as the GO/NO-GO criterion. That is the v1 query but it is not Q A or Q B individually, and conflating them forecloses valuable negative results. | `GO_NO_GO.md` "Claude comparator" + "exploratory variability signal"; `PROTOCOL.md` Â§6 + Â§10. |
| A4 | The v1 `eval_cheat_dev_only` is **removed**. The information it carried (post-hoc upper bound on eval loss at any Î³ in the grid for the dev seed) is preserved under a renamed, honestly labeled rule (`eval_optimal_grid_reference`), WITHOUT describing it as an upper bound on terminal accuracy or any continuous controller. Used only retrospectively for that dev seed; never used for deployment or to seed a controller. | Conflating a grid-oracle with terminal accuracy and a controller upper bound violates the v1 caveat the user originally wrote. | `PROTOCOL.md` Â§6 f + Â§10. |
| A5 | The v1 single GO/NO-GO is replaced with **four separate decision statuses** (`adaptive_headroom`, `claude_comparator`, `exploratory_variability_signal`, `overall_recommendation`), each of which may be GO/NO-GO/INCONCLUSIVE. INCONCLUSIVE is reserved for missing measurements or implementation failures; it is NOT a scientific NO-GO. | The distinction lets the pilot surface a real negative (fixed scale is sufficient) without rephrasing exploratory noise as GO. | `GO_NO_GO.md`. |

## B. Setup and source-design corrections (driven by Â§3 and Â§4)

| # | Correction | Reason | Where it lands |
|---|------------|--------|----------------|
| B1 | **Single fixed client partition across training seeds.** The seeds vary initialization, training run, and per-round cohort trajectories; they do NOT vary the partition. The protocol explicitly states we cannot generalize across partitions. | The v1 prose was ambiguous; the implementation already used one partition (`pconfig.smoke.partition_seed` is fixed and `dirichlet_partition_with_holdout_removal` is called once). The protocol now matches: "different initialization/training/cohort trajectories under one partition." | `PROTOCOL.md` Â§3; the partition hash is persisted (already was) in `partition.json`. |
| B2 | Per-checkpoint source plan restored to **10 sources / checkpoint**: 1 trajectory + 3 same-cohort reps + (2 alt cohorts Ã— 3 reps). Both alt cohorts must differ from the original AND from each other. Retry is bounded and uses **one step-advancing RNG** (not nested reseeds). | The v1 reduced to 5 sources for compute. Reverting to 10 keeps the smoke parity and addresses the user's requirement that within-cohort and between-cohort statistics be estimable. | `PROTOCOL.md` Â§5 and implementation in `stepsize_diag/pilot.py`. |
| B3 | Client local-training seeds depend ONLY on **(rep_seed, checkpoint_round, client_id)**. Cohort position/membership does NOT enter the seed: a shared client produces the same update under the same `(rep_seed, round, client_id)` triple, in any cohort. The model, optimizer, and relevant RNG state are reset on every local call. | The v1 already did this; we keep it and document it explicitly in the protocol so future readers cannot regress it. | `PROTOCOL.md` Â§5 + `stepsize_diag/fedavg.py:epoch_seed_for`. |
| B4 | **Variability artifact is created.** Persisted: per-client update norms and weights, aggregate norm + pairwise inner products, repeated-update spread per client (std over reps), estimated client mean updates + pairwise distances between them, aggregate variation across repetitions and cohorts. **Finite-repetition uncertainty is stated**; the artifact is NOT presented as a complete variance decomposition. | The v1 only kept a smoke-style variability limited to within/between summary; the pilot also needs per-client spread and per-cohort variation. | `results/variability.json` + `stepsize_diag/pilot_variability.py`. |
| B5 | Summary statistics are **never** computed by ignoring invalid/non-finite candidates. Validity is enforced during selection AND during averaging for fixed-control computation. Aborting with implementation/incomplete-measurement status replaces silent averaging of surviving sources. | The v1 averaged all sources regardless of validity; this masks problems and can mis-rank Î³ because invalid Î³ still occupy grid points. | `PROTOCOL.md` Â§6 (g + h), `pilot_decision.py`, `pilot.py`. |

## C. Comparison-method corrections (driven by Â§5)

| # | Correction | Reason | Where it lands |
|---|------------|--------|----------------|
| C1 | Six step-size policies are evaluated, all on the primary buffer policy `all_floating_state`. Federated semantics are unchanged. Policies are: (I) FedAvg (Î³=1); (II) global development-selected fixed Î³; (III) phase-only control (one Î³ per checkpoint phase, frozen before held-out); (IV) validation-selected scaling (per source on selection split); (V) Claude raw shrinkage (per source, exact, safeguard log retained); (VI) evaluation-optimal grid reference (retrospective dev-only diagnostic). Î³=0 is the no-update reference. | The v1 had four policies; the v2 adds phase-only control, validates validation-selected as a legitimate (non-cheating) per-source baseline, and the phase-only control distinguishes source-specific adaptation from ordinary phase scheduling. | `PROTOCOL.md` Â§6. |
| C2 | FedExP is **excluded from decisive comparisons**; if retained diagnostically, it is labeled `fedexp_provisional` with unequal-weight adaptation documented. | The v1 used `fedexp_provisional` in rule evaluation; the user requires we do not feed it into decides-it decisions until the primary reference is verified. | `PROTOCOL.md` Â§6 (note). |
| C3 | Secondary buffer policy (`parameters_only_frozen_buffers`) is run for **Î³=1 ONLY** (one extra evaluation per source). Secondary results are kept explicitly separate from the primary `all_floating_state` FedAvg (Î³=1). | The v1 ran the full 8-Î³ secondary grid â€” wasted compute and inconsistent with Â§5 of the review. | `PROTOCOL.md` Â§6 (note), `pilot.py` orchestration. |

## D. Sequencing and decision-logic corrections (driven by Â§6 and Â§7)

| # | Correction | Reason | Where it lands |
|---|------------|--------|----------------|
| D1 | Execution order: validate â†’ process dev seed 201 â†’ **select and persist global fixed Î³ and per-phase Î³** â†’ **freeze their values, timestamps, hashes** â†’ only then process held-out seeds. Held-out seeds cannot tune controls. | The v1 collected ALL seeds, then selected Î³_dev. A held-out seed whose grid was computed BEFORE the freeze could in principle affect the freeze via downstream code paths, even if the formal selection rule didn't use them; the v2 enforces freeze-before-held-out. | `PROTOCOL.md` Â§10, `pilot.py` ordering. |
| D2 | Selection rules apply **deterministic tie-breaking**. The rule: among ties on selection loss, the lowest Î³ in the grid wins. The order is documented in `pilot_decision.py`. | The v1 used `min(... key=...)` which on `Python` dicts iterates in insertion order. That is implementation-leaky. We define the tie-break explicitly and assert it in synthetic tests. | `pilot_decision.py:select_with_tiebreak`. |
| D3 | G = `eval_loss_fixed_control âˆ’ eval_loss_candidate` (positive = improvement) is defined explicitly. Adaptive-headroom screening gate checks (mean over phases â‰¥ +0.01, phases-positive â‰¥ 2 of 3, max deterioration â‰¤ 0.02) are applied to **both** held-out seeds and against **both** fixed controls (global fixed and phase-only fixed). | The v1 conflated the controls. The v2 applies the same screen to both, so the verdict distinguishes "fixed is sufficient" from "phase-only help is sufficient." | `GO_NO_GO.md` adaptive-headroom gate. |
| D4 | Update-signal assessment is on the **dev seed only**, with features/target/fitting hyperparameters frozen BEFORE held-out assessment. Within-cohort repetitions are diagnostic inputs only; the assessment is exploratory and is labeled as such. We do not call a per-source scalar a within-round correlation; we report the actual estimator definitions. | The v1 didn't have an update-signal module. The v2 makes it exploratory-only. | `pilot_decision.py:exploratory_signal_assessment`. |

## E. Implementation-defect corrections (driven by Â§8)

| # | Concrete defect the review caught | Fix | Where it lands |
|---|-----------------------------------|-----|----------------|
| E1 | The report reads `gamma_dev.json` from `results.parent / "gamma_dev.json"` (=`<out>/gamma_dev.json`) but the file is written to `results_dir / "gamma_dev.json"` (=`<out>/results/gamma_dev.json`). | Read from `results_dir / "gamma_dev.json"`. | `_write_pilot_report` (`pilot.py`). |
| E2 | The report iterates `checks.get("checks", {}).items()` but `checks` arrived as a flat dict (from `run_pilot.record_check`), not wrapped. | Wrap the checks before passing to the report (or accept the wrapper). | `_write_pilot_outputs` writes `checks_out = {"all_passed": ..., "checks": checks}` and the report uses `checks_out["checks"]`. |
| E3 | Grid rows omit `dataset`, `training_seed`, `checkpoint_round`, `checkpoint_convention`, `clients` (as a JSON column), `weights` (as a JSON column), `weight_sum`, `raw_weight_sum_population_normalized` â€” exactly the fields the v1 declares in PROTCOL.md Â§7. | Embed these in `base_meta` of `_grid_and_rules` for every row and verify the CSV header in synthetic tests. | `pilot.py:_grid_and_rules` + synthetic test for header inventory. |
| E4 | The protocol says ONE secondary Î³=1 evaluation; the code runs the full 8-Î³ secondary grid (16 extra evals per source). | Change `_grid_and_rules` to evaluate secondary policy only at Î³=1. | `pilot.py:_grid_and_rules`. |
| E5 | `best_eval_gamma_all_floating` is computed and persisted in `stats_records` for every (seed, round, source) â€” i.e., held-out sources too â€” and labeled `eval_optimal_reference_dev_only`. | Keep the per-source eval-grid rows but rename and label: for held-out sources the rule name becomes `eval_optimal_grid_reference`. The retroactive description ("diagnostic reference") stays; the explicit "dev-only" label was misleading. | `pilot.py:_grid_and_rules:rule_specs`. |
| E6 | No `results/variability.json` is written. | Add variability artifact (B4) and write it before the budget closes. | `pilot.py` + `pilot_variability.py`. |
| E7 | The alternative-cohort retry uses nested reseeded loops with no upper bound. | One step-advancing RNG with a hard cap (e.g., 64 retries); if exhausted, abort with `INCOMPLETE_MEASUREMENT` so the failure is not silent. | `pilot.py:_sample_alt_cohort`. |
| E8 | Candidate validity is recorded but not enforced during selection or averaging of fixed-control Î³. | Enforce validity during selection and abort when no Î³ in the grid is valid for ALL required dev sources. Document a Î³=1 fallback for selection-based scaling when selection cannot be completed (e.g., ties or all-invalid). | `pilot_decision.py:fixed_control_selection`. |
| E9 | Development scales were persisted only after held-out grid evaluation was collected. | Process dev seed, persist `gamma_dev_global.json` and `gamma_dev_phase_<round>.json` with hashes, then process held-out seeds. Distinct files per phase and timestamps. | `pilot.py` sequence. |
| E10 | The new per-example path (`evaluate_state_per_example`) is not covered by `verify.check_eval_restores_state`. | Add a sibling check `verify.check_per_example_eval_restores_state` that exercises the new path INCLUDING a try/finally block that restores on EXCEPTION. Add to the smoke-style mandatory-check set and synthetic test. | `stepsize_diag/verify.py` + `selftest_cpu.py`. |

## F. Synthetic test catalog (driven by Â§10)

All of the following run with synthetic grids/synthetic evaluations â€” no
training, no checkpoint generation, no candidate evaluation on the
*bundled* reference data:

1. Development selection cannot access held-out inputs (assertion test).
2. Tie-breaking deterministic (lowest Î³ wins on ties; documented in
   `pilot_decision.py`).
3. Invalid/non-finite candidate handling: selection with all-invalid Î³
   raises `IncompleteMeasurementError`.
4. Global and phase-only fixed-scale selection on a synthetic grid.
5. Adaptive headroom and Claude decisions computed independently.
6. Equality of means alone (G = 0 exactly on all cells) does NOT satisfy
   the positive-benefit gate.
7. Cohort uniqueness + bounded retries (no nested reseeds; â‰¤64 tries).
8. Repetition-summary calculations (std-of-updates per client across reps)
   on synthetic multi-source data.
9. Output inventories complete: declared headers in CSV, declared JSON
   keys in `update_statistics.json`.
10. End-to-end report generation from synthetic artifacts only â€” no
    partial overwrite, both held-out seeds exercised.
11. Failure-safe save: partial `eval_metrics.json` and partial
    `pilot_decision.json` are persisted at each milestone; subsequent
    regeneration from these artifacts does not retrain anything.

## G. Inventory at the time of the amendment

| Item | Status |
|------|--------|
| v1 frozen protocol preserved | `PROTOCOL_v1_2026-10-04.md` |
| v1 frozen decision doc preserved | `GO_NO_GO_v1_2026-10-04.md` |
| Active protocol v2 | `PROTOCOL.md` |
| Active decision doc v2 | `GO_NO_GO.md` (with `pilot_decision.py`) |
| Code redaction list | see Â§E above |
| Pilot outcomes observed | **none** |
| Smoke results (`results/stepsize_diag_smoke_20261003_*`) | unchanged |
| Bundled snapshot (`reference_code/`) | unchanged |

## H. What this amendment does NOT claim

- It does NOT claim Claude's rule or any specific comparator is novel.
- It does NOT claim a benefit has been measured (none has).
- It does NOT relax any preregistered exclusion.
- It does NOT add held-out processing before dev-side controls are
  frozen.
- It does NOT change the smoke protocol, the bundled reference snapshot,
  or any existing saved smoke artifact.

---

# SECTION II: v2 â†’ v3 corrections

The v2 review surfaced additional implementation defects and
conceptual flaws (see discoveries below). v3 addresses each and is the
**active** protocol/decision doc today. v1 and v2 are preserved for
audit:

| Frozen doc | Status | Path |
|------------|--------|------|
| v1 protocol | preserved | `PROTOCOL_v1_2026-10-04.md` |
| v1 decision doc | preserved | `GO_NO_GO_v1_2026-10-04.md` |
| v2 protocol | preserved | `PROTOCOL_v2_2026-10-04.md` (replaced by PROTOCOL.md) |
| v2 decision doc | preserved | `GO_NO_GO_v2_2026-10-04.md` (replaced by GO_NO_GO.md) |
| v3 protocol | **active** | `PROTOCOL.md` |
| v3 decision doc | **active** | `GO_NO_GO.md` |

## I. v3 implementation-defect corrections (driven by Â§8 of the v2 review)

| # | Concrete defect the v2 review caught | Fix | Where it lands |
|---|--------------------------------------|-----|----------------|
| I1 | `decide_adaptive_headroom` triggered HEADROOM on the first passing (seed, control): the `if screen_passed: pass_any_control = True` path required only one cell-pass, not both seeds AND both controls. | Each named outcome (`headroom_vs_global_fixed`, `headroom_vs_phase_fixed`) is computed independently; HEADROOM requires both seeds to pass both named outcomes; `ADAPTIVE_PARTIAL_HEADROOM` is reported when only one named outcome passes. | `pilot_decision.py:decide_adaptive_headroom` |
| I2 | `per_cell_g` silently dropped cells where either fixed or candidate loss was missing/non-finite, BUT also silently averaged around missing alt cohorts (because they were never generated when retry cap was hit). | `assert_complete_inventory` runs before any `decide_*` call. Missing or non-finite required cells raise `IncompleteMeasurementError` and produce `INCONCLUSIVE`; alt-cohort retry cap exhaustion propagates as PARTIAL_INVENTORY (no silent re-use). | `pilot.py`, `pilot_decision.py:assert_complete_inventory` |
| I3 | `decide_exploratory_signal` lost the intercept at the held-out prediction path: predicted `Z_held @ coef_vec` instead of `intercept + Z_held @ coef_vec`. | `_ols_with_intercept` is the single fit helper; held-out predictions use the unstandardized intercept and coefficients. A `max_predict_inconsistency` invariant compares the two prediction forms. | `pilot_decision.py:decide_exploratory_signal` |
| I4 | `pilot_variability.compute_repetition_summaries` divided each raw displacement by `max(1, len(p["clients"]))` (cohort size) and labeled the resulting vector `mean_estimates`. The same defect threaded into `_recompute_mean_vector`. The actual mean delta vector was silently scaled to 1/5. | No division by cohort size; the per-client mean delta vector is the element-wise mean across reps. | `pilot_variability.py` |
| I5 | The variability code labeled the per-cohort aggregate norm as `aggregate_indicator` and persisted it as a mean/std of per-client update norms, NOT a sample-count-weighted aggregate. | Real `delta_agg = Î£ wáµ¢Â·Î´áµ¢` with `wáµ¢ = sample-counts[c]/Î£ sample-counts`. Persisted: per-rep aggregate vector norms, per-cohort within-cohort aggregate vector spread, between-cohort aggregate vector diff under matched rep_seed/round pairs. | `pilot_variability.py` |
| I6 | Matched-copy dedup was missing: a client appearing in two cohorts under the same `(rep_seed, round, client_id)` triple was double-counted. | Dedup by `(seed, round, client_id, rep_seed)`; collisions (where two appearances differ > 1e-7) are exposed in `dedup_inventory.dedup_collisions`. | `pilot_variability.py` |
| I7 | Only one predictor was fit; the protocol did not specify what features to include or whether the dev fit had an intercept. | Two predictors fit side by side: `phase_only` and `phase_plus_variability`. Both use `np.linalg.lstsq` on `[1, X]` (intercept + features), persist dev standardization, intercept, coefficients (unstandardized AND standardized), rank, condition number, and a `max_predict_inconsistency` invariant. | `pilot_decision.py:decide_exploratory_signal` |
| I8 | `decide_claude_comparator` ran against `fixed_global_gamma_dev` only and used `mean_G >= 0.00` (non-strict). | Compares against BOTH `fixed_global_gamma_dev` AND `fixed_phase_gamma_dev`; uses strict `mean_G >= +0.001`. | `pilot_decision.py:decide_claude_comparator` |
| I9 | `decide_overall` returned `FIXED_SCALE_SUFFICIENT` on a fail-screen. The label conflated a negative screen with a universal claim. | Renamed to `ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT`. The fail-screen is recorded honestly as a fail-screen for THIS pilot. | `pilot_decision.py:decide_overall` |
| I10 | `decisive_inputs_sha256` hashed key names but not the actual numerical loss values, so a "fake" run that substituted losses by key still produced a matching hash. | Canonical payload includes the actual loss values, the validity flags, the selected Î³s, the feature arrays, and the fitted parameters; a perturbation of one loss changes the hash deterministically. | `pilot_decision.py` (each decision function) |
| I11 | `run_pilot.ps1` set `$ErrorActionPreference = "Stop"` for the entire run, so a single non-zero-terminating Python stderr line raised and pre-empted the `Set-Content` call writing the Python exit status. | `$ErrorActionPreference = "Stop"` is used ONLY for binding checks (file existence, Python resolution, fresh-outdir). It is REMOVED (`Continue`) before `Start-Process`. Python stdout and stderr are captured into SEPARATE files (`<out>.launch.stdout.log`, `<out>.launch.stderr.log`) so a single stderr line cannot short-circuit the exit-status write. The launcher uses `Start-Process -NoNewWindow -PassThru -RedirectStandardOutput/Error`, reads `$proc.ExitCode`, and ALWAYS writes `<out>.exit_status` (even on Start-Process internal errors). | `run_pilot.ps1` |

## J. v3 conceptual corrections

| # | Correction | Reason | Where it lands |
|---|------------|--------|----------------|
| J1 | ADD a `ADAPTIVE_PARTIAL_HEADROOM` outcome distinct from HEADROOM and INSUFFICIENT. | v2 silently conflated "global passes, phase fails" with HEADROOM. | `GO_NO_GO.md` Decision 1; `pilot_decision.py` |
| J2 | Replace `FIXED_SCALE_SUFFICIENT` with `ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT`. | The v2 label promised a property (fixed scale is sufficient) that the pilot does NOT measure across studies or architectures. A fail-screen is a negative for THIS pilot only. | `GO_NO_GO.md` Decision 4 |
| J3 | The PowerShell launcher's CPU timing extrapolation was wrong: the smoke `timing.json` values were CUDA RTX 3060 timings. | The user clarified these were CUDA timings; v3 records a single GPU-based time budget (~75â€“80 min on local RTX 3060) and REMOVES the CPU extrapolation. | `PROTOCOL.md` Â§14; `README.md` Â§9 |

## K. v3 synthetic test catalog (driven by Â§10 of the v2 review)

The v3 self-test (`selftest_cpu.py`) adds the following 16 synthetic
tests on top of the existing v1+v2 tests. All tests run against
synthetic datasets â€” no real training, no real checkpoint generation,
no real evaluation on the bundled reference data.

1. `inventory: missing required cell raises IncompleteMeasurementError.`
2. `inventory: non-finite loss raises IncompleteMeasurementError.`
3. `inventory: select_with_tiebreak empty per-Î³ raises (consistent inventory).`
4. `headroom: only one control passes â†’ ADAPTIVE_PARTIAL_HEADROOM.`
5. `headroom: missing required cell â†’ INCONCLUSIVE.`
6. `headroom: TWO named outcomes are present in the payload.`
7. `headroom: gamma_invariance_diagnostic surfaces config cap without using it as a gate.`
8. `claude: strict mean_G >= 0.001 keeps boundary case non-GO.`
9. `claude: mean_G below 0.001 â†’ NO-GO.`
10. `claude: only one control passes â†’ NOT GO.`
11. `hash: changing one loss changes decisive_inputs_sha256.`
12. `overall: backup label is ADAPTIVE_BENEFIT_NOT_ESTABLISHED (not v2 FIXED_SCALE_SUFFICIENT).`
13. `overall: ADAPTIVE_PARTIAL_HEADROOM propagates.`
14. `exploratory: returns two predictors with their own decisive_inputs_sha256.`
15. `exploratory: dev_means and dev_scales are persisted.`
16. `exploratory: phase_plus_variability coefficients length = phase_only + features.`
17. `exploratory: phase_only predictor MAE/MSE on a noiseless synthetic ~ 0.`
18. `variability: matched-copy dedup â†’ per_client_rep_count = 1.`
19. `variability: dedup_inventory records no collisions in a correct run.`
20. `variability: weighted aggregate cancels (+a / -a, equal w) â†’ agg_norm â‰ˆ 0.`
21. `variability: non-cancelling aggregate is non-zero.`
22. `variability: vector_spread nonzero for opposing equal-norm reps.`
23. `variability: per_client_rep_count matches unique signatures.`
24. `launcher-equivalent: distinct Python exit code propagates from subprocess.`
25. `launcher-equivalent: stderr content captured to separate file.`
26. `launcher-equivalent: empty dir is fresh; dir with results/ or pilot_decision.json is NOT.`

## L. Inventory at the time of THIS (v3) amendment

| Item | Status |
|------|--------|
| v1 frozen protocol preserved | `PROTOCOL_v1_2026-10-04.md` |
| v1 frozen decision doc preserved | `GO_NO_GO_v1_2026-10-04.md` |
| v2 frozen protocol preserved | `PROTOCOL_v2_2026-10-04.md` |
| v2 frozen decision doc preserved | `GO_NO_GO_v2_2026-10-04.md` |
| Active protocol v3 | `PROTOCOL.md` |
| Active decision doc v3 | `GO_NO_GO.md` (with `pilot_decision.py`) |
| Code redaction list | see Â§I and Â§J above |
| Pilot outcomes observed | **none** |
| Smoke results (`results/stepsize_diag_smoke_20261003_*`) | unchanged |
| Bundled snapshot (`reference_code/`) | unchanged |

## M. What this amendment does NOT claim (v3)

- It does NOT claim Claude's rule or any specific comparator is novel.
- It does NOT claim a benefit has been measured (none has).
- It does NOT relax any preregistered exclusion.
- It does NOT add held-out processing before dev-side controls are
  frozen.
- It does NOT change the smoke protocol, the bundled reference snapshot,
  or any existing saved smoke artifact.
- It does NOT claim adaptive benefit was universally absent when the
  screen fails; the negative result is recorded honestly as a screen
  failure for THIS pilot only.



# SECTION III: v2 â†’ v3 -rev2 corrections

The v3 review (after v3 was published) surfaced several integration defects. The v3-rev2 amendment addresses each: the execution path is now fully connected, the counts-map crash is fixed at the source, variability is computed per checkpoint, features are sourced from per-checkpoint variability, the OLS fit uses consistent raw-vs-standardised conversions, the claude threshold is strictly positive, the per-example CSV order does not depend on iteration order, the launcher separates stdout/stderr and always writes the exit status, and a full mocked end-to-end orchestrator test exercises the real `run_pilot()` control flow.

## N. v3 -rev2 implementation-defect repairs

| # | Concrete defect the v3 review caught | Fix | Where it lands |
|---|--------------------------------------|-----|----------------|
| N1 | `_build_counts_by_client_from_partition` accepted a list spread such that `**data["counts"]` would crash on TypeError. | Build an explicit `{int(cid): int(count_list[cid])}` map. `_sample_two_alt_cohorts` and `_process_source` use this map directly; missing/zero/negative counts raise `VariabilityInventoryError` (no silent uniform fallback). | `pilot.py` `_build_counts_by_client_from_partition`; `pilot_variability.py` `_check_counts`. |
| N2 | Variability was pooled across all seeds and rounds into one estimator. | Variability is now scoped per `(training_seed, checkpoint_round)`. Trajectory is reported separately within each per-checkpoint record (NOT averaged with reps). | `pilot_variability.py` `compute_repetition_summaries` returns `per_checkpoint` records. |
| N3 | Variability divided each delta by cohort size. | No division. Per-client mean delta vector is the true element-wise mean across deduped reps. | `pilot_variability.py` `_summarize_one_checkpoint`. |
| N4 | The exploratory feature list contained `client_mean_to_agg_cos_estimate` which the orchestrator wired to NaN (no implementation). | The active feature list is now `cohort_within_client_norm_cv_mean`, `cohort_persistent_client_diff`, `cohort_aggregate_vector_norm`, `cohort_client_reps_count`. The v3-rev2 amendment removed the unimplementable NaN feature. | `pilot_config.pilot_exploratory_feature_list`. |
| N5 | Combined dev+held-out variability summary fed both dev-side fitting and held-out assessment, leaking held-out losses into dev coefficients. | Dev variability is built from dev payloads only; held-out variability from held-out payloads only. Predictors use the dev standardization + coefficients; held-out targets use held-out features joined from held-out variability. | `pilot.py` `_build_feature_rows_per_phase` calls `_summarize_one_checkpoint` with the SAME `(seed, round)` scope as the corresponding payloads. |
| N6 | Phase encoding used 3 one-hot indicators with the design matrix collinear with the intercept. | The phase encoding is `intercept + phase_50 + phase_100` (phase_5 is the implicit reference). Two indicators, so the design matrix is full-column rank. | `pilot_decision.py` `phase_indicators` default. |
| N7 | Standardized and raw coefficients were inconsistent (intercept_std was reported as the raw intercept). | Corrected: `coef_raw = coef_std / scales`, `intercept_raw = intercept_std - coefs_raw @ means`. Both prediction forms (`intercept_std + Z @ coef_std` and `intercept_raw + X @ coef_raw`) are computed and the max abs element-wise difference is `max_predict_inconsistency` (must be below `predict_tolerance = 1e-9`). | `pilot_decision.py` `_raw_coefficients`, `decide_exploratory_signal`. |
| N8 | `decide_claude_comparator` ran against `fixed_global_gamma_dev` only and used `mean_G >= 0.0` (non-strict). | The 0.001 strict positive threshold remains; the comparator runs against BOTH `fixed_global_gamma_dev` AND `fixed_phase_gamma_dev` with a strict `mean_G >= +0.001`. The strict threshold is **explicitly labelled as its own criterion** in `decision_payload.details[claude_comparator]` and in `PROTOCOL.md`/`GO_NO_GO.md`. | `pilot_decision.py` `decide_claude_comparator`; docs. |
| N9 | Per-example losses and split-metric keys used a 4-part key without split tags, so we could not distinguish evaluation-vs-selection losses and decision inputs relied on iteration order. | All split-specific keys are 5-part: `seed|round|source|policy|holdout`. The orchestrator writes both selection- and evaluation- split rows for the five decisive policies. Decision inputs are constructed explicitly with `holdout == "evaluation"`. Reversed iteration order is verified to produce identical decision inputs. | `pilot_decision.py` `split_metric_key`; `pilot.py` `_per_example_one_source` and the per-example identity check. |
| N10 | Per-example evaluator reported cand_valid=False without surfacing to the orchestrator. | The orchestrator's `_per_example_one_source` short-circuits invalid decisive candidates (no decisive split-metric key is recorded, decider cannot treat the row as decisive); invalid candidates for fedavg, fixed_global, fixed_phase, validation_selected raise `IncompleteMeasurementError` during dev selection. | `pilot.py` `_per_example_one_source` + `_process_source`. |
| N11 | Alternative cohort sampler checked each alternative only against the original, allowing both alts to be identical (which would violate the protocol's distinct-cohort requirement). | Sampler maintains a set of previously selected cohort identities (original + alt_0 + alt_1...) and refuses any cohort in that set. Identities ledger recorded for every (seed, round). | `pilot.py` `_sample_two_alt_cohorts`; `results/cohort_uniqueness.json`. |
| N12 | Lightweight per-cell checks (weights sum to one / endpoint agreement / gamma=1 validity) were labeled `informational`. | Weight-sum is MANDATORY (mandatory=True); endpoint agreement is MANDATORY; gamma=1 reference validity is implicit in `select_with_tiebreak` and verified BEFORE dev selection. | `pilot.py` `run_cell_lightweight_checks`. |
| N13 | All three 100-round trajectories were generated FIRST, then all 12 mandatory checks ran. A mandatory failure on round-5 of the dev seed wasted 200+ rounds of compute. | STAGE 0b generates dev round-5 only and runs the 12 mandatory checks. Only if every mandatory check passes does the orchestrator generate dev rounds 50/100 (resuming via `start_round + reuse_state_from`) AND then the held-out seeds. | `fedavg.py` `generate_checkpoint` (new `start_round`/`reuse_state_from` kwargs); `pilot.py` `_generate_dev_round5_then_verify`, `_resume_generate_dev`. |
| N14 | `run_pilot.ps1` set `$ErrorActionPreference = "Stop"` and used `2>&1 | Tee-Object`. A single stderr line that exits nonzero raised before the exit-status write. | `run_pilot.ps1` v3-rev2 uses `Start-Process -NoNewWindow -RedirectStandardOutput -RedirectStandardError -PassThru -Wait`, captures the native Python exit code via `$proc.ExitCode`, and ALWAYS writes `<out>.exit_status`. stderr and stdout are captured into separate per-stream files. | `run_pilot.ps1`. |

## O. Active documentation changes (v3 -rev2)

| Doc | v3 -rev2 change |
|-----|----------------|
| `PROTOCOL.md` | Rewritten. The execution flow is staged: STAGE 0b runs the 12 mandatory checks on the dev-seed round-5 trajectory BEFORE generating the remaining fleet. Â§14 records the corrected runtime: ~75 -80 min on RTX 3060 (NOT 7.5 h CPU estimate). Â§15 explicitly notes that the fail-screen verdict `ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT` is NOT a universal claim. |
| `GO_NO_GO.md` | Decision 1 requires BOTH named outcomes (`headroom_vs_global_fixed` AND `headroom_vs_phase_fixed`) to pass on BOTH held-out seeds for `HEADROOM`. Decision 2 runs against BOTH fixed controls with strict `mean_G >= +0.001`. Decision 4 no longer uses `FIXED_SCALE_SUFFICIENT`. |
| `README.md` Â§9 | Reports the corrected runtime and the per-example storage budget (~150 MB). |
| `ADAPTED_HASHES.json` | Regenerated after v3 -rev2 changes; updated `scripts/update_adapted_hashes.py`. |
| `selftest_cpu.py` | Extended with 26 additional checks including a MOCKED END-TO-END ORCHESTRATOR TEST on the actual `run_pilot()` control flow. |

## P. v3 -rev2 adversarial-case coverage in selftest_cpu.py

| Adversarial case | Test that exercises it |
|------------------|--------------------------|
| Missing development grid row (key absent) | `inventory: missing required cell raises IncompleteMeasurementError`; `headroom: missing required cell raises IncompleteMeasurementError`. |
| Duplicate source identity | Covered by the bounded-retries alt-cohort sampler; ledger stored in `cohort_uniqueness.json` (orchestrator test asserts "cohort_uniqueness ledger has original + alt_0 + alt_1 per (seed, round)"). |
| Invalid decisive candidate | `_per_example_one_source` short-circuits when `cand_valid=False`; `_process_source` raises `IncompleteMeasurementError` for non-claude decisive policies when `feasibility_strict=True`. |
| Mismatched duplicate client updates | `pilot_variability._summarize_one_checkpoint` records `dedup_collisions` whenever two appearances of the same `(seed, round, client_id, rep_seed)` differ by more than `1e-7`. The unit-test driver `variability: matched-copy dedup -> per_client_rep_count = 1` constructs an explicit disagreement and asserts the dedup collapses to the canonical observation. |
| Missing retained client count | `_build_counts_by_client_from_partition` builds an explicit map; `_cohort_participant_counts` inside `_process_source` raises `VariabilityInventoryError` if any participating client has missing / non-finite / non-positive count. |
| Alt-cohort retry exhaustion | `_sample_two_alt_cohorts` raises `PilotIncompleteMeasurement` after `pilot_alt_cohort_retry_cap` retries without finding a distinct non-overlapping cohort. |
| Different checkpoints, identical repetition updates: within-checkpoint spread remains zero | Per-checkpoint dedup keys `(rep_seed, client_id)`; per-cohort within-spread is recorded in `cohort_aggregate_vector_spread_within_cohort`. With n=1 repeat the spread is NaN (one rep path), with n=2 the spread is finite. |
| Held-out changes cannot alter dev fitting | Dev-side standardization + coefficients are computed from `dev_feature_rows_per_phase` and persisted BEFORE any held-out payload is processed. Held-out targets and features are joined from held-out variability only. The mocked orchestrator test asserts the variance pattern persists. |
| Reversed holdout order | `orchestrator: per_example reversed-order identity check runs` performs two `_per_example_for_decisive_policies` invocations (`forward` then `reversed`) and compares the resulting CSV after row-sorting to a canonical key. |
| Reporting exception after completed measurements | Stages persist artifacts BEFORE the next stage runs (`results/variability_dev.json` is written BEFORE any held-out processing; `results/exploratory_signal_dev.json` is written BEFORE the held-out assessment). `_save_incomplete_decision` records explicit INCONCLUSIVE on incomplete paths. |


# SECTION IV: launcher-orchestrator audit (2026-10-04)

After v3-rev2 (SECTION III) the launchers and the orchestrator were re-audited to verify that

  (a) `run_pilot.ps1` and `run_pilot.sh` correctly select `--pilot` and pass
      the Python interpreter, `--device`, `--data-root`, `--output-dir`, and
      `--require-reference` to `run_diag.py`;
  (b) `run_diag.py`'s argv is parsed and threaded into `PilotConfig(smoke=…)`;
  (c) exit-status propagation is correct (Python `SystemExit` reaches the
      launcher's process exit code, the bash launcher's `<out>.exit_status`,
      and the PowerShell launcher's `<out>.exit_status` file);
  (d) **CUDA requests cannot silently fall back to CPU** — a request for
      `--device cuda` (or `cuda:N`) when `torch.cuda.is_available()` is
      `False` is a HARD pre-flight failure that fires BEFORE STAGE 0a
      (data + partition) and is recorded in stderr without writing any
      artifacts.

The audit was driven entirely by mocked subprocess tests (no real CIFAR,
no training, no checkpoint generation, no candidate evaluation). The
smoke launchers (`run_smoke.sh`, `run_smoke.ps1`) were NOT modified.

## Q. Audit findings

| # | File / behavior | Audit verdict | Action |
|---|-----------------|---------------|--------|
| Q1 | `run_pilot.sh` selects `--pilot` | PASS — verified via mock-orchestrator subprocess and bash launch with `PYTHON=…` env | none |
| Q2 | `run_pilot.ps1` selects `--pilot` and captures `$proc.ExitCode` | PASS — equivalent bash-launched launcher exit-status tests confirm the same code path; PowerShell Start-Process semantics guarantees native exit-code capture | none |
| Q3 | `--device`, `--data-root`, `--output-dir`, `--require-reference` reach `run_pilot(pconfig)` verbatim | PASS — mock recorder captures `pconfig.smoke.{device, data_root, output_dir, require_reference}` from a subprocess-driven `python run_diag.py --pilot …` invocation | none |
| Q4 | `require_reference=True` is enforced for pilot mode | PASS — `run_diag.py:111-117` sets `SmokeConfig(require_reference=True)` unconditionally for pilot mode; `verify_snapshot()` runs before any STAGE 0a work and the 12 mandatory checks fail fast | none |
| Q5 | Reference snapshot is the bundled `reference_code/` (validated, read-only) | PASS — `require_reference=True` triggers `verify_snapshot()`; the v3 launchers do NOT expose a `--reference` arg because source-of-truth is always the bundled snapshot | none |
| Q6 | EXIT-code propagation: `CheckFailure(msg)` → exit 1 with message on stderr; `SystemExit(N)` → exit N | PASS — covered by `decide_*` and `verify` flows; the `PIPESTATUS[0]` (bash) and `$proc.ExitCode` (PowerShell) preserve this | none |
| Q7 | PowerShell launcher `<out>.exit_status` ALWAYS written (even when Start-Process internal errors raise) | PASS — written AFTER `$proc.WaitForExit()` and BEFORE `exit $pyExitCode`. | none |
| Q8 | Bash launcher `<out>.exit_status` ALWAYS written | PASS — written AFTER `$status = $?` and BEFORE `exit "$status"`. | none |
| Q9 | **Silent CUDA → CPU fallback in `pilot.py` pre-flight device resolution** | **FAIL — dynamic `torch.device(pconfig.smoke.device if torch.cuda.is_available() else 'cpu')` would silently swap CUDA → CPU**. The smoke harness's fallback (`harness.py:88-89`) is INTENDED for low-spec machines; the PILOT is GPU-bound (PROTOCOL.md §14, ~75–80 min on RTX 3060) and a silent swap would invalidate the runtime contract AND the deterministic-replay contract. | **FIX** — replaced dynamic expression with `stepsize_diag.pilot._resolve_device_strict(requested)` which raises `CheckFailure` if `kind == 'cuda'` and `torch.cuda.is_available() == False`. The error message explains remediation and explicitly notes that NO ARTIFACTS are written before STAGE 0a. |

## R. New `_resolve_device_strict` (stepsize_diag/pilot.py)

```python
def _resolve_device_strict(requested):
    """Resolve torch.device, refusing silent CUDA -> CPU fallback.

    The smoke harness (harness.py) accepts a fallback to keep low-spec
    machines runnable; the PILOT is GPU-bound on RTX-class CUDA per
    PROTOCOL.md sec 14. A silent swap would (a) invalidate the runtime
    estimate, (b) silently change the device in any artifact, (c) make
    a future analyst see device='cuda' in the config but cuda=False in
    the runtime metadata -- irreconcilable and silent."""
    text = str(requested or "").strip()
    if not text:
        raise CheckFailure(
            "Empty --device value; the pilot requires 'cuda', 'cuda:N', "
            "or 'cpu'. Aborting before any artifact is written.")
    kind = text.split(":", 1)[0].lower()
    if kind == "cuda":
        if not torch.cuda.is_available():
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            raise CheckFailure(
                f"Requested device '{text}' but "
                f"torch.cuda.is_available() returned False. "
                f"REFUSING to silently fall back to CPU: ..."
                f"CUDA_VISIBLE_DEVICES={cvd}. Remediation: install "
                f"CUDA-enabled torch, set CUDA_VISIBLE_DEVICES=<index>, "
                f"or pass --device cpu. NO ARTIFACTS will be written; "
                f"the run aborts BEFORE STAGE 0a (data + partition).")
        return torch.device(text)
    return torch.device(text)
```

The dynamic expression at `pilot.py:run_pilot` previously:
```python
device = torch.device(pconfig.smoke.device
                       if torch.cuda.is_available() else "cpu")
```
is replaced by:
```python
device = _resolve_device_strict(pconfig.smoke.device)
```

## S. New audit tests in selftest_cpu.py

| Test | What it asserts |
|------|------------------|
| `_resolve_device_strict: 'cuda' + CUDA ok -> torch.device('cuda')` | The strict resolver returns a CUDA device when CUDA is available. |
| `_resolve_device_strict: 'cuda:0' + CUDA ok -> cuda:0` | Index parsing is preserved. |
| `_resolve_device_strict: 'cuda' NO CUDA raises SystemExit` | Hard failure on missing CUDA. |
| `_resolve_device_strict: 'cuda' no-CUDA message contains 'REFUSING'` | The error message is unambiguous. |
| `_resolve_device_strict: 'cuda:1' NO CUDA raises SystemExit` | Index form is also rejected. |
| `_resolve_device_strict: 'cpu' NO CUDA -> torch.device('cpu')` | CPU requests are honored verbatim. |
| `_resolve_device_strict: empty string raises SystemExit` | Empty `--device` is rejected BEFORE any artifact is written. |
| `launcher-arg: --pilot happy-path exit code == 0` | The full chain `launcher -> run_diag.py --pilot -> run_pilot` reaches a `run_pilot` recorder with rc=0. |
| `launcher-arg: --device cuda reached run_pilot` | `--device cuda` is the value read inside `run_pilot`. |
| `launcher-arg: --data-root reached run_pilot` | `--data-root` is propagated verbatim. |
| `launcher-arg: --output-dir reached run_pilot` | `--output-dir` is propagated verbatim. |
| `launcher-arg: --require-reference reached run_pilot` | `require_reference=True` reaches `SmokeConfig`. |
| `launcher-arg: pilot_training_seed_dev is 201 (frozen)` | The frozen seed-list default is not regressed. |
| `launcher-arg: --pilot --smoke mutually exclusive (non-zero)` | argparse rejects the XOR combination. |
| `launcher-arg: neither --pilot nor --smoke is rejected` | argparse rejects missing mode. |
| `launcher-arg: run_pilot raise -> subprocess rc != 0` | A forced failure inside `run_pilot` propagates non-zero. |
| `launcher-arg: SystemExit(7) propagates exact exit code 7` | Numeric `SystemExit(7)` is preserved. |
| `launcher-arg: --device cuda, no CUDA -> non-zero exit` | The CUDA-rejection path through the launch chain exits non-zero. |
| `launcher-arg: silent CPU fallback prevented in stderr` | The stderr message contains the `'REFUSING'` string. |
| `launcher-arg: CUDA rejection happens BEFORE capture file write` | The capture-file write is skipped when CUDA is rejected. |
| `launcher-arg: --device cpu, no CUDA -> exit 0` | A CPU run proceeds with no CUDA and no rejection. |
| `launcher-arg: --device cpu propagates unchanged` | `--device cpu` is seen by `run_pilot`. |
| `bash-launcher: fake OK process exit code 0 propagates` | `<out>.exit_status` file records the underlying Python exit code (when bash is functional on the host; otherwise skipped with INFO). |
| `bash-launcher: stderr-without-error does not corrupt exit-status file` | The per-stream stderr/stdout file split (Section III fix N14) survives a stderr line that LOOKS like an error but exits 0. |
| `bash-launcher: fake FAIL process exit code 7 propagates` | A non-zero Python exit reaches `<out>.exit_status` unchanged. |
| `bash-launcher: exit_status file reflects underlying Python exit code (7)` | The bash launcher's recorded exit status matches the Python exit code. |
| `bash-launcher: stderr-misnomer (looks-like-error text) still exits 0 and writes exit_status=0` | A stderr line that LOOKS like an error but exits 0 is still treated as success. |
| `bash-launcher: re-running into non-empty OUTDIR aborts with code 127` | The fresh-directory refusal check is enforced before launch. |

The bash-launcher tests are skipped with INFO when `bash` is present but the
host's bash implementation is non-functional (e.g. WSL interop with
broken systemd init). The test runner records this as informational,
not as a failure.

## T. Inventory at the time of THIS (SECTION IV) audit

| Item | Status |
|------|--------|
| Active protocol v3 | `PROTOCOL.md` (unchanged in SECTION IV) |
| Active decision doc v3 | `GO_NO_GO.md` (unchanged in SECTION IV) |
| Active orchestrator | `stepsize_diag/pilot.py` (added `_resolve_device_strict`) |
| Active launchers | `run_pilot.ps1`, `run_pilot.sh` (audit-only; no edits required) |
| Smoke launchers | `run_smoke.ps1`, `run_smoke.sh` (NOT modified per audit scope) |
| Selftest | `selftest_cpu.py` (+ 22 new audit checks; total 89+22 = 111 PASS) |
| Hashes adapted | `ADAPTED_HASHES.json` regenerated via `scripts/update_adapted_hashes.py` |
| Pilot outcomes observed | **none** (this audit is CPU synthetic only) |



# SECTION V: implementation repairs (2026-10-04, post-SECTION-IV source review)

Four source-confirmed defects were repaired. These are IMPLEMENTATION
repairs only: PROTOCOL.md, GO_NO_GO.md, and the frozen decision logic are
UNCHANGED. No pilot training, checkpoint generation, candidate evaluation,
or pilot execution has been performed; verification is CPU-synthetic and
mocked-orchestrator only.

## U. The four repairs

| # | Defect | Repair | Where it lands |
|---|--------|--------|----------------|
| R1 | The distribution ZIP excluded EVERY directory named `data`, which silently dropped `reference_code/data/{__init__,datasets,partition}.py` and broke the bundled reference snapshot's imports after extraction. | The ONLY dataset exclusion is now the ROOT dataset storage directory `data/` (the CIFAR payload). `reference_code/data/` and every other directory named `data` are included. The finished ZIP is extracted into a clean temporary directory and its imports + snapshot hashes are verified there (`scripts/make_distribution_zip.py --verify`). | `scripts/make_distribution_zip.py`. |
| R2 | `_process_source` computed the per-source best selection/evaluation gamma as `_argmin_excluding_invalid(...) or GAMMA_ONE`. Because `0.0` is falsy, a legitimately winning gamma=0.0 was silently converted to 1.0; an empty valid set was silently substituted with 1.0. | Explicit `None` handling: gamma=0.0 is preserved when it wins; an empty valid set raises `IncompleteMeasurementError` under `feasibility_strict` (consistent with the STAGE 3 contract) and falls back to `GAMMA_ONE` with an explicit warning only when `feasibility_strict=False`. Tests exercise the actual `_process_source()` path for zero-winning selection AND evaluation losses (split-specific winners: selection 0.0 / evaluation 1.5 in the same call), the all-invalid strict raise, and the non-strict documented fallback. | `stepsize_diag/pilot.py:_process_source`; `selftest_cpu.py` section 37d. |
| R3 | Feature construction rebuilt `counts_by_client` from payload `sample_counts` with a silent count=1 fallback, so the feature-path cohort aggregate weights could silently diverge from the real FedAvg weights. | Retained sample counts are persisted INSIDE every variability payload AND the canonical STAGE 0a `counts_by_client` is passed explicitly into feature construction. The count=1 fallback is removed; a missing payload count, a client missing from the canonical map, or a payload/canonical disagreement raises `VariabilityInventoryError`. `_weights_from_counts` now raises on a missing client instead of propagating a bare `KeyError`. Feature-path aggregate weights are verified equal to the real FedAvg `participant_weights` under UNEQUAL counts, both by direct unit test and on the mocked-orchestrator artifacts. | `stepsize_diag/pilot.py` (`_process_source` payload, `_build_feature_rows_per_phase`), `stepsize_diag/pilot_variability.py:_weights_from_counts`; `selftest_cpu.py` sections 34b, 37b, 37c. |
| R4 | Parameter-name lists were obtained from guessed names / state-dict-key inference (`_param_names_or_default`, `_param_names_from_payloads_or_default`). | One canonical list is computed ONCE from `model.named_parameters()` filtered by `requires_grad` (`_param_names_for_model_ctor`) and that exact list is passed through every variability and feature helper (`_process_source`, `compute_repetition_summaries`, `_build_feature_rows_per_phase`, per-example statistics). The guessed-name and payload/state-dict-inference paths are deleted (no aliases remain). Verified: classifier parameters and all trainable biases are included, requires_grad=False parameters are excluded, and BatchNorm buffers (`running_mean`, `running_var`, `num_batches_tracked`) are excluded, on both a TinyBN fixture and the bundled SmallCNN. | `stepsize_diag/pilot.py`; `selftest_cpu.py` section 37a. |

## V. Test-harness repair discovered during verification

The mocked-orchestrator test patched `stepsize_diag.data.*` but `pilot.py`
binds the data functions into its OWN namespace (`from .data import ...`),
so the mock never reached the orchestrator: the run silently used the real
CIFAR partition and the synthetic unequal counts never flowed. The mock now
patches `pilot_mod.*` as well; the mocked run is fully synthetic with the
declared unequal counts `[120, 80, 200, 60, 100, 90]`, and the mock states
use the bundled SmallCNN's ACTUAL state_dict keys (previously synthetic
`'w'`/`'b'` keys). Per-seed update scales bake in gamma=0-winning cells
(seeds 201/202) and gamma=1.5-winning cells (seed 203). The test-36
bash-launcher block's undefined names (`fake_ok`, `fake_fail`,
`fake_stderr_warn`, `out_a/b/c`, `_read_status`) are now defined (latent
NameError if a functional bash is present).

## W. Verification performed (CPU only; no training, no candidate
evaluation, no pilot launch)

- `selftest_cpu.py` full pass, including the mocked end-to-end
  `run_pilot()` orchestration with SmallCNN actual keys, unequal client
  counts, and gamma=0-winning cases; assertions prove gamma=0.0 survives
  selection, persistence (`gamma_dev_global.json`), rule rows, and
  `pilot_decision.json`.
- The distribution ZIP was built, extracted into a clean temporary
  directory, and its imports + snapshot hashes verified there.

## X. Inventory at the time of THIS (SECTION V) repair

| Item | Status |
|------|--------|
| Active protocol v3 | `PROTOCOL.md` (UNCHANGED) |
| Active decision doc v3 | `GO_NO_GO.md` (UNCHANGED) |
| Orchestrator | `stepsize_diag/pilot.py` (R2, R3, R4 repairs) |
| Variability | `stepsize_diag/pilot_variability.py` (R3 missing-count raise) |
| Selftest | `selftest_cpu.py` (sections 34b + 37a-d; mock namespace repair; SmallCNN actual keys) |
| Packaging | `scripts/make_distribution_zip.py` (R1) |
| Hashes adapted | `ADAPTED_HASHES.json` regenerated via `scripts/update_adapted_hashes.py` |
| Pilot outcomes observed | **none** (CPU synthetic + mocked orchestration only) |

# SECTION VI: server self-test repairs (2026-10-04, second repair pass)

Three failures from the server self-test log were fixed. No experiments were
run. Test defects and production defects are distinguished explicitly below;
the frozen protocol documents (PROTOCOL.md, GO_NO_GO.md) are UNCHANGED.

## Y. Repairs

| # | Failure | Classification | Repair |
|---|---------|----------------|--------|
| R5 | `stepsize_diag/verify.py:check_per_example_eval_restores_state` contained a function-level `import pytest` (annotated "not used"). On the server pytest is not installed, so the import raised `ModuleNotFoundError` when the check executed. | **Production defect** (unused dependency executed at runtime in a shipped module). | The line is removed. No behavior change where pytest was present. |
| T1 | Mocked launcher happy-path test (`launcher-arg: --pilot happy-path exit code == 0`) relied on the HOST's real CUDA: with GPUs hidden (`CUDA_VISIBLE_DEVICES=""`), `torch.cuda.is_available()` was False and the REAL `_resolve_device_strict` correctly rejected `--device cuda`. | **Test defect** (the strict resolver behaved correctly by design). | The test shim now mocks CUDA availability INSIDE the subprocess (`torch.cuda.is_available = lambda: True` for the `with-cuda` branch) before invoking the entry point. The separate no-CUDA rejection tests (which mock `is_available = False`) and the production `_resolve_device_strict` are UNCHANGED. |
| T2 | Bash-launcher fresh-directory test (`bash-launcher: re-running into non-empty OUTDIR aborts with code 127`) re-ran into `out_a`, but the launcher writes its logs BESIDE the output dir, so `out_a` stayed EMPTY and the fresh-check legitimately did not trigger. | **Test defect** (the launcher is correct; verified directly: with an existing `results/` artifact it returns 127 with `FATAL: OutputDir not fresh` BEFORE invoking Python). | The test now explicitly creates an existing result artifact (`out_a/results/pilot_decision.json`) in the output directory and re-runs with the FAIL fake-python: rc==127 proves the abort happened before Python (a wrongly-proceeding launcher would exit 7); a second assertion checks the refusal message names the blocking artifact. The launcher itself is UNCHANGED. |

## Z. Verification performed (CPUs only; GPUs hidden; no experiments)

- Full `selftest_cpu.py` run with `CUDA_VISIBLE_DEVICES=""` (GPUs hidden):
  all checks PASS.
- Launcher verified directly under a functional bash: existing-artifact
  output dir -> exit 127 pre-Python; fresh dir -> fake-python runs, exit 0,
  `<out>.exit_status` records 0.
- `ADAPTED_HASHES.json` regenerated via `scripts/update_adapted_hashes.py`.
- Distribution ZIP rebuilt; extracted to a clean temporary directory;
  imports + snapshot hashes + the full self-test verified in the extracted
  tree with GPUs hidden.

## AA. Inventory at the time of THIS (SECTION VI) repair

| Item | Status |
|------|--------|
| Active protocol v3 | `PROTOCOL.md` (UNCHANGED) |
| Active decision doc v3 | `GO_NO_GO.md` (UNCHANGED) |
| Production code changed | `stepsize_diag/verify.py` (R5 only) |
| Test code changed | `selftest_cpu.py` (T1, T2) |
| Launchers | `run_pilot.sh` / `run_pilot.ps1` (UNCHANGED; verified correct) |
| Hashes adapted | `ADAPTED_HASHES.json` regenerated |
| Pilot outcomes observed | **none** |

# SECTION VII: exploratory-pipeline repair after the completed server run
(2026-10-04)

The completed server run `stepsize_diag_pilot_20261004_175620` produced
`fit_status="no_cells"` with zero development/held-out cells for BOTH
exploratory predictors, and `report.md` incorrectly stated "two predictors
fit" while displaying no variability records. The audit (see
`audit_stepsize_diag_pilot_20261004_175620/AUDIT.md`) traced all 81
candidate cells (27 dev + 54 held-out) to the TARGET JOIN in
`decide_exploratory_signal`, plus one feature-scoping defect and two
reporting defects. All repairs below are IMPLEMENTATION repairs: frozen
features, targets, splits, and thresholds are UNCHANGED; PROTOCOL.md and
GO_NO_GO.md are UNCHANGED. No experiment was rerun; corrected analysis is
recomputed solely from saved artifacts into a separate directory.

## AB. Repairs

| # | Defect | Classification | Repair |
|---|--------|----------------|--------|
| R6 | Dev seed never received `fixed_global_gamma_dev` / `fixed_phase_gamma_dev` split-metric keys (dev processed before the dev-side freeze), so the exploratory target join found 0/27 dev cells. | Production implementation defect | STAGE 3b in `pilot.py` (`_label_dev_fixed_policy_rows`) labels the ALREADY-EVALUATED dev grid rows at the frozen gammas (the frozen gammas are grid points). No re-evaluation. `split_metric_keys_dev_only.json` / `dev_keys_sha` intentionally still capture the pre-labeling freeze state. |
| R7 | `decide_exploratory_signal` looked up held-out targets under the DEV seed, so 0/54 held-out cells joined. | Production implementation defect | Held-out units are built PER HELD-OUT SEED. |
| R8 | Held-out feature rows mixed seeds 202/203 under a dev-seed label (grouped by round only). | Production implementation defect | `_build_feature_rows_per_phase` keys rows by each payload's own `(seed, round)`; `common_held_keys` is now the (round, source) cell-identity intersection. |
| R9 | `report.md` said "two predictors fit" unconditionally and showed no variability records (per-checkpoint payload is a dict, not a list). | Reporting defect | The report renders actual per-predictor `fit_status` and cell counts, and reads `per_checkpoint` from the dict payload. |

## AC. Independence of the decisive outcomes

`decide_adaptive_headroom` and `decide_claude_comparator` consume ONLY
held-out-seed keys for the fixed/validation/claude policies (present in the
run); the inventory gate checks held-out cells only; `decide_overall` never
reads the exploratory verdict. The measured negative outcomes of
`adaptive_headroom` and `claude_comparator` therefore STAND; the
exploratory probe is marked a FAILED ANALYSIS (no scientific content) and
the variability-signal hypothesis remains UNANSWERED until recomputation.

## AD. Artifact-only recomputation

`scripts/recompute_exploratory_from_artifacts.py` rebuilds the corrected
exploratory analysis from `config.json`, `partition.json`,
`results/split_metric_keys.json`, `results/aggregate_evaluation.json`,
`results/gamma_dev_global.json`, `results/gamma_dev_phase_<round>.json`,
`results/variability_dev.json`, `results/variability_held_out.json` —
writing into a SEPARATE `<run_dir>_audit_corrected/` directory with
`audit_trace.json` (per-stage cell counts and exclusion reasons). Nothing
required is missing from the persisted artifact set; if any file is absent
the script lists it and exits non-zero without fabricating values.
Self-test section 38 verifies the recompute end-to-end on a legacy-shaped
artifact copy (defective keys stripped) with originals untouched.

## AE. Inventory at the time of THIS (SECTION VII) repair

| Item | Status |
|------|--------|
| Active protocol v3 | `PROTOCOL.md` (UNCHANGED) |
| Active decision doc v3 | `GO_NO_GO.md` (UNCHANGED) |
| Orchestrator | `stepsize_diag/pilot.py` (R6, R8, R9) |
| Decision module | `stepsize_diag/pilot_decision.py` (R7, R8) |
| Recompute script | `scripts/recompute_exploratory_from_artifacts.py` (AD) |
| Selftest | `selftest_cpu.py` (section 34b exploratory assertions, section 38 legacy recompute, (seed, round) keying migration in sections 17/32/37c; plus a bash-probe robustness guard found during this round's verification: a hung/unstartable bash sanity probe (TimeoutExpired/OSError) now skips the bash-launcher block with INFO instead of crashing) |
| Audit report | `audit_stepsize_diag_pilot_20261004_175620/AUDIT.md` |
| Hashes adapted | `ADAPTED_HASHES.json` regenerated |
| New experiments run | **none** |

# SECTION VIII: per-predictor design-matrix repair (2026-10-04)

`decide_exploratory_signal.fit_one()` fit BOTH named predictors on the FULL
outer feature matrix: "phase_only" was not phase-only, and the two
predictors were numerically identical (identical held MAE
0.020016339820771462 in the SECTION VII correction). This is an
implementation defect; frozen features, targets, splits, and thresholds are
UNCHANGED.

## AF. Repair

- Each predictor now builds its own design matrix explicitly:
  `phase_only` = exactly [phase_50, phase_100];
  `phase_plus_variability` = those indicators plus the frozen variability
  features (`design_columns` is persisted per predictor).
- Dev-only standardization and the OLS fit are performed separately per
  matrix.
- A runtime contract asserts matrix width, coefficient counts,
  feature-order length, means, scales, and the constant-feature mask all
  agree with the predictor's own design (raises RuntimeError on violation).
- Predictor records now include held-out MAE/MSE broken down BY TRAINING
  SEED and BY CHECKPOINT ROUND, with the clustering caveat (the 3
  repetitions within a (cohort, seed, round) cell are not independent).
- Incidental: `n_held_cells_used_total` was an element count
  (`X_held.size`); now a row count (`X_held.shape[0]`).

## AG. Verification

- Selftest section 39: synthetic target depending ONLY on a variability
  feature with opposing signs inside every phase — phase_only held MAE
  2.96, phase_plus_variability held MAE ~0; the predictors differ for the
  intended reason (widths and fitted coefficients), not merely by label.
- Full GPU-hidden selftest: 225/225 PASS.
- Recomputed from the saved artifacts of run
  `stepsize_diag_pilot_20261004_175620` into a NEW directory
  `results/stepsize_diag_pilot_20261004_175620_audit_corrected_v2/`
  (previous correction pass and all original outputs preserved):
  phase_only held MAE 0.009036 / MSE 0.0003136;
  phase_plus_variability held MAE 0.020016 / MSE 0.0006962.
  Per instruction, no claim about the variability hypothesis is made until
  the corrected comparison is inspected.

## AH. Inventory at the time of THIS (SECTION VIII) repair

| Item | Status |
|------|--------|
| Active protocol v3 | `PROTOCOL.md` (UNCHANGED) |
| Active decision doc v3 | `GO_NO_GO.md` (UNCHANGED) |
| Decision module | `stepsize_diag/pilot_decision.py` (fit_one per-predictor design matrices) |
| Recompute script | `scripts/recompute_exploratory_from_artifacts.py` (by-seed/by-round MAE/MSE report) |
| Selftest | `selftest_cpu.py` (section 39) |
| Corrected analysis | `results/stepsize_diag_pilot_20261004_175620_audit_corrected_v2/` |
| Hashes adapted | `ADAPTED_HASHES.json` regenerated |
| New experiments run | **none** |

# SECTION IX: pilot closure (2026-10-04)

The pilot is CLOSED. No training, no new candidate evaluations, no
additional experiments were run in this section. `PROTOCOL.md` and
`GO_NO_GO.md` are UNCHANGED.

## AI. Finalization actions

- Revised buffer-policy analysis written to the FRESH directory
  `analysis_buffer_policy_v2/` (the earlier `analysis_buffer_policy/`
  is preserved unchanged): dev seed 201 and held-out seeds 202/203 are
  separated in every summary; sign counts are reported for the 27
  all-seed cohort cells and the 18 held-out cohort cells; the
  interaction claim is withdrawn (both γ=1-reference comparisons share
  the same validation-selected candidate, so their difference reflects
  the different reference losses); the conclusion is limited to frozen
  checkpoint buffers at γ=1 in this setting; the report records input
  artifact hashes and the exact command.
- `PILOT_CLOSURE.md` written: the final decision record. All quoted
  numbers were verified programmatically against the saved artifacts
  (pilot_decision.json, the corrected exploratory JSONs, and
  `analysis_buffer_policy_v2/cohort_summary.csv`).
- Research recommendation recorded: the tested raw shrinkage rule and
  the frozen variability-predictor formulation are PARKED for this
  CIFAR-100 setting; step-size control in general is NOT claimed
  ineffective. The harness, checkpoints, partition, paired evaluations,
  and tuned fixed controls (γ_global=0.5; γ_phase=0.75/0.5/0.5) are
  preserved. The evaluation-only frozen-buffer grid extension is NOT
  launched, and no further tuning against held-out seeds 202/203 is
  performed.

## AJ. Provenance

Original run outputs and both exploratory correction directories are
unchanged. No data deleted; no scientific outputs regenerated
unnecessarily. New/changed files in this section:
`PILOT_CLOSURE.md`, `analysis_buffer_policy_v2/` (4 files),
`PRE_EXECUTION_AMENDMENT.md` (this section), `ADAPTED_HASHES.json`
(regenerated), and the distribution ZIP (rebuilt).

# SECTION X: post-closure feasibility audit (2026-10-04)

`COHORT_BIAS_FEASIBILITY.md` audits whether the saved artifacts support
testing a possible follow-up hypothesis (cohort label composition vs the
classifier bias added during a round). Source + CPU checkpoint inspection
only; no experiments; originals unchanged. Verdict: FEASIBLE for the
trajectory cohorts (9 exact before/after matched pairs; per-client states,
weights, counts, class counts, and holdout indices all persisted and
verified). The counterfactual-cohort client states were not persisted; the
smallest additional measurement (deterministic local retraining from saved
pre-round states) is identified but NOT launched. The pilot closure
(SECTION IX) is unaffected.

# SECTION XI: cohort-bias exploratory screen (2026-10-04)

Following the feasibility audit (SECTION X), the exploratory paired
analysis of the nine saved trajectory checkpoints was implemented and run
(`analysis_cohort_bias/`): all nine before/after pairs validated
(seed/round identity, cohort IDs, weights, counts; post-aggregation state
reconstructed from saved client states with every entry compared, max
|diff| = 0). Selection split only; evaluation split and test set unread;
forward evaluation restores/asserts state, buffers, mode, and RNG.
Associations between cohort label exposure (add-0.5 smoothed log ratios,
fixed a priori) and incremental classifier changes are positive in all
nine checkpoints for parameter and functional measures (36/36 positive;
35/36 above all 200 deterministic label permutations). Controls are
exploratory, not causal. Recommendation recorded in
`analysis_cohort_bias/report.md`: the evidence justifies DESIGNING a
correction experiment under a new frozen protocol (fresh seeds and
partition), not implementing one. No training, no new candidate
evaluation, no test-set access; pilot closure (SECTION IX) and frozen
documents are unaffected.

# SECTION XII: cohort-bias screen v2 (2026-10-04)

Review of `analysis_cohort_bias/` found that v1 measured own-class logits
(logit_c on examples of true class c), which indicate learning on those
examples rather than a broader class-wide output shift. v2
(`analysis_cohort_bias_v2/`, fresh directory; v1 preserved) keeps the
own-class measurements correctly labeled and adds: per-example
class-centered logits; the mean centered-logit change per output class
across ALL selection examples (equal weight per true class) and its
non-own-class restriction; and the correct-class margin change
(z_c - logsumexp(others)). All computed for the exact full-state
before/after models and the head-only intervention. Reporting repairs:
try/finally restoration with asserted state/buffer/mode/RNG preservation;
output-directory freshness enforced; selection class-count balance (25 per
class) and selection/evaluation disjointness asserted; script and input
hashes recorded; permutation exceedances reported as counts; other-cohort
controls described as dependent comparisons without a conservativeness
claim; conclusions are conditional on the measured results (no hardcoded
findings or recommendation). Measured: broad/non-own associations mirror
the own-class ones in 8/9 checkpoints (201|100 weaker on full state,
0.21-0.24, 2-5 exceedances of 200 permutations); margin associations 9/9;
mean margin change +0.0450 with mixed signs per checkpoint. Under the
pre-declared rule in the script, the pattern is mixed between own-class
learning and broader output preference, so no design decision is taken
from this screen alone. No training, no new cohorts, no correction
fitting, no test-set access; evaluation split unread; frozen documents
unchanged.

# SECTION XIII: cohort-correction utility test (2026-10-04)

After the v2 screen (SECTION XII), a small exploratory utility test of a
cohort-derived logit correction was run on the nine saved trajectory
checkpoints under a plan frozen BEFORE evaluation
(`analysis_cohort_correction/PLAN.md`): z_corrected = z_after - tau * s_c
with the v2 add-0.5 smoothed centered log-prior-ratio exposure; fixed grid
tau in {-0.10, -0.05, -0.025, 0, 0.025, 0.05, 0.10} (negative values are
an opposite-direction control); ONE common tau selected by mean
selection-split NLL across seed 201's three checkpoints (ties toward
zero); applied frozen to seeds 202/203. No training, no new cohorts, no
test-set access, no in-loop model changes. Measured: tau* = 0.10 (grid
edge; recorded as a measured fact, grid not expanded); held-out mean NLL
improvement vs tau=0: seed 202 +0.0306, seed 203 +0.0237; shuffled-class
exposure controls (n=25) mean -0.0005 with 0/25 reaching the actual
improvement; other-cohort exposure controls (dependent comparisons)
+0.0102 / +0.0151, below the actual +0.0271. The held-out evaluation
split was used in the earlier step-size pilot, so all outcomes are labeled
exploratory, not fresh confirmation. Under the pre-declared
recommendation rule in PLAN.md (all three conditions held), the recorded
recommendation is a FRESH-SEED CONFIRMATION (new training seeds, new
partition, pre-registered decision rule), not adoption of the correction.
The rule was not changed after seeing results; no grid expansion; no
training launched; frozen documents unchanged.

# SECTION XIV: fresh-seed confirmation prepared, not launched (2026-10-04)

Following the utility test (SECTION XIII), a fresh-seed confirmation was
PREPARED under `confirmation_cohort_correction/` and is NOT launched:
- `PROTOCOL_CONFIRMATION.md` freezes the design before execution: verified
  unused training seeds 301/302/303 and partition seeds 43/44, fresh split
  seed 1009; SmallCNN without AIB; 10 clients, 5 participants, alpha=0.3,
  E=2, 100 rounds; exact before/after states and cohort metadata saved at
  rounds 5/50/100; calibration and evaluation splits reserved before
  partitioning; test set untouched. The correction is preserved exactly
  (centered add-0.5 log-prior ratio, tau=0.10, post-hoc, no feedback into
  training). Arms: uncorrected, actual tau=+0.10, opposite tau=-0.10, 25
  deterministic shuffled-exposure controls, a predefined wrong-cohort
  control (other partition, same round), and scalar temperature
  calibration fitted only on the calibration split. Primary endpoint
  (round-100 evaluation NLL improvement), meaningful-effect threshold
  (+0.01), accuracy guardrail (-0.002), control comparisons, and the
  GO/NO-GO rule are frozen; results are reported by partition and training
  seed with checkpoints as repeated observations and round-100 reported
  separately from the three-checkpoint average.
- `run_confirmation.py` implements the protocol; `--selftest` passes 12/12
  implementation checks (correction arithmetic, tau=0 exactness, offset
  invariance, stable log-sum-exp, finite-value checks, temperature tie
  rule, shuffle determinism, GO/NO-GO logic). `--run` is prepared but NOT
  launched; all `--run` imports resolve.
- `COMMANDS.md` records exact Windows/Linux commands and the compute
  estimate (~80-90 min on the pilot server, derived from the measured
  pilot timing; the checkpoint-generation timer fields were stubs, so the
  per-seed training estimate is derived, not directly instrumented).
- Deployment requirements are documented in the protocol (server-side
  population/cohort class counts, ~8 KB one-time communication, label-
  distribution disclosure implications; NO privacy guarantees claimed).
Frozen documents (PROTOCOL.md, GO_NO_GO.md) are unchanged.

# SECTION XV: confirmation-script audit repairs (2026-10-04)

`confirmation_cohort_correction/run_confirmation.py` was audited against
`PROTOCOL_CONFIRMATION.md` and repaired. Implementation repairs only; tau,
seeds, arms, and the scientific decision criteria are UNCHANGED; the
protocol document itself changed only in its runtime-documentation section
(RTX A6000 correction, download vs cached-data separation, estimate
labeled approximate) — no scientific content changed. No training or
candidate evaluation was launched.

Confirmed defects repaired:
- `shared_model` was referenced by `_forward` but never created in
  `run()`; it is now created before any `_forward()` call.
- The arm-evaluation loop referenced an undefined `y`; it now uses
  `y_eva`.
- `--data-root` is configurable, defaulting to the bundle's data/
  directory; the empty-string dataset root is removed and the commands
  were updated.

Safeguards added:
- `verify_helper_contracts()` checks the ACTUAL signatures of every
  imported helper (it caught the mocked test's own signature drift);
  `generate_checkpoint`'s return type/count/rounds are verified at call
  time.
- `verify_inventory()` gates the verdict: exactly 6 unique
  (partition, seed) runs x rounds 5/50/100 x 5 named arms x 25 shuffled
  controls; missing/duplicate/non-finite measurements are rejected; an
  empty inventory can never return GO.
- Persisted additionally: protocol/script hashes, environment versions,
  actual split indices, retained client indices/counts, per-checkpoint
  cohort exposures, and individual shuffled-control results.
- The bundled reference snapshot is hash-validated before training;
  CUDA RNG preservation is asserted in addition to CPU RNG; the
  identical-seed training check goes through the existing diagnostic
  verification path (`stepsize_diag.verify.check_identical_seed_retrain`).
- Incremental persistence: validation records, temperatures, exposures,
  per-example losses (flushed) and results_partial.json are written as
  each checkpoint completes; fresh-output refusal preserved.

Testing (no training, no download): `--selftest` 20/20 PASS;
`test_run_confirmation_mocked.py` 25/25 PASS — mocked end-to-end
orchestration reaches result writing and verdict computation with
synthetic data and structurally correct fake checkpoints; failure tests
cover missing checkpoints (loud abort, no verdict), mid-course failure
(incremental artifacts survive), non-fresh output directories,
unavailable CUDA, wrong-cohort mapping, and calibration/evaluation label
alignment.

# SECTION XVI: confirmation-script audit repairs, second pass (2026-10-04)

Three confirmed defects in `confirmation_cohort_correction/run_confirmation.py`
were repaired. Implementation repairs only; tau, seeds, arms, splits,
endpoints, and decision thresholds are UNCHANGED. No training or candidate
evaluation was launched.

- CUDA/CPU checkpoint reconstruction mismatch: the reference
  average_states() returns CPU tensors, so checkpoint states are now loaded
  with map_location="cpu" (`_load_checkpoint_cpu`, asserts CPU residence)
  for reconstruction/validation; the evaluation model stays on the
  requested device (load_state_dict copies). The reference aggregation is
  unchanged.
- Inventory rejection after CSV closure: the per-example CSV is now closed
  in a try/finally (success AND failure paths), and the inventory gate sits
  inside it, so a rejection reports the intended INVENTORY INCOMPLETE error
  and partial artifacts survive; _flush_partial never flushes a closed
  file.
- Non-finite accuracy guardrail: the inventory gate now validates every
  arm's NLL, NLL improvement, AND accuracy change for presence and
  finiteness, and a new endpoint validation rejects missing/NaN/infinite
  endpoint values (including derived control margins) before any verdict.

Regression checks: selftest now 28/28 (adds NaN-accuracy, infinite-NLL,
missing-arm-measurement rejections; endpoint validation; CPU-load residence;
exact reference reconstruction on CPU; CUDA-saved-tensors-land-on-CPU).
The mocked orchestration test now records torch.load map_location (all
"cpu"), exercises the inventory-rejection resource path end-to-end, and
runs the full mocked pipeline on CUDA (CPU-load reconstruction + CUDA
evaluation). New `--preflight` mode: a data-free CUDA hardware check using
the actual reference aggregation and actual forward path (PASSED on this
machine's RTX 3060 Laptop GPU); on a machine without CUDA it reports the
check as UNVERIFIED and exits 0.

# SECTION XVII: EuroSAT replication prepared, not launched (2026-10-05)

After the CIFAR-100 cohort-correction confirmation passed its frozen gate,
a separate EuroSAT replication was PREPARED under `replication_eurosat/`
and is NOT launched. No training, candidate evaluation, or dataset
download was performed.

- `EUROSAT_COMPATIBILITY_AUDIT.md`: what the bundle already supports
  (reference get_eurosat loader, EuroSAT normalization constants and
  augmentation recipe, input-size-agnostic SmallCNN, generic
  partitioner/aggregation) and what was missing (raw-array loader,
  functional EuroSAT transforms, EuroSAT local-training binding, config
  entry) — reported before reconstruction, never silently filled with
  CIFAR settings.
- Additive, backward-compatible change to verified shared code:
  `generate_checkpoint` and `check_identical_seed_retrain` accept an
  optional `train_client_fn` (None = the verified CIFAR path,
  bit-identical; the main self-test re-passes 225/225).
- `PROTOCOL_REPLICATION.md` frozen before execution: EuroSAT RGB (10
  classes, 64x64) from the actual reference implementation; 10 clients,
  5 participants, alpha=0.3, E=2 retained; verified-unused seeds
  (training 401/402/403, partitions 45/46, split 1011); disjoint
  stratified calibration/evaluation splits reserved before partitioning
  with retained counts recomputed; the same six paired arms; frozen
  terminal-checkpoint (round-100) primary endpoint and GO/NO-GO gate; the
  CIFAR thresholds (NLL +0.01, accuracy -0.002) retained with a written
  pre-measurement justification; the temperature arm reported separately,
  never in the gate. Image-level splitting limitations declared; no
  geographic generalization claimed.
- The intervention is preserved exactly (centered add-0.5 log-prior
  ratio, tau=0.10, post-hoc only).
- Verification without experiments: imports, 19/19 CPU synthetic checks,
  30/30 mocked orchestration checks (incl. failure tests and a full
  mocked CUDA run), data-free CUDA preflight of the actual EuroSAT
  evaluation path (PASSED on this machine). No real data was read.
- `READINESS.md` lists unresolved blockers (EuroSAT dataset not present;
  approximate runtime estimate; target-server preflight pending).

Frozen documents (PROTOCOL.md, GO_NO_GO.md, the CIFAR confirmation
protocol and results) are unchanged; no tuning against the existing
confirmation results was performed.

# SECTION XVIII: EuroSAT replication protocol v2 amendment (2026-10-05)

A documented pre-execution amendment to the EuroSAT replication; v1 is
archived unchanged as
`replication_eurosat/PROTOCOL_REPLICATION_v1_2026-10-05.md`. The CIFAR
confirmation is untouched. No downloads, training, or real candidate
evaluation were performed.

- New paired controls (same models; reported separately from the
  UNCHANGED GO gate): previous-saved-checkpoint-round cohort exposure at
  tau=0.10 (undefined at round 5 by construction); history-averaged
  exposure at tau=0.10 (plain cumulative mean over rounds r' <= r,
  initialized from round 1, no EMA coefficient); and a zero-centered
  L2-regularized classwise additive bias fitted only on the calibration
  split (objective mean NLL + 0.01||b||^2, L-BFGS-B from b=0, gtol=1e-8,
  maxiter=200, zero-centered post-fit). Explicit interpretation rules
  (R1 recency, R2 persistence, R3 calibration sufficiency) are frozen in
  the protocol before any measurements.
- Wording corrections: +0.01 nats is an absolute continuity threshold,
  NOT scale-free; recovered_fedavg.json is a CIFAR-derived configuration
  transferred to EuroSAT unless an actual EuroSAT reference config is
  recovered (read-only server search is part of readiness); threshold
  justification no longer relies on observed CIFAR gains.
- Cohort IDs, retained sample counts, and cohort class-count vectors are
  now persisted for EVERY training round (additive `round_observer`
  parameter on `generate_checkpoint`; default None keeps the CIFAR paths
  bit-identical; main self-test re-passes 225/225).
- Functional EuroSAT transforms were checked against the reference
  implementation (operation order, parameters, nearest interpolation,
  fill 0, per-image independent randomness) via a source-text structural
  check; intentional differences (tensor-level float ops vs PIL 8-bit
  quantization; batch-order RNG consumption under the same determinism
  policy) are documented in `eurosat_data.py`.
- `replication_eurosat/server_locate_eurosat.sh`: read-only server
  commands to locate existing EuroSAT data (ImageFolder + train.csv) and
  historical EuroSAT configurations; the dataset's local absence is not
  treated as evidence of server absence.

Verification re-run (synthetic/mock only): replication --selftest 31/31;
mocked orchestration 36/36 (per-round persistence, previous-round and
history mappings from persisted records, fitted-bias metadata,
round-aware inventory, all failure modes); confirmation --selftest
28/28; main selftest 225/225.

# SECTION XIX: EuroSAT replication protocol v3 amendment (2026-10-05)

A documented pre-execution amendment; v1 and v2 are archived unchanged
(`PROTOCOL_REPLICATION_v1_2026-10-05.md`,
`PROTOCOL_REPLICATION_v2_2026-10-05.md`). No downloads, training, or real
candidate evaluation were performed.

- `previous_round_tau+0.10` corrected: the control now uses the exposure
  of TRAINING ROUND r-1 from the persisted per-round cohort records
  (checkpoints 5/50/100 -> rounds 4/49/99); required at ALL three
  checkpoints; missing history aborts loudly and is never substituted;
  the previous saved checkpoint is never substituted.
- Interpretation rules revised (protocol section 4b; the GO gate is
  UNCHANGED): paired NLL AND accuracy differences for actual vs every
  control by partition/seed/checkpoint; the ±0.002-nat band is symmetric
  and labeled a descriptive practical band (not statistical
  equivalence); categories are consistent advantage / consistent
  disadvantage / within-band / mixed-inconclusive; round 100 is the
  explicit main-summary checkpoint with rounds 5/50 descriptive.
  fitted_bias_calib is documented as a comparator requiring labeled
  calibration data, NOT an upper bound. No rule claims that calibration
  "explains" the effect, that history proves persistence, or that failed
  superiority proves added value.
- Fitted-bias optimization status and parameter finiteness are checked;
  non-finite parameters abort loudly; non-convergence is recorded
  (`converged: false`) and excluded from the R3 comparison.

Kept fixed: the cumulative mean-of-exposures definition, tau=0.10,
training settings, seeds, arms, thresholds, and the original GO gate.

Verification re-run (synthetic/mock only): replication --selftest 32/32;
mocked orchestration 38/38 (exact previous-round mappings, missing-history
rejection, all-arms-at-every-checkpoint inventory, plus all prior checks).

# SECTION XX: EuroSAT replication protocol v4 amendment (2026-10-05)

A documented pre-execution amendment; v1, v2, and v3 are archived
unchanged. No downloads, training, or real candidate evaluation were
performed. Seeds, tau, training settings, split fractions, endpoints, and
decision criteria are UNCHANGED from v3.

- Server dataset layout CONFIRMED:
  `/home/whu/disk1/infoopt_dl_server/datasets/eurosat`; `train.csv`
  header `,Filename,Label,ClassName` (18,900 data rows; the leading
  unnamed column is a saved index, NOT the label); `label_map.json`
  authoritative (AnnualCrop=0 ... SeaLake=9) and cross-checked against
  the ImageFolder ordering. Only train.csv is used; test.csv and any
  validation.csv are never read or added to the pool.
- `eurosat_data.py:load_eurosat_raw` rewritten: columns parsed BY NAME;
  full row validation before training (unique filenames, existing files,
  Label/ClassName agreement, path-directory agreement, all ten classes,
  64x64x3 RGB); unique filenames + asserted split disjointness imply no
  duplicate image across calibration/evaluation/client splits.
- New `--validate-data` mode: data-validation-only command saving counts
  and input hashes (train.csv, label_map.json, image name+size manifest)
  into a fresh output directory; no training, no evaluation.
- Dataset-size provenance and the approximate runtime/storage estimate
  updated from 21,600 to 18,900 images (total ~95-100 min, < 250 MB;
  still approximate).

Verification re-run (synthetic/mock only): replication --selftest 41/41
(including synthetic-layout loader validation and validate-data
end-to-end); mocked orchestration 38/38.

# SECTION XXI: post-hoc classwise analysis of the completed confirmation
(2026-10-05)

A read-only exploratory analysis of the completed CIFAR-100 confirmation
run (`results/confirmation_20261005_060421`) was added under
`analysis_confirmation_posthoc/`. The frozen protocol, outputs, and
verdict are UNCHANGED; no tau tuning, no training, no new candidate
evaluation.

Evidence-based premise correction: the persisted verdict is GO with zero
gate failures and six passing runs (round-100 endpoints reproduce the
quoted established result exactly: mean NLL improvement 0.027907, mean
accuracy improvement 0.003667). There were NO failed runs; the request's
"NO-GO verdict / two failed runs" premise does not match the artifacts, so
the classwise analysis covers all six runs with a descriptive focus on the
two weakest-margin runs (p44_s303, p43_s303). The earlier directory
`cohort_correction_confirmation_20261004_230402` is an incomplete/aborted
attempt (checkpoints only, zero per-example rows) and was not analyzed.

Measured (exploratory): round-100 classwise NLL/recall changes per run
(`classwise_round100.csv`); best shuffled controls reconstructed
deterministically and cross-checked against the persisted
shuffled_improvements (max |diff| recorded in findings.json); correction
direction aligns with classwise prediction bias: exposure vs
over-prediction Pearson +0.47..+0.52 (positive in 6/6 runs, exceeding all
200 deterministic label permutations in 6/6), and the classwise NLL
benefit is anti-correlated with exposure (-0.96..-0.99), i.e.
under-represented classes gain most — the expected direction for a
correction that down-weights over-represented classes. Under the
pre-stated rule in the analysis, the result SUPPORTS a concrete new
hypothesis (cohort composition shifts the classwise prediction
distribution toward over-represented classes; the correction counteracts
it). All findings are labeled exploratory and associational.

# SECTION XXII: post-hoc classwise analysis of the EuroSAT replication
(2026-10-05)

A read-only exploratory analysis of the EuroSAT replication run
(`results/replication_20261005_094928`) was added under
`analysis_eurosat_posthoc/`. The frozen protocol, all outputs, and the
NO-GO verdict are UNCHANGED; no training, no tau tuning, no new candidate
evaluation.

Identity verified from provenance.json: EuroSAT, 10 classes, partitions
45/46, training seeds 401/402/403, tau=0.10, split seed 1011. The frozen
verdict was RECOMPUTED from the persisted per-checkpoint metrics via the
frozen go_no_go rule: NO-GO reproduced exactly, with the same failure
records (p46_s401 control_not_exceeded; p46_s402 nll_below_threshold AND
control_not_exceeded). Integrity: CSV-vs-logits NLL max |diff| 8.9e-16;
deterministically reconstructed shuffled-control improvements match the
persisted ones exactly (max |diff| 0.0); the correction's mean-logit shift
equals -tau*s to 5.2e-15 (tautological, labeled as such).

Measured (exploratory): round-100 classwise NLL/recall changes for all six
runs (`classwise_round100.csv`); failed-run class detail for p46_s401 and
p46_s402 (in both, the best reconstructed shuffled control EXCEEDS the
actual correction's improvement — consistent with the saved gate
failures). Direction analysis separates the mechanical consequence of the
correction from independent evidence: the uncorrected model's
over-prediction bias aligns positively with cohort exposure in 6/6 runs
(Pearson +0.32..+0.94) but exceeds the 200-permutation reference in only
2/6 runs — weak/mixed independent evidence; mean-centered-logit alignment
is sign-mixed. The classwise NLL benefit is anti-correlated with exposure
(-0.87..-0.99), a pattern that is partly mechanical by construction and is
labeled as such (not independent evidence).

# SECTION XXIII: tau-direction diagnostics for the EuroSAT replication
(2026-10-05)

An exploratory diagnostic over the saved logits of
`results/replication_20261005_094928` was added
(`analysis_eurosat_posthoc/analyze_tau_diagnostics.py` ->
`tau_diagnostics.json` / `tau_diagnostics.md`). The frozen protocol,
outputs, and NO-GO verdict are UNCHANGED; no tau search, no new
correction candidates, no training.

Measured: per run, g_c = mean softmax - mean one-hot; the analytic NLL
derivative at tau=0 (-s^T g) verified against central finite differences
(max error 1.1e-9); curvature kappa (mean per-example variance of s under
the softmax); realized vs linear/quadratic NLL change at tau=0.10; the
actual exposure direction is the steepest initial descent direction among
all 25 deterministically reconstructed shuffled directions in 4/6 runs
(rank 1 and 3 in the two failed runs — i.e. the direction is locally
strong even there). The two failed runs show WEAKER initial descent
(-0.232 and -0.080 vs -0.282 mean of the others), NOT greater curvature
penalties (+0.0012 and +0.0009 vs +0.0016 mean of the others). The
existing two-sided permutation comparison of exposure vs over-prediction
bias is clarified with exceedance counts and Monte Carlo p-values
(p = (1 + exceedances)/201; per-run table in tau_diagnostics.md).

# SECTION XXIV: EuroSAT checkpoint feasibility audit (2026-10-05)

A read-only feasibility audit of the EuroSAT replication checkpoints was
added (`analysis_eurosat_posthoc/CHECKPOINT_FEASIBILITY_AUDIT.md`). The
`.pt` checkpoint files are NOT present locally (no `checkpoints/`
directory in `results/replication_20261005_094928/`); this is reported
explicitly and CIFAR artifacts were NOT substituted. The audit combines
code-exact facts (the checkpoint-writing code) with the run's own metadata
(`validation.json`: 18/18 checkpoints, reconstruction max |diff| = 0.0,
identical-seed retrain passed; `cohorts_per_round.json`: 100 rounds x 6
runs of cohort IDs/counts/class counts). Findings: only the 5 participant
clients' states are saved per checkpoint, all from the same pre-round
global state, count-normalized weights, buffers averaged (num_batches_tracked
from the first client), fresh SGD per local call, no cross-client coupling
within a round; exactly reconstructable without training = the 31 non-empty
sub-cohorts of the five saved participants (any weighting); cohorts with
non-participants or cross-checkpoint combinations are NOT reconstructable.
Composition can be isolated only WITHIN the five participants at fixed
count; a population-level participation effect cannot be isolated without
new local training. Smallest feasibility pilot proposed by position (not
observed gains): p45_s401 round 100 — Design A: 0 new local-training calls,
24 evaluations; Design B: 5 deterministic local-training calls for the
missing clients. No training or candidate evaluation was performed.
