# Pilot decision logic v3 (active)

**Date frozen:** 2026-10-04 (BEFORE any pilot execution; replaces v2
preserved under `GO_NO_GO_v2_2026-10-04.md` and v1 preserved under
`GO_NO_GO_v1_2026-10-04.md`).
**Status:** Frozen. Change record is in
`PRE_EXECUTION_AMENDMENT.md`.

This document defines the **four decision statuses** the pilot
produces. They are computed independently. INCONCLUSIVE is reserved
for missing measurements or implementation failures; INCONCLUSIVE is
NOT a scientific NO-GO.

The v1 single GO/NO-GO (Claude-vs-fixed-dev) was replaced in v2
because it conflated two distinct questions (PROTCOL.md §1) and
because it relied on the dev-only `eval_cheat` baseline that leaked
the evaluation holdout (PRE_EXECUTION_AMENDMENT.md §A4, C1).

In v3 the screens are tightened:

  - **HEADROOM** is a per-seed per-control verdict. It now requires
    EVERY held-out seed to pass the SAME four sub-criteria against
    EACH named fixed control (`headroom_vs_global_fixed` and
    `headroom_vs_phase_fixed`). Two named outcomes are reported
    independently; a screen that holds against only one fixed control
    is `ADAPTIVE_PARTIAL_HEADROOM`, not a HEADROOM verdict.
  - **Claude comparator** is evaluated against BOTH fixed controls;
    `mean_G ≥ 0.00` becomes `mean_G ≥ +0.001` (strict).
  - **Exploratory signal** fits a real ordinary-least-squares model
    with intercept on the dev seed; **two predictor variants** are
    fit side by side (`phase_only` and `phase_plus_variability`).
    Dev-side standardization is persisted. The verdict stays
    `EXPLORATORY_PROBE`; it is NOT a deployability claim.
  - **Overall recommendation** semantics are honest: a fail-screen
    is named
    `ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT`, NOT
    `FIXED_SCALE_SUFFICIENT` (v2 confounded a negative screen with a
    universal claim).

## Notation

For every cell `(seed, checkpoint_round, source)` the pilot records
mean eval-loss on the **evaluation** holdout for each decisive
policy. Let

```
G(x, y) := L_fixed(x) − L_y
```

be the improvement of policy `y` over fixed control `x`; `G>0` means
`y` is better than `x` on that cell.

`fixed_global` is the single development-selected γ frozen before
held-out processing. `fixed_phase[r]` is the per-phase γ frozen
before held-out processing (one γ per `r ∈ {5, 50, 100}`).
`validation_selected` is selected per source on its selection split
and is independently valid on held-out seeds.

`G_val_vs_global(r)` = `G(fixed_global, validation_selected)`
**averaged within (cohort × rep) for cohort**, then averaged within
cohort for the phase, then averaged equally across phases for the
seed:

```
G_val_vs_global(s) = mean over r in {5,50,100} of
                      mean over c of cohorts of
                        mean over reps of cohort c of G_val_vs_global(s, r, c, rep)
```

The trajectory reference is reported separately and NOT included in
these cohort averages (to avoid overweighting the original cohort).

## Decision 1 — adaptive_headroom (v3)

**Question:** Does selection-based aggregate scaling improve
independent evaluation loss beyond a development-tuned fixed scale?

In v3 we report TWO named outcomes
(`headroom_vs_global_fixed`, `headroom_vs_phase_fixed`), each
computed independently:

  - `headroom_vs_global_fixed(s)` = `G_val_vs_global(s)`.
  - `headroom_vs_phase_fixed(s)` = mean over `r` of phase-aggregate
    `G_val_vs_phase(s, r)`.

For each named outcome, the screening gate has 4 sub-criteria,
applied to BOTH held-out seeds 202 AND 203:

  H1. **Mean improvement ≥ +0.01** on each held-out seed. (`mean_over_r
      G(s, r) >= +0.01`.)
  H2. **Phases-positive ≥ 2 of 3** on each held-out seed. (At least
      two of the three phases have a positive phase-aggregate
      `G_val_vs_fixed(s, r)`.)
  H3. **Max deterioration ≤ +0.02** on each held-out seed. (No phase
      has a phase-aggregate `G(s, r)` < `-0.02`; i.e., the worst
      shift is at most 0.02 worse than the fixed control.)
  H4. **All required cells are present and finite** on each
      held-out seed. This is the v3 inventory check; the
      `assert_complete_inventory` function raises
      `IncompleteMeasurementError` if any required (seed, round,
      source, policy) cell is missing or non-finite.

`adaptive_headroom` verdict (top-level):

  - **HEADROOM** iff, against EACH of `fixed_global` AND
    `fixed_phase`, ALL FOUR sub-criteria hold on BOTH held-out seeds;
  - **ADAPTIVE_PARTIAL_HEADROOM** iff exactly one of the two
    named outcomes holds on both held-out seeds;
  - **INSUFFICIENT** iff neither named outcome holds on both
    held-out seeds;
  - **INCONCLUSIVE** iff a required measurement is missing
    (`IncompleteMeasurementError` from
    `assert_complete_inventory`) or a required γ is invalid; we
    do NOT report INSUFFICIENT in that case.

The γ≈1.5 invariance is recorded as a `gamma_invariance_diagnostic`
block (mean and max per-source validation-selected γ across
held-out seeds) but is NOT a gate.

A `HEADROOM` verdict means: there is adaptive headroom beyond BOTH
tuned fixed scales (global AND phase), with the screening
thresholds above. It does NOT mean a deployable controller exists.
A screen is a mechanism measurement.

## Decision 2 — claude_comparator (v3; BOTH fixed controls)

**Question:** Did Claude's experimental raw shrinkage rule on held-out
seeds beat the fixed controls? This is recorded for context; it does
NOT affect Decision 1.

In v3, `G_claude_vs_<control>(s)` is computed against BOTH
`fixed_global` and `fixed_phase`. The gate for `claude_comparator`:

  C1. `mean_over_r G_claude_vs_<control>(s, r) >= +0.001` on each
      held-out seed AND against each fixed control (Claude at
      least not worse on average; strict positive);
  C2. `max_over_r |G_claude_vs_<control>(s, r)| <= +0.10` on each
      held-out seed AND against each fixed control (no catastrophic
      swing);
  C3. `n_worse_rounds(s, <control>) <= 1` on each held-out seed
      against each fixed control (Claude NOT consistently
      dominated).

`claude_comparator` verdict:

  - **GO** iff all three sub-criteria pass on BOTH held-out seeds
    against EACH fixed control;
  - **NO-GO** otherwise;
  - **INCONCLUSIVE** iff a required measurement is missing.

Failure of `claude_comparator` (Claude worse than fixed) does NOT
translate to insufficient headroom. Claude is one experimental
comparator; its deletion from the comparison set cannot change
Decision 1.

## Decision 3 — exploratory_variability_signal (dev only, v3)

**Question:** Do separated estimates of stochastic variability and
persistent client differences carry signal about which scaling
benefits held-out eval-loss?

In v3 we fit **two predictor variants** side by side on dev seed
only, against the same held-out seeds:

  - **phase_only**: per-phase one-hot features only.
  - **phase_plus_variability**: per-phase one-hot features plus the
    variability summaries (per_client_norm_cv, client_mean_norm_
    estimate, client_mean_to_agg_cos_estimate, cohort_aggregate_
    norm_mean).

Each predictor is fit with `np.linalg.lstsq` on the design matrix
`Z = [1, X]` (intercept + features) and reported with its dev-side
standardization (means and scales), intercept, coefficients
(unstandardized AND standardized for transparency), rank, condition
number, and per-phase sign-agreement on the held-out cells. Each
predictor's `mae_held` and `mse_held` are recorded, AND a
`max_predict_inconsistency` invariant: re-running the prediction on
the UNSTANDARDIZED scale
(`intercept + Σᵢ coef_i · x_i`) MUST equal the prediction on the
standardized scale to within numerical precision; the recorded value
is the max absolute element-wise difference.

The verdict is always `EXPLORATORY_PROBE` for both predictors. The
record includes:
  - `n_dev_cells_used`, `n_held_cells_used`, `feature_order`,
    `n_features_X`, `rank`, `condition_number`,
    `rank_deficient`, `dev_cells_too_few`,
    `constant_feature_mask`, `max_predict_inconsistency`,
    `decisive_inputs_sha256`.

A change to ANY field above (other than the literal `verdict`
string) is a pilot protocol modification and requires an update to
this document.

## Decision 4 — overall_recommendation (v3)

Synthesizes. Rules below are explicitly biased to NOT escalate a
single decision. **An INCONCLUSIVE anywhere can keep the overall
recommendation at INCONCLUSIVE; an overall NO-GO is reserved for
clear INSUFFICIENT headroom AND a NO-GO claude.**

  - **PROCEED_TO_CONTROLLER_PROTOTYPE** iff
      adaptive_headroom == HEADROOM
    AND  claude_comparator != NO-GO
    AND  no required measurement absent.
    (Note: this is a recommendation for follow-on controller work,
    not a claim that the controller exists. A
    `PROCEED_TO_CONTROLLER_PROTOTYPE` recommendation requires its
    OWN preregistered follow-on protocol.)
  - **ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT** iff
      adaptive_headroom == INSUFFICIENT
    OR   claude_comparator == NO-GO.
    The negative result is the point of the pilot and is recorded
    honestly. We do NOT claim a fixed scale is universally
    sufficient; we record that THIS pilot's screen did not
    establish a benefit.
  - **ADAPTIVE_PARTIAL_HEADROOM** iff
      adaptive_headroom == ADAPTIVE_PARTIAL_HEADROOM.
    We preserve the partial result explicitly. The four-decision
    statuses record WHICH fixed control passed and which did not.
  - **INCONCLUSIVE** iff
      adaptive_headroom == INCONCLUSIVE
    OR   any decisive measurement was missing
    OR   the principle "check the inputs, not the verdict" applies
         (any required input SHA failed to match a frozen input).
  - **EXPLORE_FURTHER** in any other case.

## Tie-breaking

Among ties on selection loss, the **lowest γ in the grid** wins.
This is documented in
`stepsize_diag.pilot_decision.select_with_tiebreak`.

## Audit (per decision)

After the pilot, `results/pilot_decision.json` records per decision:

- `verdict` ∈ {HEADROOM, INSUFFICIENT, INCONCLUSIVE,
  ADAPTIVE_PARTIAL_HEADROOM, GO, NO-GO,
  EXPLORATORY_PROBE, PROCEED_TO_CONTROLLER_PROTOTYPE,
  ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT,
  EXPLORE_FURTHER} (one per status).
- `decisive_inputs_sha256` per status, computed over the canonical
  JSON payload that includes ACTUAL numerical loss values. Changing
  one loss/feature value MUST change the hash.
- Per-decision details:
  - adaptive_headroom: per-seed per-control numbers, the two named
    outcomes (`headroom_vs_global_fixed`,
    `headroom_vs_phase_fixed`) with per-seed coverage, and the
    `gamma_invariance_diagnostic` block.
  - claude_comparator: per-seed per-control numbers (BOTH
    `vs_global_fixed` and `vs_phase_fixed`).
  - exploratory_variability_signal: TWO predictors (`phase_only`,
    `phase_plus_variability`) with their own
    `decisive_inputs_sha256` plus the `max_predict_inconsistency`
    invariant.
- The global inputs hash `protocol_sha256_glob` over
  `PROTOCOL.md` + `GO_NO_GO.md` +
  `PRE_EXECUTION_AMENDMENT.md`.

## What this protocol does NOT claim

- It does NOT claim Claude's rule or any comparator is novel.
- It does NOT claim a benefit has been measured (none has).
- It does NOT promise a deployable controller.
- It does NOT claim a fixed scale is universally sufficient when
  the pilot screen fails. A fail-screen is negative for THIS pilot
  only.
- It does NOT generalize beyond CIFAR-100, beyond the bundled
  reference SmallCNN, or beyond one client partition.
- It does NOT present the variability summary as a complete
  variance decomposition.
