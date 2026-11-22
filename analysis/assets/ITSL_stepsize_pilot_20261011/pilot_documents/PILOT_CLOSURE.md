# Pilot closure record — CIFAR-100 step-size pilot

**Run:** `results/stepsize_diag_pilot_20261004_175620`
**Closed:** 2026-10-04
**Frozen documents (unchanged):** `PROTOCOL.md` (v3), `GO_NO_GO.md` (v3),
`PRE_EXECUTION_AMENDMENT.md` (change record).
**Verification:** every number below was read directly from the saved
artifacts listed under "Provenance". Repetitions within a
(cohort, seed, round) cell are NOT independent training runs; all
summary statistics average repetitions within each cohort first, and
trajectory sources are reported separately.

## 1. Measured findings (scientific outcomes, frozen criteria)

### 1.1 Adaptive headroom — INSUFFICIENT (measured failure to meet the
frozen gate)

The frozen gate requires, on BOTH held-out seeds against BOTH fixed
controls: mean G ≥ +0.01, ≥ 2 of 3 checkpoint phases positive, and worst
phase deterioration ≤ 0.02 (G = eval_loss_fixed_control −
eval_loss_validation_selected, positive = improvement). Measured:

| Control | Seed 202 mean G (phases r5/r50/r100) | Seed 203 mean G (phases) |
|---------|---------------------------------------|--------------------------|
| global fixed (γ=0.5) | +0.005737 (+0.016813 / +0.000398 / 0.0) — 2/3 positive | +0.022656 (+0.067968 / 0.0 / 0.0) — 1/3 positive |
| phase fixed (0.75/0.5/0.5) | +0.002900 (+0.008303 / +0.000398 / 0.0) — 2/3 positive | +0.003649 (+0.010948 / 0.0 / 0.0) — 1/3 positive |

Gate failure (persisted):
`screen_unsatisfied_against_both_controls_on_at_least_one_held_out_seed`.
The gate fails on at least one sub-criterion for at least one held-out
seed against each control.

### 1.2 Claude raw shrinkage rule — NO-GO

Claude's per-source raw shrinkage rule was worse than BOTH fixed controls
on EVERY checkpoint round of BOTH held-out seeds (persisted gate failure:
`at_least_one_seed_or_control_failed_claude_screen`; frozen threshold
mean G ≥ +0.001):

| Control | Seed 202 mean G | Seed 203 mean G | rounds worse (of 3) |
|---------|-----------------|-----------------|---------------------|
| vs global fixed | −0.019722 | −0.035515 | 3/3 both seeds |
| vs phase fixed | −0.022559 | −0.054522 | 3/3 both seeds |

### 1.3 Overall recommendation

`ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT` (persisted in
`pilot_decision.json`; this label is a measured negative for THIS pilot,
not a universal claim that fixed scales are always sufficient).

### 1.4 Corrected exploratory comparison (phase-only vs
phase-plus-variability)

Recomputed from saved artifacts only, into
`results/stepsize_diag_pilot_20261004_175620_audit_corrected_v2/`, after
repairing the two implementation defects that invalidated the original
exploratory output (§2). 27 dev cells / 54 held-out cells; dev-only OLS
fit; held-out evaluation.

| Predictor | held MAE | held MSE |
|-----------|----------|----------|
| phase_only | 0.009036 | 0.0003136 |
| phase_plus_variability | 0.020016 | 0.0006962 |

By training seed (MAE / MSE): phase_only — 202: 0.005331 / 0.0001171,
203: 0.012741 / 0.0005100; phase_plus_variability — 202: 0.023525 /
0.0008249, 203: 0.016507 / 0.0005675.

By checkpoint round (MAE / MSE): phase_only — r5: 0.026910 / 0.0009400,
r50: 0.000199 / 7.11e-07, r100: 3.47e-18 / 1.20e-35;
phase_plus_variability — r5: 0.039113 / 0.0017480, r50: 0.009576 /
0.0001574, r100: 0.011360 / 0.0001831.

Read with the clustering caveat (3 repetitions per (cohort, seed, round)
cell are not independent). Per the audit instruction, no success/failure
claim about the variability hypothesis is recorded here; the corrected
comparison stands as the measured outcome.

### 1.5 Buffer-policy analysis (saved artifacts; revised report)

`analysis_buffer_policy_v2/` (supersedes `analysis_buffer_policy/`,
which is preserved). On the evaluation split, repetitions pre-averaged
within cohort, dev seed 201 and held-out seeds 202/203 separated:

- Freezing checkpoint buffers at γ=1 did NOT improve over full-state
  FedAvg in this setting: held-out cohort-mean loss improvement
  −0.0629 (dev −0.0705); sign counts 3+/24− of 27 all-seed cells and
  3+/15− of 18 held-out cells. Conclusion limited to frozen checkpoint
  buffers at γ=1 here; no generalization to other normalization methods.
- Validation-selected damping improved over both γ=1 references
  (held-out: +0.0646 vs full-state, +0.1276 vs frozen-buffer). Both
  comparisons share the same candidate; their difference reflects the
  different reference losses, not an interaction.
- Against the dev-selected fixed scales (held-out): +0.0142 vs global,
  +0.0033 vs phase — consistent with, and subsumed by, the frozen
  INSUFFICIENT verdict in §1.1.
- The frozen-buffer scale grid was not measured; parameter-scaling and
  buffer-scaling effects remain incompletely separated.

## 2. Implementation failures and their artifact-only repairs

These are implementation defects, NOT scientific findings. None required
retraining or new candidate evaluation; all repairs were verified by the
CPU self-test and by artifact-only recomputation. Recorded in
`PRE_EXECUTION_AMENDMENT.md` SECTIONS V–VIII and
`audit_stepsize_diag_pilot_20261004_175620/AUDIT.md`.

- Distribution ZIP excluded every directory named `data` (dropped
  `reference_code/data/`); repaired packaging.
- Per-source best-γ used `argmin(...) or GAMMA_ONE`, coercing γ=0.0 wins
  to 1.0; explicit `None` handling.
- Feature-path counts fell back to count 1; canonical
  `counts_by_client` now flows explicitly and missing counts fail.
- Parameter names were guessed/state-dict-inferred; single-sourced from
  `named_parameters()` filtered by `requires_grad`.
- Unused runtime `import pytest` in `verify.py` removed.
- Exploratory probe: dev seed lacked fixed-policy keys (processed before
  the freeze), held-out targets were looked up under the dev seed, and
  held-out features mixed both held-out seeds — all three produced
  `fit_status="no_cells"`; repaired, and the analysis was recomputed
  from saved artifacts (`..._audit_corrected/`).
- Both exploratory predictors fit the full outer feature matrix
  ("phase_only" was not phase-only); per-predictor design matrices with
  a runtime structural contract; recomputed (`..._audit_corrected_v2/`).
- Report wording claimed "two predictors fit" and dropped variability
  records; fixed.

## 3. Research recommendation

- PARK the tested raw shrinkage rule for this CIFAR-100 setting
  (measured NO-GO against both fixed controls on both held-out seeds).
- PARK the frozen variability-predictor formulation (the 4-feature
  per-checkpoint summary + phase indicators) for this setting: the
  corrected comparison did not improve held-out prediction over
  phase-only indicators. This is a statement about THIS formulation,
  not about step-size control in general — per-source
  validation-selected scaling did improve over the γ=1 references
  (measured), though it did not meet the frozen headroom gate against
  the dev-selected fixed scales.
- Do NOT claim that step-size control is ineffective.
- Preserve for reuse: the harness (`stepsize_diag/`), the checkpoints
  (`results/stepsize_diag_pilot_20261004_175620/checkpoints/`), the
  partition (`partition.json`), the paired evaluations
  (`per_example_losses.csv`, `aggregate_evaluation.*`,
  `rule_evaluation.*`), and the tuned fixed controls
  (γ_global = 0.5; γ_phase = 0.75 / 0.5 / 0.5 for rounds 5 / 50 / 100).
- The previously identified evaluation-only frozen-buffer grid
  extension is NOT launched, and no further tuning against the same
  held-out seeds (202, 203) is performed.

## 4. Limits

- One dataset (CIFAR-100), one fixed partition, one development training
  seed (201), two held-out training seeds (202, 203).
- Clustered repetitions: 3 reps per (cohort, seed, round) cell are not
  independent measurements.
- Fixed-checkpoint one-step outcomes: all candidate evaluations are
  single interpolation steps from saved checkpoints at rounds
  5 / 50 / 100 — NOT end-to-end adaptive training.
- The frozen-buffer scale grid was not measured.
- The exploratory predictors were fit on 27 dev cells with up to 6
  regressors.

## 5. Provenance

- Original run outputs `results/stepsize_diag_pilot_20261004_175620/` —
  UNCHANGED.
- Exploratory corrections `..._audit_corrected/` and
  `..._audit_corrected_v2/` — UNCHANGED.
- Revised buffer analysis: `analysis_buffer_policy_v2/` (fresh
  directory; `analysis_buffer_policy/` preserved unchanged). Its
  `report.md` records the sha256 of every input artifact and the exact
  command.
- Exploratory recomputation command:
  `python -X utf8 scripts/recompute_exploratory_from_artifacts.py
  --run-dir results/stepsize_diag_pilot_20261004_175620
  --out-dir results/stepsize_diag_pilot_20261004_175620_audit_corrected_v2`
- Buffer analysis command:
  `python -X utf8 analysis_buffer_policy_v2/analyze_buffer_policy.py
  --run-dir results/stepsize_diag_pilot_20261004_175620
  --out-dir analysis_buffer_policy_v2`
- Change record: `PRE_EXECUTION_AMENDMENT.md` SECTION IX.
- No data deleted; no scientific outputs regenerated unnecessarily;
  `PROTOCOL.md` and `GO_NO_GO.md` unchanged.

This task closes the current pilot. Proposing a new method is outside
its scope.
