# Frozen Pilot Protocol v3 (active)

**Date frozen:** 2026-10-04 (BEFORE any pilot execution; replaces v2
preserved under `PROTOCOL_v2_2026-10-04.md` and v1 preserved under
`PROTOCOL_v1_2026-10-04.md`).
**Status:** Frozen. The change record is in
`PRE_EXECUTION_AMENDMENT.md`.

This pilot is a **mechanism investigation**, not a novelty claim and not
a terminal training superiority claim.

## 1. Questions the pilot answers

The pilot answers **two separate questions**, and reports them
**separately**:

- **Question A — Adaptive headroom.** Does selection-based aggregate
  scaling (per-source γ chosen on a selection split) improve independent
  evaluation loss beyond a development-tuned fixed scale?
- **Question B — Update signal.** Do separated estimates of stochastic
  variability and persistent client differences (over the dev seed
  repetitions and cohorts) provide any signal that helps identify
  beneficial scaling beyond training phase?

Claude's experimental raw shrinkage rule is **one experimental
comparator only**: failure does not imply absence of headroom; success
does not imply a deployable controller. The validation-selected
baseline is legitimate (no eval-split leak) and is computed per source
on the selection split; the eval-split is reserved for evaluation only.

A deployable controller — if warranted by this pilot's results — is a
**separate follow-on** with its own preregistered protocol; this pilot
does not develop one.

## 2. Dataset and model (unchanged from smoke and v2)

- CIFAR-100 (raw `train` only); normalize constants and augmentation
  recipe taken from the bundled reference snapshot
  (`reference_code/data/datasets.py`).
- Bundled reference SmallCNN, `use_aib=False`, channels 64/128/256,
  GAP-fed 256-feature linear classifier — hash- and fact-validated
  against `COPY_MANIFEST.json` / `ADAPTED_HASHES.json`. The verification
  chain (`ref_adapter.verify_snapshot`, `load_reference_model`) MUST
  pass before any training starts; the launchers set `--require-reference`.

## 3. Federated protocol (unchanged from smoke and v2)

10 clients; 5 participants/round; Dirichlet α=0.3; min client size 50;
local epochs E=2; batch 64; SGD lr=0.01, momentum=0.9, weight_decay=5e-4;
gradient clip 5.0; selection holdout 5% + evaluation holdout 5% from
the ORIGINAL training pool, stratified, disjoint, `split_seed=999`
(unchanged from smoke); clients train only on the RETAINED indices.
Aggregation weights are retained sample counts normalized over the
participating cohort.

Determinism policy (unchanged): `cudnn.deterministic=True`,
`cudnn.benchmark=False`, reapplied after every `set_seed` call; the
DataLoader uses `num_workers=0` with an explicit shuffle generator.

## 4. Training seeds and partition

**One fixed client partition across training seeds.** The three training
seeds (201 / 202 / 203) provide **different initialization, training
runs, and per-round cohort trajectories** under **one partition**
(`partition_seed=42`). The seeds do **not** provide independent
partition realizations. We CANNOT generalize across partitions; the
protocol explicitly does not claim that.

| Role         | seed | Used for |
|--------------|------|----------|
| Development  | 201  | training run, source generations, **all** scale selection (global and per-phase), validation-selected gammas, exploration fit; per-source eval. Held-out processing does NOT use its selection losses for anything other than the fixed-scale selection table. |
| Held-out     | 202  | training run, source generations, eval only. |
| Held-out     | 203  | training run, source generations, eval only. |

The split holdouts (selection/evaluation) are IDENTICAL across all three
seeds because `split_seed` is fixed and the partitioner runs on the same
raw training pool. This is intentional: it puts the seed variation on
initialization and cohort trajectory, not on which examples are
holdout/eval/train.

## 5. Checkpoints

For every training seed, run FedAvg for 100 rounds and save the global
state **immediately before** the local updates of rounds 5, 50, and
100:

| checkpoint_round | phase      | saved as                                          |
|------------------|------------|---------------------------------------------------|
| 5                | early      | `checkpoints/seed_<s>/checkpoint_round_5.pt`     |
| 50               | middle     | `checkpoints/seed_<s>/checkpoint_round_50.pt`    |
| 100              | late       | `checkpoints/seed_<s>/checkpoint_round_100.pt`   |

Convention: file holds the state BEFORE round `r`'s local updates, i.e.
the state at end of round `r − 1`. The file also stores the round `r`
cohort and its post-local-update state dicts.

## 6. Aggregate-source plan per checkpoint: 10 sources

Per `(seed, checkpoint)`, the diagnostic evaluates against **10
sources**:

| source type          | count | rep seeds | notes |
|----------------------|-------|-----------|-------|
| `trajectory_cohort`  | 1     | (state from training) | recovered from saved `selected_clients`. |
| `original_cohort_rep`| 3     | `(1001, 1002, 1003)` | three same-cohort reps under three seeds. |
| `alt_cohort`         | 2 × 3 | cohorts `alt_0`, `alt_1` × seeds `(1001, 1002, 1003)` | both alt cohorts must differ from each other AND from the original. |

**Total per checkpoint: 10 sources = 1 + 3 + 2 × 3.** Per pilot =
30 sources per seed × 3 seeds = **90 sources total**.

- **Cohort-position independence.** Client local-training seeds depend
  ONLY on `(rep_seed, checkpoint_round, client_id)`:
  `epoch_seed = rep_seed * 100000 + round_num * 1000 + client_id`.
  Cohort position or membership does NOT enter the seed. A shared
  client produces the same update under the same `(rep_seed, round,
  client_id)` triple, in any cohort.
- **RNG reset on every local call.** The model, optimizer, and RNG
  state are reset before every local call (`reset_all_rng(epoch_seed)`
  then fresh model + load_state_dict + fresh SGD in
  `stepsize_diag/fedavg.py:train_client_local`). Persisted and not
  regressed.
- **Alternative-cohort sampling.** `np.random.RandomState(seed +
  cohort_seed_offset + round * 17 + alt_index * 257)` is the advancing
  RNG that draws the two alt cohorts, per checkpoint, per seed. **One
  RNG, bounded retries** (≤64 attempts per alt cohort). If both alt
  cohorts cannot be drawn distinct from each other and from the
  original within 64 attempts, the run fails closed with
  `INCOMPLETE_MEASUREMENT` (NOT silent re-use).
- **Trajectory is reported separately.** The trajectory cohort is
  **never** averaged into cohort-totals to avoid overweighting the
  original cohort; it is reported as a separate reference stream.

## 7. Step-size policies (six, preregistered)

Primary buffer policy for **all** decisive policies is
`all_floating_state` (matches the recovered fixed-γ intervention).
Secondary buffer policy is `parameters_only_frozen_buffers` and is run
for **γ=1 only** as a sensitivity analysis explicitly separate from
FedAvg.

| # | Policy | γ source | Computed on | Notes |
|---|--------|----------|-------------|-------|
| I+ | FedAvg | γ = 1 fixed | every (seed, round, source) | ground truth + every other policy's reference. |
| I | global development-selected fixed γ | argmin over γ of mean over (phase × cohort × rep-source-retain) of selection loss on **dev seed 201**; required γ valid for every required dev source; ties broken on smallest γ. **FROZEN before held-out processing.** | dev seed only | applied unchanged to all (held-out seed, round, source) cells. |
| II | phase-only control | argmin over γ of mean over (cohort × rep-source-retain with at least one valid γ per cohort) of selection loss, **per checkpoint phase**, on dev seed 201; same validity mask; ties broken on smallest γ. **FROZEN before held-out processing.** Distinct fixed γ per phase. | dev seed only | applied unchanged to all (held-out seed, source) cells within that phase. |
| III | validation-selected scaling | per source on its **selection split**; evaluate on the **independent evaluation split**. | every (seed, round, source, both holdouts used for SELECTION and EVALUATION respectively) | legitimate per-source baseline; no eval-split leak. |
| IV | Claude raw shrinkage | per source from `compute_update_statistics` + `claude_raw_shrinkage`. Exact γ; safeguard log retained. | every (seed, round, source) | experimental comparator. |
| V | evaluation-optimal grid reference | argmin over the grid of eval-loss for each source. | every (seed, round, source); but **retrospectively labeled**: it is NOT a deployable upper bound on terminal accuracy or any continuous controller. | dev seed values inform pilot, held-out values are recorded but **described as retrospective**, not as upper bounds. |
| γ=0 reference | no update | γ = 0 | every (seed, round, source) | sanity reference, never used in decisions. |

### a. Required `γ=1` validity

For decisive comparisons, the γ=1 reference MUST be valid for every
required cell. If γ=1 is invalid for any required cell on a held-out
seed, that cell is `INCOMPLETE_MEASUREMENT`; it is NOT silently
substituted with another γ.

### b. γ_dev fallback when selection cannot be completed

For the global fixed and phase-only fixed controls, **all γ in the
grid are required to satisfy `valid_for_required_dev_sources`**. If
none satisfies that, the control is unset (`absent`) and the pilot
returns `INCONCLUSIVE` on adaptive-headroom. The γ=1 value is NOT a
silent fallback for the global/phase fixed control selection. γ=1 IS
a documented fallback for **validation-selected scaling** if no valid γ
exists for that source: γ=1 is recorded as the source's
validation-selected γ with a clear marker.

### c. FedExP exclusion

FedExP is **excluded from decisive comparisons** unless independently
verified against its primary reference or official implementation. If
retained diagnostically, it is labeled `fedexp_provisional` and its
unequal-weight adaptation is documented in
`stepsize_diag/candidates.py`.

### d. Secondary buffer policy scope

Secondary (`parameters_only_frozen_buffers`) is run for **γ=1
ONLY** — one evaluation per (seed, round, source). Results are
labeled explicitly as the secondary `frozen_buffer_parameter_candidate`
distinct from actual FedAvg. Its γ=1 is NOT FedAvg.

## 8. Outputs (the second half of the protocol)

Beyond the smoke artifacts (`provenance.json`, `config.json`,
`partition.json`, `results/verification.json`,
`results/reference_comparison.json`,
`results/aggregate_evaluation.{csv,json}`,
`results/rule_evaluation.{csv,json}`,
`results/update_statistics.json`, `results/timing.json`,
`results/report.md`, `pilot_protocol_sha256.json`), the pilot adds
(exactly):

| Path | Contents |
|------|----------|
| `checkpoints/seed_<s>/checkpoint_round_<r>.pt` | full checkpoint (state, cohort, weights, post-state, partition hashes, fingerprint). |
| `results/aggregate_evaluation.csv` | full γ-grid × primary-policy × sources (declared header columns verified). |
| `results/aggregate_evaluation.json` | same content as CSV (full record). |
| `results/secondary_buffer_evaluation.csv` | one row per (seed, round, source) for `parameters_only_frozen_buffers` γ=1. |
| `results/rule_evaluation.csv` | per rule: γ + safeguards + origin + note + uses_heldout flag. |
| `results/update_statistics.json` | per-source stats with explicit `_inventory` block. |
| `results/variability.json` | per-client norms + spread + deduped mean delta vector + per-client vector_spread + weighted cohort aggregate vector norms + per-cohort within-cohort aggregate vector spread + between-cohort aggregate vector diff under matched rep_seed/round pairs + finite-repetition uncertainty explicitly stated + estimator_definitions per field. |
| `results/per_example_losses.csv` | per-example losses / correctness for the **decisive policies** (FedAvg, fixed_dev_global, fixed_dev_phase, validation_selected_scaling, claude_raw). |
| `results/gamma_dev_global.json` | frozen global fixed γ with selection table, hashes, timestamp. |
| `results/gamma_dev_phase_<round>.json` | frozen per-phase fixed γ with selection table, hashes, timestamp. |
| `results/selection_table_dev.json` | dev-seed full selection-loss / eval-loss matrix (rows = sources, cols = (phase × γ)); frozen before held-out processing. |
| `results/eval_metrics.json` | per (seed, round, source, policy) averaged eval-loss on eval holdout — the input to decisions. |
| `results/source_keys_dev_only.json` | SHA256 over the keys going into `pilot_decision.dev_selection_input`. Proves dev-before-held-out separation. |
| `results/pilot_decision.json` | four decision statuses + per-status `decisive_inputs_sha256` (hashing real numerical loss values) + executive summary; v3 also reports two named outcomes for adaptive-headroom (`headroom_vs_global_fixed`, `headroom_vs_phase_fixed`) and two predictors (`phase_only`, `phase_plus_variability`) with their own input hashes. |
| `results/checkpoint_metadata.json` | per-seed per-round metadata. |
| `results/report.md` | the rendered four-decision report with every input hash. |
| `results/timing.json` | per-phase timings. |

### Output column completeness (PROTOCOL.md §7 — declared columns)

`results/aggregate_evaluation.csv` MUST contain every declared
column: `seed, round, dataset, training_seed, checkpoint_round,
checkpoint_convention, source, cohort_id, rep_seed, clients,
sample_counts, weights, weight_sum,
raw_weight_sum_population_normalized, buffer_policy, gamma,
row_label, candidate_valid, invalid_entries, sel_loss, sel_acc,
eval_loss, eval_acc, n_sel, n_eval`. The synthetic test asserts this.

### Per-example storage budget

Each of the FIVE decisive policies emits per-example losses on
TWO holdouts per source. Per source, per holdout, per policy: N_eval
rows (N_eval ∈ {~2500} depending on the holdout). Per (seed, round,
source, holdout, policy): one CSV row per example.
Total rows: 5 policies × 2 holdouts × 90 sources × ~2500 examples =
~2.25M CSV rows. Per-row size ~70 bytes → ~150 MB CSV for one full
pilot. We do NOT launch the pilot from this protocol; storage is
budgeted above for the operator.

## 9. Verification — preserved 12 mandatory checks

The pilot runs the SAME 12 mandatory verification checks as the smoke.
Concretely: the pilot hashes the bundled reference snapshot
(`ref_adapter.verify_snapshot`), runs the CUDA evaluation-path probe
when `device=cuda`, runs all twelve checks on **one** cell (dev seed
201, round 5, trajectory cohort). **Per-checkpoint lightweight
checks** (weight-sums finite, endpoint same as γ=0 reference, γ=1
reference valid) are run for every cell; the expensive retraining
checks (identical-seed retrain reproducibility, shared-client matched
seed) run once. Their scope is recorded in `provenance.json`. Updating
these checks is a pilot protocol change.

A new mandatory check **also runs**:
`verify.check_per_example_eval_restores_state` exercises
`evaluate_state_per_example` including restoration under EXCEPTION
(`try/finally`). It is added to the pilot's mandatory verification
set.

The v3 pilot additionally enforces **exact measurement inventory
validation** before any `decide_*` call:

- `assert_complete_inventory(eval_metrics, pconfig, policies=…)`
  is invoked in `run_pilot` BEFORE the four decision calls. If a
  required cell is missing or non-finite, a partial-decision artifact
  is persisted and the pilot exits with
  `PilotIncompleteMeasurement`.

## 10. Preregistered exclusions

- No post-dev processing of held-out selection losses for
  fixed-control selection: held-out selection losses may select
  **validation-selected scaling** (per source) on the held-out seed,
  but they MUST NOT influence the fixed controls (global, phase-only),
  any predictor, or any threshold.
- No post-execution fitting on held-out data. Any fitting is done on
  the dev seed only, with hyperparameters frozen before held-out
  processing.
- No retraining of any cell that the synthetic test cannot exercise
  via mocked data.
- No retroactive change to the selection rule. The selection rule
  and tie-break are documented in
  `pilot_decision.py:select_with_tiebreak`.
- No exclusion of invalid sources silently: invalid candidates are
  preserved with their reasons and the selection process raises
  `IncompleteMeasurementError` rather than averaging around them; the
  v3 inventory check refuses to silently average around missing or
  non-finite required cells.
- No silent re-use of a client that appears in multiple cohorts under
  the same `(rep_seed, round, client_id)` triple. The dedup
  inventory records any matched-copy collision; the deduped signature
  is the basis for `per_client_norm_mean`, `per_client_mean_delta_`
  `vector_norm`, `per_client_vector_spread`,
  `per_pairwise_mean_distance`, and the cohort aggregate computation.

## 11. Sequencing (dev-before-held-out)

1. Validate (this means: imports, snapshot hash, CUDA eval-path probe
   if applicable, dummy-check return values, no ModuleNotFoundError).
2. Process dev seed 201 (training, sources, grid, per-example).
3. Compute and persist dev-only selection tables and grids.
4. **Select and persist**:
   - `gamma_dev_global.json` (with selection rule, table, hash,
     timestamp),
   - `gamma_dev_phase_<r>.json` for each `r ∈ {5, 50, 100}`
     (with per-phase table, hash, timestamp).
5. **Freeze** by storing their hashes via
   `results/source_keys_dev_only.json`.
6. Only then process held-out seeds 202 and 203.

Held-out seeds' selection losses are used for **per-source
validation-selected scaling only**. They do not enter any other
decision.

## 12. Provenance / fingerprinting

- `provenance.json` records the SHA256 of every `.py / .sh / .ps1 /
  .md` in the bundle (existing); the pilot additionally records the
  SHA256 of `PROTOCOL.md`, `GO_NO_GO.md`, `PRE_EXECUTION_AMENDMENT.md`,
  and the per-phase/global `gamma_dev_*.json` selection tables.
- Each saved checkpoint `.pt` records the same `config_fingerprint`
  computed from preregistered fields + determinism policy. Reuse is
  governed by `check 12`.
- `pilot_config_fingerprint` is the SHA256 over the preregistered
  fields specific to the pilot (training seeds, checkpoint rounds,
  source plan, policy list, gamma_dev thresholds, screening margins,
  selection rule name, tie-break rule name AND the two frozen fit
  procedure names). Any deviation invalidates every prior
  measurement.
- `decisive_inputs_sha256` per decision statuses (and per predictor
  in v3) hash the canonical numeric payload (including actual
  numerical loss values, selected scales, validity, feature/target
  arrays, fitted parameters). Changing one loss/feature changes the
  hash for that exact decision.

## 13. Launchers and required flags

The pilot launchers (`run_pilot.sh`, `run_pilot.ps1`) MUST:

- refuse to run unless `--require-reference` is set (no
  reconstruction fallback);
- refuse to run unless the bundled reference snapshot hash-validates
  AND `PROTOCOL.md`, `GO_NO_GO.md`, and `PRE_EXECUTION_AMENDMENT.md`
  exist;
- pass `PYTHONDONTWRITEBYTECODE=1`;
- emit `<out>.launch.stdout.log`, `<out>.launch.stderr.log`, and
  `<out>.exit_status` BESIDE the fresh timestamped output directory.
  `run_pilot.ps1` v3 additionally captures stdout and stderr into
  separate per-stream files (so a non-zero-terminating stderr line
  cannot short-circuit the `Set-Content -Path <out>.exit_status`
  write);
- NOT run the three seeds in parallel by default (serial execution
  is the preregistered default).
- The Windows launcher may take an explicit `-Python` path so the
  CUDA-enabled Python interpreter is selected deterministically.
- The PowerShell launcher v3 starts Python via `Start-Process
  -NoNewWindow -RedirectStandardOutput/Error`, captures the native
  exit code, and ALWAYS writes `<out>.exit_status` (even on
  `Start-Process` internal errors). `$ErrorActionPreference =
  "Stop"` is applied ONLY to binding checks (file-existence, Python
  resolution, fresh-outdir check); it is removed before launching
  Python.
- The fresh-output check is case-insensitive on the directory name
  AND trims trailing separators; a non-empty `<out>` containing
  `results/`, `checkpoints/`, `provenance.json`, `config.json`,
  `partition.json`, `pilot_protocol_sha256.json`, or any
  pilot-output `.json/.csv/.pt/.md/.launch.{,stdout.,stderr.}log`
  /`exit_status` is REJECTED.

## 14. Approximate run-time budget (CUDA-class, RTX 3060)

The `results/timing.json` from the smoke run is for a single 5-round
training run. The v3 pilot runs three 100-round training runs
(sequentially) plus the eval grid × policies × sources × holdouts
× 5 decisive policies per source. Operator test budget:

- Three 100-round training runs (one per seed) on RTX 3060: ~37.5
  min (matches per-round smoke timings × 100 rounds × 3 seeds on
  the same device).
- PHASE 2 (dev-side 30 sources × γ grid × primary policy): the
  eval-grid overhead per source dominates; estimate ~5–10 min on
  RTX 3060.
- PHASE 4 (held-out seeds × 60 sources × eval grid + 5 decisive
  policies × 2 holdouts × per-example loss): the per-example
  pass dominates; estimate ~30 min on RTX 3060.
- Total: ~75–80 min on the local RTX 3060. The legacy CPU
  extrapolation to ~7.5 h is REMOVED v3: the smoke's measured
  timings were CUDA RTX 3060 timings (the dedup et. al. is local).
  A CPU-only run is NOT recommended.

## 15. What this protocol does NOT claim

- It does NOT claim Claude's rule or any comparator is novel.
- It does NOT claim a benefit has been measured (none has).
- It does NOT promise a deployable controller.
- It does NOT generalize beyond CIFAR-100, beyond the bundled
  reference SmallCNN, or beyond one client partition.
- It does NOT present the variability summary as a complete
  variance decomposition.
- It does NOT claim a fixed scale is universally sufficient when
  the pilot fails the headroom screen; a fail-screen is a fail-
  screen for THIS pilot only. The decision label is
  `ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT` (was
  `FIXED_SCALE_SUFFICIENT` in v2; the v2 label confounded a
  negative screen with a universal claim).
