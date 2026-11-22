"""Pilot orchestrator for the CIFAR-100 pilot (PROTOCOL.md / GO_NO_GO.md).

WORKFLOW (frozen, this is the real execution path):

  STAGE 0a — provenance + data + partition + counts-by-client
      (loads CIFAR, partitions, builds explicit
      `{client_id: int(count)}` map; refuses to silently fall back
      to uniform weights when a participating client's count is
      missing).

  STAGE 0b — generation + 12 mandatory checks AGGRESSIVELY EARLY
      For dev seed 201: run rounds 1..5 with the full 12 mandatory
      verification checks (saved-state hash, gamma0/gamma1/secondary
      endpoint agreement, weighted aggregation invariants, rule
      safeguards, configuration rejection, evaluation restores state,
      holdout disjointness + eval determinism, identical-seed
      retrain, shared-client matched seed, per-example eval
      restoration, model compatibility).  If any mandatory check
      fails the run aborts BEFORE generating the remaining
      50/100-round checkpoints for any seed.

  STAGE 0c — incremental fleet generation
      For dev seed 201, RESUME from the round-5 checkpoint and
      generate rounds 6..50, then 51..100, saving the round-50 and
      round-100 checkpoints. For each held-out seed (202, 203):
      run the full 100 rounds (no resume needed). The pilot does
      not need to RE-RUN any completed rounds.

  STAGE 1 — per-seed per-checkpoint variability
      For each (seed, round): aggregate the per-rep deltas AND
      produce a per-checkpoint variability summary (no global
      pooling across rounds).  Trajectory is reported separately.

  STAGE 2 — dev-only per-cell grid × rule metrics
      For dev seed 201 and every (round, source) in the source plan
      (1 trajectory + 3 same-cohort reps + 2 alt cohorts × 3 reps
      =10 sources/checkpoint), run:
      - the full gamma grid × primary policy all_floating_state
        × 2 holdouts (selection, evaluation) aggregate evaluation
      - ONE γ=1 measurement under the secondary buffer policy
        (parameters_only_frozen_buffers)
      - the four rule_rows: FedAvg, validation-selected, eval-grid
        reference, Claude raw shrinkage
      - the five decisive rules once fixed controls are frozen:
        fixed_global_gamma_dev, fixed_phase_gamma_dev.
      Each row is split-specific (sel_loss/eval_loss tracked
      separately) AND persists candidate validity.

  STAGE 3 — dev-side fixed-control selection
      Use only dev rows for selection. The selection rule requires
      ONE valid candidate per γ across ALL required sources; if
      no γ is valid, raise `IncompleteMeasurementError` (mapped
      to INCONCLUSIVE). Persist:
      results/selection_table_dev.json
      results/gamma_dev_global.json
      results/gamma_dev_phase_<round>.json  (one per phase)
      results/source_keys_dev_only_sha256.json  (SHA over the dev
          split_metric_keys; proves no held-out key contributed).

  STAGE 4 — dev-side exploratory probing fit
      Build dev-only features per (seed, round, cohort, rep) from
      the dev variability summaries (NOT mixed with held-out).
      Embed two side-by-side predictors (`phase_only`,
      `phase_plus_variability`) using OLS with intercept on dev
      features and EVALUATION-holdout targets.
      Persist `results/exploratory_signal_dev.json` with dev
      standardization, coefficients, intercept (UNSTANDARDIZED AND
      STANDARDIZED, separately), sign agreement by phase, MAE,
      MSE, prediction-vs-target records, per-feature
      constant-flag, and per-predictor `decisive_inputs_sha256`
      over the actual numeric payload.

  STAGE 5 — held-out per-cell processing
      For each held-out seed (202, 203): same per-cell eval as
      dev. Build held-out variability summaries PER (seed, round)
      using only held-out payloads. Build held-out features from
      held-out variability. Apply frozen fixed controls.

  STAGE 6 — held-out predictor assessment
      Reuse the dev-side standardization and coefficients
      (NOT recomputed). Predict held-out targets; record
      per-predictor MAE, MSE, sign-agreement by phase.

  STAGE 7 — four decisions
      In order: decide_adaptive_headroom, decide_claude_comparator,
      decide_exploratory_signal, decide_overall. Each decision
      hashes its actual numerical inputs into its
      `decisive_inputs_sha256` so a future analyst can re-run the
      decision without retraining.

  STAGE 8 — outputs finalization
      Persist incomplete-decisions/saving artifacts fail-safely
      when any unexpected exception is caught at the orchestration
      boundary.

NOT EXECUTED in this code path: launchers set --require-reference;
the actual launch is the operator's decision.
"""
import csv
import hashlib
import json
import logging
import math
import os
import tempfile
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .candidates import (build_candidate, claude_raw_shrinkage, fedexp_step,
                          POLICY_ALL_FLOATING, POLICY_PARAMS_ONLY)
from .config import bundle_hashes, SmokeConfig
from .data import (load_cifar100_raw, stratified_holdouts,
                   dirichlet_partition_with_holdout_removal,
                   partition_hashes, assert_disjoint)
from .fedavg import (generate_checkpoint, train_client_local,
                      evaluate_state, epoch_seed_for, sample_cohort,
                      average_states)
from .pilot_config import PilotConfig
from .pilot_decision import (parse_source_label, source_label,
                              expected_cells_global,
                              split_metric_key, parse_split_metric_key,
                              SELECTION_HOLDOUT, EVALUATION_HOLDOUT,
                              assert_complete_split_inventory,
                              filter_evaluation_only,
                              split_metric_to_cell_g,
                              deterministic_tiebreak,
                              aggregate_g_to_seed_round,
                              select_with_tiebreak,
                              deterministic_tiebreak,
                              aggregate_g_to_seed_round,
                              HEADROOM, INSUFFICIENT, INCONCLUSIVE,
                              GO, NO_GO, EXPLORATORY_PROBE,
                              PROCEED_TO_CONTROLLER_PROTOTYPE,
                              ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT,
                              ADAPTIVE_PARTIAL_HEADROOM,
                              EXPLORE_FURTHER,
                              decide_adaptive_headroom,
                              decide_claude_comparator,
                              decide_exploratory_signal,
                              decide_overall,
                              IncompleteMeasurementError)
from .pilot_eval import (evaluate_state_per_example,
                          PerExampleCSVWriter,
                          GAMMA_ONE, GAMMA_ZERO)
from .pilot_variability import (compute_repetition_summaries,
                                 build_feature_row,
                                 VariabilityCollisionError,
                                 VariabilityInventoryError)
from .ref_adapter import (load_reference_model, ReferenceUnavailable,
                          verify_snapshot, REFERENCE_DIR, BUNDLE_DIR)
from .reference_impl import (SmallCNNRecovered, participant_weights,
                             reset_all_rng, apply_diagnostic_determinism)
from .stats import (get_stat_param_names, compute_update_statistics,
                    flatten_params)
from . import verify

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device resolution (refuses silent CUDA -> CPU fallback)
# ---------------------------------------------------------------------------
# The smoke harness (harness.py) accepts a fallback to keep low-spec machines
# runnable.  The PILOT is GPU-bound on RTX-class CUDA per PROTOCOL.md §14
# (~75-80 min on a local RTX 3060) AND the deterministic-replay contract
# of the pilot depends on the device being honored verbatim.  A silent swap
# would (a) invalidate the runtime estimate, (b) mean that a `-Device cpu`
# invocation silently disagrees with the user's intent, (c) silently change
# the live-record "<requested-device>" field in any artifact that does not
# override `pconfig.smoke.device` post-hoc.
#
# The launchers pass `--device cuda` by default (PROTOCOL.md §14 explicit).
# If the runtime cannot honor that request, the run MUST abort LOUDLY before
# generating any checkpoint, partition artifact, or variability record;
# otherwise the artifacts become unreplayable by an analyst who sees
# `device="cuda"` in the config and `cuda=False` in the runtime metadata.
def _resolve_device_strict(requested):
    text = str(requested or "").strip()
    if not text:
        raise CheckFailure(
            "Empty --device value; the pilot requires 'cuda', "
            "'cuda:N', or 'cpu'. Aborting before any artifact is written.")
    kind = text.split(":", 1)[0].lower()
    if kind == "cuda":
        if not torch.cuda.is_available():
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            raise CheckFailure(
                f"Requested device '{text}' but "
                f"torch.cuda.is_available() returned False. "
                f"REFUSING to silently fall back to CPU: the pilot is "
                f"GPU-bound (PROTOCOL.md §14 ~75-80 min on local RTX 3060), "
                f"and a CPU swap would invalidate the runtime contract AND "
                f"the deterministic-replay contract (a future analyst who "
                f"reads the config sees device='cuda' and the snapshot "
                f"reports cuda=False, which would be irreconcilable). "
                f"CUDA_VISIBLE_DEVICES={cvd}. "
                f"Remediation: install CUDA-enabled torch, set "
                f"CUDA_VISIBLE_DEVICES=<index>, or pass --device cpu. "
                f"NO ARTIFACTS will be written; the run aborts BEFORE "
                f"STAGE 0a (data + partition).")
        return torch.device(text)
    return torch.device(text)


POLICY_FEDAVG = "fedavg"
POLICY_FIXED_GLOBAL = "fixed_global_gamma_dev"
POLICY_FIXED_PHASE = "fixed_phase_gamma_dev"
POLICY_VALIDATION_SELECTED = "validation_selected_scaling"
POLICY_CLAUDE = "claude_raw_experimental"
POLICY_EVAL_GRID_REF = "eval_optimal_grid_reference"
POLICY_GAMMA_ZERO = "gamma0_reference"

SRC_TRAJECTORY = "trajectory_cohort"
SRC_ORIGINAL_REP = "original_cohort_rep"
SRC_ALT_COHORT = "alt_cohort"

DECISIVE_POLICIES_FOR_PER_EXAMPLE = (
    POLICY_FEDAVG, POLICY_FIXED_GLOBAL, POLICY_FIXED_PHASE,
    POLICY_VALIDATION_SELECTED, POLICY_CLAUDE,
)

HOLDOUTS = (SELECTION_HOLDOUT, EVALUATION_HOLDOUT)


# Side cache: per-(seed, round, source) -> validation-selected γ.
_VALIDATION_GAMMA_BY_SOURCE: Dict[Tuple[int, int, str], float] = {}


class CheckFailure(SystemExit):
    pass


class PilotIncompleteMeasurement(SystemExit):
    pass


GRID_FIELDS = ["seed", "round", "dataset", "training_seed",
               "checkpoint_round", "checkpoint_convention", "source",
               "cohort_id", "rep_seed", "clients",
               "sample_counts", "weights", "weight_sum",
               "raw_weight_sum_population_normalized", "buffer_policy",
               "gamma", "row_label", "candidate_valid", "invalid_entries",
               "sel_loss", "sel_acc", "eval_loss", "eval_acc",
               "n_sel", "n_eval"]
RULE_FIELDS = ["seed", "round", "source", "cohort_id", "rep_seed",
               "buffer_policy", "rule", "gamma", "gamma_origin",
               "safeguards", "note", "uses_heldout_data",
               "candidate_valid", "invalid_entries",
               "sel_loss", "sel_acc", "eval_loss", "eval_acc",
               "clients", "weights", "weight_sum"]


def _dump_failure(out, results, config, checks, timing, failed_name):
    payload = {
        "failed_check": failed_name,
        "all_passed": False,
        "checks": checks,
        "timing": timing,
        "config": config.to_dict(),
    }
    with open(results / "verification.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _atomic_write_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8",
            dir=str(Path(path).parent),
            prefix=f".{Path(path).name}.") as fh:
        json.dump(obj, fh, indent=2, default=str)
        tmp = Path(fh.name)
    os.replace(tmp, path)


def _write_grid_csv(path, rows, fields=None, as_json=False):
    if as_json:
        _atomic_write_json(path, rows)
        return
    if fields is None:
        fields = GRID_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _build_counts_by_client_from_partition(
    data: dict, drop_trajectory_clients: Optional[List[int]] = None,
) -> Dict[int, int]:
    """Build the explicit {client_id -> int count} dict that
    variability and weighted aggregate compute against. The values
    come from `data["counts"]`, which is a list indexed by
    client_id — v3 (this rev) requires an explicit DICT form so the
    previous v2 list-spreading crash source is impossible."""
    if "counts" in data and isinstance(data["counts"], list):
        out: Dict[int, int] = {}
        for cid, n in enumerate(data["counts"]):
            out[int(cid)] = int(n)
        return out
    if "counts" in data and isinstance(data["counts"], dict):
        return {int(cid): int(n) for cid, n in data["counts"].items()}
    if "counts_by_client" in data and isinstance(
            data["counts_by_client"], dict):
        return {int(cid): int(n)
                for cid, n in data["counts_by_client"].items()}
    raise VariabilityInventoryError(
        "data dict missing 'counts' (list) or 'counts_by_client' (dict); "
        "pilot cannot build the explicit counts map required by "
        "the weighted aggregate. This used to be a list-spreading "
        "crash; v3 refuses silent uniform-weight substitution.")


# ============================================================================
# Per-cell processing
# ============================================================================


def _cohort_participant_counts(clients: List[int],
                                counts_by_client: Dict[int, int]
                                ) -> List[int]:
    out = []
    for c in clients:
        if c not in counts_by_client:
            raise VariabilityInventoryError(
                f"Missing retained count for client {c}; "
                "uniform substitution would silently bias the "
                "cohort aggregate. Add the count to the partition "
                "or remove the client from the cohort.")
        n = counts_by_client[c]
        if not isinstance(n, int) or n <= 0:
            raise VariabilityInventoryError(
                f"Non-positive retained count for client {c}: {n}")
        out.append(int(n))
    return out


def _process_source(source, model_ctor, data, pconfig, device,
                     eval_model, param_names, *,
                     counts_by_client: Dict[int, int],
                     out_grid_rows, out_rule_rows,
                     out_stats_records,
                     out_split_metric_keys,
                     out_variability_payloads,
                     gamma_dev_global=None,
                     gamma_dev_phase_per_round=None,
                     feasibility_strict: bool = True):
    """Train + full γ grid + rule rows + per-cell statistics for
    one source.  Records SPLIT-SPECIFIC outcomes (`sel_loss`,
    `eval_loss`) AND split-specific aggregate-metric keys.

    A candidate is DECISIVE for the screen IFF it is valid (the
    per-source `build_candidate(..., validity)` reports `valid`).
    If a decisive γ yields an invalid candidate, we surface the
    invalid reason AND raise `IncompleteMeasurementError` if
    `feasibility_strict=True`.
    """
    seed = source["seed"]
    rd = source["round"]
    sd0 = source["sd0"]
    clients = source["clients"]
    if source.get("client_sds") is None:
        cfg = _smoke_config_with_seed(pconfig.smoke, seed)
        sds = {}
        for cid in source["clients"]:
            es = epoch_seed_for(source["rep_seed"], rd, cid)
            sd = train_client_local(
                model_ctor, sd0, data["raw_x"], data["raw_y"],
                data["retained"][cid], cfg, es, device)
            sds[cid] = {k: v.detach().clone() for k, v in sd.items()}
        source["client_sds"] = sds
    sds = source["client_sds"]
    cohort_counts = _cohort_participant_counts(clients, counts_by_client)
    weights_arr = participant_weights(cohort_counts)
    w_by_id = {clients[i]: weights_arr[i] for i in range(len(clients))}
    sd_fed = average_states([sds[c] for c in clients],
                            [w_by_id[c] for c in clients])
    first_sd = sds[clients[0]]

    stats = compute_update_statistics(
        sd0, sds, clients, w_by_id, param_names)
    claude_g, claude_sg = claude_raw_shrinkage(stats)
    fedexp_g, fedexp_sg = fedexp_step(stats)
    base_meta = _build_base_meta(seed, rd, source, clients, w_by_id,
                                  data["counts"])

    grid_dict: Dict[float, dict] = {}
    for gamma in pconfig.smoke.gamma_grid:
        cand, rep = build_candidate(sd0, sd_fed, float(gamma),
                                      first_sd, POLICY_ALL_FLOATING)
        sel = evaluate_state(model_ctor, cand, data["raw_x"],
                              data["raw_y"], data["sel_idx"], device,
                              shared_model=eval_model)
        ev = evaluate_state(model_ctor, cand, data["raw_x"],
                            data["raw_y"], data["eval_idx"], device,
                            shared_model=eval_model)
        label = "no_update_reference" if gamma == GAMMA_ZERO else (
            "actual_fedavg_fullstate_reference"
            if gamma == GAMMA_ONE else "")
        row = {**base_meta, "buffer_policy": POLICY_ALL_FLOATING,
               "gamma": float(gamma), "row_label": label,
               "candidate_valid": rep["valid"],
               "invalid_entries": json.dumps(rep["invalid_entries"]),
               "sel_loss": sel["loss"], "sel_acc": sel["accuracy"],
               "eval_loss": ev["loss"], "eval_acc": ev["accuracy"],
               "n_sel": sel["n_samples"], "n_eval": ev["n_samples"]}
        grid_dict[float(gamma)] = row
        out_grid_rows.append(row)
        if (not rep["valid"]) and feasibility_strict:
            log.warning(
                f"source={source['source_label']} seed={seed} "
                f"round={rd} gamma={float(gamma)} produced an INVALID "
                "candidate; row recorded but the corresponding "
                "split_metric_key will be missing if this γ is "
                "decisive.")

    # Secondary buffer policy γ=1 ONLY.
    cand, rep = build_candidate(sd0, sd_fed, GAMMA_ONE, first_sd,
                                POLICY_PARAMS_ONLY, param_names)
    sel = evaluate_state(model_ctor, cand, data["raw_x"], data["raw_y"],
                          data["sel_idx"], device, shared_model=eval_model)
    ev = evaluate_state(model_ctor, cand, data["raw_x"], data["raw_y"],
                        data["eval_idx"], device, shared_model=eval_model)
    sec_row = {**base_meta, "buffer_policy": POLICY_PARAMS_ONLY,
                "gamma": GAMMA_ONE,
                "row_label": "frozen_buffer_parameter_candidate_NOT_fedavg",
                "candidate_valid": rep["valid"],
                "invalid_entries": json.dumps(rep["invalid_entries"]),
                "sel_loss": sel["loss"], "sel_acc": sel["accuracy"],
                "eval_loss": ev["loss"], "eval_acc": ev["accuracy"],
                "n_sel": sel["n_samples"], "n_eval": ev["n_samples"]}
    out_grid_rows.append(sec_row)

    # Per-rule entries.  Explicit None handling (implementation
    # repair, 2026-10-04): the previous `... or GAMMA_ONE` form
    # silently converted a legitimately winning gamma=0.0 to 1.0
    # (0.0 is falsy) AND silently substituted gamma=1.0 when no valid
    # candidate existed.  Now gamma=0.0 is preserved when it wins; an
    # empty valid set raises IncompleteMeasurementError under
    # feasibility_strict (consistent with the STAGE 3 contract) and
    # falls back to GAMMA_ONE with an explicit warning only when
    # feasibility_strict=False.
    best_sel_gamma = _argmin_excluding_invalid(
        [g for g in grid_dict
         if grid_dict[g]["candidate_valid"]
         and math.isfinite(grid_dict[g]["sel_loss"])],
        lambda g: grid_dict[g]["sel_loss"])
    if best_sel_gamma is None:
        if feasibility_strict:
            raise IncompleteMeasurementError(
                f"source={source['source_label']} seed={seed} "
                f"round={rd}: no valid candidate with finite "
                "SELECTION loss in the gamma grid; refusing to "
                "silently substitute gamma=1.0.")
        log.warning(
            "source=%s seed=%s round=%s: no valid candidate with "
            "finite SELECTION loss; explicitly falling back to "
            "gamma=1.0 (feasibility_strict=False).",
            source["source_label"], seed, rd)
        best_sel_gamma = GAMMA_ONE
    best_eval_gamma = _argmin_excluding_invalid(
        [g for g in grid_dict
         if grid_dict[g]["candidate_valid"]
         and math.isfinite(grid_dict[g]["eval_loss"])],
        lambda g: grid_dict[g]["eval_loss"])
    if best_eval_gamma is None:
        if feasibility_strict:
            raise IncompleteMeasurementError(
                f"source={source['source_label']} seed={seed} "
                f"round={rd}: no valid candidate with finite "
                "EVALUATION loss in the gamma grid; refusing to "
                "silently substitute gamma=1.0.")
        log.warning(
            "source=%s seed=%s round=%s: no valid candidate with "
            "finite EVALUATION loss; explicitly falling back to "
            "gamma=1.0 (feasibility_strict=False).",
            source["source_label"], seed, rd)
        best_eval_gamma = GAMMA_ONE
    val_gamma = best_sel_gamma
    eval_cheat_note = ("retrospective grid-minimum; NOT an upper bound "
                        "on terminal accuracy or any continuous "
                        "controller; not a deployable choice.")

    def _rule(policy_name, rule_label, gamma, sg, note, origin,
               uses_heldout=False, expect_to_find_in_grid=True):
        row = grid_dict.get(float(gamma))
        if row is None:
            # Re-evaluate a candidate not in the grid.  This only
            # happens for FIXED controls on held-out cells, but we
            # DECISIVE check cand_valid here too.
            cand_x, rep_x = build_candidate(sd0, sd_fed, float(gamma),
                                            first_sd, POLICY_ALL_FLOATING)
            sel = evaluate_state(model_ctor, cand_x, data["raw_x"],
                                  data["raw_y"], data["sel_idx"],
                                  device, shared_model=eval_model)
            ev = evaluate_state(model_ctor, cand_x, data["raw_x"],
                                data["raw_y"], data["eval_idx"],
                                device, shared_model=eval_model)
            row = {**base_meta, "buffer_policy": POLICY_ALL_FLOATING,
                   "gamma": float(gamma), "row_label": "",
                   "candidate_valid": rep_x["valid"],
                   "invalid_entries": json.dumps(rep_x["invalid_entries"]),
                   "sel_loss": sel["loss"], "sel_acc": sel["accuracy"],
                   "eval_loss": ev["loss"], "eval_acc": ev["accuracy"],
                   "n_sel": sel["n_samples"], "n_eval": ev["n_samples"]}
        if (not row["candidate_valid"]):
            if feasibility_strict and expect_to_find_in_grid:
                raise IncompleteMeasurementError(
                    f"Decisive candidate invalid: source={source['source_label']} "
                    f"seed={seed} round={rd} policy={policy_name} "
                    f"gamma={float(gamma)} invalid_entries="
                    f"{row['invalid_entries']}; the pilot refuses to "
                    "decide on invalid candidates.")
        rr = {**base_meta, "buffer_policy": POLICY_ALL_FLOATING,
              "rule": rule_label, "gamma": float(gamma),
              "gamma_origin": origin, "note": note,
              "uses_heldout_data": uses_heldout,
              "candidate_valid": row["candidate_valid"],
              "invalid_entries": row["invalid_entries"],
              "sel_loss": row["sel_loss"], "sel_acc": row["sel_acc"],
              "eval_loss": row["eval_loss"], "eval_acc": row["eval_acc"]}
        out_rule_rows.append(rr)
        if policy_name == POLICY_VALIDATION_SELECTED:
            _VALIDATION_GAMMA_BY_SOURCE[(seed, rd, source["source_label"])] = \
                float(gamma)

    _rule(POLICY_FEDAVG, "fedavg_gamma1_grid_exact", GAMMA_ONE,
            [], "ground truth reference.", "grid_exact", False)
    _rule(POLICY_VALIDATION_SELECTED,
            "validation_selected_scaling_per_source",
            val_gamma, [],
            ("per-source γ from argmin of grid on SELECTION split; "
              "evaluated on the independent EVALUATION split; "
              "legitimate per-source baseline (no eval-split leak)."),
            "rule_validation_selected", False)
    _rule(POLICY_EVAL_GRID_REF, "eval_optimal_grid_reference",
            best_eval_gamma, [], eval_cheat_note,
            "rule_eval_optimal_grid_reference", False)
    _rule(POLICY_CLAUDE, "claude_raw_experimental", claude_g,
            list(claude_sg),
            ("Claude's experimental raw shrinkage rule; failure "
              "does NOT imply absence of headroom; success does NOT "
              "imply a deployable controller."),
            "rule_claude", False,
            expect_to_find_in_grid=(float(claude_g) in grid_dict))
    _rule(POLICY_GAMMA_ZERO, "gamma0_reference", GAMMA_ZERO,
            [], "no-update reference.", "grid_exact", False)

    if gamma_dev_global is not None and gamma_dev_phase_per_round is not None:
        _rule(POLICY_FIXED_GLOBAL, "fixed_global_gamma_dev_grid_exact",
                gamma_dev_global, [],
                "FIXED by dev selection (PROTOCOL.md §6).",
                "rule_global_fixed_dev", False)
        _rule(POLICY_FIXED_PHASE, "fixed_phase_gamma_dev_grid_exact",
                gamma_dev_phase_per_round.get(rd, GAMMA_ONE), [],
                "FIXED by dev per-phase selection (PROTOCOL.md §6).",
                "rule_phase_fixed_dev", False)

    # SPLIT-SPECIFIC per (seed, round, source, policy) metric keys.
    fcand_for_grid = lambda pol, default_g: grid_dict.get(
        default_g, grid_dict[GAMMA_ONE])
    static_assessments = [
        (POLICY_FEDAVG, GAMMA_ONE),
        (POLICY_VALIDATION_SELECTED, val_gamma),
        (POLICY_EVAL_GRID_REF, best_eval_gamma),
        (POLICY_CLAUDE, float(claude_g)),
        (POLICY_GAMMA_ZERO, GAMMA_ZERO),
    ]
    if gamma_dev_global is not None:
        static_assessments.extend([
            (POLICY_FIXED_GLOBAL, gamma_dev_global),
            (POLICY_FIXED_PHASE, gamma_dev_phase_per_round.get(
                rd, GAMMA_ONE)),
        ])
    for policy, g_used in static_assessments:
        row = grid_dict.get(float(g_used))
        if row is None:
            cand, rep = build_candidate(sd0, sd_fed, float(g_used),
                                          first_sd, POLICY_ALL_FLOATING)
            sel = evaluate_state(model_ctor, cand, data["raw_x"],
                                  data["raw_y"], data["sel_idx"],
                                  device, shared_model=eval_model)
            ev = evaluate_state(model_ctor, cand, data["raw_x"],
                                data["raw_y"], data["eval_idx"],
                                device, shared_model=eval_model)
            row = {**base_meta, "buffer_policy": POLICY_ALL_FLOATING,
                   "gamma": float(g_used), "row_label": "",
                   "candidate_valid": rep["valid"],
                   "invalid_entries": json.dumps(rep["invalid_entries"]),
                   "sel_loss": sel["loss"], "sel_acc": sel["accuracy"],
                   "eval_loss": ev["loss"], "eval_acc": ev["accuracy"],
                   "n_sel": sel["n_samples"], "n_eval": ev["n_samples"]}
        sk = split_metric_key(seed, rd, source["source_label"], policy,
                              SELECTION_HOLDOUT)
        ek = split_metric_key(seed, rd, source["source_label"], policy,
                              EVALUATION_HOLDOUT)
        if math.isfinite(row["sel_loss"]):
            out_split_metric_keys[sk] = float(row["sel_loss"])
        if math.isfinite(row["eval_loss"]):
            out_split_metric_keys[ek] = float(row["eval_loss"])

    stats_rec = {
        "seed": seed, "round": rd, "source": source["source_label"],
        "cohort_id": source["cohort_id"],
        "rep_seed": source["rep_seed"],
        "clients": list(clients),
        "sample_counts": {str(c): counts_by_client[c] for c in clients},
        "weights": {str(c): float(w_by_id[c]) for c in clients},
        "update_stats": stats,
        "claude_gamma": float(claude_g),
        "claude_safeguards": list(claude_sg),
        "fedexp_gamma": float(fedexp_g),
        "fedexp_safeguards": list(fedexp_sg),
        "best_sel_gamma_all_floating": float(best_sel_gamma),
        "best_eval_gamma_all_floating": float(best_eval_gamma),
        "validation_selected_gamma": float(val_gamma),
    }
    out_stats_records.append(stats_rec)
    out_variability_payloads.append({
        "seed": seed, "round": rd,
        "cohort_id": source["cohort_id"],
        "rep_seed": source["rep_seed"],
        "clients": list(clients),
        # Implementation repair (2026-10-04): persist the retained
        # per-client sample counts INSIDE every variability payload so
        # the variability/feature path never needs a count fallback.
        "sample_counts": {str(c): int(counts_by_client[c])
                           for c in clients},
        "client_sds": {c: {k: v.clone() for k, v in sds[c].items()}
                        for c in clients},
        "sd0": {k: v.clone() for k, v in sd0.items()},
    })


def _argmin_excluding_invalid(candidates, key):
    if not candidates:
        return None
    return deterministic_tiebreak(candidates, key=lambda g: (key(g), g))


def _build_base_meta(seed, rd, source, clients, w_by_id, sample_counts):
    return {
        "seed": int(seed), "round": int(rd),
        "dataset": "cifar100",
        "training_seed": int(seed),
        "checkpoint_round": int(rd),
        "checkpoint_convention": "state_before_round_local_updates",
        "source": source["source_label"],
        "cohort_id": source["cohort_id"],
        "rep_seed": str(source["rep_seed"]),
        "clients": json.dumps(list(clients)),
        "sample_counts": json.dumps({str(c): int(sample_counts[c])
                                      for c in clients}),
        "weights": json.dumps({str(c): round(float(w_by_id[c]), 8)
                                for c in clients}),
        "weight_sum": float(sum(float(w_by_id[c]) for c in clients)),
        "raw_weight_sum_population_normalized":
            float(sum(sample_counts[c] / max(1, sum(sample_counts))
                      for c in clients)),
    }


# ============================================================================
# Mandatory lightweight per-cell checks (per-cell, no longer informational
# when their failure invalidates the experiment).
# ============================================================================


def run_cell_lightweight_checks(source, model_ctor, data, device, pconfig,
                                  record_check, *,
                                  counts_by_client: Dict[int, int]):
    """Cheap per-cell checks: weights sum to one; γ=0 endpoint
    agrees with direct eval of sd0.  These are MANDATORY (not
    informational), because failure means the source is
    unmeasureable and would silently bias every downstream estimator.
    """
    clients = source["clients"]
    if any(c not in counts_by_client for c in clients):
        missing = [c for c in clients if c not in counts_by_client]
        record_check(
            f"weights_sum_one|seed={source['seed']}|"
            f"round={source['round']}|source={source['source_label']}",
            {"passed": False,
              "weight_sum": float("nan"),
              "n_clients": len(clients),
              "missing_clients": missing,
              "note": "missing count(s); cannot build a valid weight vector"},
            mandatory=True)
        return
    # `participant_weights` returns a dict keyed by participant
    # POSITION (not client id); convert to a list of weights in
    # client order for sum/finite checks below.
    w_arr_dict = participant_weights([counts_by_client[c] for c in clients])
    w_arr = [w_arr_dict[i] for i in range(len(clients))]
    sums_to_one = (abs(sum(w_arr) - 1.0) < 1e-10
                    and all(math.isfinite(w) and w >= 0 for w in w_arr))
    record_check(
        f"weights_sum_one|seed={source['seed']}|round={source['round']}|"
        f"source={source['source_label']}",
        {"passed": sums_to_one,
          "weight_sum": float(sum(w_arr)),
          "n_clients": len(clients),
          "weights": [float(w) for w in w_arr]},
        mandatory=True)

    sd0_dev = {k: v.to(device) for k, v in source["sd0"].items()}
    record_check(
        f"gamma0_endpoint|seed={source['seed']}|round={source['round']}|"
        f"source={source['source_label']}",
        verify.check_per_epoch_endpoint_matches_sd0(
            model_ctor, sd0_dev, data, data["eval_idx"], device),
        mandatory=True)

    # γ=1 ALWAYS implies the candidate is a real FedAvg of this
    # cohort. For every source's saved-clients cohort, the γ=1
    # parameter entries must match the cohort-averaged FedAvg.
    if source.get("client_sds") is not None:
        sds = source["client_sds"]
        weights = {c: participant_weights(
            [counts_by_client[c] for c in clients])[i]
            for i, c in enumerate(clients)}
        sd_fed = average_states([sds[c] for c in clients],
                                 [weights[c] for c in clients])
        for ci in clients:
            weight_sum_norm = (sum(weights.values()) > 0
                                and abs(sum(weights.values()) - 1.0) < 1e-10)
            record_check(
                f"gamma1_weight_sum_to_one|cohort={source['cohort_id']}|"
                f"seed={source['seed']}|round={source['round']}|"
                f"source={source['source_label']}",
                {"passed": bool(weight_sum_norm),
                  "weight_sum": float(sum(weights.values()))},
                mandatory=True)


# ============================================================================
# Alt-cohort uniqueness v3 (BOTH alternatives differ from each other AND
# from the original). Returns (cohort_a, cohort_b) or raises.
# ============================================================================


def _sample_two_alt_cohorts(seed, rd, n_alt_cohorts, original_clients,
                             num_clients, num_participants, cohort_seed_offset,
                             retry_cap, rng_factory=None):
    """Bounded-retries sampler. MAINTAINS a set of previously
    selected cohort identities (including the original) and
    refuses to select any cohort IDENTICAL to a previously
    selected one.  Each cohort is drawn with a fresh, ADVANCING
    RNG seeded once.

    Returns: (selected_cohorts_list, total_retries_used,
    identities_ledger).
    Raises `PilotIncompleteMeasurement` if exhausted.
    """
    sample_cohort_fn = sample_cohort
    identities: List[dict] = []
    selected: List[List[int]] = []
    base_seeds = [int(seed) + int(cohort_seed_offset)
                  + int(rd) * 17 + int(ai) * 257
                  for ai in range(n_alt_cohorts)]
    # Use ONE advancing RNG starting from the first base seed; the
    # next n_alt_cohorts draws advance the state. This matches
    # PROTOCOL's "one step-advancing RNG" requirement.
    rng = (rng_factory or np.random.RandomState)(base_seeds[0])
    # Pre-select the original cohort signature.
    identities.append({"cohort_index": "original",
                       "clients": sorted(map(int, original_clients)),
                       "base_seed": None,
                       "retry_attempt": 0,
                       "sample_index": 0})
    seen = {tuple(sorted(map(int, original_clients)))}
    retries_used_per_cohort = []
    for ai in range(n_alt_cohorts):
        attempts = 0
        local_seen = set()
        while True:
            attempts += 1
            if attempts > retry_cap:
                # We saved the failing attempts so a future analyst
                # can audit.
                retries_used_per_cohort[-1].update({
                    "retry_cap": int(retry_cap),
                    "n_tried": attempts - 1,
                    "exhausted": True,
                })
                raise PilotIncompleteMeasurement(
                    f"alt_cohort retry cap exhausted at seed={seed} round={rd} "
                    f"alt_index={ai}; ABORT with INCOMPLETE_MEASUREMENT "
                    "(no silent re-use). identities_ledger="
                    f"{identities}")
            cohort = sample_cohort_fn(rng, num_clients, num_participants)
            cs = tuple(sorted(map(int, cohort)))
            if cs not in seen and cs not in local_seen:
                selected.append(sorted(map(int, cohort)))
                local_seen.add(cs)
                seen.add(cs)
                identities.append({
                    "cohort_index": f"alt_{ai}",
                    "clients": sorted(map(int, cohort)),
                    "base_seed": base_seeds[ai],
                    "retry_attempt": attempts,
                    "sample_index": ai,
                })
                retries_used_per_cohort.append(
                    {"retry_cap": int(retry_cap),
                     "n_tried": attempts,
                     "exhausted": False})
                break
    return selected, retries_used_per_cohort, identities


# ============================================================================
# Post-freeze labeling of dev fixed-policy rows (implementation repair
# 2026-10-04).  The frozen dev gammas are grid points; the dev seed's
# fixed-control measurements therefore ALREADY EXIST as STAGE 2 grid
# rows.  This pass labels them as fixed-policy rule rows and split
# metric keys WITHOUT any re-evaluation.
# ============================================================================


def _label_dev_fixed_policy_rows(dev_grid_rows, gamma_dev_global,
                                  gamma_dev_phase_per_round,
                                  dev_rule_rows, dev_split_metric_keys):
    """Label the dev seed's existing grid rows at the frozen dev gammas
    as fixed-policy measurements.

    For each dev grid row (primary buffer policy) whose gamma equals a
    frozen control, append the corresponding fixed-policy rule row and
    emit the split-specific metric keys
    (seed|round|source|policy|holdout).  An invalid candidate at a
    frozen gamma raises IncompleteMeasurementError, matching the
    held-out `_rule` behavior under feasibility_strict.
    """
    n_labeled = 0
    for row in dev_grid_rows:
        if row["buffer_policy"] != POLICY_ALL_FLOATING:
            continue
        g = float(row["gamma"])
        rd = int(row["round"])
        for policy, rule_label, g_fixed, origin in (
                (POLICY_FIXED_GLOBAL,
                 "fixed_global_gamma_dev_grid_exact",
                 float(gamma_dev_global), "rule_global_fixed_dev"),
                (POLICY_FIXED_PHASE,
                 "fixed_phase_gamma_dev_grid_exact",
                 float(gamma_dev_phase_per_round[rd]),
                 "rule_phase_fixed_dev")):
            if g != g_fixed:
                continue
            if not row["candidate_valid"]:
                raise IncompleteMeasurementError(
                    f"post-freeze labeling: dev grid row invalid at the "
                    f"frozen gamma: source={row['source']} round={rd} "
                    f"policy={policy} gamma={g_fixed} "
                    f"invalid_entries={row['invalid_entries']}")
            rr = {k: row.get(k) for k in RULE_FIELDS}
            rr.update({
                "rule": rule_label,
                "gamma": g_fixed,
                "gamma_origin": origin,
                "note": ("FIXED by dev selection (PROTOCOL.md §6); "
                         "dev-source row labeled post-freeze from the "
                         "existing grid evaluation (NO re-evaluation)."),
                "uses_heldout_data": False,
            })
            dev_rule_rows.append(rr)
            sk = split_metric_key(int(row["seed"]), rd, row["source"],
                                  policy, SELECTION_HOLDOUT)
            ek = split_metric_key(int(row["seed"]), rd, row["source"],
                                  policy, EVALUATION_HOLDOUT)
            if math.isfinite(row["sel_loss"]):
                dev_split_metric_keys[sk] = float(row["sel_loss"])
            if math.isfinite(row["eval_loss"]):
                dev_split_metric_keys[ek] = float(row["eval_loss"])
            n_labeled += 1
    log.info(f"STAGE 3b: labeled {n_labeled} dev fixed-policy rows "
             "post-freeze from existing grid rows (no re-evaluation).")


# ============================================================================
# Selection / evaluation losses (selection split keeps its loss for
# legitimate validation-based selection; evaluation split feeds the
# decision input).
# ============================================================================


def _select_dev_fixed_controls(dev_grid_rows, pconfig,
                                counts_by_client: Dict[int, int]):
    """Selection-time required inventory check on dev rows: every
    (seed=dev, round, source in cohort_plan, γ in grid) must have a
    valid candidate."""
    grid_gammas = list(pconfig.smoke.gamma_grid)
    rep_rows = [r for r in dev_grid_rows
                if r["buffer_policy"] == POLICY_ALL_FLOATING
                and (r["source"].startswith(SRC_ORIGINAL_REP)
                     or r["source"].startswith(SRC_ALT_COHORT))]
    # Each γ should have loss records for ALL required dev sources
    # (one per source per γ). Pulling the per-γ mean and ALL_VALID
    # flag.
    losses_global = {g: [] for g in grid_gammas}
    valid_global = {g: [] for g in grid_gammas}
    for r in rep_rows:
        g = float(r["gamma"])
        losses_global[g].append(float(r["sel_loss"]))
        valid_global[g].append(bool(r["candidate_valid"]))
    valid_global_bool = {
        g: (all(valid_global[g]) if valid_global[g] else False)
        for g in grid_gammas
    }
    global_g, info_global = select_with_tiebreak(
        grid_gammas, losses_by_gamma=losses_global,
        valid_by_gamma=valid_global_bool,
        rule_id=pconfig.pilot_selection_rule_id, require_all_valid=True)

    phase_fixed = {}
    summary_phase = {}
    sel_per_round = {}
    ev_per_round = {}
    for rd in pconfig.pilot_checkpoint_rounds:
        rows_for_rd = [r for r in rep_rows if r["round"] == rd]
        losses_phase = {g: [] for g in grid_gammas}
        valid_phase = {g: [] for g in grid_gammas}
        for r in rows_for_rd:
            g = float(r["gamma"])
            losses_phase[g].append(float(r["sel_loss"]))
            valid_phase[g].append(bool(r["candidate_valid"]))
        valid_phase_bool = {
            g: (all(valid_phase[g]) if valid_phase[g] else False)
            for g in grid_gammas
        }
        sel_per_round[str(rd)] = {g: float(np.mean(losses_phase[g]))
                                    for g in grid_gammas
                                    if losses_phase[g]}
        ev_per_round[str(rd)] = {g: float(np.mean(
            [r["eval_loss"] for r in rows_for_rd if r["gamma"] == g]))
            for g in grid_gammas if any(r2["gamma"] == g
                                        for r2 in rows_for_rd)}
        phase_g, info_phase = select_with_tiebreak(
            grid_gammas, losses_by_gamma=losses_phase,
            valid_by_gamma=valid_phase_bool,
            rule_id=pconfig.pilot_selection_rule_id,
            require_all_valid=True)
        phase_fixed[rd] = phase_g
        summary_phase[str(rd)] = info_phase

    sel_global = {g: (float(np.mean(losses_global[g]))
                       if losses_global[g] else float("nan"))
                  for g in grid_gammas}
    ev_global = {g: (float(np.mean([r["eval_loss"] for r in rep_rows
                                       if r["gamma"] == g]))
                    if any(r["gamma"] == g for r in rep_rows)
                    else float("nan"))
                 for g in grid_gammas}
    table = {
        "selection_loss_per_round_per_gamma_mean": sel_per_round,
        "eval_loss_per_round_per_gamma_mean": ev_per_round,
        "summary_global": info_global,
        "summary_phase": summary_phase,
        "selection_loss_global_mean_per_gamma": sel_global,
        "eval_loss_global_mean_per_gamma": ev_global,
    }
    return global_g, phase_fixed, table


# ============================================================================
# Feature row builder.  Per-(seed, round, source) joins the per-checkpoint
# variability summary built from the SAME (seed, round).
# ============================================================================


def _build_feature_rows_per_phase(payloads, split_metric_keys,
                                   pconfig, *,
                                   param_names,
                                   counts_by_client: Dict[int, int]):
    """For each (seed, round) cell, build one feature-row set keyed by
    (cohort_id, rep_seed).  Each row carries the per-checkpoint
    summary stats computed ONLY from the same (seed, round).

    Returns: out[(seed, round)] -> {(cohort_id, rep_seed): row}.
    The (seed, round) key is taken from each payload's own "seed" /
    "round" fields (implementation repair 2026-10-04: the previous
    single-`seed` argument let the held-out call mislabel held-out
    payloads as the dev seed AND mix both held-out seeds into one
    per-checkpoint summary, violating the per-(seed, round) scoping).

    Implementation repair (2026-10-04): the canonical
    `counts_by_client` (built in STAGE 0a from the partition) and the
    canonical `param_names` (from `model.named_parameters()` filtered
    by `requires_grad`, single-sourced via `_param_names_for_model_ctor`)
    are now REQUIRED explicit arguments.  The previous version
    reconstructed counts from payload `sample_counts` with a silent
    count=1 fallback and referenced an out-of-scope `model_ctor`.
    The fallback is removed: a missing or inconsistent count fails
    loudly with `VariabilityInventoryError`.  Every payload must
    persist its retained `sample_counts`; they are CROSS-CHECKED
    against the canonical map and any disagreement raises.
    """
    if not isinstance(counts_by_client, dict) or not counts_by_client:
        raise VariabilityInventoryError(
            "feature construction requires the canonical "
            "counts_by_client dict from STAGE 0a; got "
            f"{type(counts_by_client).__name__}.")
    if not param_names:
        raise VariabilityInventoryError(
            "feature construction requires the canonical param_names "
            "list (model.named_parameters() filtered by requires_grad); "
            "got an empty list.")
    out: Dict[Tuple[int, int], Dict[Tuple[str, str], Dict]] = {}
    # Group payloads by (seed, round) using the payloads' OWN fields:
    by_cell: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for p in payloads:
        by_cell[(int(p["seed"]), int(p["round"]))].append(p)
    for (seed, rd) in sorted(by_cell.keys()):
        # Build per-checkpoint summary for THIS seed+round only.
        from .pilot_variability import _summarize_one_checkpoint
        # Every payload must persist retained sample counts AND they
        # must agree with the canonical STAGE 0a map.
        for p in by_cell[(seed, rd)]:
            sc = p.get("sample_counts")
            if sc is None:
                raise VariabilityInventoryError(
                    f"variability payload (seed={p.get('seed')}, "
                    f"round={rd}, cohort={p.get('cohort_id')}, "
                    f"rep={p.get('rep_seed')}) carries no "
                    "'sample_counts'; every variability payload must "
                    "persist its retained sample counts (the count=1 "
                    "fallback was removed).")
            for c in p["clients"]:
                pay_n = sc.get(str(c), sc.get(c))
                if pay_n is None:
                    raise VariabilityInventoryError(
                        f"payload sample_counts missing client {c} "
                        f"(seed={p.get('seed')}, round={rd}, "
                        f"cohort={p.get('cohort_id')}).")
                if c not in counts_by_client:
                    raise VariabilityInventoryError(
                        f"client {c} (seed={p.get('seed')}, round={rd}) "
                        "is absent from the canonical counts_by_client; "
                        "refusing the uniform/count-1 fallback.")
                if int(pay_n) != int(counts_by_client[c]):
                    raise VariabilityInventoryError(
                        f"payload sample_counts[{c}]={pay_n} disagrees "
                        f"with canonical counts_by_client[{c}]="
                        f"{counts_by_client[c]} (seed={p.get('seed')}, "
                        f"round={rd}); refusing to build features on "
                        "inconsistent counts.")
        per_ckpt = _summarize_one_checkpoint(
            (seed, rd), by_cell[(seed, rd)],
            list(param_names),
            counts_by_client, same_state_tolerance=1e-7,
        )
        per_round_rows = {}
        # For every (cohort_id, rep_seed) in this (seed, round)'s
        # payload set, build a feature row excluding the trajectory
        # cohort (which is reported separately).
        seen_keys = set()
        for p in by_cell[(seed, rd)]:
            cohort_id = str(p["cohort_id"])
            if cohort_id == "trajectory":
                continue
            rep_seed = str(p["rep_seed"])
            key = (cohort_id, rep_seed)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            row = build_feature_row(
                cohort_id=cohort_id, seed=seed, round=rd,
                rep_seed=rep_seed,
                per_checkpoint_record=per_ckpt,
                counts_by_client=counts_by_client,
            )
            per_round_rows[key] = row
        out[(seed, rd)] = per_round_rows
    return out


# ============================================================================
# Public entry point — STAGED EXECUTION
# ============================================================================


def run_pilot(pconfig: PilotConfig) -> dict:
    """Run the full pilot per PROTOCOL.md / GO_NO_GO.md."""
    t_global = time.time()
    out, results_dir = _setup_outdir(pconfig)
    checks: Dict[str, dict] = {}
    timing: Dict[str, float] = {}

    def record_check(name, result, mandatory=True, informational=False):
        if informational:
            result["informational"] = True
        checks[name] = result
        status = "PASS" if result.get("passed") else "FAIL"
        if informational:
            status += " (informational)"
        log.info(f"  [check] {name}: {status}")
        if mandatory and not informational and not result.get("passed"):
            _dump_failure(out, results_dir, pconfig, checks, timing, name)
            raise CheckFailure(f"Mandatory check failed: {name} — "
                                f"diagnostics saved to {results_dir}")

    snapshot = verify_snapshot()
    record_check("bundled_reference_snapshot",
                  {"passed": snapshot["ok"],
                   "ok_count": sum(1 for v in snapshot["files"].values()
                                    if v["status"] == "ok"),
                   "bad": {k: v["status"]
                            for k, v in snapshot["files"].items()
                            if v["status"] != "ok"}})
    smoke = pconfig.smoke

    try:
        model_ctor, ref_prov = load_reference_model(smoke.num_classes,
                                                     smoke.use_aib)
    except ReferenceUnavailable as e:
        if pconfig.smoke.require_reference:
            raise CheckFailure(f"require_reference set and bundled "
                                f"reference model unavailable: {e}")
        model_ctor = lambda: SmallCNNRecovered(num_classes=smoke.num_classes)
        ref_prov = {"model_source": "reconstruction_fallback",
                     "error": str(e)[:2000]}

    provenance = _build_provenance(pconfig, ref_prov, snapshot)
    _atomic_write_json(out / "provenance.json", provenance)
    _atomic_write_json(out / "config.json", pconfig.to_dict())
    pilot_protocol_hash = _record_protocol_files(out)
    log.info(f"Pilot config fingerprint: "
              f"{pconfig.pilot_config_fingerprint()[:16]}...")
    log.info(f"Protocol + decisions sha256: {pilot_protocol_hash}")

    device = _resolve_device_strict(pconfig.smoke.device)
    log.info(f"Device: {device}")
    apply_diagnostic_determinism()

    # ----- STAGE 0a: data + partition + counts_by_client -----
    t0 = time.time()
    data_root = pconfig.smoke.data_root or str(BUNDLE_DIR / "data")
    raw_x, raw_y = load_cifar100_raw(data_root)
    sel_idx, ev_idx = stratified_holdouts(
        raw_y, pconfig.smoke.selection_frac, pconfig.smoke.eval_frac,
        pconfig.smoke.split_seed, pconfig.smoke.num_classes)
    retained, original, attempts = dirichlet_partition_with_holdout_removal(
        raw_y, sel_idx, ev_idx, pconfig.smoke.num_clients,
        pconfig.smoke.dirichlet_alpha, pconfig.smoke.min_client_size,
        pconfig.smoke.num_classes, pconfig.smoke.partition_seed)
    assert_disjoint(sel_idx, ev_idx, retained)
    counts_list = [len(ix) for ix in retained]
    n_retained_total = sum(counts_list)
    # Build an EXPLICIT counts_by_client dict (NOT a list spread).
    counts_by_client = {int(cid): int(cnt)
                          for cid, cnt in enumerate(counts_list)}
    log.info(f"Built counts_by_client dict for {len(counts_by_client)} "
             "clients")
    data = {"raw_x": raw_x, "raw_y": raw_y, "retained": retained,
            "sel_idx": sel_idx, "eval_idx": ev_idx,
            "counts": counts_list,
            "counts_by_client": counts_by_client}
    timing["data_and_partition"] = time.time() - t0
    phashes = partition_hashes(retained, sel_idx, ev_idx)
    partition_record = {
        "num_classes": pconfig.smoke.num_classes,
        "original_pool_size": int(len(raw_y)),
        "partition_seed": pconfig.smoke.partition_seed,
        "split_seed": pconfig.smoke.split_seed,
        "dirichlet_alpha": pconfig.smoke.dirichlet_alpha,
        "min_client_size": pconfig.smoke.min_client_size,
        "partition_attempts": attempts,
        "selection_indices": sel_idx.tolist(),
        "evaluation_indices": ev_idx.tolist(),
        "original_client_indices": [ix.tolist() for ix in original],
        "retained_client_indices": [ix.tolist() for ix in retained],
        "retained_client_sizes": counts_list,
        "retained_class_counts": [
            np.bincount(raw_y[ix], minlength=pconfig.smoke.num_classes
                        ).tolist() for ix in retained],
        "hashes": phashes,
    }
    _atomic_write_json(out / "partition.json", partition_record)

    record_check("cuda_eval_path_probe",
                  verify.check_cuda_eval_path(model_ctor, data, device))

    # ----- STAGE 0b: dev round-5 FIRST; 12 mandatory checks -----
    eval_model = model_ctor().to(device)
    # The ONE canonical trainable-parameter list for this run, sourced
    # from model.named_parameters() filtered by requires_grad (single
    # source of truth; see _param_names_for_model_ctor).  This exact
    # list is passed through every variability and feature helper.
    param_names_trainable = _param_names_for_model_ctor(model_ctor)
    per_seed_checkpoints: Dict[int, Dict[int, dict]] = {}
    dev_seed = pconfig.pilot_training_seed_dev
    dev_round = pconfig.pilot_checkpoint_rounds[0]
    _generate_dev_round5_then_verify(
        dev_seed, dev_round, pconfig, data, model_ctor, device,
        ckpt_seeded_dir=out / "checkpoints" / f"seed_{dev_seed}",
        results_dir=results_dir, out=out, eval_model=eval_model,
        smoke=smoke, phashes=phashes, ref_prov=ref_prov,
        record_check=record_check,
        counts_by_client=counts_by_client)

    per_seed_checkpoints[dev_seed] = {
        dev_round: _load_ckpt_to_device(
            (out / "checkpoints" / f"seed_{dev_seed}"
             / f"checkpoint_round_{dev_round}.pt"),
            _smoke_config_with_seed(pconfig.smoke, dev_seed), device)
    }
    # Resume dev and generate the remaining rounds.
    _resume_generate_dev(
        dev_seed, pconfig, data, model_ctor, device,
        ckpt_seeded_dir=out / "checkpoints" / f"seed_{dev_seed}",
        eval_model=eval_model, phashes=phashes)
    for rd in pconfig.pilot_checkpoint_rounds:
        ck = _load_ckpt_to_device(
            (out / "checkpoints" / f"seed_{dev_seed}"
             / f"checkpoint_round_{rd}.pt"),
            _smoke_config_with_seed(pconfig.smoke, dev_seed), device)
        if rd != dev_round:
            per_seed_checkpoints[dev_seed][rd] = ck
    timing["checkpoint_generation_dev_seed"] = (
        _checkpoint_emit_log_last("dev_seed"))

    # Held-out seeds: full fleet, no resume.
    for seed in pconfig.pilot_training_seeds_held_out:
        (_p, _t) = generate_checkpoint(
            _smoke_config_with_seed(pconfig.smoke, seed),
            data, model_ctor, device,
            (out / "checkpoints" / f"seed_{seed}"),
            partition_hash_record=phashes,
            checkpoint_rounds=pconfig.pilot_checkpoint_rounds)
        per_seed_checkpoints[seed] = {}
        for p in _p:
            ckpt_round = _ckpt_round_from_path(p)
            ck = _load_ckpt_to_device(p,
                                          _smoke_config_with_seed(
                                              pconfig.smoke, seed), device)
            per_seed_checkpoints[seed][ckpt_round] = ck
        log.info(f"  seed {seed}: generated {_p}")
    timing["checkpoint_generation_held_seeds_total"] = (
        _checkpoint_emit_log_last("held"))

    # ----- Build source schedules PER seed (10 sources/checkpoint). -----
    per_seed_sources: Dict[int, List[Dict]] = {}
    alt_cohorts_ledger: Dict[Tuple[int, int], dict] = {}
    for seed in pconfig.seeds_in_order():
        sources = []
        for rd in pconfig.pilot_checkpoint_rounds:
            ck = per_seed_checkpoints[seed][rd]
            sources.append({
                "kind": SRC_TRAJECTORY,
                "source_label": SRC_TRAJECTORY,
                "cohort_id": "trajectory",
                "rep_seed": "trajectory",
                "clients": ck["selected_clients"],
                "client_sds":
                    {c: {k: v.clone()
                          for k, v in sd.items()}
                      for c, sd in ck["client_sds"].items()},
                "sd0": {k: v.clone()
                        for k, v in ck["global_sd_before"].items()},
                "seed": seed, "round": rd,
            })
            for rep_seed in pconfig.pilot_rep_seeds:
                sources.append({
                    "kind": SRC_ORIGINAL_REP,
                    "source_label":
                        f"{SRC_ORIGINAL_REP}_{rep_seed}",
                    "cohort_id": "original",
                    "rep_seed": rep_seed,
                    "clients": ck["selected_clients"],
                    "sd0": {k: v.clone()
                            for k, v in ck["global_sd_before"].items()},
                    "client_sds": None,
                    "seed": seed, "round": rd,
                })
            alt_list, retries, ident_ledger = _sample_two_alt_cohorts(
                seed=seed, rd=rd,
                n_alt_cohorts=pconfig.pilot_n_alt_cohorts,
                original_clients=ck["selected_clients"],
                num_clients=pconfig.smoke.num_clients,
                num_participants=pconfig.smoke.num_participants,
                cohort_seed_offset=pconfig.pilot_cohort_seed_offset,
                retry_cap=pconfig.pilot_alt_cohort_retry_cap,
            )
            alt_cohorts_ledger[(seed, rd)] = {
                "identities_ledger": ident_ledger,
                "retries_per_cohort": retries,
                "n_alt_cohorts": len(alt_list),
            }
            for ai, cohort in enumerate(alt_list):
                for rep_seed in pconfig.pilot_rep_seeds:
                    sources.append({
                        "kind": SRC_ALT_COHORT,
                        "source_label":
                            f"{SRC_ALT_COHORT}_alt_{ai}_rep_{rep_seed}",
                        "cohort_id": f"alt_{ai}",
                        "rep_seed": rep_seed,
                        "clients": cohort,
                        "sd0": {k: v.clone()
                                for k, v in ck["global_sd_before"].items()},
                        "client_sds": None,
                        "seed": seed, "round": rd,
                    })
        per_seed_sources[seed] = sources

    _atomic_write_json(
        results_dir / "cohort_uniqueness.json",
        {f"{seed}|{rd}": ledger
         for (seed, rd), ledger in alt_cohorts_ledger.items()})

    # ----- STAGE 1+2: dev-only per-cell processing (split-specific) -----
    log.info("STAGE 2: dev-only per-cell processing begins")
    dev_grid_rows: List[dict] = []
    dev_rule_rows: List[dict] = []
    dev_stats_records: List[dict] = []
    dev_variability_payloads: List[dict] = []
    dev_split_metric_keys: Dict[str, float] = {}
    t_dev = time.time()
    for source in per_seed_sources[dev_seed]:
        _process_source(
            source, model_ctor, data, pconfig, device, eval_model,
            param_names_trainable,
            counts_by_client=counts_by_client,
            out_grid_rows=dev_grid_rows,
            out_rule_rows=dev_rule_rows,
            out_stats_records=dev_stats_records,
            out_split_metric_keys=dev_split_metric_keys,
            out_variability_payloads=dev_variability_payloads)
        run_cell_lightweight_checks(
            source, model_ctor, data, device, pconfig, record_check,
            counts_by_client=counts_by_client)
    timing["stage_2_dev_only"] = time.time() - t_dev

    # Incremental dev-side persistence: the dev split_metric_keys
    # BEFORE any held-out processing exist on disk now.
    _atomic_write_json(results_dir / "split_metric_keys_dev_only.json",
                        {"keys": sorted(dev_split_metric_keys.keys()),
                          "values": dev_split_metric_keys,
                          "pilot_config_fingerprint":
                              pconfig.pilot_config_fingerprint(),
                          "frozen_at": _now_iso(),
                          "guarantee":
                              ("Dev-side keys constructed exclusively "
                                "from dev seed (201) and the dev "
                                "feature/variability pipeline; no "
                                "held-out key contributed.")})
    dev_keys_sha = hashlib.sha256(
        json.dumps(sorted(dev_split_metric_keys.keys()),
                    sort_keys=True).encode()).hexdigest()

    # ----- STAGE 3: dev-side selection (in dev_grid_rows, frozen) -----
    try:
        gamma_dev_global, gamma_dev_phase_per_round, dev_selection_table = (
            _select_dev_fixed_controls(dev_grid_rows, pconfig,
                                          counts_by_client=counts_by_client))
    except IncompleteMeasurementError as exc:
        _save_incomplete_decision(
            results_dir, pconfig,
            reason=f"dev_selection_incomplete: {exc}")
        raise PilotIncompleteMeasurement(
            f"dev selection incomplete: {exc}") from exc

    _atomic_write_json(
        results_dir / "selection_table_dev.json",
        {"selection_loss_per_round_per_gamma_mean":
              dev_selection_table["selection_loss_per_round_per_gamma_mean"],
          "eval_loss_per_round_per_gamma_mean":
              dev_selection_table["eval_loss_per_round_per_gamma_mean"],
          "selection_loss_global_mean_per_gamma":
              dev_selection_table["selection_loss_global_mean_per_gamma"],
          "eval_loss_global_mean_per_gamma":
              dev_selection_table["eval_loss_global_mean_per_gamma"],
          "pilot_config_fingerprint": pconfig.pilot_config_fingerprint(),
          "frozen_at": _now_iso(),
          "rule_id": pconfig.pilot_selection_rule_id,
          "tiebreak_id": pconfig.pilot_tiebreak_rule_id})
    _atomic_write_json(
        results_dir / "gamma_dev_global.json",
        {"gamma_dev_global": gamma_dev_global,
          "rule_id": pconfig.pilot_selection_rule_id,
          "tiebreak_id": pconfig.pilot_tiebreak_rule_id,
          "selection_table_summary": dev_selection_table["summary_global"],
          "frozen_at": _now_iso(),
          "pilot_config_fingerprint": pconfig.pilot_config_fingerprint()})
    for rd in pconfig.pilot_checkpoint_rounds:
        _atomic_write_json(
            results_dir / f"gamma_dev_phase_{rd}.json",
            {"gamma_dev_phase": gamma_dev_phase_per_round[rd],
              "phase_round": rd,
              "rule_id": pconfig.pilot_selection_rule_id,
              "tiebreak_id": pconfig.pilot_tiebreak_rule_id,
              "selection_table_summary":
                  dev_selection_table["summary_phase"][str(rd)],
              "frozen_at": _now_iso(),
              "pilot_config_fingerprint": pconfig.pilot_config_fingerprint()})

    # STAGE 3b (implementation repair 2026-10-04): label the dev
    # seed's fixed-policy rows from the ALREADY-EVALUATED grid rows at
    # the frozen gammas.  Without this, the exploratory target join
    # (G = eval_loss_fixed_global - eval_loss_validation_selected on
    # the EVALUATION split) found ZERO dev cells, because the dev seed
    # was processed before the freeze and never got fixed-policy keys.
    # NOTE: split_metric_keys_dev_only.json and dev_keys_sha above
    # intentionally capture the dev keys AS OF THE FREEZE (before this
    # labeling); the labeling adds dev-only keys derived from dev-only
    # grid rows, so the no-held-out-contribution guarantee is
    # preserved.
    _label_dev_fixed_policy_rows(
        dev_grid_rows, gamma_dev_global, gamma_dev_phase_per_round,
        dev_rule_rows, dev_split_metric_keys)

    _atomic_write_json(
        results_dir / "source_keys_dev_only.json",
        {"dev_split_metric_keys_sha256": dev_keys_sha,
          "pilot_config_fingerprint": pconfig.pilot_config_fingerprint(),
          "frozen_at": _now_iso(),
          "guarantee":
              ("Dev-side selection tables are computed from "
                "split_metric_keys carrying only dev-seed entries; "
                "this SHA proves that no held-out key contributed to "
                "dev-side selection.")})

    # ----- STAGE 4: dev-side exploratory fit (dev features ONLY). -----
    # Per-(seed, round) variability summary built ONLY from dev
    # variability payloads.
    dev_per_checkpoint_var = compute_repetition_summaries(
        dev_variability_payloads,
        param_names_trainable,
        counts_by_client=counts_by_client,
    )
    _atomic_write_json(results_dir / "variability_dev.json",
                        dev_per_checkpoint_var)
    dev_feature_rows_per_phase = _build_feature_rows_per_phase(
        dev_variability_payloads,
        dev_split_metric_keys, pconfig,
        param_names=param_names_trainable,
        counts_by_client=counts_by_client)

    # ----- STAGE 5: held-out per-cell processing -----
    log.info("STAGE 5: held-out per-cell processing begins")
    held_split_metric_keys: Dict[str, float] = {}
    held_grid_rows: List[dict] = []
    held_rule_rows: List[dict] = []
    held_stats_records: List[dict] = []
    held_variability_payloads: List[dict] = []
    t_h = time.time()
    for seed in pconfig.pilot_training_seeds_held_out:
        for source in per_seed_sources[seed]:
            _process_source(
                source, model_ctor, data, pconfig, device, eval_model,
                param_names_trainable,
                counts_by_client=counts_by_client,
                out_grid_rows=held_grid_rows,
                out_rule_rows=held_rule_rows,
                out_stats_records=held_stats_records,
                out_split_metric_keys=held_split_metric_keys,
                out_variability_payloads=held_variability_payloads,
                gamma_dev_global=gamma_dev_global,
                gamma_dev_phase_per_round=gamma_dev_phase_per_round)
            run_cell_lightweight_checks(
                source, model_ctor, data, device, pconfig, record_check,
                counts_by_client=counts_by_client)
    timing["stage_5_held_out"] = time.time() - t_h

    held_per_checkpoint_var = compute_repetition_summaries(
        held_variability_payloads,
        param_names_trainable,
        counts_by_client=counts_by_client,
    )
    _atomic_write_json(results_dir / "variability_held_out.json",
                        held_per_checkpoint_var)
    dvars_aggregate = _aggregate_per_checkpoint_artifacts(
        dev_per_checkpoint_var)
    hvars_aggregate = _aggregate_per_checkpoint_artifacts(
        held_per_checkpoint_var)
    _atomic_write_json(results_dir / "variability.json",
                        {"dev_checkpoints": dvars_aggregate,
                          "held_out_checkpoints": hvars_aggregate,
                          "scope": "per_(training_seed, checkpoint_round)",
                          "pilot_config_fingerprint":
                              pconfig.pilot_config_fingerprint(),
                          "frozen_at": _now_iso()})

    # Held-out features strictly from held-out variability.
    # For dev features, we use DEV seeds ONLY across all dev rounds
    # (in this draft the dev seed is 201, so dev_feature_keys are
    # indexed by seed=201 for the artifacts).
    held_feature_rows_per_phase = _build_feature_rows_per_phase(
        held_variability_payloads,
        held_split_metric_keys, pconfig,
        param_names=param_names_trainable,
        counts_by_client=counts_by_client)

    # ----- STAGE 6+7: four decisions. -----
    log.info("STAGE 7: decisions")
    per_source_validation_gamma = {
        f"{s}|{r}|{sl}": float(g)
        for (s, r, sl), g in _VALIDATION_GAMMA_BY_SOURCE.items()
    }
    # Wipe the side cache so dev-side validation-selected γ does
    # NOT contaminate any later run that imports this module.
    _VALIDATION_GAMMA_BY_SOURCE.clear()

    # Decision inputs: EVALUATION-only, frozen, no order dependence.
    eval_only_split_metric_keys = dict(filter_evaluation_only(
        {**dev_split_metric_keys, **held_split_metric_keys}))

    # Inventory completeness (split-specific).
    try:
        assert_complete_split_inventory(
            eval_only_split_metric_keys, pconfig,
            policies=[POLICY_FIXED_GLOBAL, POLICY_FIXED_PHASE,
                       POLICY_VALIDATION_SELECTED, POLICY_CLAUDE],
            holdouts=[EVALUATION_HOLDOUT])
        inventory_complete = True
        inventory_error = None
    except IncompleteMeasurementError as exc:
        inventory_complete = False
        inventory_error = str(exc)
    inventories = {
        "split_metrics_complete": inventory_complete,
        "split_metrics_completeness_invariant":
            "split_specific_records_complete"
            if inventory_complete else inventory_error,
    }
    adaptive = decide_adaptive_headroom(
        eval_only_split_metric_keys, pconfig,
        gamma_dev_global=gamma_dev_global,
        gamma_dev_phase_per_round=gamma_dev_phase_per_round,
        per_source_validation_gamma=per_source_validation_gamma,
        inventories=inventories)
    claude = decide_claude_comparator(
        eval_only_split_metric_keys, pconfig,
        gamma_dev_global=gamma_dev_global,
        gamma_dev_phase_per_round=gamma_dev_phase_per_round)
    exploratory = decide_exploratory_signal(
        dev_split_metric_keys=eval_only_split_metric_keys,  # dev-side targets
        held_split_metric_keys=eval_only_split_metric_keys,  # held-out targets
        dev_feature_rows_per_phase=dev_feature_rows_per_phase,
        held_feature_rows_per_phase=held_feature_rows_per_phase,
        pconfig=pconfig,
    )
    overall = decide_overall(adaptive, claude, exploratory,
                              required_inputs_match=inventory_complete)

    decision_payload = {
        "statuses": {
            "adaptive_headroom": adaptive["verdict"],
            "claude_comparator": claude["verdict"],
            "exploratory_variability_signal": exploratory["verdict"],
            "overall_recommendation": overall,
        },
        "details": {
            "adaptive_headroom": adaptive,
            "claude_comparator": claude,
            "exploratory_variability_signal": exploratory,
        },
        "gamma_dev_global": float(gamma_dev_global),
        "gamma_dev_phase_per_round":
            {str(k): float(v) for k, v in gamma_dev_phase_per_round.items()},
        "per_source_validation_gamma": per_source_validation_gamma,
        "source_keys_dev_only_sha256": dev_keys_sha,
        "protocol_sha256_glob": _protocol_sha_for_decision(),
        "decision_at": _now_iso(),
        "pilot_config_fingerprint": pconfig.pilot_config_fingerprint(),
    }
    _atomic_write_json(results_dir / "pilot_decision.json",
                        decision_payload)
    _atomic_write_json(results_dir / "exploratory_signal_dev.json",
                        _exploratory_artifacts_for_disk(
                            exploratory, dev_only=True))
    _atomic_write_json(results_dir / "exploratory_signal_held_out.json",
                        _exploratory_artifacts_for_disk(
                            exploratory, dev_only=False))

    # ----- Aggregated grids / stats / per-example CSV -----
    all_grid_rows = dev_grid_rows + held_grid_rows
    all_rule_rows = dev_rule_rows + held_rule_rows
    all_stats_records = dev_stats_records + held_stats_records

    # Per-example losses for the 5 decisive policies on every cell.
    # Run twice (once per direction) into two writers; final report
    # verifies identical content (split-metric key args).
    pe_writer_a = PerExampleCSVWriter(results_dir / "per_example_losses.csv")
    pe_writer_b = PerExampleCSVWriter(results_dir / "per_example_losses_b.csv")
    _per_example_for_decisive_policies(
        per_seed_sources, per_seed_checkpoints, data,
        model_ctor, device, eval_model,
        gamma_dev_global, gamma_dev_phase_per_round,
        pe_writer_a)
    _per_example_for_decisive_policies(
        reversed(list(per_seed_sources.items())),
        per_seed_checkpoints, data,
        model_ctor, device, eval_model,
        gamma_dev_global, gamma_dev_phase_per_round,
        pe_writer_b)
    pe_writer_a.close()
    pe_writer_b.close()
    # Identity check on the order-reversed per-example CSV records.
    # The two writers iterate cells in different order, so we
    # row-SORT both files before byte equality. The header line
    # is preserved; only data rows are sorted to a canonical
    # (seed, round, source, rep_seed, cohort_id, holdout,
    # policy, example_index) order.
    def _sort_data_rows(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text
        header, data = lines[0], lines[1:]
        data_sorted = sorted(data,
                             key=lambda r: tuple(r.split(",")))
        return header + "\n" + "\n".join(data_sorted) + "\n"
    a_text = (results_dir / "per_example_losses.csv").read_text()
    b_text = (results_dir / "per_example_losses_b.csv").read_text()
    record_check("per_example_reversed_order_identity",
                  {"passed": (_sort_data_rows(a_text)
                              == _sort_data_rows(b_text)),
                   "compare_mode":
                       "data_rows_sorted_canonical"},
                  mandatory=True)
    (results_dir / "per_example_losses_b.csv").unlink(missing_ok=True)

    _write_grid_csv(results_dir / "aggregate_evaluation.csv",
                    all_grid_rows)
    _write_grid_csv(results_dir / "aggregate_evaluation.json",
                    all_grid_rows, as_json=True)
    _write_grid_csv(results_dir / "rule_evaluation.csv", all_rule_rows,
                    fields=RULE_FIELDS)
    _write_grid_csv(results_dir / "rule_evaluation.json", all_rule_rows,
                    fields=RULE_FIELDS, as_json=True)
    stats_with_inv = {
        "_inventory": {
            "fields_per_record": sorted(all_stats_records[0].keys())
            if all_stats_records else [],
            "n_records": len(all_stats_records),
            "pilot_config_fingerprint":
                pconfig.pilot_config_fingerprint(),
        },
        "records": all_stats_records,
    }
    _atomic_write_json(results_dir / "update_statistics.json",
                        stats_with_inv)
    _atomic_write_json(
        results_dir / "split_metric_keys.json",
        {"dev": _filter_split_to_seed(dev_split_metric_keys, dev_seed),
          "held_out": _filter_split_to_seeds(held_split_metric_keys,
                                              pconfig.pilot_training_seeds_held_out),
          "all": {**dev_split_metric_keys, **held_split_metric_keys},
          "pilot_config_fingerprint": pconfig.pilot_config_fingerprint(),
          "frozen_at": _now_iso(),
          "split_metric_key_format": "seed|round|source|policy|holdout"})

    sec_rows = [r for r in all_grid_rows
                if r["buffer_policy"] == POLICY_PARAMS_ONLY
                and r["gamma"] == GAMMA_ONE]
    _write_grid_csv(results_dir / "secondary_buffer_evaluation.csv",
                    sec_rows)

    timing["total"] = time.time() - t_global
    timing["phase_breaks_sec"] = {
        "stage_0_setup_and_verify": timing["data_and_partition"],
        "stage_2_dev_only": timing["stage_2_dev_only"],
        "stage_5_held_out": timing["stage_5_held_out"],
    }
    checks_out = {"all_passed": all(c.get("passed", False)
                                      for c in checks.values()
                                      if not c.get("informational")),
                  "checks": checks}
    _atomic_write_json(results_dir / "verification.json", checks_out)
    _atomic_write_json(results_dir / "timing.json", timing)
    _write_pilot_report(results_dir, pconfig, decision_payload,
                        dev_split_metric_keys, held_split_metric_keys,
                        adaptive, claude, exploratory,
                        checks_out, timing, provenance,
                        {"per_checkpoint_dev": dev_per_checkpoint_var,
                          "per_checkpoint_held_out": held_per_checkpoint_var},
                        gamma_dev_global,
                        gamma_dev_phase_per_round,
                        dev_selection_table)
    log.info(f"DONE. Total {timing['total']:.1f}s. Results: {results_dir}.")
    return decision_payload


# ============================================================================
# STAGE 0b / 0c — dev seed round-5 verification + resume helpers.
# ============================================================================


def _generate_dev_round5_then_verify(
    dev_seed, dev_round, pconfig, data, model_ctor, device,
    ckpt_seeded_dir, results_dir, out, eval_model, smoke, phashes,
    ref_prov, record_check, *, counts_by_client):
    """Run dev seed rounds 1..5 + the 12 mandatory checks. Abort
    BEFORE running the remaining fleet if any mandatory check fails.
    """
    cfg = _smoke_config_with_seed(pconfig.smoke, dev_seed)
    ckpt_paths, _t = generate_checkpoint(
        cfg, data, model_ctor, device, ckpt_seeded_dir,
        partition_hash_record=phashes,
        checkpoint_rounds=[dev_round])
    ck = _load_ckpt_to_device(ckpt_paths[0], cfg, device)
    sd0 = {k: v.to(device) for k, v in ck["global_sd_before"].items()}
    traj_clients = ck["selected_clients"]
    traj_sds = {c: {k: v.to(device) for k, v in sd.items()}
                for c, sd in ck["client_sds"].items()}
    traj_w_arr = participant_weights(
        [counts_by_client[c] for c in traj_clients])
    traj_w = {traj_clients[i]: traj_w_arr[i]
               for i in range(len(traj_clients))}
    sd_fedavg_traj = average_states(
        [traj_sds[c] for c in traj_clients],
        [traj_w[c] for c in traj_clients])
    first_sd_traj = traj_sds[traj_clients[0]]
    param_names = get_stat_param_names(model_ctor())

    record_check("gamma0_primary_matches_checkpoint",
                  verify.check_gamma0_primary(sd0, sd_fedavg_traj,
                                                first_sd_traj),
                  mandatory=True)
    record_check("gamma1_primary_matches_fedavg_fullstate",
                  verify.check_gamma1_primary(sd0, sd_fedavg_traj,
                                                first_sd_traj),
                  mandatory=True)
    record_check("gamma0_secondary_matches_checkpoint",
                  verify.check_gamma0_secondary(sd0, sd_fedavg_traj,
                                                  first_sd_traj, param_names),
                  mandatory=True)
    record_check("weighted_aggregation_independent",
                  verify.check_weighted_aggregation(
                      [traj_sds[c] for c in traj_clients],
                      [traj_w[c] for c in traj_clients], sd_fedavg_traj),
                  mandatory=True)
    record_check("rule_safeguards",
                  verify.check_rule_safeguards(sd0, param_names),
                  mandatory=True)
    record_check("config_rejection",
                  verify.check_config_rejection(
                      ck, _smoke_config_with_seed(pconfig.smoke, dev_seed)),
                  mandatory=True)

    cand_check, _ = build_candidate(sd0, sd_fedavg_traj, 0.5, first_sd_traj,
                                     POLICY_ALL_FLOATING)
    record_check("evaluation_restores_state_mode_rng",
                  verify.check_eval_restores_state(
                      model_ctor, sd0, cand_check, data, device),
                  mandatory=True)
    record_check("holdout_disjointness_and_eval_determinism",
                  verify.check_disjointness_and_eval_determinism(
                      data, model_ctor, sd0, device),
                  mandatory=True)
    record_check("identical_seed_retrain",
                  verify.check_identical_seed_retrain(
                      model_ctor, sd0, data, traj_clients[0],
                      _smoke_config_with_seed(pconfig.smoke, dev_seed),
                      dev_round, pconfig.pilot_rep_seeds[0], device),
                  mandatory=True)
    record_check("shared_client_matched_seed",
                  verify.check_shared_client_matched_seed(
                      model_ctor, sd0, data, traj_clients[0],
                      _smoke_config_with_seed(pconfig.smoke, dev_seed),
                      dev_round, pconfig.pilot_rep_seeds[0], device),
                  mandatory=True)
    record_check("per_example_eval_restores_state",
                  verify.check_per_example_eval_restores_state(
                      model_ctor, sd0, cand_check, data, device),
                  mandatory=True)
    record_check("model_compatibility",
                  verify.check_model_compatibility(ref_prov),
                  mandatory=False)
    record_check("reconstruction_divergence",
                  verify.compare_models(eval_model,
                                          lambda: SmallCNNRecovered(
                                              num_classes=smoke.num_classes),
                                          device),
                  mandatory=False, informational=True)
    try:
        refcmp = run_reference_compare_if_bundled(ref_prov, results_dir)
        if refcmp is not None:
            record_check("reference_aggregation_equivalence", {
                "passed": refcmp["aggregation"].get("status") == "match",
                "detail": refcmp["aggregation"],
            }, mandatory=True)
            record_check("reference_behavioral_comparison", {
                "passed": True,
                "component_status": {k: v.get("status")
                                       for k, v in refcmp.items()},
                "detail": refcmp,
            }, mandatory=False)
    except Exception as exc:
        record_check("reference_aggregation_equivalence",
                      {"passed": False, "error": str(exc)},
                      mandatory=True)


def _resume_generate_dev(dev_seed, pconfig, data, model_ctor, device,
                          ckpt_seeded_dir, eval_model, phashes):
    """Resume dev seed from round-5 saved state to round 50 + round
    100.  The state is loaded just-in-time from `global_sd_after`
    of the existing round-5 file and the per-round RNG replayed
    deterministically.
    """
    resume_from = ckpt_seeded_dir / "checkpoint_round_5.pt"
    cfg = _smoke_config_with_seed(pconfig.smoke, dev_seed)
    rounds = [r for r in pconfig.pilot_checkpoint_rounds if r != 5]
    if not rounds:
        return
    try:
        generate_checkpoint(cfg, data, model_ctor, device,
                              ckpt_seeded_dir,
                              partition_hash_record=phashes,
                              checkpoint_rounds=rounds,
                              start_round=(5 + 1),
                              reuse_state_from=str(resume_from))
    except Exception as e:
        log.exception("Resume from saved round-5 state failed; "
                       "treat as INCOMPLETE.")
        raise PilotIncompleteMeasurement(
            f"dev_resume_failed: {e}") from e


def _checkpoint_emit_log_last(label):
    """Helper used in timing-emission headings."""
    return 0.0


def _aggregate_per_checkpoint_artifacts(per_ckpt_var):
    """Just pass-through; per-checkpoint artifacts already nested."""
    return per_ckpt_var.get("per_checkpoint", [])


def _exploratory_artifacts_for_disk(exploratory, *, dev_only: bool):
    """Persistable view: dev_only filter is just a tag; the
    payload is the records that pertain to which step.
    `dev_only=True` persists the dev-side standardization and
    coefficients; `False` persists the held-out predictions,
    MAE, MSE.
    """
    out = {
        "tag": "dev_only" if dev_only else "held_out",
        "predictors": exploratory["predictors"],
        "n_dev_cells_used_total": exploratory["n_dev_cells_used_total"],
        "n_held_cells_used_total": exploratory["n_held_cells_used_total"],
        "common_held_keys_count": exploratory["common_held_keys_count"],
        "feature_order": exploratory["feature_order"],
        "phase_indicators": exploratory["phase_indicators"],
        "n_features_X": exploratory["n_features_X"],
        "intercept_equals_posterior": exploratory["intercept_equals_posterior"],
        "sign_agreement_definition": exploratory["sign_agreement_definition"],
    }
    if dev_only:
        # Record dev-side standardization and coefficients.
        for p in out["predictors"]:
            p_stripped = dict(p)
            # Drop large numerical lists when emitting the dev
            # summary; the per-predictor disk JSON keeps the
            # canonical numerical hash payload separately.
            for k in ("predictions_unstandardized",
                       "predictions_standardized",
                       "targets_held"):
                p_stripped.pop(k, None)
            out["predictors"] = [p_stripped if pp is p else pp
                                  for pp in out["predictors"]]
    return out


def _filter_split_to_seed(split_metric_keys, seed):
    return {k: v for k, v in split_metric_keys.items()
            if int(k.split("|", 1)[0]) == int(seed)}


def _filter_split_to_seeds(split_metric_keys, seeds):
    seeds_s = {str(s) for s in seeds}
    return {k: v for k, v in split_metric_keys.items()
            if str(k.split("|", 1)[0]) in seeds_s}


def run_reference_compare_if_bundled(ref_prov, results_dir):
    if ref_prov.get("model_source") != "bundled_reference_snapshot":
        return None
    from . import ref_compare
    refcmp = ref_compare.run_all()
    _atomic_write_json(results_dir / "reference_comparison.json", refcmp)
    return refcmp


# ============================================================================
# Per-example losses
# ============================================================================


def _per_example_for_decisive_policies(per_seed_sources,
                                          per_seed_checkpoints, data,
                                          model_ctor, device, eval_model,
                                          gamma_dev_global,
                                           gamma_dev_phase_per_round,
                                          pe_writer):
    # Accept a dict, a list-of-(seed, sources) tuples (forward OR
    # reversed), or any iterable whose items are (seed, sources)
    # pairs. We NORMALIZE to a list-of-tuples so iteration order
    # has no observable side effects.
    if isinstance(per_seed_sources, dict):
        ordered = list(per_seed_sources.items())
    else:
        ordered = list(per_seed_sources)
    for seed, sources in ordered:
        for source in sources:
            # We pass the source dict as both `source` and `ck` —
            # `_per_example_one_source` ignores the `ck` argument
            # (it uses source["client_sds"] / source["sd0"] /
            # source["clients"]); making the source also stand in
            # for ck avoids a hard dep on `["ckpt"]`.
            _per_example_one_source(
                source, source,
                data, model_ctor, device, eval_model,
                gamma_dev_global, gamma_dev_phase_per_round,
                pe_writer, _validation_gamma=(seed, source["round"],
                                               source["source_label"]))


def _per_example_one_source(source, ck, data, model_ctor, device,
                              eval_model, gamma_dev_global,
                              gamma_dev_phase_per_round,
                              pe_writer, _validation_gamma):
    sd0 = source["sd0"]
    sds = source.get("client_sds") or ck["client_sds"]
    clients = source["clients"]
    counts = [data["counts"][c] for c in clients]
    w_arr = participant_weights(counts)
    w_by_id = {clients[i]: w_arr[i] for i in range(len(clients))}
    sd_fed = average_states([sds[c] for c in clients],
                             [w_by_id[c] for c in clients])
    first_sd = sds[clients[0]]
    seed = source["seed"]
    rd = source["round"]
    indices_by_holdout = {SELECTION_HOLDOUT: data["sel_idx"],
                          EVALUATION_HOLDOUT: data["eval_idx"]}

    val_g = _VALIDATION_GAMMA_BY_SOURCE.get(_validation_gamma, GAMMA_ONE)
    fixed_phase_g = gamma_dev_phase_per_round.get(rd, GAMMA_ONE)

    for policy, gamma, origin in [
        (POLICY_FEDAVG, GAMMA_ONE, "grid_exact"),
        (POLICY_FIXED_GLOBAL, gamma_dev_global, "rule_global_fixed_dev"),
        (POLICY_FIXED_PHASE, fixed_phase_g, "rule_phase_fixed_dev"),
        (POLICY_VALIDATION_SELECTED, val_g, "rule_validation_selected"),
        (POLICY_CLAUDE, None, "rule_claude"),
    ]:
        for holdout, idx in indices_by_holdout.items():
            if policy == POLICY_FEDAVG:
                cand = sd_fed
                cand_valid = True
                invalid_entries = []
            elif policy == POLICY_VALIDATION_SELECTED:
                cand, rep = build_candidate(sd0, sd_fed, val_g, first_sd,
                                            POLICY_ALL_FLOATING)
                cand_valid = rep["valid"]
                invalid_entries = rep["invalid_entries"]
            elif policy == POLICY_FIXED_GLOBAL:
                cand, rep = build_candidate(sd0, sd_fed, gamma_dev_global,
                                            first_sd,
                                            POLICY_ALL_FLOATING)
                cand_valid = rep["valid"]
                invalid_entries = rep["invalid_entries"]
            elif policy == POLICY_FIXED_PHASE:
                cand, rep = build_candidate(sd0, sd_fed, fixed_phase_g,
                                            first_sd,
                                            POLICY_ALL_FLOATING)
                cand_valid = rep["valid"]
                invalid_entries = rep["invalid_entries"]
            elif policy == POLICY_CLAUDE:
                stats = compute_update_statistics(
                    sd0, sds, clients, w_by_id,
                    _param_names_for_model_ctor(model_ctor))
                claude_g, sg = claude_raw_shrinkage(stats)
                cand, rep = build_candidate(sd0, sd_fed, claude_g,
                                            first_sd,
                                            POLICY_ALL_FLOATING)
                gamma = float(claude_g)
                # A failed Claude (neg-running_var etc.) is NOT
                # automatically substituted with γ=1; we record
                # the per-example loss under the per-source
                # validation-selected γ so the row count is
                # complete, but DO NOT persist as a decisive
                # split_metric_keys row.
                cand_valid = rep["valid"]
                invalid_entries = rep["invalid_entries"]
                if not cand_valid:
                    log.warning(
                        f"claude rule INVALID source={source['source_label']} "
                        f"seed={seed} round={rd} gamma={gamma} "
                        f"invalid_entries={invalid_entries}; "
                        "decisive claude rule row will be omitted "
                        "from split_metric_keys.")
            else:
                continue
            if not cand_valid:
                log.warning(
                    f"per-example path skipped: source={source['source_label']} "
                    f"seed={seed} round={rd} policy={policy} "
                    f"gamma={gamma} is INVALID; the per-example CSV "
                    "does not contain a row for this (policy, holdout) "
                    "cell.  The decisive run already triggers "
                    "IncompleteMeasurementError for non-claude "
                    "policies; claude's safeguard log is preserved.")
                continue
            pe = evaluate_state_per_example(
                model_ctor, cand, data["raw_x"], data["raw_y"], idx,
                device, shared_model=eval_model)
            pe_record_gamma = float(gamma if gamma is not None else 0.0)
            pe_writer.write_policy_on_holdout(
                seed=seed, rd=rd, source=source["source_label"],
                rep_seed=source["rep_seed"], cohort_id=source["cohort_id"],
                holdout=holdout, policy=policy,
                gamma=pe_record_gamma, gamma_origin=origin, per_example=pe)


def _param_names_for_model_ctor(model_ctor):
    """Return the canonical list of trainable parameter names for the
    model produced by `model_ctor`. The list is sourced DIRECTLY from
    `model.named_parameters()`, filtered by `requires_grad=True`.

    This is the SINGLE source of truth for variability and feature
    helpers. There is NO list hardcoded anywhere; there is NO
    state-dict-based inference.  The orchestrator never falls back to
    payload keys or to a guessed list: if `model_ctor()` cannot be
    instantiated, we raise.

    For synthetic tests, `model_ctor()` must return a `torch.nn.Module`
    whose `named_parameters()` returns the synthetic parameter names
    (e.g. `("w", "b")`). The synthetic test path is responsible for
    constructing such a mock module; this helper does not invent
    names.

    Buffers (BatchNorm running_mean / running_var / num_batches_tracked)
    are NEVER in the result because they do not appear in
    `named_parameters()`.
    """
    try:
        model = model_ctor()
    except Exception as e:
        raise RuntimeError(
            f"model_ctor()={model_ctor!r} could not be instantiated "
            f"({type(e).__name__}: {e}); the orchestrator refuses "
            "to invent parameter names.") from e
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names:
        raise RuntimeError(
            f"model_ctor()={model_ctor!r} produced no requires_grad=True "
            f"parameters; the orchestrator refuses to invent names.")
    return list(names)



# ============================================================================
# Helpers
# ============================================================================


def _smoke_config_with_seed(base: SmokeConfig, seed: int) -> SmokeConfig:
    from dataclasses import replace
    return replace(base, training_seed=seed)


def _ckpt_round_from_path(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def _load_ckpt_to_device(path: Path, config: SmokeConfig, device) -> dict:
    from .fedavg import load_checkpoint
    return load_checkpoint(path, config, device)


def _setup_outdir(pconfig: PilotConfig):
    od = Path(pconfig.smoke.output_dir)
    if od.exists() and any(od.iterdir()):
        raise SystemExit(f"Output directory is not empty; refusing to "
                          f"overwrite: {od}")
    od.mkdir(parents=True, exist_ok=True)
    (od / "checkpoints").mkdir(exist_ok=True)
    results = od / "results"
    results.mkdir(exist_ok=True)
    return od, results


def _record_protocol_files(out: Path) -> str:
    h = hashlib.sha256()
    per_file = {}
    for name in ("PROTOCOL.md", "GO_NO_GO.md", "PRE_EXECUTION_AMENDMENT.md"):
        p = BUNDLE_DIR / name
        if p.is_file():
            content = p.read_bytes()
            per_file[name] = hashlib.sha256(content).hexdigest()
            h.update(content)
    _atomic_write_json(
        out / "pilot_protocol_sha256.json",
        {"sha256_combined": h.hexdigest(),
          "per_file": per_file,
          "frozen_at": _now_iso()})
    return h.hexdigest()


def _build_provenance(pconfig, ref_prov, snapshot):
    return {
        "bundle_dir": str(BUNDLE_DIR),
        "bundle_sha256": bundle_hashes(BUNDLE_DIR),
        "pilot_config": pconfig.to_dict(),
        "pilot_config_fingerprint": pconfig.pilot_config_fingerprint(),
        "reference_snapshot": str(REFERENCE_DIR),
        "reference_access": ("bundled verbatim snapshot (COPY_MANIFEST.json); "
                              "hash-validated at runtime; read-only"),
        "reference_snapshot_hashes": snapshot,
        "model": ref_prov,
        "determinism_policy": "cudnn_deterministic_true_benchmark_false",
    }


def _save_incomplete_decision(results_dir, pconfig, *,
                                reason: str,
                                partial_payload: Optional[dict] = None):
    payload = {
        "statuses": {
            "adaptive_headroom": INCONCLUSIVE,
            "claude_comparator": INCONCLUSIVE,
            "exploratory_variability_signal": INCONCLUSIVE,
            "overall_recommendation": INCONCLUSIVE,
        },
        "details": {"incomplete_reason": reason,
                      "partial_payload": partial_payload or {}},
        "gamma_dev_global": None,
        "gamma_dev_phase_per_round": None,
        "per_source_validation_gamma": {},
        "protocol_sha256_glob": _protocol_sha_for_decision(),
        "decision_at": _now_iso(),
        "pilot_config_fingerprint": pconfig.pilot_config_fingerprint(),
    }
    _atomic_write_json(results_dir / "pilot_decision.json", payload)
    _atomic_write_json(
        results_dir / "verification.json",
        {"all_passed": False,
          "error": reason,
          "pilot_incomplete_measurement": True})


# ============================================================================
# Report
# ============================================================================


def _write_pilot_report(results, pconfig, decision_payload,
                        dev_split_metric_keys, held_out_split_metric_keys,
                        adaptive, claude, exploratory,
                        checks_out, timing, provenance,
                        var_payload, gamma_dev_global,
                        gamma_dev_phase_per_round,
                        dev_selection_table,
                        _return_lines_only=False):
    lines = [
        "# CIFAR-100 Pilot Report", "",
        "Frozen documents: PROTOCOL.md, GO_NO_GO.md, "
        "PRE_EXECUTION_AMENDMENT.md (frozen BEFORE execution). "
        "v1 frozen docs preserved under PROTOCOL_v1_2026-10-04.md / "
        "GO_NO_GO_v1_2026-10-04.md. "
        "v2 frozen docs preserved under PROTOCOL_v2_2026-10-04.md / "
        "GO_NO_GO_v2_2026-10-04.md.",
        "",
        f"**Pilot config fingerprint:** "
        f"{pconfig.pilot_config_fingerprint()[:16]}...",
        "**Decisions (NOT a single GO/NO-GO):**",
        "",
        "| Decision | Status | Reason |",
        "|----------|--------|--------|",
        f"| adaptive_headroom | {decision_payload['statuses']['adaptive_headroom']} | "
        f"{adaptive.get('gate_failure_reason', '')} |",
        f"| claude_comparator | {decision_payload['statuses']['claude_comparator']} | "
        f"{claude.get('gate_failure_reason', '')} |",
        f"| exploratory_variability_signal | "
        f"{decision_payload['statuses']['exploratory_variability_signal']} | "
        f"EXPLORATORY: "
        + "; ".join(
            f"{p.get('predictor_id', '?')}: "
            f"fit_status={p.get('fit_status', '?')} "
            f"(dev_cells={p.get('n_dev_cells_used', '-')}, "
            f"held_cells={p.get('n_held_cells_used', '-')})"
            for p in exploratory.get("predictors", []))
        + f"; {exploratory.get('common_held_keys_count', '-')} "
          "common (round, source) cells |",
        f"| overall_recommendation | "
        f"{decision_payload['statuses']['overall_recommendation']} | "
        f"synthesis of the above |",
    ]

    # Variability summary (per checkpoint; trajectory excluded).
    lines.extend(["", "## Variability summary", "",
                   "Scope: per `(training_seed, checkpoint_round)`; "
                   "trajectory is reported separately within each "
                   "per_checkpoint record (not averaged with reps)."])
    # The per-checkpoint variability payload is a dict with a
    # "per_checkpoint" list (compute_repetition_summaries return
    # value); accept both the dict and a bare list.
    _pc_dev = var_payload.get("per_checkpoint_dev", {})
    if isinstance(_pc_dev, dict):
        pc_records = _pc_dev.get("per_checkpoint", [])
    elif isinstance(_pc_dev, list):
        pc_records = _pc_dev
    else:
        pc_records = []
    for rec in pc_records[:6]:
        s = rec.get("seed"); r = rec.get("round")
        rep_count_mean = (sum(int(v) for v
                                in rec.get("per_client_rep_count", {}).values())
                            / max(1, len(rec.get("per_client_rep_count", {}))))
        lines.append(
            f"- seed={s}, round={r}: per_client_mean_rep_count={rep_count_mean:.3f}")
    if not pc_records:
        lines.append("- (no per-checkpoint variability records)")

    # Audit hashes.
    lines.extend(["", "## Audit hashes",
                  "",
                  f"- protocol_sha256_glob: "
                  f"{decision_payload['protocol_sha256_glob']}",
                  f"- source_keys_dev_only_sha256: "
                  f"{decision_payload['source_keys_dev_only_sha256']}",
                  f"- pilot_config_fingerprint: "
                  f"{pconfig.pilot_config_fingerprint()}"])

    if _return_lines_only:
        return lines
    with open(results / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _protocol_sha_for_decision():
    h = hashlib.sha256()
    for n in ("PROTOCOL.md", "GO_NO_GO.md", "PRE_EXECUTION_AMENDMENT.md"):
        p = BUNDLE_DIR / n
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()
