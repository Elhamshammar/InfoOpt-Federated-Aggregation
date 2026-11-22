#!/usr/bin/env python
"""Saved-artifact analysis: buffer policy and selected-vs-fixed scaling.

REVISED VERSION (2026-10-04, pilot closure).  Changes from
analysis_buffer_policy/ (which is preserved unchanged):
  - development seed 201 and held-out seeds 202/203 are reported
    SEPARATELY in every summary (never pooled);
  - sign counts are reported separately for the 27 all-seed cohort
    cells and the 18 held-out cohort cells;
  - the interaction claim is removed: the two gamma=1-reference
    comparisons share the same validation-selected candidate, so their
    difference reflects the different reference losses;
  - the conclusion is limited to freezing checkpoint buffers at
    gamma=1 in this setting (no generalization to other normalization
    methods);
  - input artifact hashes and the exact command are recorded in the
    report.

Analyzes the COMPLETED pilot run
results/stepsize_diag_pilot_20261004_175620 from its saved CSV/JSON
artifacts ONLY.  No training, no candidate evaluation, no modification
of existing results.

Comparisons (all on the independent EVALUATION split; positive loss
improvement = loss_reference - loss_candidate):

  A. Frozen buffers vs full state at gamma=1:
       reference  = all_floating_state FedAvg gamma=1 (grid rows)
       candidate  = parameters_only_frozen_buffers gamma=1
                    (secondary_buffer_evaluation)
  B. Validation-selected scaling vs the two gamma=1 references.
  C. Validation-selected scaling vs the development-selected fixed
     scales (global and phase).

Aggregation: repetitions are averaged WITHIN each cohort first
(cohort_id in {original, alt_0, alt_1}); trajectory sources are
reported separately and never pooled with repetition sources.
Checks: candidate validity, missing pairs, matching sample counts
(grid/secondary rows persist n_eval; rule rows do not persist n_eval —
the decisive policies' per-example row counts are verified against the
evaluation split size from per_example_losses.csv instead).

Note on the dev seed (201): the run predates the SECTION VII repair, so
the dev seed has no fixed-policy rule rows.  The frozen gammas are grid
points, so the dev fixed-scale measurements are recovered from the
EXISTING grid rows (labeled recovered_from_grid_row=True); no new
evaluation is performed.

Outputs (into --out-dir, a NEW folder):
  paired_by_source.csv    per-(seed, round, source, comparison) pairs
  cohort_summary.csv      per-(seed, round, cohort, comparison) with
                          repetitions pre-averaged within cohort
  report.md               measured findings vs interpretations
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

POLICY_FULL = "all_floating_state"
POLICY_PARAMS_ONLY = "parameters_only_frozen_buffers"

REF_FULL_G1 = "full_state_fedavg_gamma1"
REF_PARAMS_ONLY_G1 = "params_only_frozen_buffers_gamma1"
CAND_VALSEL = "validation_selected_fullstate"
REF_FIXED_GLOBAL = "fixed_global_gamma_dev"
REF_FIXED_PHASE = "fixed_phase_gamma_dev"

COMPARISONS = [
    ("A_freeze_buffers_at_gamma1", REF_FULL_G1, REF_PARAMS_ONLY_G1),
    ("B_valsel_vs_full_gamma1", REF_FULL_G1, CAND_VALSEL),
    ("B_valsel_vs_params_only_gamma1", REF_PARAMS_ONLY_G1, CAND_VALSEL),
    ("C_valsel_vs_fixed_global", REF_FIXED_GLOBAL, CAND_VALSEL),
    ("C_valsel_vs_fixed_phase", REF_FIXED_PHASE, CAND_VALSEL),
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/stepsize_diag_pilot_20261004_175620")
    ap.add_argument("--out-dir", default="analysis_buffer_policy_v2")
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = run / "results"

    for rel in ("aggregate_evaluation.csv", "secondary_buffer_evaluation.csv",
                "rule_evaluation.csv", "gamma_dev_global.json",
                "per_example_losses.csv"):
        if not (res / rel).is_file():
            print(f"MISSING required artifact: {res / rel}")
            return 2

    grid = list(csv.DictReader(open(res / "aggregate_evaluation.csv")))
    secondary = list(csv.DictReader(
        open(res / "secondary_buffer_evaluation.csv")))
    rules = list(csv.DictReader(open(res / "rule_evaluation.csv")))
    gamma_dev_global = _f(json.loads(
        (res / "gamma_dev_global.json").read_text())["gamma_dev_global"])
    gamma_dev_phase = {}
    for p in res.glob("gamma_dev_phase_*.json"):
        d = json.loads(p.read_text())
        gamma_dev_phase[int(d["phase_round"])] = _f(d["gamma_dev_phase"])

    # ---- provenance: hash every input artifact ----------------------
    input_hashes = {}
    for rel in ("aggregate_evaluation.csv",
                "secondary_buffer_evaluation.csv",
                "rule_evaluation.csv", "per_example_losses.csv",
                "gamma_dev_global.json"):
        input_hashes[f"results/{rel}"] = hashlib.sha256(
            (res / rel).read_bytes()).hexdigest()
    for p in sorted(res.glob("gamma_dev_phase_*.json")):
        input_hashes[f"results/{p.name}"] = hashlib.sha256(
            p.read_bytes()).hexdigest()

    # ---- schema verification ---------------------------------------
    grid_fields = set(grid[0].keys())
    sec_fields = set(secondary[0].keys())
    rule_fields = set(rules[0].keys())
    need_grid = {"seed", "round", "source", "cohort_id", "rep_seed",
                 "buffer_policy", "gamma", "candidate_valid",
                 "eval_loss", "eval_acc", "n_eval"}
    need_rule = {"seed", "round", "source", "cohort_id", "rep_seed",
                 "rule", "gamma", "candidate_valid",
                 "eval_loss", "eval_acc"}
    schema_ok = (need_grid <= grid_fields and need_grid <= sec_fields
                 and need_rule <= rule_fields)

    # ---- per-(seed, round, source) observations --------------------
    obs = {}  # (seed, round, source) -> name -> record
    invalid_grid_rows = []

    def _cell(r):
        return (int(r["seed"]), int(r["round"]), r["source"])

    def _put(cell, name, r, recovered=False):
        obs.setdefault(cell, {})[name] = {
            "eval_loss": _f(r["eval_loss"]),
            "eval_acc": _f(r["eval_acc"]),
            "n_eval": (int(r["n_eval"]) if r.get("n_eval") not in
                       (None, "") else None),
            "candidate_valid": (str(r["candidate_valid"]) == "True"),
            "gamma": _f(r["gamma"]),
            "recovered_from_grid_row": recovered,
        }

    for r in grid:
        if str(r["candidate_valid"]) != "True":
            invalid_grid_rows.append((r["seed"], r["round"], r["source"],
                                      r["buffer_policy"], r["gamma"],
                                      r["invalid_entries"]))
        if r["buffer_policy"] == POLICY_FULL and _f(r["gamma"]) == 1.0:
            _put(_cell(r), REF_FULL_G1, r)
    for r in secondary:
        if r["buffer_policy"] == POLICY_PARAMS_ONLY \
                and _f(r["gamma"]) == 1.0:
            _put(_cell(r), REF_PARAMS_ONLY_G1, r)
    for r in rules:
        if r["rule"] == "validation_selected_scaling_per_source":
            _put(_cell(r), CAND_VALSEL, r)
        elif r["rule"] == "fixed_global_gamma_dev_grid_exact":
            _put(_cell(r), REF_FIXED_GLOBAL, r)
        elif r["rule"] == "fixed_phase_gamma_dev_grid_exact":
            _put(_cell(r), REF_FIXED_PHASE, r)

    # Dev-seed fixed-scale recovery from EXISTING grid rows (the frozen
    # gammas are grid points).  No new evaluation.
    n_recovered = 0
    for r in grid:
        if r["buffer_policy"] != POLICY_FULL:
            continue
        cell = _cell(r)
        rd = cell[1]
        slot = obs.setdefault(cell, {})
        if REF_FIXED_GLOBAL not in slot \
                and _f(r["gamma"]) == gamma_dev_global:
            _put(cell, REF_FIXED_GLOBAL, r, recovered=True)
            n_recovered += 1
        if REF_FIXED_PHASE not in slot and rd in gamma_dev_phase \
                and _f(r["gamma"]) == gamma_dev_phase[rd]:
            _put(cell, REF_FIXED_PHASE, r, recovered=True)
            n_recovered += 1

    # ---- sample-count cross-check for rule-side policies -----------
    # Rule rows do not persist n_eval; verify each (seed, round,
    # source, policy, evaluation) per-example block has the same row
    # count as the grid rows' n_eval.
    grid_n_eval = sorted({int(r["n_eval"]) for r in grid})
    pe_counts = defaultdict(int)
    with open(res / "per_example_losses.csv") as f:
        for r in csv.DictReader(f):
            if r["holdout"] == "evaluation":
                pe_counts[(r["seed"], r["round"], r["source"],
                           r["policy"])] += 1
    pe_bad = {k: v for k, v in pe_counts.items()
              if grid_n_eval and v != grid_n_eval[0]}

    # ---- paired comparisons ----------------------------------------
    pair_rows = []
    n_missing = defaultdict(int)
    n_invalid_pairs = defaultdict(int)
    n_ncount_mismatch = 0
    for cell in sorted(obs.keys()):
        for comp, ref_name, cand_name in COMPARISONS:
            ref = obs[cell].get(ref_name)
            cand = obs[cell].get(cand_name)
            if ref is None or cand is None:
                n_missing[comp] += 1
                continue
            if not (ref["candidate_valid"] and cand["candidate_valid"]):
                n_invalid_pairs[comp] += 1
                continue
            if (ref["n_eval"] is not None and cand["n_eval"] is not None
                    and ref["n_eval"] != cand["n_eval"]):
                n_ncount_mismatch += 1
                continue
            pair_rows.append({
                "seed": cell[0], "round": cell[1], "source": cell[2],
                "cohort_id": next(
                    r["cohort_id"] for r in grid
                    if _cell(r) == cell),
                "rep_seed": next(
                    r["rep_seed"] for r in grid
                    if _cell(r) == cell),
                "comparison": comp,
                "reference": ref_name, "candidate": cand_name,
                "gamma_reference": ref["gamma"],
                "gamma_candidate": cand["gamma"],
                "loss_reference": ref["eval_loss"],
                "loss_candidate": cand["eval_loss"],
                "loss_improvement": ref["eval_loss"] - cand["eval_loss"],
                "acc_reference": ref["eval_acc"],
                "acc_candidate": cand["eval_acc"],
                "acc_improvement": cand["eval_acc"] - ref["eval_acc"],
                "n_eval_reference": ref["n_eval"],
                "n_eval_candidate": cand["n_eval"],
                "recovered_from_grid_row":
                    ref["recovered_from_grid_row"]
                    or cand["recovered_from_grid_row"],
            })

    with open(out / "paired_by_source.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    # ---- cohort summary: average repetitions WITHIN cohort first ---
    # Trajectory rows (cohort_id == "trajectory") are kept separate and
    # never pooled with repetition cohorts.
    cohort_groups = defaultdict(list)
    for r in pair_rows:
        cohort_groups[(r["seed"], r["round"], r["cohort_id"],
                       r["comparison"])].append(r)
    cohort_rows = []
    for (seed, rd, cohort, comp), members in sorted(cohort_groups.items()):
        n = len(members)
        row = {
            "seed": seed, "round": rd, "cohort_id": cohort,
            "comparison": comp, "n_reps_averaged": n,
            "is_trajectory": cohort == "trajectory",
            "mean_loss_improvement": sum(
                m["loss_improvement"] for m in members) / n,
            "mean_acc_improvement": sum(
                m["acc_improvement"] for m in members) / n,
            "mean_loss_reference": sum(
                m["loss_reference"] for m in members) / n,
            "mean_loss_candidate": sum(
                m["loss_candidate"] for m in members) / n,
            "mean_gamma_candidate": sum(
                m["gamma_candidate"] for m in members) / n,
            "any_recovered_from_grid_row": any(
                m["recovered_from_grid_row"] for m in members),
        }
        cohort_rows.append(row)
    with open(out / "cohort_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cohort_rows[0].keys()))
        w.writeheader()
        w.writerows(cohort_rows)

    # ---- report -----------------------------------------------------
    rep_cohorts = [c for c in cohort_rows if not c["is_trajectory"]]
    traj = [c for c in cohort_rows if c["is_trajectory"]]

    def _table(rows, comp, seeds=(201, 202, 203)):
        lines = [f"| seed | round | cohort | n_reps | "
                 f"mean loss ref | mean loss cand | loss improvement | "
                 f"acc improvement |",
                 f"|------|-------|--------|--------|---------------|"
                 f"----------------|------------------|-----------------|"]
        for r in rows:
            if r["comparison"] != comp or r["seed"] not in seeds:
                continue
            lines.append(
                f"| {r['seed']} | {r['round']} | {r['cohort_id']} | "
                f"{r['n_reps_averaged']} | "
                f"{r['mean_loss_reference']:.4f} | "
                f"{r['mean_loss_candidate']:.4f} | "
                f"{r['mean_loss_improvement']:+.4f} | "
                f"{r['mean_acc_improvement']:+.4f} |")
        return lines

    def _overall(rows, comp, seeds=(202, 203)):
        sel = [r for r in rows
               if r["comparison"] == comp and r["seed"] in seeds]
        if not sel:
            return None
        return (sum(r["mean_loss_improvement"] for r in sel) / len(sel),
                sum(r["mean_acc_improvement"] for r in sel) / len(sel),
                len(sel))

    L = []
    L.append("# Buffer-policy and scaling analysis (saved artifacts "
             "only, REVISED)")
    L.append("")
    L.append(f"- run: `{run.name}`")
    L.append(f"- analysis output dir: `{out}` (originals preserved; "
             "supersedes analysis_buffer_policy/ with the dev/held-out "
             "separation and without the withdrawn interaction claim)")
    L.append("- positive loss improvement = loss_reference "
             "- loss_candidate; positive accuracy improvement = "
             "acc_candidate - acc_reference")
    L.append("- repetitions averaged WITHIN each cohort first; "
             "trajectory sources are reported separately and never "
             "pooled with repetition cohorts")
    L.append("- development seed 201 (used to select the fixed scales) "
             "and held-out seeds 202/203 are reported SEPARATELY in "
             "every summary")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(f"- command: `python -X utf8 "
             f"{Path(sys.argv[0]).as_posix()} --run-dir "
             f"{args.run_dir} --out-dir {args.out_dir}`")
    for rel, h in input_hashes.items():
        L.append(f"- sha256 `{rel}`: `{h}`")
    L.append("")
    L.append("## Checks")
    L.append("")
    L.append(f"- schema: required columns present in all three CSVs: "
             f"{schema_ok}")
    L.append(f"- grid rows: {len(grid)}; secondary rows: "
             f"{len(secondary)}; rule rows: {len(rules)}")
    L.append(f"- invalid grid candidates: {len(invalid_grid_rows)} "
             f"(all at gamma=2.0 — negative running_var; none intersect "
             f"the compared rows at gamma=1 / rule rows; rule rows "
             f"invalid: 0)")
    L.append(f"- missing pairs per comparison: "
             f"{dict(n_missing) if n_missing else 'none'}")
    L.append(f"- invalid pairs excluded: "
             f"{dict(n_invalid_pairs) if n_invalid_pairs else 'none'}")
    L.append(f"- n_eval mismatches in pairs: {n_ncount_mismatch}; "
             f"grid/secondary n_eval values: {grid_n_eval}")
    L.append(f"- rule rows do not persist n_eval; per-example row "
             f"counts per (seed, round, source, policy, evaluation) "
             f"block vs n_eval={grid_n_eval}: "
             f"{'all match' if not pe_bad else pe_bad}")
    L.append(f"- dev-seed fixed-scale measurements recovered from "
             f"existing grid rows at the frozen gammas "
             f"(gamma_global={gamma_dev_global}, gamma_phase="
             f"{gamma_dev_phase}): {n_recovered} rows labeled "
             f"recovered_from_grid_row (no new evaluation)")
    L.append("")
    L.append("## Measured findings")
    for comp, _, _ in COMPARISONS:
        L.append("")
        L.append(f"### {comp}")
        L.append("")
        L.extend(_table(rep_cohorts, comp))
        for label, sd in (("dev seed 201 (selection source; NOT a "
                           "held-out evaluation)", (201,)),
                          ("held-out seeds (202, 203)", (202, 203))):
            o = _overall(rep_cohorts, comp, sd)
            if o:
                L.append(f"- cohort-mean over {label}: "
                         f"loss improvement {o[0]:+.4f}, "
                         f"acc improvement {o[1]:+.4f} "
                         f"(n_cohort_cells={o[2]})")
        tr = [c for c in traj if c["comparison"] == comp]
        if tr:
            L.append("- trajectory (single observation per "
                     "seed/round, reported separately):")
            for r in tr:
                L.append(f"    - seed {r['seed']} round {r['round']}: "
                         f"loss improvement {r['mean_loss_improvement']:+.4f}, "
                         f"acc improvement {r['mean_acc_improvement']:+.4f}")
    L.append("")
    L.append("## Interpretations (separate from the measurements above)")
    L.append("")
    # Sign consistency per comparison: reported SEPARATELY for all 27
    # cohort cells (3 seeds x 3 rounds x 3 repetition cohorts) and for
    # the 18 held-out cohort cells (2 seeds x 3 rounds x 3 cohorts).
    for comp, _, _ in COMPARISONS:
        for label, sdgroup in (
                ("all seeds (27 cohort cells)", (201, 202, 203)),
                ("held-out seeds only (18 cohort cells)", (202, 203))):
            sel = [c for c in rep_cohorts
                   if c["comparison"] == comp and c["seed"] in sdgroup]
            n_pos = sum(1 for c in sel
                        if c["mean_loss_improvement"] > 1e-12)
            n_neg = sum(1 for c in sel
                        if c["mean_loss_improvement"] < -1e-12)
            n_zero = len(sel) - n_pos - n_neg
            L.append(f"- {comp} [{label}]: {n_pos} positive / "
                     f"{n_neg} negative / {n_zero} zero of "
                     f"{len(sel)} cohort cells.")
    L.append("")
    a_dev = _overall(rep_cohorts, "A_freeze_buffers_at_gamma1", (201,))
    a_held = _overall(rep_cohorts, "A_freeze_buffers_at_gamma1",
                      (202, 203))
    if a_held:
        verdict = ("improves" if a_held[0] > 0 else
                   "does not improve" if a_held[0] < 0 else
                   "is neutral vs")
        L.append(f"- Freezing checkpoint buffers at gamma=1 in THIS "
                 f"setting {verdict} full-state FedAvg on the "
                 f"evaluation split (held-out cohort-mean loss "
                 f"improvement {a_held[0]:+.4f}; dev seed "
                 f"{a_dev[0]:+.4f}). This conclusion is limited to "
                 f"freezing checkpoint buffers at gamma=1 in this "
                 f"setting; it does NOT generalize to other "
                 f"normalization or buffering methods.")
    b1 = _overall(rep_cohorts, "B_valsel_vs_full_gamma1", (202, 203))
    b2 = _overall(rep_cohorts, "B_valsel_vs_params_only_gamma1",
                  (202, 203))
    if b1 and b2:
        L.append(f"- Validation-selected damping vs full-state "
                 f"gamma=1: {b1[0]:+.4f}; vs frozen-buffer gamma=1: "
                 f"{b2[0]:+.4f} (held-out cohort means). "
                 + ("It improves over BOTH gamma=1 references."
                    if b1[0] > 0 and b2[0] > 0 else
                    "It does NOT improve over both gamma=1 references."))
        L.append(f"  Both comparisons use the SAME "
                 f"validation-selected candidate, so the difference "
                 f"between the two improvements ({b2[0]:+.4f} vs "
                 f"{b1[0]:+.4f}) reflects the different REFERENCE "
                 f"losses; it is not evidence of an interaction "
                 f"between damping and buffer handling.")
    c1 = _overall(rep_cohorts, "C_valsel_vs_fixed_global", (202, 203))
    c2 = _overall(rep_cohorts, "C_valsel_vs_fixed_phase", (202, 203))
    c1d = _overall(rep_cohorts, "C_valsel_vs_fixed_global", (201,))
    c2d = _overall(rep_cohorts, "C_valsel_vs_fixed_phase", (201,))
    if c1 and c2:
        L.append(f"- Validation-selected vs dev-selected fixed scales "
                 f"(held-out): vs global {c1[0]:+.4f}, vs phase "
                 f"{c2[0]:+.4f}. (Dev seed, not a held-out "
                 f"evaluation: vs global {c1d[0]:+.4f}, vs phase "
                 f"{c2d[0]:+.4f}.)")
    L.append("")
    L.append("- Caveat (frozen): the frozen-buffer scale grid was NOT "
             "measured (only gamma=1 under the secondary buffer "
             "policy), so parameter-scaling and buffer-scaling effects "
             "remain INCOMPLETELY SEPARATED.")
    L.append("")
    L.append("## Additional measurements")
    L.append("")
    L.append("An evaluation-only extension of the secondary buffer "
             "policy over the existing gamma grid on the existing "
             "checkpoints was identified earlier as the smallest "
             "separating measurement. Per the pilot closure decision "
             "it is NOT launched, and no further tuning against the "
             "same held-out seeds is performed.")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out / 'paired_by_source.csv'} ({len(pair_rows)} pairs)")
    print(f"wrote {out / 'cohort_summary.csv'} ({len(cohort_rows)} rows)")
    print(f"wrote {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
