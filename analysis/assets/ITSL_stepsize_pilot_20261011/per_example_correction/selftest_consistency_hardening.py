#!/usr/bin/env python
"""Regression tests for the hardened consistency checker.

Builds a synthetic run directory (grid/rule summaries, pilot_decision,
gamma_dev files, partition) and a clean per-example export, then verifies
that the checker PASSES the clean case and FAILS each of these false-PASS
cases:

  1. non-finite per-example loss
  2. unknown policy label
  3. undecidable cell (no authoritative reference)
  4. duplicate authoritative grid key (after buffer-policy filter)
  5. wrong example set (one example swapped for an out-of-holdout ID)
  6. duplicate example IDs within a cell
  7. the original 2026-10-08 defect pattern (gamma=1 duplicating FedAvg
     where intended gamma is 0.5)
  8. empty per-example file
  9. partial inventory with one missing cell
 10. partial inventory rejecting an unexpected extra cell
 11. non-finite reference value (NaN in gamma_dev JSON)
 12. gamma_origin mismatch
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
CHECKER = REPO / "scripts" / "verify_per_example_consistency.py"

SEED, RD, SRC, COHORT, REP = "201", "5", "trajectory_cohort", "trajectory", "trajectory"
SEL = [0, 1, 2, 3]
EV = [4, 5, 6, 7]
HOLD = {"selection": SEL, "evaluation": EV}
N_SEL, N_EV = 4, 4

POLICIES = {
    "fedavg": ("grid_exact", 1.0, 1.0, 2.0),
    "fixed_global_gamma_dev": ("rule_global_fixed_dev", 0.5, 1.1, 2.1),
    "fixed_phase_gamma_dev": ("rule_phase_fixed_dev", 0.75, 1.2, 2.2),
    "validation_selected_scaling": ("rule_validation_selected", 0.5, 1.3, 2.3),
    "claude_raw_experimental": ("rule_claude", 0.5, 1.4, 2.4),
}

GRID_FIELDS = ["seed", "round", "source", "cohort_id", "rep_seed",
               "buffer_policy", "gamma", "sel_loss", "sel_acc", "eval_loss",
               "eval_acc", "n_sel", "n_eval"]
RULE_FIELDS = ["seed", "round", "source", "cohort_id", "rep_seed",
               "buffer_policy", "rule", "gamma", "sel_loss", "eval_loss",
               "candidate_valid"]
PE_FIELDS = ["seed", "round", "source", "rep_seed", "cohort_id", "holdout",
             "policy", "gamma", "gamma_origin", "example_index", "truth",
             "loss", "correct"]


def build_run_dir(root):
    res = root / "results"
    res.mkdir(parents=True)
    # grid: gamma rows 0.5, 0.75, 1.0 for the source (fixed policies read
    # their losses from grid rows at their gamma)
    grid_losses = {0.5: (1.1, 2.1), 0.75: (1.2, 2.2), 1.0: (1.0, 2.0)}
    with open(res / "aggregate_evaluation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(GRID_FIELDS)
        for g, (sl, el) in grid_losses.items():
            w.writerow([SEED, RD, SRC, COHORT, REP, "all_floating_state",
                        g, sl, 0.5, el, 0.5, N_SEL, N_EV])
        # a params_only row that MUST be ignored by the filter
        w.writerow([SEED, RD, SRC, COHORT, REP,
                    "parameters_only_frozen_buffers", 1.0, 9.9, 0.1, 9.9,
                    0.1, N_SEL, N_EV])
    with open(res / "rule_evaluation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RULE_FIELDS)
        w.writerow([SEED, RD, SRC, COHORT, REP, "all_floating_state",
                    "fedavg_gamma1_grid_exact", 1.0, 1.0, 2.0, "True"])
        w.writerow([SEED, RD, SRC, COHORT, REP, "all_floating_state",
                    "validation_selected_scaling_per_source", 0.5, 1.3,
                    2.3, "True"])
        w.writerow([SEED, RD, SRC, COHORT, REP, "all_floating_state",
                    "claude_raw_experimental", 0.5, 1.4, 2.4, "True"])
    json.dump({"per_source_validation_gamma":
               {f"{SEED}|{RD}|{SRC}": 0.5}},
              open(res / "pilot_decision.json", "w"))
    json.dump({"gamma_dev_global": 0.5},
              open(res / "gamma_dev_global.json", "w"))
    json.dump({"gamma_dev_phase": 0.75, "phase_round": int(RD)},
              open(res / f"gamma_dev_phase_{RD}.json", "w"))
    json.dump({"selection_indices": SEL, "evaluation_indices": EV},
              open(root / "partition.json", "w"))


def write_pe(path, mutate=None):
    rows = []
    for policy, (origin, gamma, sl, el) in POLICIES.items():
        for holdout, idxs in HOLD.items():
            mean = sl if holdout == "selection" else el
            for ex in idxs:
                rows.append([SEED, RD, SRC, REP, COHORT, holdout, policy,
                             f"{gamma:.6f}", origin, ex, 0, mean, 0])
    if mutate:
        rows = mutate(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PE_FIELDS)
        w.writerows(rows)
    return rows


def inventory(cells_policy_subset=None, path=None):
    cells = []
    for policy in (cells_policy_subset or POLICIES):
        for holdout in HOLD:
            cells.append([SEED, RD, SRC, COHORT, REP, holdout, policy])
    inv = {"cells": cells,
           "expected_total_rows": len(cells) * N_SEL}
    if path:
        json.dump(inv, open(path, "w"))
    return inv


def run_checker(run_dir, pe, expect=None):
    cmd = [sys.executable, str(CHECKER), "--run-dir", str(run_dir),
           "--per-example", str(pe)]
    if expect:
        cmd += ["--expect-inventory", str(expect)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


RESULTS = []


def case(name, fn, expect_pass):
    with tempfile.TemporaryDirectory() as td:
        code, out = fn(Path(td))
    ok = (code == 0) if expect_pass else (code != 0)
    RESULTS.append((name, ok, out[-400:] if not ok else ""))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} (exit={code})")


def base(td):
    run = td / "run"
    build_run_dir(run)
    return run


def main():
    # 0. clean case passes (full inventory mode)
    def clean(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe)
        return run_checker(run, pe)
    case("clean export passes", clean, True)

    # 0b. clean case passes with exact partial inventory
    def clean_inv(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe)
        inv = td / "inv.json"
        inventory(path=inv)
        return run_checker(run, pe, expect=inv)
    case("clean export passes with expected inventory", clean_inv, True)

    # 1. non-finite loss
    def nan_loss(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe, mutate=lambda rows: [rows[0][:11] + ["nan"] + rows[0][12:]] + rows[1:])
        return run_checker(run, pe)
    case("non-finite loss rejected", nan_loss, False)

    # 2. unknown policy
    def bad_policy(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe, mutate=lambda rows: [rows[0][:6] + ["mystery_policy"] + rows[0][7:]] + rows[1:])
        return run_checker(run, pe)
    case("unknown policy fails", bad_policy, False)

    # 3. undecidable cell (source absent from summaries)
    def undecidable(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe, mutate=lambda rows: [rows[0][:2] + ["ghost_source"] + rows[0][3:]] + rows[1:])
        return run_checker(run, pe)
    case("undecidable cell fails", undecidable, False)

    # 4. duplicate authoritative grid key
    def dup_grid(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe)
        with open(run / "results" / "aggregate_evaluation.csv", "a",
                  newline="") as f:
            csv.writer(f).writerow(
                [SEED, RD, SRC, COHORT, REP, "all_floating_state", 1.0,
                 1.0, 0.5, 2.0, 0.5, N_SEL, N_EV])
        return run_checker(run, pe)
    case("duplicate authoritative key rejected", dup_grid, False)

    # 5. wrong example set (out-of-holdout ID)
    def wrong_example(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe, mutate=lambda rows: [rows[0][:9] + [999] + rows[0][10:]] + rows[1:])
        return run_checker(run, pe)
    case("out-of-holdout example ID fails", wrong_example, False)

    # 6. duplicate example IDs within a cell
    def dup_example(td):
        run = base(td)
        pe = td / "pe.csv"
        def m(rows):
            rows[3] = rows[3][:9] + [rows[0][9]] + rows[0][10:]
            return rows
        write_pe(pe, mutate=m)
        return run_checker(run, pe)
    case("duplicate example IDs fail", dup_example, False)

    # 7. original defect pattern: gamma=1 duplicating fedavg where 0.5 intended
    def defect(td):
        run = base(td)
        pe = td / "pe.csv"
        def m(rows):
            out = []
            for r in rows:
                if r[6] == "validation_selected_scaling":
                    r = r[:7] + ["1.000000"] + r[8:11] + [1.0] + r[12:]
                out.append(r)
            return out
        write_pe(pe, mutate=m)
        return run_checker(run, pe)
    case("2026-10-08 defect pattern detected", defect, False)

    # 8. empty file
    def empty(td):
        run = base(td)
        pe = td / "pe.csv"
        with open(pe, "w", newline="") as f:
            csv.writer(f).writerow(PE_FIELDS)
        return run_checker(run, pe)
    case("empty per-example file fails", empty, False)

    # 9. partial inventory with one missing cell
    def inv_missing(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe, mutate=lambda rows: [r for r in rows
                                          if not (r[6] == "fedavg"
                                                  and r[5] == "selection")])
        inv = td / "inv.json"
        inventory(path=inv)
        return run_checker(run, pe, expect=inv)
    case("missing cell vs inventory fails", inv_missing, False)

    # 10. unexpected extra cell vs inventory
    def inv_extra(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe)
        inv = td / "inv.json"
        inventory(cells_policy_subset=["fedavg"], path=inv)
        return run_checker(run, pe, expect=inv)
    case("unexpected extra cell vs inventory fails", inv_extra, False)

    # 11. non-finite reference value (NaN in gamma_dev JSON)
    def nan_ref(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe)
        with open(run / "results" / "gamma_dev_global.json", "w") as f:
            f.write('{"gamma_dev_global": NaN}')
        return run_checker(run, pe)
    case("non-finite reference value rejected", nan_ref, False)

    # 12. gamma_origin mismatch
    def bad_origin(td):
        run = base(td)
        pe = td / "pe.csv"
        write_pe(pe, mutate=lambda rows: [rows[0][:8] + ["wrong_origin"] + rows[0][9:]] + rows[1:])
        return run_checker(run, pe)
    case("gamma_origin mismatch fails", bad_origin, False)

    print()
    failed = [n for n, ok, detail in RESULTS if not ok]
    for n, ok, detail in RESULTS:
        if not ok:
            print(f"--- detail for {n} ---\n{detail}")
    if failed:
        print(f"REGRESSION SUITE FAILED: {failed}")
        sys.exit(1)
    print(f"REGRESSION SUITE PASSED: {len(RESULTS)} cases")


if __name__ == "__main__":
    main()
