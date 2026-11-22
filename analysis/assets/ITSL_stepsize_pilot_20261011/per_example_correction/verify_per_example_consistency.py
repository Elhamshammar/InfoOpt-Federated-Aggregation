#!/usr/bin/env python
"""Consistency verification for per-example loss exports (HARDENED).

For every cell (seed, round, source, cohort_id, rep_seed, holdout, policy)
present in a per-example CSV, verify against the retained grid/rule
summaries of the same run — NOT merely order-reversed self-agreement.

Hardening (2026-10-08):
  1. Non-finite values are rejected wherever they matter: ANY non-finite
     per-example loss/gamma is fatal; any value actually USED as an
     authoritative reference (gammas, summary losses, retained mapping,
     dev gammas) must be finite or the cell fails. Summary rows that
     deliberately record NaN for INVALID candidates (candidate_valid=
     False, e.g. gamma=2.0 extrapolations) are catalogued and never
     consumed; if such a row is ever selected as a reference, that cell
     fails loudly.
  2. Unknown policies and undecidable cells (no authoritative reference)
     are FAILURES, not warnings.
  3. Summaries are filtered explicitly to buffer_policy=all_floating_state;
     duplicate authoritative keys abort the run.
  4. Example IDs must be unique within a cell and the per-cell example
     set must equal the expected holdout index set from partition.json.
  5. Missing cells are ALWAYS reported. A partial verification REQUIRES
     --expect-inventory <json> listing the exact expected cells and total
     row count; present cells must equal that inventory exactly. An empty
     per-example file always fails, in every mode.

Exit status 1 on any verification failure, 2 on malformed inputs.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

DECISIVE = {
    "fedavg": "grid_exact",
    "fixed_global_gamma_dev": "rule_global_fixed_dev",
    "fixed_phase_gamma_dev": "rule_phase_fixed_dev",
    "validation_selected_scaling": "rule_validation_selected",
    "claude_raw_experimental": "rule_claude",
}

PRIMARY_BUFFER_POLICY = "all_floating_state"


def fail(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def load_json(p):
    with open(p) as f:
        return json.load(f)


def parse_finite(value, what):
    """Used references MUST be finite; anything else aborts."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        fail(f"unparseable numeric value for {what}: {value!r}")
    if not math.isfinite(v):
        fail(f"non-finite value for {what}: {value!r}")
    return v


def parse_tolerant(value):
    """Summary values: NaN/inf are returned (not raised) so deliberately
    recorded invalid-candidate rows can be catalogued. Values actually
    USED as references are validated at use time."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_grid(res):
    """Grid summary filtered to the primary buffer policy; duplicate keys
    are fatal. Non-finite loss rows are catalogued, never silently used."""
    grid, n_by_source = {}, {}
    with open(res / "aggregate_evaluation.csv") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            if r.get("buffer_policy") != PRIMARY_BUFFER_POLICY:
                continue
            gamma = parse_tolerant(r["gamma"])
            if not math.isfinite(gamma):
                fail(f"non-finite grid gamma line {i}: {r['gamma']!r}")
            key = (r["seed"], r["round"], r["source"], r["cohort_id"],
                   r["rep_seed"], round(gamma, 6))
            if key in grid:
                fail(f"duplicate authoritative grid key after "
                     f"{PRIMARY_BUFFER_POLICY} filter: {key}")
            grid[key] = {
                "sel_loss": parse_tolerant(r["sel_loss"]),
                "eval_loss": parse_tolerant(r["eval_loss"]),
                "candidate_valid": r.get("candidate_valid"),
            }
            skey = key[:5]
            try:
                n = (int(r["n_sel"]), int(r["n_eval"]))
            except (TypeError, ValueError):
                fail(f"unparseable n_sel/n_eval line {i}")
            if skey in n_by_source and n_by_source[skey] != n:
                fail(f"inconsistent n_sel/n_eval within source {skey}")
            n_by_source[skey] = n
    return grid, n_by_source


def load_rules(res):
    rules = {}
    with open(res / "rule_evaluation.csv") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            if r.get("buffer_policy") != PRIMARY_BUFFER_POLICY:
                continue
            key = (r["seed"], r["round"], r["source"], r["cohort_id"],
                   r["rep_seed"], r["rule"])
            if key in rules:
                fail(f"duplicate authoritative rule key after "
                     f"{PRIMARY_BUFFER_POLICY} filter: {key}")
            gamma = parse_tolerant(r["gamma"])
            if not math.isfinite(gamma):
                fail(f"non-finite rule gamma line {i}: {r['gamma']!r}")
            rules[key] = {
                "gamma": gamma,
                "sel_loss": parse_tolerant(r["sel_loss"]),
                "eval_loss": parse_tolerant(r["eval_loss"]),
                "candidate_valid": r.get("candidate_valid"),
            }
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--per-example", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--loss-atol", type=float, default=3e-4)
    ap.add_argument("--expect-inventory", default=None,
                    help="JSON: {\"cells\": [[seed, round, source, "
                         "cohort_id, rep_seed, holdout, policy], ...], "
                         "\"expected_total_rows\": int}. REQUIRED for "
                         "partial verification; present cells must equal "
                         "this inventory exactly.")
    args = ap.parse_args()

    run = Path(args.run_dir)
    res = run / "results"
    pe_path = Path(args.per_example) if args.per_example else \
        res / "per_example_losses.csv"
    if not pe_path.is_file():
        fail(f"per-example file not found: {pe_path}")

    grid, n_by_source = load_grid(res)
    rules = load_rules(res)
    dec = load_json(res / "pilot_decision.json")
    vg = {str(k): parse_finite(v, f"retained gamma {k}")
          for k, v in dec["per_source_validation_gamma"].items()}
    g_global = parse_finite(
        load_json(res / "gamma_dev_global.json")["gamma_dev_global"],
        "gamma_dev_global")
    g_phase = {}
    for p in sorted(res.glob("gamma_dev_phase_*.json")):
        d = load_json(p)
        g_phase[int(d["phase_round"])] = parse_finite(
            d["gamma_dev_phase"], f"gamma_dev_phase in {p.name}")

    part = load_json(run / "partition.json")
    holdout_index_sets = {
        "selection": set(int(i) for i in part["selection_indices"]),
        "evaluation": set(int(i) for i in part["evaluation_indices"]),
    }

    expect_inv = None
    if args.expect_inventory:
        inv = load_json(args.expect_inventory)
        expect_inv = {
            "cells": {tuple(str(x) for x in cell) for cell in inv["cells"]},
            "expected_total_rows": int(parse_finite(
                inv["expected_total_rows"], "expected_total_rows")),
        }

    def expected(policy, seed, rd, source, cohort, rep, holdout):
        """Return (gamma, loss, n, ref_problems) or None if undecidable.
        ref_problems is non-empty when the selected authoritative row is
        itself invalid (non-finite loss or candidate_valid=False)."""
        n_sel, n_eval = n_by_source.get((seed, rd, source, cohort, rep),
                                        (None, None))
        n = n_sel if holdout == "selection" else n_eval
        lossf = "sel_loss" if holdout == "selection" else "eval_loss"

        def ref(row):
            problems = []
            if row.get("candidate_valid") == "False":
                problems.append("authoritative reference row is marked "
                                "candidate_valid=False")
            if not math.isfinite(row[lossf]):
                problems.append(f"authoritative reference {lossf} is "
                                f"non-finite ({row[lossf]})")
            return problems

        if policy == "fedavg":
            row = rules.get((seed, rd, source, cohort, rep,
                             "fedavg_gamma1_grid_exact"))
            return (1.0, row[lossf], n, ref(row)) if row else None
        if policy == "validation_selected_scaling":
            k = f"{seed}|{rd}|{source}"
            if k not in vg:
                return None
            row = rules.get((seed, rd, source, cohort, rep,
                             "validation_selected_scaling_per_source"))
            return (vg[k], row[lossf], n, ref(row)) if row else None
        if policy == "fixed_global_gamma_dev":
            row = grid.get((seed, rd, source, cohort, rep,
                            round(g_global, 6)))
            return (g_global, row[lossf], n, ref(row)) if row else None
        if policy == "fixed_phase_gamma_dev":
            g = g_phase.get(int(rd))
            if g is None:
                return None
            row = grid.get((seed, rd, source, cohort, rep, round(g, 6)))
            return (g, row[lossf], n, ref(row)) if row else None
        if policy == "claude_raw_experimental":
            row = rules.get((seed, rd, source, cohort, rep,
                             "claude_raw_experimental"))
            return (row["gamma"], row[lossf], n, ref(row)) if row else None
        return None

    # --- stream the per-example CSV ---
    cells = {}
    total_rows = 0
    with open(pe_path) as f:
        for line_no, r in enumerate(csv.DictReader(f), start=2):
            total_rows += 1
            key = (r["seed"], r["round"], r["source"], r["cohort_id"],
                   r["rep_seed"], r["holdout"], r["policy"])
            loss = parse_finite(r["loss"],
                                f"per-example loss line {line_no}")
            gamma = parse_finite(r["gamma"],
                                 f"per-example gamma line {line_no}")
            try:
                ex = int(r["example_index"])
            except (TypeError, ValueError):
                fail(f"unparseable example_index line {line_no}: "
                     f"{r['example_index']!r}")
            c = cells.get(key)
            if c is None:
                c = cells[key] = {"n": 0, "sum_loss": 0.0, "gammas": set(),
                                  "origins": set(), "examples": set()}
            c["n"] += 1
            c["sum_loss"] += loss
            c["gammas"].add(gamma)
            c["origins"].add(r["gamma_origin"])
            c["examples"].add(ex)

    if total_rows == 0:
        fail(f"per-example file is EMPTY: {pe_path}; an empty export is "
             f"never acceptable")

    mismatches, undecidable, checked = [], [], 0
    for key, c in sorted(cells.items()):
        seed, rd, source, cohort, rep, holdout, policy = key
        if policy not in DECISIVE:
            mismatches.append({"cell": key,
                               "problems": [f"unknown policy {policy!r}"]})
            continue
        exp = expected(policy, seed, rd, source, cohort, rep, holdout)
        if exp is None:
            undecidable.append({"cell": key,
                                "reason": "no authoritative reference"})
            continue
        checked += 1
        g_exp, loss_exp, n_exp, ref_problems = exp
        problems = list(ref_problems)
        if c["origins"] != {DECISIVE[policy]}:
            problems.append(f"gamma_origin {sorted(c['origins'])} != "
                            f"{DECISIVE[policy]}")
        if len(c["gammas"]) != 1:
            problems.append(f"multiple gammas in cell: "
                            f"{sorted(c['gammas'])}")
        elif abs(next(iter(c["gammas"])) - g_exp) > 1e-6:
            problems.append(f"gamma {next(iter(c['gammas']))} != "
                            f"expected {g_exp}")
        if n_exp is not None and c["n"] != n_exp:
            problems.append(f"n {c['n']} != expected {n_exp}")
        if len(c["examples"]) != c["n"]:
            problems.append(f"duplicate example IDs: {c['n']} rows but "
                            f"{len(c['examples'])} unique example_index")
        expected_set = holdout_index_sets.get(holdout)
        if expected_set is None:
            problems.append(f"unknown holdout {holdout!r}")
        elif c["examples"] != expected_set:
            only_cell = len(c["examples"] - expected_set)
            only_ref = len(expected_set - c["examples"])
            problems.append(f"example set mismatch: {only_cell} IDs only "
                            f"in cell, {only_ref} only in partition "
                            f"{holdout} set")
        mean_loss = c["sum_loss"] / c["n"] if c["n"] else float("nan")
        if math.isfinite(loss_exp) and (
                not math.isfinite(mean_loss)
                or abs(mean_loss - loss_exp) > args.loss_atol):
            problems.append(f"mean_loss {mean_loss:.6f} != summary "
                            f"{loss_exp:.6f} (atol {args.loss_atol})")
        if problems:
            mismatches.append({"cell": key, "problems": problems})

    # --- missing cells (always reported) ---
    missing = []
    if expect_inv is not None:
        present = {tuple(str(x) for x in key) for key in cells}
        for cell in sorted(expect_inv["cells"] - present):
            missing.append({"cell": cell, "vs": "expect-inventory"})
        unexpected = present - expect_inv["cells"]
        if unexpected:
            mismatches.append({
                "cell": sorted(unexpected)[:10],
                "problems": [f"{len(unexpected)} cells present but NOT in "
                             f"the expected inventory"]})
        if total_rows != expect_inv["expected_total_rows"]:
            mismatches.append({
                "cell": "(file)",
                "problems": [f"total rows {total_rows} != expected "
                             f"{expect_inv['expected_total_rows']}"]})
    else:
        for (seed, rd, source, cohort, rep) in n_by_source:
            for policy in DECISIVE:
                for holdout in ("selection", "evaluation"):
                    key = (seed, rd, source, cohort, rep, holdout, policy)
                    if key not in cells:
                        crow = rules.get((seed, rd, source, cohort, rep,
                                          "claude_raw_experimental"))
                        if policy == "claude_raw_experimental" and crow \
                                and crow.get("candidate_valid") == "False":
                            continue  # legitimately absent invalid candidate
                        missing.append({"cell": key})

    failed = bool(mismatches) or bool(undecidable) or bool(missing)
    report = {
        "per_example_file": str(pe_path),
        "run_dir": str(run),
        "buffer_policy_filter": PRIMARY_BUFFER_POLICY,
        "total_rows": total_rows,
        "cells_present": len(cells),
        "cells_checked": checked,
        "n_mismatched": len(mismatches),
        "mismatches": mismatches[:50],
        "n_undecidable": len(undecidable),
        "undecidable_cells": undecidable[:20],
        "n_missing_cells": len(missing),
        "missing_cells_sample": missing[:20],
        "expect_inventory": args.expect_inventory,
        "verdict": "FAIL" if failed else "PASS",
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("total_rows", "cells_present", "cells_checked",
                       "n_mismatched", "n_undecidable", "n_missing_cells",
                       "verdict")}, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
