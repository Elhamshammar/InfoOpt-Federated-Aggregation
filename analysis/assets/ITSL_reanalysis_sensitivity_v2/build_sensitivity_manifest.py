#!/usr/bin/env python
"""Build the frozen sensitivity/RA source inventory (v11).

Separately dated, hashed source inventory for:
  - latest-repeat sensitivity (one latest-in-stratum run per
    repeat-selection cell);
  - CIFAR FedOpt sensitivity-only (the five completed, never-triggered
    early-stop-enabled slr0.01 runs);
  - RA-InfoOpt per-stratum summaries (all eligible RA runs, stratum-labelled;
    per-seed earliest selection happens at analysis time).

Every record carries run identity, all locations, and SHA-256 of
metrics.csv / config_used.json / summary.json. Sources are verified
against these hashes before any calculation (run_sensitivity_reanalysis.py
aborts on any mismatch — no substitution, no fallback).
"""
import csv
import json
from collections import Counter
from pathlib import Path

ELIG = "candidate_eligibility_map_v4.csv"
MANIFEST = "reanalysis_20261008/frozen_inputs/proposed_source_manifest_FROZEN_20261008.csv"
STRATA = "campaign_strata_v4.json"
OUT = "sensitivity_source_inventory.csv"

DS2EXP = {"cifar100": "cifar100_sweep_v2", "eurosat": "eurosat_sweep",
          "pathmnist": "pathmnist_sweep"}


def main():
    elig = list(csv.DictReader(open(ELIG, encoding="utf-8")))
    elig_ok = [r for r in elig if r["eligible_draft"] == "True"]
    man = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    strata = json.load(open(STRATA, encoding="utf-8"))["cell_statuses"]

    def fam_of(ds, method, slug, gamma):
        return (DS2EXP[ds], method, slug, gamma)

    records = []

    # --- latest-repeat sensitivity: one latest run per repeat-selection cell
    for r in man:
        if not r["status"].startswith("PROPOSED (retrospective_repeat"):
            continue
        ds, row, seed = r["dataset"], r["manuscript_row"], int(r["seed"])
        fam = tuple(r["family"].split("|"))
        key = "|".join(fam) + f"|{seed}"
        consistent = strata.get(key, {}).get("consistent_strata") or []
        assert len(consistent) == 1, f"no unique stratum for {key}"
        cands = [c for c in elig_ok
                 if (c["experiment"], c["method"], c["param_slug"],
                     c["gamma_override"]) == fam
                 and int(c["seed"]) == seed
                 and c["config_stratum"] == consistent[0]]
        cands.sort(key=lambda c: c["timestamp"])
        latest = cands[-1]
        assert latest["run_name"] != r["run_name"] or len(cands) == 1
        records.append(_rec("latest_repeat_sensitivity", ds, row, seed,
                            consistent[0], latest,
                            note=f"latest in stratum; earliest is primary "
                                 f"({r['run_name']})"))

    # --- CIFAR FedOpt sensitivity-only runs ---
    for r in man:
        if r["manuscript_row"] == "fedopt" and r["dataset"] == "cifar100" \
                and "sensitivity only" in r["sensitivity_candidates"]:
            lname = r["sensitivity_candidates"].split(" (")[0]
            c = next(c for c in elig if c["run_name"] == lname)
            records.append(_rec("cifar_fedopt_sensitivity_only",
                                "cifar100", "fedopt", int(r["seed"]),
                                c["config_stratum"], c,
                                note=("completed 100 rounds; early stop "
                                      "enabled but never triggered; strict "
                                      "E3 keeps it out of primary")))

    # --- RA per-stratum: ALL eligible RA runs (selection at analysis time)
    for r in elig_ok:
        if r["method"] != "RA_InfoOpt" or r["param_slug"] \
                or r["gamma_override"]:
            continue
        ds = {v: k for k, v in DS2EXP.items()}[r["experiment"]]
        records.append(_rec("ra_stratum_member", ds, "RA_InfoOpt",
                            int(r["seed"]), r["config_stratum"], r,
                            note="per-stratum summary member"))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(dict(Counter(r["role"] for r in records)))
    print(f"wrote {OUT} ({len(records)} records)")


def _rec(role, ds, row, seed, stratum, c, note):
    return {
        "role": role, "dataset": ds, "manuscript_row": row, "seed": seed,
        "config_stratum": stratum,
        "run_name": c["run_name"], "timestamp": c.get("timestamp", ""),
        "locations": c["locations"],
        "sha256_metrics.csv": c["sha256_metrics.csv"],
        "sha256_config_used.json": c["sha256_config_used.json"],
        "sha256_summary.json": c["sha256_summary.json"],
        "note": note,
    }


if __name__ == "__main__":
    main()
