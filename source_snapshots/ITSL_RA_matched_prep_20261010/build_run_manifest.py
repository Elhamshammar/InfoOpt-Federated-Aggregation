#!/usr/bin/env python3
"""Build/refresh the EXECUTION run manifest for the four-arm experiment.

A cell is marked "complete" ONLY if the full verifier
(verify_run.py --arm) passes for that exact run directory. Otherwise the
cell gets an explicit status and reason:

  pending          — no run directory
  partial          — run directory exists but evidence files are missing
  invalid          — evidence present but the verifier FAILED (reason kept)
  duplicate_error  — more than one run directory for the cell
  complete         — verifier passed (report path + hashes recorded,
                     including partition_used.json)

Writes EXP_ROOT/RUN_MANIFEST.json. The preparation template in the
package is never modified. Verification reports are preserved under
EXP_ROOT/verification/.
"""
import csv
import hashlib
import math
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = Path(os.environ.get("EXP_ROOT",
                               str(HERE)))
RUNS = EXP_ROOT / "runs"
VERIF_DIR = EXP_ROOT / "verification"
DATASETS = ("cifar100", "eurosat", "pathmnist")
SEEDS = (100, 101, 102, 103, 104)
ARMS = ('fedavg_aibon', 'infoopt_aibon', 'ra_probeoff', 'ra_probeon')
EVIDENCE = ("config_used.json", "metrics.csv", "summary.json",
            "final_model.pt", "partition_used.json")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(rd, ds, seed, arm, report_out):
    sched_dir = os.environ.get("SHARED_DIR", str(EXP_ROOT / "shared"))
    r = subprocess.run(
        [sys.executable, str(HERE / "verify_run.py"),
         "--run-dir", str(rd), "--dataset", ds, "--seed", str(seed),
         "--arm", arm, "--schedules-dir", sched_dir,
         "--report-out", str(report_out)],
        capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, (tail[-1] if tail else "")


def main():
    VERIF_DIR.mkdir(parents=True, exist_ok=True)
    cells, problems = [], []
    failed_root = EXP_ROOT / "failed_attempts"
    for ds in DATASETS:
        for seed in SEEDS:
            for arm in ARMS:
                cell = {"dataset": ds, "seed": seed, "arm": arm,
                        "status": "pending", "reason": "",
                        "run_dir": None, "verification_report": None,
                        "hashes": {}, "terminal_acc": None,
                        "failed_attempts": []}
                if failed_root.is_dir():
                    for d in sorted(failed_root.iterdir()):
                        if not (d.is_dir()
                                and d.name.startswith(f"raexp_{ds}_{arm}_")
                                and f"_seed{seed}_" in d.name):
                            continue
                        entry = {"path": str(d)}
                        reason_file = d.with_suffix(".reason.txt")
                        log_file = d.with_suffix(".log")
                        if reason_file.is_file():
                            entry["reason"] = reason_file.read_text(
                                encoding="utf-8").strip()
                        if log_file.is_file():
                            entry["log"] = str(log_file)
                        cell["failed_attempts"].append(entry)
                matches = []
                if RUNS.is_dir():
                    matches = sorted(
                        d for d in RUNS.iterdir() if d.is_dir()
                        and d.name.startswith(f"raexp_{ds}_{arm}_")
                        and f"_seed{seed}_" in d.name)

                if len(matches) > 1:
                    cell["status"] = "duplicate_error"
                    cell["reason"] = ("multiple run directories: "
                                      + ";".join(m.name for m in matches))
                    cell["run_dir"] = ";".join(str(m) for m in matches)
                    problems.append(cell["reason"])
                elif matches:
                    rd = matches[0]
                    cell["run_dir"] = str(rd)
                    missing = [fn for fn in EVIDENCE
                               if not (rd / fn).is_file()]
                    if missing:
                        cell["status"] = "partial"
                        cell["reason"] = ("missing evidence files: "
                                          + ",".join(missing))
                    else:
                        report_out = (VERIF_DIR
                                      / f"{ds}_{arm}_seed{seed}.json")
                        ok, tail = verify(rd, ds, seed, arm, report_out)
                        cell["verification_report"] = str(report_out)
                        cell["hashes"] = {fn: sha256(rd / fn)
                                          for fn in EVIDENCE}
                        if arm == "ra_probeon":
                            cell["hashes"]["counterfactual_probe.csv"] = sha256(rd / "counterfactual_probe.csv")
                        if ok:
                            cell["status"] = "complete"
                            try:
                                rows = list(csv.DictReader(
                                    open(rd / "metrics.csv",
                                         encoding="utf-8")))
                                acc = float(rows[-1]["test_acc"])
                                if not math.isfinite(acc):
                                    raise ValueError("non-finite terminal "
                                                     "accuracy")
                                cell["terminal_acc"] = acc
                            except Exception as e:
                                cell["status"] = "invalid"
                                cell["reason"] = (f"terminal accuracy "
                                                  f"unreadable after "
                                                  f"verification: {e}")
                                problems.append(cell["reason"])
                        else:
                            # never parse endpoints after verifier failure:
                            # leave terminal_acc null, keep the reason
                            cell["status"] = "invalid"
                            cell["reason"] = f"verifier failed: {tail}"
                            problems.append(
                                f"invalid run {ds} {arm} seed {seed}: "
                                f"{tail}")
                cells.append(cell)

    counts = {}
    for c in cells:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    payload = {"experiment": "four_arm_20261009",
               "grid": "3 datasets x 5 seeds x 4 arms = 60 cells",
               "completion_rule": ("a cell is 'complete' only if the full "
                                   "verifier passes for that exact run "
                                   "and arm"),
               "counts": counts, "problems": problems, "cells": cells}
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    out = EXP_ROOT / "RUN_MANIFEST.json"
    out.write_text(json.dumps(payload, indent=1,
                                allow_nan=False),
                        encoding="utf-8")
    print(f"wrote {out}: {counts}" + (f" | problems: {len(problems)}"
                                     if problems else ""))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
