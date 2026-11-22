#!/usr/bin/env python3
"""Build/refresh the EXECUTION run manifest for the Stage-3
heterogeneity experiment (3 datasets x 2 alphas x 5 seeds x 6 arms
= 180 cells).

A cell is marked "complete" ONLY if the full verifier
(verify_run.py --arm --alpha) passes for that exact run directory.
Otherwise the cell gets an explicit status and reason:

  pending          — no run directory
  partial          — run directory exists but evidence files are missing
  invalid          — evidence present but the verifier FAILED (reason kept)
  duplicate_error  — more than one run directory for the cell
  complete         — verifier passed (report path + hashes recorded,
                     including partition_used.json and init_state_hash.json)

Writes EXP_ROOT/RUN_MANIFEST.json. The preparation template in the
package is never modified. Verification reports are preserved under
EXP_ROOT/verification/.

Archived attempts live under EXP_ROOT/failed_attempts/. Companion files
(reason, log) are addressed by FULL directory name plus suffix
(Path(str(d) + ".reason.txt")) — never Path.with_suffix(), which
mis-parses timestamp- and alpha-bearing names (the '.' in alpha0.1 is
treated as an extension separator).
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
BACKBONE = "resnet18"
ALPHAS = ("0.3",)
SEEDS = (100, 101, 102, 103, 104)
ARMS = ("fedavg", "uniform", "infoopt", "shuffled_infoopt",
        "damping_050", "damping_075")
EVIDENCE = ("config_used.json", "metrics.csv", "summary.json",
            "final_model.pt", "partition_used.json",
            "diagnostics.jsonl", "init_state_hash.json")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(rd, ds, alpha, seed, arm, report_out):
    sched_dir = os.environ.get("SHARED_DIR", str(EXP_ROOT / "shared"))
    r = subprocess.run(
        [sys.executable, str(HERE / "verify_run.py"),
         "--run-dir", str(rd), "--dataset", ds, "--seed", str(seed),
         "--alpha", alpha, "--backbone", BACKBONE, "--arm", arm,
         "--schedules-dir", sched_dir,
         "--report-out", str(report_out)],
        capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, (tail[-1] if tail else "")


def main():
    VERIF_DIR.mkdir(parents=True, exist_ok=True)
    cells, problems = [], []
    failed_root = EXP_ROOT / "failed_attempts"
    for ds in DATASETS:
        for alpha in ALPHAS:
            for seed in SEEDS:
                for arm in ARMS:
                    cell = {"dataset": ds, "alpha": alpha, "seed": seed,
                            "arm": arm, "backbone": BACKBONE,
                            "status": "pending", "reason": "",
                            "run_dir": None, "verification_report": None,
                            "hashes": {}, "terminal_acc": None,
                            "failed_attempts": []}
                    prefix = f"stage4_{BACKBONE}_{ds}_{arm}_"
                    infix = f"_seed{seed}_alpha{alpha}_"
                    if failed_root.is_dir():
                        for d in sorted(failed_root.iterdir()):
                            if not (d.is_dir()
                                    and d.name.startswith(prefix)
                                    and infix in d.name):
                                continue
                            entry = {"path": str(d)}
                            # FULL name + suffix; never with_suffix()
                            reason_file = Path(str(d) + ".reason.txt")
                            log_file = Path(str(d) + ".log")
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
                            and d.name.startswith(prefix)
                            and infix in d.name)

                    if len(matches) > 1:
                        cell["status"] = "duplicate_error"
                        cell["reason"] = ("multiple run directories: "
                                          + ";".join(m.name
                                                     for m in matches))
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
                                          / f"{BACKBONE}_{ds}_{arm}"
                                            f"_alpha{alpha}"
                                            f"_seed{seed}.json")
                            ok, tail = verify(rd, ds, alpha, seed, arm,
                                              report_out)
                            cell["verification_report"] = str(report_out)
                            cell["hashes"] = {fn: sha256(rd / fn)
                                              for fn in EVIDENCE}
                            if ok:
                                cell["status"] = "complete"
                                try:
                                    rows = list(csv.DictReader(
                                        open(rd / "metrics.csv",
                                             encoding="utf-8")))
                                    r100 = [x for x in rows
                                            if int(x["round"]) == 100]
                                    if len(r100) != 1:
                                        raise ValueError(
                                            "no unique round-100 row")
                                    acc = float(r100[0]["test_acc"])
                                    if not math.isfinite(acc):
                                        raise ValueError(
                                            "non-finite terminal accuracy")
                                    cell["terminal_acc"] = acc
                                except Exception as e:
                                    cell["status"] = "invalid"
                                    cell["reason"] = (f"terminal accuracy "
                                                      f"unreadable after "
                                                      f"verification: {e}")
                                    problems.append(cell["reason"])
                            else:
                                # never parse endpoints after verifier
                                # failure: terminal_acc stays null
                                cell["status"] = "invalid"
                                cell["reason"] = f"verifier failed: {tail}"
                                problems.append(
                                    f"invalid run {ds} {arm} "
                                    f"alpha{alpha} seed {seed}: {tail}")
                    cells.append(cell)

    counts = {}
    for c in cells:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    payload = {"experiment": "stage4_backbone_20261014",
               "grid": ("1 backbone (resnet18) x 3 datasets x alpha=0.3 "
                        "x 5 seeds x 6 arms = 90 cells"),
               "completion_rule": ("a cell is 'complete' only if the full "
                                   "verifier passes for that exact run, "
                                   "arm and alpha"),
               "counts": counts, "problems": problems, "cells": cells}
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    out = EXP_ROOT / "RUN_MANIFEST.json"
    out.write_text(json.dumps(payload, indent=1, allow_nan=False),
                   encoding="utf-8")
    print(f"wrote {out}: {counts}" + (f" | problems: {len(problems)}"
                                     if problems else ""))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
