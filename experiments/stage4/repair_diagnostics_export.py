#!/usr/bin/env python3
"""Deterministic diagnostics-schema correction for runs produced with the
pre-repair2 snapshot on non-InfoOpt arms (repair1-era defect: the
baseline info dict leaked non-null scores/nu/epsilon/mixture_js into
diagnostics.jsonl).

This correction touches ONLY diagnostics.jsonl and ONLY the
InfoOpt-specific fields (score/constraint/solver/shuffle). It:
  1. strict-parses every row of the original diagnostics.jsonl;
  2. verifies the arm is a non-InfoOpt arm (correction is not defined
     for InfoOpt arms — those rows must be produced by training);
  3. nulls exactly the InfoOpt-specific fields, recording every changed
     field per round;
  4. copies ALL other evidence files byte-identically into a NEW derived
     directory (refuses to overwrite; originals untouched);
  5. records original and corrected SHA-256 for every file;
  6. runs the full verify_run.py against the derived directory and
     reports its verdict.

Usage:
  python repair_diagnostics_export.py --run-dir <original run> \
      --arm fedavg --dataset cifar100 --seed 100 \
      --schedules-dir <shared> --out-dir <new derived dir>
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = ("config_used.json", "metrics.csv", "summary.json",
            "final_model.pt", "partition_used.json", "diagnostics.jsonl",
            "init_state_hash.json")
NON_INFOOPT_ARMS = ("fedavg", "uniform", "damping_050", "damping_075")
CORRECTED_FIELDS = ("scores", "temperature_used", "mixture_js",
                    "epsilon", "constraint_feasible", "constraint_active",
                    "nu", "solver_calls", "solver_final_status",
                    "fallback_used", "fallback_reason",
                    "shuffle_original_scores", "shuffle_solver_scores",
                    "shuffle_permutation", "shuffle_seed_int")


def _strict_json(text):
    def _reject(x):
        raise ValueError(f"non-strict JSON constant {x!r}")
    return json.loads(text, parse_constant=_reject)


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", required=True, choices=NON_INFOOPT_ARMS)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--schedules-dir", required=True)
    args = ap.parse_args()
    alpha = str(float(args.alpha))

    rd = Path(args.run_dir)
    od = Path(args.out_dir)
    if od.exists():
        raise SystemExit(f"refusing to overwrite existing out dir: {od}")
    for fn in EVIDENCE:
        if not (rd / fn).is_file():
            raise SystemExit(f"missing evidence file: {fn}")

    rows = [_strict_json(ln) for ln in
            (rd / "diagnostics.jsonl").read_text(
                encoding="utf-8").splitlines() if ln.strip()]
    changes = []
    corrected = []
    for row in rows:
        row = dict(row)
        changed = {}
        for f in CORRECTED_FIELDS:
            if row.get(f) is not None:
                changed[f] = row[f]
                row[f] = None
        if changed:
            changes.append({"round": row.get("round"),
                            "nulled": sorted(changed.keys())})
        corrected.append(row)

    od.mkdir(parents=True)
    record = {"tool": "repair_diagnostics_export.py",
              "source_run_dir": str(rd), "derived_dir": str(od),
              "arm": args.arm, "dataset": args.dataset,
              "seed": args.seed, "alpha": alpha,
              "backbone": args.backbone,
              "correction": ("null InfoOpt-specific diagnostics fields on "
                             "a non-InfoOpt arm (repair2 schema)"),
              "fields": list(CORRECTED_FIELDS),
              "rows_changed": len(changes),
              "changes": changes,
              "hashes": {}}
    for fn in EVIDENCE:
        record["hashes"][fn] = {"original_sha256": _sha256(rd / fn)}
        if fn == "diagnostics.jsonl":
            blob = "\n".join(json.dumps(r, allow_nan=False)
                             for r in corrected) + "\n"
            (od / fn).write_text(blob, encoding="utf-8")
        else:
            shutil.copyfile(rd / fn, od / fn)
        record["hashes"][fn]["corrected_sha256"] = _sha256(od / fn)
        if fn != "diagnostics.jsonl":
            assert record["hashes"][fn]["original_sha256"] == \
                record["hashes"][fn]["corrected_sha256"]

    # full verification of the derived evidence
    r = subprocess.run(
        [sys.executable, str(HERE / "verify_run.py"),
         "--run-dir", str(od), "--dataset", args.dataset,
         "--seed", str(args.seed), "--alpha", alpha,
         "--backbone", args.backbone, "--arm", args.arm,
         "--schedules-dir", args.schedules_dir,
         "--report-out", str(od / "verification_report_repair2.json")],
        capture_output=True, text=True)
    record["derived_verification"] = {
        "returncode": r.returncode,
        "tail": (r.stdout + r.stderr).strip().splitlines()[-1:]
    }
    (od / "correction_record.json").write_text(
        json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
    print(json.dumps({"rows_changed": len(changes),
                      "verifier_returncode": r.returncode,
                      "record": str(od / "correction_record.json")}))
    if r.returncode != 0:
        raise SystemExit("derived evidence FAILED verification; the run "
                         "cannot be retained by this correction alone")
    print("derived evidence verified; correction is retention-eligible")


if __name__ == "__main__":
    main()
