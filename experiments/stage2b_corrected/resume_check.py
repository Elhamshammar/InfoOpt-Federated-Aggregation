#!/usr/bin/env python3
"""Resume decision helper for the four-arm launcher.

For one cell (dataset, seed, arm):
  - 0 run directories        -> "run"
  - exactly 1 run directory  -> run the FULL verifier;
      verification passes     -> "skip_complete"
      verification fails      -> "abort_invalid"
  - more than 1 directory    -> "abort_duplicate" (no latest/head
    selection; a human must resolve duplicates)

Prints exactly one of: run | skip_complete | abort_invalid | abort_duplicate
(exit 0), or exits nonzero on verifier invocation errors. Verification
reports are preserved under EXP_ROOT/verification/.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--schedules-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    matches = sorted(
        d for d in runs.iterdir() if d.is_dir()
        and d.name.startswith(f"fourarm_{args.dataset}_{args.arm}_")
        and f"_seed{args.seed}_" in d.name) if runs.is_dir() else []

    # archived failed attempts live OUTSIDE runs/ under failed_attempts/
    # and are never counted as active candidates; they are surfaced for
    # the record only.
    failed_dir = runs.parent / "failed_attempts"
    archived = sorted(
        d for d in failed_dir.iterdir() if d.is_dir()
        and d.name.startswith(f"fourarm_{args.dataset}_{args.arm}_")
        and f"_seed{args.seed}_" in d.name) \
        if failed_dir.is_dir() else []
    for a in archived:
        print(f"ARCHIVED ATTEMPT (not an active candidate): {a}",
              file=sys.stderr)

    if len(matches) == 0:
        print("run")
        return
    if len(matches) > 1:
        print("abort_duplicate")
        for m in matches:
            print(f"DUPLICATE: {m}", file=sys.stderr)
        return

    rd = matches[0]
    report_out = Path(runs).parent / "verification" / \
        f"{args.dataset}_{args.arm}_seed{args.seed}.json"
    r = subprocess.run(
        [args.python, str(HERE / "verify_run.py"),
         "--run-dir", str(rd), "--dataset", args.dataset,
         "--seed", str(args.seed), "--arm", args.arm,
         "--schedules-dir", args.schedules_dir,
         "--report-out", str(report_out)],
        capture_output=True, text=True)
    # verifier diagnostics go to stderr; stdout carries exactly one
    # decision line, nothing else
    sys.stderr.write(r.stdout)
    sys.stderr.write(r.stderr)
    print("skip_complete" if r.returncode == 0 else "abort_invalid")


if __name__ == "__main__":
    main()
