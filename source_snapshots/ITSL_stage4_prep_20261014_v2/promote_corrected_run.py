#!/usr/bin/env python3
"""Promote a verified corrected (derived) run into runs/ and archive the
original completed run outside active-run discovery. Stage-3 version:
every name is alpha-namespaced.

Hardening relative to the repair3 version:
  - Companion files are addressed by FULL directory name plus suffix
    (Path(str(d) + ".log")), never Path.with_suffix() — alpha-bearing
    names like stage3_..._alpha0.1_... contain a '.' that with_suffix
    would mis-parse.
  - ALL destination paths (archive dir, reason file, log targets,
    promotion target, promotion record) are checked BEFORE any file is
    moved; existing destinations abort the whole operation.
  - An explicit promotion record (source -> derived -> active) is
    written under EXP_ROOT/promotions/.
  - If the post-promotion resume check fails, the moves are rolled back
    (derived restored to its original path, original restored to runs/).

Order of operations:
  1. The derived directory must contain the seven evidence files plus
     correction_record.json, and must PASS the full verifier.
  2. Exactly one active original run directory for the cell must exist
     in EXP_ROOT/runs (zero or multiple aborts for human resolution).
  3. Pre-flight all destination paths (nothing may exist).
  4. Move the original (and sibling .log/.out) to failed_attempts/ with
     a .reason.txt. Nothing is deleted.
  5. Move the derived directory into runs/ as
     stage3_<dataset>_<arm>_seed<seed>_alpha<alpha>_repairderived.
  6. Write the promotion record; resume_check.py must return exactly
     "skip_complete" (rollback otherwise).

The execution manifest is NOT touched here — rebuild it normally with
build_run_manifest.py afterwards.

Usage:
  python promote_corrected_run.py --dataset cifar100 --alpha 0.1 \
      --seed 100 --arm fedavg --derived-dir <path> [--exp-root $EXP_ROOT]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = ("config_used.json", "metrics.csv", "summary.json",
            "final_model.pt", "partition_used.json", "diagnostics.jsonl",
            "init_state_hash.json")


def _abort(msg):
    raise SystemExit(f"PROMOTE ABORT: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--derived-dir", required=True)
    ap.add_argument("--exp-root",
                    default=os.environ.get("EXP_ROOT"))
    args = ap.parse_args()
    if not args.exp_root:
        _abort("--exp-root (or EXP_ROOT) is required")
    alpha = str(float(args.alpha))
    exp = Path(args.exp_root)
    runs = exp / "runs"
    shared = Path(os.environ.get("SHARED_DIR", str(exp / "shared")))
    derived = Path(args.derived_dir)
    if not derived.is_dir():
        _abort(f"derived directory missing: {derived}")
    for fn in EVIDENCE:
        if not (derived / fn).is_file():
            _abort(f"derived directory lacks evidence file: {fn}")
    if not (derived / "correction_record.json").is_file():
        _abort("derived directory lacks correction_record.json — "
               "this tool only promotes outputs of "
               "repair_diagnostics_export.py")

    # 1. derived copy must pass the full verifier
    r = subprocess.run(
        [sys.executable, str(HERE / "verify_run.py"),
         "--run-dir", str(derived), "--dataset", args.dataset,
         "--seed", str(args.seed), "--alpha", alpha,
         "--backbone", args.backbone, "--arm", args.arm,
         "--schedules-dir", str(shared)],
        capture_output=True, text=True)
    if r.returncode != 0:
        _abort(f"derived copy fails verification: "
               f"{(r.stdout + r.stderr).strip().splitlines()[-1][:300]}")

    # 2. exactly one active original run
    prefix = f"stage4_{args.backbone}_{args.dataset}_{args.arm}_"
    infix = f"_seed{args.seed}_alpha{alpha}_"
    matches = sorted(
        d for d in runs.iterdir() if d.is_dir()
        and d.name.startswith(prefix)
        and infix in d.name) if runs.is_dir() else []
    if len(matches) == 0:
        _abort("no active original run found for this cell")
    if len(matches) > 1:
        _abort(f"multiple active run directories: "
               f"{[m.name for m in matches]} — resolve manually")
    orig = matches[0]

    # 3. pre-flight ALL destination paths before moving anything
    failed_root = exp / "failed_attempts"
    arch = failed_root / orig.name
    reason_dst = Path(str(arch) + ".reason.txt")
    side_pairs = []
    for ext in (".log", ".out"):
        src_side = Path(str(orig) + ext)
        if src_side.is_file():
            side_pairs.append((src_side, Path(str(arch) + ext)))
    target = runs / (f"stage4_{args.backbone}_{args.dataset}"
                     f"_{args.arm}_seed{args.seed}_alpha{alpha}"
                     "_repairderived")
    prom_rec = exp / "promotions" / f"{target.name}.json"
    for dst in [arch, reason_dst, target, prom_rec] + [d for _, d
                                                       in side_pairs]:
        if dst.exists():
            _abort(f"destination already exists, refusing to overwrite: "
                   f"{dst}")

    # 4-5. move original aside, promote derived (with rollback)
    failed_root.mkdir(parents=True, exist_ok=True)
    prom_rec.parent.mkdir(parents=True, exist_ok=True)
    moved = []
    try:
        shutil.move(str(orig), str(arch))
        moved.append((arch, orig))
        for src_side, dst_side in side_pairs:
            shutil.move(str(src_side), str(dst_side))
            moved.append((dst_side, src_side))
        reason_dst.write_text(
            f"superseded by verified corrected copy (diagnostics-schema "
            f"correction); original evidence preserved here; not an "
            f"accuracy-based retry\n", encoding="utf-8")
        shutil.move(str(derived), str(target))
        moved.append((target, derived))
    except Exception as e:
        for dst, src in reversed(moved):
            try:
                shutil.move(str(dst), str(src))
            except Exception as e2:
                print(f"ROLLBACK FAILURE: {dst} -> {src}: {e2}",
                      file=sys.stderr)
        _abort(f"move failed, rolled back: {e}")

    # 6. promotion record + resume gate
    record = {"tool": "promote_corrected_run.py",
              "source_original": str(arch),
              "derived_from": str(derived),
              "active": str(target),
              "dataset": args.dataset, "alpha": alpha,
              "backbone": args.backbone,
              "seed": args.seed, "arm": args.arm,
              "correction_record": str(target / "correction_record.json")}
    rc = subprocess.run(
        [sys.executable, str(HERE / "resume_check.py"),
         "--dataset", args.dataset, "--seed", str(args.seed),
         "--alpha", alpha, "--backbone", args.backbone,
         "--arm", args.arm, "--runs-dir", str(runs),
         "--schedules-dir", str(shared)],
        capture_output=True, text=True)
    decision = rc.stdout.strip()
    record["resume_decision"] = decision
    if decision != "skip_complete":
        for dst, src in reversed(moved):
            try:
                shutil.move(str(dst), str(src))
            except Exception as e2:
                print(f"ROLLBACK FAILURE: {dst} -> {src}: {e2}",
                      file=sys.stderr)
        reason_dst.unlink(missing_ok=True)
        _abort(f"resume check returned {decision!r} after promotion "
               f"(expected 'skip_complete'); rolled back. stderr: "
               f"{rc.stderr.strip()[-300:]}")
    prom_rec.write_text(json.dumps(record, indent=1, allow_nan=False),
                        encoding="utf-8")
    print(f"PROMOTED {target.name}; original archived at {arch}; "
          f"promotion record {prom_rec}; resume decision: skip_complete")
    print("Next: rebuild the manifest normally with "
          "build_run_manifest.py")


if __name__ == "__main__":
    main()
