#!/usr/bin/env python3
"""Frozen-integrity verification for the four-arm experiment.

Verifies, before launch and on every resume:
  1. every file listed in snapshot/SNAPSHOT_HASHES.sha256 matches its hash;
  2. all 12 effective configs exist under EXP_ROOT/configs_effective and
     match RESOLUTION.json hashes (if phase 0a has run);
  3. all frozen shared partition/schedule files match SHARED_MANIFEST.json
     hashes (if phase 0b has run).

Any mismatch aborts. The preparation manifest (RUN_MANIFEST.json in the
package) is never touched; execution manifests live under EXP_ROOT.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = Path(os.environ.get("EXP_ROOT",
                               str(HERE)))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-complete-schedules", action="store_true",
                    help="require SHARED_MANIFEST.json and all 15 "
                         "dataset-seed pairs (use AFTER generation)")
    cli = ap.parse_args()

    problems = []
    sums = HERE / "SNAPSHOT_HASHES.sha256"
    for ln in sums.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        h, name = ln.split(None, 1)
        name = name.lstrip("*")
        # entries are PACKAGE-ROOT-relative (e.g. "snapshot/methods/x.py")
        p = HERE / name
        if not p.is_file():
            problems.append(f"missing snapshot file: {name}")
        elif sha256(p) != h:
            problems.append(f"snapshot hash mismatch: {name}")

    res = EXP_ROOT / "configs_effective" / "RESOLUTION.json"
    if res.is_file():
        for rec in json.loads(res.read_text(encoding="utf-8")):
            p = EXP_ROOT / "configs_effective" / rec["config"]
            if not p.is_file():
                problems.append(f"missing effective config: {p.name}")
            elif sha256(p) != rec["sha256"]:
                problems.append(f"effective config mismatch: {p.name}")

    sman_dir = Path(os.environ.get(
        "SHARED_DIR", str(EXP_ROOT / "shared")))
    sman = sman_dir / "SHARED_MANIFEST.json"
    if sman.is_file():
        for f in json.loads(sman.read_text(encoding="utf-8"))["files"]:
            for side in ("partition", "schedule"):
                p = Path(f[side]["path"])
                if not p.is_file():
                    problems.append(f"missing shared file: {p.name}")
                elif sha256(p) != f[side]["sha256"]:
                    problems.append(f"shared file mismatch: {p.name}")
    if cli.require_complete_schedules:
        if not sman.is_file():
            problems.append(f"shared manifest missing: {sman}")
        else:
            subset = os.environ.get("DATASETS", "").split()
            expected = 5 * (len(subset) if subset else 3)
            parts = list(sman_dir.glob("partition_*_seed*.json"))
            scheds = list(sman_dir.glob("cohort_schedule_*_seed*.json"))
            if len(parts) != expected or len(scheds) != expected:
                problems.append(
                    f"shared schedule coverage: {len(parts)} partition + "
                    f"{len(scheds)} cohort files, expected "
                    f"{expected} + {expected} "
                    f"({len(subset) if subset else 3} datasets x 5 seeds)")

    if problems:
        for p in problems:
            print("FROZEN-INTEGRITY FAIL:", p)
        sys.exit(1)
    print("frozen integrity OK")


if __name__ == "__main__":
    main()
