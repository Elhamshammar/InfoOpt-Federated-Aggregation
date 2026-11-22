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
                         "dataset-alpha-seed combinations (use AFTER "
                         "the verified Stage-1 copy)")
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
            datasets = subset if subset else ["cifar100", "eurosat",
                                              "pathmnist"]
            alphas = ("0.3",)
            seeds = (100, 101, 102, 103, 104)
            expected_keys = {(d, a, s) for d in datasets for a in alphas
                             for s in seeds}
            expected_files = set()
            for d, a, s in expected_keys:
                expected_files.add(f"partition_{d}_alpha{a}_seed{s}.json")
                expected_files.add(
                    f"cohort_schedule_{d}_alpha{a}_seed{s}.json")
            try:
                entries = json.loads(sman.read_text(
                    encoding="utf-8"))["files"]
            except Exception as e:
                problems.append(f"unreadable shared manifest: {e}")
                entries = []
            # unique keys, exact set equality with the expected grid
            mkeys = [(e.get("dataset"), str(e.get("alpha")),
                      e.get("seed")) for e in entries]
            dupes = sorted({k for k in mkeys if mkeys.count(k) > 1})
            if dupes:
                problems.append(f"duplicate manifest keys: {dupes[:3]}")
            mkey_set = {(d, str(a), s) for d, a, s in mkeys}
            for k in sorted(expected_keys - mkey_set):
                problems.append(f"manifest missing combination: {k}")
            for k in sorted(mkey_set - expected_keys):
                problems.append(f"manifest has unexpected combination: {k}")
            # exact filenames + hash verification for every entry
            for e in entries:
                d, a, s = e.get("dataset"), str(e.get("alpha")), \
                    e.get("seed")
                for kind, tmpl in (("partition",
                                    f"partition_{d}_alpha{a}_seed{s}.json"),
                                   ("schedule",
                                    f"cohort_schedule_{d}_alpha{a}_seed{s}"
                                    f".json")):
                    rec = e.get(kind, {})
                    if Path(rec.get("path", "")).name != tmpl:
                        problems.append(f"manifest {kind} filename mismatch "
                                        f"for {d}/{a}/{s}: "
                                        f"{rec.get('path')}")
                        continue
                    f = sman_dir / tmpl
                    if not f.is_file():
                        problems.append(f"missing shared file: {tmpl}")
                    elif sha256(f) != rec.get("sha256"):
                        problems.append(f"shared file hash mismatch "
                                        f"(substituted or changed): {tmpl}")
            # on-disk set must be exactly the expected filenames — extras
            # rejected even when counts would match
            actual = {p.name for p in sman_dir.glob("partition_*_seed*.json")}
            actual |= {p.name for p in
                       sman_dir.glob("cohort_schedule_*_seed*.json")}
            for name in sorted(actual - expected_files):
                problems.append(f"unexpected file in shared dir: {name}")
            for name in sorted(expected_files - actual):
                problems.append(f"expected file absent on disk: {name}")

    if problems:
        for p in problems:
            print("FROZEN-INTEGRITY FAIL:", p)
        sys.exit(1)
    print("frozen integrity OK")


if __name__ == "__main__":
    main()
