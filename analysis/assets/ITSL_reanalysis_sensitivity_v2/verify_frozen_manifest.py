#!/usr/bin/env python
"""Standalone verification of the frozen manifest's 160 selected sources
against the LOCALLY RECOVERED server metadata.

Does NOT assume original training runs were executed locally. The local
copies come from server recovery archives; their authoritative location is
the server path recorded per run. Any missing or hash-mismatching source
is reported with its exact server path for targeted copying. No run is
ever substituted.

Checks per selected source:
  - run directory exists among recovered copies
  - metrics.csv, config_used.json, summary.json exist and SHA-256-match
    the frozen manifest
  - metrics.csv has exactly one finite round-100 test_acc
Also verifies the frozen manifest's hash against FREEZE_RECORD.json.
"""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOTS = [Path("run_metadata/runs"), Path("confirmatory_recovery/runs")]
FREEZE = Path("reanalysis_20261008/frozen_inputs")
MANIFEST = FREEZE / "proposed_source_manifest_FROZEN_20261008.csv"
SERVER_RUNS = "<REDACTED_HOME>/disk1/infoopt_dl_server/runs"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = {"manifest": str(MANIFEST), "checks": [], "problems": []}

    # frozen manifest integrity
    fr = json.load(open(FREEZE / "FREEZE_RECORD.json", encoding="utf-8"))
    actual = sha256(MANIFEST)
    expected = fr["sha256"].get(MANIFEST.name)
    out["freeze_integrity"] = {
        "manifest_sha256_matches_freeze_record": actual == expected,
        "frozen_manifest_sha256": actual,
    }
    if actual != expected:
        out["problems"].append("FATAL: frozen manifest hash does not match "
                               "FREEZE_RECORD.json")

    # index recovered run dirs by name
    runs_by_name = {}
    for root in ROOTS:
        for d in root.rglob("*"):
            if d.is_dir() and (d / "metrics.csv").is_file():
                runs_by_name.setdefault(d.name, []).append(d)

    man = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    proposed = [r for r in man if r["run_name"]]
    n_verified = 0
    for r in proposed:
        cell = f"{r['dataset']}|{r['manuscript_row']}|{r['seed']}"
        cands = runs_by_name.get(r["run_name"], [])
        if not cands:
            out["problems"].append({
                "cell": cell, "run_name": r["run_name"],
                "problem": "run directory NOT in any local recovery copy",
                "server_path": f"{SERVER_RUNS}/{r['run_name']}/",
                "server_files_needed": [
                    f"{SERVER_RUNS}/{r['run_name']}/metrics.csv",
                    f"{SERVER_RUNS}/{r['run_name']}/config_used.json",
                    f"{SERVER_RUNS}/{r['run_name']}/summary.json"],
            })
            continue
        # hash-verify; collapse content-identical copies
        matched = []
        for d in cands:
            ok = True
            for fn, col in (("metrics.csv", "sha256_metrics.csv"),
                            ("config_used.json", "sha256_config_used.json"),
                            ("summary.json", "sha256_summary.json")):
                p = d / fn
                if not p.is_file() or sha256(p) != r[col]:
                    ok = False
                    break
            if ok:
                matched.append(d)
        if not matched:
            out["problems"].append({
                "cell": cell, "run_name": r["run_name"],
                "problem": "hash mismatch in all local copies",
                "local_locations": [str(d) for d in cands],
                "server_path": f"{SERVER_RUNS}/{r['run_name']}/",
                "server_files_needed": [
                    f"{SERVER_RUNS}/{r['run_name']}/metrics.csv",
                    f"{SERVER_RUNS}/{r['run_name']}/config_used.json",
                    f"{SERVER_RUNS}/{r['run_name']}/summary.json"],
            })
            continue
        # endpoint check on the verified copy
        rows = list(csv.DictReader(open(matched[0] / "metrics.csv",
                                        encoding="utf-8")))
        r100 = [x for x in rows if x["round"] == "100"]
        if len(r100) != 1 or not math.isfinite(float(r100[0]["test_acc"])):
            out["problems"].append({
                "cell": cell, "run_name": r["run_name"],
                "problem": f"invalid endpoint: {len(r100)} round-100 rows",
                "server_path": f"{SERVER_RUNS}/{r['run_name']}/"})
            continue
        n_verified += 1
        out["checks"].append({"cell": cell, "run_name": r["run_name"],
                              "verified": True,
                              "identical_copies": len(matched)})

    out["n_proposed"] = len(proposed)
    out["n_verified"] = n_verified
    out["n_problems"] = len(out["problems"])
    out["verdict"] = ("ALL SOURCES VERIFIED" if not out["problems"]
                      else "INCOMPLETE — see problems")
    rp = Path("reanalysis_20261008/frozen_manifest_verification.json")
    json.dump(out, open(rp, "w", encoding="utf-8"), indent=1)

    print(f"frozen manifest integrity: "
          f"{out['freeze_integrity']['manifest_sha256_matches_freeze_record']}")
    print(f"proposed sources: {len(proposed)}; verified: {n_verified}; "
          f"problems: {len(out['problems'])}")
    for p in out["problems"]:
        print(f"  MISSING/MISMATCH: {p['cell']} {p['run_name']}")
        print(f"    server path: {p['server_path']}")
    print(f"report: {rp}")
    sys.exit(0 if not out["problems"] else 1)


if __name__ == "__main__":
    main()
