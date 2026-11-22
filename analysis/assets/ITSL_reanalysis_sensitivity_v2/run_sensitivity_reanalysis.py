#!/usr/bin/env python
"""Sensitivity/RA reanalysis from the FROZEN, verified source inventory.

- Every selected source is hash-verified (metrics.csv, config_used.json,
  summary.json) against the frozen sensitivity inventory and endpoint-
  checked (exactly one finite round-100 record) BEFORE any calculation.
- Any missing/mismatched/invalid source ABORTS with a clear reason.
  There is NO substitution and NO fallback to primary values.
- Cells without a distinct latest repeat are labelled
  "same run (single source)" — this is a label, not a fallback.
- RA summaries use only verified inventory records; per-stratum, per-seed
  earliest selection; run identities and per-seed values exported; no
  combined five-seed estimate.
"""
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOTS = [Path("run_metadata/runs"), Path("confirmatory_recovery/runs")]
SEEDS = (100, 101, 102, 103, 104)
DATASETS = ("cifar100", "eurosat", "pathmnist")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_t(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    m = d.mean()
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n)
    tval = m / se if se > 0 else 0.0
    tcrit = stats.t.ppf(0.975, n - 1)
    return m, sd, (m - tcrit * se, m + tcrit * se), \
        (2 * stats.t.sf(abs(tval), n - 1) if se > 0 else 1.0), n


def main(inventory_csv="sensitivity_source_inventory.csv",
         primary_manifest="reanalysis_20261008/frozen_inputs/"
                          "proposed_source_manifest_FROZEN_20261008.csv",
         out_dir=None):
    if out_dir is None:
        out_dir = Path("reanalysis_sensitivity_" + time.strftime("%Y%m%d"))
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output dir not empty: {out_dir}")
    out_dir.mkdir(parents=True)

    inv = list(csv.DictReader(open(inventory_csv, encoding="utf-8")))

    runs_by_name = {}
    for root in ROOTS:
        for d in root.rglob("*"):
            if d.is_dir() and (d / "metrics.csv").is_file():
                runs_by_name.setdefault(d.name, []).append(d)

    # --- verify EVERY selected source before calculating anything ---
    acc = {}      # id(rec) -> round-100 acc %
    vlog = []
    for i, r in enumerate(inv):
        where = f"{r['role']}:{r['dataset']}|{r['manuscript_row']}|{r['seed']}"
        cands = runs_by_name.get(r["run_name"], [])
        if not cands:
            raise SystemExit(f"FATAL: source not found locally: {where} "
                             f"run={r['run_name']} — no substitution; copy "
                             f"from server and rerun")
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
            raise SystemExit(f"FATAL: hash mismatch for {where} "
                             f"run={r['run_name']} — no substitution")
        rows = list(csv.DictReader(open(matched[0] / "metrics.csv",
                                        encoding="utf-8")))
        r100 = [x for x in rows if x["round"] == "100"]
        if len(r100) != 1 or not math.isfinite(float(r100[0]["test_acc"])):
            raise SystemExit(f"FATAL: invalid endpoint for {where} "
                             f"run={r['run_name']}")
        acc[i] = float(r100[0]["test_acc"]) * 100
        vlog.append({"role": r["role"], "cell": where,
                     "run_name": r["run_name"], "verified": True,
                     "identical_copies": len(matched)})
    with open(out_dir / "verification_log.json", "w",
              encoding="utf-8") as f:
        json.dump({"verified_sources": len(vlog), "log": vlog}, f,
                  indent=1)
    print(f"verified {len(vlog)}/{len(inv)} sensitivity/RA sources")

    # --- latest-repeat sensitivity (no fallback) ---
    # Primary per-seed values come from the FROZEN primary per-seed output
    # (computed from hash-verified sources), NOT from rereading run dirs.
    # Its integrity is checked against the primary package's SHA256SUMS.
    prim = list(csv.DictReader(open(primary_manifest, encoding="utf-8")))
    prim_status = {}
    prim_run = {}
    for r in prim:
        if r["run_name"]:
            key = (r["dataset"], r["manuscript_row"], int(r["seed"]))
            prim_status[key] = r["status"]
            prim_run[key] = r["run_name"]

    prim_psv_path = Path("reanalysis_20261008/outputs/per_seed_round100.csv")
    sums_path = Path("reanalysis_20261008/SHA256SUMS.txt")
    if not prim_psv_path.is_file() or not sums_path.is_file():
        raise SystemExit("FATAL: frozen primary per-seed output or its "
                         "SHA256SUMS not found; refusing to reread run "
                         "directories unverified")
    sums = {}
    for ln in sums_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        h, name = ln.split(None, 1)
        sums[name.lstrip("*")] = h
    expected_hash = sums.get("outputs/per_seed_round100.csv")
    actual_hash = sha256(prim_psv_path)
    if expected_hash is None or expected_hash != actual_hash:
        raise SystemExit(f"FATAL: frozen primary per-seed output hash "
                         f"mismatch ({actual_hash[:16]}… != "
                         f"{str(expected_hash)[:16]}…)")

    prim_acc = {}
    for r in csv.DictReader(open(prim_psv_path, encoding="utf-8")):
        key = (r["dataset"], r["manuscript_row"], int(r["seed"]))
        if key in prim_acc:
            raise SystemExit(f"FATAL: duplicate per-seed primary row for "
                             f"{key}")
        if prim_run.get(key) != r["run_name"]:
            raise SystemExit(f"FATAL: per-seed run identity mismatch for "
                             f"{key}: manifest={prim_run.get(key)} vs "
                             f"per-seed={r['run_name']}")
        prim_acc[key] = float(r["round100_acc_pct"])

    latest = {}
    for i, r in enumerate(inv):
        if r["role"] == "latest_repeat_sensitivity":
            latest[(r["dataset"], r["manuscript_row"], int(r["seed"]))] = \
                (acc[i], r["run_name"])

    def latest_or_fail(key):
        """Latest-repeat value for a cell. A repeat-selection cell WITHOUT
        an inventory entry is a hard error (no silent fallback); a genuine
        single-source cell reuses its only run under an explicit label."""
        ds, row, s = key
        if key in latest:
            return latest[key][0], f"latest={latest[key][1]}"
        st = prim_status.get(key)
        if st is None:
            return None, "no proposed primary source"
        if st.startswith("PROPOSED (retrospective_repeat_selection"):
            raise SystemExit(
                f"FATAL: repeat-selection cell {key} has NO latest-repeat "
                f"entry in the frozen sensitivity inventory; refusing to "
                f"fall back to the primary value")
        return prim_acc[key], "same run (single source)"

    sens_rows = []
    for ds in DATASETS:
        for row in ("fedavg", "InfoOpt_Full"):
            base, lat, notes = [], [], []
            for s in SEEDS:
                key = (ds, row, s)
                if key not in prim_acc:
                    continue
                base.append(prim_acc[key])
                lv, note = latest_or_fail(key)
                lat.append(lv)
                notes.append(f"seed {s}: {note}")
            sens_rows.append({
                "dataset": ds, "row": row, "n": len(base),
                "primary_mean_earliest": float(np.mean(base)),
                "sensitivity_mean_latest": float(np.mean(lat)),
                "delta": float(np.mean(lat) - np.mean(base)),
                "per_seed_notes": notes})
    for ds in DATASETS:
        pairs, cnotes = [], []
        for s in SEEDS:
            ka = (ds, "InfoOpt_Full", s)
            kb = (ds, "fedavg", s)
            if ka not in prim_acc or kb not in prim_acc:
                continue
            a, note_a = latest_or_fail(ka)
            b, note_b = latest_or_fail(kb)
            cnotes.append(f"seed {s} InfoOpt: {note_a}")
            cnotes.append(f"seed {s} fedavg: {note_b}")
            pairs.append(a - b)
        if len(pairs) == 5:
            m_, sd_, ci_, p_, n_ = paired_t(pairs)
            sens_rows.append({"dataset": ds, "row": "InfoOpt_Full-FedAvg",
                              "n": 5, "sensitivity_mean_latest_diff": m_,
                              "ci95": [ci_[0], ci_[1]], "p_raw": p_,
                              "per_seed_notes": cnotes})
    json.dump({"label": "SENSITIVITY (latest repeat within stratum); "
                        "sources verified; no fallback to primary values; "
                        "single-source cells labelled 'same run'",
               "results": sens_rows},
              open(out_dir / "sensitivity_latest_repeat.json", "w",
                   encoding="utf-8"), indent=1)

    # --- CIFAR FedOpt sensitivity-only ---
    fedopt = []
    for i, r in enumerate(inv):
        if r["role"] == "cifar_fedopt_sensitivity_only":
            fedopt.append({"seed": int(r["seed"]), "run": r["run_name"],
                           "round100_acc_pct": acc[i]})
    fv = [x["round100_acc_pct"] for x in fedopt]
    json.dump({"label": "SENSITIVITY ONLY (strict E3 primary exclusion)",
               "n": len(fv),
               "mean": float(np.mean(fv)) if fv else None,
               "sd": float(np.std(fv, ddof=1)) if len(fv) > 1 else None,
               "runs": fedopt},
              open(out_dir / "sensitivity_cifar_fedopt.json", "w",
                   encoding="utf-8"), indent=1)

    # --- RA per-stratum summaries from verified records ---
    ra_by = defaultdict(list)
    for i, r in enumerate(inv):
        if r["role"] == "ra_stratum_member":
            ra_by[(r["dataset"], r["config_stratum"])].append(
                (int(r["seed"]), r["timestamp"] if "timestamp" in r else "",
                 r["run_name"], acc[i]))
    ra_out = []
    for (ds, st), members in sorted(ra_by.items()):
        by_seed = defaultdict(list)
        for seed, ts, name, a in members:
            by_seed[seed].append((ts, name, a))
        per_seed = []
        for s, cand in sorted(by_seed.items()):
            # earliest per seed within the stratum (need timestamps; the
            # inventory CSV does not carry them — recover from run name)
            cand_sorted = sorted(cand, key=lambda c: c[1])
            ts, name, a = cand_sorted[0]
            per_seed.append({"seed": s, "run_name": name,
                             "round100_acc_pct": a})
        vals = [p["round100_acc_pct"] for p in per_seed]
        ra_out.append({"dataset": ds, "stratum": st,
                       "n_seeds": len(vals),
                       "per_seed": per_seed,
                       "mean": float(np.mean(vals)),
                       "sd": float(np.std(vals, ddof=1))
                       if len(vals) > 1 else None})
    json.dump({"label": "RA-InfoOpt per-stratum summaries from VERIFIED "
                        "frozen inventory records; per-seed earliest; run "
                        "identities exported; no combined estimate",
               "strata": ra_out},
              open(out_dir / "ra_strata_summaries.json", "w",
                   encoding="utf-8"), indent=1)

    print(f"wrote {out_dir}: latest-repeat ({len(sens_rows)} rows), "
          f"FedOpt sensitivity (n={len(fv)}), RA strata "
          f"({len(ra_out)} blocks)")


if __name__ == "__main__":
    main(*sys.argv[1:])
