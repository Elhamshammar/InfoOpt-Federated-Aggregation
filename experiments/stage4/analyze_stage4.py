#!/usr/bin/env python3
"""Stage-4 analysis: the fixed 15-test primary family (second backbone).

Per PROTOCOL: InfoOpt minus each of {fedavg, uniform, shuffled_infoopt,
damping_050, damping_075} for each of the three datasets, at alpha=0.3
with the ResNet18SmallImage backbone — 15 two-sided paired tests over
seeds 100-104, paired by seed only after comparability verification.

The completed Stage-1 SmallCNN alpha=0.3 family is a SEPARATE reference
and is never read, merged or re-analyzed here. Cross-backbone
comparisons are descriptive in this design; significance in one
campaign and nonsignificance in another is NOT evidence of a backbone
interaction.

VERIFICATION CHAIN (all enforced before any calculation):
  1. RUN_MANIFEST.json must contain EXACTLY the 90 expected unique cell
     keys (backbone resnet18 x 3 datasets x alpha=0.3 x 5 seeds x
     6 arms). Duplicate keys, missing cells or unexpected cells abort.
  2. Frozen inputs re-hashed (snapshot + shared manifest).
  3. pending/partial/duplicate cells abort; invalid cells need a
     documented reason and a supporting verification report recording
     verified=false and a nonempty failure list.
  4. Every complete cell is RE-VERIFIED (backbone-aware verify_run.py);
     the manifest hash set must contain exactly the seven required
     evidence files with valid SHA-256 values and must match disk.
  5. Terminal accuracy from the verified metrics.csv row round == 100,
     cross-checked against the manifest's terminal_acc.
  6. Cross-arm initialization identity per dataset-alpha-seed within
     the resnet18 backbone.
  7. Output opened with mode "x"; existence checked up front.

Holm within the fixed family of 15; individual (non-simultaneous) CIs;
non-estimable counts as non-rejection and the family never shrinks.
Zero-variance differences are degenerate: report the constant
difference with a point CI and label the degeneracy.
"""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP_ROOT = Path(os.environ.get("EXP_ROOT",
                               str(HERE)))
SHARED_DIR = Path(os.environ.get("SHARED_DIR", str(EXP_ROOT / "shared")))
DATASETS = ("cifar100", "eurosat", "pathmnist")
ALPHAS = ("0.3",)
BACKBONE = "resnet18"
SEEDS = (100, 101, 102, 103, 104)
ARMS = ("fedavg", "uniform", "infoopt", "shuffled_infoopt",
        "damping_050", "damping_075")
CONTROLS = ("fedavg", "uniform", "shuffled_infoopt",
            "damping_050", "damping_075")
FAMILY_N = 15
EXPECTED = {(ds, alpha, arm, seed) for ds in DATASETS for alpha in ALPHAS
            for arm in ARMS for seed in SEEDS}
EVIDENCE = ("config_used.json", "metrics.csv", "summary.json",
            "final_model.pt", "partition_used.json",
            "diagnostics.jsonl", "init_state_hash.json")
_KNOWN_STATUSES = {"complete", "invalid", "pending", "partial",
                   "duplicate_error"}
_EXEC_INCOMPLETE = {"pending", "partial", "duplicate_error"}
_HEX64 = set("0123456789abcdef")


def _abort(msg):
    raise SystemExit(f"ANALYSIS ABORT: {msg}")


def _strict_json(text):
    def _reject(x):
        raise ValueError(f"non-strict JSON constant {x!r}")
    return json.loads(text, parse_constant=_reject)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_inputs():
    snap_manifest = HERE / "SNAPSHOT_HASHES.sha256"
    if not snap_manifest.is_file():
        _abort(f"snapshot hash manifest missing: {snap_manifest}")
    for line in snap_manifest.read_text(encoding="utf-8").splitlines():
        want, rel = line.split(" ", 1)
        rel = rel.lstrip("*")
        f = HERE / rel
        if not f.is_file():
            _abort(f"snapshot file missing: {rel}")
        if _sha256(f) != want:
            _abort(f"snapshot file changed since freeze: {rel}")
    shared_manifest = SHARED_DIR / "SHARED_MANIFEST.json"
    if not shared_manifest.is_file():
        _abort(f"shared manifest missing: {shared_manifest}")
    sm = _strict_json(shared_manifest.read_text(encoding="utf-8"))
    for entry in sm["files"]:
        for kind in ("partition", "schedule"):
            rec = entry[kind]
            f = SHARED_DIR / Path(rec["path"]).name
            if not f.is_file():
                _abort(f"shared file missing: {f.name}")
            if _sha256(f) != rec["sha256"]:
                _abort(f"shared file changed since generation: {f.name}")


def load_manifest_cells():
    manifest = EXP_ROOT / "RUN_MANIFEST.json"
    if not manifest.is_file():
        _abort(f"execution manifest missing: {manifest} "
               f"(run the launcher's phase 2 first)")
    m = _strict_json(manifest.read_text(encoding="utf-8"))
    cells_list = m["cells"]
    keys = [(c["dataset"], str(float(c["alpha"])), c["arm"], c["seed"])
            for c in cells_list]
    if len(keys) != len(set(keys)):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        _abort(f"duplicate cell keys in manifest: {dupes}")
    if set(keys) != EXPECTED:
        missing = sorted(EXPECTED - set(keys))
        extra = sorted(set(keys) - EXPECTED)
        _abort(f"manifest grid != expected 90 cells; "
               f"missing={missing[:5]} extra={extra[:5]}")
    return {(c["dataset"], str(float(c["alpha"])), c["arm"], c["seed"]): c
            for c in cells_list}


def paired_t(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / math.sqrt(n)
    if se > 0:
        t = m / se
        tc = stats.t.ppf(0.975, n - 1)
        return m, sd, [m - tc * se, m + tc * se], \
            float(2 * stats.t.sf(abs(t), n - 1)), False
    # degenerate zero-variance case
    return m, 0.0, [m, m], (1.0 if m == 0 else 0.0), True


def holm(pairs, family_n=FAMILY_N):
    """Holm over a fixed family; None p (non-estimable) never rejects."""
    valid = sorted(((k, p) for k, p in pairs if p is not None),
                   key=lambda x: x[1])
    out, prev = {}, 0.0
    for rank, (k, p) in enumerate(valid):
        out[k] = max(min(1.0, p * (family_n - rank)), prev)
        prev = out[k]
    for k, p in pairs:
        if p is None:
            out[k] = None
    return out


def main():
    out_path = EXP_ROOT / "analysis_stage4.json"
    if out_path.exists():
        _abort(f"output exists, refusing to overwrite: {out_path}")
    verify_frozen_inputs()
    cells = load_manifest_cells()

    unknown = [c for c in cells.values()
               if c["status"] not in _KNOWN_STATUSES]
    if unknown:
        u = unknown[0]
        _abort(f"unknown cell status {u['status']!r} at "
               f"{(u['dataset'], u.get('alpha'), u['arm'], u['seed'])}")
    bad_exec = [c for c in cells.values()
                if c["status"] in _EXEC_INCOMPLETE]
    if bad_exec:
        _abort(f"{len(bad_exec)} cells are pending/partial/duplicate - "
               f"execution is not finished (first: "
               f"{(bad_exec[0]['dataset'], bad_exec[0].get('alpha'), bad_exec[0]['arm'], bad_exec[0]['seed'])} "
               f"{bad_exec[0]['status']})")
    for key, c in cells.items():
        if c["status"] != "invalid":
            continue
        if not (isinstance(c.get("reason"), str) and c["reason"].strip()):
            _abort(f"invalid cell {key} without a documented reason")
        rep = c.get("verification_report")
        if not rep or not Path(rep).is_file():
            _abort(f"invalid cell {key} without a supporting "
                   f"verification report")
        try:
            repj = _strict_json(Path(rep).read_text(encoding="utf-8"))
        except Exception as e:
            _abort(f"invalid cell {key}: unreadable verification "
                   f"report: {e}")
        if repj.get("verified") is not False or not repj.get("failures"):
            _abort(f"invalid cell {key}: verification report does not "
                   f"record a failure")

    verif_dir = EXP_ROOT / "analysis_verification"
    verif_dir.mkdir(parents=True, exist_ok=True)

    terminal = {}
    skipped = []
    init_hashes = {}
    for (ds, alpha, arm, seed), c in sorted(cells.items()):
        if c["status"] != "complete":
            skipped.append({"cell": [ds, alpha, arm, seed],
                            "status": c["status"],
                            "reason": c.get("reason", "")})
            continue
        rd = Path(c["run_dir"])
        # fresh full verification — never trust the manifest's label
        report_out = verif_dir / (f"{BACKBONE}_{ds}_{arm}_alpha{alpha}"
                                  f"_seed{seed}.json")
        r = subprocess.run(
            [sys.executable, str(HERE / "verify_run.py"),
             "--run-dir", str(rd), "--dataset", ds, "--seed", str(seed),
             "--alpha", alpha, "--backbone", BACKBONE, "--arm", arm,
             "--schedules-dir", str(SHARED_DIR),
             "--report-out", str(report_out)],
            capture_output=True, text=True)
        if r.returncode != 0:
            _abort(f"fresh verification failed for "
                   f"{ds}/alpha{alpha}/{arm}/{seed}: "
                   f"{(r.stdout + r.stderr).strip().splitlines()[-1][:300]}")
        # the manifest must carry exactly the seven required evidence
        # hashes, each a valid SHA-256, and they must match disk
        rec = c.get("hashes")
        if not isinstance(rec, dict) or set(rec.keys()) != set(EVIDENCE):
            _abort(f"incomplete evidence-hash set for "
                   f"{ds}/alpha{alpha}/{arm}/{seed}: expected "
                   f"{sorted(EVIDENCE)}, got "
                   f"{sorted(rec.keys()) if isinstance(rec, dict) else rec!r}")
        for fn, want in rec.items():
            if not (isinstance(want, str) and len(want) == 64
                    and set(want) <= _HEX64):
                _abort(f"invalid SHA-256 value for {fn} in manifest "
                       f"for {ds}/alpha{alpha}/{arm}/{seed}")
            f = rd / fn
            if not f.is_file() or _sha256(f) != want:
                _abort(f"evidence changed since manifest for "
                       f"{ds}/alpha{alpha}/{arm}/{seed}: {fn}")
        # terminal accuracy from the verified metrics, round == 100
        rows = list(csv.DictReader(open(rd / "metrics.csv",
                                        encoding="utf-8")))
        r100 = [x for x in rows if int(x["round"]) == 100]
        if len(r100) != 1:
            _abort(f"no unique round-100 row for "
                   f"{ds}/alpha{alpha}/{arm}/{seed}")
        acc = float(r100[0]["test_acc"])
        if not math.isfinite(acc):
            _abort(f"non-finite round-100 accuracy for "
                   f"{ds}/alpha{alpha}/{arm}/{seed}")
        if c.get("terminal_acc") is None or \
                abs(acc - float(c["terminal_acc"])) > 1e-12:
            _abort(f"manifest terminal_acc disagrees with verified "
                   f"metrics for {ds}/alpha{alpha}/{arm}/{seed}")
        terminal[(ds, alpha, arm, seed)] = acc
        ih = _strict_json((rd / "init_state_hash.json")
                          .read_text(encoding="utf-8"))
        init_hashes.setdefault((ds, alpha, seed), {})[arm] = \
            ih["init_state_sha256"]

    # cross-arm initialization identity within each dataset-alpha-seed
    for (ds, alpha, seed), per_arm in sorted(init_hashes.items()):
        if len(set(per_arm.values())) != 1:
            _abort(f"cross-arm init hash mismatch at {ds} alpha{alpha} "
                   f"seed {seed}: {per_arm}")

    tests = []
    for alpha in ALPHAS:
        for ds in DATASETS:
            for ctrl in CONTROLS:
                name = f"{BACKBONE} {ds} alpha={alpha}: infoopt - {ctrl}"
                pairs = []
                for s in SEEDS:
                    a = terminal.get((ds, alpha, "infoopt", s))
                    b = terminal.get((ds, alpha, ctrl, s))
                    if a is not None and b is not None:
                        pairs.append((a - b) * 100)  # percentage points
                if len(pairs) < 3:
                    tests.append({"contrast": name, "estimable": False,
                                  "n": len(pairs),
                                  "reason": "fewer than 3 valid paired "
                                            "seeds",
                                  "mean_diff_pp": None, "sd": None,
                                  "ci95": None, "p_raw": None,
                                  "p_holm": None, "degenerate": None})
                    continue
                m, sd, ci, p, degen = paired_t(pairs)
                tests.append({"contrast": name, "estimable": True,
                              "n": len(pairs), "n_flag": "n<5"
                              if len(pairs) < 5 else "",
                              "mean_diff_pp": m, "sd": sd, "ci95": ci,
                              "p_raw": p, "degenerate": degen})
    holm_adj = holm([(t["contrast"], t["p_raw"]) for t in tests])
    for t in tests:
        t["p_holm"] = holm_adj[t["contrast"]]

    out = {
        "label": "Stage-4 fixed 15-test primary family "
                 "(ResNet18SmallImage, alpha=0.3); Holm within the "
                 "family of 15; individual (non-simultaneous) CIs; "
                 "Stage-1 SmallCNN family kept separate",
        "backbone": BACKBONE,
        "verification": {
            "frozen_inputs_rehashed": True,
            "manifest_cells": 90,
            "complete_cells_reverified": len(terminal),
            "documented_failures": skipped,
            "terminal_acc_source": "verified metrics.csv round == 100",
            "cross_arm_init_identity": "enforced per dataset-alpha-seed "
                                       "within the resnet18 backbone",
        },
        "tests": tests,
    }
    try:
        with open(out_path, "x", encoding="utf-8") as f:
            f.write(json.dumps(out, indent=1, allow_nan=False))
    except FileExistsError:
        _abort(f"output exists, refusing to overwrite: {out_path}")
    print(f"wrote {out_path}")
    for t in tests:
        if t["estimable"]:
            print(f"{t['contrast']}: n={t['n']} diff={t['mean_diff_pp']:+.2f} "
                  f"CI=[{t['ci95'][0]:.2f},{t['ci95'][1]:.2f}] "
                  f"p={t['p_raw']:.4f} holm={t['p_holm']:.4f}"
                  + (" DEGENERATE" if t["degenerate"] else ""))
        else:
            print(f"{t['contrast']}: NOT ESTIMABLE ({t['reason']})")


if __name__ == "__main__":
    main()
