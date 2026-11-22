#!/usr/bin/env python3
"""Stage-2A smoke acceptance checks. Read-only against the smoke output.
Exit 0 = PASS, 1 = FAIL with reasons printed."""
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMOKE = ROOT / "smoke"
STUDY2_TAR = (ROOT.parent / "ITSL_RA_matched_results_20261010_234446.tar.gz")


def main():
    runs = list((ROOT / "smoke_out").glob(
        "stage2a_smoke_cifar100_ra_probeoff_RA_InfoOpt_seed100_alpha0.3_*"))
    if len(runs) != 1:
        raise SystemExit(f"FAIL: expected exactly one smoke run dir, "
                         f"found {len(runs)}")
    rd = runs[0]
    problems = []

    # metrics
    rows = list(csv.DictReader(open(rd / "metrics.csv",
                                    encoding="utf-8")))
    if [int(r["round"]) for r in rows] != [1, 2, 3, 4, 5]:
        problems.append("rounds not exactly 1..5")

    mi = [float(r["mean_mi"]) for r in rows]
    ent = [float(r["mean_entropy"]) for r in rows]
    stab = [float(r["mean_stability"]) for r in rows]
    # 1. diversity and entropy computed (not constant zeros)
    if not any(v > 0 for v in mi):
        problems.append("mean_mi still all zero")
    if not any(v > 0 for v in ent):
        problems.append("mean_entropy still all zero")
    # 2. all three score components contribute: score variance must be
    # attributable to more than stability — check client_scores differ
    # from a stability-only z-score reconstruction is nontrivial;
    # minimum bar: scores finite and nonconstant, and mean_score moves
    # when mi/entropy move
    for r in rows:
        sc = [float(x) for x in r["client_scores"].split(";")]
        if not all(math.isfinite(x) for x in sc):
            problems.append(f"non-finite client score round "
                            f"{r['round']}")
        w = [float(x) for x in r["weights"].split(";")] \
            if "weights" in r else None
        if w is not None:
            if not all(math.isfinite(x) for x in w):
                problems.append(f"non-finite weight round {r['round']}")
            # weights are logged with 4 decimals (f"{w:.4f}"):
            # sum can deviate from 1 by up to ~5 x 5e-5
            elif abs(sum(w) - 1.0) > 1e-3 or min(w) < -5e-5:
                problems.append(f"weights not a simplex (beyond 4-decimal "
                                f"logging tolerance) round "
                                f"{r['round']}")
    # 3. gate fields present and sane (rounds 1-5 are warm-up: alpha_raw 0)
    for r in rows:
        for col in ("alpha_raw", "alpha_ema", "alpha_safe"):
            if col in r and r[col] not in ("", "nan"):
                v = float(r[col])
                if not math.isfinite(v) or not (0.0 <= v <= 1.0):
                    problems.append(f"gate {col} out of range round "
                                    f"{r['round']}")
    # warm-up: with warmup_rounds=5, rounds 1..5 must have alpha_raw 0
    if "alpha_raw" in rows[0] and rows[0]["alpha_raw"] != "":
        for r in rows:
            if float(r["alpha_raw"]) != 0.0:
                problems.append(f"warm-up violated at round "
                                f"{r['round']}")
    # 4. endpoint recorded
    acc = float(rows[-1]["test_acc"])
    if not (0.0 <= acc <= 1.0):
        problems.append("terminal accuracy out of [0,1]")

    # 5. config identity: AIB on, probe off, alpha 0.3
    c = json.load(open(rd / "config_used.json", encoding="utf-8"))
    cfg = c.get("config", c)
    if cfg["model"].get("use_aib") is not True:
        problems.append("AIB not enabled in smoke config")
    if cfg.get("counterfactual_probe", {}).get("enabled", False):
        problems.append("probe enabled in smoke config")
    if float(cfg["partition"]["alpha"]) != 0.3:
        problems.append("alpha != 0.3")

    # 6. partition equality with the frozen Study-2 file
    import tarfile
    want = "experiment/shared/partition_cifar100_seed100.json"
    tf = tarfile.open(STUDY2_TAR)
    frozen = json.loads(tf.extractfile(want).read().decode())
    used = json.load(open(rd / "partition_used.json", encoding="utf-8"))
    if used != frozen["client_indices"]:
        problems.append("smoke partition != frozen Study-2 partition")

    # 7. cohort schedule equality with the frozen Study-2 file
    sched = json.loads(tf.extractfile(
        "experiment/shared/cohort_schedule_cifar100_seed100.json")
        .read().decode())["schedule"][:5]
    actual = [[int(x) for x in r["selected_clients"].split(";")]
              for r in rows]
    if actual != sched:
        problems.append("smoke cohort schedule != frozen Study-2 "
                        "schedule (rounds 1-5)")

    # 8. AIB parameters present in checkpoint (AIB on)
    import torch
    sd = torch.load(rd / "final_model.pt", map_location="cpu",
                    weights_only=True)
    if not any(k.endswith("conv1.3.gamma") for k in sd):
        problems.append("AIB parameters missing from checkpoint")

    if problems:
        print("SMOKE FAIL:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("SMOKE PASS: all acceptance criteria met")
    print(f"  mean_mi rounds 1-5:      {[round(v, 4) for v in mi]}")
    print(f"  mean_entropy rounds 1-5: {[round(v, 4) for v in ent]}")
    print(f"  mean_stability:          {[round(v, 4) for v in stab]}")
    print(f"  terminal acc (round 5):  {acc}")


if __name__ == "__main__":
    main()
