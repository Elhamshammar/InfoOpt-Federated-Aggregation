#!/usr/bin/env python
"""Regenerate validation_selected_scaling per-example rows for the
TRAJECTORY-cohort sources from retained pilot artifacts. NO training.

Scope (2026-10-08 correction): the pilot checkpoints contain post-local
client states ONLY for the trajectory cohort of each (seed, round). The
repetition and alternative-cohort sources' client states were computed
in-memory and never persisted, so those rows cannot be regenerated
without retraining (which is out of scope). This script therefore
regenerates ONLY the 9 trajectory-cohort sources (3 seeds x 3 rounds),
into a NEW dated directory. Originals are never touched.

Inputs (all retained from the pilot run):
  - checkpoints/seed_*/checkpoint_round_*.pt  (sd0, client_sds, weights)
  - partition.json                            (selection/eval indices)
  - results/pilot_decision.json               (per-source selected gamma)
  - bundled reference model (provenance hash verified by ref_adapter)
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stepsize_diag.candidates import build_candidate, POLICY_ALL_FLOATING
from stepsize_diag.data import load_cifar100_raw, partition_hashes
from stepsize_diag.fedavg import average_states
from stepsize_diag.pilot_decision import (SELECTION_HOLDOUT,
                                          EVALUATION_HOLDOUT)
from stepsize_diag.pilot_eval import (PerExampleCSVWriter,
                                      evaluate_state_per_example)
from stepsize_diag.ref_adapter import load_reference_model
from stepsize_diag.reference_impl import apply_diagnostic_determinism

POLICY = "validation_selected_scaling"
GAMMA_ORIGIN = "rule_validation_selected"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default="results/stepsize_diag_pilot_20261004_175620")
    ap.add_argument("--out", default="results/per_example_correction_20261008")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output dir not empty; refusing to overwrite: {out}")
    out.mkdir(parents=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    apply_diagnostic_determinism()

    # --- partition indices (exact saved examples) + hash verification ---
    part = json.load(open(run_dir / "partition.json"))
    sel_idx = np.array(part["selection_indices"], dtype=np.int64)
    eval_idx = np.array(part["evaluation_indices"], dtype=np.int64)
    retained = [np.array(ix, dtype=np.int64)
                for ix in part["retained_client_indices"]]
    recomputed = partition_hashes(retained, sel_idx, eval_idx)
    if recomputed != part["hashes"]:
        raise SystemExit(f"partition hash mismatch: {recomputed} vs "
                         f"{part['hashes']}")
    counts = {int(c): int(n) for c, n in
              enumerate(part["retained_client_sizes"])}

    # --- data + model (bundled reference, provenance-checked) ---
    raw_x, raw_y = load_cifar100_raw(args.data_root)
    model_ctor, ref_prov = load_reference_model(100, False)
    eval_model = model_ctor().to(device)

    # --- retained per-source selected gammas ---
    dec = json.load(open(run_dir / "results" / "pilot_decision.json"))
    vg = dec["per_source_validation_gamma"]

    seeds = [201, 202, 203]
    rounds = [5, 50, 100]
    ckpt_hashes = {}
    cells = []

    t0 = time.time()
    csv_path = out / "per_example_losses_validation_selected_trajectory_only.csv"
    writer = PerExampleCSVWriter(csv_path)
    for seed in seeds:
        for rd in rounds:
            key = f"{seed}|{rd}|trajectory_cohort"
            if key not in vg:
                raise SystemExit(f"retained mapping missing key {key}")
            gamma = float(vg[key])
            ckpt_path = (run_dir / "checkpoints" / f"seed_{seed}"
                         / f"checkpoint_round_{rd}.pt")
            ckpt_hashes[str(ckpt_path)] = sha256_file(ckpt_path)
            ck = torch.load(ckpt_path, map_location=device,
                            weights_only=False)
            sd0 = ck["global_sd_before"]
            clients = list(ck["selected_clients"])
            sds = {int(c): sd for c, sd in ck["client_sds"].items()}
            weights = {int(c): float(w)
                       for c, w in ck["weights"].items()}
            wsum = sum(weights[c] for c in clients)
            assert abs(wsum - 1.0) < 1e-9, f"weights sum {wsum}"
            sd_fed = average_states([sds[c] for c in clients],
                                    [weights[c] for c in clients])
            first_sd = sds[clients[0]]
            cand, rep = build_candidate(sd0, sd_fed, gamma, first_sd,
                                        POLICY_ALL_FLOATING)
            if not rep["valid"]:
                raise SystemExit(
                    f"FATAL: invalid candidate for {key} gamma={gamma}: "
                    f"{rep['invalid_entries']}; refusing to export losses "
                    f"for an invalid candidate")
            for holdout, idx in ((SELECTION_HOLDOUT, sel_idx),
                                 (EVALUATION_HOLDOUT, eval_idx)):
                pe = evaluate_state_per_example(
                    model_ctor, cand, raw_x, raw_y, idx, device,
                    shared_model=eval_model)
                writer.write_policy_on_holdout(
                    seed=seed, rd=rd, source="trajectory_cohort",
                    rep_seed="trajectory", cohort_id="trajectory",
                    holdout=holdout, policy=POLICY,
                    gamma=gamma, gamma_origin=GAMMA_ORIGIN,
                    per_example=pe)
                cells.append({"key": key, "gamma": gamma,
                              "holdout": holdout,
                              "n": pe["n_samples"],
                              "mean_loss": pe["loss"],
                              "mean_acc": pe["accuracy"]})
            print(f"  {key}: gamma={gamma} done")
    writer.close()

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": ("PARTIAL: trajectory_cohort sources only (9 of 90). "
                  "The 81 repetition/alt-cohort sources were NOT "
                  "regenerated: their post-local client states were never "
                  "persisted and regenerating them requires retraining."),
        "missing_states": {
            "original_cohort_rep": ("per (seed, round): post-local client "
                                    "state dicts for the 5 trajectory-cohort "
                                    "clients x 3 repetition seeds"),
            "alt_cohort": ("per (seed, round): post-local client state "
                           "dicts for the 2 alternative cohorts x 5 clients "
                           "x 3 repetition seeds"),
            "note": ("these were computed in-memory by the pilot and not "
                     "written to disk; only trajectory_cohort client_sds "
                     "were checkpointed"),
        },
        "gamma_source": str(run_dir / "results" / "pilot_decision.json"),
        "run_dir": str(run_dir),
        "checkpoint_sha256": ckpt_hashes,
        "partition_hashes_verified": True,
        "model_provenance": ref_prov,
        "device": str(device),
        "runtime_seconds": time.time() - t0,
        "cells": cells,
    }
    with open(out / "regeneration_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"WROTE {csv_path}")
    print(f"runtime: {meta['runtime_seconds']:.1f}s, cells: {len(cells)}")


if __name__ == "__main__":
    main()
