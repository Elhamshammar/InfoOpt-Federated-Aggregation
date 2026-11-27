#!/usr/bin/env python3
"""Direct verification of the historical InfoOpt controller on oracle scores.

This script:
  1. Imports the original InfoOptController and solver from the Stage 3 snapshot
     (read-only; no source modification).
  2. Builds the real CIFAR-100 Dirichlet partitions and participation schedules
     for alpha=0.1, seeds 100-104.
  3. Feeds dummy ClientResult objects whose mi_proxy equals the oracle score,
     so the controller's composite score reduces to the oracle after robust
     normalization.
  4. Runs both InfoOpt-Full and InfoOpt-NoJS variants.
  5. Records aligned and deterministic-shuffled weights, fallback flags,
     solver status, effective clients, temperature, and compares the Full
     controller output against the MLJ-P0 audit replica.
  6. For NoJS, enumerates all distinct oracle score assignments in each
     informative cohort to compute the exact permutation expectation.

No neural-network training is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results_nojs"
DATASET_CACHE = ROOT / "_dataset_cache"
SNAPSHOT = (ROOT.parent / "InfoOpt_Code_For_Journal" /
            "source_snapshots" / "ITSL_stage3_prep_20261012_v2" / "snapshot")

sys.path.insert(0, str(SNAPSHOT))
from data.partition import class_histogram, dirichlet_partition  # noqa: E402
from methods.infoopt import InfoOptController, solve_entropic_weights  # noqa: E402

# Also import the MLJ-P0 replica for cross-check
sys.path.insert(0, str(ROOT / "scripts"))
from mlj_p0_cpu_audit import (  # noqa: E402
    compute_oracle_weights,
    load_cifar100_labels,
    generate_schedule,
    select_corrupted_clients,
)


@dataclass
class DummyResult:
    """Minimal object satisfying InfoOptController.compute_weights()."""
    client_id: int
    mi_proxy: float
    entropy: float
    update_norm: float
    train_loss_std: float
    label_hist: np.ndarray
    num_samples: int


def build_controller_cfg(seed: int, variant: str, shuffle: bool) -> dict:
    """Load Study 3 config and force the controller to use oracle scores."""
    cfg_path = (ROOT.parent / "InfoOpt_GitHub_Release" /
                "configs" / "stage3" / "configs_effective" /
                "stage3_cifar100_infoopt_alpha0.1.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["seed"] = seed
    cfg["dataset"]["name"] = "CIFAR100"
    io = cfg.setdefault("infoopt", {})
    io["signal_ema_beta"] = 1.0   # no cross-round smoothing
    io["lambda_mi"] = 1.0
    io["lambda_entropy"] = 0.0
    io["lambda_stability"] = 0.0
    io["lambda_js"] = 0.0
    io["js_mode"] = "constraint"
    io["epsilon_rel"] = 1.0
    io["temperature"] = 1.0
    io["kappa_frac"] = 0.6
    io["robust_scale"] = True
    io["score_clip"] = None
    io["fedavg_mix"] = 0.0
    io["min_weight"] = 0.0
    io["min_weight_fraction"] = 0.0
    io["early_stop"] = {"enabled": False}
    if shuffle:
        io["score_shuffle"] = "stage1"
    else:
        io.pop("score_shuffle", None)
    return cfg


def make_dummy_results(
    selected: Sequence[int],
    oracle_scores: Sequence[float],
    hists: np.ndarray,
    sizes: np.ndarray,
) -> list[DummyResult]:
    return [
        DummyResult(
            client_id=int(cid),
            mi_proxy=float(score),
            entropy=0.0,
            update_norm=0.0,
            train_loss_std=0.0,
            label_hist=hists[i].astype(float),
            num_samples=int(sizes[i]),
        )
        for i, (cid, score) in enumerate(zip(selected, oracle_scores))
    ]


def run_controller_audit(alpha: float = 0.1):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_cifar100_labels()
    num_classes = int(labels.max()) + 1
    global_hist = np.bincount(labels, minlength=num_classes).astype(float)

    seeds = (100, 101, 102, 103, 104)
    rounds = 100
    num_clients = 10
    clients_per_round = 5
    num_corrupted = 3

    records = []
    full_records = []

    for seed in seeds:
        client_indices = dirichlet_partition(
            labels, num_clients=num_clients, alpha=alpha,
            min_samples_per_client=50, seed=seed)
        schedule = generate_schedule(num_clients, clients_per_round, rounds, seed)
        corrupted_clients = select_corrupted_clients(num_clients, num_corrupted, seed)

        hists_all = np.stack([
            class_histogram(labels, client_indices[cid], num_classes)
            for cid in range(num_clients)
        ]).astype(float)
        sizes_all = np.array([len(client_indices[cid]) for cid in range(num_clients)])

        cfg_full = build_controller_cfg(seed, "InfoOpt-Full", shuffle=False)
        cfg_full_shuf = build_controller_cfg(seed, "InfoOpt-Full", shuffle=True)
        cfg_nojs = build_controller_cfg(seed, "InfoOpt-NoJS", shuffle=False)
        cfg_nojs_shuf = build_controller_cfg(seed, "InfoOpt-NoJS", shuffle=True)

        ctrl_full = InfoOptController(cfg_full, global_hist, variant="InfoOpt-Full")
        ctrl_full_shuf = InfoOptController(cfg_full_shuf, global_hist, variant="InfoOpt-Full")
        ctrl_nojs = InfoOptController(cfg_nojs, global_hist, variant="InfoOpt-NoJS")
        ctrl_nojs_shuf = InfoOptController(cfg_nojs_shuf, global_hist, variant="InfoOpt-NoJS")

        for rnd, selected in enumerate(schedule, start=1):
            selected = sorted(selected)
            n_corrupt = sum(1 for cid in selected if cid in corrupted_clients)
            raw_oracle = np.array([
                -0.8 if cid in corrupted_clients else 0.0
                for cid in selected
            ], dtype=float)
            cohort_hists = hists_all[selected]
            cohort_sizes = sizes_all[selected]

            # Full controller, aligned
            res_full = make_dummy_results(selected, raw_oracle, cohort_hists, cohort_sizes)
            w_full, info_full = ctrl_full.compute_weights(res_full, round_idx=rnd)
            w_full = np.asarray(w_full, dtype=float)

            # Full controller, shuffled
            res_full_shuf = make_dummy_results(selected, raw_oracle, cohort_hists, cohort_sizes)
            w_full_shuf, info_full_shuf = ctrl_full_shuf.compute_weights(res_full_shuf, round_idx=rnd)
            w_full_shuf = np.asarray(w_full_shuf, dtype=float)

            # NoJS controller, aligned
            res_nojs = make_dummy_results(selected, raw_oracle, cohort_hists, cohort_sizes)
            w_nojs, info_nojs = ctrl_nojs.compute_weights(res_nojs, round_idx=rnd)
            w_nojs = np.asarray(w_nojs, dtype=float)

            # NoJS controller, deterministic shuffled
            res_nojs_shuf = make_dummy_results(selected, raw_oracle, cohort_hists, cohort_sizes)
            w_nojs_shuf, info_nojs_shuf = ctrl_nojs_shuf.compute_weights(res_nojs_shuf, round_idx=rnd)
            w_nojs_shuf = np.asarray(w_nojs_shuf, dtype=float)

            # Replica (MLJ-P0) for cross-check
            w_rep_aligned, info_rep_aligned = compute_oracle_weights(
                raw_oracle, cohort_sizes, cohort_hists, global_hist,
                robust=True, shuffle=False, seed=seed, round_idx=rnd)
            w_rep_shuffled, info_rep_shuffled = compute_oracle_weights(
                raw_oracle, cohort_sizes, cohort_hists, global_hist,
                robust=True, shuffle=True, seed=seed, round_idx=rnd)

            corrupt_mask = np.array([
                cid in corrupted_clients for cid in selected], dtype=bool)

            def corrupt_mass(w):
                return float(np.sum(np.asarray(w)[corrupt_mask]))

            # Exact permutation expectation for NoJS
            expected_nojs_mass = None
            if n_corrupt > 0:
                masses = []
                seen = set()
                base_scores = [0.0] * (5 - n_corrupt) + [-0.8] * n_corrupt
                for perm in permutations(base_scores):
                    if perm in seen:
                        continue
                    seen.add(perm)
                    res_perm = make_dummy_results(selected, perm, cohort_hists, cohort_sizes)
                    w_perm, _ = ctrl_nojs.compute_weights(res_perm, round_idx=rnd)
                    w_perm = np.asarray(w_perm, dtype=float)
                    # corrupted positions are those where perm score is -0.8
                    perm_corrupt_mask = np.array([s == -0.8 for s in perm])
                    masses.append(float(w_perm[perm_corrupt_mask].sum()))
                expected_nojs_mass = float(np.mean(masses))

            records.append({
                "seed": seed,
                "round": rnd,
                "selected": selected,
                "n_corrupt": n_corrupt,
                "full_aligned_mass": corrupt_mass(w_full),
                "full_shuffled_mass": corrupt_mass(w_full_shuf),
                "full_fallback": bool(info_full.get("fallback_used")),
                "full_solver_success": info_full.get("solver_final_status", {}).get("success"),
                "full_effective_clients": info_full.get("effective_clients"),
                "nojs_aligned_mass": corrupt_mass(w_nojs),
                "nojs_shuffled_mass": corrupt_mass(w_nojs_shuf),
                "nojs_expected_mass": expected_nojs_mass,
                "nojs_effective_clients": info_nojs.get("effective_clients"),
                "nojs_kappa_satisfied": info_nojs.get("effective_clients", 0.0) >= 3.0 - 1e-6,
                "nojs_temperature": info_nojs.get("temperature_used"),
                "nojs_solver_calls": info_nojs.get("solver_calls"),
                "replica_aligned_mass": corrupt_mass(w_rep_aligned),
                "replica_shuffled_mass": corrupt_mass(w_rep_shuffled),
                "replica_fallback": bool(info_rep_aligned.get("fallback_used")),
                "full_vs_replica_l1_aligned": float(np.sum(np.abs(w_full - w_rep_aligned))),
                "full_vs_replica_l1_shuffled": float(np.sum(np.abs(w_full_shuf - w_rep_shuffled))),
            })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "controller_audit_alpha0.1.csv", index=False)

    agg = df.groupby("n_corrupt").agg({
        "full_aligned_mass": "mean",
        "full_shuffled_mass": "mean",
        "full_fallback": "mean",
        "nojs_aligned_mass": "mean",
        "nojs_shuffled_mass": "mean",
        "nojs_expected_mass": "mean",
        "nojs_kappa_satisfied": "mean",
        "nojs_effective_clients": "mean",
        "replica_aligned_mass": "mean",
        "replica_shuffled_mass": "mean",
        "replica_fallback": "mean",
        "full_vs_replica_l1_aligned": "mean",
        "full_vs_replica_l1_shuffled": "mean",
    }).reset_index()
    agg.to_csv(RESULTS_DIR / "controller_audit_alpha0.1_summary.csv", index=False)

    print(f"[alpha={alpha}] wrote {len(df)} cohort records")
    print(agg.to_string(index=False))

    # Save a small compatibility verdict
    max_l1 = max(df["full_vs_replica_l1_aligned"].max(),
                 df["full_vs_replica_l1_shuffled"].max())
    verdict = {
        "historical_controller_imported": True,
        "max_full_vs_replica_l1": float(max_l1),
        "full_fallback_fraction_mixed": float(df[df["n_corrupt"] > 0]["full_fallback"].mean()),
        "nojs_kappa_satisfied_fraction": float(df[df["n_corrupt"] > 0]["nojs_kappa_satisfied"].mean()),
        "nojs_aligned_below_expected_mass": bool(
            df[df["n_corrupt"] > 0]["nojs_aligned_mass"].mean() <
            df[df["n_corrupt"] > 0]["nojs_expected_mass"].mean()
        ),
    }
    (RESULTS_DIR / "controller_audit_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    run_controller_audit(alpha=0.1)
