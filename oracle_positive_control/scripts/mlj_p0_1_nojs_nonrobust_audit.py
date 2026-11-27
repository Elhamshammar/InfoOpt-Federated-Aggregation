#!/usr/bin/env python3
"""NoJS CPU audit with non-robust normalization for the oracle positive control.

Robust normalization of the binary oracle {0, -0.8} produces MAD=0 and an
infinite normalized gap, which causes the historical kappa floor to be violated
for cohorts with three corrupted clients. This audit evaluates the smallest
justified correction: switch `robust_scale` to False while keeping all other
historical InfoOpt-NoJS mechanics (softmax, temperature search, kappa,
stage-1 shuffle, no JS constraint).

No neural-network training is performed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results_nojs_nonrobust"
SNAPSHOT = (ROOT.parent / "InfoOpt_Code_For_Journal" /
            "source_snapshots" / "ITSL_stage3_prep_20261012_v2" / "snapshot")

sys.path.insert(0, str(SNAPSHOT))
from data.partition import class_histogram, dirichlet_partition  # noqa: E402
from methods.infoopt import InfoOptController  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from mlj_p0_cpu_audit import (  # noqa: E402
    load_cifar100_labels,
    generate_schedule,
    select_corrupted_clients,
)


@dataclass
class DummyResult:
    client_id: int
    mi_proxy: float
    entropy: float
    update_norm: float
    train_loss_std: float
    label_hist: np.ndarray
    num_samples: int


def build_cfg(seed: int, shuffle: bool) -> dict:
    cfg_path = (ROOT.parent / "InfoOpt_GitHub_Release" /
                "configs" / "stage3" / "configs_effective" /
                "stage3_cifar100_infoopt_alpha0.1.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["seed"] = seed
    cfg["dataset"]["name"] = "CIFAR100"
    io = cfg.setdefault("infoopt", {})
    io["signal_ema_beta"] = 1.0
    io["lambda_mi"] = 1.0
    io["lambda_entropy"] = 0.0
    io["lambda_stability"] = 0.0
    io["js_mode"] = "constraint"
    io["epsilon_rel"] = 1.0
    io["temperature"] = 1.0
    io["kappa_frac"] = 0.6
    io["robust_scale"] = False   # correction
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


def make_results(selected, oracle_scores, hists, sizes):
    return [
        DummyResult(
            client_id=int(cid),
            mi_proxy=float(score),
            entropy=0.0,
            update_norm=0.0,
            train_loss_std=0.0,
            label_hist=hists[cid].astype(float),
            num_samples=int(sizes[cid]),
        )
        for cid, score in zip(selected, oracle_scores)
    ]


def run_audit(alpha: float = 0.1):
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

        cfg_aligned = build_cfg(seed, shuffle=False)
        cfg_shuffled = build_cfg(seed, shuffle=True)
        ctrl_aligned = InfoOptController(cfg_aligned, global_hist, variant="InfoOpt-NoJS")
        ctrl_shuffled = InfoOptController(cfg_shuffled, global_hist, variant="InfoOpt-NoJS")

        for rnd, selected in enumerate(schedule, start=1):
            selected = sorted(selected)
            n_corrupt = sum(1 for cid in selected if cid in corrupted_clients)
            raw_oracle = [-0.8 if cid in corrupted_clients else 0.0 for cid in selected]

            # Aligned
            res_a = make_results(selected, raw_oracle, hists_all, sizes_all)
            w_a, info_a = ctrl_aligned.compute_weights(res_a, round_idx=rnd)
            w_a = np.asarray(w_a, dtype=float)

            # Deterministic shuffled
            res_s = make_results(selected, raw_oracle, hists_all, sizes_all)
            w_s, info_s = ctrl_shuffled.compute_weights(res_s, round_idx=rnd)
            w_s = np.asarray(w_s, dtype=float)

            corrupt_mask = np.array([cid in corrupted_clients for cid in selected])

            # Exact permutation expectation over distinct score assignments
            expected_mass = None
            if n_corrupt > 0:
                masses = []
                seen = set()
                base = [0.0] * (5 - n_corrupt) + [-0.8] * n_corrupt
                for perm in permutations(base):
                    if perm in seen:
                        continue
                    seen.add(perm)
                    res_p = make_results(selected, perm, hists_all, sizes_all)
                    w_p, _ = ctrl_aligned.compute_weights(res_p, round_idx=rnd)
                    w_p = np.asarray(w_p, dtype=float)
                    p_mask = np.array([s == -0.8 for s in perm])
                    masses.append(float(w_p[p_mask].sum()))
                expected_mass = float(np.mean(masses))

            fedavg = sizes_all[selected] / sizes_all[selected].sum()
            uniform = np.ones(5) / 5.0

            records.append({
                "seed": seed,
                "round": rnd,
                "selected": selected,
                "n_corrupt": n_corrupt,
                "aligned_mass": float(w_a[corrupt_mask].sum()),
                "shuffled_mass": float(w_s[corrupt_mask].sum()),
                "expected_mass": expected_mass,
                "fedavg_mass": float(fedavg[corrupt_mask].sum()),
                "uniform_mass": float(uniform[corrupt_mask].sum()),
                "aligned_minus_expected": (float(w_a[corrupt_mask].sum()) - expected_mass
                                            if expected_mass is not None else 0.0),
                "aligned_minus_shuffled": float(w_a[corrupt_mask].sum() - w_s[corrupt_mask].sum()),
                "l1_aligned_vs_shuffled": float(np.sum(np.abs(w_a - w_s))),
                "effective_clients": info_a["effective_clients"],
                "kappa_satisfied": info_a["effective_clients"] >= 3.0 - 1e-6,
                "temperature": info_a["temperature_used"],
                "solver_calls": info_a.get("solver_calls"),
            })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "nojs_nonrobust_alpha0.1.csv", index=False)

    agg = df.groupby("n_corrupt").agg({
        "aligned_mass": "mean",
        "shuffled_mass": "mean",
        "expected_mass": "mean",
        "fedavg_mass": "mean",
        "uniform_mass": "mean",
        "aligned_minus_expected": "mean",
        "aligned_minus_shuffled": "mean",
        "l1_aligned_vs_shuffled": "mean",
        "effective_clients": "mean",
        "kappa_satisfied": "mean",
        "temperature": "mean",
        "solver_calls": "mean",
    }).reset_index()
    agg.to_csv(RESULTS_DIR / "nojs_nonrobust_alpha0.1_summary.csv", index=False)

    print(f"[NoJS non-robust alpha={alpha}] wrote {len(df)} cohort records")
    print(agg.to_string(index=False))

    mixed = df[df["n_corrupt"] > 0]
    verdict = {
        "mean_aligned_mass": float(mixed["aligned_mass"].mean()),
        "mean_expected_mass": float(mixed["expected_mass"].mean()),
        "mean_reduction_vs_expected": float((mixed["aligned_mass"] - mixed["expected_mass"]).mean()),
        "mean_l1_aligned_vs_shuffled": float(mixed["l1_aligned_vs_shuffled"].mean()),
        "kappa_satisfied_fraction": float(mixed["kappa_satisfied"].mean()),
        "mean_effective_clients": float(mixed["effective_clients"].mean()),
    }
    (RESULTS_DIR / "nojs_nonrobust_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    run_audit(alpha=0.1)
