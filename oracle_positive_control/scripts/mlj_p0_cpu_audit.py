#!/usr/bin/env python3
"""CPU-only feasibility audit for the MLJ positive-control experiment.

This script:
  1. Loads CIFAR-100 training labels.
  2. Generates Dirichlet partitions for alpha=0.3 and alpha=0.1 (for
     comparison) using the exact Study-3 partition function with the run
     seed as partition seed.
  3. Generates participation schedules using random.Random(seed).sample,
     matching Study 3.
  4. Selects three corrupted clients deterministically per seed.
  5. Computes oracle scores (clean=0.0, corrupted=-0.8).
  6. Replicates the InfoOpt weighting pipeline on these oracle scores,
     including robust normalization, optional clipping, softmax solve,
     JS constraint, kappa effective-client constraint, temperature search,
     and sample-size FedAvg mixing.
  7. Evaluates aligned-vs-shuffled weight differences for every cohort
     composition (0/1/2/3 corrupted clients in the selected five).
  8. Writes machine-readable results to results/.

No neural-network training is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.optimize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATASET_CACHE = ROOT / "_dataset_cache"

sys.path.insert(0, str(ROOT.parent / "InfoOpt_Code_For_Journal" /
                        "source_snapshots" / "ITSL_stage3_prep_20261012_v2" /
                        "snapshot"))

from data.partition import class_histogram, dirichlet_partition  # noqa: E402

# ---------------------------------------------------------------------------
# Replicated InfoOpt solver helpers (from methods/infoopt.py)
# ---------------------------------------------------------------------------


def _entropy_of(w: np.ndarray) -> float:
    w = np.clip(w, 1e-12, 1.0)
    return float(-np.sum(w * np.log(w)))


def _js_and_grad(w: np.ndarray, P: np.ndarray, q: np.ndarray):
    p_mix = np.clip(w @ P, 1e-12, None)
    p_mix = p_mix / p_mix.sum()
    m = 0.5 * (p_mix + q)
    js = 0.5 * float(np.sum(p_mix * np.log(p_mix / m))) + \
         0.5 * float(np.sum(q * np.log(q / m)))
    grad = 0.5 * (P @ np.log(p_mix / m))
    return js, grad


def _solve_entropic_weights(
    scores: np.ndarray,
    tau: float,
    P: np.ndarray | None = None,
    q: np.ndarray | None = None,
    epsilon: float | None = None,
) -> tuple[np.ndarray, dict]:
    s = np.asarray(scores, dtype=float)
    C = s.size

    def resp(logits: np.ndarray) -> np.ndarray:
        z = (logits - logits.max()) / max(tau, 1e-8)
        e = np.exp(z)
        return e / e.sum()

    w_free = resp(s)
    if P is None or q is None or epsilon is None:
        return w_free, {"js": float("nan"), "feasible": True,
                        "fallback_used": False}

    js_free, _ = _js_and_grad(w_free, P, q)
    if js_free <= epsilon:
        return w_free, {"js": float(js_free), "feasible": True,
                        "fallback_used": False}

    lo = 1e-9

    def neg_obj(w):
        wc = np.clip(w, lo, None)
        return -(float(w @ s) + tau * float(-np.sum(wc * np.log(wc))))

    def neg_obj_grad(w):
        wc = np.clip(w, lo, None)
        return -(s - tau * (np.log(wc) + 1.0))

    def cons_val(w):
        js, _ = _js_and_grad(np.clip(w, lo, None), P, q)
        return epsilon - js

    def cons_grad(w):
        _, g = _js_and_grad(np.clip(w, lo, None), P, q)
        return -g

    uniform = np.ones(C) / C
    x0 = 0.5 * uniform + 0.5 * w_free
    res = scipy.optimize.minimize(
        neg_obj, x0, jac=neg_obj_grad, method="SLSQP",
        bounds=[(lo, 1.0)] * C,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones(C)},
            {"type": "ineq", "fun": cons_val, "jac": cons_grad},
        ],
        options={"maxiter": 300, "ftol": 1e-12},
    )
    w = np.clip(res.x, lo, None)
    w = w / w.sum()
    js_w, _ = _js_and_grad(w, P, q)

    fallback_used = False
    fallback_reason = None
    if (not res.success) or js_w > epsilon * (1.0 + 1e-4):
        fallback_used = True
        fallback_reason = ("slsqp_failed" if not res.success
                           else "constraint_violated_after_solve")
        g_lo, g_hi = 0.0, 1.0
        for _ in range(60):
            gamma = 0.5 * (g_lo + g_hi)
            w_try = (1.0 - gamma) * w_free + gamma * uniform
            js_try, _ = _js_and_grad(w_try, P, q)
            if js_try <= epsilon:
                g_hi = gamma
            else:
                g_lo = gamma
        w = (1.0 - g_hi) * w_free + g_hi * uniform
        w = w / w.sum()
        js_w, _ = _js_and_grad(w, P, q)

    return w, {"js": float(js_w), "feasible": bool(js_w <= epsilon * (1.0 + 1e-3)),
               "fallback_used": fallback_used,
               "fallback_reason": fallback_reason}


def _z(x: np.ndarray, robust: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if robust:
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return (x - med) / (mad + 1e-8)
    return (x - x.mean()) / (x.std() + 1e-8)


def stage1_shuffle_seed(dataset: str, seed: int, round_idx: int) -> int:
    key = f"stage1-score-shuffle-v1|{dataset}|{seed}|{round_idx}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def compute_oracle_weights(
    oracle_scores: np.ndarray,
    client_sizes: np.ndarray,
    client_histograms: np.ndarray,
    global_hist: np.ndarray,
    robust: bool = True,
    score_clip: float | None = None,
    temperature: float = 1.0,
    kappa_frac: float = 0.6,
    epsilon_rel: float = 1.0,
    js_mode: str = "constraint",
    fedavg_mix: float = 0.0,
    min_weight: float = 0.0,
    min_weight_fraction: float = 0.0,
    shuffle: bool = False,
    dataset_name: str = "CIFAR100",
    seed: int = 100,
    round_idx: int = 1,
) -> tuple[np.ndarray, dict]:
    """Run the InfoOpt weighting pipeline on raw oracle scores.

    Returns aligned weights and diagnostics.  If shuffle=True, the oracle
    scores are permuted using the Study-1 deterministic shuffle seed before
    weighting, producing the shuffled-score control.
    """
    C = len(oracle_scores)
    score = _z(oracle_scores.copy(), robust)

    if score_clip is not None:
        clip = abs(float(score_clip))
        score = np.clip(score, -clip, clip)

    shuffle_record = None
    if shuffle:
        seed_int = stage1_shuffle_seed(dataset_name, seed, round_idx)
        rng = np.random.default_rng(seed_int)
        perm = rng.permutation(C)
        original = score.copy()
        score = score[perm]
        shuffle_record = {
            "seed_int": seed_int,
            "permutation": perm.tolist(),
            "original_scores": original.tolist(),
            "solver_scores": score.tolist(),
        }

    constrained = js_mode == "constraint"
    epsilon = None
    P = None
    q = None
    if constrained:
        P = client_histograms / client_histograms.sum(axis=1, keepdims=True)
        q = global_hist / global_hist.sum()
        uniform = np.ones(C) / C
        js_uniform, _ = _js_and_grad(uniform, P, q)
        epsilon = max(js_uniform * epsilon_rel, 1e-8)

    tau = max(float(temperature), 1e-6)
    weights, dual_info = _solve_entropic_weights(score, tau, P, q, epsilon)

    if kappa_frac > 0.0:
        kappa = min(max(kappa_frac, 1.0 / C), 1.0) * C
        solver_calls = 1
        for _ in range(40):
            if np.exp(_entropy_of(weights)) >= kappa:
                break
            tau *= 1.3
            solver_calls += 1
            weights, dual_info = _solve_entropic_weights(score, tau, P, q, epsilon)
    else:
        solver_calls = 1

    if fedavg_mix > 0.0:
        ns = client_sizes.astype(float)
        ns = ns / ns.sum() if ns.sum() > 0 else np.ones(C) / C
        weights = fedavg_mix * ns + (1.0 - fedavg_mix) * weights
        weights /= weights.sum()

    if min_weight_fraction > 0.0:
        min_w = max(min_weight, min_weight_fraction / C)
    else:
        min_w = min_weight
    min_w = min(max(min_w, 0.0), (1.0 - 1e-8) / C)
    if min_w > 0.0:
        weights = np.maximum(weights, min_w)
        weights /= weights.sum()

    info = {
        "raw_oracle": oracle_scores.tolist(),
        "normalized_score": score.tolist(),
        "weights": weights.tolist(),
        "temperature_used": float(tau),
        "js": dual_info.get("js", float("nan")),
        "feasible": dual_info.get("feasible", True),
        "fallback_used": dual_info.get("fallback_used", False),
        "fallback_reason": dual_info.get("fallback_reason"),
        "solver_calls": solver_calls,
        "effective_clients": float(np.exp(_entropy_of(weights))),
        "shuffle_record": shuffle_record,
    }
    return weights, info


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_cifar100_labels(cache_dir: Path = DATASET_CACHE) -> np.ndarray:
    try:
        import torchvision
        train = torchvision.datasets.CIFAR100(root=cache_dir, train=True,
                                              download=False)
        return np.array(train.targets)
    except Exception as e:
        raise RuntimeError(
            f"CIFAR-100 not found in {cache_dir}. Download it first."
        ) from e


def generate_schedule(num_clients: int, clients_per_round: int,
                      rounds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [rng.sample(range(num_clients), clients_per_round)
            for _ in range(rounds)]


def select_corrupted_clients(num_clients: int, num_corrupted: int,
                             seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(num_clients, num_corrupted, replace=False).tolist())


def corrupt_labels(labels: np.ndarray, indices: Sequence[int],
                   corruption_rate: float, seed: int,
                   num_classes: int = 100,
                   symmetric_other_classes: bool = True) -> np.ndarray:
    """Symmetric label noise on a subset of training indices.

    If symmetric_other_classes=True (default), a corrupted label is sampled
    uniformly from the *other* 99 classes.  If False, it is sampled from all
    100 classes, which can include the original label.
    """
    rng = np.random.default_rng(seed)
    corrupted_labels = labels.copy()
    n_corrupt = int(round(len(indices) * corruption_rate))
    corrupt_idx = rng.choice(len(indices), n_corrupt, replace=False)
    sample_positions = np.asarray(indices)[corrupt_idx]
    original = labels[sample_positions]
    replacements = np.empty_like(original)
    for i, y in enumerate(original):
        if symmetric_other_classes:
            candidates = [c for c in range(num_classes) if c != y]
            replacements[i] = rng.choice(candidates)
        else:
            replacements[i] = rng.choice(num_classes)
    corrupted_labels[sample_positions] = replacements
    actual_rate = len(corrupt_idx) / len(indices)
    return corrupted_labels, actual_rate, sample_positions.tolist(), replacements.tolist()


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def run_audit(alpha: float = 0.3, seeds: Sequence[int] = (100, 101, 102, 103, 104),
              rounds: int = 100, num_clients: int = 10,
              clients_per_round: int = 5, num_corrupted: int = 3,
              corruption_rate: float = 0.8):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_cifar100_labels()
    num_classes = int(labels.max()) + 1
    global_hist = np.bincount(labels, minlength=num_classes).astype(float)

    records = []
    cohort_summary = []

    for seed in seeds:
        # Partition and schedules use the run seed, matching Study 3.
        client_indices = dirichlet_partition(
            labels, num_clients=num_clients, alpha=alpha,
            min_samples_per_client=50, seed=seed)
        schedule = generate_schedule(num_clients, clients_per_round, rounds, seed)
        corrupted_clients = select_corrupted_clients(num_clients, num_corrupted, seed)

        # Compute client class histograms from ORIGINAL (uncorrupted) labels,
        # because the server histograms used by InfoOpt are computed from the
        # training labels seen by the server.  In the positive control the
        # oracle knows corruption, but the class-histogram pipeline should use
        # the same semantics as the aligned/shuffled arms.
        hists = np.stack([
            class_histogram(labels, client_indices[cid], num_classes)
            for cid in range(num_clients)
        ]).astype(float)
        sizes = np.array([len(client_indices[cid]) for cid in range(num_clients)])

        # Corrupt local labels for corrupted clients (record manifests)
        for cid in corrupted_clients:
            _, actual_rate, positions, repl = corrupt_labels(
                labels, client_indices[cid], corruption_rate,
                seed=seed * 10000 + cid, num_classes=num_classes)
            manifest = {
                "seed": seed, "client": cid,
                "corruption_rate_requested": corruption_rate,
                "corruption_rate_actual": actual_rate,
                "num_corrupted": len(positions),
                "positions": positions,
                "replacements": repl,
            }
            manifest_path = DATA_DIR / f"corruption_manifest_seed{seed}_client{cid}.json"
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

        for rnd, selected in enumerate(schedule, start=1):
            selected = sorted(selected)
            n_corrupt_in_cohort = sum(1 for cid in selected if cid in corrupted_clients)

            raw_oracle = np.array([
                -corruption_rate if cid in corrupted_clients else 0.0
                for cid in selected
            ], dtype=float)
            cohort_hists = hists[selected]
            cohort_sizes = sizes[selected]

            # Aligned oracle
            w_aligned, info_aligned = compute_oracle_weights(
                raw_oracle, cohort_sizes, cohort_hists, global_hist,
                robust=True, score_clip=None, temperature=1.0,
                kappa_frac=0.6, epsilon_rel=1.0, js_mode="constraint",
                fedavg_mix=0.0, min_weight=0.0, min_weight_fraction=0.0,
                shuffle=False, dataset_name="CIFAR100", seed=seed,
                round_idx=rnd)

            # Shuffled oracle
            w_shuffled, info_shuffled = compute_oracle_weights(
                raw_oracle, cohort_sizes, cohort_hists, global_hist,
                robust=True, score_clip=None, temperature=1.0,
                kappa_frac=0.6, epsilon_rel=1.0, js_mode="constraint",
                fedavg_mix=0.0, min_weight=0.0, min_weight_fraction=0.0,
                shuffle=True, dataset_name="CIFAR100", seed=seed,
                round_idx=rnd)

            # FedAvg (sample-size) and uniform baselines
            w_fedavg = cohort_sizes / cohort_sizes.sum()
            w_uniform = np.ones(clients_per_round) / clients_per_round

            corrupt_mask = np.array([
                cid in corrupted_clients for cid in selected], dtype=bool)

            def corrupt_mass(w):
                return float(np.sum(np.asarray(w)[corrupt_mask]))

            records.append({
                "seed": seed,
                "round": rnd,
                "selected": selected,
                "n_corrupt": n_corrupt_in_cohort,
                "corrupted_clients": corrupted_clients,
                "aligned_weight_corrupt": corrupt_mass(w_aligned),
                "shuffled_weight_corrupt": corrupt_mass(w_shuffled),
                "fedavg_weight_corrupt": corrupt_mass(w_fedavg),
                "uniform_weight_corrupt": corrupt_mass(w_uniform),
                "aligned_l1_vs_shuffled": float(np.sum(np.abs(np.asarray(w_aligned) - np.asarray(w_shuffled)))),
                "aligned_l1_vs_fedavg": float(np.sum(np.abs(np.asarray(w_aligned) - w_fedavg))),
                "aligned_l1_vs_uniform": float(np.sum(np.abs(np.asarray(w_aligned) - w_uniform))),
                "aligned_effective_clients": info_aligned["effective_clients"],
                "shuffled_effective_clients": info_shuffled["effective_clients"],
                "aligned_fallback": info_aligned["fallback_used"],
                "shuffled_fallback": info_shuffled["fallback_used"],
                "aligned_js": info_aligned["js"],
                "shuffled_js": info_shuffled["js"],
                "aligned_temperature": info_aligned["temperature_used"],
                "shuffled_temperature": info_shuffled["temperature_used"],
                "aligned_solver_calls": info_aligned["solver_calls"],
                "shuffled_solver_calls": info_shuffled["solver_calls"],
            })

        # Per-seed summary
        for n_c in range(num_clients + 1):
            subset = [r for r in records if r["seed"] == seed and r["n_corrupt"] == n_c]
            if not subset:
                continue
            cohort_summary.append({
                "seed": seed,
                "n_corrupt_in_cohort": n_c,
                "n_cohorts": len(subset),
                "aligned_corrupt_mass_mean": np.mean([r["aligned_weight_corrupt"] for r in subset]),
                "shuffled_corrupt_mass_mean": np.mean([r["shuffled_weight_corrupt"] for r in subset]),
                "fedavg_corrupt_mass_mean": np.mean([r["fedavg_weight_corrupt"] for r in subset]),
                "uniform_corrupt_mass_mean": np.mean([r["uniform_weight_corrupt"] for r in subset]),
                "aligned_minus_shuffled_mean": np.mean([
                    r["aligned_weight_corrupt"] - r["shuffled_weight_corrupt"]
                    for r in subset]),
                "aligned_l1_vs_shuffled_mean": np.mean([r["aligned_l1_vs_shuffled"] for r in subset]),
                "aligned_fallback_fraction": np.mean([r["aligned_fallback"] for r in subset]),
                "shuffled_fallback_fraction": np.mean([r["shuffled_fallback"] for r in subset]),
            })

    df = pd.DataFrame(records)
    summary_df = pd.DataFrame(cohort_summary)
    df.to_csv(RESULTS_DIR / f"mlj_p0_oracle_weights_alpha{alpha}.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / f"mlj_p0_cohort_summary_alpha{alpha}.csv", index=False)

    # Aggregate across seeds
    agg = summary_df.groupby("n_corrupt_in_cohort").agg({
        "n_cohorts": "sum",
        "aligned_corrupt_mass_mean": "mean",
        "shuffled_corrupt_mass_mean": "mean",
        "fedavg_corrupt_mass_mean": "mean",
        "uniform_corrupt_mass_mean": "mean",
        "aligned_minus_shuffled_mean": "mean",
        "aligned_l1_vs_shuffled_mean": "mean",
        "aligned_fallback_fraction": "mean",
        "shuffled_fallback_fraction": "mean",
    }).reset_index()
    agg.to_csv(RESULTS_DIR / f"mlj_p0_aggregate_alpha{alpha}.csv", index=False)

    print(f"[alpha={alpha}] wrote {len(df)} cohort records")
    print(agg.to_string(index=False))
    return df, summary_df, agg


if __name__ == "__main__":
    # Primary audit at proposed alpha=0.3
    run_audit(alpha=0.3)
    # Comparison audit at Study-3 alpha=0.1
    run_audit(alpha=0.1)
