"""CPU-only tests for the MLJ-P0 positive-control components."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SNAPSHOT_ROOT = (
    ROOT / "historical_snapshot"
    if (ROOT / "historical_snapshot").exists()
    else ROOT.parent
    / "InfoOpt_Code_For_Journal"
    / "source_snapshots"
    / "ITSL_stage3_prep_20261012_v2"
    / "snapshot"
)
sys.path.insert(0, str(SNAPSHOT_ROOT))

from data.partition import class_histogram, dirichlet_partition  # noqa: E402
from mlj_p0_cpu_audit import (  # noqa: E402
    compute_oracle_weights,
    corrupt_labels,
    generate_schedule,
    select_corrupted_clients,
    stage1_shuffle_seed,
)


def test_dirichlet_partition_matches_study3():
    labels = np.arange(100).repeat(50)
    seed = 100
    partition = dirichlet_partition(labels, num_clients=10, alpha=0.3,
                                    min_samples_per_client=50, seed=seed)
    assert set(partition.keys()) == set(range(10))
    flat = [i for v in partition.values() for i in v]
    assert len(flat) == len(set(flat)) == 5000
    assert all(len(v) >= 50 for v in partition.values())


def test_schedule_deterministic_from_seed():
    s1 = generate_schedule(10, 5, 100, seed=100)
    s2 = generate_schedule(10, 5, 100, seed=100)
    s3 = generate_schedule(10, 5, 100, seed=101)
    assert s1 == s2
    assert s1 != s3
    assert all(len(cohort) == 5 and len(set(cohort)) == 5 for cohort in s1)


def test_corrupted_client_selection_deterministic():
    c1 = select_corrupted_clients(10, 3, seed=100)
    c2 = select_corrupted_clients(10, 3, seed=100)
    c3 = select_corrupted_clients(10, 3, seed=101)
    assert c1 == c2
    assert len(c1) == 3
    assert set(c1).issubset(set(range(10)))
    assert c1 != c3


def test_corrupt_labels_preserves_test_labels():
    labels = np.arange(100).repeat(50)
    indices = list(range(0, 500))
    corrupted, rate, positions, repl = corrupt_labels(
        labels, indices, corruption_rate=0.8, seed=42, num_classes=100)
    assert corrupted is not labels
    assert len(corrupted) == len(labels)
    # All positions outside indices remain unchanged
    outside = [i for i in range(len(labels)) if i not in positions]
    assert np.array_equal(corrupted[outside], labels[outside])
    # Realized rate close to requested
    assert abs(rate - 0.8) < 0.05
    # Corrupted positions differ from original
    for pos in positions:
        assert corrupted[pos] != labels[pos]


def _make_realistic_cohort(
    n_clean: int = 3,
    n_corrupt: int = 2,
    num_classes: int = 100,
    samples_per_client: int = 500,
    alpha_dirichlet: float = 0.3,
    seed: int = 12345,
):
    """Return histograms, sizes, and global histogram for a realistic cohort.

    Each client distribution is a Dirichlet-multinomial over all classes,
    which is what the JS constraint sees in the real CIFAR-100 partition.
    """
    rng = np.random.default_rng(seed)
    hists = []
    for _ in range(n_clean + n_corrupt):
        p = rng.dirichlet(np.full(num_classes, alpha_dirichlet))
        hists.append(rng.multinomial(samples_per_client, p).astype(float))
    hists = np.stack(hists)
    sizes = np.full(n_clean + n_corrupt, samples_per_client, dtype=float)
    global_hist = hists.sum(axis=0) + 1.0  # small smoothing like the controller
    return hists, sizes, global_hist


def test_oracle_weight_basic_clean_cohort():
    # Five clean clients: oracle scores all 0 -> uniform weights
    scores = np.zeros(5)
    sizes = np.ones(5)
    hists, _, global_hist = _make_realistic_cohort(
        n_clean=5, n_corrupt=0, seed=100)
    w, info = compute_oracle_weights(scores, sizes, hists, global_hist)
    assert np.allclose(w, np.ones(5) / 5, atol=1e-6)
    assert info["effective_clients"] == pytest.approx(5.0, abs=1e-4)


def test_oracle_weight_mad_zero_robust():
    # Mixed cohort with MAD=0: robust normalization divides by epsilon.
    # The solver must still return finite, normalized weights and the shuffle
    # must produce a deterministic reassignment of the score multiset.
    scores = np.array([0.0, 0.0, 0.0, -0.8, -0.8])
    sizes = np.ones(5)
    hists, _, global_hist = _make_realistic_cohort(
        n_clean=3, n_corrupt=2, seed=101)
    w_aligned, _ = compute_oracle_weights(scores, sizes, hists, global_hist,
                                          robust=True, shuffle=False,
                                          seed=100, round_idx=1)
    w_shuffled, _ = compute_oracle_weights(scores, sizes, hists, global_hist,
                                           robust=True, shuffle=True,
                                           seed=100, round_idx=1)
    assert np.isfinite(w_aligned).all()
    assert abs(w_aligned.sum() - 1.0) < 1e-6
    assert abs(w_shuffled.sum() - 1.0) < 1e-6
    # Deterministic: same call twice must be identical
    w_shuffled2, _ = compute_oracle_weights(scores, sizes, hists, global_hist,
                                            robust=True, shuffle=True,
                                            seed=100, round_idx=1)
    assert np.allclose(w_shuffled, w_shuffled2, atol=1e-12)


def test_oracle_weight_shuffled_preserves_score_multiset():
    scores = np.array([0.0, 0.0, 0.0, -0.8, -0.8])
    sizes = np.ones(5)
    hists, _, global_hist = _make_realistic_cohort(
        n_clean=3, n_corrupt=2, seed=102)
    w_aligned, info_a = compute_oracle_weights(scores, sizes, hists, global_hist,
                                               shuffle=False, seed=100, round_idx=1)
    w_shuffled, info_s = compute_oracle_weights(scores, sizes, hists, global_hist,
                                                shuffle=True, seed=100, round_idx=1)
    # The raw oracle score multiset is preserved by the permutation
    assert sorted(info_s["raw_oracle"]) == sorted(info_a["raw_oracle"])
    assert sorted(info_s["shuffle_record"]["original_scores"]) == \
        sorted(info_s["shuffle_record"]["solver_scores"])
    assert abs(sum(w_aligned) - 1.0) < 1e-6
    assert abs(sum(w_shuffled) - 1.0) < 1e-6


def test_oracle_weight_constraint_feasible():
    scores = np.array([0.0, 0.0, 0.0, -0.8, -0.8])
    sizes = np.ones(5)
    hists, _, global_hist = _make_realistic_cohort(
        n_clean=3, n_corrupt=2, seed=103)
    w, info = compute_oracle_weights(scores, sizes, hists, global_hist,
                                     js_mode="constraint", epsilon_rel=1.0)
    assert info["feasible"]
    assert info["js"] <= info.get("epsilon", 1.0) * (1.0 + 1e-3)


def test_stage1_shuffle_seed_deterministic():
    assert stage1_shuffle_seed("CIFAR100", 100, 1) == stage1_shuffle_seed("CIFAR100", 100, 1)
    assert stage1_shuffle_seed("CIFAR100", 100, 1) != stage1_shuffle_seed("CIFAR100", 100, 2)


def test_oracle_weight_no_clip_extreme_scores():
    # One clean + four corrupted with MAD=0: the clean client should receive
    # the largest share of the weight, and weights must remain normalized.
    scores = np.array([0.0, -0.8, -0.8, -0.8, -0.8])
    sizes = np.array([100, 100, 100, 100, 100])
    hists, _, global_hist = _make_realistic_cohort(
        n_clean=1, n_corrupt=4, seed=104)
    w, info = compute_oracle_weights(scores, sizes, hists, global_hist,
                                     robust=True, score_clip=None)
    assert np.isfinite(w).all()
    assert abs(w.sum() - 1.0) < 1e-6
    assert w[0] == max(w)


def test_oracle_weight_kappa_constraint_attempted():
    # Two clean + three corrupted: the kappa effective-client floor may or may
    # not be satisfiable depending on the histograms; weights must remain valid.
    scores = np.array([0.0, 0.0, -0.8, -0.8, -0.8])
    sizes = np.ones(5)
    hists, _, global_hist = _make_realistic_cohort(
        n_clean=2, n_corrupt=3, seed=105)
    w, info = compute_oracle_weights(scores, sizes, hists, global_hist,
                                     kappa_frac=0.6)
    assert np.isfinite(w).all()
    assert abs(w.sum() - 1.0) < 1e-6
    assert 1.0 <= info["effective_clients"] <= 5.0


@pytest.mark.parametrize("n_corrupt", [0, 1, 2, 3])
def test_oracle_mad_zero_for_all_mixed_compositions(n_corrupt):
    # For binary scores {0, -0.8} in a 5-client cohort, the median absolute
    # deviation is zero for every composition. This test locks in the corrected
    # arithmetic and prevents regressions.
    n_clean = 5 - n_corrupt
    scores = np.array([0.0] * n_clean + [-0.8] * n_corrupt)
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    assert mad == 0.0
    normalized = (scores - med) / (mad + 1e-8)
    if n_corrupt > 0:
        # Mixed cohorts produce extreme normalized magnitudes
        extreme = 80_000_000.0
        assert np.allclose(np.abs(normalized)[normalized != 0.0], extreme)
    else:
        # Clean cohort: all scores are zero
        assert np.allclose(normalized, 0.0)


@pytest.mark.parametrize("n_corrupt", [1, 2, 3])
def test_oracle_weight_nojs_nonrobust_downweights_corrupted(n_corrupt):
    # Approved MLJ-P0.1 correction: InfoOpt-NoJS with standard z-score
    # (robust_scale=False) avoids the MAD-zero pathology and satisfies kappa.
    n_clean = 5 - n_corrupt
    scores = np.array([0.0] * n_clean + [-0.8] * n_corrupt)
    sizes = np.ones(5)
    hists, _, global_hist = _make_realistic_cohort(
        n_clean=n_clean, n_corrupt=n_corrupt, seed=200 + n_corrupt)
    w_aligned, info_a = compute_oracle_weights(
        scores, sizes, hists, global_hist,
        robust=False, js_mode="off", kappa_frac=0.6)
    corrupt_mask = np.array([False] * n_clean + [True] * n_corrupt)
    aligned_mass = w_aligned[corrupt_mask].sum()
    # Uniform baseline would give n_corrupt/5 to corrupted clients
    assert aligned_mass < n_corrupt / 5.0
    # Effective clients should satisfy the kappa floor (>= 3.0 for kappa_frac=0.6)
    assert info_a["effective_clients"] >= 3.0 - 1e-6
    assert np.isfinite(w_aligned).all()
    assert abs(w_aligned.sum() - 1.0) < 1e-6


@pytest.mark.parametrize("n_corrupt", [1, 2])
def test_oracle_weight_robust_nojs_downweights_when_kappa_satisfied(n_corrupt):
    # Robust NoJS can work for 1- and 2-corrupt cohorts where kappa is still
    # satisfiable, but it fails for 3-corrupt cohorts because MAD=0 creates
    # effectively infinite score separation.
    n_clean = 5 - n_corrupt
    scores = np.array([0.0] * n_clean + [-0.8] * n_corrupt)
    sizes = np.ones(5)
    hists, _, global_hist = _make_realistic_cohort(
        n_clean=n_clean, n_corrupt=n_corrupt, seed=200 + n_corrupt)
    w_aligned, info_a = compute_oracle_weights(
        scores, sizes, hists, global_hist,
        robust=True, js_mode="off", kappa_frac=0.6)
    corrupt_mask = np.array([False] * n_clean + [True] * n_corrupt)
    aligned_mass = w_aligned[corrupt_mask].sum()
    assert aligned_mass < n_corrupt / 5.0
    assert np.isfinite(w_aligned).all()
    assert abs(w_aligned.sum() - 1.0) < 1e-6


def test_oracle_weight_robust_nojs_three_corrupt_violates_kappa():
    # This is the historical robust normalization failure that motivated the
    # MLJ-P0.1 correction. With three corrupted clients, MAD=0 and the resulting
    # extreme normalized scores make the kappa=0.6 floor infeasible.
    n_corrupt = 3
    n_clean = 2
    scores = np.array([0.0] * n_clean + [-0.8] * n_corrupt)
    sizes = np.ones(5)
    hists, _, global_hist = _make_realistic_cohort(
        n_clean=n_clean, n_corrupt=n_corrupt, seed=200 + n_corrupt)
    w_aligned, info_a = compute_oracle_weights(
        scores, sizes, hists, global_hist,
        robust=True, js_mode="off", kappa_frac=0.6)
    assert np.isfinite(w_aligned).all()
    assert abs(w_aligned.sum() - 1.0) < 1e-6
    assert info_a["effective_clients"] < 3.0 - 1e-6
