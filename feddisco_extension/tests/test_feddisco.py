"""CPU-only validation tests for the official-source-aligned FedDisco controller.

No model training or GPU use.
"""
from __future__ import annotations

import copy
import hashlib
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Allow importing both the pilot code and the official reference repo.
PILOT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = Path(tempfile.gettempdir()).resolve() / "FedDisco_official"
sys.path.insert(0, str(PILOT_ROOT / "snapshot"))
sys.path.insert(0, str(OFFICIAL_ROOT))

from data.partition import class_histogram, dirichlet_partition  # noqa: E402
from methods.feddisco import FedDiscoController, _shuffle_seed  # noqa: E402
from models.build import build_model  # noqa: E402
from utils.aggregation import average_states  # noqa: E402
from utils.seed import set_seed  # noqa: E402

from disco import (  # noqa: E402  official reference
    disco_weight_adjusting,
    get_distribution_difference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dirichlet_partition(num_clients: int = 10, num_classes: int = 100, alpha: float = 0.3, seed: int = 100):
    rng = np.random.default_rng(seed)
    labels = np.arange(num_classes).repeat(50)
    rng.shuffle(labels)
    return dirichlet_partition(labels, num_clients=num_clients, alpha=alpha, min_samples_per_client=50, seed=seed), labels


def _make_controller_from_partition(client_indices: dict, labels: np.ndarray, seed: int = 100, shuffle: bool = False):
    num_classes = int(labels.max()) + 1
    hists = {cid: class_histogram(labels, idxs, num_classes) for cid, idxs in client_indices.items()}
    sizes = {cid: len(idxs) for cid, idxs in client_indices.items()}
    ctrl = FedDiscoController(
        num_classes=num_classes,
        a=0.5,
        b=0.1,
        discrepancy_metric="kl",
        dataset_name="CIFAR100",
        seed=seed,
    )
    ctrl.precompute_weights(hists, sizes)
    return ctrl


def _official_weights(client_counts, selected, sizes, a=0.5, b=0.1, metric="kl"):
    """Compute weights using the official repository's exact functions."""
    target = np.ones(client_counts.shape[1]) / client_counts.shape[1]
    diffs = get_distribution_difference(client_counts, selected, metric=metric, hypo_distribution=target)
    total = sum(sizes[c] for c in selected)
    old_weights = np.array([sizes[c] / total for c in selected], dtype=float)
    # Official code assumes at least one positive raw weight; for fully non-positive
    # inputs it would crash.  We guard the call by falling back to size weights.
    weight_tmp = old_weights - a * diffs + b
    if np.sum(weight_tmp > 0) == 0:
        return (old_weights / old_weights.sum()).tolist()
    return disco_weight_adjusting(old_weights, diffs, a, b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_numerical_agreement_with_official_source():
    """1. Pilot controller matches official functions on nondegenerate inputs."""
    client_indices, labels = _make_dirichlet_partition(num_clients=10, num_classes=100, alpha=0.3, seed=100)
    num_classes = 100
    counts = np.array([class_histogram(labels, client_indices[cid], num_classes) for cid in range(10)], dtype=float)
    sizes = {cid: len(client_indices[cid]) for cid in range(10)}
    selected = [0, 2, 4, 6, 8]

    ctrl = _make_controller_from_partition(client_indices, labels, seed=100)
    pilot_weights, _ = ctrl.get_weights(selected)
    official_weights = _official_weights(counts, selected, sizes)

    assert len(pilot_weights) == len(selected)
    assert np.allclose(pilot_weights, official_weights, atol=1e-9)


def test_participating_client_size_normalization():
    """2. Relative sizes sum to 1 and are computed over selected clients only."""
    client_indices, labels = _make_dirichlet_partition(num_clients=10, num_classes=100, alpha=0.5, seed=101)
    ctrl = _make_controller_from_partition(client_indices, labels, seed=101)

    selected_a = [0, 1, 2, 3, 4]
    selected_b = [5, 6, 7, 8, 9]

    weights_a, diag_a = ctrl.get_weights(selected_a)
    weights_b, diag_b = ctrl.get_weights(selected_b)

    rel_a = np.array(diag_a["relative_sizes"])
    rel_b = np.array(diag_b["relative_sizes"])

    assert np.isclose(rel_a.sum(), 1.0)
    assert np.isclose(rel_b.sum(), 1.0)
    # Sizes should differ when different clients are selected.
    assert not np.allclose(rel_a, rel_b)
    # The relative size vector should equal the selected-client size fractions.
    sizes = {cid: len(client_indices[cid]) for cid in client_indices}
    expected_a = np.array([sizes[c] for c in selected_a], dtype=float)
    expected_a /= expected_a.sum()
    assert np.allclose(rel_a, expected_a)


def test_kl_normalization_across_participants():
    """3. KL discrepancy values sum to 1 over the participating cohort."""
    client_indices, labels = _make_dirichlet_partition(num_clients=10, num_classes=100, alpha=0.3, seed=102)
    ctrl = _make_controller_from_partition(client_indices, labels, seed=102)
    selected = list(range(5))
    _, diag = ctrl.get_weights(selected)
    discrepancies = np.array(diag["discrepancies"])
    assert discrepancies.min() >= -1e-12
    assert np.isclose(discrepancies.sum(), 1.0)


def test_relu_and_fallback():
    """4. ReLU zeros negative raw weights; fallback is deterministic and logged."""
    # Build a controller manually so we can inject an extreme scenario.
    num_classes = 10
    hists = {
        0: np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        1: np.array([0, 100, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        2: np.array([0, 0, 100, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    }
    sizes = {0: 100, 1: 100, 2: 100}

    # Case 1: b is large enough that all raw weights are positive.
    ctrl = FedDiscoController(num_classes, a=0.5, b=1.0, discrepancy_metric="kl", dataset_name="TEST", seed=0)
    ctrl.precompute_weights(hists, sizes)
    weights, diag = ctrl.get_weights([0, 1, 2])
    assert all(w > 0 for w in weights)
    assert np.isclose(sum(weights), 1.0)
    assert diag["fallback_used"] == 0

    # Case 2: a is huge so all raw weights become negative -> fallback to size weights.
    ctrl2 = FedDiscoController(num_classes, a=1000.0, b=0.0, discrepancy_metric="kl", dataset_name="TEST", seed=0)
    ctrl2.precompute_weights(hists, sizes)
    weights2, diag2 = ctrl2.get_weights([0, 1, 2])
    assert diag2["fallback_used"] == 1
    assert diag2["fallback_reason"] == "all_raw_weights_nonpositive"
    assert np.allclose(weights2, [1 / 3, 1 / 3, 1 / 3])


def test_deterministic_permutation_and_discrepancy_multiset():
    """5. Shuffle uses a deterministic RNG and preserves the discrepancy multiset."""
    client_indices, labels = _make_dirichlet_partition(num_clients=10, num_classes=100, alpha=0.3, seed=103)
    ctrl = _make_controller_from_partition(client_indices, labels, seed=103)
    selected = [1, 3, 5, 7, 9]

    _, diag1 = ctrl.get_weights(selected, round_idx=1, shuffle=True)
    _, diag2 = ctrl.get_weights(selected, round_idx=1, shuffle=True)

    assert diag1["shuffle_order"] == diag2["shuffle_order"]
    assert sorted(diag1["discrepancies"]) == sorted(diag1["discrepancies_used"])
    assert np.allclose(sorted(diag1["discrepancies"]), sorted(diag2["discrepancies"]))

    # A different round should generally yield a different permutation.
    _, diag3 = ctrl.get_weights(selected, round_idx=2, shuffle=True)
    assert diag3["shuffle_order"] != diag1["shuffle_order"]


def test_unnormalized_weight_multiset_not_preserved():
    """6. Counterexample: permuting discrepancies changes the unnormalized-weight multiset."""
    # Three clients with distinct sizes and distinct discrepancies.
    num_classes = 3
    hists = {
        0: np.array([10, 0, 0], dtype=float),   # very high discrepancy
        1: np.array([4, 4, 2], dtype=float),    # lower discrepancy
        2: np.array([3, 3, 4], dtype=float),    # intermediate discrepancy
    }
    sizes = {0: 10, 1: 5, 2: 3}
    a, b = 1.0, 0.0

    ctrl = FedDiscoController(num_classes, a=a, b=b, discrepancy_metric="kl", dataset_name="TEST", seed=0)
    ctrl.precompute_weights(hists, sizes)

    # Try a few rounds until we get a non-identity permutation.
    for round_idx in range(1, 20):
        _, diag_orig = ctrl.get_weights([0, 1, 2], round_idx=round_idx, shuffle=False)
        _, diag_perm = ctrl.get_weights([0, 1, 2], round_idx=round_idx, shuffle=True)
        if diag_perm["shuffle_order"] != [0, 1, 2]:
            break

    raw_orig = np.array(diag_orig["raw_weights"])
    raw_perm = np.array(diag_perm["raw_weights"])

    # Verify clients have distinct discrepancies.
    discs = diag_orig["discrepancies"]
    assert len(set(np.round(d, 6) for d in discs)) == 3
    # Discrepancy multiset is preserved.
    assert np.allclose(sorted(diag_orig["discrepancies"]), sorted(diag_perm["discrepancies"]))
    # But raw-weight multiset is not.
    assert not np.allclose(sorted(raw_orig), sorted(raw_perm))


def test_client_id_to_weight_to_update_alignment():
    """7. Weights returned by get_weights align with the selected-client order."""
    client_indices, labels = _make_dirichlet_partition(num_clients=10, num_classes=100, alpha=0.3, seed=104)
    ctrl = _make_controller_from_partition(client_indices, labels, seed=104)
    selected = [2, 5, 8]
    weights, _ = ctrl.get_weights(selected)

    # Synthetic "updates": each client contributes a distinct constant tensor.
    updates = [torch.full((4, 4), float(cid + 1)) for cid in selected]
    agg = average_states([{"w": u} for u in updates], weights)["w"]

    # Reorder selected clients and weights together; aggregation should be identical.
    perm = [2, 0, 1]
    selected_perm = [selected[i] for i in perm]
    weights_perm = [weights[i] for i in perm]
    updates_perm = [updates[i] for i in perm]
    agg_perm = average_states([{"w": u} for u in updates_perm], weights_perm)["w"]

    assert torch.allclose(agg, agg_perm)

    # But if we permute only weights, aggregation changes.
    weights_misaligned = [weights[1], weights[0], weights[2]]
    agg_misaligned = average_states([{"w": u} for u in updates], weights_misaligned)["w"]
    assert not torch.allclose(agg, agg_misaligned)


def test_matching_partitions_schedules_and_init_hashes_across_arms():
    """8. Same seed/alpha produce identical partitions, schedules, and init hashes for all arms."""
    arms = ["fedavg", "uniform", "feddisco", "feddisco_shuffled"]
    partitions = {}
    schedules = {}
    init_hashes = {}

    for arm in arms:
        set_seed(100)
        # We do not load CIFAR-100 here; use synthetic labels matching the class count.
        labels = np.arange(100).repeat(50)
        client_indices = dirichlet_partition(labels, num_clients=10, alpha=0.3, min_samples_per_client=50, seed=100)
        partitions[arm] = {k: list(v) for k, v in client_indices.items()}

        rng = random.Random(100)
        schedules[arm] = [rng.sample(range(10), 5) for _ in range(20)]

        cfg = {
            "dataset": {"name": "CIFAR100", "num_classes": 100, "input_channels": 3},
            "model": {"name": "SmallCNN", "use_aib": False},
        }
        model = build_model(cfg)
        init_hashes[arm] = hashlib.sha256(
            b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())
        ).hexdigest()

    # All arms share the same partition, schedule, and initialization.
    assert partitions["fedavg"] == partitions["feddisco_shuffled"]
    assert schedules["fedavg"] == schedules["feddisco_shuffled"]
    assert init_hashes["fedavg"] == init_hashes["feddisco_shuffled"]


def test_rng_isolation_from_local_training():
    """9. Shuffle RNG does not share state with PyTorch/NumPy/Python global RNGs."""
    client_indices, labels = _make_dirichlet_partition(num_clients=10, num_classes=100, alpha=0.3, seed=105)
    ctrl = _make_controller_from_partition(client_indices, labels, seed=105)

    # Capture global RNG states before calling get_weights.
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    ctrl.get_weights([0, 1, 2, 3, 4], round_idx=7, shuffle=True)

    # Global states must be unchanged.
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_finite_weights_and_aggregation_outputs():
    """10. Weights are finite, nonnegative, normalized, and aggregation is finite."""
    client_indices, labels = _make_dirichlet_partition(num_clients=10, num_classes=100, alpha=0.3, seed=106)
    ctrl = _make_controller_from_partition(client_indices, labels, seed=106)
    selected = list(range(10))

    for shuffle in (False, True):
        weights, diag = ctrl.get_weights(selected, round_idx=1, shuffle=shuffle)
        w = np.array(weights)
        assert np.all(np.isfinite(w))
        assert np.all(w >= -1e-12)
        assert np.isclose(w.sum(), 1.0)

        # Aggregation of finite random states stays finite.
        states = [{"p": torch.randn(10, 10), "buf": torch.tensor(0)} for _ in selected]
        agg = average_states(states, weights)
        assert torch.all(torch.isfinite(agg["p"]))


if __name__ == "__main__":
    test_numerical_agreement_with_official_source()
    test_participating_client_size_normalization()
    test_kl_normalization_across_participants()
    test_relu_and_fallback()
    test_deterministic_permutation_and_discrepancy_multiset()
    test_unnormalized_weight_multiset_not_preserved()
    test_client_id_to_weight_to_update_alignment()
    test_matching_partitions_schedules_and_init_hashes_across_arms()
    test_rng_isolation_from_local_training()
    test_finite_weights_and_aggregation_outputs()
    print("All FedDisco CPU validation tests passed.")
