"""Validate weight reconstruction against the controller on synthetic data."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PILOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT_ROOT / "snapshot"))
sys.path.insert(0, str(PILOT_ROOT))

from data.partition import class_histogram, dirichlet_partition  # noqa: E402
from methods.feddisco import FedDiscoController  # noqa: E402
from verify_stage4 import recompute_weights  # noqa: E402


def _make_synthetic_run(num_clients: int = 10, num_classes: int = 100, alpha: float = 0.3, seed: int = 100):
    rng = np.random.default_rng(seed)
    labels = np.arange(num_classes).repeat(50)
    rng.shuffle(labels)
    client_indices = dirichlet_partition(labels, num_clients=num_clients, alpha=alpha, min_samples_per_client=50, seed=seed)
    hists = {cid: class_histogram(labels, idxs, num_classes) for cid, idxs in client_indices.items()}
    sizes = {cid: len(idxs) for cid, idxs in client_indices.items()}
    return client_indices, hists, sizes, labels


def test_reconstruction_matches_controller_original():
    client_indices, hists, sizes, labels = _make_synthetic_run()
    ctrl = FedDiscoController(
        num_classes=100,
        a=0.5,
        b=0.1,
        discrepancy_metric="kl",
        dataset_name="CIFAR100",
        seed=100,
    )
    ctrl.precompute_weights(hists, sizes)

    for round_idx in range(1, 21):
        selected = sorted(np.random.default_rng(round_idx).choice(10, 5, replace=False).tolist())
        ctrl_weights, ctrl_diag = ctrl.get_weights(selected, round_idx=round_idx, shuffle=False)
        rec_weights, rec_diag = recompute_weights(
            selected,
            hists,
            sizes,
            num_classes=100,
            a=0.5,
            b=0.1,
            shuffle=False,
            dataset_name="CIFAR100",
            seed=100,
            round_idx=round_idx,
        )
        assert np.allclose(ctrl_weights, rec_weights, atol=1e-9)
        assert np.allclose(ctrl_diag["discrepancies"], rec_diag["discrepancies"], atol=1e-9)
        assert np.allclose(ctrl_diag["weights"], rec_diag["weights"], atol=1e-9)
    print("[PASS] reconstruction matches controller for original FedDisco")


def test_reconstruction_matches_controller_shuffled():
    client_indices, hists, sizes, labels = _make_synthetic_run()
    ctrl = FedDiscoController(
        num_classes=100,
        a=0.5,
        b=0.1,
        discrepancy_metric="kl",
        dataset_name="CIFAR100",
        seed=100,
    )
    ctrl.precompute_weights(hists, sizes)

    non_identity = 0
    for round_idx in range(1, 21):
        selected = sorted(np.random.default_rng(round_idx).choice(10, 5, replace=False).tolist())
        ctrl_weights, ctrl_diag = ctrl.get_weights(selected, round_idx=round_idx, shuffle=True)
        rec_weights, rec_diag = recompute_weights(
            selected,
            hists,
            sizes,
            num_classes=100,
            a=0.5,
            b=0.1,
            shuffle=True,
            dataset_name="CIFAR100",
            seed=100,
            round_idx=round_idx,
        )
        assert np.allclose(ctrl_weights, rec_weights, atol=1e-9)
        assert ctrl_diag["shuffle_order"] == rec_diag["shuffle_order"]
        assert np.allclose(
            sorted(ctrl_diag["discrepancies"]),
            sorted(rec_diag["discrepancies"]),
            atol=1e-9,
        )
        if rec_diag["shuffle_order"] != list(range(len(selected))):
            non_identity += 1
    print(f"[PASS] reconstruction matches controller for shuffled FedDisco ({non_identity}/20 nonidentity permutations)")


def test_reconstruction_on_synthetic_run_directory():
    """Create a minimal fake run directory and verify verifier helpers run."""
    client_indices, hists, sizes, labels = _make_synthetic_run()
    ctrl = FedDiscoController(
        num_classes=100,
        a=0.5,
        b=0.1,
        discrepancy_metric="kl",
        dataset_name="CIFAR100",
        seed=100,
    )
    ctrl.precompute_weights(hists, sizes)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # We do not create the full run dir; just test reconstruction inline.
        selected = [0, 2, 4, 6, 8]
        w, d = recompute_weights(
            selected, hists, sizes, 100, 0.5, 0.1, False, "CIFAR100", 100, 1
        )
        assert len(w) == 5
        assert np.isclose(sum(w), 1.0)
        assert all(x >= -1e-12 for x in w)
    print("[PASS] reconstruction produces valid weights")


if __name__ == "__main__":
    test_reconstruction_matches_controller_original()
    test_reconstruction_matches_controller_shuffled()
    test_reconstruction_on_synthetic_run_directory()
    print("\nAll reconstruction tests passed.")
