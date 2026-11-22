"""Smoke-test that all four pilot configs parse and instantiate correctly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

PILOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT_ROOT / "snapshot"))

from data.partition import class_histogram, dirichlet_partition  # noqa: E402
from methods.feddisco import FedDiscoController  # noqa: E402
from models.build import build_model  # noqa: E402


def _load_config(name: str) -> dict:
    with open(PILOT_ROOT / "configs" / name, "r") as f:
        return yaml.safe_load(f)


def test_all_configs_parse():
    configs = [
        "feddisco_pilot_cifar100_fedavg_alpha0.3.yaml",
        "feddisco_pilot_cifar100_uniform_alpha0.3.yaml",
        "feddisco_pilot_cifar100_feddisco_alpha0.3.yaml",
        "feddisco_pilot_cifar100_feddisco_shuffled_alpha0.3.yaml",
    ]
    for cfg_name in configs:
        cfg = _load_config(cfg_name)
        assert cfg["dataset"]["name"] == "CIFAR100"
        assert cfg["dataset"]["num_classes"] == 100
        assert cfg["partition"]["num_clients"] == 10
        assert cfg["partition"]["alpha"] == 0.3
        assert cfg["training"]["clients_per_round"] == 5
        assert cfg["training"]["local_epochs"] == 2
        assert cfg["training"]["lr"] == 0.01
        assert cfg["training"]["momentum"] == 0.9
        assert cfg["training"]["batch_size"] == 64
        assert cfg["training"]["rounds"] == 100
        assert cfg["model"]["name"] == "SmallCNN"
        assert cfg["model"]["use_aib"] is False
        assert cfg["seed"] == 100
    print("[PASS] all configs parse and contain expected settings")


def test_feddisco_controller_instantiates_from_config():
    cfg = _load_config("feddisco_pilot_cifar100_feddisco_alpha0.3.yaml")

    # Synthetic labels/partition matching the config.
    labels = np.arange(100).repeat(50)
    client_indices = dirichlet_partition(labels, num_clients=10, alpha=0.3, min_samples_per_client=50, seed=100)

    ctrl = FedDiscoController(
        num_classes=cfg["dataset"]["num_classes"],
        a=cfg["feddisco"]["a"],
        b=cfg["feddisco"]["b"],
        discrepancy_metric=cfg["feddisco"]["discrepancy"],
        dataset_name=cfg["dataset"]["name"],
        seed=cfg["seed"],
    )
    ctrl.precompute_weights(
        client_label_hists={cid: class_histogram(labels, idxs, 100) for cid, idxs in client_indices.items()},
        client_num_samples={cid: len(idxs) for cid, idxs in client_indices.items()},
    )

    weights, diag = ctrl.get_weights([0, 1, 2, 3, 4], round_idx=1, shuffle=False)
    assert len(weights) == 5
    assert np.isclose(sum(weights), 1.0)
    assert all(w >= -1e-12 for w in weights)
    assert np.isclose(sum(diag["discrepancies"]), 1.0)
    print("[PASS] feddisco controller instantiates and produces valid weights")


def test_feddisco_shuffled_controller_instantiates_from_config():
    cfg = _load_config("feddisco_pilot_cifar100_feddisco_shuffled_alpha0.3.yaml")

    labels = np.arange(100).repeat(50)
    client_indices = dirichlet_partition(labels, num_clients=10, alpha=0.3, min_samples_per_client=50, seed=100)

    ctrl = FedDiscoController(
        num_classes=cfg["dataset"]["num_classes"],
        a=cfg["feddisco"]["a"],
        b=cfg["feddisco"]["b"],
        discrepancy_metric=cfg["feddisco"]["discrepancy"],
        dataset_name=cfg["dataset"]["name"],
        seed=cfg["seed"],
    )
    ctrl.precompute_weights(
        client_label_hists={cid: class_histogram(labels, idxs, 100) for cid, idxs in client_indices.items()},
        client_num_samples={cid: len(idxs) for cid, idxs in client_indices.items()},
    )

    weights, diag = ctrl.get_weights([0, 1, 2, 3, 4], round_idx=1, shuffle=True)
    assert len(weights) == 5
    assert np.isclose(sum(weights), 1.0)
    assert diag["shuffle_order"] is not None
    assert sorted(diag["discrepancies"]) == sorted(diag["discrepancies_used"])
    print("[PASS] feddisco_shuffled controller instantiates and shuffles discrepancies")


def test_model_build_without_aib():
    for cfg_name in [
        "feddisco_pilot_cifar100_fedavg_alpha0.3.yaml",
        "feddisco_pilot_cifar100_feddisco_shuffled_alpha0.3.yaml",
    ]:
        cfg = _load_config(cfg_name)
        model = build_model(cfg)
        assert "AIB2d" not in str(model)
        assert cfg["model"]["use_aib"] is False
    print("[PASS] SmallCNN built without AIB for all arms")


if __name__ == "__main__":
    import numpy as np

    test_all_configs_parse()
    test_feddisco_controller_instantiates_from_config()
    test_feddisco_shuffled_controller_instantiates_from_config()
    test_model_build_without_aib()
    print("\nAll config smoke tests passed.")
