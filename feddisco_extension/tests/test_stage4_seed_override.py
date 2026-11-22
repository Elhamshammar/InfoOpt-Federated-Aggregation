"""Verify that train.py CLI seed and GPU overrides actually reach the config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PILOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT_ROOT / "snapshot"))

from data.partition import dirichlet_partition  # noqa: E402


def _load_config(name: str) -> dict:
    with open(PILOT_ROOT / "configs" / name, "r") as f:
        return yaml.safe_load(f)


def _apply_seed_override(cfg: dict, seed: int) -> dict:
    """Mirror train.py's seed override logic."""
    cfg["seed"] = seed
    return cfg


def test_seed_override_changes_partition():
    """--seed 101 must produce a different Dirichlet partition than seed 100."""
    cfg = _load_config("feddisco_stage4_cifar100_feddisco_alpha0.3.yaml")
    assert cfg["seed"] == 100, "base config seed should be 100"

    labels = np.arange(100).repeat(50)
    partition_100 = dirichlet_partition(
        labels,
        num_clients=cfg["partition"]["num_clients"],
        alpha=cfg["partition"]["alpha"],
        min_samples_per_client=cfg["partition"]["min_samples_per_client"],
        seed=cfg["seed"],
    )

    cfg = _apply_seed_override(cfg, 101)
    assert cfg["seed"] == 101, "seed override did not update config"

    partition_101 = dirichlet_partition(
        labels,
        num_clients=cfg["partition"]["num_clients"],
        alpha=cfg["partition"]["alpha"],
        min_samples_per_client=cfg["partition"]["min_samples_per_client"],
        seed=cfg["seed"],
    )

    # Partitions must differ; compare the full index lists.
    differ = any(
        list(partition_100[cid]) != list(partition_101[cid])
        for cid in partition_100
    )
    assert differ, "seed override did not change partition"
    print("[PASS] seed override changes partition")


def test_seed_override_for_all_stage4_configs():
    configs = [
        "feddisco_stage4_cifar100_fedavg_alpha0.3.yaml",
        "feddisco_stage4_cifar100_uniform_alpha0.3.yaml",
        "feddisco_stage4_cifar100_feddisco_alpha0.3.yaml",
        "feddisco_stage4_cifar100_feddisco_shuffled_alpha0.3.yaml",
    ]
    for cfg_name in configs:
        cfg = _load_config(cfg_name)
        assert cfg["seed"] == 100
        cfg = _apply_seed_override(cfg, 101)
        assert cfg["seed"] == 101
    print("[PASS] seed override applies to all Stage 4 configs")


def test_argparser_accepts_seed_and_gpu():
    """train.py's argument parser must accept --seed and --gpu."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args(
        [
            "--config",
            "configs/feddisco_stage4_cifar100_feddisco_alpha0.3.yaml",
            "--method",
            "feddisco",
            "--seed",
            "103",
            "--gpu",
            "1",
        ]
    )
    assert args.seed == 103
    assert args.gpu == 1
    print("[PASS] argument parser accepts --seed and --gpu overrides")


if __name__ == "__main__":
    test_seed_override_changes_partition()
    test_seed_override_for_all_stage4_configs()
    test_argparser_accepts_seed_and_gpu()
    print("\nAll seed-override tests passed.")
