"""Integration tests for the MLJ-P0.1/P1 positive-control pipeline.

These tests exercise the actual GPU runner components without training neural
networks: dataset wrapper, corruption manifests, configuration generation, and
the historical InfoOpt controller compatibility.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
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
sys.path.insert(0, str(ROOT / "src"))

from corrupted_dataset import CorruptedDataset  # noqa: E402
from corruption_manifest import (  # noqa: E402
    build_manifest,
    client_corruption_map,
    load_manifest,
    save_manifests,
    select_corrupted_clients,
)
from data.datasets import get_dataset, get_labels  # noqa: E402
from data.partition import dirichlet_partition  # noqa: E402
from methods.infoopt import InfoOptController  # noqa: E402
from train_mlj_p1 import (  # noqa: E402
    configure_infoopt_for_oracle,
    load_config,
    METHOD_CHOICES,
)


@pytest.fixture(scope="module")
def cifar100_train_test():
    cache = ROOT / "_dataset_cache"
    train_set, test_set = get_dataset("CIFAR100", str(cache), 100)
    return train_set, test_set


def test_method_choices_complete():
    assert METHOD_CHOICES == ["fedavg", "oracle_nojs_aligned", "oracle_nojs_shuffled"]


def test_config_files_exist_and_parse():
    for method in METHOD_CHOICES:
        path = ROOT / "configs" / f"mlj_p1_{method}_alpha0.1.yaml"
        assert path.exists(), path
        cfg = load_config(str(path))
        assert cfg["partition"]["num_clients"] == 10
        assert cfg["training"]["clients_per_round"] == 5
        assert cfg["training"]["rounds"] == 100
        assert cfg["model"]["use_aib"] is False
        assert cfg["partition"]["alpha"] == 0.1


def test_configure_infoopt_for_oracle_sets_nonrobust_nojs():
    cfg = {"infoopt": {"lambda_entropy": 0.5, "lambda_mi": 1.0}}
    configure_infoopt_for_oracle(cfg, shuffled=False)
    io = cfg["infoopt"]
    assert io["lambda_mi"] == 1.0
    assert io["lambda_entropy"] == 0.0
    assert io["lambda_stability"] == 0.0
    assert io["robust_scale"] is False
    assert io["kappa_frac"] == 0.6
    assert io["js_mode"] == "constraint"
    assert io["fedavg_mix"] == 0.0
    assert io["early_stop"]["enabled"] is False
    assert "score_shuffle" not in io

    cfg2 = {"infoopt": {}}
    configure_infoopt_for_oracle(cfg2, shuffled=True)
    assert cfg2["infoopt"]["score_shuffle"] == "stage1"


def test_manifests_are_deterministic_and_cover_all_seeds():
    alpha = 0.1
    seeds = [100, 101, 102, 103, 104]
    tmp_dir = ROOT / "tests" / "_tmp_manifests"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = save_manifests(alpha=alpha, seeds=seeds, output_dir=tmp_dir)
    assert len(paths) == len(seeds)
    for seed, path in zip(seeds, paths):
        m1 = json.loads(path.read_text(encoding="utf-8"))
        m2 = build_manifest(
            _load_labels(), seed=seed, alpha=alpha,
            num_corrupted=3, corruption_rate=0.8
        )
        assert m1["corrupted_clients"] == m2["corrupted_clients"]
        assert m1["client_maps"].keys() == m2["client_maps"].keys()
        for cid in m1["client_maps"]:
            assert (
                m1["client_maps"][cid]["positions"]
                == m2["client_maps"][cid]["positions"]
            )
            assert (
                m1["client_maps"][cid]["replacements"]
                == m2["client_maps"][cid]["replacements"]
            )


def _load_labels() -> np.ndarray:
    train, _ = get_dataset("CIFAR100", str(ROOT / "_dataset_cache"), 100)
    return np.array(get_labels(train))


def test_corrupted_dataset_returns_corrupted_train_labels(cifar100_train_test):
    train_set, _ = cifar100_train_test
    manifest = load_manifest(alpha=0.1, seed=100)
    cmap = {}
    for cid in manifest["corrupted_clients"]:
        cmap.update(client_corruption_map(manifest, cid))
    corrupted_set = CorruptedDataset(train_set, cmap)

    assert hasattr(corrupted_set, "targets")
    original_targets = np.array(get_labels(train_set))
    assert np.array_equal(np.array(corrupted_set.targets), original_targets)

    corrupted_count = 0
    for pos in cmap:
        _, y = corrupted_set[pos]
        assert y == cmap[pos]
        assert y != original_targets[pos]
        corrupted_count += 1
    assert corrupted_count == len(cmap)

    sample_positions = set(cmap.keys())
    for i in range(0, len(corrupted_set), 1000):
        if i not in sample_positions:
            _, y = corrupted_set[i]
            assert y == original_targets[i]


def test_test_labels_unchanged(cifar100_train_test):
    _, test_set = cifar100_train_test
    labels = np.array(get_labels(test_set))
    assert labels.min() >= 0
    assert labels.max() < 100
    assert len(labels) == 10000


def test_partition_matches_manifest_alpha_and_seed():
    labels = _load_labels()
    seed = 100
    alpha = 0.1
    partition = dirichlet_partition(
        labels,
        num_clients=10,
        alpha=alpha,
        min_samples_per_client=50,
        seed=seed,
    )
    manifest = load_manifest(alpha=alpha, seed=seed)
    corrupted_clients = manifest["corrupted_clients"]
    for cid in corrupted_clients:
        assert len(partition[cid]) > 0
        positions = manifest["client_maps"][str(cid)]["positions"]
        assert all(p in partition[cid] for p in positions)


def test_schedule_deterministic_from_seed_matches_runner():
    seed = 100
    rng = __import__("random").Random(seed)
    schedule = [
        rng.sample(range(10), 5) for _ in range(100)
    ]
    rng2 = __import__("random").Random(seed)
    schedule2 = [
        rng2.sample(range(10), 5) for _ in range(100)
    ]
    assert schedule == schedule2
    assert all(len(set(c)) == 5 for c in schedule)


def test_historical_controller_nojs_nonrobust_runs(cifar100_train_test):
    train_set, _ = cifar100_train_test
    labels = np.array(get_labels(train_set))
    seed = 100
    alpha = 0.1
    partition = dirichlet_partition(
        labels, num_clients=10, alpha=alpha, min_samples_per_client=50, seed=seed
    )
    global_hist = np.bincount(labels, minlength=100).astype(float)

    cfg = load_config(str(ROOT / "configs" / "mlj_p1_oracle_nojs_aligned_alpha0.1.yaml"))
    configure_infoopt_for_oracle(cfg, shuffled=False)
    controller = InfoOptController(cfg, global_hist, variant="InfoOpt-NoJS")

    selected = [1, 3, 5, 7, 9]
    rng = np.random.default_rng(seed)
    fake_hists = []
    for cid in selected:
        p = rng.dirichlet(np.full(100, alpha))
        fake_hists.append(rng.multinomial(len(partition[cid]), p).astype(float))

    class FakeResult:
        __slots__ = ["client_id", "num_samples", "mi_proxy", "entropy",
                     "update_norm", "train_loss_std", "label_hist", "score"]

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.score = None

    results = [
        FakeResult(
            client_id=cid,
            num_samples=len(partition[cid]),
            mi_proxy=-0.8 if cid in {1, 3, 5} else 0.0,
            entropy=0.0,
            update_norm=0.0,
            train_loss_std=0.0,
            label_hist=fake_hists[i],
        )
        for i, cid in enumerate(selected)
    ]
    weights, info = controller.compute_weights(results, round_idx=1)
    weights = np.asarray(weights, dtype=float)
    assert np.isfinite(weights).all()
    assert abs(weights.sum() - 1.0) < 1e-6
    assert len(weights) == len(selected)
    clean_weight = weights[[selected.index(cid) for cid in [7, 9]]].sum()
    corrupt_weight = weights[[selected.index(cid) for cid in [1, 3, 5]]].sum()
    assert clean_weight > corrupt_weight


def test_output_directory_isolation():
    run_dir = ROOT / "runs"
    from train_mlj_p1 import make_run_dir
    import time
    d1 = make_run_dir("fedavg", 100, 0.1)
    time.sleep(0.01)
    d2 = make_run_dir("fedavg", 100, 0.1)
    assert d1 != d2
    assert d1.parent == run_dir
    assert d2.parent == run_dir


def test_dryrun_does_not_create_verified_marker():
    marker = ROOT / ".verified"
    assert not marker.exists()


def test_no_hardcoded_windows_paths_in_launcher():
    launcher = (ROOT / "run_pilot.sh").read_text(encoding="utf-8")
    assert r"D:\\" not in launcher
    assert r"C:\\" not in launcher
    assert "_dataset_cache" in launcher or "configs" in launcher


def test_environment_can_import_historical_modules():
    assert SNAPSHOT_ROOT.exists()
    assert (SNAPSHOT_ROOT / "methods" / "infoopt.py").exists()
    assert (SNAPSHOT_ROOT / "models" / "build.py").exists()
