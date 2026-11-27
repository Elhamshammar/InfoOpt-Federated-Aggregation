#!/usr/bin/env python3
"""Deterministic corruption manifest generation for the MLJ-P1 experiment.

Mirrors the logic in ``scripts/mlj_p0_cpu_audit.py`` but persists per-seed
manifests under ``data/corruption_manifests/`` so that the GPU runner can
reconstruct the exact same local-label corruption without regenerating it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE = ROOT / "_dataset_cache"
MANIFEST_DIR = ROOT / "data" / "corruption_manifests"

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
from data.partition import dirichlet_partition  # noqa: E402

DEFAULT_NUM_CLIENTS = 10
DEFAULT_NUM_CORRUPTED = 3
DEFAULT_CORRUPTION_RATE = 0.8
DEFAULT_NUM_CLASSES = 100
DEFAULT_SYMMETRIC_OTHER_CLASSES = True
DEFAULT_MIN_SAMPLES_PER_CLIENT = 50
DEFAULT_SEEDS = (100, 101, 102, 103, 104)


def _load_cifar100_labels(cache_dir: Path = DATASET_CACHE) -> np.ndarray:
    try:
        import torchvision

        train = torchvision.datasets.CIFAR100(
            root=str(cache_dir), train=True, download=False
        )
        return np.array(train.targets)
    except Exception as exc:
        raise RuntimeError(
            f"CIFAR-100 not found in {cache_dir}. Download it first."
        ) from exc


def select_corrupted_clients(
    num_clients: int, num_corrupted: int, seed: int
) -> list[int]:
    """Deterministically select the corrupted clients for a seed."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(num_clients, num_corrupted, replace=False).tolist())


def corrupt_labels(
    labels: np.ndarray,
    indices: Sequence[int],
    corruption_rate: float,
    seed: int,
    num_classes: int = 100,
    symmetric_other_classes: bool = True,
) -> tuple[np.ndarray, float, list[int], list[int]]:
    """Apply symmetric label noise to a subset of training indices.

    Returns the full label array (copy), the actual corruption rate, the
    corrupted global positions, and the replacement labels.
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
    return (
        corrupted_labels,
        float(actual_rate),
        sample_positions.tolist(),
        replacements.tolist(),
    )


def build_manifest(
    labels: np.ndarray,
    seed: int,
    alpha: float,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    num_corrupted: int = DEFAULT_NUM_CORRUPTED,
    corruption_rate: float = DEFAULT_CORRUPTION_RATE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    symmetric_other_classes: bool = DEFAULT_SYMMETRIC_OTHER_CLASSES,
    min_samples_per_client: int = DEFAULT_MIN_SAMPLES_PER_CLIENT,
) -> dict:
    """Build a single manifest describing corrupted clients for one seed/alpha."""
    client_indices = dirichlet_partition(
        labels,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_client=min_samples_per_client,
        seed=seed,
    )
    corrupted_clients = select_corrupted_clients(num_clients, num_corrupted, seed)
    client_maps: dict[str, dict] = {}
    for cid in corrupted_clients:
        per_client_seed = seed * 10000 + cid
        _, actual_rate, positions, repl = corrupt_labels(
            labels,
            client_indices[cid],
            corruption_rate,
            seed=per_client_seed,
            num_classes=num_classes,
            symmetric_other_classes=symmetric_other_classes,
        )
        client_maps[str(cid)] = {
            "corruption_rate_requested": corruption_rate,
            "corruption_rate_actual": actual_rate,
            "num_corrupted": len(positions),
            "positions": positions,
            "replacements": repl,
        }
    return {
        "seed": seed,
        "alpha": alpha,
        "num_clients": num_clients,
        "num_corrupted": num_corrupted,
        "corruption_rate": corruption_rate,
        "num_classes": num_classes,
        "symmetric_other_classes": symmetric_other_classes,
        "min_samples_per_client": min_samples_per_client,
        "corrupted_clients": corrupted_clients,
        "client_maps": client_maps,
    }


def save_manifests(
    alpha: float,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_dir: Path = MANIFEST_DIR,
) -> list[Path]:
    """Generate and save manifests for every requested seed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_cifar100_labels()
    paths: list[Path] = []
    for seed in seeds:
        manifest = build_manifest(labels, seed=seed, alpha=alpha)
        path = output_dir / f"corruption_manifest_alpha{alpha}_seed{seed}.json"
        path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        paths.append(path)
    return paths


def load_manifest(
    alpha: float,
    seed: int,
    manifest_dir: Path = MANIFEST_DIR,
) -> dict:
    """Load a previously saved corruption manifest."""
    path = manifest_dir / f"corruption_manifest_alpha{alpha}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def client_corruption_map(manifest: dict, client_id: int) -> dict[int, int]:
    """Return a global-index -> replacement mapping for one client."""
    key = str(client_id)
    if key not in manifest["client_maps"]:
        return {}
    cm = manifest["client_maps"][key]
    return {int(pos): int(rep) for pos, rep in zip(cm["positions"], cm["replacements"])}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    paths = save_manifests(alpha=args.alpha, seeds=args.seeds)
    for p in paths:
        print(f"wrote {p}")
