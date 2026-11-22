#!/usr/bin/env python3
"""Generate the shared partition indices and cohort schedules for the
Stage-3 heterogeneity experiment (alpha in {0.1, 1.0}), using the FROZEN
snapshot code so all six arms share identical partition and cohort
sequence within each dataset-alpha-seed combination.

Runs on the server BEFORE any training (launcher phase 0). Idempotent:
existing files with matching content are left untouched (never
overwritten); any mismatch aborts.

Inputs: snapshot/configs/stage3_<dataset>_fedavg_alpha<a>.yaml for
dataset labels, and seeds 100..104 (the experiment grid). train.py
derives the partition with its config's alpha and seed and the cohort
schedule from random.Random(cfg["seed"]).sample — this script reproduces
both with the snapshot's own functions, VALIDATES each partition
(coverage, duplicates, range, client count, minimum size), records
hashes, and writes descriptive statistics (client sizes, per-client
label JS divergence) into SHARED_MANIFEST.json.

The partition algorithm is the inherited dirichlet_partition()
(snapshot/data/partition.py): per-class Dirichlet(alpha) proportions,
up to 100 resampling attempts to satisfy min_samples_per_client,
RuntimeError otherwise. It is NOT changed here, and a partition is never
regenerated because of model performance.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "snapshot"
SHARED = HERE / "shared"
SEEDS = (100, 101, 102, 103, 104)
ALPHAS = (0.1, 1.0)
# schedule inputs are identical across arms (only experiment_name differs)
DATASETS = ("cifar100", "eurosat", "pathmnist")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def validate_partition(indices, n_total, num_clients, min_samples):
    """Hard checks on a generated partition. Raises ValueError."""
    if set(indices.keys()) != set(range(num_clients)):
        raise ValueError(f"client keys != 0..{num_clients - 1}")
    flat = [i for v in indices.values() for i in v]
    if len(flat) != len(set(flat)):
        raise ValueError("duplicate indices in partition")
    if sorted(flat) != list(range(n_total)):
        raise ValueError("partition does not cover exactly 0..N-1")
    sizes = {k: len(v) for k, v in indices.items()}
    if min(sizes.values()) < min_samples:
        raise ValueError(f"client below min_samples: {sizes}")


def main():
    import argparse
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=None,
                    help="directory with effective arm configs (default: "
                         "snapshot/configs; use EXP_ROOT/configs_effective "
                         "at launch)")
    ap.add_argument("--shared-dir", default=None,
                    help="output directory for shared files (default: "
                         "<package>/shared)")
    cli = ap.parse_args()
    sys.path.insert(0, str(SNAPSHOT))
    import os
    import numpy as np
    import yaml
    from data.datasets import get_dataset, get_labels
    from data.partition import dirichlet_partition

    subset = os.environ.get("DATASETS", "").split()
    datasets = [d for d in DATASETS if not subset or d in subset]
    cfg_dir = Path(cli.configs_dir) if cli.configs_dir else \
        SNAPSHOT / "configs"
    global SHARED
    if cli.shared_dir:
        SHARED = Path(cli.shared_dir)
    SHARED.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_by": "generate_shared_schedules.py",
                "experiment": "stage3_heterogeneity_20261012",
                "configs_dir": str(cfg_dir),
                "alphas": [str(a) for a in ALPHAS],
                "seeds": list(SEEDS), "files": []}

    for ds in datasets:
        for alpha in ALPHAS:
            cfg = yaml.safe_load(open(
                cfg_dir / f"stage3_{ds}_fedavg_alpha{alpha}.yaml",
                encoding="utf-8"))
            train_set, _ = get_dataset(cfg["dataset"]["name"],
                                       cfg["dataset"]["root"],
                                       cfg["dataset"]["num_classes"])
            labels = np.asarray(get_labels(train_set))
            n_total = len(labels)
            gh = np.bincount(labels, minlength=cfg["dataset"]
                             ["num_classes"]).astype(float)
            gh = gh / gh.sum()
            for seed in SEEDS:
                indices = dirichlet_partition(
                    labels,
                    num_clients=cfg["partition"]["num_clients"],
                    alpha=alpha,
                    min_samples_per_client=cfg["partition"]
                                           .get("min_samples_per_client",
                                                10),
                    seed=seed)
                validate_partition(indices, n_total,
                                   cfg["partition"]["num_clients"],
                                   cfg["partition"]
                                      .get("min_samples_per_client", 10))
                part_path = SHARED / (f"partition_{ds}_alpha{alpha}"
                                      f"_seed{seed}.json")
                part_payload = {
                    "dataset": ds, "alpha": str(alpha), "seed": seed,
                    "num_clients": cfg["partition"]["num_clients"],
                    "alpha_value": alpha,
                    "client_indices": {str(k): [int(i) for i in v]
                                       for k, v in indices.items()},
                }
                blob = json.dumps(part_payload, sort_keys=True).encode()
                _write_once(part_path, blob)

                rng = random.Random(seed)
                schedule = [rng.sample(
                    range(cfg["partition"]["num_clients"]),
                    cfg["training"]["clients_per_round"])
                    for _ in range(cfg["training"]["rounds"])]
                sched_path = SHARED / (f"cohort_schedule_{ds}_alpha{alpha}"
                                       f"_seed{seed}.json")
                sched_payload = {
                    "dataset": ds, "alpha": str(alpha), "seed": seed,
                    "rounds": cfg["training"]["rounds"],
                    "clients_per_round": cfg["training"]
                    ["clients_per_round"],
                    "schedule": schedule,
                    "note": ("random.Random(seed).sample sequence, "
                             "matching train.py's cohort sampler; "
                             "identical across alpha regimes by "
                             "construction (no alpha input)"),
                }
                blob2 = json.dumps(sched_payload, sort_keys=True).encode()
                _write_once(sched_path, blob2)

                # descriptive stats only — never used for selection
                sizes = {k: len(v) for k, v in indices.items()}
                js_vals = []
                num_classes = cfg["dataset"]["num_classes"]
                for k, v in indices.items():
                    h = np.bincount(np.asarray(labels)[v],
                                    minlength=num_classes).astype(float)
                    h = h / h.sum()
                    m = 0.5 * (h + gh)
                    js = 0.5 * float(np.sum(h * np.log2(
                        np.clip(h, 1e-12, None)
                        / np.clip(m, 1e-12, None)))) \
                        + 0.5 * float(np.sum(gh * np.log2(
                            np.clip(gh, 1e-12, None)
                            / np.clip(m, 1e-12, None))))
                    js_vals.append(js)
                manifest["files"].append({
                    "dataset": ds, "alpha": str(alpha), "seed": seed,
                    "partition": {"path": str(part_path),
                                  "sha256": sha256_bytes(
                                      part_path.read_bytes())},
                    "schedule": {"path": str(sched_path),
                                 "sha256": sha256_bytes(
                                     sched_path.read_bytes())},
                    "descriptive_stats": {
                        "client_sizes": sizes,
                        "min_client_size": min(sizes.values()),
                        "max_client_size": max(sizes.values()),
                        "mean_label_js_to_global":
                            sum(js_vals) / len(js_vals),
                        "max_label_js_to_global": max(js_vals),
                    },
                })

    mpath = SHARED / "SHARED_MANIFEST.json"
    _write_once(mpath, json.dumps(manifest, indent=1,
                                  sort_keys=True).encode())
    print(f"wrote {len(manifest['files'])} dataset-alpha-seed "
          f"combinations to {SHARED}")


def _write_once(path: Path, blob: bytes):
    if path.exists():
        if path.read_bytes() == blob:
            return
        raise SystemExit(f"existing file differs, refusing to overwrite: "
                         f"{path}")
    path.write_bytes(blob)


if __name__ == "__main__":
    main()
