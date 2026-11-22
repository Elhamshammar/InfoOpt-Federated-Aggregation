#!/usr/bin/env python3
"""Generate the shared partition indices and cohort schedules for the
four-arm experiment, using the FROZEN snapshot code so all four arms share
identical partition and cohort sequence within each dataset-seed pair.

Runs on the server BEFORE any training (launcher phase 0). Idempotent:
existing files with matching content are left untouched (never
overwritten); any mismatch aborts.

Inputs: snapshot/configs/fourarm_<dataset>.yaml for dataset labels, and
seeds 100..104 (the experiment grid). train.py derives partition with
seed=cfg["seed"] (CLI --seed overrides before partitioning) and the cohort
schedule from random.Random(cfg["seed"]).sample — this script reproduces
both with the snapshot's own functions and records hashes for later
runtime verification (verify_run.py compares each run's recorded cohort
column to these files).
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "snapshot"
SHARED = HERE / "shared"
SEEDS = (100, 101, 102, 103, 104)
# schedule inputs are identical across arms (only experiment_name differs)
DATASETS = {"cifar100": "fourarm_cifar100_fedavg_aiboff",
            "eurosat": "fourarm_eurosat_fedavg_aiboff",
            "pathmnist": "fourarm_pathmnist_fedavg_aiboff"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
    import yaml
    from data.datasets import get_dataset, get_labels
    from data.partition import dirichlet_partition

    subset = os.environ.get("DATASETS", "").split()
    datasets = {k: v for k, v in DATASETS.items()
                if not subset or k in subset}
    cfg_dir = Path(cli.configs_dir) if cli.configs_dir else \
        SNAPSHOT / "configs"
    global SHARED
    if cli.shared_dir:
        SHARED = Path(cli.shared_dir)
    SHARED.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_by": "generate_shared_schedules.py",
                "configs_dir": str(cfg_dir),
                "seeds": list(SEEDS), "files": []}

    for ds, exp in datasets.items():
        cfg = yaml.safe_load(open(cfg_dir / f"{exp}.yaml",
                                  encoding="utf-8"))
        train_set, _ = get_dataset(cfg["dataset"]["name"],
                                   cfg["dataset"]["root"],
                                   cfg["dataset"]["num_classes"])
        labels = get_labels(train_set)
        for seed in SEEDS:
            indices = dirichlet_partition(
                labels,
                num_clients=cfg["partition"]["num_clients"],
                alpha=cfg["partition"]["alpha"],
                min_samples_per_client=cfg["partition"]
                                       .get("min_samples_per_client", 10),
                seed=seed)
            part_path = SHARED / f"partition_{ds}_seed{seed}.json"
            part_payload = {
                "dataset": ds, "seed": seed,
                "num_clients": cfg["partition"]["num_clients"],
                "alpha": cfg["partition"]["alpha"],
                "client_indices": {str(k): [int(i) for i in v]
                                   for k, v in indices.items()},
            }
            blob = json.dumps(part_payload, sort_keys=True).encode()
            _write_once(part_path, blob)

            rng = random.Random(seed)
            schedule = [rng.sample(range(cfg["partition"]["num_clients"]),
                                   cfg["training"]["clients_per_round"])
                        for _ in range(cfg["training"]["rounds"])]
            sched_path = SHARED / f"cohort_schedule_{ds}_seed{seed}.json"
            sched_payload = {
                "dataset": ds, "seed": seed,
                "rounds": cfg["training"]["rounds"],
                "clients_per_round": cfg["training"]["clients_per_round"],
                "schedule": schedule,
                "note": ("random.Random(seed).sample sequence, matching "
                         "train.py's cohort sampler"),
            }
            blob2 = json.dumps(sched_payload, sort_keys=True).encode()
            _write_once(sched_path, blob2)
            manifest["files"].append({
                "partition": {"path": str(part_path),
                              "sha256": sha256_bytes(
                                  part_path.read_bytes())},
                "schedule": {"path": str(sched_path),
                             "sha256": sha256_bytes(
                                 sched_path.read_bytes())},
            })

    mpath = SHARED / "SHARED_MANIFEST.json"
    _write_once(mpath, json.dumps(manifest, indent=1,
                                  sort_keys=True).encode())
    print(f"wrote {len(manifest['files'])} dataset-seed pairs to {SHARED}")


def _write_once(path: Path, blob: bytes):
    if path.exists():
        if path.read_bytes() == blob:
            return
        raise SystemExit(f"existing file differs, refusing to overwrite: "
                         f"{path}")
    path.write_bytes(blob)


if __name__ == "__main__":
    main()
