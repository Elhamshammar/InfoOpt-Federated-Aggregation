#!/usr/bin/env python3
"""Resolve dataset roots to absolute paths and emit effective configs.

Runs at launch (before schedules and training) from ANY working directory.
For every arm config in snapshot/configs/, write an effective copy under
EXP_ROOT/configs_effective/ with dataset.root replaced by an absolute,
existence-checked path. All three datasets must exist before any training
starts; otherwise abort with the missing paths printed.

Dataset root policy (in order):
  1. $DATASETS_ROOT/<subdir> if provided (env var);
  2. EXP_ROOT/datasets/<subdir> if it exists;

CIFAR-100 may live either at <root>/cifar-100-python.tar.gz (torchvision
layout, dataset name CIFAR100) — root is the PARENT of the tar. EuroSAT
and PathMNIST use their own subdirectories per the canonical configs
(datasets/eurosat, datasets/pathmnist under the canonical tree).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP_ROOT = Path(os.environ.get("EXP_ROOT", str(HERE)))
CANONICAL = None  # removed: no public server fallback


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(name: str) -> Path:
    """Absolute dataset root for a canonical config dataset name."""
    env = os.environ.get("DATASETS_ROOT")
    submap = {
        "CIFAR100": ["", "cifar100", "CIFAR100"],
        "EuroSAT": ["eurosat", "EuroSAT"],
        "PathMNIST": ["pathmnist", "PathMNIST"],
    }
    bases = []
    if env:
        bases.append(Path(env))
    bases.append(EXP_ROOT / "datasets")
    for base in bases:
        for sub in submap.get(name, [""]):
            cand = (base / sub) if sub else base
            if _dataset_present(name, cand):
                return cand.resolve()
    raise SystemExit(f"dataset {name} not found in any of: "
                     + ", ".join(str(b) for b in bases))


def _dataset_present(name: str, root: Path) -> bool:
    if not root.is_dir():
        return False
    if name == "CIFAR100":
        return (root / "cifar-100-python.tar.gz").is_file() or \
               (root / "cifar-100-python").is_dir()
    if name == "EuroSAT":
        return any(root.iterdir())
    if name == "PathMNIST":
        return any(root.iterdir())
    return False


def main():
    eff_dir = EXP_ROOT / "configs_effective"
    eff_dir.mkdir(parents=True, exist_ok=True)
    records = []
    subset = os.environ.get("DATASETS", "").split()
    for src in sorted((HERE / "snapshot" / "configs").glob("stage4_*.yaml")):
        if subset and not any(f"_{d}_" in src.name for d in subset):
            print(f"note: skipping config outside DATASETS subset: {src.name}")
            continue
        cfg = yaml.safe_load(open(src, encoding="utf-8"))
        root = resolve(cfg["dataset"]["name"])
        cfg["dataset"]["root"] = str(root)
        dst = eff_dir / src.name
        if dst.exists() and yaml.safe_load(open(dst)) != cfg:
            raise SystemExit(f"effective config differs on disk, refusing "
                             f"to overwrite: {dst}")
        with open(dst, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        records.append({"config": dst.name,
                        "dataset": cfg["dataset"]["name"],
                        "root": str(root),
                        "sha256": sha256(dst)})
    (eff_dir / "RESOLUTION.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8")
    print(f"resolved {len(records)} effective configs into {eff_dir}")


if __name__ == "__main__":
    main()
