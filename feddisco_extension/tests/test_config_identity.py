"""Verify that the four pilot configs differ only in the aggregation method."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PILOT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PILOT_ROOT / "configs"

ARMS = {
    "fedavg": "feddisco_pilot_cifar100_fedavg_alpha0.3.yaml",
    "uniform": "feddisco_pilot_cifar100_uniform_alpha0.3.yaml",
    "feddisco": "feddisco_pilot_cifar100_feddisco_alpha0.3.yaml",
    "feddisco_shuffled": "feddisco_pilot_cifar100_feddisco_shuffled_alpha0.3.yaml",
}


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, "r") as f:
        return yaml.safe_load(f)


def _deep_equal(a, b, path="") -> list:
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: missing in one config")
            else:
                diffs.extend(_deep_equal(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(_deep_equal(x, y, f"{path}[{i}]"))
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def test_common_settings_identical():
    cfgs = {k: _load(v) for k, v in ARMS.items()}
    base = cfgs["fedavg"]

    ignored_paths = {
        ".experiment_name",
        ".stage_design.arm",
        ".stage_design.note",
        ".training.uniform_agg",
        ".feddisco.shuffle",
    }

    for arm, cfg in cfgs.items():
        all_diffs = _deep_equal(base, cfg)
        allowed = {d.split(":", 1)[0] for d in all_diffs if any(d.startswith(p) for p in ignored_paths)}
        unexpected = [d for d in all_diffs if d.split(":", 1)[0] not in allowed]
        assert not unexpected, f"Unexpected differences for {arm}: {unexpected}"

    print("[PASS] all configs share identical common settings; only method-specific fields differ")


def test_method_specific_fields():
    fedavg = _load(ARMS["fedavg"])
    uniform = _load(ARMS["uniform"])
    feddisco = _load(ARMS["feddisco"])
    shuffled = _load(ARMS["feddisco_shuffled"])

    assert fedavg["training"]["uniform_agg"] is False
    assert uniform["training"]["uniform_agg"] is True
    assert feddisco["feddisco"]["shuffle"] is False
    assert shuffled["feddisco"]["shuffle"] is True
    assert "fedavg" in fedavg["experiment_name"]
    assert "uniform" in uniform["experiment_name"]
    assert "feddisco" in feddisco["experiment_name"] and "shuffled" not in feddisco["experiment_name"]
    assert "feddisco_shuffled" in shuffled["experiment_name"]
    print("[PASS] method-specific fields are set correctly")


if __name__ == "__main__":
    test_common_settings_identical()
    test_method_specific_fields()
    print("\nAll config-identity tests passed.")
