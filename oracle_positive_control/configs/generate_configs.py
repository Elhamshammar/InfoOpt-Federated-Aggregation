#!/usr/bin/env python3
"""Generate the three MLJ-P1 frozen configs from the Study-3 effective config.

This avoids duplicating the bulk of the Stage-3 hyper-parameters; the generated
files are checked-in/frozen YAML artifacts.
"""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STUDY3_CONFIG = (
    ROOT.parent
    / "InfoOpt_GitHub_Release"
    / "configs"
    / "stage3"
    / "configs_effective"
    / "stage3_cifar100_infoopt_alpha0.1.yaml"
)
OUTPUT_DIR = ROOT / "configs"


def load_study3_config() -> dict:
    with open(STUDY3_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_base() -> dict:
    cfg = load_study3_config()
    cfg["experiment_name"] = "mlj_p1"
    cfg["output"] = {"dir": "runs"}
    cfg["seed"] = 100
    cfg["model"]["use_aib"] = False
    cfg["partition"] = {
        "num_clients": 10,
        "alpha": 0.1,
        "min_samples_per_client": 50,
    }
    cfg["training"]["clients_per_round"] = 5
    cfg["training"]["rounds"] = 100
    return cfg


def apply_oracle_overrides(cfg: dict, shuffled: bool) -> dict:
    io = cfg.setdefault("infoopt", {})
    io["lambda_mi"] = 1.0
    io["lambda_entropy"] = 0.0
    io["lambda_stability"] = 0.0
    io["signal_ema_beta"] = 1.0
    io["robust_scale"] = False
    io["kappa_frac"] = 0.6
    io["temperature"] = 1.0
    io["js_mode"] = "constraint"
    io["score_clip"] = None
    io["fedavg_mix"] = 0.0
    io["min_weight"] = 0.0
    io["min_weight_fraction"] = 0.0
    io.setdefault("early_stop", {})
    io["early_stop"]["enabled"] = False
    if shuffled:
        io["score_shuffle"] = "stage1"
    else:
        io.pop("score_shuffle", None)
    return cfg


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_fedavg = make_base()
    with open(
        OUTPUT_DIR / "mlj_p1_fedavg_alpha0.1.yaml", "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(cfg_fedavg, f, sort_keys=False)

    cfg_aligned = make_base()
    apply_oracle_overrides(cfg_aligned, shuffled=False)
    with open(
        OUTPUT_DIR / "mlj_p1_oracle_nojs_aligned_alpha0.1.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(cfg_aligned, f, sort_keys=False)

    cfg_shuffled = make_base()
    apply_oracle_overrides(cfg_shuffled, shuffled=True)
    with open(
        OUTPUT_DIR / "mlj_p1_oracle_nojs_shuffled_alpha0.1.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(cfg_shuffled, f, sort_keys=False)

    print(f"Wrote MLJ-P1 configs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
