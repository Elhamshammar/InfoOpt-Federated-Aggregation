#!/usr/bin/env python
"""Regression check for the 2026-10-08 per-example export defect.

Defect: run_pilot cleared the module side cache
_VALIDATION_GAMMA_BY_SOURCE before _per_example_one_source read it, so
every exported validation_selected_scaling row silently used gamma=1
(duplicating FedAvg).

This check:
  1. Populates the module cache with gamma=0.5 and then CLEARS it
     (the exact conditions of the defect).
  2. Calls _per_example_one_source with the retained mapping passed
     explicitly and verifies the exported validation_selected_scaling
     rows carry gamma 0.5 (NOT 1.0) and do not duplicate FedAvg.
  3. Verifies a missing mapping key raises KeyError (no silent fallback).
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stepsize_diag import pilot
from stepsize_diag.pilot_eval import PerExampleCSVWriter
from stepsize_diag.reference_impl import SmallCNNRecovered


def _perturbed(model, scale, seed):
    g = torch.Generator().manual_seed(seed)
    return {k: (v + scale * torch.randn(v.shape, generator=g)
                if torch.is_floating_point(v) else v.clone())
            for k, v in model.state_dict().items()}


def _export(validation_gamma_by_source, out_csv):
    num_classes = 5
    ctor = lambda: SmallCNNRecovered(num_classes=num_classes)
    base = ctor()
    sd0 = {k: v.clone() for k, v in base.state_dict().items()}
    sdA = _perturbed(base, 0.05, 1)
    sdB = _perturbed(base, 0.05, 2)
    rng = np.random.RandomState(0)
    raw_x = rng.randint(0, 256, (16, 32, 32, 3), dtype=np.uint8)
    raw_y = rng.randint(0, num_classes, 16).astype(np.int64)
    data = {"raw_x": raw_x, "raw_y": raw_y,
            "sel_idx": np.arange(8), "eval_idx": np.arange(8, 16),
            "counts": [64, 64]}
    source = {"sd0": sd0, "client_sds": {0: sdA, 1: sdB}, "clients": [0, 1],
              "seed": 201, "round": 5, "source_label": "trajectory_cohort",
              "rep_seed": "trajectory", "cohort_id": "trajectory"}
    w = PerExampleCSVWriter(out_csv)
    try:
        pilot._per_example_one_source(
            source, source, data, ctor, torch.device("cpu"), None,
            gamma_dev_global=0.75, gamma_dev_phase_per_round={5: 0.5},
            validation_gamma_by_source=validation_gamma_by_source,
            pe_writer=w)
    finally:
        w.close()


def main():
    # 1. Exact defect conditions: cache populated, then cleared.
    pilot._VALIDATION_GAMMA_BY_SOURCE[(201, 5, "trajectory_cohort")] = 0.5
    pilot._VALIDATION_GAMMA_BY_SOURCE.clear()
    assert not pilot._VALIDATION_GAMMA_BY_SOURCE, "cache not cleared"

    # 2. Non-1 gamma must still be exported correctly via the explicit map.
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "pe.csv"
        _export({"201|5|trajectory_cohort": 0.5}, out_csv)
        rows = list(csv.DictReader(open(out_csv)))

    vs = [r for r in rows if r["policy"] == "validation_selected_scaling"]
    assert vs, "no validation_selected_scaling rows written"
    gammas = {r["gamma"] for r in vs}
    assert gammas == {"0.500000"}, \
        f"FAIL: expected gamma 0.500000 after cache clear, got {gammas}"

    fa = [r for r in rows if r["policy"] == "fedavg"]
    keyf = lambda r: (r["holdout"], r["example_index"])
    fa_map = {keyf(r): r["loss"] for r in fa}
    n_differ = sum(1 for r in vs if fa_map[keyf(r)] != r["loss"])
    assert n_differ > 0, \
        "FAIL: validation_selected rows still duplicate FedAvg"

    # 3. Missing key must raise; no silent gamma=1 fallback.
    raised = False
    try:
        with tempfile.TemporaryDirectory() as td:
            _export({}, Path(td) / "pe2.csv")
    except KeyError:
        raised = True
    assert raised, "FAIL: missing mapping key did not raise KeyError"

    print("REGRESSION CHECK PASSED: gamma 0.5 exported after cache clear; "
          "rows differ from FedAvg; missing key raises KeyError")


if __name__ == "__main__":
    main()
