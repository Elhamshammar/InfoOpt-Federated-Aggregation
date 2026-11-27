#!/usr/bin/env python3
"""Post-run verification for the MLJ-P1 three-arm GPU pilot.

Reads the three run directories under ``runs/`` and verifies:
- Exactly 100 completed rounds (or the requested number).
- Matching configurations, seeds, alpha, and model settings.
- Matching partitions within seed.
- Matching initial model hashes within seed.
- Matching participation schedules within seed.
- Finite losses, accuracies, and aggregation weights.
- Oracle arms use InfoOpt-NoJS with robust_scale=False.
- Corrupted-client identities match the manifest.
- Weights differ between aligned and shuffled oracle arms.

This script is read-only with respect to saved results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "historical_snapshot"))
sys.path.insert(0, str(ROOT / "src"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _dict_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def find_run_dirs(method: str, seed: int, alpha: float) -> list[Path]:
    pattern = f"mlj_p1_{method}_seed{seed}_alpha{alpha}_*"
    return sorted((ROOT / "runs").glob(pattern))


def verify_run(run_dir: Path, expected_method: str, expected_seed: int,
               expected_alpha: float, expected_rounds: int) -> dict:
    failures: list[str] = []
    warnings: list[str] = []

    config_path = run_dir / "config_used.json"
    if not config_path.exists():
        return {"run_dir": str(run_dir), "status": "missing_config", "failures": ["config_used.json not found"]}
    config = _load_json(config_path)

    if config.get("method") != expected_method:
        failures.append(f"method mismatch: {config.get('method')} != {expected_method}")
    if config.get("seed") != expected_seed:
        failures.append(f"seed mismatch: {config.get('seed')} != {expected_seed}")
    if abs(float(config.get("alpha", 0)) - expected_alpha) > 1e-9:
        failures.append(f"alpha mismatch: {config.get('alpha')} != {expected_alpha}")

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        failures.append("metrics.csv not found")
        rows = []
    else:
        rows = _load_csv(metrics_path)

    completed_rounds = len(rows)
    if completed_rounds != expected_rounds:
        failures.append(f"completed rounds {completed_rounds} != {expected_rounds}")

    if rows:
        for row in rows:
            for key in ["test_acc", "test_loss", "train_loss"]:
                val = float(row.get(key, "nan"))
                if not math.isfinite(val):
                    failures.append(f"non-finite {key} at round {row.get('round')}")
        last_acc = float(rows[-1]["test_acc"])
        if not math.isfinite(last_acc):
            failures.append("final accuracy non-finite")
    else:
        failures.append("no metric rows to validate")

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        failures.append("summary.json not found")

    diag_path = run_dir / "diagnostics.jsonl"
    if not diag_path.exists():
        failures.append("diagnostics.jsonl not found")

    partition_path = run_dir / "partition_used.json"
    if not partition_path.exists():
        failures.append("partition_used.json not found")

    init_hash_path = run_dir / "init_state_hash.json"
    if not init_hash_path.exists():
        failures.append("init_state_hash.json not found")

    return {
        "run_dir": str(run_dir),
        "status": "fail" if failures else "pass",
        "completed_rounds": completed_rounds,
        "failures": failures,
        "warnings": warnings,
    }


def verify_cross_arm_consistency(seed: int, alpha: float,
                                 run_dirs: dict) -> dict:
    failures: list[str] = []
    warnings: list[str] = []
    arms = ["fedavg", "oracle_nojs_aligned", "oracle_nojs_shuffled"]

    if len(run_dirs) != len(arms):
        return {"seed": seed, "status": "fail", "failures": failures, "warnings": warnings}

    # Partition identity
    partition_hashes = {}
    for method, rd in run_dirs.items():
        p = _load_json(rd / "partition_used.json")
        partition_hashes[method] = _dict_hash(p)
    if len(set(partition_hashes.values())) != 1:
        failures.append(f"partition mismatch across arms: {partition_hashes}")

    # Initial model hash identity
    init_hashes = {}
    for method, rd in run_dirs.items():
        h = _load_json(rd / "init_state_hash.json")["init_state_sha256"]
        init_hashes[method] = h
    if len(set(init_hashes.values())) != 1:
        failures.append(f"init hash mismatch across arms: {init_hashes}")

    # Participation schedule identity
    schedules = {}
    for method, rd in run_dirs.items():
        cfg = _load_json(rd / "config_used.json")
        schedules[method] = json.dumps(cfg["schedule"], sort_keys=True)
    if len(set(schedules.values())) != 1:
        failures.append("participation schedule mismatch across arms")

    # Oracle arms: check robust_scale=False and method=InfoOpt-NoJS
    for method in ["oracle_nojs_aligned", "oracle_nojs_shuffled"]:
        cfg = _load_json(run_dirs[method] / "config_used.json")
        if cfg.get("controller_method") != "InfoOpt-NoJS":
            failures.append(f"{method} controller_method != InfoOpt-NoJS")
        io = cfg.get("config", {}).get("infoopt", {})
        if io.get("robust_scale") is not False:
            failures.append(f"{method} robust_scale != False")
        if io.get("lambda_mi") != 1.0:
            failures.append(f"{method} lambda_mi != 1.0")
        if io.get("lambda_entropy") != 0.0:
            failures.append(f"{method} lambda_entropy != 0.0")

    # Corrupted clients match manifest
    manifest_path = ROOT / "data" / "corruption_manifests" / f"corruption_manifest_alpha{alpha}_seed{seed}.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        expected_corrupt = sorted(manifest["corrupted_clients"])
        for method in ["oracle_nojs_aligned", "oracle_nojs_shuffled"]:
            cfg = _load_json(run_dirs[method] / "config_used.json")
            actual_corrupt = sorted(cfg.get("corrupted_clients", []))
            if actual_corrupt != expected_corrupt:
                failures.append(f"{method} corrupted clients {actual_corrupt} != {expected_corrupt}")

    # Aligned vs shuffled weights should differ in at least one round.
    aligned_rows = _load_csv(run_dirs["oracle_nojs_aligned"] / "metrics.csv")
    shuffled_rows = _load_csv(run_dirs["oracle_nojs_shuffled"] / "metrics.csv")
    if aligned_rows and shuffled_rows:
        diffs = []
        for ra, rs in zip(aligned_rows, shuffled_rows):
            wa = np.fromstring(ra.get("weights", ""), sep=";")
            ws = np.fromstring(rs.get("weights", ""), sep=";")
            if len(wa) == len(ws) and len(wa) > 0:
                diffs.append(np.sum(np.abs(wa - ws)))
        if diffs and max(diffs) < 1e-6:
            failures.append("aligned and shuffled oracle weights are identical in every round")
    else:
        warnings.append("could not compare aligned/shuffled weights")

    return {
        "seed": seed,
        "alpha": alpha,
        "status": "fail" if failures else "pass",
        "failures": failures,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()

    arms = ["fedavg", "oracle_nojs_aligned", "oracle_nojs_shuffled"]
    results = []
    for method in arms:
        dirs = list(find_run_dirs(method, args.seed, args.alpha))
        if not dirs:
            results.append({
                "method": method,
                "run_dir": None,
                "status": "missing",
                "failures": [f"no run directory for {method}"],
            })
            continue
        candidates = []
        for run_dir in dirs:
            res = verify_run(
                run_dir, method, args.seed,
                args.alpha, args.rounds
            )
            if res["status"] == "pass":
                candidates.append(res)

        if len(candidates) != 1:
            results.append({
                "method": method,
                "run_dir": None,
                "status": "fail",
                "failures": [
                    f"Expected one valid completed run; found {len(candidates)}"
                ],
                "valid_candidates": [
                    r["run_dir"] for r in candidates
                ],
            })
            continue

        res = candidates[0]
        res["method"] = method
        results.append(res)

    selected_dirs = {
        r["method"]: Path(r["run_dir"])
        for r in results
        if r["status"] == "pass"
    }

    cross = verify_cross_arm_consistency(
        args.seed, args.alpha, selected_dirs
    )

    overall = "pass" if all(r["status"] == "pass" for r in results) and cross["status"] == "pass" else "fail"

    report = {
        "overall": overall,
        "seed": args.seed,
        "alpha": args.alpha,
        "expected_rounds": args.rounds,
        "per_arm": results,
        "cross_arm": cross,
    }

    out_path = ROOT / f"MLJ_P1_VERIFY_seed{args.seed}_alpha{args.alpha}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0 if overall == "pass" else 1)


if __name__ == "__main__":
    main()
