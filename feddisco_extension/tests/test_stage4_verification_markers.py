"""Regression tests for Stage 4 verification-marker safety."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PILOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT_ROOT))

import verify_stage4 as v4  # noqa: E402


def _make_partition(num_clients: int = 10, alpha: float = 0.3, seed: int = 100):
    sys.path.insert(0, str(PILOT_ROOT / "snapshot"))
    from data.partition import dirichlet_partition  # noqa: E402

    labels = np.arange(100).repeat(50)
    np.random.default_rng(42).shuffle(labels)
    return dirichlet_partition(
        labels,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_client=50,
        seed=seed,
    )


def _write_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _fake_run_dir(
    runs_dir: Path,
    arm: str,
    seed: int,
    rounds: int,
    verified: bool,
    partition: dict | None = None,
) -> Path:
    slug = v4._method_slug(arm)
    run_dir = (
        runs_dir
        / f"feddisco_stage4_cifar100_{arm}_{slug}_seed{seed}_alpha0.3_20261122_000000"
    )
    run_dir.mkdir(parents=True)
    part = partition if partition is not None else _make_partition(seed=seed)
    _write_json(run_dir / "partition_used.json", part)
    _write_json(run_dir / "init_state_hash.json", {"init_state_sha256": "abc123"})
    _write_json(
        run_dir / "config_used.json",
        {
            "config": {
                "dataset": {"name": "CIFAR100", "num_classes": 100, "root": "./datasets"},
                "feddisco": {"a": 0.5, "b": 0.1},
                "partition": {"alpha": 0.3},
                "seed": seed,
            }
        },
    )
    rng = np.random.default_rng(seed)
    with open(run_dir / "diagnostics.jsonl", "w") as f:
        for rnd in range(1, rounds + 1):
            selected = sorted(rng.choice(10, 5, replace=False).tolist())
            weights = [0.2] * 5
            row = {
                "round": rnd,
                "method": arm,
                "client_ids": selected,
                "weights": weights,
                "sample_counts": [100] * 5,
            }
            f.write(json.dumps(row) + "\n")
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        f.write("round,test_acc,test_loss,fallback_used\n")
        for rnd in range(1, rounds + 1):
            f.write(f"{rnd},0.1,2.3,0\n")
    if verified:
        (run_dir / ".verified").write_text("verified", encoding="utf-8")
    return run_dir


def test_launcher_does_not_create_verified_marker():
    """The launcher must not touch .verified after training exits successfully."""
    script = (PILOT_ROOT / "SERVER_STAGE4_RUN.sh").read_text(encoding="utf-8")
    assert "_mark_verified" not in script, "launcher still contains _mark_verified"
    assert 'touch "${newest}/.verified"' not in script, "launcher still touches .verified"
    # Note: the launcher is allowed to *read* .verified in _run_exists; it must not create it.
    print("[PASS] launcher does not create .verified markers")


def test_cross_arm_mismatch_prevents_verified_markers():
    """If one arm has a different partition, no arm of that seed gets .verified."""
    labels = np.arange(100).repeat(50)
    np.random.default_rng(42).shuffle(labels)
    partition_a = _make_partition(seed=100)
    partition_b = _make_partition(seed=999)

    original_cifar100_labels = v4._cifar100_labels
    v4._cifar100_labels = lambda root: labels

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            runs_dir = tmp_root / "runs"
            runs_dir.mkdir()

            for arm in v4.ARMS:
                _fake_run_dir(
                    runs_dir,
                    arm,
                    seed=100,
                    rounds=20,
                    verified=False,
                    partition=partition_a if arm != "uniform" else partition_b,
                )

            original_seeds = v4.SEEDS
            original_expected = v4.EXPECTED_ROUNDS
            v4.SEEDS = [100]
            v4.EXPECTED_ROUNDS = 20
            try:
                verifier = v4.Stage4Verifier(runs_dir)
                ok = verifier.run()
                assert not ok, "verifier should have failed on partition mismatch"
                for arm in v4.ARMS:
                    run_dir = runs_dir / f"feddisco_stage4_cifar100_{arm}_{v4._method_slug(arm)}_seed100_alpha0.3_20261122_000000"
                    assert not (
                        run_dir / ".verified"
                    ).exists(), f"{arm} should not be .verified"
            finally:
                v4.SEEDS = original_seeds
                v4.EXPECTED_ROUNDS = original_expected
        print("[PASS] cross-arm mismatch prevents .verified markers")
    finally:
        v4._cifar100_labels = original_cifar100_labels


def test_incomplete_runs_not_skipped_as_verified():
    """A run with <100 rounds and no .verified must not be skipped by the launcher."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        stage4_dir = tmp_root / "stage4"
        (stage4_dir / "configs").mkdir(parents=True)
        for cfg_name in [
            "feddisco_stage4_cifar100_fedavg_alpha0.3.yaml",
            "feddisco_stage4_cifar100_uniform_alpha0.3.yaml",
            "feddisco_stage4_cifar100_feddisco_alpha0.3.yaml",
            "feddisco_stage4_cifar100_feddisco_shuffled_alpha0.3.yaml",
        ]:
            shutil.copy2(PILOT_ROOT / "configs" / cfg_name, stage4_dir / "configs" / cfg_name)
        shutil.copy2(PILOT_ROOT / "SERVER_STAGE4_RUN.sh", stage4_dir / "SERVER_STAGE4_RUN.sh")
        runs_dir = stage4_dir / "runs"
        runs_dir.mkdir()
        _fake_run_dir(runs_dir, "fedavg", seed=100, rounds=50, verified=False)

        inner = (
            f'export DRY_RUN=1 SEEDS="100" STAGE4_DIR="{stage4_dir.as_posix()}" CONDA_SKIP=1; '
            f'bash SERVER_STAGE4_RUN.sh'
        )
        cmd = f'bash -c "{inner}"'
        result = subprocess.run(
            cmd,
            cwd=str(stage4_dir),
            shell=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, f"launcher failed with exit code {result.returncode}"
        assert "SKIP: verified run exists" not in output, "incomplete run was skipped"
        assert "RUN: python snapshot/train.py" in output, "launcher did not schedule training"
    print("[PASS] incomplete runs are not skipped")


def test_verified_runs_are_skipped_during_recovery():
    """A run with 100 rounds and .verified must be skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        stage4_dir = tmp_root / "stage4"
        (stage4_dir / "configs").mkdir(parents=True)
        for cfg_name in [
            "feddisco_stage4_cifar100_fedavg_alpha0.3.yaml",
            "feddisco_stage4_cifar100_uniform_alpha0.3.yaml",
            "feddisco_stage4_cifar100_feddisco_alpha0.3.yaml",
            "feddisco_stage4_cifar100_feddisco_shuffled_alpha0.3.yaml",
        ]:
            shutil.copy2(PILOT_ROOT / "configs" / cfg_name, stage4_dir / "configs" / cfg_name)
        shutil.copy2(PILOT_ROOT / "SERVER_STAGE4_RUN.sh", stage4_dir / "SERVER_STAGE4_RUN.sh")
        runs_dir = stage4_dir / "runs"
        runs_dir.mkdir()
        _fake_run_dir(runs_dir, "fedavg", seed=100, rounds=100, verified=True)

        inner = (
            f'export DRY_RUN=1 SEEDS="100" STAGE4_DIR="{stage4_dir.as_posix()}" CONDA_SKIP=1; '
            f'bash SERVER_STAGE4_RUN.sh'
        )
        cmd = f'bash -c "{inner}"'
        result = subprocess.run(
            cmd,
            cwd=str(stage4_dir),
            shell=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, f"launcher failed: {output}"
        assert "SKIP: verified run exists" in output, "verified run was not skipped"
    print("[PASS] verified runs are skipped during recovery")


def test_all_twenty_commands_in_dry_run():
    """The dry-run launcher emits exactly 20 training commands."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        stage4_dir = tmp_root / "stage4"
        (stage4_dir / "configs").mkdir(parents=True)
        for cfg_name in [
            "feddisco_stage4_cifar100_fedavg_alpha0.3.yaml",
            "feddisco_stage4_cifar100_uniform_alpha0.3.yaml",
            "feddisco_stage4_cifar100_feddisco_alpha0.3.yaml",
            "feddisco_stage4_cifar100_feddisco_shuffled_alpha0.3.yaml",
        ]:
            shutil.copy2(PILOT_ROOT / "configs" / cfg_name, stage4_dir / "configs" / cfg_name)
        shutil.copy2(PILOT_ROOT / "SERVER_STAGE4_RUN.sh", stage4_dir / "SERVER_STAGE4_RUN.sh")

        inner = (
            f'export DRY_RUN=1 STAGE4_DIR="{stage4_dir.as_posix()}" CONDA_SKIP=1; '
            f'bash SERVER_STAGE4_RUN.sh'
        )
        cmd = f'bash -c "{inner}"'
        result = subprocess.run(
            cmd,
            cwd=str(stage4_dir),
            shell=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, f"launcher failed: {output}"
        commands = [line for line in output.splitlines() if line.startswith("RUN: python snapshot/train.py")]
        assert len(commands) == 20, f"expected 20 commands, got {len(commands)}"
        # Sanity check all arms and seeds appear.
        for seed in [100, 101, 102, 103, 104]:
            for arm in ["fedavg", "uniform", "feddisco", "feddisco_shuffled"]:
                pattern = f"--config configs/feddisco_stage4_cifar100_{arm}_alpha0.3.yaml"
                seed_commands = [c for c in commands if f"--seed {seed}" in c and pattern in c]
                assert len(seed_commands) == 1, f"missing command for seed {seed} arm {arm}"
    print("[PASS] dry-run emits all 20 intended commands")


if __name__ == "__main__":
    test_launcher_does_not_create_verified_marker()
    test_cross_arm_mismatch_prevents_verified_markers()
    test_incomplete_runs_not_skipped_as_verified()
    test_verified_runs_are_skipped_during_recovery()
    test_all_twenty_commands_in_dry_run()
    print("\nAll verification-marker regression tests passed.")
