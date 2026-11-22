"""Synthetic end-to-end test of verify_stage4.py logic without CIFAR-100."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PILOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT_ROOT))

import verify_stage4 as v4  # noqa: E402


def _make_synthetic_labels(num_classes: int = 100, samples_per_class: int = 50) -> np.ndarray:
    rng = np.random.default_rng(42)
    labels = np.arange(num_classes).repeat(samples_per_class)
    rng.shuffle(labels)
    return labels


def _make_partition(
    labels: np.ndarray, num_clients: int = 10, alpha: float = 0.3, seed: int = 100
) -> dict:
    sys.path.insert(0, str(PILOT_ROOT / "snapshot"))
    from data.partition import dirichlet_partition  # noqa: E402

    return dirichlet_partition(
        labels,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_client=50,
        seed=seed,
    )


def _build_histograms(partition: dict, labels: np.ndarray, num_classes: int) -> dict:
    sys.path.insert(0, str(PILOT_ROOT / "snapshot"))
    from data.partition import class_histogram  # noqa: E402

    return {
        int(cid): class_histogram(labels, idxs, num_classes)
        for cid, idxs in partition.items()
    }


def _write_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _write_diagnostics(
    path: Path,
    partition: dict,
    num_classes: int,
    a: float,
    b: float,
    shuffle: bool,
    seed: int,
) -> None:
    from data.partition import class_histogram  # noqa: E402

    labels = v4._cifar100_labels(Path("./datasets"))
    histograms = _build_histograms(partition, labels, num_classes)
    sizes = {int(cid): len(idxs) for cid, idxs in partition.items()}
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for rnd in range(1, 21):
            selected = sorted(rng.choice(10, 5, replace=False).tolist())
            weights, diag = v4.recompute_weights(
                selected,
                histograms,
                sizes,
                num_classes,
                a,
                b,
                shuffle=shuffle,
                dataset_name="CIFAR100",
                seed=seed,
                round_idx=rnd,
            )
            row = {
                "round": rnd,
                "method": "feddisco" if not shuffle else "feddisco_shuffled",
                "client_ids": selected,
                "weights": weights,
                "sample_counts": [sizes[cid] for cid in selected],
            }
            f.write(json.dumps(row) + "\n")


def test_verifier_on_synthetic_runs():
    """Run the full verifier on a synthetic 20-round Stage 4 dataset."""
    labels = _make_synthetic_labels()
    partition = _make_partition(labels)
    num_classes = 100
    a, b = 0.5, 0.1

    # Monkeypatch CIFAR-100 label loader to use synthetic labels.
    original_cifar100_labels = v4._cifar100_labels
    v4._cifar100_labels = lambda root: labels

    original_seeds = v4.SEEDS
    original_expected = v4.EXPECTED_ROUNDS
    v4.SEEDS = [100]
    v4.EXPECTED_ROUNDS = 20

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            runs_dir = tmp_root / "runs"
            runs_dir.mkdir()

            for arm in v4.ARMS:
                slug = v4._method_slug(arm)
                run_dir = (
                    runs_dir
                    / f"feddisco_stage4_cifar100_{arm}_{slug}_seed100_alpha0.3_20261122_000000"
                )
                run_dir.mkdir()

                _write_json(run_dir / "partition_used.json", partition)
                _write_json(run_dir / "init_state_hash.json", {"init_state_sha256": "abc123"})
                cfg = {
                    "dataset": {"name": "CIFAR100", "num_classes": num_classes, "root": "./datasets"},
                    "feddisco": {"a": a, "b": b},
                    "partition": {"alpha": 0.3},
                    "seed": 100,
                }
                _write_json(
                    run_dir / "config_used.json",
                    {"config": cfg},
                )

                if arm in ("feddisco", "feddisco_shuffled"):
                    _write_diagnostics(
                        run_dir / "diagnostics.jsonl",
                        partition,
                        num_classes,
                        a,
                        b,
                        shuffle=(arm == "feddisco_shuffled"),
                        seed=100,
                    )
                else:
                    # Baseline arms: just reuse the same schedule and uniform weights.
                    rng = np.random.default_rng(100)
                    with open(run_dir / "diagnostics.jsonl", "w") as f:
                        for rnd in range(1, 21):
                            selected = sorted(rng.choice(10, 5, replace=False).tolist())
                            weights = [0.2] * 5
                            row = {
                                "round": rnd,
                                "method": arm,
                                "client_ids": selected,
                                "weights": weights,
                                "sample_counts": [len(partition[cid]) for cid in selected],
                            }
                            f.write(json.dumps(row) + "\n")

                # Minimal metrics.csv for round-count check.
                with open(run_dir / "metrics.csv", "w", newline="") as f:
                    f.write("round,test_acc,test_loss,fallback_used\n")
                    for rnd in range(1, 21):
                        f.write(f"{rnd},0.1,2.3,0\n")

            verifier = v4.Stage4Verifier(runs_dir)
            ok = verifier.run()
            assert ok, f"verifier failed: {verifier.failures}"
            assert len(verifier.failures) == 0
            print("[PASS] synthetic verifier run passed")
    finally:
        v4._cifar100_labels = original_cifar100_labels
        v4.SEEDS = original_seeds
        v4.EXPECTED_ROUNDS = original_expected


if __name__ == "__main__":
    test_verifier_on_synthetic_runs()
    print("\nAll synthetic verifier tests passed.")
