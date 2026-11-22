"""Post-run verification for the FedDisco Stage 4 full experiment.

Generalizes verify_stage3b.py to five seeds and 100 rounds.  Reconstructs
FedDisco weights from saved artifacts (partition, config, diagnostics) rather
than relying on FedDisco-specific diagnostic fields, which are not written to
diagnostics.jsonl by the training loop.

Run from the InfoOpt_FedDisco_Stage4 project root after all 20 runs finish.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

# Make snapshot modules importable when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "snapshot"))

from data.partition import class_histogram  # noqa: E402

RUNS_DIR = Path("runs")
EXPECTED_ROUNDS = 100
SEEDS = [100, 101, 102, 103, 104]
ARMS = ["fedavg", "uniform", "feddisco", "feddisco_shuffled"]
METHOD_SLUG = {
    "fedavg": "fedavg",
    "uniform": "fedavg",
    "feddisco": "feddisco",
    "feddisco_shuffled": "feddisco_shuffled",
}
ALPHA = 0.3


def _shuffle_seed(dataset: str, seed: int, round_idx: int) -> int:
    key = f"feddisco-shuffle-v1|{dataset}|{seed}|{round_idx}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _method_slug(arm: str) -> str:
    return METHOD_SLUG.get(arm, arm)


def _find_run_dir(arm: str, seed: int, runs_dir: Path = RUNS_DIR) -> Path | None:
    slug = _method_slug(arm)
    pattern = re.compile(
        rf"^feddisco_stage4_cifar100_{arm}_{slug}_seed{seed}_alpha{ALPHA}_.+$"
    )
    if not runs_dir.exists():
        return None
    matches = [d for d in runs_dir.iterdir() if d.is_dir() and pattern.match(d.name)]
    if not matches:
        return None
    if len(matches) > 1:
        # Return the most recent if multiple exist, but warn.
        matches = sorted(matches)
    return matches[-1]


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _load_partition(run_dir: Path) -> dict:
    return _load_json(run_dir / "partition_used.json")


def _load_init_hash(run_dir: Path) -> str:
    return _load_json(run_dir / "init_state_hash.json")["init_state_sha256"]


def _read_diagnostics(run_dir: Path) -> list[dict]:
    path = run_dir / "diagnostics.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _cifar100_labels(root: Path) -> np.ndarray:
    """Load CIFAR-100 training labels from torchvision cache if present."""
    candidates = [
        root / "CIFAR100" / "cifar-100-python" / "train",
        root / "cifar-100-python" / "train",
    ]
    for c in candidates:
        if c.exists():
            if c.suffix == ".json":
                data = _load_json(c)
            else:
                import pickle

                with open(c, "rb") as f:
                    data = pickle.load(f, encoding="bytes")
            return np.array(data[b"fine_labels"], dtype=int)
    raise FileNotFoundError(
        f"CIFAR-100 training labels not found under {root}. "
        "Download with torchvision or place the dataset at ./datasets/CIFAR100."
    )


def _build_histograms(
    partition: dict, labels: np.ndarray, num_classes: int
) -> dict[int, np.ndarray]:
    return {
        int(cid): class_histogram(labels, idxs, num_classes)
        for cid, idxs in partition.items()
    }


def _raw_kl(hist: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(hist / hist.sum(), eps, 1.0)
    q = np.clip(target, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def recompute_weights(
    selected: list[int],
    histograms: dict[int, np.ndarray],
    sizes: dict[int, int],
    num_classes: int,
    a: float,
    b: float,
    shuffle: bool,
    dataset_name: str,
    seed: int,
    round_idx: int,
) -> tuple[list[float], dict]:
    """Reconstruct FedDisco weights from first principles."""
    sample_counts = np.array([sizes[cid] for cid in selected], dtype=float)
    relative_sizes = sample_counts / sample_counts.sum()
    target = np.full(num_classes, 1.0 / num_classes, dtype=float)

    raw_kls = np.array(
        [_raw_kl(histograms[cid], target) for cid in selected], dtype=float
    )
    total_kl = float(raw_kls.sum())
    if total_kl <= 1e-12:
        discrepancies = np.zeros_like(raw_kls)
    else:
        discrepancies = raw_kls / total_kl

    original_discrepancies = discrepancies.copy()
    shuffle_order: list[int] | None = None

    if shuffle:
        shuffle_rng = random.Random(_shuffle_seed(dataset_name, seed, round_idx))
        shuffle_order = list(range(len(selected)))
        shuffle_rng.shuffle(shuffle_order)
        discrepancies = discrepancies[shuffle_order]

    raw_weights = np.maximum(relative_sizes - a * discrepancies + b, 0.0)
    fallback = False
    if raw_weights.sum() <= 1e-12:
        weights = relative_sizes.copy()
        fallback = True
    else:
        weights = raw_weights / raw_weights.sum()

    diagnostics = {
        "relative_sizes": relative_sizes.tolist(),
        "discrepancies": original_discrepancies.tolist(),
        "discrepancies_used": discrepancies.tolist(),
        "shuffle_order": shuffle_order,
        "raw_weights": raw_weights.tolist(),
        "weights": weights.tolist(),
        "fallback": fallback,
    }
    return weights.tolist(), diagnostics


class Stage4Verifier:
    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.runs_dir = runs_dir
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.failed_runs: set[tuple[int, str]] = set()
        self.labels: np.ndarray | None = None

    def add_failure(
        self,
        msg: str,
        seed: int | None = None,
        arm: str | None = None,
    ) -> None:
        self.failures.append(msg)
        print(f"  [FAIL] {msg}")
        if seed is not None and arm is not None:
            self.failed_runs.add((seed, arm))
        elif seed is not None and arm is None:
            for a in ARMS:
                self.failed_runs.add((seed, a))

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def find_all_runs(self) -> dict[int, dict[str, Path]]:
        """Locate one run directory per (seed, arm)."""
        print("\n=== Locating run directories ===")
        by_seed: dict[int, dict[str, Path]] = {}
        for seed in SEEDS:
            by_seed[seed] = {}
            for arm in ARMS:
                d = _find_run_dir(arm, seed, runs_dir=self.runs_dir)
                if d is None:
                    self.add_failure(
                        f"seed {seed} arm {arm}: no run directory found",
                        seed=seed,
                        arm=arm,
                    )
                else:
                    by_seed[seed][arm] = d
                    print(f"  seed {seed} {arm}: {d.name}")
        return by_seed

    def load_labels(self, cfg: dict) -> np.ndarray:
        if self.labels is None:
            dataset_root = Path(cfg.get("dataset", {}).get("root", "./datasets"))
            self.labels = _cifar100_labels(dataset_root)
        return self.labels

    def verify_rounds(
        self, by_seed: dict[int, dict[str, Path]]
    ) -> dict[int, dict[str, int]]:
        print("\n=== Round counts ===")
        counts: dict[int, dict[str, int]] = {}
        for seed in SEEDS:
            counts[seed] = {}
            for arm in ARMS:
                run_dir = by_seed[seed].get(arm)
                if run_dir is None:
                    continue
                diags = _read_diagnostics(run_dir)
                metrics = _read_metrics(run_dir)
                diag_count = len(diags)
                metric_count = len(metrics)
                counts[seed][arm] = diag_count
                ok = diag_count == EXPECTED_ROUNDS and metric_count == EXPECTED_ROUNDS
                status = "OK" if ok else "FAIL"
                print(
                    f"  seed {seed} {arm}: diagnostics={diag_count} "
                    f"metrics={metric_count} [{status}]"
                )
                if not ok:
                    self.add_failure(
                        f"seed {seed} {arm}: expected {EXPECTED_ROUNDS} rounds, "
                        f"got diagnostics={diag_count} metrics={metric_count}",
                        seed=seed,
                        arm=arm,
                    )
        return counts

    def verify_partitions_and_hashes(
        self, by_seed: dict[int, dict[str, Path]]
    ) -> None:
        print("\n=== Cross-arm identity checks (per seed) ===")
        for seed in SEEDS:
            arms_present = [a for a in ARMS if a in by_seed[seed]]
            if not arms_present:
                continue
            base_arm = "fedavg"
            if base_arm not in by_seed[seed]:
                base_arm = arms_present[0]
            base_partition = _load_partition(by_seed[seed][base_arm])
            base_hash = _load_init_hash(by_seed[seed][base_arm])
            for arm in arms_present:
                part = _load_partition(by_seed[seed][arm])
                h = _load_init_hash(by_seed[seed][arm])
                part_ok = part == base_partition
                hash_ok = h == base_hash
                status = "OK" if (part_ok and hash_ok) else "FAIL"
                print(
                    f"  seed {seed} {arm}: partition={part_ok} "
                    f"init_hash={hash_ok} [{status}]"
                )
                if not part_ok:
                    # Cross-arm identity failure affects every arm of this seed.
                    self.add_failure(
                        f"seed {seed} {arm}: partition does not match {base_arm}",
                        seed=seed,
                    )
                if not hash_ok:
                    self.add_failure(
                        f"seed {seed} {arm}: init hash does not match {base_arm}",
                        seed=seed,
                    )

    def verify_schedules(self, by_seed: dict[int, dict[str, Path]]) -> None:
        print("\n=== Participation schedule checks (per seed) ===")
        for seed in SEEDS:
            arms_present = [a for a in ARMS if a in by_seed[seed]]
            if not arms_present:
                continue
            base_arm = "fedavg"
            if base_arm not in by_seed[seed]:
                base_arm = arms_present[0]
            base_diags = _read_diagnostics(by_seed[seed][base_arm])
            base_selected = [tuple(d.get("client_ids", [])) for d in base_diags]
            for arm in arms_present:
                diags = _read_diagnostics(by_seed[seed][arm])
                selected = [tuple(d.get("client_ids", [])) for d in diags]
                match = selected == base_selected
                status = "OK" if match else "FAIL"
                print(
                    f"  seed {seed} {arm}: schedule matches {base_arm}={match} "
                    f"[{status}]"
                )
                if not match:
                    # Cross-arm schedule failure affects every arm of this seed.
                    self.add_failure(
                        f"seed {seed} {arm}: participation schedule does not match "
                        f"{base_arm}",
                        seed=seed,
                    )

    def verify_feddisco_weights(
        self, by_seed: dict[int, dict[str, Path]]
    ) -> dict[int, dict[str, list[dict]]]:
        print("\n=== FedDisco weight reconstruction ===")
        recomputed_by_seed_arm: dict[int, dict[str, list[dict]]] = {}
        for seed in SEEDS:
            recomputed_by_seed_arm[seed] = {}
            cfg_path = by_seed[seed].get("fedavg")
            if cfg_path is None:
                self.add_failure(
                    f"seed {seed}: no fedavg run to load config",
                    seed=seed,
                )
                continue
            cfg = _load_json(cfg_path / "config_used.json")["config"]
            labels = self.load_labels(cfg)
            num_classes = int(cfg["dataset"]["num_classes"])
            a = float(cfg.get("feddisco", {}).get("a", 0.5))
            b = float(cfg.get("feddisco", {}).get("b", 0.1))
            dataset_name = cfg.get("dataset", {}).get("name", "CIFAR100")

            for arm in ["feddisco", "feddisco_shuffled"]:
                run_dir = by_seed[seed].get(arm)
                if run_dir is None:
                    continue
                partition = _load_partition(run_dir)
                sizes = {int(cid): len(idxs) for cid, idxs in partition.items()}
                histograms = _build_histograms(partition, labels, num_classes)
                diags = _read_diagnostics(run_dir)

                mismatches = 0
                missing = 0
                recomputed_rows = []
                for d in diags:
                    round_idx = int(d.get("round", 0))
                    selected = [int(cid) for cid in d.get("client_ids", [])]
                    logged = d.get("weights")
                    if logged is None:
                        missing += 1
                        continue
                    logged = np.array(logged, dtype=float)
                    recomputed, rec_diag = recompute_weights(
                        selected,
                        histograms,
                        sizes,
                        num_classes,
                        a,
                        b,
                        shuffle=(arm == "feddisco_shuffled"),
                        dataset_name=dataset_name,
                        seed=seed,
                        round_idx=round_idx,
                    )
                    recomputed = np.array(recomputed, dtype=float)
                    recomputed_rows.append(
                        {
                            "round": round_idx,
                            "selected": selected,
                            **rec_diag,
                            "logged_weights": logged.tolist(),
                        }
                    )
                    if not np.allclose(logged, recomputed, atol=1e-9):
                        mismatches += 1
                        if mismatches <= 3:
                            self.add_failure(
                                f"seed {seed} {arm} round {round_idx}: "
                                f"logged weights {logged.tolist()} do not match "
                                f"recomputed {recomputed.tolist()}",
                                seed=seed,
                                arm=arm,
                            )

                recomputed_by_seed_arm[seed][arm] = recomputed_rows
                status = "OK" if mismatches == 0 and missing == 0 else "FAIL"
                print(
                    f"  seed {seed} {arm}: mismatches={mismatches} "
                    f"missing={missing}/{len(diags)} [{status}]"
                )
                if mismatches > 3:
                    self.add_failure(
                        f"seed {seed} {arm}: {mismatches}/{len(diags)} "
                        f"weight reconstruction mismatches",
                        seed=seed,
                        arm=arm,
                    )
                if missing > 0:
                    self.add_failure(
                        f"seed {seed} {arm}: {missing}/{len(diags)} rounds "
                        f"missing logged weights",
                        seed=seed,
                        arm=arm,
                    )
        return recomputed_by_seed_arm

    def verify_shuffle(
        self,
        by_seed: dict[int, dict[str, Path]],
        recomputed: dict[int, dict[str, list[dict]]],
    ) -> None:
        print("\n=== Shuffled-arm checks (per seed) ===")
        for seed in SEEDS:
            rows = recomputed.get(seed, {}).get("feddisco_shuffled", [])
            if not rows:
                continue
            ok = True
            non_identity = 0
            for row in rows:
                original = np.array(row["discrepancies"], dtype=float)
                used = np.array(row["discrepancies_used"], dtype=float)
                order = row.get("shuffle_order")
                if order is None:
                    self.add_failure(
                        f"seed {seed} shuffled round {row['round']}: "
                        f"missing shuffle_order",
                        seed=seed,
                        arm="feddisco_shuffled",
                    )
                    ok = False
                    continue
                if not np.allclose(
                    sorted(original.tolist()), sorted(used.tolist()), atol=1e-9
                ):
                    self.add_failure(
                        f"seed {seed} shuffled round {row['round']}: "
                        f"discrepancy multiset not preserved",
                        seed=seed,
                        arm="feddisco_shuffled",
                    )
                    ok = False
                if order != list(range(len(order))):
                    non_identity += 1

            # Compare weights with original FedDisco arm for same rounds.
            orig_rows = recomputed.get(seed, {}).get("feddisco", [])
            differ_from_orig = 0
            l1_diffs = []
            if len(orig_rows) == len(rows):
                for row_s, row_o in zip(rows, orig_rows):
                    w_s = np.array(row_s["weights"], dtype=float)
                    w_o = np.array(row_o["weights"], dtype=float)
                    if not np.allclose(w_s, w_o, atol=1e-9):
                        differ_from_orig += 1
                        l1_diffs.append(float(np.sum(np.abs(w_s - w_o))))
            else:
                self.add_warning(
                    f"seed {seed}: feddisco and shuffled have different round counts"
                )

            status = "OK" if ok else "FAIL"
            print(
                f"  seed {seed}: discrepancy_multiset={ok} "
                f"nonidentity={non_identity}/{len(rows)} "
                f"differ_from_orig={differ_from_orig}/{len(rows)} "
                f"mean_l1_diff={np.mean(l1_diffs) if l1_diffs else 0.0:.6f} "
                f"[{status}]"
            )

    def verify_finite_outputs(
        self, by_seed: dict[int, dict[str, Path]]
    ) -> None:
        print("\n=== Finite outputs ===")
        for seed in SEEDS:
            for arm in ARMS:
                run_dir = by_seed[seed].get(arm)
                if run_dir is None:
                    continue
                diags = _read_diagnostics(run_dir)
                bad = 0
                for d in diags:
                    weights = np.array(d.get("weights", []), dtype=float)
                    if weights.size == 0:
                        bad += 1
                        continue
                    if not np.all(np.isfinite(weights)) or not np.all(
                        weights >= -1e-12
                    ):
                        bad += 1
                status = "OK" if bad == 0 else "FAIL"
                print(
                    f"  seed {seed} {arm}: bad_rounds={bad}/{len(diags)} "
                    f"[{status}]"
                )
                if bad > 0:
                    self.add_failure(
                        f"seed {seed} {arm}: {bad}/{len(diags)} rounds with "
                        f"non-finite or negative weights",
                        seed=seed,
                        arm=arm,
                    )

    def report_diagnostics(
        self,
        by_seed: dict[int, dict[str, Path]],
        recomputed: dict[int, dict[str, list[dict]]],
    ) -> None:
        print("\n=== FedDisco diagnostic measurements ===")
        for seed in SEEDS:
            for arm in ["feddisco", "feddisco_shuffled"]:
                rows = recomputed.get(seed, {}).get(arm, [])
                if not rows:
                    continue
                all_discs = []
                fedavg_devs = []
                effective_clients = []
                fallbacks = 0
                for row in rows:
                    discs = np.array(row["discrepancies_used"], dtype=float)
                    rel = np.array(row["relative_sizes"], dtype=float)
                    weights = np.array(row["weights"], dtype=float)
                    all_discs.extend(discs.tolist())
                    fedavg_devs.append(float(np.sum(np.abs(weights - rel))))
                    effective_clients.append(float(1.0 / np.sum(weights ** 2)))
                    if row.get("fallback", False):
                        fallbacks += 1

                print(f"\n  seed {seed} {arm}:")
                print(
                    f"    discrepancy mean={np.mean(all_discs):.6f}, "
                    f"range=[{np.min(all_discs):.6f}, {np.max(all_discs):.6f}]"
                )
                print(
                    f"    mean L1 deviation from FedAvg weights: "
                    f"{np.mean(fedavg_devs):.6f}"
                )
                print(
                    f"    mean effective client number: "
                    f"{np.mean(effective_clients):.2f}"
                )
                print(f"    fallback events: {fallbacks}/{len(rows)}")

    def report_final_accuracies(
        self, by_seed: dict[int, dict[str, Path]]
    ) -> dict[int, dict[str, float]]:
        print("\n=== Final test accuracy per seed/arm (descriptive only) ===")
        accs: dict[int, dict[str, float]] = {}
        for seed in SEEDS:
            accs[seed] = {}
            for arm in ARMS:
                run_dir = by_seed[seed].get(arm)
                if run_dir is None:
                    continue
                metrics = _read_metrics(run_dir)
                if not metrics:
                    continue
                last = metrics[-1]
                rnd = last.get("round", "?")
                acc = float(last.get("test_acc", float("nan")))
                loss = float(last.get("test_loss", float("nan")))
                accs[seed][arm] = acc
                print(
                    f"  seed {seed} {arm}: round={rnd} acc={acc:.4f} "
                    f"loss={loss:.4f}"
                )
        return accs

    def _run_failed(self, seed: int, arm: str) -> bool:
        return (seed, arm) in self.failed_runs

    def _mark_verified(self, by_seed: dict[int, dict[str, Path]]) -> None:
        print("\n=== Marking verified runs ===")
        for seed in SEEDS:
            for arm in ARMS:
                run_dir = by_seed[seed].get(arm)
                if run_dir is None:
                    print(f"  seed {seed} {arm}: not marked (missing run)")
                    continue
                if self._run_failed(seed, arm):
                    print(f"  seed {seed} {arm}: not marked (failed)")
                    continue
                marker = run_dir / ".verified"
                marker.write_text("verified", encoding="utf-8")
                print(f"  seed {seed} {arm}: {marker}")

    def run(self) -> bool:
        if not self.runs_dir.exists():
            self.add_failure(f"{self.runs_dir} not found")
            return False

        by_seed = self.find_all_runs()
        self.verify_rounds(by_seed)
        self.verify_partitions_and_hashes(by_seed)
        self.verify_schedules(by_seed)
        recomputed = self.verify_feddisco_weights(by_seed)
        self.verify_shuffle(by_seed, recomputed)
        self.verify_finite_outputs(by_seed)
        self.report_diagnostics(by_seed, recomputed)
        self.report_final_accuracies(by_seed)
        self._mark_verified(by_seed)

        print("\n=== Verification summary ===")
        print(f"  failures: {len(self.failures)}")
        if self.warnings:
            print(f"  warnings: {len(self.warnings)}")
            for w in self.warnings:
                print(f"    [WARN] {w}")

        if self.failures:
            print("\n[FAIL] Some verification checks failed.")
            return False
        print("\n[PASS] All critical verification checks passed.")
        return True


def main():
    verifier = Stage4Verifier()
    ok = verifier.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
