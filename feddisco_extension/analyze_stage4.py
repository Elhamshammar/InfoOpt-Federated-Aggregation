"""Statistical analysis for the FedDisco Stage 4 full experiment.

Reads the 20 saved metrics.csv files (4 arms x 5 seeds), computes final
test accuracies, the prespecified paired comparison between original and
shuffled FedDisco, and descriptive weighting diagnostics.

Run from the InfoOpt_FedDisco_Stage4 project root after verification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from verify_stage4 import (
    SEEDS,
    ARMS,
    _find_run_dir,
    _load_json,
    _load_partition,
    _read_diagnostics,
    _read_metrics,
    _cifar100_labels,
    _build_histograms,
    recompute_weights,
)

# Re-exported constants for clarity.
ARMS_FOR_ANALYSIS = ["fedavg", "uniform", "feddisco", "feddisco_shuffled"]


def _final_accuracy(run_dir: Path) -> float | None:
    metrics = _read_metrics(run_dir)
    if not metrics:
        return None
    try:
        return float(metrics[-1]["test_acc"])
    except (KeyError, ValueError):
        return None


def _load_cfg_for_seed(by_seed: dict[int, dict[str, Path]], seed: int) -> dict:
    fedavg_dir = by_seed[seed].get("fedavg")
    if fedavg_dir is None:
        raise ValueError(f"seed {seed}: missing fedavg run for config")
    return _load_json(fedavg_dir / "config_used.json")["config"]


def _compute_weight_diagnostics(
    by_seed: dict[int, dict[str, Path]],
) -> dict[int, dict[str, dict]]:
    """Compute per-seed/per-arm weighting diagnostics from reconstructed weights."""
    diagnostics: dict[int, dict[str, dict]] = {}
    for seed in SEEDS:
        diagnostics[seed] = {}
        cfg = _load_cfg_for_seed(by_seed, seed)
        labels = _cifar100_labels(Path(cfg.get("dataset", {}).get("root", "./datasets")))
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

            fedavg_devs = []
            effective_clients = []
            fallbacks = 0
            for d in diags:
                round_idx = int(d.get("round", 0))
                selected = [int(cid) for cid in d.get("client_ids", [])]
                _, rec = recompute_weights(
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
                weights = np.array(rec["weights"], dtype=float)
                rel = np.array(rec["relative_sizes"], dtype=float)
                fedavg_devs.append(float(np.sum(np.abs(weights - rel))))
                effective_clients.append(float(1.0 / np.sum(weights ** 2)))
                if rec.get("fallback", False):
                    fallbacks += 1

            diagnostics[seed][arm] = {
                "mean_l1_dev_from_fedavg": float(np.mean(fedavg_devs))
                if fedavg_devs
                else None,
                "mean_effective_clients": float(np.mean(effective_clients))
                if effective_clients
                else None,
                "fallback_rounds": fallbacks,
                "total_rounds": len(diags),
            }

        # Original vs shuffled L1 difference.
        orig_dir = by_seed[seed].get("feddisco")
        shuf_dir = by_seed[seed].get("feddisco_shuffled")
        if orig_dir is not None and shuf_dir is not None:
            orig_diags = _read_diagnostics(orig_dir)
            shuf_diags = _read_diagnostics(shuf_dir)
            l1_diffs = []
            differ_rounds = 0
            nonidentity_rounds = 0
            for d_o, d_s in zip(orig_diags, shuf_diags):
                w_o = np.array(d_o.get("weights", []), dtype=float)
                w_s = np.array(d_s.get("weights", []), dtype=float)
                if w_o.size == w_s.size == 0:
                    continue
                l1_diffs.append(float(np.sum(np.abs(w_s - w_o))))
                if not np.allclose(w_o, w_s, atol=1e-9):
                    differ_rounds += 1
                # Nonidentity permutation: discrepancies_used differs from discrepancies.
                round_idx = int(d_s.get("round", 0))
                selected = [int(cid) for cid in d_s.get("client_ids", [])]
                _, rec_s = recompute_weights(
                    selected,
                    histograms,
                    sizes,
                    num_classes,
                    a,
                    b,
                    shuffle=True,
                    dataset_name=dataset_name,
                    seed=seed,
                    round_idx=round_idx,
                )
                if rec_s.get("shuffle_order") != list(range(len(selected))):
                    nonidentity_rounds += 1

            diagnostics[seed]["feddisco_vs_shuffled"] = {
                "mean_l1_diff": float(np.mean(l1_diffs)) if l1_diffs else None,
                "rounds_differ": differ_rounds,
                "rounds_nonidentity_permutation": nonidentity_rounds,
                "total_rounds": len(orig_diags),
            }
    return diagnostics


def _paired_ttest(differences: list[float]) -> tuple[float, float, float, float]:
    """Return mean, sample std, 95% CI half-width, and two-sided p-value."""
    from scipy import stats

    diffs = np.array(differences, dtype=float)
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    n = len(diffs)
    if n < 2:
        return mean, std, float("nan"), float("nan")
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_half = t_crit * se
    _, p_value = stats.ttest_rel(diffs, np.zeros_like(diffs))
    return mean, std, ci_half, float(p_value)


def main():
    print("=== FedDisco Stage 4 statistical analysis ===")
    by_seed: dict[int, dict[str, Path]] = {}
    missing = []
    for seed in SEEDS:
        by_seed[seed] = {}
        for arm in ARMS_FOR_ANALYSIS:
            d = _find_run_dir(arm, seed)
            if d is None:
                missing.append(f"seed {seed} arm {arm}")
            else:
                by_seed[seed][arm] = d

    if missing:
        print("ERROR: missing runs:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # Final accuracies.
    accuracies: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        accuracies[seed] = {}
        for arm in ARMS_FOR_ANALYSIS:
            acc = _final_accuracy(by_seed[seed][arm])
            if acc is None:
                print(f"ERROR: cannot read final accuracy for seed {seed} {arm}")
                sys.exit(1)
            accuracies[seed][arm] = acc

    print("\nFinal test accuracy (proportion):")
    print(f"{'seed':>6}  {'FedAvg':>10}  {'Uniform':>10}  {'FedDisco':>10}  {'Shuffled':>10}")
    for seed in SEEDS:
        print(
            f"{seed:>6}  "
            f"{accuracies[seed]['fedavg']:>10.4f}  "
            f"{accuracies[seed]['uniform']:>10.4f}  "
            f"{accuracies[seed]['feddisco']:>10.4f}  "
            f"{accuracies[seed]['feddisco_shuffled']:>10.4f}"
        )

    print("\nMean final accuracy per arm:")
    mean_accs = {}
    for arm in ARMS_FOR_ANALYSIS:
        vals = [accuracies[s][arm] for s in SEEDS]
        mean_accs[arm] = float(np.mean(vals))
        print(f"  {arm}: {mean_accs[arm]:.4f}")

    # Primary paired comparison.
    paired_diffs_pp = [
        (accuracies[s]["feddisco"] - accuracies[s]["feddisco_shuffled"]) * 100.0
        for s in SEEDS
    ]
    mean_diff, std_diff, ci_half, p_value = _paired_ttest(paired_diffs_pp)

    print("\nPrimary comparison: FedDisco - Shuffled (percentage points):")
    for seed, d in zip(SEEDS, paired_diffs_pp):
        print(f"  seed {seed}: {d:+.4f}")
    print(f"  mean: {mean_diff:+.4f}")
    print(f"  sample SD: {std_diff:.4f}")
    print(f"  95% CI: [{mean_diff - ci_half:.4f}, {mean_diff + ci_half:.4f}]")
    print(f"  paired t-test p-value: {p_value:.6g}")
    print(
        "  (Unadjusted, single prespecified extension comparison; "
        "do not interpret nonsignificance as equivalence.)"
    )

    # Weighting diagnostics.
    weight_diags = _compute_weight_diagnostics(by_seed)
    print("\nWeighting diagnostics (FedDisco arms):")
    for seed in SEEDS:
        for arm in ["feddisco", "feddisco_shuffled"]:
            d = weight_diags[seed].get(arm, {})
            if not d:
                continue
            print(
                f"  seed {seed} {arm}: "
                f"mean_L1_dev_from_FedAvg={d.get('mean_l1_dev_from_fedavg', float('nan')):.6f}, "
                f"mean_eff_clients={d.get('mean_effective_clients', float('nan')):.2f}, "
                f"fallbacks={d.get('fallback_rounds', 0)}/{d.get('total_rounds', 0)}"
            )
        vs = weight_diags[seed].get("feddisco_vs_shuffled", {})
        if vs:
            print(
                f"  seed {seed} FedDisco vs Shuffled: "
                f"mean_L1_diff={vs.get('mean_l1_diff', float('nan')):.6f}, "
                f"rounds_differ={vs.get('rounds_differ', 0)}/{vs.get('total_rounds', 0)}, "
                f"nonidentity_perm={vs.get('rounds_nonidentity_permutation', 0)}/{vs.get('total_rounds', 0)}"
            )

    summary = {
        "accuracies": accuracies,
        "mean_accuracy": mean_accs,
        "paired_feddisco_minus_shuffled_pp": {
            "per_seed": {str(s): d for s, d in zip(SEEDS, paired_diffs_pp)},
            "mean_pp": mean_diff,
            "std_pp": std_diff,
            "ci95_half_width_pp": ci_half,
            "p_value_two_sided_paired_t": p_value,
        },
        "weight_diagnostics": weight_diags,
    }

    out_path = Path("stage4_analysis_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
