from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path.cwd()
OUT = ROOT / "analysis" / "MLJ_P2"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [100, 101, 102, 103, 104]
METHODS = [
    "fedavg",
    "oracle_nojs_aligned",
    "oracle_nojs_shuffled",
]

records = []
geometry = []
provenance = []

for seed in SEEDS:

    verification = json.loads(
        (ROOT / f"MLJ_P1_VERIFY_seed{seed}_alpha0.1.json")
        .read_text()
    )

    assert verification["overall"] == "pass"
    assert verification["expected_rounds"] == 100
    assert verification["seed"] == seed

    manifest_path = (
        ROOT / "data" / "corruption_manifests"
        / f"corruption_manifest_alpha0.1_seed{seed}.json"
    )

    manifest = json.loads(manifest_path.read_text())
    corrupted = set(manifest["corrupted_clients"])
    assert len(corrupted) == 3

    selected = {
        arm["method"]: Path(arm["run_dir"])
        for arm in verification["per_arm"]
        if arm["status"] == "pass"
    }

    assert set(selected) == set(METHODS)

    seed_rows = {}

    for method in METHODS:

        rd = selected[method]

        summary = json.loads(
            (rd / "summary.json").read_text()
        )

        config = json.loads(
            (rd / "config_used.json").read_text()
        )

        assert summary["rounds_completed"] == 100
        assert summary["seed"] == seed
        assert summary["method"] == method
        assert abs(summary["alpha"] - 0.1) < 1e-9
        assert set(summary["corrupted_clients"]) == corrupted
        assert set(config["corrupted_clients"]) == corrupted

        with (rd / "metrics.csv").open() as f:
            rows = list(csv.DictReader(f))

        assert [int(r["round"]) for r in rows] == list(
            range(1, 101)
        )

        seed_rows[method] = {}

        for row in rows:

            rnd = int(row["round"])

            clients = list(
                map(int, row["selected_clients"].split(";"))
            )

            weights = np.fromstring(
                row["weights"], sep=";"
            )

            assert len(clients) == len(weights) == 5
            assert len(set(clients)) == 5
            assert np.all(np.isfinite(weights))
            assert np.all(weights >= 0)
            assert abs(weights.sum() - 1) < 0.001

            # Normalize rounded logged weights for diagnostics.
            weights = weights / weights.sum()

            positive = weights[weights > 0]
            enc = np.exp(
                -np.sum(positive * np.log(positive))
            )

            corrupt_mass = sum(
                w for c, w in zip(clients, weights)
                if c in corrupted
            )

            seed_rows[method][rnd] = {
                "clients": clients,
                "weights": weights,
                "corrupt_mass": float(corrupt_mass),
                "enc": float(enc),
                "acc": float(row["test_acc"]),
            }

        informative = [
            r for r in seed_rows[method].values()
            if any(c in corrupted for c in r["clients"])
        ]

        final_acc = seed_rows[method][100]["acc"]

        assert abs(
            final_acc - summary["final_acc"]
        ) < 1e-8

        enc_values = [
            r["enc"] for r in seed_rows[method].values()
        ]

        if method != "fedavg":
            assert min(enc_values) >= 2.999

        records.append({
            "seed": seed,
            "method": method,
            "final_acc_pct": final_acc * 100,
            "last10_acc_pct": np.mean([
                seed_rows[method][r]["acc"]
                for r in range(91, 101)
            ]) * 100,
            "best_acc_pct": summary["best_acc"] * 100,
            "final_f1_pct": summary["final_macro_f1"] * 100,
            "final_loss": summary["final_loss"],
            "mean_corrupt_mass": np.mean([
                r["corrupt_mass"] for r in informative
            ]),
            "mean_enc": np.mean(enc_values),
            "min_enc": min(enc_values),
            "informative_rounds": len(informative),
        })

        provenance.append({
            "seed": seed,
            "method": method,
            "run_dir": str(rd),
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "metrics_sha256": hashlib.sha256(
                (rd / "metrics.csv").read_bytes()
            ).hexdigest(),
            "summary_sha256": hashlib.sha256(
                (rd / "summary.json").read_bytes()
            ).hexdigest(),
        })

    for rnd in range(1, 101):

        f = seed_rows["fedavg"][rnd]
        a = seed_rows["oracle_nojs_aligned"][rnd]
        s = seed_rows["oracle_nojs_shuffled"][rnd]

        assert f["clients"] == a["clients"] == s["clients"]

        same_weight_multiset = np.allclose(
            np.sort(a["weights"]),
            np.sort(s["weights"]),
            atol=0.0002,
            rtol=0,
        )

        geometry.append({
            "seed": seed,
            "round": rnd,
            "n_corrupted": sum(
                c in corrupted for c in a["clients"]
            ),
            "fedavg_corrupt_mass": f["corrupt_mass"],
            "aligned_corrupt_mass": a["corrupt_mass"],
            "shuffled_corrupt_mass": s["corrupt_mass"],
            "corrupt_mass_reduction":
                s["corrupt_mass"] - a["corrupt_mass"],
            "aligned_shuffled_l1": np.sum(
                np.abs(a["weights"] - s["weights"])
            ),
            "aligned_enc": a["enc"],
            "shuffled_enc": s["enc"],
            "same_weight_multiset": bool(
                same_weight_multiset
            ),
        })

df = pd.DataFrame(records)
geo = pd.DataFrame(geometry)

assert len(df) == 15
assert len(geo) == 500

df.to_csv(OUT / "per_seed_results.csv", index=False)
geo.to_csv(OUT / "round_geometry.csv", index=False)

(OUT / "run_provenance.json").write_text(
    json.dumps(provenance, indent=2)
)

print("\n===== PER-SEED FINAL ACCURACY (%) =====")

acc = df.pivot(
    index="seed",
    columns="method",
    values="final_acc_pct"
)

print(acc.round(4).to_string())

print("\n===== AGGREGATE RESULTS =====")

aggregate = df.groupby("method")[
    [
        "final_acc_pct",
        "last10_acc_pct",
        "final_f1_pct",
        "final_loss",
        "mean_corrupt_mass",
        "mean_enc",
        "min_enc",
    ]
].agg(["mean", "std"])

print(aggregate.round(6).to_string())

aggregate.to_csv(OUT / "aggregate_results.csv")

def paired_analysis(endpoint, method_a, method_b):

    p = df.pivot(
        index="seed",
        columns="method",
        values=endpoint
    )

    d = (
        p[method_a] - p[method_b]
    ).to_numpy()

    n = len(d)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)

    half_width = stats.t.ppf(
        0.975, df=n-1
    ) * se

    test = stats.ttest_rel(
        p[method_a], p[method_b]
    )

    result = {
        "endpoint": endpoint,
        "comparison": f"{method_a} minus {method_b}",
        "n_seeds": n,
        "paired_differences": d.tolist(),
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_lower": float(mean - half_width),
        "ci95_upper": float(mean + half_width),
        "paired_t": float(test.statistic),
        "two_sided_p": float(test.pvalue),
        "cohen_dz": mean / sd if sd > 0 else None,
        "positive_seeds": int(np.sum(d > 0)),
        "negative_seeds": int(np.sum(d < 0)),
    }

    return result

primary = paired_analysis(
    "final_acc_pct",
    "oracle_nojs_aligned",
    "oracle_nojs_shuffled"
)

secondary = [
    paired_analysis(
        "last10_acc_pct",
        "oracle_nojs_aligned",
        "oracle_nojs_shuffled"
    ),
    paired_analysis(
        "final_acc_pct",
        "oracle_nojs_aligned",
        "fedavg"
    ),
    paired_analysis(
        "final_acc_pct",
        "oracle_nojs_shuffled",
        "fedavg"
    ),
]

print("\n===== PRIMARY PAIRED ANALYSIS =====")
print(json.dumps(primary, indent=2))

print("\n===== SECONDARY ANALYSES =====")
print(json.dumps(secondary, indent=2))

mixed = geo[geo["n_corrupted"] > 0]

geometry_summary = mixed.groupby("seed")[
    [
        "fedavg_corrupt_mass",
        "aligned_corrupt_mass",
        "shuffled_corrupt_mass",
        "corrupt_mass_reduction",
        "aligned_shuffled_l1",
    ]
].mean()

print("\n===== PER-SEED WEIGHT DIAGNOSTICS =====")
print(geometry_summary.round(6).to_string())

print("\n===== MEAN WEIGHT DIAGNOSTICS =====")
print(geometry_summary.mean().round(6).to_string())

print("\nWeight-multiset matching rounds:",
      int(geo["same_weight_multiset"].sum()),
      "/", len(geo))

print("Minimum oracle ENC:",
      min(geo["aligned_enc"].min(),
          geo["shuffled_enc"].min()))

geometry_summary.to_csv(
    OUT / "per_seed_geometry.csv"
)

report = {
    "completed_runs": len(df),
    "completed_rounds": len(df) * 100,
    "primary": primary,
    "secondary_unadjusted": secondary,
    "geometry_seed_equal_weighted_means":
        geometry_summary.mean().to_dict(),
    "weight_multiset_matching_rounds":
        int(geo["same_weight_multiset"].sum()),
    "weight_multiset_total_rounds": len(geo),
    "minimum_oracle_enc": float(
        min(geo["aligned_enc"].min(),
            geo["shuffled_enc"].min())
    ),
}

(OUT / "MLJ_P2_STATISTICAL_REPORT.json").write_text(
    json.dumps(report, indent=2)
)

print("\nANALYSIS COMPLETED")
print("Results saved to:", OUT)
