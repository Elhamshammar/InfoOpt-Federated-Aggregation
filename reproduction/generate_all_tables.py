#!/usr/bin/env python3
"""Generate all reproducible tables for the MLJ-R1.1 revision.

Run from the project root:
    python reproduction/generate_all_tables.py

Outputs:
- Corrected MLJ-A1.1 synthesis tables in reproduction/outputs/MLJ_A1_1/tables.
- Positive-control tables in reproduction/outputs/tables.
- A printed status report for legacy tables whose source CSVs are not in this
  reproduction package.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

REPRO = Path(__file__).resolve().parent
OUT = REPRO / "outputs" / "MLJ_A1_1"
TAB_OUT = OUT / "tables"
TAB_OUT.mkdir(parents=True, exist_ok=True)
POS_TAB_OUT = REPRO / "outputs" / "tables"
POS_TAB_OUT.mkdir(parents=True, exist_ok=True)
POS_CTRL = REPRO / "positive_control"


def table_setting_effects():
    df = pd.read_csv(OUT / "synthesis_setting_level_corrected.csv")
    df = df.sort_values(["dataset", "stage", "alpha"])
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Setting": r["setting_id"],
            "Stage": r["stage"],
            "Dataset": r["dataset"],
            "Alpha": r["alpha"],
            "Model": r["model"],
            "n": int(r["n_seeds"]),
            "Mean diff (pp)": f"{r['mean_diff_pp']:.3f}",
            "SD (pp)": f"{r['sd_diff_pp']:.3f}",
            "95% CI (pp)": f"[{r['ci95_lower']:.3f}, {r['ci95_upper']:.3f}]",
            "Raw p": f"{r['p_two_sided']:.4f}",
            "Holm p": f"{r['p_holm']:.4f}" if pd.notna(r["p_holm"]) else "",
            "TOST p (±0.5 pp)": f"{r['p_tost_05pp']:.4f}" if pd.notna(r["p_tost_05pp"]) else "",
            "Equivalent at ±0.5 pp": "Yes" if r["equiv_within_05pp"] else "No",
        })
    pd.DataFrame(rows).to_csv(TAB_OUT / "setting_effects_corrected.csv", index=False)
    print("[OK] setting_effects_corrected.csv")


def table_descriptive_summary():
    summary = json.loads((OUT / "synthesis_equal_setting_summary_corrected.json").read_text())
    df_dataset = pd.read_csv(OUT / "synthesis_dataset_level_corrected.csv")
    rows = [
        {
            "Summary": "Equal-setting-weighted descriptive mean",
            "Value": f"{summary['equal_setting_weighted_mean_diff_pp']:.4f} pp",
            "Note": "Not an inferential estimate; settings share design.",
        },
        {
            "Summary": "Across-setting SD",
            "Value": f"{summary['across_setting_sd_pp']:.4f} pp",
            "Note": "",
        },
        {
            "Summary": "Direction count",
            "Value": f"{summary['n_positive']} positive, {summary['n_negative']} negative, {summary['n_zero']} zero",
            "Note": "",
        },
    ]
    for _, r in df_dataset.iterrows():
        rows.append({
            "Summary": f"Dataset {r['dataset']} descriptive mean",
            "Value": f"{r['descriptive_mean_diff_pp']:.4f} pp",
            "Note": r["note"],
        })
    pd.DataFrame(rows).to_csv(TAB_OUT / "descriptive_summary_corrected.csv", index=False)
    print("[OK] descriptive_summary_corrected.csv")


def table_positive_control():
    per_seed = pd.read_csv(REPRO / "outputs" / "positive_control_per_seed_accuracy.csv")
    repro = json.loads((POS_CTRL / "MLJ_P2_STATISTICAL_REPORT.json").read_text())
    rows = []
    for _, r in per_seed.iterrows():
        rows.append({
            "Seed": int(r["seed"]),
            "FedAvg (%)": f"{r['fedavg']:.2f}",
            "Oracle aligned (%)": f"{r['oracle_nojs_aligned']:.2f}",
            "Oracle shuffled (%)": f"{r['oracle_nojs_shuffled']:.2f}",
            "Aligned - shuffled (pp)": f"{r['oracle_nojs_aligned'] - r['oracle_nojs_shuffled']:.2f}",
        })
    rows.append({
        "Seed": "Mean",
        "FedAvg (%)": f"{per_seed['fedavg'].mean():.2f}",
        "Oracle aligned (%)": f"{per_seed['oracle_nojs_aligned'].mean():.2f}",
        "Oracle shuffled (%)": f"{per_seed['oracle_nojs_shuffled'].mean():.2f}",
        "Aligned - shuffled (pp)": f"{repro['primary']['mean_difference']:.2f}",
    })
    pd.DataFrame(rows).to_csv(POS_TAB_OUT / "positive_control_per_seed.csv", index=False)

    contrast_rows = [
        {
            "Comparison": "Oracle aligned - Oracle shuffled",
            "Endpoint": "Round-100 accuracy",
            "Mean diff (pp)": f"{repro['primary']['mean_difference']:.2f}",
            "SD (pp)": f"{repro['primary']['sd_difference']:.2f}",
            "95% CI (pp)": f"[{repro['primary']['ci95_lower']:.2f}, {repro['primary']['ci95_upper']:.2f}]",
            "p (paired t)": f"{repro['primary']['two_sided_p']:.4f}",
            "Cohen's dz": f"{repro['primary']['cohen_dz']:.2f}",
        },
    ]
    for sec in repro["secondary_unadjusted"]:
        contrast_rows.append({
            "Comparison": sec["comparison"].replace("_", " ").replace("minus", "-"),
            "Endpoint": sec["endpoint"],
            "Mean diff (pp)": f"{sec['mean_difference']:.2f}",
            "SD (pp)": f"{sec['sd_difference']:.2f}",
            "95% CI (pp)": f"[{sec['ci95_lower']:.2f}, {sec['ci95_upper']:.2f}]",
            "p (paired t)": f"{sec['two_sided_p']:.4f}",
            "Cohen's dz": f"{sec['cohen_dz']:.2f}",
        })
    pd.DataFrame(contrast_rows).to_csv(POS_TAB_OUT / "positive_control_contrasts.csv", index=False)

    summary = pd.read_csv(REPRO / "outputs" / "positive_control_summary.csv")
    smap = dict(zip(summary["metric"], summary["value"]))
    def fmt(key):
        v = smap[key]
        try:
            return f"{float(v):.4f}"
        except ValueError:
            return str(v)
    mech_rows = [
        ("FedAvg corrupted-client weight mass", fmt("FedAvg corrupt weight mass")),
        ("Aligned corrupted-client weight mass", fmt("Aligned corrupt weight mass")),
        ("Shuffled corrupted-client weight mass", fmt("Shuffled corrupt weight mass")),
        ("Corrupted-weight reduction", fmt("Corrupt-weight reduction")),
        ("Mean aligned-shuffled L1 weight distance", fmt("Mean aligned-shuffled L1 weight distance")),
        ("Weight-multiset matched rounds", fmt("Weight-multiset matched rounds")),
        ("Minimum oracle effective-client count", fmt("Minimum oracle effective-client number")),
    ]
    pd.DataFrame(mech_rows, columns=["Diagnostic", "Value"]).to_csv(
        POS_TAB_OUT / "positive_control_mechanistic_diagnostics.csv", index=False
    )
    print("[OK] positive-control tables")


def report_legacy_tables():
    legacy = [
        ("Table 1", "four-arm architecture/aggregation summary", "manuscript only"),
        ("Table 2", "RA-InfoOpt matched summary", "manuscript only"),
        ("Table 3", "Study 3 score-attribution summary", "manuscript only"),
        ("Table 4", "Study 4 heterogeneity summary", "manuscript only"),
        ("Table 5", "Study 5 ResNet18 summary", "manuscript only"),
        ("Table 6", "FedDisco extension summary", "manuscript only"),
        ("Supplementary Table A.1-A.18", "ancillary summaries", "manuscript only"),
        ("Supplementary Table D.1-D.3", "retrospective tables", "manuscript only"),
    ]
    print("\nLegacy table status (values in manuscript; source CSVs not in reproduction package):")
    for name, desc, reason in legacy:
        print(f"  PARTIAL: {name} — {desc} ({reason})")


def main():
    table_setting_effects()
    table_descriptive_summary()
    table_positive_control()
    report_legacy_tables()
    print("\nAll reproducible tables generated.")


if __name__ == "__main__":
    main()
