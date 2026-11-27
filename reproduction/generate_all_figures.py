#!/usr/bin/env python3
"""Generate all reproducible figures for the MLJ-R1.1 revision.

Run from the project root:
    python reproduction/generate_all_figures.py

Outputs:
- Three new manuscript figures (PDF in project root, PNG in reproduction/):
  FigureOracle_accuracy, FigureOracle_mass, FigureAttribution_forest.
- Corrected MLJ-A1.1 synthesis figures (PDF/PNG in reproduction/outputs/MLJ_A1_1/figures).
- A printed status report for legacy figures whose original plotting inputs are not
  available in this reproduction package.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPRO = Path(__file__).resolve().parent
ROOT = REPRO.parent
OUT = REPRO / "outputs" / "MLJ_A1_1"
FIG_OUT = OUT / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.compression": 9,
})


def save(fig: plt.Figure, name: str, root_pdf: bool = False):
    fig.tight_layout()
    if root_pdf:
        fig.savefig(ROOT / (name + ".pdf"))
    fig.savefig(REPRO / (name + ".png"), dpi=300)
    plt.close(fig)


def new_oracle_figures():
    """FigureOracle_accuracy and FigureOracle_mass."""
    df = pd.read_csv(REPRO / "positive_control" / "per_seed_results.csv")
    labels = {"fedavg": "FedAvg", "oracle_nojs_aligned": "Oracle aligned", "oracle_nojs_shuffled": "Oracle shuffled"}
    for field, name, ylabel in [
        ("final_acc_pct", "FigureOracle_accuracy", "Final test accuracy (%)"),
        ("mean_corrupt_mass", "FigureOracle_mass", "Corrupted-client weight mass"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        for method, label in labels.items():
            sub = df[df.method == method].sort_values("seed")
            ax.plot(sub.seed, sub[field], marker="o", label=label)
        ax.set(xticks=sorted(df.seed.unique()), xlabel="Seed", ylabel=ylabel)
        ax.legend(frameon=False)
        save(fig, name, root_pdf=True)
    print("[OK] FigureOracle_accuracy + FigureOracle_mass")


def new_attribution_forest():
    """FigureAttribution_forest: setting-specific aligned-minus-shuffled effects."""
    df = pd.read_csv(OUT / "synthesis_setting_level_corrected.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(df))
    ax.errorbar(
        df.mean_diff_pp,
        y,
        xerr=[df.mean_diff_pp - df.ci95_lower, df.ci95_upper - df.mean_diff_pp],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="gray", linestyle=":")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{dict(cifar100='CIFAR-100', eurosat='EuroSAT', pathmnist='PathMNIST').get(r.dataset, r.dataset)}, {r.model}, α={r.alpha:g}" for _, r in df.iterrows()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Aligned minus shuffled accuracy (pp)")
    save(fig, "FigureAttribution_forest", root_pdf=True)
    print("[OK] FigureAttribution_forest")


def corrected_synthesis_figures():
    """Forest and dataset summary using corrected synthesis outputs."""
    df = pd.read_csv(OUT / "synthesis_setting_level_corrected.csv")
    df = df.sort_values(["dataset", "stage", "alpha"])
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.errorbar(
        df.mean_diff_pp,
        y,
        xerr=[df.mean_diff_pp - df.ci95_lower, df.ci95_upper - df.mean_diff_pp],
        fmt="o",
        color="steelblue",
        ecolor="gray",
        capsize=3,
        markersize=5,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    labels = [f"{r.stage} {r.dataset} alpha={r.alpha} ({r.model})" for _, r in df.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Aligned - shuffled round-100 accuracy (pp)")
    ax.set_title("Setting-specific aligned-vs-shuffled InfoOpt effects\n(error bars are individual 95% paired CIs; not a pooled interval)")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "forest_12_settings_corrected.pdf", bbox_inches="tight")
    fig.savefig(FIG_OUT / "forest_12_settings_corrected.png", bbox_inches="tight")
    plt.close(fig)

    df_ds = pd.read_csv(OUT / "synthesis_dataset_level_corrected.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df_ds))
    ax.bar(x, df_ds.descriptive_mean_diff_pp, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df_ds.dataset)
    ax.set_ylabel("Mean of setting-level effects (pp)")
    ax.set_title("Dataset-level descriptive summary\n(no inferential interval; settings are dependent)")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "dataset_level_summary_corrected.pdf", bbox_inches="tight")
    fig.savefig(FIG_OUT / "dataset_level_summary_corrected.png", bbox_inches="tight")
    plt.close(fig)
    print("[OK] corrected synthesis figures")


def report_legacy_figures():
    legacy = [
        ("Figure_1_Framework", "conceptual framework diagram"),
        ("Figure2_fourarm", "Study 1 four-arm contrasts"),
        ("Figure3_final_accuracy", "Study 2B final accuracy"),
        ("Figure4_paired_differences", "Study 2B paired differences"),
        ("Figure5_stage1_contrasts", "Study 3 contrasts"),
        ("Figure6_stage3_contrasts", "Study 4 contrasts"),
        ("Figure7_stage4_contrasts", "Study 5 contrasts"),
        ("FigureA1_accuracy_trajectories", "Study 2B trajectories"),
        ("FigureA2_controller_diagnostics", "Study 2B controller diagnostics"),
        ("FigureA3_stage1_learning_curves", "Study 3 learning curves"),
        ("FigureD1_accuracy", "retrospective accuracy"),
        ("FigureD2_paired_contrasts", "retrospective paired contrasts"),
        ("FigureD3_repeat_sensitivity", "repeat-selection sensitivity"),
        ("FigureD4_ra_strata", "RA strata"),
    ]
    print("\nLegacy figure status (PDFs preserved; original plotting inputs not supplied):")
    for name, desc in legacy:
        pdf = ROOT / (name + ".pdf")
        status = "PRESENT" if pdf.exists() else "MISSING"
        print(f"  {status}: {name}.pdf — {desc}")


def main():
    new_oracle_figures()
    new_attribution_forest()
    corrected_synthesis_figures()
    report_legacy_figures()
    print("\nAll reproducible figures generated.")


if __name__ == "__main__":
    main()
