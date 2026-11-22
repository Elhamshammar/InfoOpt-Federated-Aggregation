#!/usr/bin/env python3
"""Render Figure 5 (main manuscript): the four-arm controlled experiment.

NEW figure for revision 5. Unlike scripts/replot_repeat_sensitivity.py (a
label-placement fix to an existing frozen figure), this script draws a figure
that did not exist before this revision, from the frozen four-arm package
assets/ITSL_four_arm_20261010/ (verified against its own SHA256SUMS.txt by
scripts/make_tables.py before this script is ever invoked from the build).

Panel (a): the six primary within-AIB InfoOpt-Full - FedAvg contrasts
(three datasets x {AIB off, AIB on}), 95% paired-t CIs, Holm p (m=6)
annotated.
Panel (b): the three aggregation x AIB interaction contrasts, 95% paired-t
CIs, Holm p (m=3) annotated; the CIFAR-100 interaction is the only one with
an interval excluding zero and is marked.

Output: Figures/figure_fourarm.pdf (+ .png). Values plotted are asserted
equal to data_fourarm/fourarm_primary_contrasts.json and
data_fourarm/fourarm_aib_interactions.json (working copies, themselves
checksum-verified against the frozen package by make_tables.py).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_fourarm")
OUT = os.path.join(ROOT, "Figures", "figure_fourarm")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 220})

DS = ["cifar100", "eurosat", "pathmnist"]
DS_LABEL = {"cifar100": "CIFAR-100", "eurosat": "EuroSAT", "pathmnist": "PathMNIST"}

contrasts = json.load(open(os.path.join(DATA, "fourarm_primary_contrasts.json")))
interactions = json.load(open(os.path.join(DATA, "fourarm_aib_interactions.json")))
assert len(contrasts) == 6
assert len(interactions) == 3

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})

# -------- Panel (a): six primary contrasts, grouped by dataset, AIB off/on --------
plotted_a = []
ypos = {}
y = 0
rows_a = []
for d in DS:
    for aib in ["aiboff", "aibon"]:
        c = [x for x in contrasts if x["dataset"] == d and x["aib"] == aib][0]
        rows_a.append((d, aib, c))
        ypos[(d, aib)] = y
        y += 1
    y += 0.6  # gap between datasets

colors = {"aiboff": "#6b6b6b", "aibon": "#2176ae"}
for d, aib, c in rows_a:
    yy = ypos[(d, aib)]
    lo, hi = c["ci95_low_pp"], c["ci95_high_pp"]
    mean = c["mean_difference_pp"]
    axL.plot([lo, hi], [yy, yy], "-", color=colors[aib], lw=1.6)
    axL.plot(mean, yy, "o", color=colors[aib], ms=5)
    axL.text(hi + 0.25, yy, f"Holm $p$={c['p_holm']:.3f}", va="center", fontsize=7.5)
    plotted_a.append((d, aib, mean, lo, hi, c["p_holm"]))
axL.axvline(0, color="#888888", ls="--", lw=1)
yt = [ypos[(d, aib)] for d in DS for aib in ["aiboff", "aibon"]]
ytl = [f"{DS_LABEL[d]}\nAIB {'off' if aib=='aiboff' else 'on'}" for d in DS for aib in ["aiboff", "aibon"]]
axL.set_yticks(yt, ytl, fontsize=8)
axL.invert_yaxis()
axL.set_xlabel("InfoOpt-Full $-$ FedAvg (pp)")
axL.set_title("(a) Primary contrasts within each AIB setting", fontsize=9.5)
axL.set_xlim(-12, 6)
axL.grid(axis="x", alpha=0.2)

# -------- Panel (b): three AIB x aggregation interactions --------
plotted_b = []
for i, d in enumerate(DS):
    c = [x for x in interactions if x["dataset"] == d][0]
    lo, hi = c["ci95_low_pp"], c["ci95_high_pp"]
    mean = c["mean_difference_pp"]
    sig = lo > 0 or hi < 0
    col = "#c1440e" if sig else "#6b6b6b"
    axR.plot([lo, hi], [i, i], "-", color=col, lw=1.8)
    axR.plot(mean, i, "o" if not sig else "D", color=col, ms=6)
    marker = " *" if sig else ""
    axR.text(hi + 0.3, i, f"Holm $p$={c['p_holm']:.3f}{marker}", va="center", fontsize=7.5)
    plotted_b.append((d, mean, lo, hi, c["p_holm"], sig))
axR.axvline(0, color="#888888", ls="--", lw=1)
axR.set_yticks(range(3), [DS_LABEL[d] for d in DS], fontsize=8)
axR.invert_yaxis()
axR.set_xlabel("Interaction (pp)")
axR.set_title("(b) Aggregation $\\times$ AIB interaction", fontsize=9.5)
axR.set_xlim(-6, 9)
axR.grid(axis="x", alpha=0.2)

fig.suptitle("Four-arm controlled experiment: FedAvg and InfoOpt-Full, AIB off/on ($n=5$ seed pairs)",
             fontsize=10.5)
# The full explanatory note (significance marker, Holm family sizes, partition/cohort-schedule
# identity, and experiment scope) is given in the LaTeX caption (main_revised.tex, fig:fourarm),
# not rendered into the image, so the image crops tightly to its two panels.
fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.80, wspace=0.55)
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.06)
fig.savefig(OUT + ".png", bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# Data check: everything plotted is exactly the verified value.
exp_a = {(c["dataset"], c["aib"]): c for c in contrasts}
for d, aib, mean, lo, hi, holm in plotted_a:
    c = exp_a[(d, aib)]
    assert mean == c["mean_difference_pp"] and lo == c["ci95_low_pp"] and hi == c["ci95_high_pp"] and holm == c["p_holm"]
exp_b = {c["dataset"]: c for c in interactions}
for d, mean, lo, hi, holm, sig in plotted_b:
    c = exp_b[d]
    assert mean == c["mean_difference_pp"] and lo == c["ci95_low_pp"] and hi == c["ci95_high_pp"] and holm == c["p_holm"]
    assert sig == (c["ci95_low_pp"] > 0 or c["ci95_high_pp"] < 0)
print("wrote", OUT + ".pdf; all plotted values identical to data_fourarm/ source")
