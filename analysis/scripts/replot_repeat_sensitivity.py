#!/usr/bin/env python3
"""Re-render figure_repeat_sensitivity with a label-placement fix only.

Identical to the figure_repeat_sensitivity block of the frozen
assets/ITSL_verified_figures/plot_figures.py (same source data, same marks,
axes, limits, colors, title and footnote), except that each value label is
centered above its point (ha='center') instead of right-aligned at the point,
so the PathMNIST "-0.164 pp" label no longer overlaps the y-axis spine.
No data are changed. Output: Figures/figure_repeat_sensitivity.pdf (+ .png).
The frozen original in assets/ is left untouched.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "ITSL_verified_figures", "source_data", "sensitivity_latest_repeat.json")
OUT = os.path.join(ROOT, "Figures", "figure_repeat_sensitivity")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 220})
ds = ["cifar100", "eurosat", "pathmnist"]
titles = ["CIFAR-100", "EuroSAT", "PathMNIST"]
sens = json.load(open(SRC))["results"]

fig, axs = plt.subplots(1, 3, figsize=(10, 3.8), sharex=True, sharey=True)
plotted = []
for ax, d, t in zip(axs, ds, titles):
    rr = [x for x in sens if x["dataset"] == d and x["row"] in ["fedavg", "InfoOpt_Full"]]
    for i, r in enumerate(rr):
        v = r["delta"]
        ax.plot(v, i, "o", color=["#444444", "#2176ae"][i])
        ax.text(v, i + .15, f"{v:+.3f} pp", ha="center", fontsize=9)  # only change: centered label
        plotted.append((d, r["row"], v))
    ax.axvline(0, color="#888888", ls="--", lw=1)
    ax.set(title=t, xlabel="Latest minus earliest (pp)", xlim=(-.22, .07), ylim=(-.4, 1.6))
    ax.grid(axis="x", alpha=.2)
axs[0].set_yticks([0, 1], ["FedAvg", "InfoOpt-Full (AIB)"])
fig.suptitle("Within-stratum repeat sensitivity: change in mean accuracy")
fig.subplots_adjust(left=.19, bottom=.27, top=.82, wspace=.15)
fig.text(.01, .015, "Same-source cells retain their only run. Maximum movement among these six summaries: 0.164 pp.\n"
         "This sensitivity does not assess FedDyn/SCAFFOLD or every Table 3 row; AIB mismatch persists.",
         ha="left", va="bottom", fontsize=8)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", bbox_inches="tight")
plt.close(fig)

# Data check: plotted values are exactly the verified deltas.
expected = {(r["dataset"], r["row"]): r["delta"] for r in sens if r["row"] in ["fedavg", "InfoOpt_Full"]}
assert len(plotted) == 6 and all(expected[(d, row)] == v for d, row, v in plotted)
print("wrote", OUT + ".pdf; plotted values identical to source:", [f"{v:+.3f}" for _, _, v in plotted])
