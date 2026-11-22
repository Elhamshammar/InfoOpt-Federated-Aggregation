# IT-SL verified figures — retrospective reanalysis, 2026-10-08

Four figures in vector PDF and 220 dpi PNG, with exact source outputs and a reproducible script. These are revision assets, not historical reconstructions or new experiments.

Run `python plot_figures.py` from any directory (Python, numpy and matplotlib required). Paths resolve relative to the script. No raw data, training, server access or tuning is required.

## Sources

- Primary package SHA-256: 85e0cc8af0b95231602d04c2dc708cb7e756027b6f6f70f01268a716f988cf58.
- Sensitivity v2 package SHA-256: ba5f20e430b6eafb551b4ed6c95240c4797b11141fa3ccc984fa0fa482469a96.
- source_data/primary_summary.csv and paired_contrasts.json: verified primary outputs.
- source_data/sensitivity_latest_repeat.json and ra_strata_summaries.json: verified sensitivity v2 outputs.

## Figure use and captions

1. figure_accuracy: terminal round-100 test accuracy, mean ± sample SD, n=5 for available rows. RA combined results are withheld and CIFAR FedOpt is excluded under the strict configured-early-stop rule. Orange squares mark FedDyn/SCAFFOLD audit observations with unresolved implementation provenance: do not use for method ranking. InfoOpt/RA use AIB while the other arms do not. Means are descriptive architecture-plus-method outcomes.
2. figure_paired_contrasts: InfoOpt minus FedAvg in percentage points, 95% paired-t intervals, n=5. Holm family size remains six, including three non-estimable RA comparisons. Intervals crossing zero express uncertainty, not equivalence. Partition membership is assumed identical for matched seeds, supported by schedule/count checks but not proven.
3. figure_repeat_sensitivity: latest minus earliest mean accuracy for FedAvg and InfoOpt-Full only. Maximum absolute movement 0.164 pp. Same-source cases reuse the sole selected run. No claim of sensitivity stability for other methods.
4. figure_ra_strata: one selected run per seed per configuration stratum. Points are seed-level endpoints; bars show mean ± sample SD, absent for n=1. Panels use separate vertical scales. No combined estimate or cross-stratum inferential comparison.

## LaTeX integration

Use PDFs via includegraphics at text width; copy the caveats into manuscript captions. These new figures do not validate existing convergence, alpha-sensitivity or diagnostic figures, which need their own data/provenance review. Retain all original figures unchanged in the original manuscript project.

SHA256SUMS.txt covers payload files and excludes itself.
