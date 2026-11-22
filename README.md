# InfoOpt — Reproducibility Package

This repository contains the code, configurations, numerical results, and analysis scripts needed to reproduce the findings reported in the manuscript:

> "Do Client-Specific Scores Improve Information-Guided Federated Aggregation? A Controlled Evaluation and Implementation Audit."

## What is included

- **Canonical training code** in `source_snapshots/` for each study (Studies 1–5, corrected Study 2B, and the FedDisco extension).
- **Frozen configurations** in `configs/` (effective YAMLs and resolution manifests).
- **Numerical results** in `analysis/data*/` (per-seed round-100 accuracies, contrasts, summaries).
- **Analysis and figure scripts** in `analysis/scripts/`.
- **FedDisco extension evidence** in `feddisco_extension/` (source, configs, 20 run directories, verification and analysis reports).
- **Verification reports** for the four-arm and RA-InfoOpt experiments under `analysis/data_fourarm/` and `analysis/data_ra/`.

## What is excluded

- Raw datasets (CIFAR-100, EuroSAT, PathMNIST) — see Dataset setup below.
- Large model checkpoints.
- Medical imaging arrays unrelated to the paper.
- Private credentials and internal server paths.

## Important notes

- **Historical audit-only records.** Some files under `experiments/` and
  `analysis/assets/` are retained as frozen audit evidence. They may
  contain original machine-specific paths from the development
  environment and should not be executed without reviewing those paths.
- **LaTeX manuscript source.** The final Astra-corrected LaTeX source
  (`main_revised.tex`, `supplementary.tex`, `references.bib`,
  `elsarticle-num.bst`) was not available in the workspace during
  packaging. A placeholder `manuscript/` directory is provided. Add the
  source files there to enable the full prose and cross-reference checks
  in `analysis/scripts/check_numbers.py`.
- **Missing historical patch.** The file
  `pilot_validation_gamma_fix.patch` listed in the step-size pilot
  package's `SHA256SUMS.txt` is absent. This missing patch is documented
  in the Stage-2C validation report and does not affect the retained
  numerical results of the principal controlled studies.

## Quick start

```bash
pip install -r requirements.txt

# Regenerate tables and figures from retained results
python analysis/scripts/make_tables.py
python analysis/scripts/test_make_tables.py
python analysis/scripts/plot_fourarm.py

# Run numerical cross-checks against the manuscript
python analysis/scripts/check_numbers.py .
```

## Dataset setup

The training scripts expect datasets under a configurable root. Set one of:

- `DATASETS_ROOT` environment variable (e.g., `export DATASETS_ROOT=/path/to/datasets`);
- `EXP_ROOT/datasets/` relative to the experiment root;
- or edit the `dataset.root` field in the YAML configs.

CIFAR-100 will be downloaded automatically by torchvision if not present. EuroSAT and PathMNIST must be obtained separately and placed in the configured root.

## Reproducibility notes

- All reported manuscript tables and figures can be regenerated from the included CSV/JSON summaries without retraining.
- Retraining the original 480 + 15 corrected Study 2B runs requires the datasets, a GPU environment, and the launch scripts in `source_snapshots/`.
- The 20 FedDisco extension runs are included under `feddisco_extension/runs/` and were verified independently on the Linux server; the verification report is `feddisco_extension/STAGE4_VERIFICATION.txt`.

## License

To be determined before public release.
