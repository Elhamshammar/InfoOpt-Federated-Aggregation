# InfoOpt MLJ Final GitHub Public-Release Audit

**Repository:** `D:\InfoOpt Latex\GitHub_MLJ_Final\InfoOpt-Federated-Aggregation`

**Date:** 2026-11-27

**Scope:** Final safety and reproducibility audit of the sanitized public-release overlay for the manuscript *Evaluating Client-Specific Score Attribution in Federated Aggregation: A Controlled Score-Reassignment Methodology*.

## 1. Sanitization summary

A systematic scan was performed for credentials, API keys, authentication tokens, private SSH information, personal information, internal server addresses, and machine-specific absolute paths. No credentials or secrets were found.

Machine-specific path prefixes in historical provenance records and launch scripts were redacted in the cloned repository only. Original research artifacts outside this clone remain unchanged.

### 1.1 Path redactions applied

| Pattern | Redaction | File categories |
|---|---|---|
| `/home/whu/` | `<REDACTED_HOME>/` | Historical provenance records, Stage2C reports, audit scripts, JSON manifests |
| `/mnt/diskN/` (N = 0-9) | `<REDACTED_SERVER_PATH>/` | Historical provenance records and manifests |
| `D:\InfoOpt ...` and `D:\FL_StepSize_Diagnostic\...` | `<REDACTED_LOCAL_PATH>` | `SOURCE_MANIFEST_SHA256.csv`, pre-GitHub report, Stage2 release reports |
| `C:/Users/Elham Ali/` | `C:/Users/<REDACTED_USER>/` | Historical provenance records |
| `whu-lab` | `<REDACTED_USER>` | Historical provenance identifiers |

### 1.2 Executable script updates

Selftest scripts in the following directories were updated so that they no longer rely on hardcoded Git-for-Windows paths or a fixed Windows dataset root:

- `experiments/stage1/selftest_stage1.py`
- `experiments/stage2b_corrected/selftest_fourarm.py`
- `experiments/stage3/selftest_stage3.py`
- `experiments/stage4/selftest_stage3.py`
- `experiments/stage4/selftest_stage4.py`
- Corresponding files under `source_snapshots/`

Changes:

- Bash discovery now checks the `BASH_EXE` environment variable first, then `shutil.which("bash")`.
- Default dataset roots now resolve relative to the script location (`Path(__file__).resolve().parents[N] / "datasets"`) instead of a hardcoded `D:\FL_StepSize_Diagnostic\data` path.

### 1.3 Smoke configuration updates

- `experiments/stage2b_corrected/smoke/smoke_ra_probeoff.yaml`
- Corresponding source-snapshot copy

The dataset root was changed from an absolute Windows path to `./datasets/cifar100`.

### 1.4 Other redacted records

- `experiments/stage2b_corrected/smoke_out/.../config_used.json`
- `analysis/assets/ITSL_stepsize_pilot_20261011/per_example_correction/regeneration_metadata.json`

Absolute source paths in these records were replaced with `<REDACTED_LOCAL_PATH>` or a relative placeholder while preserving run identifiers, parameters, and numerical values.

### 1.5 Gitignore update

`.gitignore` was updated to exclude `_dataset_cache/`, the directory used by oracle tests to download CIFAR-100 temporarily.

## 2. Verification results

All CPU-only verification commands were run from the repository root.

### 2.1 Statistical verification

```bash
python reproduction/scripts_reports/mlj_r4_statistics.py \
  --seed-csv reproduction/outputs/MLJ_A1_1/seed_level_effects.csv \
  --oracle-csv reproduction/outputs/positive_control_per_seed_accuracy.csv \
  --out reproduction/outputs/MLJ_R4
```

**Result:** 56/56 checks passed.

The script reproduces:

- Twelve setting-level score-attribution contrasts with paired 95% confidence intervals and retrospective TOST p-values at ±0.5 pp.
- Exact minimum detectable effect calculations that yield 80% nominal power.
- FedDisco aligned-versus-shuffled contrast: +0.046 pp, 95% CI [−0.318, +0.410], two-sided p = 0.744, TOST equivalence at ±0.5 pp.
- Oracle positive-control aligned-versus-shuffled contrast: +5.300 pp, 95% CI [1.825, 8.775], p = 0.013319, Cohen's dz = 1.894.
- Oracle comparisons with FedAvg and last-10-round analysis.
- Equal-setting descriptive mean of −0.824 pp and nine of twelve negative setting means.

### 2.2 Table generation

```bash
python reproduction/generate_all_tables.py
```

**Result:** All reproducible tables were generated. Legacy manuscript-only tables are marked as partial because their exact original plotting inputs could not be recovered.

### 2.3 Figure generation

```bash
python reproduction/generate_all_figures.py
```

**Result:** All reproducible figures were generated, including:

- `FigureAttribution_forest.pdf`
- `FigureOracle_accuracy.pdf`
- `FigureOracle_mass.pdf`

Legacy historical PDFs are retained but noted as missing original plotting inputs.

### 2.4 Oracle component tests

```bash
python -m pytest oracle_positive_control/tests/ -v
```

**Result:** 34/34 tests passed after downloading CIFAR-100 to a temporary `_dataset_cache/` directory. The cache was subsequently removed and is excluded from the repository by `.gitignore`. If the test suite is run without network access or an existing CIFAR-100 download, one test requiring the dataset will fail; the failure is a data-availability issue, not a code issue.

## 3. Integrity manifest

A new manifest of the sanitized working tree was generated:

- File: `SHA256SUMS_PUBLIC_RELEASE_SANITIZED.txt`
- Entries: 1449
- Exclusions: `.git/`, `_dataset_cache/`, `.pytest_cache/`, existing historical manifests, and run checkpoints.

The original execution manifests are preserved as historical records:

- `SHA256SUMS_Public_Release.txt`
- `SHA256SUMS_Public_Release.txt.sha256`
- `SHA256SUMS_MLJ_Update.txt`

## 4. License status

- File: `LICENSE` (placeholder)
- License review: `INFOOPT_REPRO_STAGE2C_LICENSE_REVIEW.md`

Findings:

- The federated-learning training code, analysis scripts, figure-generation scripts, and FedDisco extension implementation are original work by the manuscript authors.
- No third-party copyleft code, vendored libraries, or modified upstream files were identified.
- Runtime dependencies are standard scientific-Python packages listed in `requirements.txt`.
- `elsarticle-num.bst` is an official Elsevier publisher-supplied bibliography style file.

The `LICENSE` file currently states that the license is "to be determined before public release." The final license must be selected and inserted by the authors before any public release or GitHub upload. The license review recommends an OSI-approved permissive license (Apache-2.0 preferred), but author or institutional confirmation is required.

## 5. README and reproduction coverage

The updated `README.md`:

- States the manuscript title and the 530-run experimental accounting (480 original, 15 reliability, 20 FedDisco, 15 oracle).
- Describes the repository organization, including the recovered `oracle_positive_control/` source and the `reproduction/` verification scripts.
- Provides commands to reproduce the updated statistical analyses, tables, and figures.
- Notes that the oracle positive-control implementation and 15 run records are in `oracle_positive_control/` and refers to its README for the portable launcher.
- Documents the main limitations: raw datasets and large checkpoints are not included; full GPU retraining has not been verified in this release workspace; historical provenance gaps remain as documented.

The oracle README covers seeds 100–104.

## 6. Remaining limitations

1. **License placeholder:** The final open-source license must be inserted by the authors before public release or GitHub upload.
2. **Historical provenance records:** Sanitized audit records under `analysis/assets/` are retained in the public overlay. The authors may prefer to move these internal audit files to a separate research archive rather than include them in the public repository.
3. **Dataset and checkpoint availability:** Raw CIFAR-100, PathMNIST, EuroSAT, and other benchmark datasets are not bundled. Large model checkpoints are not included.
4. **GPU retraining verification:** Full GPU retraining has not been re-executed in this sanitized workspace. The retained run directories and source snapshots document the executed experiments.
5. **Legacy figures and tables:** A small number of historical manuscript figures and tables depend on inputs that could not be recovered exactly. Their status is documented in the reproduction README and provenance files.
6. **Server-side logs and checkpoints:** Some original execution artifacts exist only on the Linux research server and are not packaged here.

## 7. Decision

**READY PENDING LICENSE APPROVAL**

The cloned repository overlay has been sanitized, the CPU-only verification suite passes, and the reproduction package is internally consistent. No credentials, secrets, or unredacted machine-specific paths remain in the working tree. The only blocker for public release or GitHub upload is the replacement of the placeholder `LICENSE` file with a final author-approved license.

## 8. Files changed in this audit

Modified tracked files (path redactions and script/config updates):

- `.gitignore`
- `INFOOPT_FINAL_PRE_GITHUB_REPORT.md`
- `INFOOPT_REPRO_STAGE2C_FINAL_VALIDATION.md`
- `INFOOPT_REPRO_STAGE2C_LICENSE_REVIEW.md`
- `INFOOPT_REPRO_STAGE2C_RELEASE_MANIFEST.md`
- `INFOOPT_REPRO_STAGE2_DECISION.md`
- `INFOOPT_REPRO_STAGE2_REPOSITORY_MANIFEST.md`
- `INFOOPT_REPRO_STAGE2_TEST_REPORT.md`
- `README.md`
- `RELEASE_REPORT.txt`
- `SOURCE_MANIFEST_SHA256.csv`
- Historical provenance files under `analysis/assets/`
- `analysis/data_fourarm/fourarm_per_seed_round100.csv`
- Selftest scripts under `experiments/` and `source_snapshots/`
- Smoke configuration files under `experiments/stage2b_corrected/smoke/` and the source-snapshot copy
- `experiments/stage2b_corrected/smoke_out/.../config_used.json`

Untracked new files:

- `FigureAttribution_forest.pdf`
- `FigureOracle_accuracy.pdf`
- `FigureOracle_mass.pdf`
- `MLJ_RELEASE_VALIDATION.md`
- `SHA256SUMS_MLJ_Update.txt`
- `SHA256SUMS_PUBLIC_RELEASE_SANITIZED.txt`
- `oracle_positive_control/`
- `reproduction/`
- `PUBLIC_RELEASE_FINAL_AUDIT.md` (this file)

## 9. Next steps

1. Replace `LICENSE` with the final author-approved license text.
2. Decide whether to keep historical `analysis/assets/` audit records in the public repository or move them to a separate research archive.
3. Upload the sanitized repository to GitHub only after the license placeholder is resolved.
