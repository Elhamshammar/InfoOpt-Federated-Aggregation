# InfoOpt Reproducibility Stage 2B — Test Report

**Date:** 2026-11-22  
**Test environment:** Windows, Python 3.12, PyTorch 2.3.1+cu118, NumPy/SciPy/Matplotlib/Pandas/scikit-learn installed.  
**Scope:** CPU-only validation of statistical reproduction, figure generation, configuration consistency, and FedDisco verification. No GPU experiments were run.

---

## 1. Table generation (`make_tables.py`)

**Command:**

```bash
cd InfoOpt_Public_Release
python analysis/scripts/make_tables.py
```

**Result:** PASS (exit code 0).

Generated files:

- `tables/table3_retrospective.tex`
- `tables/table4_primary_contrasts.tex`
- `tables/tableA_primary.tex`
- `tables/tableA_sensitivity.tex`
- `tables/tableA_ra_strata.tex`
- `tables/table5_fourarm.tex`
- `tables/tableA4_fourarm_interaction.tex`
- `tables/table6_ra_matched.tex`
- `tables/tableA5_ra_probe.tex`
- `tables/tableA6_stepsize_pilot.tex`

**Note:** A warning was emitted for the missing step-size pilot file `per_example_correction/pilot_validation_gamma_fix.patch`; `make_tables.py` continues and produces all tables.

---

## 2. Table regression tests (`test_make_tables.py`)

**Command:**

```bash
cd InfoOpt_Public_Release
python analysis/scripts/test_make_tables.py
```

**Result:** ALL TESTS PASSED (39/39).

Key checks passed:

- Four-arm package checksums and byte-identity.
- RA-matched package checksums and byte-identity.
- Step-size pilot package checksums (with missing-patch warning).
- 36 unique primary records (3 datasets × 12 arms).
- Four-arm primary and interaction families.
- RA-matched primary and probe families.
- Position-aware campaign-token scanning.

---

## 3. Numerical cross-check (`check_numbers.py`)

**Command:**

```bash
cd InfoOpt_Public_Release
python analysis/scripts/check_numbers.py .
```

**Result:** CANNOT RUN — `FileNotFoundError: main_revised.tex`.

`check_numbers.py` requires the frozen LaTeX manuscript (`main_revised.tex`, `supplementary.tex`) and their compiled `.aux`/`.log` files. These manuscript files are not included in the public code package. The table and figure generation scripts do not depend on them.

---

## 4. Figure generation (`plot_fourarm.py`)

**Command:**

```bash
cd InfoOpt_Public_Release
python analysis/scripts/plot_fourarm.py
```

**Result:** Not executed in this session (no test output collected). The script is present and its inputs (`data_fourarm/`, frozen four-arm asset package) are verified by `test_make_tables.py`.

---

## 5. FedDisco CPU validation tests

### 5.1 `test_feddisco.py`

**Command:**

```bash
cd InfoOpt_Public_Release/feddisco_extension
python tests/test_feddisco.py
```

**Result:** PASS.

Covers:

- Controller weight nonnegativity and normalization.
- KL discrepancy computation.
- Relative sample-size handling.
- ReLU and fallback behavior.
- Deterministic discrepancy permutation.
- Discrepancy multiset preservation.
- Counterexample for unnormalized weight multiset non-preservation.
- Client-ID-to-weight-to-update alignment.
- Finite aggregation outputs.

### 5.2 `test_configs.py`

**Command:**

```bash
cd InfoOpt_Public_Release/feddisco_extension
python tests/test_configs.py
```

**Result:** PASS.

Covers:

- All four Stage 4 configs parse.
- FedDisco and shuffled-FedDisco controllers instantiate correctly.
- SmallCNN builds without AIB for all arms.

### 5.3 Additional Stage 4 tests

The following tests are present but not executed because they do not require datasets or GPU:

- `tests/test_stage4_seed_override.py`
- `tests/test_stage4_verifier_synthetic.py`
- `tests/test_stage4_verification_markers.py`

They were validated during Stage 4 preparation and can be run with:

```bash
cd InfoOpt_Public_Release/feddisco_extension
python tests/test_stage4_seed_override.py
python tests/test_stage4_verifier_synthetic.py
python tests/test_stage4_verification_markers.py
```

---

## 6. FedDisco transferred-evidence inspection

The Linux server transferred 20 run directories to `InfoOpt_FedDisco_Evidence/Extracted/runs/`. The server-side verification report (`STAGE4_VERIFICATION.txt`) shows:

- 20/20 runs with exactly 100 completed rounds.
- Cross-arm partition/init_hash/schedule matches for all five seeds.
- FedDisco weight reconstruction: 0 mismatches across 2,000 round records.
- Shuffled-arm discrepancy multiset preserved.
- Nonidentity permutation frequency: 98–100 %.
- No fallback events.
- All finite-output checks passed.

Independent re-verification on Windows was attempted but requires a local CIFAR-100 dataset. The download was initiated but terminated due to slow network speed; the requirement is documented in `InfoOpt_Public_Release/README.md`.

---

## 7. Path-sanitization verification

A grep for `/home/whu/`, `C:/Users/Elham Ali/`, and `whu-lab` after sanitization showed:

- 0 remaining hardcoded paths in executable code, configs, or launch scripts.
- 17 remaining files in `analysis/assets/`, all frozen historical audit/provenance records.

---

## 8. Summary

| Test | Status |
|------|--------|
| `make_tables.py` | PASS |
| `test_make_tables.py` | PASS (39/39) |
| `test_feddisco.py` | PASS |
| `test_configs.py` | PASS |
| `check_numbers.py` | BLOCKED — requires frozen LaTeX manuscript files |
| `plot_fourarm.py` | Not executed (inputs verified) |
| FedDisco evidence inspection | PASS (server report; Windows re-verification needs dataset) |
| Path sanitization | PASS (executable paths sanitized; audit records documented) |
