# Buffer-policy and scaling analysis (saved artifacts only, REVISED)

- run: `stepsize_diag_pilot_20261004_175620`
- analysis output dir: `analysis_buffer_policy_v2` (originals preserved; supersedes analysis_buffer_policy/ with the dev/held-out separation and without the withdrawn interaction claim)
- positive loss improvement = loss_reference - loss_candidate; positive accuracy improvement = acc_candidate - acc_reference
- repetitions averaged WITHIN each cohort first; trajectory sources are reported separately and never pooled with repetition cohorts
- development seed 201 (used to select the fixed scales) and held-out seeds 202/203 are reported SEPARATELY in every summary

## Provenance

- command: `python -X utf8 analysis_buffer_policy_v2/analyze_buffer_policy.py --run-dir results/stepsize_diag_pilot_20261004_175620 --out-dir analysis_buffer_policy_v2`
- sha256 `results/aggregate_evaluation.csv`: `aede8c6ce740084e30503c3df25b48d86da63254fd4a6010952b34f010ada7a1`
- sha256 `results/secondary_buffer_evaluation.csv`: `df18b241d5fff708a8247908ec97c79ac4772a79aa9801b7d77650806443f87a`
- sha256 `results/rule_evaluation.csv`: `ac65d9fc2867bfb6e37b22b2c59d6554120a79d979b8208213bcc3eef91b6c28`
- sha256 `results/per_example_losses.csv`: `0829d977a996f077750dc694a3f2f10cf390c565e2bb34a46a208233dec5c3c5`
- sha256 `results/gamma_dev_global.json`: `70eb546f1cc6fd028600aac804bd5c918bbdbc5be647205cd8056ca64feb1e8a`
- sha256 `results/gamma_dev_phase_100.json`: `d56c00264c3a73f857b2ed998a7bb48e05f63b8348fe642f5214243d271f6469`
- sha256 `results/gamma_dev_phase_5.json`: `f2d1c0c24015a68f15c2d906ffb33fff476a845b11d2e06e0a3ad717455c5792`
- sha256 `results/gamma_dev_phase_50.json`: `a42c0331e3ff06c6e2224171f3757e57fbd9ce48ab14e050127a8e63dd4d554b`

## Checks

- schema: required columns present in all three CSVs: True
- grid rows: 810; secondary rows: 90; rule rows: 570
- invalid grid candidates: 10 (all at gamma=2.0 — negative running_var; none intersect the compared rows at gamma=1 / rule rows; rule rows invalid: 0)
- missing pairs per comparison: none
- invalid pairs excluded: none
- n_eval mismatches in pairs: 0; grid/secondary n_eval values: [2500]
- rule rows do not persist n_eval; per-example row counts per (seed, round, source, policy, evaluation) block vs n_eval=[2500]: all match
- dev-seed fixed-scale measurements recovered from existing grid rows at the frozen gammas (gamma_global=0.5, gamma_phase={100: 0.5, 5: 0.75, 50: 0.5}): 60 rows labeled recovered_from_grid_row (no new evaluation)

## Measured findings

### A_freeze_buffers_at_gamma1

| seed | round | cohort | n_reps | mean loss ref | mean loss cand | loss improvement | acc improvement |
|------|-------|--------|--------|---------------|----------------|------------------|-----------------|
| 201 | 5 | alt_0 | 3 | 3.7151 | 3.7219 | -0.0068 | +0.0013 |
| 201 | 5 | alt_1 | 3 | 3.7275 | 3.7564 | -0.0289 | -0.0096 |
| 201 | 5 | original | 3 | 3.8060 | 3.8601 | -0.0541 | -0.0093 |
| 201 | 50 | alt_0 | 3 | 2.4391 | 2.5132 | -0.0742 | -0.0105 |
| 201 | 50 | alt_1 | 3 | 2.5228 | 2.5769 | -0.0541 | -0.0147 |
| 201 | 50 | original | 3 | 2.4571 | 2.5096 | -0.0525 | -0.0100 |
| 201 | 100 | alt_0 | 3 | 2.2158 | 2.3469 | -0.1310 | -0.0213 |
| 201 | 100 | alt_1 | 3 | 2.2142 | 2.3575 | -0.1433 | -0.0216 |
| 201 | 100 | original | 3 | 2.1536 | 2.2428 | -0.0892 | -0.0217 |
| 202 | 5 | alt_0 | 3 | 3.7031 | 3.7193 | -0.0162 | -0.0075 |
| 202 | 5 | alt_1 | 3 | 3.6849 | 3.6988 | -0.0139 | -0.0025 |
| 202 | 5 | original | 3 | 3.8786 | 3.9183 | -0.0397 | -0.0099 |
| 202 | 50 | alt_0 | 3 | 2.4766 | 2.5133 | -0.0367 | -0.0059 |
| 202 | 50 | alt_1 | 3 | 2.4281 | 2.5161 | -0.0880 | -0.0108 |
| 202 | 50 | original | 3 | 2.4998 | 2.5514 | -0.0516 | -0.0083 |
| 202 | 100 | alt_0 | 3 | 2.2585 | 2.3851 | -0.1266 | -0.0147 |
| 202 | 100 | alt_1 | 3 | 2.1915 | 2.3593 | -0.1678 | -0.0305 |
| 202 | 100 | original | 3 | 2.2074 | 2.3390 | -0.1315 | -0.0284 |
| 203 | 5 | alt_0 | 3 | 3.7438 | 3.7423 | +0.0015 | +0.0060 |
| 203 | 5 | alt_1 | 3 | 3.7228 | 3.7069 | +0.0159 | -0.0005 |
| 203 | 5 | original | 3 | 3.7991 | 3.7926 | +0.0065 | +0.0061 |
| 203 | 50 | alt_0 | 3 | 2.4426 | 2.4820 | -0.0395 | -0.0105 |
| 203 | 50 | alt_1 | 3 | 2.5360 | 2.5959 | -0.0599 | -0.0152 |
| 203 | 50 | original | 3 | 2.4546 | 2.5001 | -0.0455 | -0.0048 |
| 203 | 100 | alt_0 | 3 | 2.2251 | 2.3615 | -0.1364 | -0.0243 |
| 203 | 100 | alt_1 | 3 | 2.2183 | 2.3346 | -0.1163 | -0.0276 |
| 203 | 100 | original | 3 | 2.1155 | 2.2023 | -0.0867 | -0.0152 |
- cohort-mean over dev seed 201 (selection source; NOT a held-out evaluation): loss improvement -0.0705, acc improvement -0.0131 (n_cohort_cells=9)
- cohort-mean over held-out seeds (202, 203): loss improvement -0.0629, acc improvement -0.0114 (n_cohort_cells=18)
- trajectory (single observation per seed/round, reported separately):
    - seed 201 round 5: loss improvement -0.0700, acc improvement -0.0068
    - seed 201 round 50: loss improvement -0.0078, acc improvement -0.0016
    - seed 201 round 100: loss improvement -0.1648, acc improvement -0.0328
    - seed 202 round 5: loss improvement -0.0212, acc improvement +0.0032
    - seed 202 round 50: loss improvement -0.0695, acc improvement -0.0132
    - seed 202 round 100: loss improvement -0.1865, acc improvement -0.0324
    - seed 203 round 5: loss improvement -0.0146, acc improvement +0.0032
    - seed 203 round 50: loss improvement -0.1290, acc improvement -0.0248
    - seed 203 round 100: loss improvement -0.0188, acc improvement -0.0136

### B_valsel_vs_full_gamma1

| seed | round | cohort | n_reps | mean loss ref | mean loss cand | loss improvement | acc improvement |
|------|-------|--------|--------|---------------|----------------|------------------|-----------------|
| 201 | 5 | alt_0 | 3 | 3.7151 | 3.7018 | +0.0133 | +0.0052 |
| 201 | 5 | alt_1 | 3 | 3.7275 | 3.7280 | -0.0005 | -0.0049 |
| 201 | 5 | original | 3 | 3.8060 | 3.8049 | +0.0010 | +0.0017 |
| 201 | 50 | alt_0 | 3 | 2.4391 | 2.3743 | +0.0648 | +0.0247 |
| 201 | 50 | alt_1 | 3 | 2.5228 | 2.4103 | +0.1125 | +0.0256 |
| 201 | 50 | original | 3 | 2.4571 | 2.3995 | +0.0576 | +0.0187 |
| 201 | 100 | alt_0 | 3 | 2.2158 | 2.0386 | +0.1772 | +0.0427 |
| 201 | 100 | alt_1 | 3 | 2.2142 | 2.0900 | +0.1242 | +0.0224 |
| 201 | 100 | original | 3 | 2.1536 | 2.0899 | +0.0638 | +0.0140 |
| 202 | 5 | alt_0 | 3 | 3.7031 | 3.6753 | +0.0278 | +0.0063 |
| 202 | 5 | alt_1 | 3 | 3.6849 | 3.6769 | +0.0080 | +0.0035 |
| 202 | 5 | original | 3 | 3.8786 | 3.7944 | +0.0841 | +0.0009 |
| 202 | 50 | alt_0 | 3 | 2.4766 | 2.4247 | +0.0519 | +0.0135 |
| 202 | 50 | alt_1 | 3 | 2.4281 | 2.3853 | +0.0428 | +0.0168 |
| 202 | 50 | original | 3 | 2.4998 | 2.3961 | +0.1038 | +0.0276 |
| 202 | 100 | alt_0 | 3 | 2.2585 | 2.1336 | +0.1248 | +0.0203 |
| 202 | 100 | alt_1 | 3 | 2.1915 | 2.0655 | +0.1261 | +0.0325 |
| 202 | 100 | original | 3 | 2.2074 | 2.0894 | +0.1181 | +0.0233 |
| 203 | 5 | alt_0 | 3 | 3.7438 | 3.7438 | +0.0000 | +0.0000 |
| 203 | 5 | alt_1 | 3 | 3.7228 | 3.7238 | -0.0011 | +0.0000 |
| 203 | 5 | original | 3 | 3.7991 | 3.7991 | +0.0000 | +0.0000 |
| 203 | 50 | alt_0 | 3 | 2.4426 | 2.3925 | +0.0501 | +0.0139 |
| 203 | 50 | alt_1 | 3 | 2.5360 | 2.4123 | +0.1237 | +0.0241 |
| 203 | 50 | original | 3 | 2.4546 | 2.3770 | +0.0776 | +0.0237 |
| 203 | 100 | alt_0 | 3 | 2.2251 | 2.1504 | +0.0747 | +0.0113 |
| 203 | 100 | alt_1 | 3 | 2.2183 | 2.1404 | +0.0779 | +0.0156 |
| 203 | 100 | original | 3 | 2.1155 | 2.0422 | +0.0733 | +0.0279 |
- cohort-mean over dev seed 201 (selection source; NOT a held-out evaluation): loss improvement +0.0682, acc improvement +0.0167 (n_cohort_cells=9)
- cohort-mean over held-out seeds (202, 203): loss improvement +0.0646, acc improvement +0.0145 (n_cohort_cells=18)
- trajectory (single observation per seed/round, reported separately):
    - seed 201 round 5: loss improvement +0.0000, acc improvement +0.0000
    - seed 201 round 50: loss improvement +0.0586, acc improvement +0.0128
    - seed 201 round 100: loss improvement +0.0542, acc improvement +0.0104
    - seed 202 round 5: loss improvement +0.0560, acc improvement +0.0032
    - seed 202 round 50: loss improvement +0.1325, acc improvement +0.0324
    - seed 202 round 100: loss improvement +0.1222, acc improvement +0.0256
    - seed 203 round 5: loss improvement +0.0000, acc improvement +0.0000
    - seed 203 round 50: loss improvement +0.0830, acc improvement +0.0204
    - seed 203 round 100: loss improvement +0.0912, acc improvement +0.0248

### B_valsel_vs_params_only_gamma1

| seed | round | cohort | n_reps | mean loss ref | mean loss cand | loss improvement | acc improvement |
|------|-------|--------|--------|---------------|----------------|------------------|-----------------|
| 201 | 5 | alt_0 | 3 | 3.7219 | 3.7018 | +0.0201 | +0.0039 |
| 201 | 5 | alt_1 | 3 | 3.7564 | 3.7280 | +0.0284 | +0.0047 |
| 201 | 5 | original | 3 | 3.8601 | 3.8049 | +0.0552 | +0.0111 |
| 201 | 50 | alt_0 | 3 | 2.5132 | 2.3743 | +0.1389 | +0.0352 |
| 201 | 50 | alt_1 | 3 | 2.5769 | 2.4103 | +0.1666 | +0.0403 |
| 201 | 50 | original | 3 | 2.5096 | 2.3995 | +0.1101 | +0.0287 |
| 201 | 100 | alt_0 | 3 | 2.3469 | 2.0386 | +0.3083 | +0.0640 |
| 201 | 100 | alt_1 | 3 | 2.3575 | 2.0900 | +0.2676 | +0.0440 |
| 201 | 100 | original | 3 | 2.2428 | 2.0899 | +0.1530 | +0.0357 |
| 202 | 5 | alt_0 | 3 | 3.7193 | 3.6753 | +0.0440 | +0.0137 |
| 202 | 5 | alt_1 | 3 | 3.6988 | 3.6769 | +0.0220 | +0.0060 |
| 202 | 5 | original | 3 | 3.9183 | 3.7944 | +0.1239 | +0.0108 |
| 202 | 50 | alt_0 | 3 | 2.5133 | 2.4247 | +0.0886 | +0.0193 |
| 202 | 50 | alt_1 | 3 | 2.5161 | 2.3853 | +0.1307 | +0.0276 |
| 202 | 50 | original | 3 | 2.5514 | 2.3961 | +0.1553 | +0.0359 |
| 202 | 100 | alt_0 | 3 | 2.3851 | 2.1336 | +0.2514 | +0.0349 |
| 202 | 100 | alt_1 | 3 | 2.3593 | 2.0655 | +0.2938 | +0.0631 |
| 202 | 100 | original | 3 | 2.3390 | 2.0894 | +0.2496 | +0.0517 |
| 203 | 5 | alt_0 | 3 | 3.7423 | 3.7438 | -0.0015 | -0.0060 |
| 203 | 5 | alt_1 | 3 | 3.7069 | 3.7238 | -0.0169 | +0.0005 |
| 203 | 5 | original | 3 | 3.7926 | 3.7991 | -0.0065 | -0.0061 |
| 203 | 50 | alt_0 | 3 | 2.4820 | 2.3925 | +0.0895 | +0.0244 |
| 203 | 50 | alt_1 | 3 | 2.5959 | 2.4123 | +0.1836 | +0.0393 |
| 203 | 50 | original | 3 | 2.5001 | 2.3770 | +0.1231 | +0.0285 |
| 203 | 100 | alt_0 | 3 | 2.3615 | 2.1504 | +0.2111 | +0.0356 |
| 203 | 100 | alt_1 | 3 | 2.3346 | 2.1404 | +0.1942 | +0.0432 |
| 203 | 100 | original | 3 | 2.2023 | 2.0422 | +0.1600 | +0.0431 |
- cohort-mean over dev seed 201 (selection source; NOT a held-out evaluation): loss improvement +0.1387, acc improvement +0.0297 (n_cohort_cells=9)
- cohort-mean over held-out seeds (202, 203): loss improvement +0.1276, acc improvement +0.0259 (n_cohort_cells=18)
- trajectory (single observation per seed/round, reported separately):
    - seed 201 round 5: loss improvement +0.0700, acc improvement +0.0068
    - seed 201 round 50: loss improvement +0.0664, acc improvement +0.0144
    - seed 201 round 100: loss improvement +0.2190, acc improvement +0.0432
    - seed 202 round 5: loss improvement +0.0772, acc improvement +0.0000
    - seed 202 round 50: loss improvement +0.2020, acc improvement +0.0456
    - seed 202 round 100: loss improvement +0.3088, acc improvement +0.0580
    - seed 203 round 5: loss improvement +0.0146, acc improvement -0.0032
    - seed 203 round 50: loss improvement +0.2119, acc improvement +0.0452
    - seed 203 round 100: loss improvement +0.1100, acc improvement +0.0384

### C_valsel_vs_fixed_global

| seed | round | cohort | n_reps | mean loss ref | mean loss cand | loss improvement | acc improvement |
|------|-------|--------|--------|---------------|----------------|------------------|-----------------|
| 201 | 5 | alt_0 | 3 | 3.7321 | 3.7018 | +0.0303 | +0.0049 |
| 201 | 5 | alt_1 | 3 | 3.7615 | 3.7280 | +0.0335 | +0.0057 |
| 201 | 5 | original | 3 | 3.8304 | 3.8049 | +0.0255 | +0.0123 |
| 201 | 50 | alt_0 | 3 | 2.3743 | 2.3743 | +0.0000 | +0.0000 |
| 201 | 50 | alt_1 | 3 | 2.4103 | 2.4103 | +0.0000 | +0.0000 |
| 201 | 50 | original | 3 | 2.3995 | 2.3995 | +0.0000 | +0.0000 |
| 201 | 100 | alt_0 | 3 | 2.0386 | 2.0386 | +0.0000 | +0.0000 |
| 201 | 100 | alt_1 | 3 | 2.0900 | 2.0900 | +0.0000 | +0.0000 |
| 201 | 100 | original | 3 | 2.0899 | 2.0899 | +0.0000 | +0.0000 |
| 202 | 5 | alt_0 | 3 | 3.6945 | 3.6753 | +0.0191 | +0.0067 |
| 202 | 5 | alt_1 | 3 | 3.7082 | 3.6769 | +0.0313 | +0.0081 |
| 202 | 5 | original | 3 | 3.7944 | 3.7944 | +0.0000 | +0.0000 |
| 202 | 50 | alt_0 | 3 | 2.4247 | 2.4247 | +0.0000 | +0.0000 |
| 202 | 50 | alt_1 | 3 | 2.3865 | 2.3853 | +0.0012 | -0.0001 |
| 202 | 50 | original | 3 | 2.3961 | 2.3961 | +0.0000 | +0.0000 |
| 202 | 100 | alt_0 | 3 | 2.1336 | 2.1336 | +0.0000 | +0.0000 |
| 202 | 100 | alt_1 | 3 | 2.0655 | 2.0655 | +0.0000 | +0.0000 |
| 202 | 100 | original | 3 | 2.0894 | 2.0894 | +0.0000 | +0.0000 |
| 203 | 5 | alt_0 | 3 | 3.8126 | 3.7438 | +0.0688 | +0.0125 |
| 203 | 5 | alt_1 | 3 | 3.7889 | 3.7238 | +0.0651 | +0.0097 |
| 203 | 5 | original | 3 | 3.8692 | 3.7991 | +0.0700 | +0.0051 |
| 203 | 50 | alt_0 | 3 | 2.3925 | 2.3925 | +0.0000 | +0.0000 |
| 203 | 50 | alt_1 | 3 | 2.4123 | 2.4123 | +0.0000 | +0.0000 |
| 203 | 50 | original | 3 | 2.3770 | 2.3770 | +0.0000 | +0.0000 |
| 203 | 100 | alt_0 | 3 | 2.1504 | 2.1504 | +0.0000 | +0.0000 |
| 203 | 100 | alt_1 | 3 | 2.1404 | 2.1404 | +0.0000 | +0.0000 |
| 203 | 100 | original | 3 | 2.0422 | 2.0422 | +0.0000 | +0.0000 |
- cohort-mean over dev seed 201 (selection source; NOT a held-out evaluation): loss improvement +0.0099, acc improvement +0.0025 (n_cohort_cells=9)
- cohort-mean over held-out seeds (202, 203): loss improvement +0.0142, acc improvement +0.0023 (n_cohort_cells=18)
- trajectory (single observation per seed/round, reported separately):
    - seed 201 round 5: loss improvement +0.0299, acc improvement +0.0156
    - seed 201 round 50: loss improvement +0.0000, acc improvement +0.0000
    - seed 201 round 100: loss improvement +0.0000, acc improvement +0.0000
    - seed 202 round 5: loss improvement +0.0000, acc improvement +0.0000
    - seed 202 round 50: loss improvement +0.0000, acc improvement +0.0000
    - seed 202 round 100: loss improvement +0.0000, acc improvement +0.0000
    - seed 203 round 5: loss improvement +0.0675, acc improvement +0.0008
    - seed 203 round 50: loss improvement +0.0000, acc improvement +0.0000
    - seed 203 round 100: loss improvement +0.0000, acc improvement +0.0000

### C_valsel_vs_fixed_phase

| seed | round | cohort | n_reps | mean loss ref | mean loss cand | loss improvement | acc improvement |
|------|-------|--------|--------|---------------|----------------|------------------|-----------------|
| 201 | 5 | alt_0 | 3 | 3.7018 | 3.7018 | +0.0000 | +0.0000 |
| 201 | 5 | alt_1 | 3 | 3.7280 | 3.7280 | +0.0000 | +0.0000 |
| 201 | 5 | original | 3 | 3.8083 | 3.8049 | +0.0033 | +0.0031 |
| 201 | 50 | alt_0 | 3 | 2.3743 | 2.3743 | +0.0000 | +0.0000 |
| 201 | 50 | alt_1 | 3 | 2.4103 | 2.4103 | +0.0000 | +0.0000 |
| 201 | 50 | original | 3 | 2.3995 | 2.3995 | +0.0000 | +0.0000 |
| 201 | 100 | alt_0 | 3 | 2.0386 | 2.0386 | +0.0000 | +0.0000 |
| 201 | 100 | alt_1 | 3 | 2.0900 | 2.0900 | +0.0000 | +0.0000 |
| 201 | 100 | original | 3 | 2.0899 | 2.0899 | +0.0000 | +0.0000 |
| 202 | 5 | alt_0 | 3 | 3.6753 | 3.6753 | +0.0000 | +0.0000 |
| 202 | 5 | alt_1 | 3 | 3.6784 | 3.6769 | +0.0016 | +0.0005 |
| 202 | 5 | original | 3 | 3.8178 | 3.7944 | +0.0233 | +0.0003 |
| 202 | 50 | alt_0 | 3 | 2.4247 | 2.4247 | +0.0000 | +0.0000 |
| 202 | 50 | alt_1 | 3 | 2.3865 | 2.3853 | +0.0012 | -0.0001 |
| 202 | 50 | original | 3 | 2.3961 | 2.3961 | +0.0000 | +0.0000 |
| 202 | 100 | alt_0 | 3 | 2.1336 | 2.1336 | +0.0000 | +0.0000 |
| 202 | 100 | alt_1 | 3 | 2.0655 | 2.0655 | +0.0000 | +0.0000 |
| 202 | 100 | original | 3 | 2.0894 | 2.0894 | +0.0000 | +0.0000 |
| 203 | 5 | alt_0 | 3 | 3.7554 | 3.7438 | +0.0116 | +0.0001 |
| 203 | 5 | alt_1 | 3 | 3.7292 | 3.7238 | +0.0053 | +0.0028 |
| 203 | 5 | original | 3 | 3.8151 | 3.7991 | +0.0159 | +0.0008 |
| 203 | 50 | alt_0 | 3 | 2.3925 | 2.3925 | +0.0000 | +0.0000 |
| 203 | 50 | alt_1 | 3 | 2.4123 | 2.4123 | +0.0000 | +0.0000 |
| 203 | 50 | original | 3 | 2.3770 | 2.3770 | +0.0000 | +0.0000 |
| 203 | 100 | alt_0 | 3 | 2.1504 | 2.1504 | +0.0000 | +0.0000 |
| 203 | 100 | alt_1 | 3 | 2.1404 | 2.1404 | +0.0000 | +0.0000 |
| 203 | 100 | original | 3 | 2.0422 | 2.0422 | +0.0000 | +0.0000 |
- cohort-mean over dev seed 201 (selection source; NOT a held-out evaluation): loss improvement +0.0004, acc improvement +0.0003 (n_cohort_cells=9)
- cohort-mean over held-out seeds (202, 203): loss improvement +0.0033, acc improvement +0.0002 (n_cohort_cells=18)
- trajectory (single observation per seed/round, reported separately):
    - seed 201 round 5: loss improvement +0.0053, acc improvement +0.0072
    - seed 201 round 50: loss improvement +0.0000, acc improvement +0.0000
    - seed 201 round 100: loss improvement +0.0000, acc improvement +0.0000
    - seed 202 round 5: loss improvement +0.0058, acc improvement -0.0016
    - seed 202 round 50: loss improvement +0.0000, acc improvement +0.0000
    - seed 202 round 100: loss improvement +0.0000, acc improvement +0.0000
    - seed 203 round 5: loss improvement +0.0174, acc improvement -0.0052
    - seed 203 round 50: loss improvement +0.0000, acc improvement +0.0000
    - seed 203 round 100: loss improvement +0.0000, acc improvement +0.0000

## Interpretations (separate from the measurements above)

- A_freeze_buffers_at_gamma1 [all seeds (27 cohort cells)]: 3 positive / 24 negative / 0 zero of 27 cohort cells.
- A_freeze_buffers_at_gamma1 [held-out seeds only (18 cohort cells)]: 3 positive / 15 negative / 0 zero of 18 cohort cells.
- B_valsel_vs_full_gamma1 [all seeds (27 cohort cells)]: 23 positive / 2 negative / 2 zero of 27 cohort cells.
- B_valsel_vs_full_gamma1 [held-out seeds only (18 cohort cells)]: 15 positive / 1 negative / 2 zero of 18 cohort cells.
- B_valsel_vs_params_only_gamma1 [all seeds (27 cohort cells)]: 24 positive / 3 negative / 0 zero of 27 cohort cells.
- B_valsel_vs_params_only_gamma1 [held-out seeds only (18 cohort cells)]: 15 positive / 3 negative / 0 zero of 18 cohort cells.
- C_valsel_vs_fixed_global [all seeds (27 cohort cells)]: 9 positive / 0 negative / 18 zero of 27 cohort cells.
- C_valsel_vs_fixed_global [held-out seeds only (18 cohort cells)]: 6 positive / 0 negative / 12 zero of 18 cohort cells.
- C_valsel_vs_fixed_phase [all seeds (27 cohort cells)]: 7 positive / 0 negative / 20 zero of 27 cohort cells.
- C_valsel_vs_fixed_phase [held-out seeds only (18 cohort cells)]: 6 positive / 0 negative / 12 zero of 18 cohort cells.

- Freezing checkpoint buffers at gamma=1 in THIS setting does not improve full-state FedAvg on the evaluation split (held-out cohort-mean loss improvement -0.0629; dev seed -0.0705). This conclusion is limited to freezing checkpoint buffers at gamma=1 in this setting; it does NOT generalize to other normalization or buffering methods.
- Validation-selected damping vs full-state gamma=1: +0.0646; vs frozen-buffer gamma=1: +0.1276 (held-out cohort means). It improves over BOTH gamma=1 references.
  Both comparisons use the SAME validation-selected candidate, so the difference between the two improvements (+0.1276 vs +0.0646) reflects the different REFERENCE losses; it is not evidence of an interaction between damping and buffer handling.
- Validation-selected vs dev-selected fixed scales (held-out): vs global +0.0142, vs phase +0.0033. (Dev seed, not a held-out evaluation: vs global +0.0099, vs phase +0.0004.)

- Caveat (frozen): the frozen-buffer scale grid was NOT measured (only gamma=1 under the secondary buffer policy), so parameter-scaling and buffer-scaling effects remain INCOMPLETELY SEPARATED.

## Additional measurements

An evaluation-only extension of the secondary buffer policy over the existing gamma grid on the existing checkpoints was identified earlier as the smallest separating measurement. Per the pilot closure decision it is NOT launched, and no further tuning against the same held-out seeds is performed.
