# Matched RA experiment preparation — 10 October 2026

Status: preparation only. No full experiment runs launched. This is an internally documented prospective plan, not external preregistration. Freeze this package before launch, retaining its archive hash and preflight log. No historical results are substituted into this study.

## Questions and arms

Main question: under one matched architecture and training recipe, does RA improve terminal accuracy relative to FedAvg and InfoOpt-Full?

All arms use AIB on: fedavg_aibon (FedAvg), infoopt_aibon (InfoOpt-Full), ra_probeoff (RA-InfoOpt without counterfactual probes), ra_probeon (the same RA controller with counterfactual probes). Three datasets (CIFAR-100, EuroSAT, PathMNIST), seeds 100–104, 100 rounds: 60 fresh runs. Do not reuse historical or recent controls. Fresh controls remove source-era uncertainty.

The secondary question is instrumentation sensitivity, not selection of a best RA mechanism. The two RA configs differ ONLY in counterfactual_probe.enabled. Both explicitly freeze the same safety values: JS deviation 0.05, ENC deviation 1.0, 20 binary-search steps, tolerance 1e-8; gate/reliability/smoothing settings are preserved from the recovered design. These values are not chosen by test performance.

Recovered defaults show July absent safety limits and August explicit values coincide under this code. Thus this study is not claimed as an exact reconstruction of the historical July and August executable states.

## Matched settings

lr 0.01; momentum 0.9; weight decay 0.0005; two local epochs; batch 64; clipping 5; ten clients; five selected per round; Dirichlet alpha 0.3; minimum 50 samples/client; 100 rounds; early stopping disabled. Configs are dataset-specific where required and identical across arms except experiment identity and probe switch. CLI selects method and forces AIB true after method defaults. Shared partition indices and cohort schedules are generated once per dataset/seed and verified against each run. Effective config sections, arm identity, checkpoints and endpoints are verifier-gated.

## Instrumentation scope

Probe-on uses the existing implementation unchanged: extra probe model is initialized after the training model without RNG isolation; per-round evaluation restores CPU and active-device CUDA RNG. Probe cadence is every five rounds, with ratio 0.15 and seed+10000. Probe evaluation uses a subset of the training dataset, including its training transforms; it is not an independent validation holdout and must not select configurations or endpoints. Initialization and instrumentation overhead/RNG effects are part of this diagnostic. A difference would not identify which component caused it or demonstrate a second aggregation mechanism. No patch to isolate RNG is silently introduced.

## Verification and endpoint

Use round-100 test accuracy exclusively. Exact unique rounds 1–100, finite losses, finite accuracy in [0,1], no triggered early stop, strict finite checkpoint load with AIB, recorded partition/cohort equality and correct effective config required. Probe-on requires exactly rounds 5,10,...,100 in counterfactual_probe.csv; all other arms must lack that file. Full verification gates skips and manifest completion; duplicate directories or invalid existing evidence abort. Scripts and configs are integrity-checked before preparation/training. CPU controller tests must pass before dataset preflight.

## Analysis plan

Primary family: six paired two-sided contrasts — RA-probeoff minus FedAvg and RA-probeoff minus InfoOpt-Full, each across three datasets. Report mean difference in percentage points, sample SD of paired differences, paired-t 95% CI, raw p and Holm-adjusted p across six. Secondary family: three paired RA-probeon minus RA-probeoff contrasts with separate Holm correction across three. Also report every arm's mean, SD, per-seed values and actual n. No accuracy-based selection, best-round selection, pooling of repeats or secondary-arm substitution. CI crossing zero is uncertainty, not equivalence. No aggregation-by-AIB interaction is estimable because AIB is always on here.

Contrast pairs require both arms valid for the same seed and identical partition/cohort hashes. Failed cells remain visible; n<5 flagged, n<3 not estimable. Preserve family size with non-estimable contrasts treated as non-rejections; do not shrink the Holm family. No post-result changes to configuration or analysis.

## Operations and failure policy

Linux GPU 1 only; sequential runs; distinct EXP_ROOT <REDACTED_HOME>/disk1/ITSL_RA_matched_20261010. PREFLIGHT_ONLY=1 exits after CPU checks, integrity, dataset resolution and 15 shared schedules. It creates no training run directories beyond the empty runs container. Do not edit the existing four-arm experiment directory.

One documented manual retry is allowed for an infrastructure failure, preserving the failed directory under EXP_ROOT/failed_attempts outside active discovery, with reason/log. Numerical or methodological invalidity must be reported and not silently replaced. Launcher never retries automatically. Preserve all attempts. No arbitrary extra seeds.

## Validation status

Preparation-side syntax/static/package tests are recorded in PREPARATION_CHECKS.json. This environment has no PyTorch; controller CPU tests, checkpoint verifier execution and real-data preflight MUST run in the server environment before launch. Original four-arm regression results do not automatically certify this adaptation. Source snapshot code is byte-preserved from four-arm v8; only configs and orchestration/verification tools are adapted. AIB override and partition persistence remain the two earlier documented train.py patches.
