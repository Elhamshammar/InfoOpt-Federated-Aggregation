# Matched RA analysis — 10 October 2026

All 60 fresh runs verified from collected metadata; all arms AIB enabled. Protocol and analysis plan frozen before launch; not external preregistration. Historical runs and the previous four-arm study are not pooled.

## Primary family (Holm across six)

|Dataset|Contrast|Difference pp|95% CI pp|Holm p|
|---|---|---:|---|---:|
|cifar100|ra_probeoff minus fedavg_aibon|+0.208|[+0.030, +0.386]|0.1888|
|cifar100|ra_probeoff minus infoopt_aibon|+0.038|[-1.039, +1.115]|1.0000|
|eurosat|ra_probeoff minus fedavg_aibon|+2.096|[-1.967, +6.160]|0.9013|
|eurosat|ra_probeoff minus infoopt_aibon|+6.681|[+0.428, +12.935]|0.2064|
|pathmnist|ra_probeoff minus fedavg_aibon|-0.025|[-2.251, +2.200]|1.0000|
|pathmnist|ra_probeoff minus infoopt_aibon|+0.919|[-1.295, +3.134]|0.9400|

## Secondary family (Holm across three)

|Dataset|Probe-on minus probe-off pp|95% CI pp|Holm p|
|---|---:|---|---:|
|cifar100|-0.560|[-1.777, +0.657]|0.5408|
|eurosat|-1.044|[-2.560, +0.472]|0.3850|
|pathmnist|-0.067|[-3.041, +2.907]|0.9532|

All estimates have n=5. CIs are individual paired-t 95% intervals, not multiplicity-adjusted simultaneous intervals. Holm decisions use the frozen separate families. Zero-crossing intervals express uncertainty, not equivalence.

## Evidence and interpretive limits

537 collection file hashes and the preparation frozen hashes checked. All 60 records have the correct arm/method, AIB, seed, frozen settings, exact unique rounds 1–100, finite losses and bounded accuracies, no early stop, exact shared partition/cohort agreement, and expected probe coverage. No failed attempts are recorded. Checkpoints remain on server: local verification relies on hashed server reports for strict model loading and tensor finiteness. Diagnostic NaNs in non-endpoint optional columns are not treated as endpoint failures.

The manifest contains a stale experiment label four_arm_20261009; it is preserved and flagged, not used for run selection. RA paths, arms, frozen settings and all 60 cell keys establish campaign identity.

Probe-on versus off measures instrumentation sensitivity, including unisolated probe-model initialization RNG effects. It does not establish a different aggregation mechanism or a pure probe-evaluation causal effect. Probe subsets overlap training and do not select configurations. All arms are AIB on: no AIB interaction is estimated here. Results are specific to this code, training recipe, alpha and five seeds.

For writing AI: add this prospective matched study in a separate RA data namespace. Preserve historical tables, prior AIB factorial analysis and all frozen sources. Report all six primary and all three secondary contrasts; do not choose a better RA arm after seeing test results.

## Findings

None of the six primary comparisons survives Holm correction (minimum adjusted p=0.1888). Two individual unadjusted 95% intervals exclude zero: CIFAR RA minus FedAvg and EuroSAT RA minus InfoOpt. This is not a consistent or multiplicity-adjusted superiority result. None of the three instrumentation comparisons survives Holm correction; their individual intervals all include zero.
