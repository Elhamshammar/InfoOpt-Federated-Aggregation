# IT-SL four-arm analysis — 10 October 2026

## Verification
Archive SHA-256: 0a56c0080cb8da42566be1ca4a976937d50f1f273b10eb06d4eb76d4b660f385. All 406 collected files match their collection hashes; all 60 unique dataset–seed–arm records have matching execution-manifest evidence hashes and successful server verification reports. Independent checks confirm exact rounds 1–100; finite test accuracy, test loss and mean training loss; configured arm identity; common settings; early stopping disabled/not triggered; exact runtime partition equality and cohort equality to shared schedules. Full saved configs agree after removing only seed, experiment name and intended AIB switch. Checkpoints remain on the server: their hashes and successful load verification are recorded, but checkpoint loading was not independently repeated here.

## Primary: InfoOpt-Full minus FedAvg
Endpoint: round-100 test accuracy. Paired by seeds 100–104, n=5 for every contrast. Units: percentage points. Unadjusted paired-t 95% confidence intervals; two-sided p-values; Holm correction across all six primary tests.

| Dataset | AIB | Difference (pp) | 95% CI | Raw p | Holm p |
|---|---|---:|---|---:|---:|
| cifar100 | aiboff | +0.536 | [-0.359, 1.431] | 0.1718 | 0.6871 |
| cifar100 | aibon | -0.184 | [-1.249, 0.881] | 0.6565 | 0.7356 |
| eurosat | aiboff | -4.156 | [-10.230, 1.919] | 0.1303 | 0.6517 |
| eurosat | aibon | -3.741 | [-8.490, 1.008] | 0.0940 | 0.5640 |
| pathmnist | aiboff | -1.565 | [-5.225, 2.094] | 0.3007 | 0.7356 |
| pathmnist | aibon | -0.304 | [-0.923, 0.316] | 0.2452 | 0.7356 |

All six intervals include zero. This is uncertainty, not evidence of equivalence or proof that aggregation has no effect. These runs do not establish InfoOpt superiority.

## Secondary: aggregation × AIB interaction
Definition: (InfoOpt on − InfoOpt off) − (FedAvg on − FedAvg off), equivalently (InfoOpt − FedAvg on) − (InfoOpt − FedAvg off). Paired n=5, separate prespecified Holm family of three.

| Dataset | Interaction (pp) | 95% CI | Raw p | Holm p |
|---|---:|---|---:|---:|
| cifar100 | -0.720 | [-1.108, -0.332] | 0.0067 | 0.0202 |
| eurosat | +0.415 | [-1.984, 2.814] | 0.6562 | 0.8915 |
| pathmnist | +1.262 | [-2.885, 5.408] | 0.4457 | 0.8915 |

CIFAR-100 shows a negative interaction under the planned paired-t analysis: AIB changes InfoOpt’s relative effect by −0.72 pp (Holm p=0.0202). This does not establish that InfoOpt is worse within either AIB setting; those primary contrasts are individually uncertain. Other interactions are uncertain. With only five seeds, inference depends on the paired-t assumptions and remains specific to this fixed protocol.

## Scope
This is a new controlled experiment, separate from historical retrospective reanalyses. It addresses the architecture confound for FedAvg versus InfoOpt-Full within this experiment. It does not resolve historical RA-InfoOpt configuration differences or FedDyn/SCAFFOLD implementation provenance. No additional training, tuning or manuscript edits were performed during this analysis.

## Reproduction
Run analyze_fourarm.py from a directory containing extracted input evidence at fourarm_results/. It aborts on failed checks before calculating statistics. Requires NumPy and SciPy. Per-seed values, exact contrasts, summary mean/SD, verification report and original frozen protocol are included.
