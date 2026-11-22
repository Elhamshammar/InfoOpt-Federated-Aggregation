# FedDyn/SCAFFOLD historical audit — 10 October 2026

## Outcome

Repair-campaign run provenance has been recovered for 30 runs. Historical executable-source provenance remains unverified. These findings justify revising the provenance description, but do not authorize replacing frozen primary results or ranking these baselines. No training, tuning, evaluation, manuscript edits, or changes to prior packages were performed.

## Evidence recovered and checked

The recovered inventory contains 70 candidate directories: 35 per method. There are 60 complete 100-round records (30 earlier runs and 30 repair-campaign runs), plus 10 partial attempts. Each repair campaign covers three datasets and seeds 100–104. Its individual log names an exact saved-results directory; all 100 logged accuracies agree with that directory's metrics within printed rounding tolerance. Each linked run has unique rounds 1–100, finite test loss and accuracy, accuracy in [0,1], the expected method/seed, and AIB disabled. All 30 summaries report no triggered early stop.

`campaign_source_map.json` records exact paths, log hashes and hashes of metrics/config/summary. Selection is by the campaign's saved path, never by accuracy. The campaign launchers and both campaign logs explicitly describe repair reruns; they do not record a code commit or executed-source hash. Consequently, current recovered code cannot be certified as the implementation used for these runs.

## Descriptive campaign endpoints

These are audit summaries, not replacement manuscript tables. Accuracy is percentage; SD is sample SD; each row contains five distinct seeds.

| Method | Dataset | Mean ± SD (%) | n |
|---|---|---:|---:|
| SCAFFOLD | CIFAR-100 | 29.100 ± 0.875 | 5 |
| SCAFFOLD | EuroSAT | 74.844 ± 3.662 | 5 |
| SCAFFOLD | PathMNIST | 85.097 ± 4.294 | 5 |
| FedDyn | CIFAR-100 | 42.278 ± 1.270 | 5 |
| FedDyn | EuroSAT | 87.919 ± 1.027 | 5 |
| FedDyn | PathMNIST | 85.582 ± 2.229 | 5 |

The SCAFFOLD CIFAR campaign mean reproduces the printed 29.10%, whereas the earliest recovered complete runs average 1.764%. This provides an explicit campaign trail for the discrepancy. It does not prove the cause of the earlier poor performance, nor recover the historical Tables 3–4 selection ledger.

## Implementation findings

### FedDyn: a concrete historical code difference

The older `train.py.bak.lines` initializes one shared `feddyn_y` dictionary, passes it to clients, and updates it from the change in the aggregated global model. It does not implement the recovered newer snapshot's persistent per-client dual states and all-client bias correction. This is a substantive difference in the implemented recursion, not a filename or cosmetic difference.

The newer `utils/feddyn.py` uses the local penalty alpha/2 × ||theta−global||² + alpha × <theta,dual_client>, selected-client dual increments, and uniform selected-model averaging plus the all-client mean dual. Under dual = −historical gradient/alpha, these core recursions correspond to Algorithm 1 of Acar et al. The current local solver also uses finite SGD steps, clipping and weight decay; static correspondence is not a full numerical correctness test.

No recovered source hash binds either implementation to an earlier run or a repair run. Therefore the audit does **not** assert that a particular run executed the defective older recursion.

### SCAFFOLD: core control recursion present, adaptations require explicit scope

The newer client code applies gradient correction c−ci and Option-II control updates using the actual number of local steps. The server updates selected client controls and adds their sum divided by the total client count. These correspond to the core control recursions of Karimireddy et al., Algorithm 1.

The recovered default model aggregation uses sample-count weights; the control update averages selected control differences uniformly before multiplication by participation fraction. The original displayed algorithm averages selected model updates uniformly, while the paper allows weighted extensions. This is an implementation adaptation requiring explanation and numerical tests, not by itself proof of a bug. Gradient correction is followed by clipping; weight decay remains enabled. No older client implementation was recovered that establishes the cause of SCAFFOLD's earlier poor endpoints.

The current `local_momentum` explicitly returns zero for both methods despite saved training momentum being 0.9. Runtime settings cannot be inferred from config alone, and this override cannot be ascribed to historical runs without source linkage.

Identical FedDyn utility copies and namespace-only differences between client reference copies do not supply independent historical version evidence.

## Eligibility conflict requiring a separate decision

For all ten CIFAR repair-campaign runs, the saved InfoOpt early-stop setting is enabled; the earlier complete runs disable it. This is the only flattened saved-config difference for those comparisons. The 20 EuroSAT/PathMNIST earlier-versus-campaign comparisons have identical saved configs.

In the recovered newer train.py, the early-stop call is gated to InfoOpt/RA methods, so it would not stop FedDyn/SCAFFOLD; all linked runs complete 100 rounds and all summaries report false. Nevertheless the frozen retrospective E3 rule excludes configurations with early stopping enabled. These ten CIFAR runs must not enter that primary analysis silently. A method-applicability amendment, if adopted, must be separately dated and justified independently of accuracy, with historical-source uncertainty retained.

## Recommended next step

Keep these baseline rows qualified and out of comparative rankings. Perform one narrow read-only provenance search on the server for the August 12–13 repair source state: git commits/reflog, dated backups of train.py and methods/client.py, repair patch files, and repair-campaign notes or logs containing source hashes. Do not resend datasets or all metrics; those are already recovered.

If historical source linkage cannot be recovered, choose explicitly between retaining these as historical descriptive results with limitations and executing new, separately documented baseline comparisons. New training is not yet justified solely by the endpoint discrepancy.

Before any new baseline experiment, numerically test the intended implementation: FedDyn selected-client dual persistence and all-client bias term; SCAFFOLD Option-II controls, unselected-client persistence, participation scaling, and the chosen weighted/uniform model update. Document clipping, weight decay and effective momentum. Compare against common frozen partitions/schedules and explicitly record executed code hashes.

## Reproduction and scope

Run `audit_feddyn_scaffold.py` from the recovered workspace root, which must contain `itsl_followup/home/whu/disk1/infoopt_dl_server/` and `run_metadata_recovery/runs/`. The packaged evidence copies preserve the inspected code and repair logs; the outputs preserve exact identity hashes and config differences. The script verifies saved evidence and calculates descriptive endpoints only. It does not load models or execute training.

This audit uses recovered local copies. It did not inspect the live server, reload historical checkpoints, certify historical partitions, or run algorithm regression tests. The code comparison is static. Originals and frozen primary analyses remain untouched.

## Primary references

- Karimireddy et al. (2020), SCAFFOLD, Algorithm 1: https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf
- Acar et al. (2021), Federated Learning Based on Dynamic Regularization, Algorithm 1: https://arxiv.org/abs/2111.04263
