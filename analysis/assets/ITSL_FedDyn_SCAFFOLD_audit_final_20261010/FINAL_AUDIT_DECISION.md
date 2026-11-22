# Consolidated FedDyn/SCAFFOLD audit — 10 October 2026

This decision note supersedes the next-step recommendations in AUDIT_REPORT.md, which is preserved as the initial audit record.

## Verified outcome
The uploaded Windows execution report records PASS for 17 CPU synthetic checks (PyTorch 2.5.1+cu121). Its script SHA256 exactly matches the supplied test script; its five snapshot hashes exactly match the supplied source manifest. This confirms reported execution against the intended package identities; the tests were executed by the author, not independently rerun in this environment. The test script checks hashes before imports. Report authenticity is based on the submitted execution artifact and terminal output, not remote attestation.

The 17 checks cover SCAFFOLD local correction, actual-step Option-II controls, selected/unselected controls, exact extracted server participation update; FedDyn regularizer gradient/sign, local two-step solver, selected-client dual increments, unselected history and all-client bias across rounds; and effective zero-momentum overrides. Utility-level weighted/uniform aggregation comparisons are included. Nonzero clipping/weight decay, BN buffers, full training orchestration and CUDA behavior remain outside the test scope.

The live-server launch provenance record is byte-identical to its recovered copy (a239b9eb4bc436e6d38c7e0fe122bbdd1433ae2fdefb4f072ac30aa4a2d09e61). It records 12 August 2026 launch-time code sizes/timestamps and presence of repaired functions. This is evidence that repaired code was present at launch. There are no historical source hashes or Git commits, so exact executed bytes remain unverified.

Thirty repair-campaign runs have exact saved-path links in their individual logs and matching 100-round metrics, using provenance rather than accuracy selection. The initial report contains their descriptive means and the real older/newer FedDyn recursion difference. No specific old run has been proven to execute the older code.

## Recommendation for the manuscript
Keep the existing frozen retrospective analysis unchanged. Treat the phase-2 repair campaign as a separately labelled historical descriptive analysis if included, with source-version limitations explicit. Do not silently replace earliest-run values, pool implementation eras, or use these results to claim FedDyn/SCAFFOLD inferiority or superiority. The tests support the available repaired implementation; they do not retrospectively certify historical source execution.

For all ten CIFAR repair-campaign cells, early-stop-enabled saved configs conflict with the strict frozen E3 criterion. Current recovered code gates that controller to InfoOpt/RA, and all ten runs complete 100 rounds without triggering; this does not itself amend E3. Keep these cells outside the frozen primary analysis unless a separately dated method-applicability amendment is explicitly adopted independently of accuracy.

No additional training is required merely to document this audit or retain the diagnostic manuscript's narrow matched FedAvg/InfoOpt result. If a clean, reproducible comparative claim about FedDyn/SCAFFOLD is essential, prepare a new baseline protocol with frozen executed code, shared partitions/schedules, explicit aggregation/clipping/weight-decay/momentum settings and strict run verification. Synthetic PASS alone is not sufficient justification to launch it.

No training, real-data evaluation, tuning, manuscript edits or alteration of prior frozen outputs occurred in this audit. Synthetic local optimizer steps were executed by the author on CPU only.
