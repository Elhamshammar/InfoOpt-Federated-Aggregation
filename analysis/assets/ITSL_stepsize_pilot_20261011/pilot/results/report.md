# CIFAR-100 Pilot Report

Frozen documents: PROTOCOL.md, GO_NO_GO.md, PRE_EXECUTION_AMENDMENT.md (frozen BEFORE execution). v1 frozen docs preserved under PROTOCOL_v1_2026-10-04.md / GO_NO_GO_v1_2026-10-04.md. v2 frozen docs preserved under PROTOCOL_v2_2026-10-04.md / GO_NO_GO_v2_2026-10-04.md.

**Pilot config fingerprint:** 281ecf3f1bbb56ee...
**Decisions (NOT a single GO/NO-GO):**

| Decision | Status | Reason |
|----------|--------|--------|
| adaptive_headroom | INSUFFICIENT | screen_unsatisfied_against_both_controls_on_at_least_one_held_out_seed |
| claude_comparator | NO-GO | at_least_one_seed_or_control_failed_claude_screen |
| exploratory_variability_signal | EXPLORATORY_PROBE | EXPLORATORY: two predictors fit; 0 common keys |
| overall_recommendation | ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT | synthesis of the above |

## Variability summary

Scope: per `(training_seed, checkpoint_round)`; trajectory is reported separately within each per_checkpoint record (not averaged with reps).
- (no per-checkpoint variability records)

## Audit hashes

- protocol_sha256_glob: 2abedacdb04eaedd71da2d1f0d7529bd008d79c2d723c740e17d7fdbf2f2aa82
- source_keys_dev_only_sha256: f48ac4d0d157cc5c397c5e21e464a1948ebf2171f7e16fd5e8ee8468530ddd0d
- pilot_config_fingerprint: 281ecf3f1bbb56ee6607f4df096e9825ba0250167501d8a46e7bd69f9079ef06