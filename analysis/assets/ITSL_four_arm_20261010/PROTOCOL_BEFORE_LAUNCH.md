# FOUR-ARM EXPERIMENT PROTOCOL — frozen for review (not launched)

**Frozen:** 2026-10-09. This protocol is for a NEW experiment directory
(`<REDACTED_HOME>/disk1/ITSL_four_arm_20261009/`). It does not modify the
manuscript, existing runs, or any frozen audit artifact.

## Purpose

Isolate aggregation from architecture for the manuscript's central
comparison: **FedAvg and InfoOpt-Full, each with AIB OFF and ON** — the
2×2 grid that the historical campaign lacked (its FedAvg arm had AIB
disabled by hardcoded default; see recovered `train.py` lines 112–116).

## Grid and arms

- Datasets: CIFAR-100, EuroSAT, PathMNIST (as in the manuscript).
- Seeds: 100, 101, 102, 103, 104.
- Arms (method, aib-override):
  1. `fedavg_aiboff` — `--method fedavg` (AIB auto-off, canonical arm)
  2. `fedavg_aibon` — `--method fedavg --aib-override true` (the missing arm)
  3. `infoopt_aiboff` — `--method InfoOpt-Full --aib-override false`
  4. `infoopt_aibon` — `--method InfoOpt-Full` (canonical arm)

60 runs total. The ONLY per-arm differences are the aggregation method
and the AIB switch. Everything else (local epochs 2, lr 0.01, momentum
0.9, wd 5e-4, batch 64, clip 5, 10 clients, 5/round, Dirichlet α=0.3,
min client size 50, 100 rounds) is identical, from the canonical sweep
configs.

## Shared structure (identity guarantees)

- Partition indices: Dirichlet(α=0.3) with `seed=<run seed>` — identical
  for all four arms of a dataset-seed because train.py partitions with
  `seed=cfg["seed"]` after the CLI override. Materialized by
  `generate_shared_schedules.py` into `shared/partition_<ds>_seed<S>.json`.
- Cohort schedules: `random.Random(<seed>).sample` sequence, matching
  train.py's sampler; materialized into
  `shared/cohort_schedule_<ds>_seed<S>.json`. `verify_run.py` compares
  each run's recorded `selected_clients` column against it after training.

## Disabled mechanisms

- Early stopping: disabled (`infoopt.early_stop.enabled: false`).
- Counterfactual oracle probes: REMOVED from all arm configs (the probe
  advances the torch RNG stream and perturbs trajectories — audit
  finding; the four-arm study excludes it so all arms share one clean
  RNG policy).
- Fixed-gamma damping: not used (γ=1 for all arms).

## Endpoint

Terminal round-100 test accuracy (fraction in [0,1]), exactly as in the
manuscript. The exporter is verified: snapshot `train.py` reads
`metrics["acc"]` from `utils.metrics.evaluate()` (the canonical exporter;
the `"accuracy"` key defect existed only in the fixed-γ/DDM runners).
`verify_run.py` re-checks this per run, plus exact rounds 1–100, finite
accuracy, no early stop, and cohort-vs-schedule identity.

## Code snapshot

`snapshot/` contains the recovered server code with **two documented
changes**: (1) the `--aib-override {true,false}` CLI flag, applied after
the method-based AIB defaults (enables the FedAvg+AIB arm that the
original code cannot express); and (2) persistence of
`partition_used.json` — the exact client indices each run computed and
used — so the verifier can prove equality with the frozen shared
partition. See `PATCH_NOTES.md`. Everything else is byte-identical to the
recovered sources (hashes in `SNAPSHOT_HASHES.sha256`).

## Provenance and limitations

- The snapshot is assembled from the recovered code archive (methods/,
  utils/, train.py, configs/), the recovered models/ files, and the
  fetched data/datasets.py + data/partition.py. All hashed. The original
  train.py (`train.py.orig`, sha256 `ab595350…`) matches the live-server
  file hash verified earlier.
- Datasets are resolved to ABSOLUTE, existence-checked roots at launch by
  `resolve_dataset_roots.py` (order: `$DATASETS_ROOT/<subdir>` →
  `EXP_ROOT/datasets/<subdir>` → canonical
  `<REDACTED_HOME>/disk1/infoopt_dl_server/datasets/<subdir>`), and effective
  configs are written to `EXP_ROOT/configs_effective/` with hashes.
  All three datasets must be present before any training starts; the
  check runs from any working directory.
- Partition membership for cross-arm pairing is guaranteed by
  construction (identical partition files and schedules per dataset-seed),
  removing the disclosed assumption for THIS experiment; each run's
  `partition_used.json` is verified equal to the frozen partition.
- Results will be new evidence, not a replacement for any historical run.

## Launch (after review)

1. Copy this package to the server (path `PKG`).
2. `cd` anywhere; `PKG=… EXP_ROOT=… bash launch_four_arm_gpu1.sh`.
3. Launcher: frozen-integrity → dataset-root resolution (absolute,
   existence-checked) → schedules → integrity re-check → sequential
   training with full verification gating every skip and completion →
   execution manifest under `EXP_ROOT/RUN_MANIFEST.json`. ~8–15 GPU-hours
   on GPU 1.

## Analysis plan (pre-registered before launch)

**Endpoint:** terminal round-100 test accuracy (fraction in [0,1]) per run,
from the verified `metrics.csv` only. No best-round selection.

**Primary contrasts (family of 6):** within each AIB setting, InfoOpt-Full
minus FedAvg, paired by seed (n=5) per dataset:
- AIB off: `infoopt_aiboff − fedavg_aiboff` × 3 datasets
- AIB on: `infoopt_aibon − fedavg_aibon` × 3 datasets

**Secondary interaction (family of 3):** aggregation×AIB interaction per
dataset: `(infoopt_aibon − infoopt_aiboff) − (fedavg_aibon − fedavg_aiboff)`.

**Statistics:** per contrast — n (actual), mean paired difference, sample
SD, paired-t 95% CI, raw two-sided p, Holm–Bonferroni adjustment within
each family (6 and 3 respectively). Pairing by seed is exact for this
experiment (identical frozen partitions and cohort schedules per
dataset-seed, verified at runtime — no membership assumption remains).
Zero-including intervals are uncertainty, not equivalence.

**Failed-run policy (two distinct classes):**

1. **Infrastructure failure** (process crash, OOM, disk/GPU/transient
   error — identifiable from the run log and the absence of a valid
   run directory): a cell may be retried ONCE, but only as a DOCUMENTED
   manual step: the failed attempt's directory is MOVED out of `runs/`
   into `EXP_ROOT/failed_attempts/` (the move and reason are recorded
   under `EXP_ROOT/verification/`), so the cell again has exactly one
   active candidate directory, and the launcher is re-run. Archived
   attempts are never counted as active candidates and are listed in the
   execution manifest per cell. The launcher itself never auto-retries.
   All attempts are preserved and appear in the execution manifest.
2. **Numerical or methodological failure** (non-finite losses, diverged
   training, or content that fails verification): the run is reported as
   `invalid` with the verifier's reason. It is NEVER retried silently,
   never replaced, and never substituted by another run.

**Contrast eligibility:** a paired InfoOpt−FedAvg contrast for a seed
requires BOTH arms valid for that seed (both AIB settings computed
independently); an interaction contrast requires ALL FOUR arms valid for
that seed. Missing eligibility is reported with actual n (n<5 flagged;
n<3 → not estimable). No failed run is replaced, and no repeat is chosen
by accuracy or any outcome. No tuning of any setting based on test
accuracy is permitted; the launcher may be re-run only to complete or
repair cells, with verified-complete cells skipped by the resume rule.
