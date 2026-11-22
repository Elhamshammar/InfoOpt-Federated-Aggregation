# Intentional Differences from the Reference Pipeline

Every deliberate deviation of this diagnostic from the reference
implementation (now bundled verbatim under `reference_code/`, see
COPY_MANIFEST.json) is listed here. Anything found at runtime that is NOT
listed here is reported as a recorded difference in
`results/reference_comparison.json` (and, for aggregation, aborts the run).

## D1. Determinism policy
- Reference: `reference_code/utils/seed.py` `set_seed` sets
  `cudnn.deterministic=False`, `cudnn.benchmark=True` (verified at runtime
  by `ref_compare.compare_set_seed` against the bundled snapshot).
- This diagnostic: `cudnn.deterministic=True`, `cudnn.benchmark=False`
  everywhere, reapplied after every `set_seed` call (the REAL reference
  `set_seed` is used; only the cudnn policy is overridden).
- Reason: verification checks 7 and 8 (identical-seed retraining,
  shared-client matched-seed agreement) require bitwise-reproducible local
  updates; measured cuDNN atomics nondeterminism is ~1e-2 per element
  otherwise.

## D2. Holdout removal from client training indices
- Reference/old counterfactual setup: validation indices were not visibly
  removed from client training data.
- This diagnostic: the 5% selection + 5% evaluation holdouts are removed
  from every client's indices, disjointness is asserted, and weights are
  recomputed from retained counts.
- Reason: explicit instruction; the overlap would leak evaluation data
  into client training and invalidate the scale-selection comparisons.

## D3. Local-call construction
- Reference `train_client`: `copy.deepcopy` of the global model.
- This diagnostic: fresh instance of the REAL reference SmallCNN +
  `load_state_dict` (state-equivalent transfer), fresh SGD per call
  preserved [F4].
- Reason: equivalent state semantics without deepcopy of module objects;
  recorded so any subtle difference (e.g. module hooks) is visible.

## D4. DataLoader and augmentation plumbing
- Reference: dataset-level torchvision transforms
  (`reference_code/data/datasets.py` `_cifar_transforms`).
- This diagnostic: raw uint8 arrays (loaded via the reference `get_cifar`,
  whose dataset transforms are never invoked); the SAME recipe (pad-4
  random crop + horizontal flip + normalize with the REFERENCE constants,
  imported from `reference_code.data.datasets`) applied functionally per
  batch in the local-training loop only; evaluation uses normalize-only
  (the reference `test_tf` recipe); the shared arrays are never mutated;
  `num_workers=0`; and EVERY DataLoader gets an explicit generator
  (without one, iteration draws a base seed from the global RNG even with
  `shuffle=False`).
- Reason: instruction (do not mutate the shared training dataset's
  transform) and determinism.

## D5. Client epoch seeds
- `epoch_seed = rep_seed * 100000 + checkpoint_round * 1000 + client_id`.
  Depends only on (checkpoint, repetition seed, client ID), never on cohort
  position, so a shared client yields the same update under a matched seed
  across cohorts (verified by check 8).

## D6. Partition acceptance loop
- The core Dirichlet draw is the REAL reference
  `reference_code.data.partition.dirichlet_partition`. The reference's own
  internal retry loop checks the ORIGINAL client sizes; the diagnostic
  bypasses it (`min_samples_per_client=0`) because its size floor applies
  to the RETAINED sets after holdout removal (D2). The wrapper redraws
  (`partition_seed + attempt`) until every retained client set has ≥ 50
  samples; the attempt count is logged.

## D7. Evaluation data
- Scales are selected and evaluated only on training-derived holdouts;
  the official test set is never read by the diagnostic.

## D8. FedExP implementation
- Labelled PROVISIONAL: formula from the recorded statement of
  arXiv 2301.09604; the unequal-weight handling is an explicit adaptation,
  not the original algorithm.

## D9. Claude raw shrinkage rule
- Experimental; common-mean/noise-model justification only; not claimed to
  minimize global training loss; all safeguard activations logged.

---

## Resolved by the snapshot integration (no longer differences)

- **Normalization constants.** The pre-snapshot reconstruction used
  CIFAR-100 mean (0.5071, 0.4867, 0.4408) and std (0.2675, 0.2565, 0.2761).
  The verified reference values are (0.5071, 0.4865, 0.4409) and
  (0.2673, 0.2564, 0.2762); the diagnostic now imports them directly from
  `reference_code.data.datasets`.
- **Model architecture.** The reconstruction's flat `conv1/bn1` module
  names differ from the reference's `nn.Sequential` naming; the diagnostic
  now trains/evaluates the REAL reference SmallCNN, so the naming
  difference exists only in the audit baseline (recorded by the
  informational reconstruction-divergence entry in verification.json).
- **Aggregation / partitioner / set_seed.** Previously reconstructions;
  now the real bundled functions are used directly (aggregation via a
  device-preserving wrapper — tensor placement only, values unchanged).

