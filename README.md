# Controlled score reassignment in federated aggregation

Research code and retained evidence for the manuscript **Evaluating Client-Specific Score Attribution in Federated Aggregation: A Controlled Score-Reassignment Methodology** (submitted version in preparation).

The study evaluates client-score assignment under matched aggregation procedures, rather than proposing a universally superior aggregator. The manuscript reports 530 runs: 480 original controlled runs, 15 corrected reliability runs, 20 FedDisco runs and 15 oracle positive-control runs.

## Repository organization

- `source_snapshots/`, `configs/`: existing training snapshots and configurations for the original studies and corrected reliability follow-up.
- `feddisco_extension/`: existing training implementation, configurations and 20 run directories.
- `analysis/`, `experiments/`: historical analysis and evidence. Some scripts use original machine-specific paths; historical results are not the authoritative source for the updated equivalence analysis.
- `oracle_positive_control/`: recovered training source and retained evidence for the 15 oracle runs.
- `reproduction/`: current CPU-only statistical verification, retained oracle results, corrected score-attribution summaries and figure/table generators.

## Reproduce the updated statistical analyses

From the repository root, run:

```bash
python -m pip install -r reproduction/requirements.txt
python reproduction/scripts_reports/mlj_r4_statistics.py --seed-csv reproduction/outputs/MLJ_A1_1/seed_level_effects.csv --oracle-csv reproduction/outputs/positive_control_per_seed_accuracy.csv --out reproduction/outputs/MLJ_R4
python reproduction/generate_all_tables.py
python reproduction/generate_all_figures.py
```

The statistical script checks paired contrasts, corrected TOST, exact noncentral-t detectable effects, FedDisco equivalence and oracle contrasts. The equivalence and detectable-effect analyses are retrospective and nominal. Figure and table generators cover the updated analyses, not every historical manuscript display. The generated forest figure uses standardized dataset labels.

## Training and limitations

For existing training snapshots, install root `requirements.txt` and consult the corresponding snapshot instructions and configurations. Set `DATASETS_ROOT` or configure `dataset.root` to local datasets. Raw datasets and large checkpoints are not included. The recovered oracle training implementation, launch configurations, corruption manifests and 15 run records are in `oracle_positive_control/`. See its README for the portable launcher. Full GPU retraining has not been verified in this release workspace. Historical provenance gaps remain as documented in the original repository. The old manuscript-number checker targets an earlier manuscript and is not an R4.2 verification command.

## License

See `LICENSE`. Licensing and redistribution permission must be confirmed by the authors before public release. No new license is assigned by this update.
