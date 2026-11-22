# Synthetic implementation checks — 10 October 2026

Runtime tests NOT EXECUTED: preparation environment has no PyTorch. Syntax and unique extraction of the exact recovered SCAFFOLD server-control block validated. Run check_equations.py using your existing ITSL Python environment on CPU. No real datasets, checkpoints, full training, tuning, or manuscript edits. Tiny synthetic local SGD steps are used solely for equation tests.

Source hashes checked before imports. Tests cover actual recovered local functions, analytical softmax gradients, Option-II control updates, server participation scaling, FedDyn regularizer signs, selected/unselected dual persistence and all-client bias across rounds. Successful completion creates equation_check_results.json exclusively; an existing report aborts. Failures never produce a PASS report.

Aggregation utility tests distinguish sample-weighted and uniform averages, without executing full training orchestration. Nonzero clipping/weight decay, BN buffers and CUDA behavior are not covered. PASS would not certify historical source versions or baseline comparisons.

Live phase2_provenance.txt is byte-identical to recovered evidence (a239b9eb4bc436e6d38c7e0fe122bbdd1433ae2fdefb4f072ac30aa4a2d09e61). Launch-time repaired-function checks plus file sizes/timestamps support repaired code presence, but no executed source hash was recorded. Primary eligibility remains unchanged.
