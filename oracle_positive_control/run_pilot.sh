#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
GPU_ID="${1:-0}"
for SEED in 100; do
  for METHOD in fedavg oracle_nojs_aligned oracle_nojs_shuffled; do
    python src/train_mlj_p1.py --config "configs/mlj_p1_${METHOD}_alpha0.1.yaml" --method "$METHOD" --seed "$SEED" --rounds 100 --alpha 0.1 --gpu "$GPU_ID"
  done
done
