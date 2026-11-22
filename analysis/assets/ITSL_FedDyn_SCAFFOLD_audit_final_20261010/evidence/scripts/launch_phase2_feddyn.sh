#!/usr/bin/env bash
# Phase 2: FedDyn canonical reruns — SAFE (GPU-isolated, failure-logged)
# Usage: ./launch_phase2_feddyn.sh [start_gpu=0] [num_gpus=1]
# Runs 15 experiments sequentially on the assigned GPU(s).
# Continues after individual failures; writes PASS/FAIL per run.

set -uo pipefail

START_GPU=${1:-0}
NUM_GPUS=${2:-1}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infoopt_dl

echo "=== Phase 2: FedDyn repair campaign ==="
echo "GPU range: $START_GPU to $((START_GPU + NUM_GPUS - 1))"
echo "Total runs: 15 (3 datasets x 5 seeds)"
echo ""

mkdir -p runs/_logs

SEEDS=(100 101 102 103 104)
declare -A CONFIG=(
    [cifar100]=configs/sweep_v2.yaml
    [pathmnist]=configs/pathmnist_sweep.yaml
    [eurosat]=configs/eurosat_sweep.yaml
)

TOTAL=15
PASS=0
FAIL=0
count=0

for dataset in cifar100 pathmnist eurosat; do
    for seed in "${SEEDS[@]}"; do
        count=$((count + 1))
        gpu_id=$((START_GPU + (count - 1) % NUM_GPUS))
        logfile="runs/_logs/phase2_feddyn_run_${count}.log"
        cmd="python train.py --config ${CONFIG[$dataset]} --method feddyn --seed $seed --alpha 0.3"
        echo "[$count/$TOTAL] GPU $gpu_id: $cmd"
        if CUDA_VISIBLE_DEVICES=$gpu_id $cmd > "$logfile" 2>&1; then
            echo "PASS feddyn $dataset seed=$seed" | tee -a runs/_logs/phase2_status.log
            PASS=$((PASS + 1))
        else
            echo "FAIL feddyn $dataset seed=$seed (exit code $?)" | tee -a runs/_logs/phase2_status.log
            FAIL=$((FAIL + 1))
        fi
    done
done

echo ""
echo "=== FedDyn complete: PASS=$PASS FAIL=$FAIL ==="
echo "Status log: runs/_logs/phase2_status.log"
