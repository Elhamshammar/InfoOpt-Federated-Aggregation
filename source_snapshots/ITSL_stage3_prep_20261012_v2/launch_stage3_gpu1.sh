#!/bin/bash
# Stage-3 heterogeneity experiment launcher (six arms, two alpha regimes,
# 180 runs), GPU 1 only. Sequential. Full verifier gates every skip and
# every completion. No overwrite of existing run directories.
# LF line endings only.
#
# Usage:
#   PKG=… EXP_ROOT=… bash launch_stage3_gpu1.sh
set -o pipefail
set -u

PKG=${PKG:-$(dirname "$0")}
EXP_ROOT=${EXP_ROOT:-$(dirname "$0")}
PY=${PY:-python}

DATASETS=${DATASETS:-"cifar100 eurosat pathmnist"}
ALPHAS="0.1 1.0"
SEEDS="100 101 102 103 104"
ARMS="fedavg uniform infoopt shuffled_infoopt damping_050 damping_075"
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1
export EXP_ROOT
SHARED_DIR=${SHARED_DIR:-$EXP_ROOT/shared}   # ONE absolute shared location
export SHARED_DIR

if [ ! -x "$PY" ]; then echo "FATAL: python not found: $PY"; exit 127; fi
mkdir -p "$EXP_ROOT/runs" "$EXP_ROOT/logs" "$EXP_ROOT/verification"

echo "== phase -1: frozen integrity =="
"$PY" -u "$PKG/verify_frozen_integrity.py" || exit $?

echo "== phase 0a: dataset root resolution (absolute, existence-checked) =="
"$PY" -u "$PKG/resolve_dataset_roots.py" || exit $?

echo "== phase 0b: shared partition indices + cohort schedules =="
"$PY" -u "$PKG/generate_shared_schedules.py" \
    --configs-dir "$EXP_ROOT/configs_effective" \
    --shared-dir "$SHARED_DIR" || exit $?

echo "== phase -1b: frozen integrity (post-schedules) =="
"$PY" -u "$PKG/verify_frozen_integrity.py" --require-complete-schedules || exit $?

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "PREFLIGHT COMPLETE: integrity + dataset resolution + schedules OK; exiting before training."
  exit 0
fi

run_arm() {
  local ds=$1 alpha=$2 seed=$3 arm=$4
  local cfg="$EXP_ROOT/configs_effective/stage3_${ds}_${arm}_alpha${alpha}.yaml"
  local decision verdict
  decision=$("$PY" -u "$PKG/resume_check.py" \
      --dataset "$ds" --seed "$seed" --alpha "$alpha" --arm "$arm" \
      --runs-dir "$EXP_ROOT/runs" --schedules-dir "$SHARED_DIR" --python "$PY") \
      || return 2
  verdict=$(echo "$decision" | tail -1)
  case "$verdict" in
    skip_complete) echo "SKIP (verified complete): $ds alpha=$alpha $arm seed=$seed"; return 0 ;;
    abort_invalid) echo "ABORT: invalid existing run (verification failed): $ds alpha=$alpha $arm seed=$seed"; return 2 ;;
    abort_duplicate) echo "ABORT: duplicate run dirs for $ds alpha=$alpha $arm seed=$seed"; return 2 ;;
    run) ;;
    *) echo "ABORT: resume_check failed"; return 2 ;;
  esac

  local extra=""
  case "$arm" in
    uniform)          extra="" ;;                      # uniform_agg in config
    infoopt)          extra="--aib-override false" ;;
    shuffled_infoopt) extra="--aib-override false --score-shuffle stage1" ;;
    damping_050)      extra="--gamma 0.5" ;;
    damping_075)      extra="--gamma 0.75" ;;
    fedavg)           extra="" ;;
  esac

  local method
  case "$arm" in
    fedavg|uniform|damping_*) method="fedavg" ;;
    infoopt|shuffled_infoopt) method="InfoOpt-Full" ;;
  esac

  local log="$EXP_ROOT/logs/${ds}_${arm}_alpha${alpha}_seed${seed}_$(date +%Y%m%d_%H%M%S)_$$.log"
  echo "RUN $ds alpha=$alpha $arm seed=$seed (log: $log)"
  ( cd "$EXP_ROOT" && "$PY" -u "$PKG/snapshot/train.py" \
      --config "$cfg" --method "$method" --seed "$seed" $extra \
      > "$log" 2>&1 )
  local status=$?
  if [ $status -ne 0 ]; then
    echo "FAILED ($status): $ds alpha=$alpha $arm seed=$seed -- see $log"
    return $status
  fi

  decision=$("$PY" -u "$PKG/resume_check.py" \
      --dataset "$ds" --seed "$seed" --alpha "$alpha" --arm "$arm" \
      --runs-dir "$EXP_ROOT/runs" --schedules-dir "$SHARED_DIR" --python "$PY") \
      || return 2
  verdict=$(echo "$decision" | tail -1)
  if [ "$verdict" != "skip_complete" ]; then
    echo "ABORT: post-training verification did not pass: $ds alpha=$alpha $arm seed=$seed"
    return 2
  fi
  return 0
}

echo "== phase 1: training (sequential) =="
fail=0
for ds in $DATASETS; do
  for alpha in $ALPHAS; do
    for seed in $SEEDS; do
      for arm in $ARMS; do
        run_arm "$ds" "$alpha" "$seed" "$arm" || { fail=1; break 4; }
      done
    done
  done
done

if [ $fail -eq 0 ]; then
  echo "ALL 180 CELLS COMPLETE (trained or verified-skip)"
  echo "== phase 2: execution manifest =="
  "$PY" -u "$PKG/build_run_manifest.py" || exit $?
else
  echo "STOPPED on failure; fix and re-run (verified-complete cells will be skipped)"
fi
exit $fail
