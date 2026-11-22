#!/bin/bash
# Four-arm experiment launcher (FedAvg / InfoOpt-Full x AIB off/on) v2.
# Physical GPU 1 only. Sequential. Full verifier gates every skip and every
# completion. No overwrite of existing run directories.
#
# Usage:
#   PKG=… EXP_ROOT=… bash launch_four_arm_gpu1.sh
set -o pipefail
set -u

PKG=${PKG:-$(dirname "$0")}
EXP_ROOT=${EXP_ROOT:-$(dirname "$0")}
PY=${PY:-python}

DATASETS=${DATASETS:-"cifar100 eurosat pathmnist"}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
SEEDS="100 101 102 103 104"
ARMS="fedavg_aiboff fedavg_aibon infoopt_aiboff infoopt_aibon"

export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1
export EXP_ROOT                      # consistent for all subprocesses
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

method_of() {
  case "$1" in
    fedavg_*)   echo "fedavg" ;;
    infoopt_*)  echo "InfoOpt-Full" ;;
  esac
}

run_arm() {
  local ds=$1 seed=$2 arm=$3
  local cfg="$EXP_ROOT/configs_effective/fourarm_${ds}_${arm}.yaml"
  local decision
  decision=$("$PY" -u "$PKG/resume_check.py" \
      --dataset "$ds" --seed "$seed" --arm "$arm" \
      --runs-dir "$EXP_ROOT/runs" --schedules-dir "$SHARED_DIR" --python "$PY") \
      || return 2
  local verdict
  verdict=$(echo "$decision" | tail -1)
  case "$verdict" in
    skip_complete) echo "SKIP (verified complete): $ds $arm seed=$seed"; return 0 ;;
    abort_invalid) echo "ABORT: invalid existing run (verification failed): $ds $arm seed=$seed"; return 2 ;;
    abort_duplicate) echo "ABORT: duplicate run dirs for $ds $arm seed=$seed"; return 2 ;;
    run) ;;
    *) echo "ABORT: resume_check failed"; return 2 ;;
  esac

  local extra=""
  case "$arm" in
    fedavg_aibon)   extra="--aib-override true" ;;
    infoopt_aiboff) extra="--aib-override false" ;;
    *) extra="" ;;
  esac

  local log="$EXP_ROOT/logs/${ds}_${arm}_seed${seed}_$(date +%Y%m%d_%H%M%S)_$$.log"
  echo "RUN $ds $arm seed=$seed (log: $log)"
  ( cd "$EXP_ROOT" && "$PY" -u "$PKG/snapshot/train.py" \
      --config "$cfg" --method "$(method_of "$arm")" --seed "$seed" $extra \
      > "$log" 2>&1 )
  local status=$?
  if [ $status -ne 0 ]; then
    echo "FAILED ($status): $ds $arm seed=$seed -- see $log"
    return $status
  fi

  # completion is recognized ONLY through full verification
  decision=$("$PY" -u "$PKG/resume_check.py" \
      --dataset "$ds" --seed "$seed" --arm "$arm" \
      --runs-dir "$EXP_ROOT/runs" --schedules-dir "$SHARED_DIR" --python "$PY") \
      || return 2
  verdict=$(echo "$decision" | tail -1)
  if [ "$verdict" != "skip_complete" ]; then
    echo "ABORT: post-training verification did not pass: $ds $arm seed=$seed"
    return 2
  fi
  return 0
}

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "PREFLIGHT COMPLETE: integrity + dataset resolution + schedules OK; exiting before training."
  exit 0
fi

echo "== phase 1: training (sequential) =="
fail=0
for ds in $DATASETS; do
  for seed in $SEEDS; do
    for arm in $ARMS; do
      run_arm "$ds" "$seed" "$arm" || { fail=1; break 3; }
    done
  done
done

if [ $fail -eq 0 ]; then
  echo "ALL 60 CELLS COMPLETE (trained or verified-skip)"
  echo "== phase 2: execution manifest =="
  "$PY" -u "$PKG/build_run_manifest.py" || exit $?
else
  echo "STOPPED on failure; fix and re-run (verified-complete cells will be skipped)"
fi
exit $fail
