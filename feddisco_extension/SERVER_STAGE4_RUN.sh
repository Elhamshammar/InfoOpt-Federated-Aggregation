#!/usr/bin/env bash
# SERVER_STAGE4_RUN.sh
# Bounded full five-seed, 100-round FedDisco experiment launcher.
# Run on the Linux GPU server inside the Stage 4 project directory.
#
# Usage:
#   export GPU_INDEX=0          # 0 or 1
#   export SEEDS="100 101 102 103 104"
#   bash SERVER_STAGE4_RUN.sh
#
# Optional:
#   export DRY_RUN=1            # print commands without executing
#   export ROUNDS_OVERRIDE=2    # smoke-test override (passed to train.py)

set -euo pipefail

GPU_INDEX="${GPU_INDEX:-0}"
SEEDS="${SEEDS:-100 101 102 103 104}"
DRY_RUN="${DRY_RUN:-0}"
ROUNDS_OVERRIDE="${ROUNDS_OVERRIDE:-}"
CONDA_SKIP="${CONDA_SKIP:-0}"

STAGE3_DIR="${STAGE3_DIR:-../InfoOpt_FedDisco_Stage3}"
STAGE4_DIR="${STAGE4_DIR:-.}"
CONDA_ENV="${CONDA_ENV:-infoopt_dl}"

ARMS=(fedavg uniform feddisco feddisco_shuffled)
METHODS=(fedavg fedavg feddisco feddisco_shuffled)
# Note: uniform arm uses --method fedavg with training.uniform_agg=true in config.
ALPHA=0.3
EXPECTED_ROUNDS=100

LOG_FILE="STAGE4_RUN.log"

echo "=== FedDisco Stage 4 full experiment ===" | tee -a "${LOG_FILE}"
echo "GPU_INDEX=${GPU_INDEX}" | tee -a "${LOG_FILE}"
echo "SEEDS=${SEEDS}" | tee -a "${LOG_FILE}"
echo "STAGE3_DIR=${STAGE3_DIR}" | tee -a "${LOG_FILE}"
echo "STAGE4_DIR=${STAGE4_DIR}" | tee -a "${LOG_FILE}"
echo "START_TIME=$(date -Iseconds)" | tee -a "${LOG_FILE}"

# Activate Python environment.
if [[ "${CONDA_SKIP}" == "1" ]]; then
    echo "CONDA_SKIP=1: skipping conda activation" | tee -a "${LOG_FILE}"
elif [[ "${DRY_RUN}" != "1" ]]; then
    # Ensure conda is available for activation inside a non-login shell.
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate "${CONDA_ENV}"
    python - <<'PY'
import sys, torch
print(f"python={sys.version}")
print(f"pytorch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_devices={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}={torch.cuda.get_device_name(i)}")
PY
fi

# Prepare Stage 4 directory and reusable source from Stage 3.
echo "=== Preparing Stage 4 source tree ===" | tee -a "${LOG_FILE}"
mkdir -p "${STAGE4_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "(dry-run) rsync -av --exclude=runs --exclude=__pycache__ --exclude='*.pyc' "${STAGE3_DIR}/" "${STAGE4_DIR}/""
else
    rsync -av --exclude=runs --exclude=__pycache__ --exclude='*.pyc' "${STAGE3_DIR}/" "${STAGE4_DIR}/"
fi

cd "${STAGE4_DIR}"

# Ensure the Stage 4 configs are present in the expected location.
# They are transferred from Windows into the project root configs/ directory.
for cfg in configs/feddisco_stage4_cifar100_{fedavg,uniform,feddisco,feddisco_shuffled}_alpha0.3.yaml; do
    if [[ ! -f "${cfg}" ]]; then
        echo "ERROR: missing config ${cfg}" | tee -a "${LOG_FILE}"
        exit 1
    fi
done

# Map an arm name to the method slug used in the run directory.
_method_slug_for_arm() {
    local arm="$1"
    case "${arm}" in
        fedavg) echo "fedavg" ;;
        uniform) echo "fedavg" ;;
        feddisco) echo "feddisco" ;;
        feddisco_shuffled) echo "feddisco_shuffled" ;;
        *) echo "${arm}" ;;
    esac
}

# Helper: determine whether a verified 100-round run already exists.
_run_exists() {
    local arm="$1"
    local seed="$2"
    local method_slug
    method_slug=$(_method_slug_for_arm "${arm}")
    local pattern="runs/feddisco_stage4_cifar100_${arm}_${method_slug}_seed${seed}_alpha${ALPHA}_*"
    for d in ${pattern}; do
        if [[ -d "${d}" ]]; then
            local metrics="${d}/metrics.csv"
            local verified="${d}/.verified"
            if [[ -f "${metrics}" ]]; then
                local rounds
                rounds=$(tail -n +2 "${metrics}" | wc -l)
                if [[ "${rounds}" -ge "${EXPECTED_ROUNDS}" && -f "${verified}" ]]; then
                    echo "${d}"
                    return 0
                fi
            fi
        fi
    done
    return 1
}

# Main execution loop.
fail_count=0
completed_count=0
skipped_count=0

for seed in ${SEEDS}; do
    for idx in "${!ARMS[@]}"; do
        arm="${ARMS[$idx]}"
        method="${METHODS[$idx]}"
        cfg="configs/feddisco_stage4_cifar100_${arm}_alpha0.3.yaml"

        echo "---" | tee -a "${LOG_FILE}"
        echo "seed=${seed} arm=${arm} method=${method}" | tee -a "${LOG_FILE}"

        existing=$(_run_exists "${arm}" "${seed}" || true)
        if [[ -n "${existing}" ]]; then
            echo "SKIP: verified run exists: ${existing}" | tee -a "${LOG_FILE}"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        cmd=(python snapshot/train.py --config "${cfg}" --method "${method}" --seed "${seed}" --gpu "${GPU_INDEX}")
        if [[ -n "${ROUNDS_OVERRIDE}" ]]; then
            cmd+=(--rounds "${ROUNDS_OVERRIDE}")
        fi

        echo "RUN: ${cmd[*]}" | tee -a "${LOG_FILE}"
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "(dry-run) would execute above command"
            continue
        fi

        set +e
        "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
        exit_code=${PIPESTATUS[0]}
        set -e

        if [[ "${exit_code}" -ne 0 ]]; then
            echo "ERROR: command failed with exit code ${exit_code}" | tee -a "${LOG_FILE}"
            fail_count=$((fail_count + 1))
            # Stop on first failure to avoid hiding problems.
            echo "STOPPING due to failure." | tee -a "${LOG_FILE}"
            exit "${exit_code}"
        fi
        completed_count=$((completed_count + 1))
        echo "COMPLETED: seed=${seed} arm=${arm} method=${method}" | tee -a "${LOG_FILE}"
    done
done

echo "=== Stage 4 run summary ===" | tee -a "${LOG_FILE}"
echo "completed=${completed_count} skipped=${skipped_count} failed=${fail_count}" | tee -a "${LOG_FILE}"
echo "END_TIME=$(date -Iseconds)" | tee -a "${LOG_FILE}"

if [[ "${fail_count}" -gt 0 ]]; then
    exit 1
fi
