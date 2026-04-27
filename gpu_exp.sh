#!/usr/bin/env bash
set -euo pipefail

# Runs the full GPU experiment matrix. Safe to rerun:
# - regular scaling runs use explicit labels like trainonly / tta
# - profiling wrappers use separate labels like profile_plain / profile_nvprof
# - scaling entrypoints default to auto resume / skip-if-complete
# - profiling wrappers skip once their profiler outputs already exist

SCALING_MAX_STEPS="${SCALING_MAX_STEPS:-${1:-500}}"
PROFILE_NPROC="${PROFILE_NPROC:-4}"
PROFILE_MAX_STEPS="${PROFILE_MAX_STEPS:-200}"
SCALING_EVAL="${SCALING_EVAL:-off}"
SCALING_EVAL_EVERY_STEPS="${SCALING_EVAL_EVERY_STEPS:-100}"
SCALING_RUN_LABEL="${SCALING_RUN_LABEL:-}"

if [[ -z "${SCALING_RUN_LABEL}" ]]; then
  if [[ "${SCALING_EVAL}" == "on" ]]; then
    SCALING_RUN_LABEL="tta"
  else
    SCALING_RUN_LABEL="trainonly"
  fi
fi

run_cmd() {
  echo
  echo "[gpu_exp] $*"
  "$@"
}

run_cmd bash scripts/profile_ddp.sh "${PROFILE_NPROC}" "${PROFILE_MAX_STEPS}"
run_cmd bash scripts/profile_fsdp.sh "${PROFILE_NPROC}" "${PROFILE_MAX_STEPS}"

run_cmd torchrun --standalone --nproc_per_node=1 experiments/ddp_scaling.py --max-steps "${SCALING_MAX_STEPS}" --eval "${SCALING_EVAL}" --eval-every-steps "${SCALING_EVAL_EVERY_STEPS}" --run-label "${SCALING_RUN_LABEL}"
run_cmd torchrun --standalone --nproc_per_node=2 experiments/ddp_scaling.py --max-steps "${SCALING_MAX_STEPS}" --eval "${SCALING_EVAL}" --eval-every-steps "${SCALING_EVAL_EVERY_STEPS}" --run-label "${SCALING_RUN_LABEL}"
run_cmd torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py --max-steps "${SCALING_MAX_STEPS}" --eval "${SCALING_EVAL}" --eval-every-steps "${SCALING_EVAL_EVERY_STEPS}" --run-label "${SCALING_RUN_LABEL}"

run_cmd torchrun --standalone --nproc_per_node=1 experiments/fsdp_scaling.py --max-steps "${SCALING_MAX_STEPS}" --eval "${SCALING_EVAL}" --eval-every-steps "${SCALING_EVAL_EVERY_STEPS}" --run-label "${SCALING_RUN_LABEL}"
run_cmd torchrun --standalone --nproc_per_node=2 experiments/fsdp_scaling.py --max-steps "${SCALING_MAX_STEPS}" --eval "${SCALING_EVAL}" --eval-every-steps "${SCALING_EVAL_EVERY_STEPS}" --run-label "${SCALING_RUN_LABEL}"
run_cmd torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py --max-steps "${SCALING_MAX_STEPS}" --eval "${SCALING_EVAL}" --eval-every-steps "${SCALING_EVAL_EVERY_STEPS}" --run-label "${SCALING_RUN_LABEL}"
