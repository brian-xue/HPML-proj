#!/usr/bin/env bash
set -euo pipefail

# Runs the full GPU experiment matrix. Safe to rerun:
# - scaling entrypoints default to auto resume / skip-if-complete
# - profiling wrappers skip once their report files already exist

SCALING_MAX_STEPS="${SCALING_MAX_STEPS:-${1:-500}}"
PROFILE_NPROC="${PROFILE_NPROC:-4}"
PROFILE_MAX_STEPS="${PROFILE_MAX_STEPS:-200}"

run_cmd() {
  echo
  echo "[gpu_exp] $*"
  "$@"
}

run_cmd bash scripts/profile_ddp.sh "${PROFILE_NPROC}" "${PROFILE_MAX_STEPS}"
run_cmd bash scripts/profile_fsdp.sh "${PROFILE_NPROC}" "${PROFILE_MAX_STEPS}"

run_cmd torchrun --standalone --nproc_per_node=1 experiments/ddp_scaling.py --max-steps "${SCALING_MAX_STEPS}"
run_cmd torchrun --standalone --nproc_per_node=2 experiments/ddp_scaling.py --max-steps "${SCALING_MAX_STEPS}"
run_cmd torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py --max-steps "${SCALING_MAX_STEPS}"

run_cmd torchrun --standalone --nproc_per_node=1 experiments/fsdp_scaling.py --max-steps "${SCALING_MAX_STEPS}"
run_cmd torchrun --standalone --nproc_per_node=2 experiments/fsdp_scaling.py --max-steps "${SCALING_MAX_STEPS}"
run_cmd torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py --max-steps "${SCALING_MAX_STEPS}"
