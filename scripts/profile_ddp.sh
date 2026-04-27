#!/usr/bin/env bash
set -euo pipefail

NPROC="${1:-4}"
MAX_STEPS="${2:-200}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-10}"
PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-10}"
PROFILE_MAX_STEPS="${PROFILE_MAX_STEPS:-$((PROFILE_WARMUP_STEPS + PROFILE_ACTIVE_STEPS + 2))}"

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found. Activate your PyTorch env first." >&2
  exit 1
fi

EXP="experiments/ddp_scaling.py"
NVPROF_PREFIX="output/nvprof_ddp_${NPROC}gpu"
NCU_PREFIX="output/ncu_ddp_${NPROC}gpu"
PLAIN_LABEL="profile_plain"
NVPROF_LABEL="profile_nvprof"
NCU_LABEL="profile_ncu_apprange"

echo "[ddp] plain run: nproc=${NPROC} max_steps=${MAX_STEPS}"
torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${MAX_STEPS}" --run-label "${PLAIN_LABEL}"

if command -v nvprof >/dev/null 2>&1; then
  if ls "${NVPROF_PREFIX}"* >/dev/null 2>&1; then
    echo "[ddp] nvprof outputs already exist for nproc=${NPROC}; skipping."
  else
    echo "[ddp] nvprof run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
    HPML_PROFILE_CUDA=1 \
    HPML_PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
    HPML_PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS}" \
    nvprof \
      --profile-from-start off \
      --log-file "${NVPROF_PREFIX}.log" \
      torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${PROFILE_MAX_STEPS}" --run-label "${NVPROF_LABEL}"
  fi
else
  echo "[ddp] nvprof not found; skipping profiler run." >&2
fi

if command -v ncu >/dev/null 2>&1; then
  if ls "${NCU_PREFIX}"* >/dev/null 2>&1; then
    echo "[ddp] ncu outputs already exist for nproc=${NPROC}; skipping."
  else
    echo "[ddp] ncu run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
    HPML_PROFILE_CUDA=1 \
    HPML_PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
    HPML_PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS}" \
    ncu --target-processes all --replay-mode app-range -o "${NCU_PREFIX}" \
      torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${PROFILE_MAX_STEPS}" --run-label "${NCU_LABEL}"
  fi
else
  echo "[ddp] ncu not found; skipping Nsight Compute." >&2
fi
