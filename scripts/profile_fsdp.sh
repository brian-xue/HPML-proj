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

EXP="experiments/fsdp_scaling.py"
NSYS_PREFIX="output/nsys_fsdp_${NPROC}gpu"
NVPROF_PREFIX="output/nvprof_fsdp_${NPROC}gpu"
NCU_PREFIX="output/ncu_fsdp_${NPROC}gpu"

echo "[fsdp] plain run: nproc=${NPROC} max_steps=${MAX_STEPS}"
torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${MAX_STEPS}" --run-label plain

if command -v nsys >/dev/null 2>&1; then
  if ls "${NSYS_PREFIX}"* >/dev/null 2>&1; then
    echo "[fsdp] nsys outputs already exist for nproc=${NPROC}; skipping."
  else
    echo "[fsdp] nsys run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
    HPML_PROFILE_CUDA=1 \
    HPML_PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
    HPML_PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS}" \
    nsys profile \
      -o "${NSYS_PREFIX}" \
      --trace=cuda,nvtx,osrt,nccl,cublas,cudnn \
      --capture-range=cudaProfilerApi \
      --stop-on-range-end=true \
      torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${PROFILE_MAX_STEPS}" --run-label nsys
  fi
elif command -v nvprof >/dev/null 2>&1; then
  if ls "${NVPROF_PREFIX}"* >/dev/null 2>&1; then
    echo "[fsdp] nvprof outputs already exist for nproc=${NPROC}; skipping."
  else
    echo "[fsdp] nvprof run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
    HPML_PROFILE_CUDA=1 \
    HPML_PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
    HPML_PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS}" \
    nvprof \
      --profile-from-start off \
      --log-file "${NVPROF_PREFIX}.log" \
      torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${PROFILE_MAX_STEPS}" --run-label nvprof
  fi
else
  echo "[fsdp] neither nsys nor nvprof found; skipping system profiling." >&2
fi

if command -v ncu >/dev/null 2>&1; then
  if ls "${NCU_PREFIX}"* >/dev/null 2>&1; then
    echo "[fsdp] ncu outputs already exist for nproc=${NPROC}; skipping."
  else
    echo "[fsdp] ncu run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
    HPML_PROFILE_CUDA=1 \
    HPML_PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
    HPML_PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS}" \
    ncu --target-processes all --replay-mode range -o "${NCU_PREFIX}" \
      torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${PROFILE_MAX_STEPS}" --run-label ncu
  fi
else
  echo "[fsdp] ncu not found; skipping Nsight Compute." >&2
fi
