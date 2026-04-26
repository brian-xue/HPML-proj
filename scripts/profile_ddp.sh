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
NSYS_PREFIX="output/nsys_ddp_${NPROC}gpu"
NVPROF_PREFIX="output/nvprof_ddp_${NPROC}gpu"
NCU_PREFIX="output/ncu_ddp_${NPROC}gpu"

echo "[ddp] plain run: nproc=${NPROC} max_steps=${MAX_STEPS}"
torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${MAX_STEPS}" --run-label plain

if command -v nsys >/dev/null 2>&1; then
  if ls "${NSYS_PREFIX}"* >/dev/null 2>&1; then
    echo "[ddp] nsys outputs already exist for nproc=${NPROC}; skipping."
  else
    echo "[ddp] nsys run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
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
    echo "[ddp] nvprof outputs already exist for nproc=${NPROC}; skipping."
  else
    echo "[ddp] nvprof run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
    HPML_PROFILE_CUDA=1 \
    HPML_PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
    HPML_PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS}" \
    nvprof \
      --profile-from-start off \
      --log-file "${NVPROF_PREFIX}.log" \
      torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${PROFILE_MAX_STEPS}" --run-label nvprof
  fi
else
  echo "[ddp] neither nsys nor nvprof found; skipping system profiling." >&2
fi

if command -v ncu >/dev/null 2>&1; then
  if ls "${NCU_PREFIX}"* >/dev/null 2>&1; then
    echo "[ddp] ncu outputs already exist for nproc=${NPROC}; skipping."
  else
    echo "[ddp] ncu run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
    HPML_PROFILE_CUDA=1 \
    HPML_PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
    HPML_PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS}" \
    ncu --target-processes all --replay-mode range -o "${NCU_PREFIX}" \
      torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${PROFILE_MAX_STEPS}" --run-label ncu
  fi
else
  echo "[ddp] ncu not found; skipping Nsight Compute." >&2
fi
