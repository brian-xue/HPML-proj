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

echo "[ddp] plain run: nproc=${NPROC} max_steps=${MAX_STEPS}"
torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${MAX_STEPS}" --run-label plain

if command -v nsys >/dev/null 2>&1; then
  echo "[ddp] nsys run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
  HPML_PROFILE_CUDA=1 \
  HPML_PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
  HPML_PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS}" \
  nsys profile \
    -o "output/nsys_ddp_${NPROC}gpu" \
    --trace=cuda,nvtx,osrt,nccl,cublas,cudnn \
    --capture-range=cudaProfilerApi \
    --stop-on-range-end=true \
    torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${PROFILE_MAX_STEPS}" --run-label nsys
else
  echo "[ddp] nsys not found; skipping Nsight Systems." >&2
fi

if command -v ncu >/dev/null 2>&1; then
  echo "[ddp] ncu run (warmup=${PROFILE_WARMUP_STEPS} active=${PROFILE_ACTIVE_STEPS})"
  HPML_PROFILE_CUDA=1 \
  HPML_PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
  HPML_PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS}" \
  ncu --target-processes all --profile-from-start off -o "output/ncu_ddp_${NPROC}gpu" \
    torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${PROFILE_MAX_STEPS}" --run-label ncu
else
  echo "[ddp] ncu not found; skipping Nsight Compute." >&2
fi
