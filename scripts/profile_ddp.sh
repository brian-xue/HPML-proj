#!/usr/bin/env bash
set -euo pipefail

NPROC="${1:-4}"
MAX_STEPS="${2:-200}"

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found. Activate your PyTorch env first." >&2
  exit 1
fi

EXP="experiments/ddp_scaling.py"

echo "[ddp] plain run: nproc=${NPROC} max_steps=${MAX_STEPS}"
torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps "${MAX_STEPS}" --run-label plain

if command -v nsys >/dev/null 2>&1; then
  echo "[ddp] nsys run (short)"
  nsys profile \
    -o "output/nsys_ddp_${NPROC}gpu" \
    --trace=cuda,nvtx,osrt,nccl,cublas,cudnn \
    torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps 50 --run-label nsys
else
  echo "[ddp] nsys not found; skipping Nsight Systems." >&2
fi

if command -v ncu >/dev/null 2>&1; then
  echo "[ddp] ncu run (very short)"
  ncu --target-processes all -o "output/ncu_ddp_${NPROC}gpu" \
    torchrun --standalone --nproc_per_node="${NPROC}" "${EXP}" --max-steps 20 --run-label ncu
else
  echo "[ddp] ncu not found; skipping Nsight Compute." >&2
fi

