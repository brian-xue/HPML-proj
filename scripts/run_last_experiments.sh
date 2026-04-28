#!/usr/bin/env bash
set -u

run_and_log() {
  local label="$1"
  shift
  echo "[last] Running ${label}"
  "$@"
  local status=$?
  echo "Completed ${label} exit status: ${status}"
}

cd /Users/sanchitsahay/code/hpml/HPML-proj || exit 1

run_and_log ddp_4gpu_memlen1536 \
  torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py \
  --mode new \
  --max-steps 200 \
  --peft off \
  --max-length 1536 \
  --run-label memlen1536

run_and_log fsdp_4gpu_memlen1536 \
  torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py \
  --mode new \
  --max-steps 200 \
  --peft off \
  --max-length 1536 \
  --run-label memlen1536

run_and_log ddp_4gpu_peft_memlen1536 \
  torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py \
  --mode new \
  --max-steps 200 \
  --peft on \
  --max-length 1536 \
  --run-label peft_memlen1536

run_and_log fsdp_4gpu_peft_memlen1536 \
  torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py \
  --mode new \
  --max-steps 200 \
  --peft on \
  --max-length 1536 \
  --run-label peft_memlen1536

run_and_log ddp_4gpu_peft_profile_nsys \
  env HPML_PROFILE_CUDA=1 HPML_PROFILE_WARMUP_STEPS=10 HPML_PROFILE_ACTIVE_STEPS=100 \
  nsys profile \
  -o output/nsys_ddp_4gpu_peft \
  --trace=cuda,nvtx,osrt,nccl,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py \
  --mode new \
  --max-steps 112 \
  --peft on \
  --run-label peft_profile_nsys

run_and_log fsdp_4gpu_peft_profile_nsys \
  env HPML_PROFILE_CUDA=1 HPML_PROFILE_WARMUP_STEPS=10 HPML_PROFILE_ACTIVE_STEPS=100 \
  nsys profile \
  -o output/nsys_fsdp_4gpu_peft \
  --trace=cuda,nvtx,osrt,nccl,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py \
  --mode new \
  --max-steps 112 \
  --peft on \
  --run-label peft_profile_nsys

run_and_log ddp_4gpu_tta_final \
  torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py \
  --mode new \
  --max-steps 500 \
  --peft off \
  --eval on \
  --eval-every-steps 100 \
  --run-label tta_final

run_and_log fsdp_4gpu_tta_final \
  torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py \
  --mode new \
  --max-steps 500 \
  --peft off \
  --eval on \
  --eval-every-steps 100 \
  --run-label tta_final
