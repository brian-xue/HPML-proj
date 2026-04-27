#!/usr/bin/env bash
set +e

cd "$(dirname "$0")/.."

mkdir -p output

COMPLETED_LOG="output/experiment_matrix.completed.log"
FAILED_LOG="output/experiment_matrix.failed.log"

run_and_log() {
  label="$1"
  shift

  echo
  echo "[matrix] Running ${label}"
  "$@"
  status=$?
  echo "Completed ${label} exit status: ${status}"

  if [ "${status}" -eq 0 ]; then
    echo "Completed ${label} exit status: ${status}" >> "${COMPLETED_LOG}"
  else
    echo "Completed ${label} exit status: ${status}" >> "${FAILED_LOG}"
  fi
}

run_and_log "ddp_4gpu_profile" bash scripts/profile_ddp.sh 4 200
run_and_log "fsdp_4gpu_profile" bash scripts/profile_fsdp.sh 4 200

run_and_log "ddp_4gpu_memlen512" torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py --mode new --max-steps 200 --peft off --eval off --train-batch-size 4 --max-length 512 --run-label memlen512
run_and_log "ddp_4gpu_memlen1024" torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py --mode new --max-steps 200 --peft off --eval off --train-batch-size 4 --max-length 1024 --run-label memlen1024
run_and_log "fsdp_4gpu_memlen512" torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py --mode new --max-steps 200 --peft off --eval off --train-batch-size 4 --max-length 512 --run-label memlen512
run_and_log "fsdp_4gpu_memlen1024" torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py --mode new --max-steps 200 --peft off --eval off --train-batch-size 4 --max-length 1024 --run-label memlen1024

run_and_log "ddp_1gpu_scale_r1" torchrun --standalone --nproc_per_node=1 experiments/ddp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r1
run_and_log "ddp_2gpu_scale_r1" torchrun --standalone --nproc_per_node=2 experiments/ddp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r1
run_and_log "ddp_4gpu_scale_r1" torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r1
run_and_log "fsdp_1gpu_scale_r1" torchrun --standalone --nproc_per_node=1 experiments/fsdp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r1
run_and_log "fsdp_2gpu_scale_r1" torchrun --standalone --nproc_per_node=2 experiments/fsdp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r1
run_and_log "fsdp_4gpu_scale_r1" torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r1

run_and_log "ddp_1gpu_scale_r2" torchrun --standalone --nproc_per_node=1 experiments/ddp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r2
run_and_log "ddp_2gpu_scale_r2" torchrun --standalone --nproc_per_node=2 experiments/ddp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r2
run_and_log "ddp_4gpu_scale_r2" torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r2
run_and_log "fsdp_1gpu_scale_r2" torchrun --standalone --nproc_per_node=1 experiments/fsdp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r2
run_and_log "fsdp_2gpu_scale_r2" torchrun --standalone --nproc_per_node=2 experiments/fsdp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r2
run_and_log "fsdp_4gpu_scale_r2" torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py --mode new --max-steps 500 --peft off --eval off --run-label scale_r2

run_and_log "ddp_2gpu_tta" torchrun --standalone --nproc_per_node=2 experiments/ddp_scaling.py --mode new --max-steps 500 --peft off --eval on --eval-every-steps 100 --run-label tta
run_and_log "ddp_4gpu_tta" torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py --mode new --max-steps 500 --peft off --eval on --eval-every-steps 100 --run-label tta
run_and_log "fsdp_2gpu_tta" torchrun --standalone --nproc_per_node=2 experiments/fsdp_scaling.py --mode new --max-steps 500 --peft off --eval on --eval-every-steps 100 --run-label tta
run_and_log "fsdp_4gpu_tta" torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py --mode new --max-steps 500 --peft off --eval on --eval-every-steps 100 --run-label tta


