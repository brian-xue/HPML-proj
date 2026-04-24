#!/usr/bin/env bash
set -euo pipefail

# Lists the GPU experiment commands to run (DDP/FSDP scaling + profiling).
#
# Usage:
#   bash gpu_exp.sh
#   bash gpu_exp.sh 500

MAX_STEPS="${1:-500}"

cat <<EOF
DDP scaling (weak scaling; eval disabled):
  torchrun --standalone --nproc_per_node=1 experiments/ddp_scaling.py --max-steps ${MAX_STEPS}
  torchrun --standalone --nproc_per_node=2 experiments/ddp_scaling.py --max-steps ${MAX_STEPS}
  torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py --max-steps ${MAX_STEPS}

FSDP scaling (weak scaling; eval disabled):
  torchrun --standalone --nproc_per_node=1 experiments/fsdp_scaling.py --max-steps ${MAX_STEPS}
  torchrun --standalone --nproc_per_node=2 experiments/fsdp_scaling.py --max-steps ${MAX_STEPS}
  torchrun --standalone --nproc_per_node=4 experiments/fsdp_scaling.py --max-steps ${MAX_STEPS}

Profiling wrappers (plain + nsys + ncu if available):
  bash scripts/profile_ddp.sh 4
  bash scripts/profile_fsdp.sh 4

Resume (example):
  torchrun --standalone --nproc_per_node=4 experiments/ddp_scaling.py --mode resume --resume-version latest
EOF

