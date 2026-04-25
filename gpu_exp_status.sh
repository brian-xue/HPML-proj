#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-output}"

json_get() {
  local file="$1"
  local jq_expr="$2"
  local grep_pattern="$3"
  if [[ ! -f "$file" ]]; then
    return 1
  fi
  if command -v jq >/dev/null 2>&1; then
    jq -r "$jq_expr // empty" "$file" 2>/dev/null
  else
    grep -Eo "$grep_pattern" "$file" | head -n1 | sed -E 's/.*: *"?([^",}]+)"?/\1/'
  fi
}

latest_version_dir() {
  local exp_root="$1"
  if [[ ! -d "$exp_root" ]]; then
    return 1
  fi
  find "$exp_root" -maxdepth 1 -mindepth 1 -type d -name 'v*' | sort | tail -n1
}

report_run() {
  local label="$1"
  local exp_name="$2"
  local exp_root="${OUTPUT_ROOT}/${exp_name}"
  local latest_dir

  latest_dir="$(latest_version_dir "$exp_root" 2>/dev/null || true)"
  if [[ -z "${latest_dir}" ]]; then
    printf '%-20s missing\n' "${label}"
    return
  fi

  local final_json="${latest_dir}/final_results.json"
  local ckpt_ptr="${latest_dir}/checkpoints/latest_checkpoint.json"
  local status="partial"
  local steps=""
  local global_step=""

  if [[ -f "$final_json" ]]; then
    status="completed"
    steps="$(json_get "$final_json" '.runtime.total_steps' '"total_steps"[[:space:]]*:[[:space:]]*[0-9]+' || true)"
    global_step="$(json_get "$final_json" '.state.global_step' '"global_step"[[:space:]]*:[[:space:]]*[0-9]+' || true)"
  elif [[ -f "$ckpt_ptr" ]]; then
    local ckpt_dir
    ckpt_dir="$(json_get "$ckpt_ptr" '.checkpoint_dir' '"checkpoint_dir"[[:space:]]*:[[:space:]]*"[^"]+"' || true)"
    if [[ -n "$ckpt_dir" ]]; then
      local meta_json="${ckpt_dir}/metadata.json"
      if [[ ! -f "$meta_json" ]]; then
        meta_json="$(find "$ckpt_dir" -maxdepth 1 -type f -name 'metadata*.json' | sort | head -n1)"
      fi
      if [[ -f "$meta_json" ]]; then
        global_step="$(json_get "$meta_json" '.global_step' '"global_step"[[:space:]]*:[[:space:]]*[0-9]+' || true)"
      fi
    fi
  else
    status="created"
  fi

  printf '%-20s %-10s latest=%s' "${label}" "${status}" "$(basename "$latest_dir")"
  if [[ -n "$global_step" ]]; then
    printf ' global_step=%s' "$global_step"
  fi
  if [[ -n "$steps" ]]; then
    printf ' total_steps=%s' "$steps"
  fi
  printf '\n'
}

report_profile() {
  local label="$1"
  local prefix="$2"
  if ls "${OUTPUT_ROOT}/${prefix}"* >/dev/null 2>&1; then
    printf '%-20s present\n' "${label}"
  else
    printf '%-20s missing\n' "${label}"
  fi
}

echo "GPU experiment status from ${OUTPUT_ROOT}"
echo
report_profile "profile_ddp_nsys" "nsys_ddp_4gpu"
report_profile "profile_ddp_ncu" "ncu_ddp_4gpu"
report_profile "profile_fsdp_nsys" "nsys_fsdp_4gpu"
report_profile "profile_fsdp_ncu" "ncu_fsdp_4gpu"
echo
report_run "ddp_1gpu" "ddp_scaling_1gpu"
report_run "ddp_2gpu" "ddp_scaling_2gpu"
report_run "ddp_4gpu" "ddp_scaling_4gpu"
report_run "ddp_profile_4gpu" "ddp_scaling_4gpu_plain"
report_run "fsdp_1gpu" "fsdp_scaling_1gpu"
report_run "fsdp_2gpu" "fsdp_scaling_2gpu"
report_run "fsdp_4gpu" "fsdp_scaling_4gpu"
report_run "fsdp_profile_4gpu" "fsdp_scaling_4gpu_plain"
