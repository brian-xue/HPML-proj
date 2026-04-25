from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_runner import run_experiment
from src.distributed import get_world_size, is_main_process


EXPERIMENT = {
    "name": "ddp_scaling",
    "description": "DDP scaling run (weak scaling; eval disabled).",
    "base_config": "configs/base.yaml",
    "device_config": "configs/distributed/ddp.yaml",
    "run": {"mode": "auto", "resume_version": None},
    "artifacts": {"results_filename": "final_results.json", "save_eval_metrics_json": False},
    "overrides": {
        "evaluation": {"enabled": False},
        "checkpoint": {"save_trainable_only": True},
        "training": {"eval_every_steps": 0},
        "peft": {
            "enabled": True,
            "method": "lora",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": None,
            "target_modules_strategy": "auto",
            "modules_to_save": None,
        },
        "profile": {"nvtx": True},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DDP scaling benchmark.")
    parser.add_argument("--mode", choices=["auto", "new", "resume"], default="auto")
    parser.add_argument("--resume-version", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-label", type=str, default=None, help="Optional suffix label for experiment name.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = deepcopy(EXPERIMENT)
    world_size = max(1, int(get_world_size()))
    exp["run"]["mode"] = args.mode
    exp["run"]["resume_version"] = args.resume_version
    exp.setdefault("overrides", {}).setdefault("training", {})["max_steps"] = int(args.max_steps)
    exp.setdefault("overrides", {}).setdefault("profile", {})
    exp["overrides"]["profile"]["cuda_profiler"] = bool(int(os.environ.get("HPML_PROFILE_CUDA", "0")))
    exp["overrides"]["profile"]["warmup_steps"] = int(os.environ.get("HPML_PROFILE_WARMUP_STEPS", "0"))
    exp["overrides"]["profile"]["active_steps"] = int(os.environ.get("HPML_PROFILE_ACTIVE_STEPS", "0"))
    exp["name"] = f"{exp['name']}_{world_size}gpu"

    if args.output_root is not None:
        exp.setdefault("overrides", {})["output_root"] = str(args.output_root)

    if args.run_label:
        exp["name"] = f"{exp['name']}_{args.run_label}"

    run_dir = run_experiment(exp, dry_run=bool(args.dry_run))
    if is_main_process():
        print(run_dir)


if __name__ == "__main__":
    main()
