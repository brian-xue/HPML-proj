from __future__ import annotations
import argparse, sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_runner import run_experiment

EXPERIMENT = {
    "name": "gradient_checkpointing_benchmark",
    "description": "LoRA + gradient checkpointing on GSM8K.",
    "base_config": "configs/base.yaml",
    "device_config": "configs/distributed/single_gpu.yaml",
    "run": {"mode": "new", "resume_version": None},
    "artifacts": {"results_filename": "final_results.json", "save_eval_metrics_json": False},
    "overrides": {
        "model": {
            "gradient_checkpointing": True,
            "use_cache": False,
        },
        "training": {
            "num_epochs": 1,
            "max_steps": 50,
            "log_every_steps": 10,
            "eval_every_steps": 999,
            "save_every_steps": 999,
            "save_at_end": False,
        },
        "dataloader": {"train_batch_size": 4, "eval_batch_size": 4},
        "optimizer": {"lr": 1.0e-4, "weight_decay": 0.0},
        "data": {"max_length": 256, "num_preprocessing_workers": 1},
        "peft": {
            "enabled": True,
            "method": "lora",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": [
                "q_proj", "k_proj", "v_proj",
                "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
            "target_modules_strategy": "auto",
            "modules_to_save": None,
        },
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["new", "resume"], default="new")
    parser.add_argument("--resume-version", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    exp = deepcopy(EXPERIMENT)
    exp["run"]["mode"] = args.mode
    exp["run"]["resume_version"] = args.resume_version
    if args.max_steps is not None:
        exp.setdefault("overrides", {}).setdefault("training", {})["max_steps"] = int(args.max_steps)
    if args.output_root is not None:
        exp.setdefault("overrides", {})["output_root"] = str(args.output_root)
    run_dir = run_experiment(exp, dry_run=bool(args.dry_run))
    print(run_dir)

if __name__ == "__main__":
    main()