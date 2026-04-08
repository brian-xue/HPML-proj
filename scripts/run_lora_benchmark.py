from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import get_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloaders
from src.peft import apply_peft_to_model
from src.model import load_model_and_tokenizer
from src.trainer_base import BaseTrainer
from src.utils import (
    count_parameters,
    deep_update,
    default_config,
    format_metrics,
    load_config,
    make_versioned_output_dir,
    resolve_versioned_run_dir,
    save_config,
    save_json,
    set_random_seed,
    setup_logger,
)


# Edit this list to add, remove, resume, or pin future LoRA benchmark runs.
EXPERIMENTS = [
    {
        "name": "qwen25_15b_lora_auto",
        "resume_mode": "new",
        "resume_version": None,
        "overrides": {
            "training": {
                "num_epochs": 2,
                "max_steps": 1000,
                "log_every_steps": 10,
                "eval_every_steps": 200,
                "save_every_steps": 200,
                "save_at_end": True,
            },
            "dataloader": {
                "train_batch_size": 8,
                "eval_batch_size": 8,
            },
            "optimizer": {
                "lr": 5e-5,
                "weight_decay": 0.0,
            },
            "peft": {
                "enabled": True,
                "method": "lora",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": ["q_proj", "v_proj"],
                "target_modules_strategy": None,
                "modules_to_save": None,
            },
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run versioned LoRA benchmark experiments on GSM8K.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device-config", type=str, default="configs/distributed/single_gpu.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="Run only the named experiments from EXPERIMENTS.")
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config.get("lr", 2e-5)),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1e-8)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
) -> Any:
    scheduler_config = config["scheduler"]
    training_config = config["training"]
    num_epochs = int(training_config.get("num_epochs", 1))
    max_steps = training_config.get("max_steps")
    total_steps = int(max_steps) if max_steps is not None else len(train_dataloader) * num_epochs
    return get_scheduler(
        name=scheduler_config.get("name", "linear"),
        optimizer=optimizer,
        num_warmup_steps=int(scheduler_config.get("num_warmup_steps", 0)),
        num_training_steps=total_steps,
    )


def filter_experiments(only: list[str] | None) -> list[Dict[str, Any]]:
    if not only:
        return EXPERIMENTS
    selected_names = set(only)
    return [experiment for experiment in EXPERIMENTS if experiment["name"] in selected_names]


def prepare_new_run_config(
    base_config_path: str,
    device_config_path: str | None,
    experiment: Dict[str, Any],
) -> Dict[str, Any]:
    config = load_config(base_config_path, default=default_config())
    if device_config_path:
        config = load_config(device_config_path, default=config)
    config = deep_update(config, deepcopy(experiment.get("overrides", {})))
    config["run_name"] = experiment["name"]
    return config


def prepare_resume_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Resume config not found: {config_path}")
    return load_config(config_path, default=default_config())


def resolve_run_directory(config: Dict[str, Any], experiment: Dict[str, Any]) -> Path:
    output_root = config["output_root"]
    if experiment["resume_mode"] == "new":
        return make_versioned_output_dir(output_root, experiment["name"])
    if experiment["resume_mode"] == "resume":
        return resolve_versioned_run_dir(output_root, experiment["name"], experiment.get("resume_version"))
    raise ValueError(f"Unsupported resume mode: {experiment['resume_mode']}")


def apply_lora_and_persist(
    model: torch.nn.Module,
    config: Dict[str, Any],
    run_dir: Path,
    logger: Any,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    peft_config = deepcopy(config.get("peft", {}))
    resolved_target_modules = peft_config.get("resolved_target_modules")
    if resolved_target_modules and not peft_config.get("target_modules"):
        peft_config["target_modules"] = resolved_target_modules

    model, peft_metadata = apply_peft_to_model(model, peft_config)
    if peft_metadata.get("enabled"):
        config["peft"]["resolved_target_modules"] = peft_metadata["target_modules"]
        config["peft"]["target_modules"] = peft_metadata["target_modules"]
        config["peft"]["target_modules_source"] = peft_metadata["target_modules_source"]

        if peft_metadata["target_modules_source"] == "auto":
            logger.info("Auto-detected LoRA target modules: %s", peft_metadata["target_modules"])
        else:
            logger.info("Using explicit LoRA target modules: %s", peft_metadata["target_modules"])
        logger.info("Matched module names: %s", peft_metadata["matched_module_names"])
        logger.info(
            "Parameter efficiency: %s",
            format_metrics(
                {
                    "total_parameters": peft_metadata["total_parameters"],
                    "trainable_parameters": peft_metadata["trainable_parameters"],
                    "trainable_ratio": peft_metadata["trainable_ratio"],
                }
            ),
        )
        save_json(peft_metadata, run_dir / "resolved_peft_config.json")
    return model, peft_metadata


def run_experiment(
    base_config_path: str,
    device_config_path: str | None,
    experiment: Dict[str, Any],
) -> None:
    is_resume = experiment["resume_mode"] == "resume"
    if is_resume:
        bootstrap_config = load_config(base_config_path, default=default_config())
        if device_config_path:
            bootstrap_config = load_config(device_config_path, default=bootstrap_config)
        bootstrap_config["run_name"] = experiment["name"]
        run_dir = resolve_run_directory(bootstrap_config, experiment)
        config = prepare_resume_run_config(run_dir)
    else:
        config = prepare_new_run_config(base_config_path, device_config_path, experiment)
        run_dir = resolve_run_directory(config, experiment)

    logger_name = f"benchmark.{experiment['name']}.{run_dir.name}"
    logger = setup_logger(logger_name, output_dir=run_dir)
    logger.info("Starting experiment '%s' in %s mode at %s", experiment["name"], experiment["resume_mode"], run_dir)

    set_random_seed(int(config["seed"]))
    model, tokenizer, device = load_model_and_tokenizer(config)
    model, peft_metadata = apply_lora_and_persist(model, config, run_dir, logger)

    save_config(config, run_dir / "config.yaml")
    save_json(experiment, run_dir / "experiment_definition.json")

    dataloaders = build_dataloaders(tokenizer=tokenizer, config=config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, dataloaders["train"], config)

    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=dataloaders["train"],
        eval_dataset=dataloaders["eval_generation"].dataset,
        device=device,
        config=config,
        output_dir=run_dir,
        logger=logger,
    )
    if is_resume:
        trainer.resume_from_latest_checkpoint()

    logger.info("Device: %s", device)
    logger.info("Total parameters before training: %s", count_parameters(model))
    results = trainer.train()
    results["peft"] = peft_metadata
    save_json(results, run_dir / "final_results.json")
    logger.info("Finished experiment '%s'. Results saved to %s", experiment["name"], run_dir / "final_results.json")


def main() -> None:
    args = parse_args()
    experiments = filter_experiments(args.only)
    if not experiments:
        raise ValueError("No experiments selected. Check --only against the EXPERIMENTS list.")

    for experiment in experiments:
        run_experiment(
            base_config_path=args.config,
            device_config_path=args.device_config,
            experiment=experiment,
        )


if __name__ == "__main__":
    main()
