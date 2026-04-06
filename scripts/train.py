from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import get_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloaders
from src.model import load_model_and_tokenizer
from src.utils import (
    count_parameters,
    default_config,
    make_output_dir,
    save_config,
    set_random_seed,
    setup_logger,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared training setup entrypoint.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
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
    config: dict,
):
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config, default=default_config())
    if args.run_name is not None:
        config["run_name"] = args.run_name

    set_random_seed(int(config["seed"]))
    output_dir = make_output_dir(config["output_root"], config, run_name=config.get("run_name"))
    logger = setup_logger("train", output_dir=output_dir)
    save_config(config, output_dir / "config.yaml")

    model, tokenizer, device = load_model_and_tokenizer(config)
    dataloaders = build_dataloaders(tokenizer=tokenizer, config=config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, dataloaders["train"], config)

    logger.info("Training scaffolding initialized.")
    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)
    logger.info("Model parameters: %s", count_parameters(model))
    logger.info("Train batches per epoch: %s", len(dataloaders["train"]))
    logger.info("Eval batches: %s", len(dataloaders["eval"]))
    logger.info("Optimizer: %s", optimizer.__class__.__name__)
    logger.info("Scheduler: %s", scheduler.__class__.__name__)
    logger.info("Shared setup complete. Trainer-specific loops can be layered on top of this entrypoint.")


if __name__ == "__main__":
    main()
