from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import get_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloaders
from src.model import load_model_and_tokenizer
from src.trainer_base import BaseTrainer
from src.utils import default_config, ensure_dir, load_config, save_json, set_random_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight end-to-end smoke test.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model-name", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--output-dir", type=str, default="output/smoke_test")
    return parser.parse_args()


def build_synthetic_dataset() -> DatasetDict:
    train_examples = {
        "question": [
            "If Ana has 2 apples and buys 3 more, how many apples does she have?",
            "Tom has 10 books and gives away 4. How many books remain?",
            "A box has 6 red marbles and 5 blue marbles. How many marbles are there in total?",
            "Sara solved 7 problems on Monday and 8 on Tuesday. How many total problems did she solve?",
        ],
        "answer": [
            "Ana starts with 2 apples and buys 3 more. 2 + 3 = 5. #### 5",
            "Tom gives away 4 from 10. 10 - 4 = 6. #### 6",
            "There are 6 red and 5 blue marbles. 6 + 5 = 11. #### 11",
            "Sara solved 7 and then 8 problems. 7 + 8 = 15. #### 15",
        ],
    }
    eval_examples = {
        "question": [
            "There are 9 birds on a tree and 2 fly away. How many birds are left?",
            "Liam has 4 bags with 3 candies in each bag. How many candies does he have?",
        ],
        "answer": [
            "Two birds fly away from 9. 9 - 2 = 7. #### 7",
            "There are 4 groups of 3 candies. 4 * 3 = 12. #### 12",
        ],
    }
    return DatasetDict(
        {
            "train": Dataset.from_dict(train_examples),
            "validation": Dataset.from_dict(eval_examples),
            "test": Dataset.from_dict(eval_examples),
        }
    )


def save_synthetic_dataset(output_dir: Path) -> Path:
    dataset_dir = ensure_dir(output_dir / "synthetic_gsm8k")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_synthetic_dataset()
    dataset.save_to_disk(str(dataset_dir))
    return dataset_dir


def build_optimizer(model, config):
    optimizer_config = config["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config.get("lr", 5e-5)),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1e-8)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, default=default_config())
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("smoke_test", output_dir=output_dir)

    dataset_path = save_synthetic_dataset(output_dir)

    # The smoke test uses a tiny public model and CPU-friendly settings so the
    # shared pipeline can be verified quickly without the full Qwen checkpoint.
    config["model"]["name"] = args.model_name
    config["model"]["dtype"] = "fp32"
    config["model"]["device"] = "cpu"
    config["model"]["use_cache"] = True
    config["data"]["load_from_disk_path"] = str(dataset_path)
    config["data"]["max_length"] = 128
    config["data"]["num_preprocessing_workers"] = 1
    config["dataloader"]["train_batch_size"] = 2
    config["dataloader"]["eval_batch_size"] = 2
    config["dataloader"]["num_workers"] = 0
    config["training"]["num_epochs"] = 1
    config["training"]["max_steps"] = 2
    config["training"]["gradient_accumulation_steps"] = 1
    config["training"]["log_every_steps"] = 1
    config["generation"]["max_new_tokens"] = 16
    config["generation"]["do_sample"] = False
    config["generation"]["temperature"] = 0.0
    config["optimizer"]["lr"] = 5e-5
    config["optimizer"]["weight_decay"] = 0.0
    config["run_name"] = "smoke_test"
    config["output_root"] = str(output_dir)

    set_random_seed(int(config["seed"]))
    model, tokenizer, device = load_model_and_tokenizer(config)
    dataloaders = build_dataloaders(tokenizer=tokenizer, config=config)
    optimizer = build_optimizer(model, config)
    scheduler = get_scheduler(
        name=config["scheduler"]["name"],
        optimizer=optimizer,
        num_warmup_steps=int(config["scheduler"].get("num_warmup_steps", 0)),
        num_training_steps=int(config["training"]["max_steps"]),
    )

    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=dataloaders["train"],
        eval_dataset=dataloaders["eval_generation"].dataset,
        device=device,
        config=config,
        output_dir=output_dir,
        logger=logger,
    )
    results = trainer.train()

    save_json(results, output_dir / "smoke_test_results.json")
    logger.info("Smoke test finished successfully.")
    logger.info("Results saved to %s", output_dir / "smoke_test_results.json")


if __name__ == "__main__":
    main()
