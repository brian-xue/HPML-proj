from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import load_checkpoint
from src.data import load_gsm8k_dataset, preprocess_dataset
from src.evaluator import evaluate_generation
from src.model import load_model_and_tokenizer
from src.utils import default_config, load_config, save_json, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared GSM8K evaluation.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, default=default_config())
    logger = setup_logger("eval")

    model, tokenizer, device = load_model_and_tokenizer(config)
    metadata = load_checkpoint(args.checkpoint, model=model, map_location=device)
    logger.info("Loaded checkpoint from %s", args.checkpoint)
    logger.info("Checkpoint metadata: %s", metadata)

    dataset = load_gsm8k_dataset(config["data"])
    processed = preprocess_dataset(dataset, tokenizer=None, config=config["data"])
    eval_split = config["data"].get("eval_split", "validation")
    results = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        dataset=processed[eval_split],
        config=config,
        device=device,
    )

    output_path = args.output or f"{args.checkpoint.rstrip('/')}/eval_metrics.json"
    save_json(results, output_path)
    logger.info("Saved evaluation metrics to %s", output_path)
    logger.info("Accuracy: %.4f", results["metrics"]["accuracy"])


if __name__ == "__main__":
    main()
