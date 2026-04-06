from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_gsm8k_dataset, preprocess_dataset, save_processed_dataset
from src.model import load_tokenizer
from src.utils import default_config, ensure_dir, load_config, save_config, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSM8K data for shared fine-tuning experiments.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--tokenize", action="store_true", help="Tokenize during preprocessing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached processed data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, default=default_config())
    logger = setup_logger("prepare_data")

    processed_root = Path(config["data"]["processed_dir"])
    output_path = processed_root / "gsm8k"
    if output_path.exists() and not args.overwrite:
        logger.info("Processed dataset already exists at %s", output_path)
        return
    if output_path.exists():
        shutil.rmtree(output_path)

    ensure_dir(processed_root)
    dataset = load_gsm8k_dataset(config["data"])
    tokenizer = load_tokenizer(config["model"]) if args.tokenize else None
    processed = preprocess_dataset(dataset, tokenizer=tokenizer, config=config["data"])
    saved_path = save_processed_dataset(processed, processed_root)
    save_config(config, saved_path / "prepare_config.yaml")
    logger.info("Saved processed dataset to %s", saved_path)


if __name__ == "__main__":
    main()
