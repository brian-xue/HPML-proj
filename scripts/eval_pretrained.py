from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained base-model accuracy with LoRA benchmark settings.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional base config override. Defaults to the LoRA benchmark base_config.",
    )
    parser.add_argument(
        "--device-config",
        type=str,
        default=None,
        help="Optional device config override. Defaults to the LoRA benchmark device_config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to evaluate. Defaults to data.eval_split from the config.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit evaluation to the first N examples for a faster baseline test.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Exit with an error if accuracy is below this threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results. Defaults under output/pretrained_lora_settings_eval/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from experiments.lora_benchmark import EXPERIMENT as LORA_EXPERIMENT
    from src.evaluator import evaluate_pretrained_generation
    from src.experiment_runner import load_effective_config
    from src.utils import (
        generate_run_name,
        save_config,
        save_json,
        set_random_seed,
        setup_logger,
    )

    exp = deepcopy(LORA_EXPERIMENT)
    base_config = args.config or exp.get("base_config", "configs/base.yaml")
    device_config = args.device_config if args.device_config is not None else exp.get("device_config")
    overrides = deepcopy(exp.get("overrides") or {})
    overrides.setdefault("peft", {})["enabled"] = False

    config = load_effective_config(
        base_config_path=PROJECT_ROOT / base_config,
        device_config_path=PROJECT_ROOT / device_config if device_config else None,
        overrides=overrides,
    )
    set_random_seed(int(config.get("seed", 42)))

    run_name = generate_run_name(config, prefix="pretrained_lora_settings_eval")
    output_path = (
        Path(args.output)
        if args.output
        else Path(config["output_root"]) / "pretrained_lora_settings_eval" / run_name / "results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("eval_pretrained", output_dir=output_path.parent)
    save_config(config, output_path.parent / "config.yaml")

    split_name = args.split or config["data"].get("eval_split", "validation")
    logger.info("Evaluating pretrained model with LoRA benchmark settings and 4-shot CoT prompting.")
    results = evaluate_pretrained_generation(config=config, split=split_name, max_examples=args.max_examples)
    results["metadata"] = {
        "model_name": config["model"].get("name"),
        "split": split_name,
        "max_examples": args.max_examples,
        "base_config": str(base_config),
        "device_config": str(device_config) if device_config else None,
        "matched_experiment": "lora_benchmark",
        "prompting": "4-shot CoT",
        "peft_enabled": False,
    }

    save_json(results, output_path)
    accuracy = float(results["metrics"]["accuracy"])
    logger.info("Saved pretrained evaluation results to %s", output_path)
    logger.info("Accuracy: %.4f (%s/%s)", accuracy, results["metrics"]["num_correct"], results["metrics"]["num_examples"])

    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        raise SystemExit(f"Accuracy {accuracy:.4f} is below required threshold {args.min_accuracy:.4f}")


if __name__ == "__main__":
    main()
