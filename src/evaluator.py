from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

import torch

from src.data import build_generation_dataloader
from src.metrics import RuntimeTracker
from src.utils import move_batch_to_device


NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_final_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip()

    lowered = text.lower()
    markers = ["final answer:", "answer:", "the answer is", "therefore"]
    for marker in markers:
        if marker in lowered:
            start = lowered.rfind(marker)
            return text[start + len(marker) :].strip()

    matches = NUMBER_PATTERN.findall(text)
    if matches:
        return matches[-1].replace(",", "")
    return text.strip()


def answers_match(prediction: str, reference: str) -> bool:
    pred = extract_final_answer(prediction)
    ref = extract_final_answer(reference)
    pred_norm = normalize_answer(pred).replace(",", "")
    ref_norm = normalize_answer(ref).replace(",", "")
    return pred_norm == ref_norm


def limit_dataset(dataset: Any, max_examples: Optional[int]) -> Any:
    if max_examples is None:
        return dataset
    if max_examples <= 0:
        raise ValueError("max_examples must be a positive integer when provided.")
    return dataset.select(range(min(max_examples, len(dataset))))


@torch.inference_mode()
def evaluate_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Any,
    config: Mapping[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    generation_config = config.get("generation", {})
    dataloader_config = config.get("dataloader", {})
    target_device = device or next(model.parameters()).device

    dataloader = build_generation_dataloader(
        dataset,
        batch_size=int(dataloader_config.get("eval_batch_size", 4)),
        shuffle=False,
        num_workers=int(dataloader_config.get("num_workers", 0)),
    )

    tracker = RuntimeTracker(device=target_device)
    tracker.start_run()

    model.eval()
    predictions: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for batch in dataloader:
            prompts = batch["prompt"]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=int(config["data"].get("max_length", 512)),
                return_tensors="pt",
            )
            encoded = move_batch_to_device(encoded, target_device)

            tracker.start_step()
            generated = model.generate(
                **encoded,
                max_new_tokens=int(generation_config.get("max_new_tokens", 128)),
                do_sample=bool(generation_config.get("do_sample", False)),
                temperature=float(generation_config.get("temperature", 0.0)),
                top_p=float(generation_config.get("top_p", 1.0)),
                num_beams=int(generation_config.get("num_beams", 1)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            tracker.end_step(samples=len(prompts), tokens=int(generated.numel()))

            sequence_length = encoded["input_ids"].shape[1]
            decoded_outputs = tokenizer.batch_decode(generated[:, sequence_length:], skip_special_tokens=True)

            for question, reference, prediction in zip(
                batch["question"],
                batch["reference_answer"],
                decoded_outputs,
            ):
                is_correct = answers_match(prediction, reference)
                correct += int(is_correct)
                total += 1
                predictions.append(
                    {
                        "question": question,
                        "reference_answer": reference,
                        "prediction": prediction,
                        "predicted_answer": extract_final_answer(prediction),
                        "correct": is_correct,
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side

    tracker.stop_run()

    metrics = tracker.summary()
    metrics["accuracy"] = correct / total if total else 0.0
    metrics["num_examples"] = total
    metrics["num_correct"] = correct

    return {
        "metrics": metrics,
        "predictions": predictions,
    }


def evaluate_pretrained_generation(
    config: Mapping[str, Any],
    split: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    from src.data import build_dataloaders
    from src.model import load_model_and_tokenizer

    model, tokenizer, device = load_model_and_tokenizer(config)
    dataloaders = build_dataloaders(tokenizer=tokenizer, config=config)

    configured_split = config["data"].get("eval_split", "validation")
    split_name = split or configured_split
    if split_name != configured_split:
        raise ValueError(
            f"split={split_name!r} does not match the configured eval split {configured_split!r}. "
            "Update config['data']['eval_split'] to change the eval setting."
        )

    eval_dataset = limit_dataset(dataloaders["eval_generation"].dataset, max_examples)
    return evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        config=config,
        device=device,
    )
