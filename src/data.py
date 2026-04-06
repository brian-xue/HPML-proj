from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader

from src.utils import ensure_dir


GSM8K_DATASET_NAME = "gsm8k"
GSM8K_CONFIG_NAME = "main"


def extract_reference_answer(answer_text: str) -> str:
    marker = "####"
    if marker in answer_text:
        return answer_text.split(marker)[-1].strip()
    return answer_text.strip()


def build_instruction_text(example: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    question = str(example["question"]).strip()
    template = config.get(
        "instruction_template",
        "Solve the following math word problem. Show your reasoning before giving the final answer.\n\nQuestion: {question}",
    )
    return template.format(question=question)


def format_gsm8k_example(
    example: Mapping[str, Any],
    tokenizer: Optional[Any] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(config or {})
    system_prompt = cfg.get(
        "system_prompt",
        "You are a helpful math reasoning assistant. Solve the problem carefully and end with a final answer.",
    )
    prompt_text = build_instruction_text(example, cfg)
    response_text = str(example["answer"]).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_messages = messages + [{"role": "assistant", "content": response_text}]
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    else:
        prompt = f"System: {system_prompt}\nUser: {prompt_text}\nAssistant: "
        full_text = prompt + response_text

    return {
        "id": example.get("id"),
        "question": example["question"],
        "answer": response_text,
        "reference_answer": extract_reference_answer(response_text),
        "prompt": prompt,
        "response": response_text,
        "text": full_text,
    }


def load_gsm8k_dataset(config: Mapping[str, Any]) -> DatasetDict:
    load_from_disk_path = config.get("load_from_disk_path")
    if load_from_disk_path:
        return load_from_disk(str(load_from_disk_path))

    dataset_name = config.get("dataset_name", GSM8K_DATASET_NAME)
    dataset_config_name = config.get("dataset_config_name", GSM8K_CONFIG_NAME)
    cache_dir = config.get("cache_dir", "data/raw")
    dataset = load_dataset(dataset_name, dataset_config_name, cache_dir=cache_dir)
    return ensure_validation_split(dataset, config)


def ensure_validation_split(dataset: DatasetDict, config: Mapping[str, Any]) -> DatasetDict:
    if "validation" in dataset:
        return dataset

    validation_size = float(config.get("validation_size", 0.05))
    split_seed = int(config.get("split_seed", 42))
    split = dataset["train"].train_test_split(test_size=validation_size, seed=split_seed, shuffle=True)
    return DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": dataset["test"],
        }
    )


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: Optional[Any],
    config: Mapping[str, Any],
) -> DatasetDict:
    num_proc = int(config.get("num_preprocessing_workers", 1))
    max_length = int(config.get("max_length", 512))
    append_eos_token = bool(config.get("append_eos_token", True))

    def _format_fn(example: Mapping[str, Any]) -> Dict[str, Any]:
        return format_gsm8k_example(example, tokenizer=tokenizer, config=config)

    formatted = dataset.map(
        _format_fn,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc if num_proc > 1 else None,
        desc="Formatting GSM8K examples",
    )

    if tokenizer is None:
        return formatted

    def _tokenize_fn(example: Mapping[str, Any]) -> Dict[str, Any]:
        prompt_tokens = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        response_tokens = tokenizer(
            example["response"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]

        if append_eos_token and (not input_ids or input_ids[-1] != tokenizer.eos_token_id):
            input_ids.append(tokenizer.eos_token_id)
            attention_mask.append(1)
            labels.append(tokenizer.eos_token_id)

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": len(prompt_tokens["input_ids"]),
            "response_length": len(response_tokens["input_ids"]),
        }

    return formatted.map(
        _tokenize_fn,
        num_proc=num_proc if num_proc > 1 else None,
        desc="Tokenizing GSM8K examples",
    )


@dataclass
class CausalLMCollator:
    tokenizer: Any
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded = label + [self.label_pad_token_id] * (max_length - len(label))
            padded_labels.append(padded[:max_length])

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def build_dataloader(
    dataset: Dataset,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    collator = CausalLMCollator(tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collator,
    )


def build_generation_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    def _collate(features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        keys = features[0].keys()
        return {key: [feature[key] for feature in features] for key in keys}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
    )


def build_dataloaders(
    tokenizer: Any,
    config: Mapping[str, Any],
    include_generation_eval: bool = True,
) -> Dict[str, DataLoader]:
    data_config = config["data"]
    dataloader_config = config["dataloader"]

    dataset = load_gsm8k_dataset(data_config)
    processed = preprocess_dataset(dataset, tokenizer=tokenizer, config=data_config)

    train_split = data_config.get("train_split", "train")
    eval_split = data_config.get("eval_split", "validation")

    loaders = {
        "train": build_dataloader(
            processed[train_split],
            tokenizer=tokenizer,
            batch_size=int(dataloader_config.get("train_batch_size", 4)),
            shuffle=True,
            num_workers=int(dataloader_config.get("num_workers", 0)),
            pin_memory=bool(dataloader_config.get("pin_memory", True)),
        ),
        "eval": build_dataloader(
            processed[eval_split],
            tokenizer=tokenizer,
            batch_size=int(dataloader_config.get("eval_batch_size", 4)),
            shuffle=False,
            num_workers=int(dataloader_config.get("num_workers", 0)),
            pin_memory=bool(dataloader_config.get("pin_memory", True)),
        ),
    }

    if include_generation_eval:
        loaders["eval_generation"] = build_generation_dataloader(
            processed[eval_split],
            batch_size=int(dataloader_config.get("eval_batch_size", 4)),
            shuffle=False,
            num_workers=int(dataloader_config.get("num_workers", 0)),
        )

    return loaders


def save_processed_dataset(dataset: DatasetDict, output_dir: str | Path) -> Path:
    output_path = ensure_dir(output_dir) / "gsm8k"
    dataset.save_to_disk(str(output_path))
    return output_path
