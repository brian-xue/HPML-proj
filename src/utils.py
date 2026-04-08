from __future__ import annotations

import json
import logging
import os
import random
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import torch
import yaml


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_for_path(value: Any) -> str:
    text = str(value).strip().replace(" ", "_").replace("/", "-")
    return "".join(char for char in text if char.isalnum() or char in {"_", "-", "."})


def generate_run_name(config: Mapping[str, Any], prefix: Optional[str] = None) -> str:
    model_name = sanitize_for_path(Path(config.get("model", {}).get("name", DEFAULT_MODEL_NAME)).name)
    dataset_name = sanitize_for_path(config.get("data", {}).get("dataset_name", "gsm8k"))
    seed = config.get("seed", 42)
    explicit_name = config.get("run_name")
    if explicit_name:
        stem = sanitize_for_path(explicit_name)
    else:
        stem = f"{dataset_name}_{model_name}_seed{seed}"
    if prefix:
        stem = f"{sanitize_for_path(prefix)}_{stem}"
    return f"{stem}_{timestamp()}"


def make_output_dir(
    base_dir: str | Path,
    config: Mapping[str, Any],
    run_name: Optional[str] = None,
    exist_ok: bool = False,
) -> Path:
    base_path = ensure_dir(base_dir)
    final_run_name = run_name or generate_run_name(config)
    output_dir = base_path / final_run_name
    output_dir.mkdir(parents=True, exist_ok=exist_ok)
    return output_dir


def _close_logger_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def setup_logger(
    name: str,
    output_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
    filename: str = "run.log",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        _close_logger_handlers(logger)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_dir is not None:
        log_path = ensure_dir(output_dir) / filename
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(path: Optional[str | Path] = None, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = deepcopy(default or {})
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(handle) or {}
        elif config_path.suffix == ".json":
            loaded = json.load(handle)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return deep_update(config, loaded)


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            yaml.safe_dump(dict(config), handle, sort_keys=False)
        elif path.suffix == ".json":
            json.dump(config, handle, indent=2, sort_keys=True)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    return path


def save_json(data: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_json_serializable(data), handle, indent=2, sort_keys=True)
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_json_serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def flatten_dict(data: Mapping[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_dict(value, prefix=flat_key, sep=sep))
        else:
            flat[flat_key] = value
    return flat


def format_metric_value(value: Any, precision: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, torch.Tensor):
        return format_metric_value(value.item(), precision=precision)
    return str(value)


def format_metrics(metrics: Mapping[str, Any], precision: int = 4) -> Dict[str, str]:
    return {key: format_metric_value(value, precision=precision) for key, value in metrics.items()}


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (param for param in parameters if param.requires_grad)
    return sum(param.numel() for param in parameters)


def get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def chunked(items: Iterable[Any], chunk_size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


VERSION_PATTERN = re.compile(r"^v(\d+)$")


def get_experiment_root(output_root: str | Path, experiment_name: str) -> Path:
    return ensure_dir(Path(output_root) / sanitize_for_path(experiment_name))


def list_versioned_run_dirs(experiment_root: str | Path) -> list[Path]:
    root = Path(experiment_root)
    versioned_dirs = []
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and VERSION_PATTERN.match(child.name):
            versioned_dirs.append(child)
    return sorted(versioned_dirs, key=lambda path: int(VERSION_PATTERN.match(path.name).group(1)))


def get_latest_version_dir(experiment_root: str | Path) -> Optional[Path]:
    versioned_dirs = list_versioned_run_dirs(experiment_root)
    return versioned_dirs[-1] if versioned_dirs else None


def make_versioned_output_dir(output_root: str | Path, experiment_name: str) -> Path:
    experiment_root = get_experiment_root(output_root, experiment_name)
    latest_dir = get_latest_version_dir(experiment_root)
    next_version = 1 if latest_dir is None else int(VERSION_PATTERN.match(latest_dir.name).group(1)) + 1
    run_dir = experiment_root / f"v{next_version:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_versioned_run_dir(
    output_root: str | Path,
    experiment_name: str,
    version: Optional[str] = None,
) -> Path:
    experiment_root = get_experiment_root(output_root, experiment_name)
    if version is None or version == "latest":
        latest_dir = get_latest_version_dir(experiment_root)
        if latest_dir is None:
            raise FileNotFoundError(f"No versioned runs found for experiment: {experiment_name}")
        return latest_dir

    run_dir = experiment_root / version
    if not run_dir.exists():
        raise FileNotFoundError(f"Versioned run not found: {run_dir}")
    return run_dir


def default_config() -> Dict[str, Any]:
    return {
        "seed": 42,
        "output_root": "output",
        "run_name": None,
        "model": {
            "name": DEFAULT_MODEL_NAME,
            "dtype": "bf16",
            "device": None,
            "trust_remote_code": False,
            "use_cache": True,
            "padding_side": "right",
            "truncation_side": "right",
        },
        "data": {
            "dataset_name": "gsm8k",
            "dataset_config_name": "main",
            "cache_dir": "data/raw",
            "processed_dir": "data/processed",
            "load_from_disk_path": None,
            "train_split": "train",
            "eval_split": "validation",
            "test_split": "test",
            "validation_size": 0.05,
            "split_seed": 42,
            "max_length": 256,
            "num_preprocessing_workers": 1,
            "overwrite_cache": False,
            "pack_to_max_length": False,
            "append_eos_token": True,
            "system_prompt": "You are a helpful math reasoning assistant. Solve the problem carefully and end with a final answer.",
            "instruction_template": "Solve the following math word problem. Show your reasoning before giving the final answer.\n\nQuestion: {question}",
        },
        "dataloader": {
            "train_batch_size": 8,
            "eval_batch_size": 8,
            "num_workers": 0,
            "pin_memory": True,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "scheduler": {
            "name": "linear",
            "num_warmup_steps": 0,
        },
        "training": {
            "num_epochs": 1,
            "max_steps": None,
            "gradient_accumulation_steps": 1,
            "log_every_steps": 10,
            "eval_every_steps": 200,
            "save_every_steps": 200,
            "save_at_end": True,
            "resume_from": None,
        },
        "generation": {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "peft": {
            "enabled": False,
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
        "checkpoint": {
            "dir_name": "checkpoints",
            "metric_name": "accuracy",
            "metric_mode": "max",
        },
    }
