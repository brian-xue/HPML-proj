from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import DEFAULT_MODEL_NAME, get_device


DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_torch_dtype(dtype_name: str | None) -> Optional[torch.dtype]:
    if dtype_name is None:
        return None
    normalized = str(dtype_name).lower()
    if normalized not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return DTYPE_MAP[normalized]


def load_tokenizer(config: Mapping[str, Any]) -> Any:
    model_name = config.get("name", DEFAULT_MODEL_NAME)
    cache_dir = config.get("cache_dir")
    local_files_only = bool(config.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.get("padding_side", "right")
    tokenizer.truncation_side = config.get("truncation_side", "right")
    return tokenizer


def load_model(
    config: Mapping[str, Any],
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    model_name = config.get("name", DEFAULT_MODEL_NAME)
    dtype = resolve_torch_dtype(config.get("dtype"))
    target_device = device or get_device(config.get("device"))
    cache_dir = config.get("cache_dir")
    local_files_only = bool(config.get("local_files_only", False))
    trust_remote_code = bool(config.get("trust_remote_code", False))

    if bool(config.get("qlora", False)):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "QLoRA loading requires transformers BitsAndBytes support. "
                "Install compatible transformers/bitsandbytes packages to enable model.qlora=true."
            ) from exc

        compute_dtype = resolve_torch_dtype(config.get("qlora_compute_dtype")) or dtype or torch.bfloat16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("qlora_quant_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(config.get("qlora_use_double_quant", True)),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map={"": str(target_device)} if target_device.type == "cuda" else None,
        )
        model.config.use_cache = bool(config.get("use_cache", True))
        return model

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )
    model.config.use_cache = bool(config.get("use_cache", True))
    model.to(target_device)
    return model


def load_model_and_tokenizer(config: Mapping[str, Any]) -> Tuple[torch.nn.Module, Any, torch.device]:
    model_config = config.get("model", config)
    device = get_device(model_config.get("device"))
    tokenizer = load_tokenizer(model_config)
    model = load_model(model_config, device=device)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, device
