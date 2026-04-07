from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model

from src.utils import count_parameters


PREFERRED_QWEN_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
EXCLUDED_MODULE_SUFFIXES = {
    "lm_head",
    "embed_tokens",
    "wte",
    "word_embeddings",
    "output",
    "score",
    "classifier",
}


def _normalize_modules_to_save(value: Any) -> List[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def discover_lora_target_modules(model: torch.nn.Module) -> Tuple[List[str], List[str]]:
    matched_by_leaf: Dict[str, List[str]] = {}
    fallback_by_leaf: Dict[str, List[str]] = {}

    for name, module in model.named_modules():
        if not name or not isinstance(module, torch.nn.Linear):
            continue

        leaf_name = name.split(".")[-1]
        if leaf_name in EXCLUDED_MODULE_SUFFIXES or "lm_head" in name:
            continue

        fallback_by_leaf.setdefault(leaf_name, []).append(name)
        if leaf_name in PREFERRED_QWEN_TARGETS:
            matched_by_leaf.setdefault(leaf_name, []).append(name)

    if matched_by_leaf:
        target_modules = [name for name in PREFERRED_QWEN_TARGETS if name in matched_by_leaf]
        matched_module_names = [full_name for leaf_name in target_modules for full_name in matched_by_leaf[leaf_name]]
        return target_modules, matched_module_names

    target_modules = sorted(
        leaf_name
        for leaf_name in fallback_by_leaf
        if leaf_name not in EXCLUDED_MODULE_SUFFIXES and "head" not in leaf_name
    )
    matched_module_names = [full_name for leaf_name in target_modules for full_name in fallback_by_leaf[leaf_name]]
    return target_modules, matched_module_names


def summarize_parameter_efficiency(model: torch.nn.Module) -> Dict[str, Any]:
    total_parameters = count_parameters(model, trainable_only=False)
    trainable_parameters = count_parameters(model, trainable_only=True)
    trainable_ratio = (trainable_parameters / total_parameters) if total_parameters else 0.0
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_ratio": trainable_ratio,
    }


def build_lora_metadata(
    peft_config: Mapping[str, Any],
    resolved_target_modules: Sequence[str],
    matched_module_names: Sequence[str],
    model: torch.nn.Module,
    target_modules_source: str,
) -> Dict[str, Any]:
    metadata = {
        "enabled": True,
        "method": peft_config.get("method", "lora"),
        "task_type": peft_config.get("task_type", "CAUSAL_LM"),
        "r": int(peft_config.get("r", 16)),
        "lora_alpha": int(peft_config.get("lora_alpha", 32)),
        "lora_dropout": float(peft_config.get("lora_dropout", 0.05)),
        "bias": peft_config.get("bias", "none"),
        "target_modules_source": target_modules_source,
        "target_modules": list(resolved_target_modules),
        "matched_module_names": list(matched_module_names),
        "modules_to_save": _normalize_modules_to_save(peft_config.get("modules_to_save")),
    }
    metadata.update(summarize_parameter_efficiency(model))
    return metadata


def apply_peft_to_model(
    model: torch.nn.Module,
    peft_config: Mapping[str, Any],
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    if not peft_config.get("enabled", False):
        return model, {"enabled": False, **summarize_parameter_efficiency(model)}

    method = str(peft_config.get("method", "lora")).lower()
    if method != "lora":
        raise ValueError(f"Unsupported PEFT method: {method}")

    explicit_target_modules = peft_config.get("target_modules")
    if explicit_target_modules:
        resolved_target_modules = [str(item) for item in explicit_target_modules]
        matched_module_names = []
        for name, _ in model.named_modules():
            if any(name.endswith(f".{leaf_name}") or name == leaf_name for leaf_name in resolved_target_modules):
                matched_module_names.append(name)
        target_modules_source = "explicit"
    else:
        strategy = str(peft_config.get("target_modules_strategy", "auto")).lower()
        if strategy != "auto":
            raise ValueError(f"Unsupported target_modules_strategy: {strategy}")
        resolved_target_modules, matched_module_names = discover_lora_target_modules(model)
        target_modules_source = "auto"

    if not resolved_target_modules:
        raise ValueError("No LoRA target modules were resolved for the model.")

    lora_config = LoraConfig(
        task_type=getattr(TaskType, str(peft_config.get("task_type", "CAUSAL_LM")).upper()),
        r=int(peft_config.get("r", 16)),
        lora_alpha=int(peft_config.get("lora_alpha", 32)),
        lora_dropout=float(peft_config.get("lora_dropout", 0.05)),
        bias=str(peft_config.get("bias", "none")),
        target_modules=list(resolved_target_modules),
        modules_to_save=_normalize_modules_to_save(peft_config.get("modules_to_save")),
    )
    wrapped_model = get_peft_model(model, lora_config)
    metadata = build_lora_metadata(
        peft_config=peft_config,
        resolved_target_modules=resolved_target_modules,
        matched_module_names=matched_module_names,
        model=wrapped_model,
        target_modules_source=target_modules_source,
    )
    return wrapped_model, metadata
