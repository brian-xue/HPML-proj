from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn

from src.utils import count_parameters, move_batch_to_device


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
        target_modules = [
            name for name in PREFERRED_QWEN_TARGETS if name in matched_by_leaf]
        matched_module_names = [
            full_name for leaf_name in target_modules for full_name in matched_by_leaf[leaf_name]]
        return target_modules, matched_module_names

    target_modules = sorted(
        leaf_name
        for leaf_name in fallback_by_leaf
        if leaf_name not in EXCLUDED_MODULE_SUFFIXES and "head" not in leaf_name
    )
    matched_module_names = [
        full_name for leaf_name in target_modules for full_name in fallback_by_leaf[leaf_name]]
    return target_modules, matched_module_names


def summarize_parameter_efficiency(model: torch.nn.Module) -> Dict[str, Any]:
    total_parameters = count_parameters(model, trainable_only=False)
    trainable_parameters = count_parameters(model, trainable_only=True)
    trainable_ratio = (trainable_parameters /
                       total_parameters) if total_parameters else 0.0
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


@dataclass
class GoRAResolvedConfig:
    rank: int
    alpha: int
    dropout: float
    init_method: str
    weight_a_init_method: str | None
    weight_b_init_method: str | None
    run_in_fp32: bool
    rank_stabilize: bool
    dynamic_scaling: bool
    importance_type: str
    scale_importance: bool
    temperature: float
    softmax_importance: bool
    allocate_strategy: str
    features_func: str | None
    min_rank: int
    max_rank: int | None
    stable_gamma: float
    lr: float
    gradient_estimation_steps: int
    rank_pattern: Dict[str, int] | None


class GoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: GoRAResolvedConfig, initial_rank: int = 0):
        super().__init__()
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.avg_rank = int(config.rank)
        self.alpha = int(config.alpha)
        self.dropout = nn.Dropout(float(config.dropout)) if float(
            config.dropout) > 0 else nn.Identity()
        self.init_method = str(config.init_method)
        self.weight_a_init_method = config.weight_a_init_method
        self.weight_b_init_method = config.weight_b_init_method
        self.run_in_fp32 = bool(config.run_in_fp32)
        self.rank_stabilize = bool(config.rank_stabilize)
        self.dynamic_scaling = bool(config.dynamic_scaling)

        self.weight = nn.Parameter(
            base_layer.weight.detach().clone(), requires_grad=False)
        if base_layer.bias is not None:
            self.bias = nn.Parameter(
                base_layer.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.active_rank = 0
        self.scale_rank = 1.0
        self.lora_scaler = float(self.alpha) / max(float(self.avg_rank), 1.0)
        self.grad_sum: torch.Tensor | None = None
        self.grad_steps = 0
        self.importance_score: float | tuple[float, float] | None = None
        self.register_buffer("error", torch.tensor(0.0))
        self.register_buffer("relative_error", torch.tensor(0.0))

        if initial_rank > 0:
            self.allocate_rank(initial_rank)
            self.init_lora_weights()

    def _get_lora_dtype(self) -> torch.dtype:
        return torch.float32 if self.run_in_fp32 else self.weight.dtype

    def _set_scaling(self, real_rank: int) -> None:
        base_rank = real_rank if self.dynamic_scaling else self.avg_rank
        scale_rank = max(float(base_rank), 1.0)
        if self.rank_stabilize:
            scale_rank = math.sqrt(scale_rank)
        self.scale_rank = scale_rank
        self.lora_scaler = float(self.alpha) / scale_rank

    def allocate_rank(self, rank: int) -> None:
        rank = max(0, min(int(rank), min(self.in_features, self.out_features)))
        self.active_rank = rank
        self._set_scaling(rank if rank > 0 else self.avg_rank)
        if rank <= 0:
            if hasattr(self, "weight_a"):
                delattr(self, "weight_a")
            if hasattr(self, "weight_b"):
                delattr(self, "weight_b")
            return

        dtype = self._get_lora_dtype()
        device = self.weight.device
        self.weight_a = nn.Parameter(torch.empty(
            (rank, self.in_features), device=device, dtype=dtype))
        self.weight_b = nn.Parameter(torch.empty(
            (self.out_features, rank), device=device, dtype=dtype))

    def _init_weight(self, weight_name: str, method: str | None) -> None:
        weight = getattr(self, weight_name)
        if weight_name == "weight_a":
            if method == "kaiming":
                nn.init.kaiming_uniform_(weight, a=5**0.5)
            elif method == "normal":
                nn.init.normal_(weight, mean=0.0, std=0.02)
            else:
                nn.init.normal_(weight, mean=0.0, std=1 /
                                (self.in_features**0.5))
            return

        if method == "kaiming":
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        elif method in {"normal", "gaussian"}:
            nn.init.normal_(weight, mean=0.0, std=0.02 if method ==
                            "normal" else 1 / max(self.active_rank, 1) ** 0.5)
        elif method == "orthogonal":
            nn.init.orthogonal_(weight)
        else:
            nn.init.zeros_(weight)

    def init_lora_weights(self) -> None:
        if self.active_rank <= 0:
            return
        self._init_weight("weight_a", self.weight_a_init_method)
        self._init_weight("weight_b", self.weight_b_init_method)

    def _compute_lora_weight(self) -> torch.Tensor:
        if self.active_rank <= 0:
            return torch.zeros_like(self.weight)
        lora_weight = torch.matmul(self.weight_b.to(
            self._get_lora_dtype()), self.weight_a.to(self._get_lora_dtype()))
        return (self.lora_scaler * lora_weight).to(self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.active_rank <= 0:
            return result

        dropped = self.dropout(x).to(self._get_lora_dtype())
        lora_out = F.linear(F.linear(dropped, self.weight_a),
                            self.weight_b).to(result.dtype)
        return result + self.lora_scaler * lora_out

    def reset_gradient_stats(self) -> None:
        self.grad_sum = None
        self.grad_steps = 0
        self.importance_score = None

    def accumulate_gradient(self) -> None:
        if self.weight.grad is None:
            return
        grad = self.weight.grad.detach().cpu()
        self.grad_sum = grad if self.grad_sum is None else self.grad_sum + grad
        self.grad_steps += 1

    def averaged_gradient(self) -> torch.Tensor | None:
        if self.grad_sum is None or self.grad_steps <= 0:
            return None
        return self.grad_sum / float(self.grad_steps)

    def weight_svd_init(self) -> None:
        if self.active_rank <= 0:
            return
        weight = self.weight.detach().to(torch.float32)
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        rank = min(self.active_rank, s.shape[0])
        s = s[:rank] / max(self.lora_scaler, 1e-8)
        sqrt_s = torch.sqrt(torch.clamp(s, min=1e-12))
        weight_a = torch.diag(sqrt_s) @ vh[:rank]
        weight_b = u[:, :rank] @ torch.diag(sqrt_s)
        self.weight_a.data.copy_(weight_a.to(self.weight_a.dtype))
        self.weight_b.data.copy_(weight_b.to(self.weight_b.dtype))
        self.weight.data.copy_(
            (weight - self._compute_lora_weight().to(weight.dtype)).to(self.weight.dtype))

    def grad_svd_init(self) -> None:
        grad = self.averaged_gradient()
        if self.active_rank <= 0 or grad is None:
            return
        grad = grad.to(self.weight.device, dtype=torch.float32)
        u, s, vh = torch.linalg.svd(grad, full_matrices=False)
        rank = min(self.active_rank, s.shape[0])
        s = s[:rank] / max(self.lora_scaler, 1e-8)
        sqrt_s = torch.sqrt(torch.clamp(s, min=1e-12))
        weight_a = torch.diag(sqrt_s) @ vh[:rank]
        weight_b = u[:, :rank] @ torch.diag(sqrt_s)
        self.weight_a.data.copy_(weight_a.to(self.weight_a.dtype))
        self.weight_b.data.copy_(weight_b.to(self.weight_b.dtype))

    def grad_compress_init(self, lr: float, stable_gamma: float) -> None:
        grad = self.averaged_gradient()
        if self.active_rank <= 0 or grad is None:
            return

        device = self.weight.device
        grad = grad.to(device=device, dtype=torch.float32)
        if self.weight_a_init_method == "weight_svd":
            weight = self.weight.detach().to(torch.float32)
            _, s, vh = torch.linalg.svd(weight, full_matrices=False)
            rank = min(self.active_rank, s.shape[0])
            seed_a = torch.diag(s[:rank]) @ vh[:rank]
            self.weight_a.data.copy_(seed_a.to(self.weight_a.dtype))
        elif self.weight_a_init_method == "grad_svd":
            _, s, vh = torch.linalg.svd(grad, full_matrices=False)
            rank = min(self.active_rank, s.shape[0])
            seed_a = torch.diag(s[:rank]) @ vh[:rank]
            self.weight_a.data.copy_(seed_a.to(self.weight_a.dtype))
        else:
            self._init_weight("weight_a", self.weight_a_init_method)

        a = self.weight_a.detach().to(torch.float32)
        ata = a @ a.T
        ata_inv = torch.linalg.pinv(
            ata + 1e-8 * torch.eye(ata.shape[0], device=device, dtype=torch.float32))
        a_pinv_right = a.T @ ata_inv
        weight_b = grad @ a_pinv_right
        scale = stable_gamma / max(float(self.alpha), 1e-8)
        if lr > 0:
            scale *= lr
        self.weight_b.data.copy_((weight_b * scale).to(self.weight_b.dtype))

        approx = self._compute_lora_weight().to(torch.float32)
        target = (-grad * lr) if lr > 0 else grad
        reconstruction_error = torch.norm(target - approx, p="fro")
        relative_error = reconstruction_error / \
            torch.clamp(torch.norm(target, p="fro"), min=1e-8)
        self.error.copy_(reconstruction_error.to(self.error.dtype))
        self.relative_error.copy_(relative_error.to(self.relative_error.dtype))

    def dynamic_init(self, rank: int, lr: float, stable_gamma: float) -> None:
        self.allocate_rank(rank)
        if self.active_rank <= 0:
            return
        if self.init_method == "weight_svd":
            self.weight_svd_init()
        elif self.init_method == "grad_svd":
            self.grad_svd_init()
        elif self.init_method == "compress":
            self.grad_compress_init(lr=lr, stable_gamma=stable_gamma)
        else:
            self.init_lora_weights()
        self.grad_sum = None
        self.grad_steps = 0


def _resolve_target_modules(
    model: nn.Module,
    peft_config: Mapping[str, Any],
) -> tuple[List[str], List[str], str]:
    explicit_target_modules = peft_config.get("target_modules")
    if explicit_target_modules:
        resolved_target_modules = [str(item)
                                   for item in explicit_target_modules]
        matched_module_names = []
        for name, _ in model.named_modules():
            if any(name.endswith(f".{leaf_name}") or name == leaf_name for leaf_name in resolved_target_modules):
                matched_module_names.append(name)
        return resolved_target_modules, matched_module_names, "explicit"

    strategy = str(peft_config.get("target_modules_strategy", "auto")).lower()
    if strategy != "auto":
        raise ValueError(f"Unsupported target_modules_strategy: {strategy}")
    resolved_target_modules, matched_module_names = discover_lora_target_modules(
        model)
    return resolved_target_modules, matched_module_names, "auto"


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _full_name_matches(full_name: str, target_modules: Sequence[str]) -> bool:
    return any(full_name == target or full_name.endswith(f".{target}") for target in target_modules)


def _build_gora_resolved_config(peft_config: Mapping[str, Any]) -> GoRAResolvedConfig:
    return GoRAResolvedConfig(
        rank=int(peft_config.get("r", 16)),
        alpha=int(peft_config.get("lora_alpha", 32)),
        dropout=float(peft_config.get("lora_dropout", 0.0)),
        init_method=str(peft_config.get("gora_init_method", "weight_svd")),
        weight_a_init_method=peft_config.get("weight_a_init_method"),
        weight_b_init_method=peft_config.get("weight_b_init_method"),
        run_in_fp32=bool(peft_config.get("run_lora_in_fp32", False)),
        rank_stabilize=bool(peft_config.get("gora_rank_stablize", False)),
        dynamic_scaling=bool(peft_config.get("gora_dynamic_scaling", False)),
        importance_type=str(peft_config.get(
            "gora_importance_type", "union_frobenius_norm")),
        scale_importance=bool(peft_config.get("gora_scale_importance", False)),
        temperature=float(peft_config.get("gora_temperature", 0.5)),
        softmax_importance=bool(peft_config.get(
            "gora_softmax_importance", False)),
        allocate_strategy=str(peft_config.get(
            "gora_allocate_stretagy", "moderate")),
        features_func=peft_config.get("gora_features_func"),
        min_rank=int(peft_config.get("gora_min_rank", 1)),
        max_rank=(None if peft_config.get("gora_max_rank")
                  is None else int(peft_config.get("gora_max_rank"))),
        stable_gamma=float(peft_config.get("gora_stable_gemma", 0.02)),
        lr=float(peft_config.get("gora_lr", 1e-3)),
        gradient_estimation_steps=int(
            peft_config.get("gradient_estimation_steps", 8)),
        rank_pattern=(
            {str(k): int(v)
             for k, v in dict(peft_config.get("rank_pattern")).items()}
            if peft_config.get("rank_pattern")
            else None
        ),
    )


def _apply_gora_to_model(
    model: nn.Module,
    peft_config: Mapping[str, Any],
    resolved_target_modules: Sequence[str],
    matched_module_names: Sequence[str],
    target_modules_source: str,
) -> tuple[nn.Module, Dict[str, Any]]:
    gora_config = _build_gora_resolved_config(peft_config)
    rank_pattern = gora_config.rank_pattern or {}
    applied_names: List[str] = []

    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _full_name_matches(full_name, resolved_target_modules):
            continue
        initial_rank = int(rank_pattern.get(full_name, 0))
        gora_module = GoRALinear(
            module, config=gora_config, initial_rank=initial_rank)
        _replace_module(model, full_name, gora_module)
        applied_names.append(full_name)

    metadata = build_lora_metadata(
        peft_config=peft_config,
        resolved_target_modules=resolved_target_modules,
        matched_module_names=matched_module_names or applied_names,
        model=model,
        target_modules_source=target_modules_source,
    )
    metadata.update(
        {
            "gora_init_method": gora_config.init_method,
            "gora_rank_pattern": dict(rank_pattern),
            "gradient_estimation_steps": gora_config.gradient_estimation_steps,
        }
    )
    return model, metadata


def apply_peft_to_model(
    model: torch.nn.Module,
    peft_config: Mapping[str, Any],
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    if not peft_config.get("enabled", False):
        return model, {"enabled": False, **summarize_parameter_efficiency(model)}

    method = str(peft_config.get("method", "lora")).lower()
    resolved_target_modules, matched_module_names, target_modules_source = _resolve_target_modules(
        model, peft_config)
    if not resolved_target_modules:
        raise ValueError("No PEFT target modules were resolved for the model.")

    if method == "lora":
        lora_config = LoraConfig(
            task_type=getattr(TaskType, str(
                peft_config.get("task_type", "CAUSAL_LM")).upper()),
            r=int(peft_config.get("r", 16)),
            lora_alpha=int(peft_config.get("lora_alpha", 32)),
            lora_dropout=float(peft_config.get("lora_dropout", 0.05)),
            bias=str(peft_config.get("bias", "none")),
            target_modules=list(resolved_target_modules),
            modules_to_save=_normalize_modules_to_save(
                peft_config.get("modules_to_save")),
        )
        wrapped_model = get_peft_model(model, lora_config)
        if not bool(peft_config.get("run_lora_in_fp32", False)):
            base_model_dtype = next(model.parameters()).dtype
            for param in wrapped_model.parameters():
                if getattr(param, "requires_grad", False) and param.dtype != base_model_dtype:
                    param.data = param.data.to(base_model_dtype)
        metadata = build_lora_metadata(
            peft_config=peft_config,
            resolved_target_modules=resolved_target_modules,
            matched_module_names=matched_module_names,
            model=wrapped_model,
            target_modules_source=target_modules_source,
        )
        return wrapped_model, metadata

    if method == "gora":
        return _apply_gora_to_model(
            model=model,
            peft_config=peft_config,
            resolved_target_modules=resolved_target_modules,
            matched_module_names=matched_module_names,
            target_modules_source=target_modules_source,
        )

    raise ValueError(f"Unsupported PEFT method: {method}")


def _extract_loss(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise ValueError("Model outputs do not contain a loss tensor.")


def _compute_gora_importance(
    module: GoRALinear,
    importance_type: str,
    scale_importance: bool,
) -> float | tuple[float, float]:
    grad = module.averaged_gradient()
    if grad is None:
        return 0.0

    weight = module.weight.detach().to(torch.float32)
    grad = grad.to(dtype=torch.float32, device=weight.device)
    param_grad = weight * grad

    def maybe_scale(value: float | tuple[float, float]) -> float | tuple[float, float]:
        if not scale_importance:
            return value
        if isinstance(value, tuple):
            return tuple(math.sqrt(max(v, 0.0)) for v in value)
        return math.sqrt(max(value, 0.0))

    if importance_type == "union_frobenius_norm":
        return maybe_scale(float(torch.linalg.matrix_norm(param_grad).item()))
    if importance_type == "union_2ord_norm":
        return maybe_scale(float(torch.mean(torch.linalg.norm(param_grad, dim=1)).item()))
    if importance_type == "union_mean":
        return maybe_scale(float(torch.mean(torch.abs(param_grad)).item()))
    if importance_type == "union_nuc_norm":
        return maybe_scale(float(torch.linalg.matrix_norm(param_grad, ord="nuc").item()))
    if importance_type == "grad_nuc_norm":
        return maybe_scale(float(torch.linalg.matrix_norm(grad, ord="nuc").item()))
    if importance_type == "grad_frobenius_norm":
        return maybe_scale(float(torch.linalg.matrix_norm(grad).item()))
    if importance_type == "grad_mean":
        return maybe_scale(float(torch.mean(torch.abs(grad)).item()))
    if importance_type == "grad_entropy":
        flat = grad.flatten()
        sigma = torch.clamp(torch.std(flat), min=1e-8)
        entropy = torch.log(
            sigma) + 0.5 * (torch.log(torch.tensor(2 * torch.pi, device=grad.device)) + 1)
        return maybe_scale(float(entropy.item()))
    if importance_type == "union_mean_grad_nuc_norm":
        return maybe_scale(
            (
                float(torch.mean(torch.abs(param_grad)).item()),
                float(torch.linalg.matrix_norm(grad, ord="nuc").item()),
            )
        )
    if importance_type == "union_mean_union_nuc_norm":
        return maybe_scale(
            (
                float(torch.mean(torch.abs(param_grad)).item()),
                float(torch.linalg.matrix_norm(param_grad, ord="nuc").item()),
            )
        )
    if importance_type == "grad_mean_grad_nuc_norm":
        return maybe_scale(
            (
                float(torch.mean(torch.abs(grad)).item()),
                float(torch.linalg.matrix_norm(grad, ord="nuc").item()),
            )
        )
    raise ValueError(f"Unsupported GoRA importance type: {importance_type}")


def _normalize_importances(importances: Sequence[float], softmax_importance: bool, temperature: float) -> torch.Tensor:
    tensor = torch.tensor(importances, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor
    if softmax_importance:
        denom = torch.clamp(tensor.max() - tensor.min(), min=1e-8)
        return torch.softmax((tensor - tensor.min()) / denom / max(temperature, 1e-8), dim=0)
    total = torch.clamp(tensor.sum(), min=1e-8)
    return tensor / total


def _feature_adjuster(name: str | None) -> callable:
    if name == "sqrt":
        return math.sqrt
    if name == "log1p":
        return math.log1p
    return lambda x: x


def _allocate_gora_ranks(
    model: nn.Module,
    peft_config: Mapping[str, Any],
) -> tuple[Dict[str, int], Dict[str, float | tuple[float, float]], int, int]:
    gora_config = _build_gora_resolved_config(peft_config)
    target_modules = [(name, module) for name, module in model.named_modules(
    ) if isinstance(module, GoRALinear)]
    if not target_modules:
        return {}, {}, 0, 0

    allocate_func = {
        "radical": math.ceil,
        "moderate": round,
        "conserved": math.floor,
    }.get(gora_config.allocate_strategy, round)
    feature_adjust_func = _feature_adjuster(gora_config.features_func)

    named_importances: Dict[str, float | tuple[float, float]] = {}
    named_features: Dict[str, int] = {}
    named_smooth_features: Dict[str, float] = {}
    total_budget = 0
    smooth_total_budget = 0.0

    for name, module in target_modules:
        importance = _compute_gora_importance(
            module=module,
            importance_type=gora_config.importance_type,
            scale_importance=gora_config.scale_importance,
        )
        module.importance_score = importance
        features = module.in_features + module.out_features
        named_importances[name] = importance
        named_features[name] = features
        adjusted_features = float(feature_adjust_func(features))
        named_smooth_features[name] = adjusted_features
        total_budget += features * gora_config.rank
        smooth_total_budget += adjusted_features * gora_config.rank

    first_component: List[float] = []
    second_component: List[float] = []
    has_tuple = any(isinstance(value, tuple)
                    for value in named_importances.values())
    if has_tuple:
        for value in named_importances.values():
            if isinstance(value, tuple):
                first_component.append(float(value[0]))
                second_component.append(float(value[1]))
            else:
                first_component.append(float(value))
                second_component.append(float(value))
        normalized = 0.5 * _normalize_importances(
            first_component,
            softmax_importance=gora_config.softmax_importance,
            temperature=gora_config.temperature,
        ) + 0.5 * _normalize_importances(
            second_component,
            softmax_importance=gora_config.softmax_importance,
            temperature=gora_config.temperature,
        )
    else:
        normalized = _normalize_importances(
            [float(value) for value in named_importances.values()],
            softmax_importance=gora_config.softmax_importance,
            temperature=gora_config.temperature,
        )

    named_ranks: Dict[str, int] = {}
    actual_trainable = 0
    for name, normalized_importance in zip(named_importances.keys(), normalized):
        smooth_trainable = allocate_func(
            smooth_total_budget * float(normalized_importance.item()))
        smooth_features = max(named_smooth_features[name], 1.0)
        rank = int(smooth_trainable // smooth_features)
        max_rank = min(
            named_features[name], gora_config.max_rank) if gora_config.max_rank is not None else named_features[name]
        rank = min(max(rank, gora_config.min_rank), max_rank)
        named_ranks[name] = rank
        actual_trainable += rank * named_features[name]

    return named_ranks, named_importances, total_budget, actual_trainable


def initialize_gora(
    model: nn.Module,
    train_dataloader: Iterable[Any],
    device: torch.device,
    peft_config: Mapping[str, Any],
) -> Dict[str, Any]:
    gora_config = _build_gora_resolved_config(peft_config)
    if gora_config.rank_pattern:
        return {
            "rank_pattern": dict(gora_config.rank_pattern),
            "gradient_estimation_steps": gora_config.gradient_estimation_steps,
        }

    target_modules = [(name, module) for name, module in model.named_modules(
    ) if isinstance(module, GoRALinear)]
    if not target_modules:
        return {"rank_pattern": {}, "gradient_estimation_steps": gora_config.gradient_estimation_steps}

    was_training = model.training
    model.train()

    for param in model.parameters():
        param.requires_grad = False
        param.grad = None

    for _, module in target_modules:
        module.reset_gradient_stats()
        module.weight.requires_grad = True

    for step, batch in enumerate(train_dataloader):
        if step >= gora_config.gradient_estimation_steps:
            break
        batch = move_batch_to_device(batch, device)
        loss = _extract_loss(model(**batch))
        loss.backward()
        for _, module in target_modules:
            module.accumulate_gradient()
            module.weight.grad = None
        model.zero_grad(set_to_none=True)

    named_ranks, named_importances, total_budget, actual_trainable = _allocate_gora_ranks(
        model, peft_config)
    for name, module in target_modules:
        module.weight.requires_grad = False
        module.dynamic_init(
            rank=named_ranks.get(name, 0),
            lr=gora_config.lr,
            stable_gamma=gora_config.stable_gamma,
        )

    for param in model.parameters():
        param.grad = None

    if not was_training:
        model.eval()

    metadata = {
        "rank_pattern": named_ranks,
        "importance_scores": {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in named_importances.items()
        },
        "gradient_estimation_steps": gora_config.gradient_estimation_steps,
        "total_budget": total_budget,
        "actual_trainable": actual_trainable,
    }
    metadata.update(summarize_parameter_efficiency(model))
    return metadata
