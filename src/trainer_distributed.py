from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from src.trainer_base import BaseTrainer
from src.distributed import (
    all_reduce_max_float,
    all_reduce_sum_float,
    barrier,
    is_main_process,
)


class DistributedTrainer(BaseTrainer):
    def __init__(
        self,
        *args,
        train_sampler: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.train_sampler = train_sampler

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)
        return super().train_epoch(epoch)

    def evaluate(self) -> Dict[str, Any]:
        if not is_main_process():
            return {"metrics": {}}
        return super().evaluate()

    def maybe_checkpoint(self, eval_metrics: Optional[Mapping[str, Any]] = None):
        if not is_main_process():
            return None
        return super().maybe_checkpoint(eval_metrics=eval_metrics)

    def train(self) -> Dict[str, Any]:
        results = super().train()

        # Aggregate runtime counters across ranks for paper-ready global throughput.
        runtime = results.get("runtime", {}) or {}
        device = self.device if getattr(self, "device", None) is not None else torch.device("cuda")
        device_str = str(device) if device.type == "cuda" else "cpu"

        total_tokens = float(runtime.get("total_tokens", 0.0))
        total_samples = float(runtime.get("total_samples", 0.0))
        total_runtime_s = float(runtime.get("total_runtime_seconds", 0.0))
        peak_mem_mb = float(runtime.get("gpu_peak_memory_allocated_mb", 0.0))

        summed_tokens = all_reduce_sum_float(total_tokens, device=device_str if device.type == "cuda" else None)
        summed_samples = all_reduce_sum_float(total_samples, device=device_str if device.type == "cuda" else None)
        max_runtime = all_reduce_max_float(total_runtime_s, device=device_str if device.type == "cuda" else None)
        max_peak_mem = all_reduce_max_float(peak_mem_mb, device=device_str if device.type == "cuda" else None)

        if max_runtime > 0:
            runtime["global_tokens_per_second"] = summed_tokens / max_runtime
            runtime["global_samples_per_second"] = summed_samples / max_runtime
        else:
            runtime["global_tokens_per_second"] = 0.0
            runtime["global_samples_per_second"] = 0.0

        runtime["global_total_tokens"] = summed_tokens
        runtime["global_total_samples"] = summed_samples
        runtime["global_total_runtime_seconds"] = max_runtime
        runtime["global_peak_gpu_memory_allocated_mb_max_rank"] = max_peak_mem

        results["runtime"] = runtime

        barrier()
        return results
