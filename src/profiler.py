from __future__ import annotations

from typing import Any, Mapping

import torch


class CudaProfileWindow:
    def __init__(self, config: Mapping[str, Any], logger: Any = None) -> None:
        profile_cfg = config.get("profile", {}) or {}
        self.enabled = bool(profile_cfg.get("cuda_profiler", False))
        self.warmup_steps = int(profile_cfg.get("warmup_steps", 0) or 0)
        self.active_steps = int(profile_cfg.get("active_steps", 0) or 0)
        self.logger = logger
        self.started = False
        self.finished = False

    def _log(self, message: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.info(message, *args)

    def maybe_start(self, global_step: int) -> None:
        if not self.enabled or self.started or self.finished:
            return
        if self.active_steps <= 0 or not torch.cuda.is_available():
            return
        if int(global_step) < self.warmup_steps:
            return
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        self.started = True
        self._log(
            "CUDA profiler window started at global_step=%s (warmup_steps=%s active_steps=%s)",
            global_step,
            self.warmup_steps,
            self.active_steps,
        )

    def maybe_stop(self, global_step: int) -> None:
        if not self.enabled or not self.started or self.finished:
            return
        if int(global_step) < self.warmup_steps + self.active_steps:
            return
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        self.finished = True
        self._log("CUDA profiler window stopped at global_step=%s", global_step)

    def close(self) -> None:
        if not self.enabled or not self.started or self.finished:
            return
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        self.finished = True
        self._log("CUDA profiler window stopped during cleanup")
