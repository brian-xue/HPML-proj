from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

import torch


def _cuda_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
            "gpu_peak_memory_allocated_mb": 0.0,
        }

    index = (device.index if device is not None and device.index is not None else torch.cuda.current_device())
    allocated = torch.cuda.memory_allocated(index) / (1024**2)
    reserved = torch.cuda.memory_reserved(index) / (1024**2)
    peak = torch.cuda.max_memory_allocated(index) / (1024**2)
    return {
        "gpu_memory_allocated_mb": allocated,
        "gpu_memory_reserved_mb": reserved,
        "gpu_peak_memory_allocated_mb": peak,
    }


class Timer:
    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        if self._start_time is None:
            return self.elapsed
        self.elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return self.elapsed

    @contextmanager
    def track(self) -> Iterator["Timer"]:
        self.start()
        try:
            yield self
        finally:
            self.stop()


@dataclass
class RuntimeTracker:
    device: Optional[torch.device] = None
    total_steps: int = 0
    total_samples: int = 0
    total_tokens: int = 0
    total_runtime_seconds: float = 0.0
    total_step_time_seconds: float = 0.0
    _run_start_time: Optional[float] = field(default=None, init=False, repr=False)
    _step_start_time: Optional[float] = field(default=None, init=False, repr=False)
    _cuda_start_event: Optional[torch.cuda.Event] = field(default=None, init=False, repr=False)
    _cuda_end_event: Optional[torch.cuda.Event] = field(default=None, init=False, repr=False)

    def _use_cuda_events(self) -> bool:
        if not (self.device and torch.device(self.device).type == "cuda"):
            return False
        # Nsight Compute range replay can fail if timing events straddle the
        # capture boundary. During profiler-driven runs, use synchronized wall
        # clock timing instead of CUDA events to keep the capture window clean.
        if os.environ.get("HPML_PROFILE_CUDA", "0") == "1":
            return False
        return True


    def start_run(self) -> None:
        self._run_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def stop_run(self) -> float:
        if self._run_start_time is None:
            return self.total_runtime_seconds
        self.total_runtime_seconds = time.perf_counter() - self._run_start_time
        self._run_start_time = None
        return self.total_runtime_seconds

    def start_step(self) -> None:
        if self._use_cuda_events():
            torch.cuda.synchronize()
            self._cuda_start_event = torch.cuda.Event(enable_timing=True)
            self._cuda_end_event = torch.cuda.Event(enable_timing=True)
            self._cuda_start_event.record()
        elif self.device and torch.device(self.device).type == "cuda":
            torch.cuda.synchronize()
            self._step_start_time = time.perf_counter()
        else:
            self._step_start_time = time.perf_counter()

    def end_step(self, samples: int = 0, tokens: int = 0) -> float:
        if self._use_cuda_events():
            self._cuda_end_event.record()
            torch.cuda.synchronize()
            step_time = self._cuda_start_event.elapsed_time(self._cuda_end_event) / 1000.0
        else:
            if self._step_start_time is None:
                raise RuntimeError("start_step() must be called before end_step().")
            if self.device and torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - self._step_start_time
            self._step_start_time = None
        # ...existing stats update...
        self.total_steps += 1
        self.total_samples += int(samples)
        self.total_tokens += int(tokens)
        self.total_step_time_seconds += step_time
        return step_time

    @property
    def average_step_time_seconds(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_step_time_seconds / self.total_steps

    @property
    def samples_per_second(self) -> float:
        runtime = self.total_runtime_seconds or self.total_step_time_seconds
        if runtime <= 0:
            return 0.0
        return self.total_samples / runtime

    @property
    def tokens_per_second(self) -> float:
        runtime = self.total_runtime_seconds or self.total_step_time_seconds
        if runtime <= 0:
            return 0.0
        return self.total_tokens / runtime

    def summary(self) -> Dict[str, Any]:
        metrics = {
            "total_steps": self.total_steps,
            "total_samples": self.total_samples,
            "total_tokens": self.total_tokens,
            "total_runtime_seconds": self.total_runtime_seconds,
            "total_step_time_seconds": self.total_step_time_seconds,
            "average_step_time_seconds": self.average_step_time_seconds,
            "samples_per_second": self.samples_per_second,
            "tokens_per_second": self.tokens_per_second,
        }
        metrics.update(_cuda_memory_stats(self.device))
        return metrics


@contextmanager
def measure_runtime(device: Optional[torch.device] = None) -> Iterator[RuntimeTracker]:
    tracker = RuntimeTracker(device=device)
    tracker.start_run()
    try:
        yield tracker
    finally:
        tracker.stop_run()
