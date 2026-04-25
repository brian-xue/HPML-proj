from __future__ import annotations

import os
from typing import Any


def is_distributed() -> bool:
    # torchrun sets RANK/WORLD_SIZE even for a single process. Treat distributed
    # as "multi-process" to avoid requiring NCCL for nproc_per_node=1 runs.
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    import torch
    import torch.distributed as dist

    if not is_distributed():
        return
    if not dist.is_available() or dist.is_initialized():
        return
    backend = str(backend).lower()
    if backend == "nccl" and not dist.is_nccl_available():
        raise RuntimeError(
            "PyTorch distributed was built without NCCL support. "
            "Install a CUDA/NCCL-enabled PyTorch build, or set distributed.backend=gloo for CPU-only testing."
        )
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but CUDA is not available.")
        torch.cuda.set_device(get_local_rank())
    dist.init_process_group(backend=backend)


def barrier() -> None:
    import torch
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[get_local_rank()])
        else:
            dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return obj
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def all_reduce_sum_float(value: float, device: str | None = None) -> float:
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor(float(value), device=device or "cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_max_float(value: float, device: str | None = None) -> float:
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor(float(value), device=device or "cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())
