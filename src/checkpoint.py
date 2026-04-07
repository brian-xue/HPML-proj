from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from src.utils import ensure_dir, load_json, save_json


def get_checkpoint_root(output_dir: str | Path, dir_name: str = "checkpoints") -> Path:
    return ensure_dir(Path(output_dir) / dir_name)


def get_latest_checkpoint_metadata_path(output_dir: str | Path, dir_name: str = "checkpoints") -> Path:
    return get_checkpoint_root(output_dir, dir_name=dir_name) / "latest_checkpoint.json"


def build_checkpoint_dir(
    output_dir: str | Path,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    dir_name: str = "checkpoints",
) -> Path:
    checkpoint_root = get_checkpoint_root(output_dir, dir_name=dir_name)
    if step is not None:
        return checkpoint_root / f"step_{step:08d}"
    if epoch is not None:
        return checkpoint_root / f"epoch_{epoch:04d}"
    return checkpoint_root / "latest"


def save_checkpoint(
    output_dir: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    dir_name: str = "checkpoints",
) -> Path:
    checkpoint_dir = build_checkpoint_dir(output_dir, step=global_step, epoch=epoch, dir_name=dir_name)
    ensure_dir(checkpoint_dir)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metadata": {
            "epoch": epoch,
            "global_step": global_step,
            **dict(metadata or {}),
        },
    }
    torch.save(state, checkpoint_dir / "training_state.pt")
    save_json(state["metadata"], checkpoint_dir / "metadata.json")
    latest_metadata_path = get_latest_checkpoint_metadata_path(output_dir, dir_name=dir_name)
    save_json({"checkpoint_dir": str(checkpoint_dir.resolve())}, latest_metadata_path)
    if hasattr(model, "save_pretrained"):
        adapter_dir = checkpoint_dir / "adapter"
        try:
            model.save_pretrained(adapter_dir)
        except TypeError:
            pass
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir) / "training_state.pt"
    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model"])

    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    return state.get("metadata", {})


def get_latest_checkpoint_dir(output_dir: str | Path, dir_name: str = "checkpoints") -> Path:
    latest_metadata_path = get_latest_checkpoint_metadata_path(output_dir, dir_name=dir_name)
    if not latest_metadata_path.exists():
        raise FileNotFoundError(f"Latest checkpoint metadata not found: {latest_metadata_path}")
    metadata = load_json(latest_metadata_path)
    checkpoint_dir = Path(metadata["checkpoint_dir"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Latest checkpoint directory does not exist: {checkpoint_dir}")
    return checkpoint_dir


def maybe_save_best_checkpoint(
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    metric_name: str,
    metric_value: float,
    mode: str = "max",
    dir_name: str = "checkpoints",
) -> Optional[Path]:
    checkpoint_root = get_checkpoint_root(output_dir, dir_name=dir_name)
    best_metadata_path = checkpoint_root / "best_checkpoint.json"
    best_dir = checkpoint_root / "best"

    is_better = False
    if best_metadata_path.exists():
        best_metadata = load_json(best_metadata_path)
        best_value = float(best_metadata["metric_value"])
        is_better = metric_value > best_value if mode == "max" else metric_value < best_value
    else:
        is_better = True

    if not is_better:
        return None

    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(checkpoint_dir, best_dir)
    save_json(
        {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "source_checkpoint": str(Path(checkpoint_dir).resolve()),
        },
        best_metadata_path,
    )
    return best_dir
