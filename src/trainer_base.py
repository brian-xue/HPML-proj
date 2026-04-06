from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from src.checkpoint import maybe_save_best_checkpoint, save_checkpoint
from src.evaluator import evaluate_generation
from src.metrics import RuntimeTracker
from src.utils import format_metrics, move_batch_to_device


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
        }


class BaseTrainer:
    """Shared trainer core for future single-device and distributed trainers."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        train_dataloader: Any,
        eval_dataset: Optional[Any],
        device: torch.device,
        config: Mapping[str, Any],
        output_dir: str | Path,
        logger: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.eval_dataset = eval_dataset
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.state = TrainerState()
        self.runtime = RuntimeTracker(device=device)

    @property
    def training_config(self) -> Mapping[str, Any]:
        return self.config.get("training", {})

    @property
    def checkpoint_config(self) -> Mapping[str, Any]:
        return self.config.get("checkpoint", {})

    def compute_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**batch)
        return outputs.loss

    def training_step(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = move_batch_to_device(batch, self.device)

        gradient_accumulation_steps = int(self.training_config.get("gradient_accumulation_steps", 1))
        loss = self.compute_loss(batch)
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        should_step = (self.state.global_step + 1) % gradient_accumulation_steps == 0
        if should_step:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": float(loss.detach().item()),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        epoch_loss = 0.0
        steps_in_epoch = 0
        max_steps = self.training_config.get("max_steps")

        for batch in self.train_dataloader:
            if max_steps is not None and self.state.global_step >= int(max_steps):
                break

            batch_size = int(batch["input_ids"].shape[0])
            token_count = int(batch["attention_mask"].sum().item())

            self.runtime.start_step()
            step_metrics = self.training_step(batch)
            self.runtime.end_step(samples=batch_size, tokens=token_count)

            self.state.global_step += 1
            epoch_loss += step_metrics["loss"]
            steps_in_epoch += 1

            log_every = int(self.training_config.get("log_every_steps", 1))
            if log_every > 0 and self.state.global_step % log_every == 0:
                self.logger.info(
                    "epoch=%s step=%s %s",
                    epoch,
                    self.state.global_step,
                    format_metrics(
                        {
                            "loss": step_metrics["loss"],
                            "avg_step_time_s": self.runtime.average_step_time_seconds,
                        }
                    ),
                )

        average_loss = epoch_loss / steps_in_epoch if steps_in_epoch else 0.0
        return {
            "train_loss": average_loss,
            "train_steps": steps_in_epoch,
        }

    def evaluate(self) -> Dict[str, Any]:
        if self.eval_dataset is None:
            return {"metrics": {}}
        return evaluate_generation(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.eval_dataset,
            config=self.config,
            device=self.device,
        )

    def maybe_checkpoint(self, eval_metrics: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
        metric_name = self.checkpoint_config.get("metric_name", "accuracy")
        metric_mode = self.checkpoint_config.get("metric_mode", "max")
        dir_name = self.checkpoint_config.get("dir_name", "checkpoints")

        checkpoint_dir = save_checkpoint(
            output_dir=self.output_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            metadata=self.state.as_dict(),
            dir_name=dir_name,
        )

        if not eval_metrics or metric_name not in eval_metrics:
            return checkpoint_dir

        metric_value = float(eval_metrics[metric_name])
        best_dir = maybe_save_best_checkpoint(
            output_dir=self.output_dir,
            checkpoint_dir=checkpoint_dir,
            metric_name=metric_name,
            metric_value=metric_value,
            mode=metric_mode,
            dir_name=dir_name,
        )
        if best_dir is not None:
            self.state.best_metric_name = metric_name
            self.state.best_metric_value = metric_value
        return checkpoint_dir

    def train(self) -> Dict[str, Any]:
        num_epochs = int(self.training_config.get("num_epochs", 1))
        self.optimizer.zero_grad(set_to_none=True)
        self.runtime.start_run()
        final_results: Dict[str, Any] = {"train": {}, "eval": {}}

        try:
            for epoch in range(num_epochs):
                self.state.epoch = epoch
                train_metrics = self.train_epoch(epoch)
                final_results["train"] = train_metrics

                eval_results = self.evaluate() if self.eval_dataset is not None else {"metrics": {}}
                final_results["eval"] = eval_results
                self.maybe_checkpoint(eval_results.get("metrics", {}))

                summary = {
                    **train_metrics,
                    **eval_results.get("metrics", {}),
                }
                self.logger.info("epoch=%s summary=%s", epoch, format_metrics(summary))
        finally:
            self.runtime.stop_run()

        final_results["runtime"] = self.runtime.summary()
        final_results["state"] = self.state.as_dict()
        return final_results
