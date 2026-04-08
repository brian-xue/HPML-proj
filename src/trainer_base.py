from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from src.checkpoint import get_latest_checkpoint_dir, load_checkpoint, maybe_save_best_checkpoint, save_checkpoint
from src.evaluator import evaluate_generation
from src.metrics import RuntimeTracker
from src.utils import format_metrics, move_batch_to_device


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    micro_step: int = 0
    epoch_step: int = 0
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "epoch_step": self.epoch_step,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainerState":
        return cls(
            epoch=int(data.get("epoch", 0)),
            global_step=int(data.get("global_step", 0)),
            micro_step=int(data.get("micro_step", data.get("global_step", 0))),
            epoch_step=int(data.get("epoch_step", 0)),
            best_metric_name=data.get("best_metric_name"),
            best_metric_value=data.get("best_metric_value"),
        )


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
        self._grad_accum_counter = 0
        self._log_interval_tokens = 0
        self._log_interval_step_time_s = 0.0

    @property
    def training_config(self) -> Mapping[str, Any]:
        return self.config.get("training", {})

    @property
    def checkpoint_config(self) -> Mapping[str, Any]:
        return self.config.get("checkpoint", {})

    def compute_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**batch)
        return outputs.loss

    def load_state(self, metadata: Mapping[str, Any]) -> None:
        self.state = TrainerState.from_dict(metadata)
        self._grad_accum_counter = 0

    def resume_from_latest_checkpoint(self) -> Dict[str, Any]:
        checkpoint_dir = get_latest_checkpoint_dir(
            self.output_dir,
            dir_name=self.checkpoint_config.get("dir_name", "checkpoints"),
        )
        metadata = load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.load_state(metadata)
        self.logger.info("Resumed from checkpoint %s", checkpoint_dir)
        return metadata

    def _perform_optimizer_step(self) -> None:
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._grad_accum_counter = 0
        self.state.global_step += 1

    def training_step(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = move_batch_to_device(batch, self.device)

        gradient_accumulation_steps = int(self.training_config.get("gradient_accumulation_steps", 1))
        loss = self.compute_loss(batch)
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()
        self._grad_accum_counter += 1
        self.state.micro_step += 1
        self.state.epoch_step += 1

        should_step = self._grad_accum_counter >= gradient_accumulation_steps
        if should_step:
            self._perform_optimizer_step()

        return {
            "loss": float(loss.detach().item()),
            "optimizer_stepped": float(1 if should_step else 0),
        }

    def should_stop(self) -> bool:
        max_steps = self.training_config.get("max_steps")
        return max_steps is not None and self.state.global_step >= int(max_steps)

    def should_log(self) -> bool:
        log_every = int(self.training_config.get("log_every_steps", 10))
        return log_every > 0 and self.state.global_step > 0 and self.state.global_step % log_every == 0

    def should_evaluate(self) -> bool:
        eval_every = int(self.training_config.get("eval_every_steps", 0))
        return eval_every > 0 and self.state.global_step > 0 and self.state.global_step % eval_every == 0

    def should_checkpoint(self) -> bool:
        save_every = int(self.training_config.get("save_every_steps", 0))
        return save_every > 0 and self.state.global_step > 0 and self.state.global_step % save_every == 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        epoch_loss = 0.0
        steps_in_epoch = 0
        skipped_batches = self.state.epoch_step if self.state.epoch == epoch else 0

        for batch_index, batch in enumerate(self.train_dataloader):
            if skipped_batches and batch_index < skipped_batches:
                continue
            if self.should_stop():
                break

            batch_size = int(batch["input_ids"].shape[0])
            token_count = int(batch["attention_mask"].sum().item())

            self.runtime.start_step()
            step_metrics = self.training_step(batch)
            step_time_s = self.runtime.end_step(samples=batch_size, tokens=token_count)
            self._log_interval_tokens += token_count
            self._log_interval_step_time_s += step_time_s

            epoch_loss += step_metrics["loss"]
            steps_in_epoch += 1

            if step_metrics["optimizer_stepped"] and self.should_log():
                interval_tok_s = (
                    self._log_interval_tokens / self._log_interval_step_time_s
                    if self._log_interval_step_time_s > 0
                    else 0.0
                )
                gpu_mem = None
                if self.device.type == "cuda" and torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    reserved_mem = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                    max_gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    self.logger.info(
                        "epoch=%s step=%s %s gpu_mem=%.2fMB reserved_mem=%.2fMB max_gpu_mem=%.2fMB",
                        epoch,
                        self.state.global_step,
                        format_metrics(
                            {
                                "loss": step_metrics["loss"],
                                "avg_step_time_s": self.runtime.average_step_time_seconds,
                                "tok_s": interval_tok_s,
                                "gpu_mem": gpu_mem if gpu_mem is not None else 0.0,
                                "reserved_mem": reserved_mem if reserved_mem is not None else 0.0,
                                "max_gpu_mem": max_gpu_mem if max_gpu_mem is not None else 0.0,
                            }
                        ),
                    )
                else:
                    self.logger.info(
                        "epoch=%s step=%s %s",
                        epoch,
                        self.state.global_step,
                        format_metrics(
                            {
                                "loss": step_metrics["loss"],
                                "avg_step_time_s": self.runtime.average_step_time_seconds,
                                "tok_s": interval_tok_s,
                            }
                        ),
                    )

                self._log_interval_tokens = 0
                self._log_interval_step_time_s = 0.0

            if step_metrics["optimizer_stepped"] and self.should_evaluate():
                eval_results = self.evaluate()
                metrics = eval_results.get("metrics", {})
                self.logger.info("step=%s eval=%s", self.state.global_step, format_metrics(metrics))
                if self.should_checkpoint():
                    self.maybe_checkpoint(metrics)
            elif step_metrics["optimizer_stepped"] and self.should_checkpoint():
                self.maybe_checkpoint()

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
        metric_value = None
        should_update_best = False

        if eval_metrics and metric_name in eval_metrics:
            metric_value = float(eval_metrics[metric_name])
            if self.state.best_metric_value is None:
                should_update_best = True
            elif metric_mode == "max":
                should_update_best = metric_value > float(self.state.best_metric_value)
            else:
                should_update_best = metric_value < float(self.state.best_metric_value)

            if should_update_best:
                self.state.best_metric_name = metric_name
                self.state.best_metric_value = metric_value

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

        if metric_value is None:
            return checkpoint_dir

        best_dir = None
        if should_update_best:
            best_dir = maybe_save_best_checkpoint(
                output_dir=self.output_dir,
                checkpoint_dir=checkpoint_dir,
                metric_name=metric_name,
                metric_value=metric_value,
                mode=metric_mode,
                dir_name=dir_name,
            )
        return checkpoint_dir

    def train(self) -> Dict[str, Any]:
        num_epochs = int(self.training_config.get("num_epochs", 1))
        self.optimizer.zero_grad(set_to_none=True)
        self.runtime.start_run()
        final_results: Dict[str, Any] = {"train": {}, "eval": {}}

        try:
            start_epoch = int(self.state.epoch)
            for epoch in range(start_epoch, num_epochs):
                self.state.epoch = epoch
                train_metrics = self.train_epoch(epoch)
                final_results["train"] = train_metrics

                if self._grad_accum_counter > 0 and not self.should_stop():
                    self._perform_optimizer_step()

                eval_results = self.evaluate() if self.eval_dataset is not None else {"metrics": {}}
                final_results["eval"] = eval_results
                if self.training_config.get("save_at_end", True) or self.should_checkpoint():
                    self.maybe_checkpoint(eval_results.get("metrics", {}))

                summary = {
                    **train_metrics,
                    **eval_results.get("metrics", {}),
                }
                self.logger.info("epoch=%s summary=%s", epoch, format_metrics(summary))
                self.state.epoch_step = 0
                if self.should_stop():
                    break
        finally:
            self.runtime.stop_run()

        final_results["runtime"] = self.runtime.summary()
        final_results["state"] = self.state.as_dict()
        return final_results
