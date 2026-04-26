from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.utils import (
    count_parameters,
    deep_update,
    default_config,
    format_metrics,
    load_json,
    list_versioned_run_dirs,
    load_config,
    make_versioned_output_dir,
    resolve_versioned_run_dir,
    sanitize_for_path,
    save_config,
    save_json,
    set_random_seed,
    setup_logger,
)
from src.distributed import (
    barrier,
    broadcast_object,
    destroy_distributed,
    get_local_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentNotImplementedError(RuntimeError):
    pass


def _read_results(run_dir, results_filename):
    results_path = Path(run_dir) / str(results_filename)
    if not results_path.exists():
        return None
    try:
        return load_json(results_path)
    except Exception:
        return None


def _has_resume_artifacts(run_dir):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    latest_checkpoint_path = run_dir / "checkpoints" / "latest_checkpoint.json"
    return config_path.exists() and latest_checkpoint_path.exists()


def is_completed_run(run_dir, results_filename="final_results.json"):
    results = _read_results(run_dir, results_filename)
    if not isinstance(results, dict):
        return False
    if str(results.get("status", "")).lower() == "completed":
        return True
    return True


def resolve_auto_run_state(output_root, experiment_name, results_filename="final_results.json"):
    experiment_root = Path(output_root) / sanitize_for_path(experiment_name)
    latest_run_dir = list_versioned_run_dirs(experiment_root)
    if not latest_run_dir:
        return "new", None

    latest_run_dir = latest_run_dir[-1]
    if is_completed_run(latest_run_dir, results_filename=results_filename):
        return "skip", latest_run_dir
    if not _has_resume_artifacts(latest_run_dir):
        return "new", None
    return "resume", latest_run_dir


def load_effective_config(base_config_path, device_config_path=None, overrides=None):
    config = load_config(base_config_path, default=default_config())
    if device_config_path:
        config = load_config(device_config_path, default=config)
    if overrides:
        config = deep_update(config, deepcopy(dict(overrides)))
    return config


def _next_versioned_run_dir(output_root, experiment_name):
    experiment_root = Path(output_root) / sanitize_for_path(experiment_name)
    versioned = list_versioned_run_dirs(experiment_root)
    if not versioned:
        return experiment_root / "v001"
    latest_version = int(versioned[-1].name.lstrip("v"))
    return experiment_root / f"v{latest_version + 1:03d}"


def resolve_run_dir(output_root, experiment_name, mode, resume_version=None, dry_run=False):
    mode = str(mode or "new").lower()
    if mode == "new":
        if dry_run:
            return _next_versioned_run_dir(output_root, experiment_name)
        return make_versioned_output_dir(output_root, experiment_name)
    if mode == "resume":
        return resolve_versioned_run_dir(output_root, experiment_name, resume_version)
    if mode == "auto":
        resolved_mode, resolved_run_dir = resolve_auto_run_state(
            output_root, experiment_name)
        if resolved_mode == "new":
            if dry_run:
                return _next_versioned_run_dir(output_root, experiment_name)
            return make_versioned_output_dir(output_root, experiment_name)
        if resolved_run_dir is None:
            raise FileNotFoundError(
                f"Unable to resolve run dir for experiment: {experiment_name}")
        return resolved_run_dir
    raise ValueError(f"Unsupported run mode: {mode}")


def _validate_supported_features(config, experiment_name):
    peft_cfg = config.get("peft", {}) or {}
    if peft_cfg.get("enabled", False):
        method = str(peft_cfg.get("method", "lora")).lower()
        if method != "lora":
            raise ExperimentNotImplementedError(
                f"Experiment '{experiment_name}' requests PEFT method '{
                    method}', but only 'lora' is implemented."
            )

    # DDP/FSDP are supported; keep this for future methods (pipeline parallel, etc).
    execution_cfg = config.get("execution", {}) or {}
    execution_mode = str(execution_cfg.get("mode", "single_gpu")).lower()
    if execution_mode not in {"single_gpu", "ddp", "fsdp"}:
        raise ExperimentNotImplementedError(
            f"Experiment '{experiment_name}' requests unsupported execution.mode '{
                execution_mode}'."
        )

    training_cfg = config.get("training", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    if training_cfg.get("gradient_checkpointing", False) or runtime_cfg.get("gradient_checkpointing", False):
        raise ExperimentNotImplementedError(
            f"Experiment '{
                experiment_name}' requests gradient checkpointing, but it is not implemented yet."
        )


def _build_optimizer(model, config):
    import torch

    optimizer_config = config.get("optimizer", {}) or {}
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config.get("lr", 2e-5)),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1e-8)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
    )


def _build_scheduler(optimizer, train_dataloader, config):
    from transformers import get_scheduler

    scheduler_config = config.get("scheduler", {}) or {}
    training_config = config.get("training", {}) or {}
    num_epochs = int(training_config.get("num_epochs", 1))
    max_steps = training_config.get("max_steps")
    total_steps = int(max_steps) if max_steps is not None else len(
        train_dataloader) * num_epochs
    return get_scheduler(
        name=scheduler_config.get("name", "linear"),
        optimizer=optimizer,
        num_warmup_steps=int(scheduler_config.get("num_warmup_steps", 0)),
        num_training_steps=total_steps,
    )


def apply_lora_and_persist(model, config, run_dir, logger):
    from src.peft import apply_peft_to_model

    peft_config = deepcopy(config.get("peft", {}) or {})
    model, peft_metadata = apply_peft_to_model(model, peft_config)
    if peft_metadata.get("enabled"):
        config.setdefault("peft", {})
        config["peft"]["resolved_target_modules"] = peft_metadata.get(
            "target_modules")
        config["peft"]["target_modules"] = peft_metadata.get("target_modules")
        config["peft"]["target_modules_source"] = peft_metadata.get(
            "target_modules_source")

        logger.info("PEFT enabled. Target modules source: %s",
                    peft_metadata.get("target_modules_source"))
        logger.info("PEFT target modules: %s",
                    peft_metadata.get("target_modules"))
        save_json(peft_metadata, run_dir / "resolved_peft_config.json")
    return model, peft_metadata


def _load_resume_config(run_dir):
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Resume requested but config not found: {config_path}")
    return load_config(config_path, default=default_config())


def run_experiment(exp, dry_run=False):
    """
    Runs a benchmark experiment described by a plain dict.
    """
    if not isinstance(exp, dict):
        raise ValueError("run_experiment(exp): exp must be a dict")

    name = exp.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Experiment requires a non-empty string 'name'.")
    name = name.strip()

    base_config = exp.get("base_config", "configs/base.yaml")
    device_config = exp.get("device_config")

    run_cfg = exp.get("run") or {}
    mode = str(run_cfg.get("mode", "auto")).lower()
    resume_version = run_cfg.get("resume_version")

    overrides = exp.get("overrides") or {}
    artifacts = exp.get("artifacts") or {}
    results_filename = artifacts.get("results_filename", "final_results.json")
    save_eval_metrics_json = bool(
        artifacts.get("save_eval_metrics_json", False))

    base_config_path = (PROJECT_ROOT / base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"base_config not found: {base_config_path}")
    device_config_path = None
    if device_config:
        device_config_path = (PROJECT_ROOT / device_config).resolve()
        if not device_config_path.exists():
            raise FileNotFoundError(f"device_config not found: {
                                    device_config_path}")

    bootstrap_config = load_effective_config(
        base_config_path=base_config_path,
        device_config_path=device_config_path,
        overrides=overrides if mode != "resume" else None,
    )
    _validate_supported_features(bootstrap_config, name)

    output_root = bootstrap_config.get("output_root", "output")
    execution_cfg = bootstrap_config.get("execution", {}) or {}
    distributed_cfg = bootstrap_config.get("distributed", {}) or {}
    execution_mode = str(execution_cfg.get("mode", "single_gpu")).lower()
    distributed_enabled = bool(distributed_cfg.get(
        "enabled", False)) or execution_mode in {"ddp", "fsdp"}
    launched_world_size = int(get_world_size()) if distributed_enabled else 1

    if distributed_enabled and is_distributed():
        init_distributed(backend=str(
            distributed_cfg.get("backend", "nccl") or "nccl"))

    effective_mode = mode
    if mode == "auto":
        if distributed_enabled and is_distributed():
            if is_main_process():
                auto_mode, auto_run_dir = resolve_auto_run_state(
                    output_root, name, results_filename=results_filename)
                resolved_run_dir = auto_run_dir
                if resolved_run_dir is None:
                    resolved_run_dir = resolve_run_dir(
                        output_root,
                        name,
                        auto_mode,
                        resume_version=resume_version,
                        dry_run=dry_run,
                    )
                payload = (auto_mode, resolved_run_dir)
            else:
                payload = None
            effective_mode, run_dir = broadcast_object(payload, src=0)
            barrier()
        else:
            effective_mode, auto_run_dir = resolve_auto_run_state(
                output_root, name, results_filename=results_filename)
            run_dir = auto_run_dir if auto_run_dir is not None else resolve_run_dir(
                output_root,
                name,
                effective_mode,
                resume_version=resume_version,
                dry_run=dry_run,
            )
    elif distributed_enabled and is_distributed():
        if is_main_process():
            resolved_run_dir = resolve_run_dir(
                output_root,
                name,
                effective_mode,
                resume_version=resume_version,
                dry_run=dry_run,
            )
        else:
            resolved_run_dir = None
        resolved_run_dir = broadcast_object(resolved_run_dir, src=0)
        run_dir = resolved_run_dir
        barrier()
    else:
        run_dir = resolve_run_dir(
            output_root, name, effective_mode, resume_version=resume_version, dry_run=dry_run)

    if mode == "auto" and effective_mode == "skip":
        if distributed_enabled and is_distributed():
            destroy_distributed()
        return Path(run_dir)
    if dry_run:
        if distributed_enabled and is_distributed():
            destroy_distributed()
        return run_dir

    # For resume correctness: load the exact config used to create this run dir.
    config = _load_resume_config(
        run_dir) if effective_mode == "resume" else bootstrap_config
    config["run_name"] = name
    config.setdefault("distributed", {})
    config["distributed"]["world_size"] = launched_world_size
    if distributed_enabled and is_distributed():
        config["distributed"]["local_rank"] = get_local_rank()
    _validate_supported_features(config, name)

    rank_suffix = ""
    filename = "run.log"
    if distributed_enabled and is_distributed():
        import torch.distributed as dist

        rank_suffix = f".rank{dist.get_rank()}"
        if not is_main_process():
            filename = f"run{rank_suffix}.log"
    logger = setup_logger(f"experiment.{name}.{Path(run_dir).name}{
                          rank_suffix}", output_dir=run_dir, filename=filename)
    if exp.get("description"):
        logger.info("Description: %s", exp.get("description"))
    logger.info("Starting experiment '%s' (%s) at %s",
                name, effective_mode, run_dir)

    set_random_seed(int(config.get("seed", 42)))

    from src.model import load_model_and_tokenizer
    from src.trainer_base import BaseTrainer
    from src.trainer_distributed import DistributedTrainer

    if distributed_enabled and is_distributed():
        local_rank = get_local_rank()
        config.setdefault("model", {})
        config["model"]["device"] = f"cuda:{local_rank}"

    model, tokenizer, device = load_model_and_tokenizer(config)
    model, peft_metadata = apply_lora_and_persist(
        model, config, Path(run_dir), logger)

    if is_main_process() or not (distributed_enabled and is_distributed()):
        save_config(config, Path(run_dir) / "config.yaml")
        save_json(
            {
                "name": name,
                "base_config": base_config,
                "device_config": device_config,
                "run": {"mode": effective_mode, "resume_version": resume_version},
                "launch": {"world_size": launched_world_size},
                "overrides": overrides,
                "artifacts": {"results_filename": results_filename, "save_eval_metrics_json": save_eval_metrics_json},
            },
            Path(run_dir) / "experiment.json",
        )
    barrier()

    from src.data import build_dataloaders

    include_eval = bool((config.get("evaluation", {})
                        or {}).get("enabled", True))
    dataloaders = build_dataloaders(
        tokenizer=tokenizer, config=config, include_generation_eval=include_eval)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, dataloaders["train"], config)

    trainer_cls = BaseTrainer
    if distributed_enabled and is_distributed():
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        if execution_mode == "ddp":
            model = DistributedDataParallel(
                model, device_ids=[get_local_rank()])
        elif execution_mode == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            model = FSDP(model, device_id=get_local_rank(),
                         use_orig_params=True)
        trainer_cls = DistributedTrainer

    eval_dataset = None
    if include_eval and "eval_generation" in dataloaders:
        eval_dataset = dataloaders["eval_generation"].dataset

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=dataloaders["train"],
        eval_dataset=eval_dataset,
        device=device,
        config=config,
        output_dir=run_dir,
        logger=logger,
    )
    if trainer_cls is DistributedTrainer:
        trainer_kwargs["train_sampler"] = dataloaders.get("train_sampler")
    trainer = trainer_cls(**trainer_kwargs)

    if effective_mode == "resume":
        trainer.resume_from_latest_checkpoint()

    logger.info("Device: %s", device)
    logger.info("Total parameters before training: %s",
                count_parameters(model))
    results = trainer.train()
    results["peft"] = peft_metadata
    results["launch"] = {"world_size": launched_world_size}

    if is_main_process() or not (distributed_enabled and is_distributed()):
        results_path = Path(run_dir) / str(results_filename)
        save_json(results, results_path)
        if save_eval_metrics_json:
            save_json(results.get("eval", {}), Path(
                run_dir) / "eval_metrics.json")
        logger.info("Finished experiment '%s'. Results saved to %s",
                    name, results_path)
    barrier()
    if distributed_enabled and is_distributed():
        destroy_distributed()
    return Path(run_dir)
