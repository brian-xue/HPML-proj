from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.utils import (
    count_parameters,
    deep_update,
    default_config,
    format_metrics,
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentNotImplementedError(RuntimeError):
    pass


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
    raise ValueError(f"Unsupported run mode: {mode}")


def _validate_supported_features(config, experiment_name):
    peft_cfg = config.get("peft", {}) or {}
    if peft_cfg.get("enabled", False):
        method = str(peft_cfg.get("method", "lora")).lower()
        if method not in {"lora", "gora"}:
            raise ExperimentNotImplementedError(
                f"Experiment '{experiment_name}' requests PEFT method '{method}', but only 'lora' and 'gora' are implemented."
            )

    distributed_cfg = config.get("distributed", {}) or {}
    execution_cfg = config.get("execution", {}) or {}
    if distributed_cfg.get("enabled", False) or str(execution_cfg.get("mode", "single_gpu")).lower() in {"ddp", "fsdp"}:
        raise ExperimentNotImplementedError(
            f"Experiment '{experiment_name}' requests distributed mode, but DDP/FSDP runners are not implemented yet."
        )

    training_cfg = config.get("training", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    if training_cfg.get("gradient_checkpointing", False) or runtime_cfg.get("gradient_checkpointing", False):
        raise ExperimentNotImplementedError(
            f"Experiment '{experiment_name}' requests gradient checkpointing, but it is not implemented yet."
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
    total_steps = int(max_steps) if max_steps is not None else len(train_dataloader) * num_epochs
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
        config["peft"]["resolved_target_modules"] = peft_metadata.get("target_modules")
        config["peft"]["target_modules"] = peft_metadata.get("target_modules")
        config["peft"]["target_modules_source"] = peft_metadata.get("target_modules_source")
        if peft_metadata.get("gora_rank_pattern") is not None:
            config["peft"]["rank_pattern"] = peft_metadata.get("gora_rank_pattern")

        logger.info("PEFT enabled. Target modules source: %s", peft_metadata.get("target_modules_source"))
        logger.info("PEFT target modules: %s", peft_metadata.get("target_modules"))
        save_json(peft_metadata, run_dir / "resolved_peft_config.json")
    return model, peft_metadata


def _load_resume_config(run_dir):
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Resume requested but config not found: {config_path}")
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
    mode = str(run_cfg.get("mode", "new")).lower()
    resume_version = run_cfg.get("resume_version")

    overrides = exp.get("overrides") or {}
    artifacts = exp.get("artifacts") or {}
    results_filename = artifacts.get("results_filename", "final_results.json")
    save_eval_metrics_json = bool(artifacts.get("save_eval_metrics_json", False))

    base_config_path = (PROJECT_ROOT / base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"base_config not found: {base_config_path}")
    device_config_path = None
    if device_config:
        device_config_path = (PROJECT_ROOT / device_config).resolve()
        if not device_config_path.exists():
            raise FileNotFoundError(f"device_config not found: {device_config_path}")

    bootstrap_config = load_effective_config(
        base_config_path=base_config_path,
        device_config_path=device_config_path,
        overrides=overrides if mode != "resume" else None,
    )
    _validate_supported_features(bootstrap_config, name)

    output_root = bootstrap_config.get("output_root", "output")
    run_dir = resolve_run_dir(output_root, name, mode, resume_version=resume_version, dry_run=dry_run)
    if dry_run:
        return run_dir

    # For resume correctness: load the exact config used to create this run dir.
    config = _load_resume_config(run_dir) if mode == "resume" else bootstrap_config
    config["run_name"] = name
    _validate_supported_features(config, name)

    logger = setup_logger(f"experiment.{name}.{Path(run_dir).name}", output_dir=run_dir)
    if exp.get("description"):
        logger.info("Description: %s", exp.get("description"))
    logger.info("Starting experiment '%s' (%s) at %s", name, mode, run_dir)

    set_random_seed(int(config.get("seed", 42)))

    from src.data import build_dataloaders
    from src.model import load_model_and_tokenizer
    from src.peft import initialize_gora
    from src.trainer_base import BaseTrainer

    model, tokenizer, device = load_model_and_tokenizer(config)
    model, peft_metadata = apply_lora_and_persist(model, config, Path(run_dir), logger)

    save_json(
        {
            "name": name,
            "base_config": base_config,
            "device_config": device_config,
            "run": {"mode": mode, "resume_version": resume_version},
            "overrides": overrides,
            "artifacts": {"results_filename": results_filename, "save_eval_metrics_json": save_eval_metrics_json},
        },
        Path(run_dir) / "experiment.json",
    )

    dataloaders = build_dataloaders(tokenizer=tokenizer, config=config)
    if peft_metadata.get("enabled") and peft_metadata.get("method") == "gora" and not config.get("peft", {}).get("rank_pattern"):
        logger.info(
            "Initializing GoRA with %s gradient-estimation steps.",
            config.get("peft", {}).get("gradient_estimation_steps", 8),
        )
        gora_metadata = initialize_gora(
            model=model,
            train_dataloader=dataloaders["train"],
            device=device,
            peft_config=config["peft"],
        )
        peft_metadata.update(
            {
                "gora_rank_pattern": gora_metadata.get("rank_pattern", {}),
                "gora_importance_scores": gora_metadata.get("importance_scores", {}),
                "gradient_estimation_steps": gora_metadata.get("gradient_estimation_steps"),
                "gora_total_budget": gora_metadata.get("total_budget"),
                "gora_actual_trainable": gora_metadata.get("actual_trainable"),
                "trainable_parameters": gora_metadata.get("trainable_parameters", peft_metadata.get("trainable_parameters")),
                "trainable_ratio": gora_metadata.get("trainable_ratio", peft_metadata.get("trainable_ratio")),
            }
        )
        config.setdefault("peft", {})["rank_pattern"] = gora_metadata.get("rank_pattern", {})
        logger.info("GoRA initialized with rank pattern: %s", config["peft"]["rank_pattern"])
        save_json(peft_metadata, Path(run_dir) / "resolved_peft_config.json")

    save_config(config, Path(run_dir) / "config.yaml")
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, dataloaders["train"], config)

    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=dataloaders["train"],
        eval_dataset=dataloaders["eval_generation"].dataset,
        device=device,
        config=config,
        output_dir=run_dir,
        logger=logger,
    )

    if mode == "resume":
        trainer.resume_from_latest_checkpoint()

    logger.info("Device: %s", device)
    logger.info("Total parameters before training: %s", count_parameters(model))
    results = trainer.train()
    results["peft"] = peft_metadata

    results_path = Path(run_dir) / str(results_filename)
    save_json(results, results_path)
    if save_eval_metrics_json:
        save_json(results.get("eval", {}), Path(run_dir) / "eval_metrics.json")

    logger.info("Finished experiment '%s'. Results saved to %s", name, results_path)
    return Path(run_dir)
