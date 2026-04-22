# Shared LLM Fine-Tuning Core

This project now includes the shared implementation core for GSM8K-style math reasoning fine-tuning and evaluation using `Qwen/Qwen2.5-1.5B-Instruct`, PyTorch, Hugging Face Transformers, and Hugging Face Datasets.

## What Was Implemented

The current work focuses only on the reusable foundation that all later experiment paths can build on.

Included:

- Shared utilities for configuration, logging, seeding, output directories, run naming, and small helper functions
- A reusable GSM8K data pipeline with deterministic splitting, instruction-response formatting, tokenization, dataloaders, and a causal language modeling collator
- Shared model and tokenizer loading with config-driven dtype and device handling
- Lightweight runtime measurement utilities for timing, throughput, and GPU memory tracking
- Shared evaluation logic for batched text generation, answer extraction, exact-match accuracy, and metric aggregation
- Shared checkpoint save/load utilities with metadata handling and best-checkpoint support
- Thin orchestration scripts for data preparation, training setup, evaluation, and result summarization

Not included yet:

- Distributed training logic
- Advanced profiler integrations
- Trainer subclasses for single-device or distributed execution

## Files Added

### `src/utils.py`

Provides lightweight project-wide helpers:

- random seed setup for `random`, `numpy`, `torch`, and CUDA
- logger creation for console and file output
- config load/save helpers for YAML and JSON
- output directory creation
- run name generation from config
- metric formatting
- device, parameter counting, and batch movement helpers
- a default shared config structure

### `src/data.py`

Implements the shared GSM8K dataset pipeline:

- loading GSM8K from Hugging Face
- optional loading from a previously saved disk dataset
- deterministic train/validation split creation when needed
- formatting examples into a Qwen chat-style instruction/response structure
- tokenization with truncation and label masking for causal LM fine-tuning
- PyTorch dataloader builders for both training and generation-based evaluation
- a reusable causal LM collator

### `src/model.py`

Provides model and tokenizer loading:

- tokenizer loading for `Qwen/Qwen2.5-1.5B-Instruct`
- pad token and padding-side handling
- config-driven dtype selection for `fp32`, `fp16`, and `bf16`
- model loading and movement to the configured device
- a single helper to rebuild model, tokenizer, and device together

### `src/metrics.py`

Implements lightweight shared runtime measurements:

- wall-clock timing
- per-step timing and average step time
- total runtime
- samples/sec and tokens/sec
- current allocated GPU memory
- reserved GPU memory
- peak allocated GPU memory

### `src/evaluator.py`

Implements reusable GSM8K evaluation logic:

- batched generation for decoder-only models
- answer extraction from generated text
- exact-match comparison against GSM8K references
- structured metric aggregation
- runtime measurement integration

### `src/checkpoint.py`

Implements shared checkpoint handling:

- saving model state
- saving optimizer state
- saving scheduler state
- saving metadata such as epoch and global step
- loading checkpoints for resume or evaluation
- tracking and copying the best checkpoint by metric
- storing a latest-checkpoint pointer for resumable runs

### `src/peft.py`

Implements shared LoRA support:

- config-driven LoRA wrapping via `peft`
- auto-detection of target projection layers for Qwen-style models
- support for explicit fixed `target_modules`
- resolved target-module reporting for reproducible follow-up runs
- trainable vs total parameter summaries

## Scripts Added

### `scripts/prepare_data.py`

Standalone data preparation script that:

- loads GSM8K
- preprocesses examples into the shared instruction-response format
- optionally tokenizes data
- saves processed outputs under `data/processed`

### `scripts/train.py`

Thin shared training setup script that:

- loads config
- sets the random seed
- creates an output directory
- loads model and tokenizer
- builds dataloaders
- initializes optimizer and scheduler

This script does not implement a training loop. It is only the shared setup layer for future trainer-specific paths.

### `scripts/eval.py`

Standalone evaluation script that:

- loads config
- rebuilds model and tokenizer
- restores a checkpoint
- prepares evaluation data
- runs shared generation-based evaluation
- saves metrics to JSON

### `scripts/eval_pretrained.py`

Standalone pretrained baseline evaluation script that:

- loads the same effective config stack as `experiments/lora_benchmark.py`
- delegates model/dataloader/generation evaluation to `src.evaluator`
- evaluates the base model without requiring a checkpoint
- disables PEFT so the model remains pretrained-only
- uses the shared 4-shot chain-of-thought prompt from `src.data`
- supports quick subset runs with `--max-examples`
- optionally fails when accuracy is below `--min-accuracy`
- saves metrics and per-example predictions to JSON

### `scripts/summarizing.py`

Shared result summarization script that:

- scans experiment directories for saved evaluation metrics
- aggregates them into JSON and CSV
- summarizes accuracy, runtime, throughput, and peak memory

### `scripts/run_lora_benchmark.py`

Main editable benchmark runner for full GSM8K LoRA experiments:

- holds a top-level `EXPERIMENTS` Python list
- supports explicit `new` or `resume` modes per experiment
- creates versioned run directories such as `v001`, `v002`, ...
- applies LoRA and saves the resolved target-layer list
- runs full training, checkpointing, resume, and evaluation

## Design Choices

The implementation was kept intentionally simple and config-driven so later work can add PEFT, distributed execution, and trainer subclasses without major refactoring.

Reusable logic lives in `src/`.
Scripts in `scripts/` stay thin and orchestration-focused.
The default model is consistently `Qwen/Qwen2.5-1.5B-Instruct`.
LoRA target discovery is reproducible because the resolved target list is saved
to each benchmark run and can be copied back into future explicit configs.

## Validation

The implemented files were checked with:

```bash
PYTHONPYCACHEPREFIX=/tmp/python_cache python3 -m compileall src scripts
```

This passed successfully.

## Smoke Test

You can run a lightweight end-to-end smoke test to verify that the shared
pipeline is wired correctly.

### 1. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2. Run the smoke test

```bash
python3 scripts/smoke_test.py
```

This script will:

- create a tiny synthetic GSM8K-style dataset locally
- load a very small Hugging Face causal LM by default
- run a short train pass through `src/trainer_base.py`
- run shared generation-based evaluation
- save outputs under `output/smoke_test`

### 3. Optional overrides

To explicitly choose the tiny model or change the output directory:

```bash
python3 scripts/smoke_test.py \
  --model-name sshleifer/tiny-gpt2 \
  --output-dir output/smoke_test
```

### 4. What to check after it finishes

If the smoke test succeeds, you should see:

- `output/smoke_test/smoke_test_results.json`
- `output/smoke_test/run.log`
- `output/smoke_test/checkpoints/`
- `output/smoke_test/synthetic_gsm8k/`

### Notes

- The smoke test defaults to the project model (see `configs/base.yaml`), but you can
  override `--model-name` to use something smaller (e.g. `sshleifer/tiny-gpt2`) for a quicker wiring check.
- The smoke test still requires network access the first time so Transformers
  can download the model.
- This is only a wiring/integration check, not a meaningful training run.

## Benchmark Runs

Benchmark experiments are intended to be run as standalone Python entrypoints under `experiments/`.

Run the LoRA benchmark:

```bash
python3 experiments/lora_benchmark.py
```

## Pretrained Accuracy Baseline

Evaluate the configured base model before fine-tuning using the LoRA benchmark
eval settings and 4-shot chain-of-thought prompting:

```bash
python3 scripts/eval_pretrained.py --max-examples 100
```

To use it as a pass/fail eval test, add a threshold:

```bash
python3 scripts/eval_pretrained.py --max-examples 100 --min-accuracy 0.10
```

Dry-run (validate and print the resolved output directory without training):

```bash
python3 experiments/lora_benchmark.py --dry-run
```

Each run writes a versioned directory under the experiment name, for example:

- `output/lora_benchmark/v001/`
- `output/lora_benchmark/v002/`

Important run artifacts:

- `config.yaml`
- `resolved_peft_config.json`
- `final_results.json`
- `checkpoints/`
- `run.log`

When LoRA auto-detect mode is used, the resolved LoRA target list is:

- printed to the console and run log
- saved in `resolved_peft_config.json`
- written back into `config.yaml`

That saved `target_modules` list is meant to be copied into future experiments
to lock the setting for reproducible benchmarks.

To resume an unfinished run, set the experiment spec `run.mode: resume` and
optionally set `run.resume_version` to `latest` or a specific `v###`.

### Backwards compatibility

`scripts/run_lora_benchmark.py` still exists as a deprecated wrapper that runs
`experiments/lora_benchmark.py`.

## Next Steps

The codebase is now ready for the next layer of work, such as:

- implementing the actual shared training loop base
- adding trainer subclasses for single-device and distributed paths
- adding additional PEFT methods (QLoRA, GoRA)
- adding more advanced profiling and experiment tracking
