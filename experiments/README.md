# Experiments

This project defines benchmark experiments as standalone Python entrypoints under `experiments/`.

```bash
python3 experiments/lora_benchmark.py
```

Note: make sure you run the command with a Python environment that has the project
dependencies installed (e.g., your local venv).

## Common commands

Run the LoRA benchmark:

```bash
python3 experiments/lora_benchmark.py
```

Run the GoRA benchmark:

```bash
python3 experiments/gora_benchmark.py
```

Dry-run (validate and show the resolved output directory without launching training):

```bash
python3 experiments/lora_benchmark.py --dry-run
```

```bash
python3 experiments/gora_benchmark.py --dry-run
```

## Output layout

Runs write to versioned directories under `output/<experiment-name>/v###/`, for example:

- `output/lora_benchmark/v001/config.yaml`
- `output/lora_benchmark/v001/experiment.json`
- `output/lora_benchmark/v001/resolved_peft_config.json` (when PEFT enabled)
- `output/lora_benchmark/v001/final_results.json`
- `output/lora_benchmark/v001/run.log`

## Adding a new experiment

1. Create a new Python file under `experiments/` (e.g., `experiments/qlora_benchmark.py`).
2. Add a small `EXPERIMENT = {...}` dict and call `run_experiment(EXPERIMENT)` from `src/experiment_runner.py`.
3. Run it directly: `python3 experiments/qlora_benchmark.py`.
