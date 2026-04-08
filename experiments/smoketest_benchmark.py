from __future__ import annotations

import argparse
import shutil
import sys
from copy import deepcopy
from pathlib import Path

from datasets import Dataset, DatasetDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_runner import run_experiment


EXPERIMENT = {
    "name": "smoke_local",
    "description": "Very small local smoke test (synthetic dataset + tiny model).",
    "base_config": "configs/base.yaml",
    "device_config": None,
    "run": {"mode": "new", "resume_version": None},
    "artifacts": {"results_filename": "final_results.json", "save_eval_metrics_json": False},
    "overrides": {
        "model": {
            "name": "sshleifer/tiny-gpt2",
            "dtype": "fp32",
            "device": "cpu",
            "use_cache": True,
        },
        "data": {
            "max_length": 128,
            "num_preprocessing_workers": 1,
            "load_from_disk_path": None,  # filled in at runtime
        },
        "dataloader": {
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {
            "num_epochs": 1,
            "max_steps": 10,
            "gradient_accumulation_steps": 1,
            "log_every_steps": 1,
            "eval_every_steps": 0,
            "save_every_steps": 0,
            "save_at_end": False,
        },
        "generation": {
            "max_new_tokens": 8,
            "do_sample": False,
            "temperature": 0.0,
        },
        "optimizer": {"lr": 5e-5, "weight_decay": 0.0},
        "peft": {"enabled": False},
    },
}


def _build_synthetic_gsm8k_like_dataset() -> DatasetDict:
    train_examples = {
        "question": [
            "If Ana has 2 apples and buys 3 more, how many apples does she have?",
            "Tom has 10 books and gives away 4. How many books remain?",
            "A box has 6 red marbles and 5 blue marbles. How many marbles are there in total?",
            "Sara solved 7 problems on Monday and 8 on Tuesday. How many total problems did she solve?",
        ],
        "answer": [
            "Ana starts with 2 apples and buys 3 more. 2 + 3 = 5. #### 5",
            "Tom gives away 4 from 10. 10 - 4 = 6. #### 6",
            "There are 6 red and 5 blue marbles. 6 + 5 = 11. #### 11",
            "Sara solved 7 and then 8 problems. 7 + 8 = 15. #### 15",
        ],
    }
    eval_examples = {
        "question": [
            "There are 9 birds on a tree and 2 fly away. How many birds are left?",
            "Liam has 4 bags with 3 candies in each bag. How many candies does he have?",
        ],
        "answer": [
            "Two birds fly away from 9. 9 - 2 = 7. #### 7",
            "There are 4 groups of 3 candies. 4 * 3 = 12. #### 12",
        ],
    }
    return DatasetDict(
        {
            "train": Dataset.from_dict(train_examples),
            "validation": Dataset.from_dict(eval_examples),
            "test": Dataset.from_dict(eval_examples),
        }
    )


def _prepare_dataset_dir(dataset_dir: Path, *, overwrite: bool) -> Path:
    if dataset_dir.exists() and overwrite:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset = _build_synthetic_gsm8k_like_dataset()
    dataset.save_to_disk(str(dataset_dir))
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny local smoke test experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved output dir without running training.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/tmp/hpml_smoke_local_dataset",
        help="Where to store the synthetic DatasetDict (load_from_disk_path).",
    )
    parser.add_argument("--overwrite-dataset", action="store_true", help="Re-create the synthetic dataset directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = deepcopy(EXPERIMENT)

    if not args.dry_run:
        dataset_dir = _prepare_dataset_dir(Path(args.dataset_dir), overwrite=bool(args.overwrite_dataset))
        exp["overrides"]["data"]["load_from_disk_path"] = str(dataset_dir)

    run_dir = run_experiment(exp, dry_run=bool(args.dry_run))
    print(run_dir)


if __name__ == "__main__":
    main()

