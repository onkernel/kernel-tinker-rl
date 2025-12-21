#!/usr/bin/env python3
"""
split_dataset.py - Shuffle and split a JSONL dataset into train/eval files.

Creates a reproducible 80/20 train/test split with random shuffling.

Usage:
    uv run python scripts/split_dataset.py examples/agent_auth/tasks.jsonl
    uv run python scripts/split_dataset.py examples/agent_auth/tasks.jsonl --train-ratio 0.9 --seed 123
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_dataset(
    input_file: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    output_dir: str | None = None,
) -> tuple[Path, Path]:
    """
    Shuffle and split a JSONL dataset into train/eval files.

    Args:
        input_file: Path to the input JSONL file
        train_ratio: Fraction of data to use for training (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
        output_dir: Output directory (default: same as input file)

    Returns:
        Tuple of (train_path, eval_path)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load all tasks
    tasks = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if not tasks:
        raise ValueError(f"No tasks found in {input_path}")

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(tasks)

    # Split
    split_idx = int(len(tasks) * train_ratio)
    train_tasks = tasks[:split_idx]
    eval_tasks = tasks[split_idx:]

    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = input_path.parent

    # Write output files
    train_path = out_dir / f"{input_path.stem}_train.jsonl"
    eval_path = out_dir / f"{input_path.stem}_eval.jsonl"

    with open(train_path, "w") as f:
        for task in train_tasks:
            f.write(json.dumps(task) + "\n")

    with open(eval_path, "w") as f:
        for task in eval_tasks:
            f.write(json.dumps(task) + "\n")

    return train_path, eval_path


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle and split a JSONL dataset into train/eval files"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="examples/agent_auth/tasks.jsonl",
        help="Path to input JSONL file (default: examples/agent_auth/tasks.jsonl)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as input file)",
    )

    args = parser.parse_args()

    print(f"Splitting {args.input_file}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Seed: {args.seed}")

    train_path, eval_path = split_dataset(
        args.input_file,
        train_ratio=args.train_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Count tasks
    with open(train_path) as f:
        train_count = sum(1 for line in f if line.strip())
    with open(eval_path) as f:
        eval_count = sum(1 for line in f if line.strip())

    print(f"\nCreated:")
    print(f"  {train_path} ({train_count} tasks)")
    print(f"  {eval_path} ({eval_count} tasks)")


if __name__ == "__main__":
    main()

