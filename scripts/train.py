#!/usr/bin/env python3
"""
train.py - Generalized RL training entry point.

This script provides a flexible interface for training computer use agents
using Tinker's RL infrastructure with:
- Kernel browser pools for efficient browser management
- WebJudge for reward computation
- Weights & Biases for experiment tracking

Usage:
    cd kernel-tinker-rl

    # Agent Auth training with browser pool
    uv run python -m scripts.train \
        --env agent_auth \
        --pool-name my-browser-pool \
        --batch-size 4 \
        --group-size 2

    # With W&B logging
    uv run python -m scripts.train \
        --env agent_auth \
        --pool-name my-browser-pool \
        --wandb-project my-project

    # Custom task file
    uv run python -m scripts.train \
        --env agent_auth \\
        --task-file examples/agent_auth/tasks.jsonl \
        --pool-name my-browser-pool

    # Dry run (print config without running)
    uv run python -m scripts.train --dry-run

Environment Variables:
    KERNEL_API_KEY: Required for Kernel browser API
    TINKER_API_KEY: Required for Tinker RL training
    OPENROUTER_API_KEY: Required for WebJudge (gpt-5-mini)
    WANDB_API_KEY: Required for --wandb-project
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Prevent tokenizer parallelism warning when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tinker_cookbook.rl import train

console = Console()

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Available environments
AVAILABLE_ENVS = ["agent_auth"]

# Default model for training
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


@dataclass
class TrainConfig:
    """Configuration for RL training."""

    # Environment selection
    env: str = "agent_auth"

    # Model parameters
    model_name: str = DEFAULT_MODEL
    lora_rank: int = 32
    learning_rate: float = 4e-5

    # Training parameters
    batch_size: int = 4  # Tasks per batch
    group_size: int = 2  # Rollouts per task (for GRPO baseline)
    max_steps: int = 5  # Max actions per episode
    max_tokens: int = 512  # Max tokens per VLM response

    # Dataset parameters
    max_tasks: int | None = None  # Limit tasks (None = all)
    task_file: str | None = None  # Path to task JSONL file
    seed: int = 42

    # Browser pool parameters
    pool_name: str = "rl-browser-pool"
    acquire_timeout_seconds: int = 60
    image_max_size: int = 512  # Max dimension for screenshots
    max_screenshots_in_context: int = 3  # Limit screenshots in VLM context

    # WebJudge parameters
    webjudge_model: str = "openai/gpt-5-mini"
    webjudge_enabled: bool = True

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 10  # 0 = disabled
    save_every: int = 10  # 0 = disabled

    # Checkpoint parameters
    load_checkpoint_path: str | None = None

    # Control flags
    dry_run: bool = False


def parse_args() -> TrainConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RL training for computer use agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Environment selection
    parser.add_argument(
        "--env",
        default="agent_auth",
        choices=AVAILABLE_ENVS,
        help=f"Environment to use (default: agent_auth). Available: {AVAILABLE_ENVS}",
    )

    # Model parameters
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank (default: 32)")
    parser.add_argument(
        "--learning-rate", type=float, default=4e-5, help="Learning rate (default: 4e-5)"
    )

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=4, help="Tasks per batch (default: 4)")
    parser.add_argument("--group-size", type=int, default=2, help="Rollouts per task (default: 2)")
    parser.add_argument(
        "--max-steps", type=int, default=5, help="Max actions per episode (default: 5)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens per VLM response (default: 512)"
    )

    # Dataset parameters
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Limit number of tasks (default: all)"
    )
    parser.add_argument("--task-file", default=None, help="Path to task JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Browser pool parameters
    parser.add_argument(
        "--pool-name",
        default="rl-browser-pool",
        help="Browser pool name (default: rl-browser-pool)",
    )
    parser.add_argument(
        "--acquire-timeout", type=int, default=60, help="Browser acquire timeout (default: 60)"
    )
    parser.add_argument(
        "--image-max-size", type=int, default=512, help="Max image dimension (default: 512)"
    )
    parser.add_argument(
        "--max-screenshots", type=int, default=3, help="Max screenshots in VLM context (default: 3)"
    )

    # WebJudge parameters
    parser.add_argument(
        "--webjudge-model",
        default="openai/gpt-5-mini",
        help="WebJudge model (default: openai/gpt-5-mini)",
    )
    parser.add_argument("--no-webjudge", action="store_true", help="Disable WebJudge")

    # Logging parameters
    parser.add_argument("--log-path", default=None, help="Log directory path")
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    parser.add_argument("--wandb-name", default=None, help="W&B run name")
    parser.add_argument(
        "--eval-every", type=int, default=10, help="Eval frequency (default: 10, 0=disabled)"
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save frequency (default: 10, 0=disabled)"
    )

    # Checkpoint parameters
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="Path to checkpoint to load (e.g., tinker://model_id/name)",
    )

    # Control flags
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")

    args = parser.parse_args()

    return TrainConfig(
        env=args.env,
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        max_tasks=args.max_tasks,
        task_file=args.task_file,
        seed=args.seed,
        pool_name=args.pool_name,
        acquire_timeout_seconds=args.acquire_timeout,
        image_max_size=args.image_max_size,
        max_screenshots_in_context=args.max_screenshots,
        webjudge_model=args.webjudge_model,
        webjudge_enabled=not args.no_webjudge,
        log_path=args.log_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        eval_every=args.eval_every,
        save_every=args.save_every,
        load_checkpoint_path=args.load_checkpoint,
        dry_run=args.dry_run,
    )


def print_config(cfg: TrainConfig) -> None:
    """Print configuration table."""
    table = Table(title="RL Training Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    table.add_row("Environment", cfg.env)
    table.add_row("Model", cfg.model_name)
    table.add_row("LoRA Rank", str(cfg.lora_rank))
    table.add_row("Learning Rate", str(cfg.learning_rate))
    table.add_row("Batch Size", str(cfg.batch_size))
    table.add_row("Group Size", str(cfg.group_size))
    table.add_row("Max Steps", str(cfg.max_steps))
    table.add_row("Max Tasks", str(cfg.max_tasks or "all"))
    table.add_row("Task File", cfg.task_file or "(default for env)")
    table.add_row("Pool Name", cfg.pool_name)
    table.add_row(
        "WebJudge Model", cfg.webjudge_model if cfg.webjudge_enabled else "disabled"
    )
    table.add_row("W&B Project", cfg.wandb_project or "disabled")
    if cfg.load_checkpoint_path:
        table.add_row("Load Checkpoint", cfg.load_checkpoint_path)

    console.print(table)


def get_dataset_builder(cfg: TrainConfig):
    """Get the appropriate dataset builder for the selected environment."""
    if cfg.env == "agent_auth":
        from examples.agent_auth.environment import AgentAuthRLDatasetBuilder

        # Determine task file path
        task_file = cfg.task_file or "examples/agent_auth/tasks.jsonl"

        return AgentAuthRLDatasetBuilder(
            model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            group_size=cfg.group_size,
            max_tasks=cfg.max_tasks,
            task_data_path=task_file,
            seed=cfg.seed,
            pool_name=cfg.pool_name,
            acquire_timeout_seconds=cfg.acquire_timeout_seconds,
            max_steps=cfg.max_steps,
            image_max_size=cfg.image_max_size,
            max_screenshots_in_context=cfg.max_screenshots_in_context,
            webjudge_model=cfg.webjudge_model,
            webjudge_enabled=cfg.webjudge_enabled,
        )
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")


async def train_main(cfg: TrainConfig) -> int:
    """Main entry point for RL training."""

    console.print("\n[bold cyan]Kernel + Tinker RL Training[/]")
    console.print("=" * 60)

    # Load .env file
    load_dotenv()

    # Print configuration
    print_config(cfg)

    # Validate environment
    console.print("\n[bold blue]Checking environment...[/]")

    kernel_key = os.getenv("KERNEL_API_KEY")
    if not kernel_key and not cfg.dry_run:
        console.print("[red]✗ KERNEL_API_KEY not set[/]")
        return 1
    console.print("  ✓ KERNEL_API_KEY")

    tinker_key = os.getenv("TINKER_API_KEY")
    if not tinker_key and not cfg.dry_run:
        console.print("[red]✗ TINKER_API_KEY not set[/]")
        return 1
    console.print("  ✓ TINKER_API_KEY")

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key and cfg.webjudge_enabled and not cfg.dry_run:
        console.print("[red]✗ OPENROUTER_API_KEY not set (required for WebJudge)[/]")
        return 1
    console.print("  ✓ OPENROUTER_API_KEY")

    if cfg.wandb_project:
        wandb_key = os.getenv("WANDB_API_KEY")
        if not wandb_key:
            console.print("[red]✗ WANDB_API_KEY not set (required for --wandb-project)[/]")
            return 1
        console.print("  ✓ WANDB_API_KEY")

    if cfg.dry_run:
        console.print("\n[yellow]Dry run mode - not executing[/]")
        return 0

    # Build run name
    model_name_short = cfg.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"{cfg.env}_{model_name_short}_bs{cfg.batch_size}_gs{cfg.group_size}_"
        f"lr{cfg.learning_rate}_rank{cfg.lora_rank}_{date_and_time}"
    )

    # Set log path
    if cfg.log_path is not None:
        log_path = cfg.log_path
    else:
        log_path = f"/tmp/kernel-tinker-rl/{run_name}"

    # Set wandb name
    wandb_name = cfg.wandb_name or run_name

    # Ensure log directory exists
    Path(log_path).mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold blue]Log path:[/] {log_path}")

    # Build dataset builder
    console.print("\n[bold blue]Building dataset...[/]")
    dataset_builder = get_dataset_builder(cfg)

    # Build training config
    console.print("\n[bold blue]Starting training...[/]")
    train_config = train.Config(
        model_name=cfg.model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cfg.learning_rate,
        max_tokens=cfg.max_tokens,
        lora_rank=cfg.lora_rank,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        wandb_project=cfg.wandb_project,
        wandb_name=wandb_name,
        load_checkpoint_path=cfg.load_checkpoint_path,
        loss_fn="importance_sampling",
    )

    # Run training
    try:
        await train.main(train_config)
        console.print("\n[bold green]✓ Training completed successfully![/]")
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/]")
        return 130
    except Exception as e:
        console.print(f"\n[red]✗ Training failed: {e}[/]")
        raise


def main() -> int:
    """Entry point."""
    cfg = parse_args()
    return asyncio.run(train_main(cfg))


if __name__ == "__main__":
    sys.exit(main())


