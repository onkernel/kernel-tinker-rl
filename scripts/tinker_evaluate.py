#!/usr/bin/env python3
"""
tinker_evaluate.py - Evaluate finetuned checkpoints using Tinker's native sampling.

This script uses Tinker's SamplingClient for VLM inference, which supports
vision models unlike the OpenAI-compatible endpoint. This enables evaluation
of finetuned checkpoints that fail when using the OpenAI SDK.

Usage:
    cd kernel-tinker-rl

    # Evaluate a finetuned checkpoint
    uv run python -m scripts.tinker_evaluate \
        --model-path tinker://488643ee-3be8-523e-9297-aecf5f8bb48f:train:0/sampler_weights/final \
        --base-model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --pool-name eval-browser-pool \
        --output results/finetuned_eval.json

    # Evaluate the base model (no finetuning)
    uv run python -m scripts.tinker_evaluate \
        --base-model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --pool-name eval-browser-pool \
        --output results/baseline_eval.json

    # Evaluate with specific task file
    uv run python -m scripts.tinker_evaluate \
        --model-path tinker://... \
        --base-model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --task-file examples/agent_auth/tasks_eval.jsonl \
        --output results/eval.json

    # Dry run (print config without running)
    uv run python -m scripts.tinker_evaluate --dry-run

Environment Variables:
    KERNEL_API_KEY: Required for Kernel browser API
    TINKER_API_KEY: Required for Tinker sampling
    OPENROUTER_API_KEY: Required for WebJudge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

console = Console()

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Default models
DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_WEBJUDGE_MODEL = "openai/gpt-5-mini"


@dataclass
class TinkerEvalConfig:
    """Configuration for Tinker-based evaluation."""

    # Required - path to output JSON file
    output_file: str

    # Model parameters
    base_model: str = DEFAULT_BASE_MODEL
    model_path: str | None = None  # Tinker checkpoint path (e.g., tinker://...)
    renderer_name: str = "qwen3_vl_instruct"

    # WebJudge parameters
    webjudge_model: str = DEFAULT_WEBJUDGE_MODEL
    webjudge_enabled: bool = True

    # Task parameters
    env: str = "agent_auth"
    task_file: str | None = None
    max_tasks: int | None = None
    start_index: int | None = None
    end_index: int | None = None

    # Evaluation parameters
    max_steps: int = 10
    max_tokens: int = 512
    temperature: float = 0.0
    concurrency: int = 4

    # Browser pool parameters
    pool_name: str = "eval-browser-pool"
    acquire_timeout_seconds: int = 60
    image_max_size: int = 512

    # Control flags
    dry_run: bool = False
    verbose: bool = False


def parse_args() -> TinkerEvalConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned checkpoints using Tinker's native sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model parameters
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model name (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Tinker checkpoint path (e.g., tinker://...). If not provided, evaluates base model.",
    )
    parser.add_argument(
        "--renderer-name",
        default="qwen3_vl_instruct",
        help="Tinker renderer name (default: qwen3_vl_instruct)",
    )

    # WebJudge parameters
    parser.add_argument(
        "--webjudge-model",
        default=DEFAULT_WEBJUDGE_MODEL,
        help=f"WebJudge model (default: {DEFAULT_WEBJUDGE_MODEL})",
    )
    parser.add_argument(
        "--no-webjudge",
        action="store_true",
        help="Disable WebJudge scoring (use heuristics)",
    )

    # Task parameters
    parser.add_argument(
        "--env",
        default="agent_auth",
        choices=["agent_auth"],
        help="Environment to evaluate (default: agent_auth)",
    )
    parser.add_argument(
        "--task-file",
        default=None,
        help="Path to task JSONL file (default: examples/{env}/tasks.jsonl)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start index for task subset (0-based, inclusive)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index for task subset (0-based, inclusive)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Max actions per episode (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per model response (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent episodes (default: 4)",
    )

    # Browser pool parameters
    parser.add_argument(
        "--pool-name",
        default="eval-browser-pool",
        help="Browser pool name (default: eval-browser-pool)",
    )
    parser.add_argument(
        "--acquire-timeout",
        type=int,
        default=60,
        help="Browser acquire timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--image-max-size",
        type=int,
        default=512,
        help="Max image dimension (default: 512)",
    )

    # Output and control
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    return TinkerEvalConfig(
        output_file=args.output,
        base_model=args.base_model,
        model_path=args.model_path,
        renderer_name=args.renderer_name,
        webjudge_model=args.webjudge_model,
        webjudge_enabled=not args.no_webjudge,
        env=args.env,
        task_file=args.task_file,
        max_tasks=args.max_tasks,
        start_index=args.start_index,
        end_index=args.end_index,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        pool_name=args.pool_name,
        acquire_timeout_seconds=args.acquire_timeout,
        image_max_size=args.image_max_size,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def print_config(cfg: TinkerEvalConfig) -> None:
    """Print configuration table."""
    table = Table(title="Tinker Evaluation Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    table.add_row("Base Model", cfg.base_model)
    table.add_row("Model Path", cfg.model_path or "(base model only)")
    table.add_row("Renderer", cfg.renderer_name)
    table.add_row("WebJudge Model", cfg.webjudge_model if cfg.webjudge_enabled else "disabled")
    table.add_row("Environment", cfg.env)
    table.add_row("Task File", cfg.task_file or f"(default for {cfg.env})")
    table.add_row("Max Tasks", str(cfg.max_tasks or "all"))

    if cfg.start_index is not None or cfg.end_index is not None:
        start_str = str(cfg.start_index) if cfg.start_index is not None else "0"
        end_str = str(cfg.end_index) if cfg.end_index is not None else "end"
        table.add_row("Task Index Range", f"{start_str} to {end_str}")

    table.add_row("Max Steps", str(cfg.max_steps))
    table.add_row("Concurrency", str(cfg.concurrency))
    table.add_row("Pool Name", cfg.pool_name)
    table.add_row("Output File", cfg.output_file)

    console.print(table)


def load_tasks(cfg: TinkerEvalConfig):
    """Load tasks for the selected environment."""
    if cfg.env == "agent_auth":
        from examples.agent_auth.dataset import load_tasks as load_agent_auth_tasks

        task_file = cfg.task_file or "examples/agent_auth/tasks.jsonl"
        tasks = load_agent_auth_tasks(task_file, limit=None)
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")

    # Apply index filtering
    if cfg.start_index is not None or cfg.end_index is not None:
        start = cfg.start_index if cfg.start_index is not None else 0
        end = (cfg.end_index + 1) if cfg.end_index is not None else len(tasks)
        tasks = tasks[start:end]

    # Apply max_tasks limit
    if cfg.max_tasks is not None:
        tasks = tasks[:cfg.max_tasks]

    return tasks


async def eval_main(cfg: TinkerEvalConfig) -> int:
    """Main entry point for Tinker-based evaluation."""
    import tinker

    from examples.agent_auth.evaluator import (
        AgentAuthSamplingEvaluator,
        EvalConfig as EvaluatorConfig,
    )

    console.print("\n[bold cyan]Tinker Native Sampling Evaluation[/]")
    console.print("=" * 60)

    # Load .env
    load_dotenv()

    # Print config
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
    if cfg.webjudge_enabled:
        if not openrouter_key and not cfg.dry_run:
            console.print("[red]✗ OPENROUTER_API_KEY not set (required for WebJudge)[/]")
            return 1
        console.print("  ✓ OPENROUTER_API_KEY")

    if cfg.dry_run:
        console.print("\n[yellow]Dry run mode - not executing[/]")
        return 0

    # Load tasks
    console.print("\n[bold blue]Loading tasks...[/]")
    tasks = load_tasks(cfg)
    console.print(f"  Loaded {len(tasks)} tasks")

    # Create Tinker service client and sampling client
    console.print("\n[bold blue]Initializing Tinker...[/]")
    service_client = tinker.ServiceClient()

    # Create sampling client - with or without checkpoint
    if cfg.model_path:
        console.print(f"  Creating sampling client with checkpoint...")
        console.print(f"    Base model: {cfg.base_model}")
        console.print(f"    Checkpoint: {cfg.model_path}")
        sampling_client = service_client.create_sampling_client(
            base_model=cfg.base_model,
            model_path=cfg.model_path,
        )
    else:
        console.print(f"  Creating sampling client for base model...")
        console.print(f"    Model: {cfg.base_model}")
        sampling_client = service_client.create_sampling_client(
            base_model=cfg.base_model,
        )

    console.print("  ✓ Sampling client created")

    # Create evaluator
    console.print("\n[bold blue]Creating evaluator...[/]")
    eval_config = EvaluatorConfig(
        pool_name=cfg.pool_name,
        acquire_timeout_seconds=cfg.acquire_timeout_seconds,
        max_steps=cfg.max_steps,
        image_max_size=cfg.image_max_size,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        concurrency=cfg.concurrency,
    )

    evaluator = AgentAuthSamplingEvaluator(
        model_name=cfg.base_model,
        tasks=tasks,
        config=eval_config,
        webjudge_model=cfg.webjudge_model,
        webjudge_enabled=cfg.webjudge_enabled,
        renderer_name=cfg.renderer_name,
    )

    console.print("  ✓ Evaluator created")

    # Run evaluation
    console.print("\n[bold blue]Running evaluation...[/]")
    try:
        metrics = await evaluator(sampling_client)
    except Exception as e:
        console.print(f"\n[red]✗ Evaluation failed: {e}[/]")
        raise

    # Get detailed results
    results = evaluator.get_results()

    # Write output
    output_data = {
        "config": {
            "base_model": cfg.base_model,
            "model_path": cfg.model_path,
            "webjudge_model": cfg.webjudge_model,
            "max_steps": cfg.max_steps,
            "env": cfg.env,
        },
        "summary": {
            "total_tasks": len(results),
            "success_count": int(metrics.get("total_success", 0)),
            "success_rate": metrics.get("success_rate", 0.0),
            "avg_score": metrics.get("avg_score", 0.0),
            "total_time_seconds": metrics.get("total_time_seconds", 0.0),
        },
        "results": [
            {
                "task_id": r.task_id,
                "domain": r.domain,
                "success": r.success,
                "score": r.score,
                "reasoning": r.reasoning,
                "steps": r.steps,
                "terminal_action": r.terminal_action,
                "error": r.error,
                "duration_seconds": r.duration_seconds,
            }
            for r in results
        ],
    }

    # Ensure output directory exists
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cfg.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"\n  Results written to: {cfg.output_file}")

    return 0


def main() -> int:
    """Entry point."""
    cfg = parse_args()
    return asyncio.run(eval_main(cfg))


if __name__ == "__main__":
    sys.exit(main())

