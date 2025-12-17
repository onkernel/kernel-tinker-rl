#!/usr/bin/env python3
"""
evaluate.py - Batch WebJudge evaluation of agent trajectories.

This script runs an agent on a set of tasks and evaluates the trajectories
using WebJudge. Useful for:
- Evaluating model checkpoints
- Benchmarking agent performance
- Testing changes to the agent or environment

Usage:
    cd kernel-tinker-rl

    # Evaluate on agent_auth tasks
    uv run python -m scripts.evaluate \\
        --env agent_auth \\
        --pool-name my-browser-pool \\
        --max-tasks 10

    # Evaluate with a specific model
    uv run python -m scripts.evaluate \\
        --env agent_auth \\
        --model openrouter/qwen/qwen3-vl-8b-instruct \\
        --max-tasks 5

    # Output results to JSON file
    uv run python -m scripts.evaluate \\
        --env agent_auth \\
        --max-tasks 10 \\
        --output results.json

    # Dry run (print config without running)
    uv run python -m scripts.evaluate --dry-run

Environment Variables:
    KERNEL_API_KEY: Required for Kernel browser API
    OPENROUTER_API_KEY: Required for WebJudge and VLM inference
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from kernel import Kernel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from core.agent import AgentConfig, QwenAgent
from core.browser import PoolBrowserAdapter
from core.reward_models import Trajectory, WebJudge
from core.utils import resize_image

console = Console()

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Available environments
AVAILABLE_ENVS = ["agent_auth"]

# Default models
DEFAULT_AGENT_MODEL = "qwen/qwen3-vl-8b-instruct"
DEFAULT_WEBJUDGE_MODEL = "openai/o4-mini"


@dataclass
class EvalConfig:
    """Configuration for batch evaluation."""

    # Environment selection
    env: str = "agent_auth"

    # Model parameters
    agent_model: str = DEFAULT_AGENT_MODEL
    webjudge_model: str = DEFAULT_WEBJUDGE_MODEL

    # Evaluation parameters
    max_tasks: int | None = None
    max_steps: int = 10
    task_file: str | None = None

    # Browser pool parameters
    pool_name: str = "eval-browser-pool"
    acquire_timeout_seconds: int = 60
    image_max_size: int = 512

    # Output
    output_file: str | None = None
    verbose: bool = False

    # Control flags
    dry_run: bool = False


def parse_args() -> EvalConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch WebJudge evaluation of agent trajectories",
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
        "--model",
        default=DEFAULT_AGENT_MODEL,
        help=f"Agent model to use (default: {DEFAULT_AGENT_MODEL})",
    )
    parser.add_argument(
        "--webjudge-model",
        default=DEFAULT_WEBJUDGE_MODEL,
        help=f"WebJudge model (default: {DEFAULT_WEBJUDGE_MODEL})",
    )

    # Evaluation parameters
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Limit number of tasks (default: all)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10, help="Max actions per episode (default: 10)"
    )
    parser.add_argument("--task-file", default=None, help="Path to task JSONL file")

    # Browser pool parameters
    parser.add_argument(
        "--pool-name",
        default="eval-browser-pool",
        help="Browser pool name (default: eval-browser-pool)",
    )
    parser.add_argument(
        "--acquire-timeout", type=int, default=60, help="Browser acquire timeout (default: 60)"
    )
    parser.add_argument(
        "--image-max-size", type=int, default=512, help="Max image dimension (default: 512)"
    )

    # Output
    parser.add_argument("--output", "-o", default=None, help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Control flags
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")

    args = parser.parse_args()

    return EvalConfig(
        env=args.env,
        agent_model=args.model,
        webjudge_model=args.webjudge_model,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
        task_file=args.task_file,
        pool_name=args.pool_name,
        acquire_timeout_seconds=args.acquire_timeout,
        image_max_size=args.image_max_size,
        output_file=args.output,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


def print_config(cfg: EvalConfig) -> None:
    """Print configuration table."""
    table = Table(title="Evaluation Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    table.add_row("Environment", cfg.env)
    table.add_row("Agent Model", cfg.agent_model)
    table.add_row("WebJudge Model", cfg.webjudge_model)
    table.add_row("Max Tasks", str(cfg.max_tasks or "all"))
    table.add_row("Max Steps", str(cfg.max_steps))
    table.add_row("Task File", cfg.task_file or "(default for env)")
    table.add_row("Pool Name", cfg.pool_name)
    table.add_row("Output File", cfg.output_file or "(stdout only)")

    console.print(table)


def get_tasks_and_system_prompt(cfg: EvalConfig):
    """Get tasks and system prompt for the selected environment."""
    if cfg.env == "agent_auth":
        from examples.agent_auth.config import get_agent_auth_system_prompt
        from examples.agent_auth.dataset import load_tasks

        task_file = cfg.task_file or "examples/agent_auth/tasks.jsonl"
        tasks = load_tasks(task_file, limit=cfg.max_tasks)
        system_prompt = get_agent_auth_system_prompt()

        # Get extra actions
        from examples.agent_auth.actions import AGENT_AUTH_ACTIONS

        extra_actions = AGENT_AUTH_ACTIONS

        return tasks, system_prompt, extra_actions
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")


async def run_single_episode(
    task,
    kernel: Kernel,
    agent: QwenAgent,
    cfg: EvalConfig,
) -> tuple[Trajectory, dict]:
    """
    Run a single evaluation episode.

    Returns:
        Tuple of (trajectory, metadata)
    """
    screenshots = []
    action_history = []
    metadata = {
        "task_id": task.id,
        "initial_url": task.initial_url,
        "task": task.task,
        "steps": 0,
        "terminal_action": None,
    }

    # Acquire browser from pool
    adapter = PoolBrowserAdapter(
        kernel=kernel,
        pool_name=cfg.pool_name,
        acquire_timeout_seconds=cfg.acquire_timeout_seconds,
    )

    try:
        adapter.acquire()
        await adapter.start_heartbeat()

        # Navigate to initial URL
        try:
            adapter.navigate(task.initial_url)
        except Exception as e:
            metadata["error"] = f"Navigation failed: {e}"
            return Trajectory(
                task_id=task.id,
                task=task.task,
                action_history=action_history,
                screenshots=screenshots,
                initial_url=task.initial_url,
            ), metadata

        # Initial screenshot
        screenshot = adapter.capture_screenshot()
        screenshot_resized = resize_image(screenshot, max_size=cfg.image_max_size)
        screenshots.append(screenshot_resized.copy())
        action_history.append(f"Navigate to {task.initial_url}")

        # Reset agent for new episode
        agent.reset()

        # Run agent loop
        for step in range(1, cfg.max_steps + 1):
            metadata["steps"] = step

            # Get action from agent
            try:
                action = agent.predict(task.task, screenshot_resized)
            except Exception as e:
                metadata["error"] = f"Agent error at step {step}: {e}"
                break

            if action is None:
                metadata["error"] = f"Failed to parse action at step {step}"
                break

            # Record action
            action_desc = action.to_description()
            if action.model_description:
                action_desc = f"{action.model_description} ({action_desc})"
            action_history.append(action_desc)

            # Check for terminal action
            if getattr(action, "is_terminal", False):
                metadata["terminal_action"] = getattr(action, "action_type", "unknown")
                break

            # Execute action
            try:
                baseline = adapter.capture_screenshot()
                should_continue = adapter.execute_action(action)

                if not should_continue:
                    metadata["terminal_action"] = getattr(action, "action_type", "unknown")
                    break

                # Wait for screen to settle
                if not getattr(action, "skip_screen_settle", False):
                    adapter.wait_for_screen_settle(baseline=baseline)

                # Capture new screenshot
                screenshot = adapter.capture_screenshot()
                screenshot_resized = resize_image(screenshot, max_size=cfg.image_max_size)
                screenshots.append(screenshot_resized.copy())

            except Exception as e:
                metadata["error"] = f"Execution error at step {step}: {e}"
                break

    finally:
        try:
            await adapter.release_async(reuse=True)
        except Exception:
            pass

    trajectory = Trajectory(
        task_id=task.id,
        task=task.task,
        action_history=action_history,
        screenshots=screenshots,
        initial_url=task.initial_url,
    )

    return trajectory, metadata


async def eval_main(cfg: EvalConfig) -> int:
    """Main entry point for batch evaluation."""

    console.print("\n[bold cyan]Kernel + Tinker Batch Evaluation[/]")
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

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key and not cfg.dry_run:
        console.print("[red]✗ OPENROUTER_API_KEY not set[/]")
        return 1
    console.print("  ✓ OPENROUTER_API_KEY")

    if cfg.dry_run:
        console.print("\n[yellow]Dry run mode - not executing[/]")
        return 0

    # Load tasks and environment config
    console.print("\n[bold blue]Loading tasks...[/]")
    tasks, system_prompt, extra_actions = get_tasks_and_system_prompt(cfg)
    console.print(f"  Loaded {len(tasks)} tasks")

    # Initialize components
    console.print("\n[bold blue]Initializing components...[/]")

    kernel = Kernel()
    console.print("  ✓ Kernel client")

    webjudge = WebJudge(model=cfg.webjudge_model, api_key=openrouter_key)
    console.print(f"  ✓ WebJudge ({cfg.webjudge_model})")

    agent_config = AgentConfig(
        model=cfg.agent_model,
        api_key=openrouter_key,
        system_prompt=system_prompt,
        extra_actions=extra_actions,
    )
    agent = QwenAgent(config=agent_config)
    console.print(f"  ✓ Agent ({cfg.agent_model})")

    # Run evaluation
    console.print("\n[bold blue]Running evaluation...[/]")

    results = []
    success_count = 0
    total_time = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task("Evaluating...", total=len(tasks))

        for i, task in enumerate(tasks):
            progress.update(
                task_progress,
                description=f"[{i+1}/{len(tasks)}] {task.domain}",
            )

            t_start = time.perf_counter()

            try:
                trajectory, metadata = await run_single_episode(
                    task=task,
                    kernel=kernel,
                    agent=agent,
                    cfg=cfg,
                )

                # Evaluate with WebJudge
                if trajectory.screenshots:
                    try:
                        wj_result = await webjudge.evaluate(trajectory)
                        success = wj_result.success
                        score = wj_result.score
                        reasoning = wj_result.reasoning
                    except Exception as e:
                        success = False
                        score = 0.0
                        reasoning = f"WebJudge error: {e}"
                else:
                    success = False
                    score = 0.0
                    reasoning = "No screenshots captured"

            except Exception as e:
                success = False
                score = 0.0
                reasoning = f"Episode error: {e}"
                metadata = {"error": str(e)}

            t_elapsed = time.perf_counter() - t_start
            total_time += t_elapsed

            result = {
                "task_id": task.id,
                "domain": task.domain,
                "success": success,
                "score": score,
                "reasoning": reasoning,
                "steps": metadata.get("steps", 0),
                "terminal_action": metadata.get("terminal_action"),
                "error": metadata.get("error"),
                "duration_seconds": round(t_elapsed, 2),
            }
            results.append(result)

            if success:
                success_count += 1

            if cfg.verbose:
                status = "[green]SUCCESS[/]" if success else "[red]FAILURE[/]"
                console.print(f"  {task.domain}: {status} ({t_elapsed:.1f}s)")

            progress.advance(task_progress)

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Evaluation Summary[/]")
    console.print("=" * 60)

    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value")

    success_rate = success_count / len(tasks) * 100 if tasks else 0
    avg_time = total_time / len(tasks) if tasks else 0

    summary_table.add_row("Total Tasks", str(len(tasks)))
    summary_table.add_row("Successful", str(success_count))
    summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
    summary_table.add_row("Total Time", f"{total_time:.1f}s")
    summary_table.add_row("Avg Time/Task", f"{avg_time:.1f}s")

    console.print(summary_table)

    # Write results to file
    if cfg.output_file:
        output_data = {
            "config": {
                "env": cfg.env,
                "agent_model": cfg.agent_model,
                "webjudge_model": cfg.webjudge_model,
                "max_steps": cfg.max_steps,
            },
            "summary": {
                "total_tasks": len(tasks),
                "success_count": success_count,
                "success_rate": success_rate,
                "total_time_seconds": round(total_time, 2),
            },
            "results": results,
        }

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


