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
    uv run python -m scripts.evaluate \
        --env agent_auth \
        --pool-name my-browser-pool \
        --max-tasks 10

    # Evaluate with a specific model
    uv run python -m scripts.evaluate \
        --env agent_auth \
        --model openrouter/qwen/qwen3-vl-8b-instruct \
        --max-tasks 5

    # Output results to JSON file
    uv run python -m scripts.evaluate \
        --env agent_auth \
        --max-tasks 10 \
        --output results.json

    # Dry run (print config without running)
    uv run python -m scripts.evaluate --dry-run

Environment Variables:
    KERNEL_API_KEY: Required for Kernel browser API
    OPENROUTER_API_KEY: Required for WebJudge and VLM inference
    RAINDROP_WRITE_KEY: Optional, enables Raindrop AI tracking
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
from datetime import datetime

from dotenv import load_dotenv
from kernel import Kernel
from rich.console import Console
from rich.table import Table

from core.agent import AgentConfig, QwenAgent
from core.agent_loop import run_agent_loop
from core.browser import KernelBrowserAdapter
from core.reward_models import Trajectory, WebJudge
from core.tracking import (
    begin_episode,
    create_step_callbacks,
    finish_episode,
    generate_id,
    init_raindrop,
    is_raindrop_enabled,
    shutdown_raindrop,
    track_webjudge_signal,
)

def ts() -> str:
    """Return short timestamp [HH:MM:SS] for log lines."""
    return datetime.now().strftime("[%H:%M:%S]")

console = Console()

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Available environments
AVAILABLE_ENVS = ["osworld", "agent_auth"]

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
    pool_size: int | None = None  # None means use pool's configured size
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
        default="osworld",
        choices=AVAILABLE_ENVS,
        help=f"Environment to use (default: osworld). Available: {AVAILABLE_ENVS}",
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
        "--pool-size",
        type=int,
        default=None,
        help="Number of concurrent tasks (default: pool's configured size)",
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
        pool_size=args.pool_size,
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
    table.add_row("Pool Size", str(cfg.pool_size or "(query from pool)"))
    table.add_row("Output File", cfg.output_file or "(stdout only)")

    console.print(table)


def get_tasks_and_system_prompt(cfg: EvalConfig):
    """Get tasks and system prompt for the selected environment."""
    if cfg.env == "osworld":
        from examples.osworld.actions import OSWORLD_ACTIONS
        from examples.osworld.config import get_osworld_system_prompt
        from examples.osworld.dataset import load_tasks

        task_file = cfg.task_file or "examples/osworld/tasks.jsonl"
        tasks = load_tasks(task_file, limit=cfg.max_tasks)
        system_prompt = get_osworld_system_prompt()
        extra_actions = OSWORLD_ACTIONS

        return tasks, system_prompt, extra_actions

    elif cfg.env == "agent_auth":
        from examples.agent_auth.actions import AGENT_AUTH_ACTIONS
        from examples.agent_auth.config import get_agent_auth_system_prompt
        from examples.agent_auth.dataset import load_tasks

        task_file = cfg.task_file or "examples/agent_auth/tasks.jsonl"
        tasks = load_tasks(task_file, limit=cfg.max_tasks)
        system_prompt = get_agent_auth_system_prompt()
        extra_actions = AGENT_AUTH_ACTIONS

        return tasks, system_prompt, extra_actions

    else:
        raise ValueError(f"Unknown environment: {cfg.env}")


def _run_episode_sync(
    task,
    adapter: KernelBrowserAdapter,
    agent_config: AgentConfig,
    cfg: EvalConfig,
    convo_id: str | None = None,
) -> tuple[Trajectory, dict]:
    """
    Run the synchronous part of an episode (runs in thread pool).

    Browser acquire/release and heartbeat are managed by the caller.
    """
    task_label = f"{task.domain} ({task.id})"
    metadata = {
        "task_id": task.id,
        "initial_url": task.initial_url,
        "task": task.task,
        "steps": 0,
        "terminal_action": None,
    }

    # Create agent for this episode (each concurrent task gets its own)
    agent = QwenAgent(config=agent_config)

    # Navigate to initial URL
    t_nav_start = time.perf_counter()
    try:
        adapter.navigate(task.initial_url)
        t_nav = time.perf_counter() - t_nav_start
        console.print(f"    {ts()} [dim]{task_label}: navigate ({t_nav:.1f}s) → {task.initial_url}[/]")
    except Exception as e:
        metadata["error"] = f"Navigation failed: {e}"
        return Trajectory(
            task_id=task.id,
            task=task.task,
            action_history=[],
            screenshots=[],
            initial_url=task.initial_url,
        ), metadata

    # Initial screenshot
    initial_screenshot = adapter.capture_screenshot()

    # Create step callbacks - Raindrop tracking (if enabled) + console logging (always)
    raindrop_on_step_start = None
    raindrop_on_step_complete = None
    raindrop_on_action_overlay = None

    if convo_id and is_raindrop_enabled():
        raindrop_on_step_start, raindrop_on_step_complete, raindrop_on_action_overlay, _ = create_step_callbacks(
            model=agent_config.model,
            convo_id=convo_id,
            nav_step_offset=1,  # Navigation is step 1
        )

    # Wrap callbacks to always log to console AND optionally track to Raindrop
    def on_step_start(step: int, screenshot):
        if raindrop_on_step_start:
            raindrop_on_step_start(step, screenshot)

    def on_action_overlay(step: int, action, overlay):
        if raindrop_on_action_overlay:
            raindrop_on_action_overlay(step, action, overlay)

    def on_step_complete(step: int, step_result):
        # Always log to console with timing
        display_step = step + 1  # Navigation is step 1
        if step_result.error or step_result.action is None:
            console.print(f"    {ts()} [dim]{task_label}: step {display_step} error: {step_result.error or 'Failed to parse action'}[/]")
        elif step_result.action_desc:
            desc = step_result.action_desc
            if len(desc) > 80:
                desc = desc[:77] + "..."
            if step_result.is_terminal:
                timing = f"total={step_result.total_time:.1f}s predict={step_result.predict_time:.1f}s"
            else:
                timing = f"total={step_result.total_time:.1f}s predict={step_result.predict_time:.1f}s exec={step_result.exec_time:.1f}s"
            console.print(f"    {ts()} [dim]{task_label}: step {display_step} {timing}: {desc}[/]")
        # Also track to Raindrop if enabled
        if raindrop_on_step_complete:
            raindrop_on_step_complete(step, step_result)

    # Run the shared agent loop
    loop_result = run_agent_loop(
        agent=agent,
        adapter=adapter,
        task=task.task,
        initial_screenshot=initial_screenshot,
        max_steps=cfg.max_steps,
        image_max_size=cfg.image_max_size,
        on_step_start=on_step_start,
        on_step_complete=on_step_complete,
        on_action_overlay=on_action_overlay,
    )

    # Build action history (prepend navigation)
    action_history = [f"Navigate to {task.initial_url}"] + loop_result.action_history

    # Update metadata from loop result
    metadata["steps"] = loop_result.steps_completed
    metadata["terminal_action"] = loop_result.terminal_action
    if loop_result.error:
        metadata["error"] = loop_result.error

    trajectory = Trajectory(
        task_id=task.id,
        task=task.task,
        action_history=action_history,
        screenshots=loop_result.screenshots,
        initial_url=task.initial_url,
    )

    return trajectory, metadata


def _acquire_and_reset_browser_sync(pool_name: str, acquire_timeout: int, task_id: str):
    """
    Create Kernel, acquire browser, create adapter and reset - all in thread.

    This runs entirely in a thread pool worker to avoid blocking the event loop.
    The browser reset (which makes HTTP calls) happens here in the thread.

    Returns:
        Tuple of (kernel, adapter, acquire_time_seconds)
    """
    t_start = time.perf_counter()

    kernel = Kernel()

    browser = kernel.browser_pools.acquire(
        pool_name,
        acquire_timeout_seconds=acquire_timeout,
    )

    # Create adapter WITH reset_on_init=True (the default)
    # This resets the browser to a clean state - and it happens in this thread,
    # not on the event loop, so it doesn't block other tasks
    adapter = KernelBrowserAdapter(kernel, browser, reset_on_init=True)

    total_time = time.perf_counter() - t_start
    return kernel, adapter, total_time


async def run_single_episode(
    task,
    agent_config: AgentConfig,
    cfg: EvalConfig,
    convo_id: str | None = None,
) -> tuple[Trajectory, dict]:
    """
    Run a single evaluation episode with proper async/thread handling.

    - Kernel creation + browser acquire run together in thread pool
    - Heartbeat runs in async context (needs event loop)
    - Agent loop runs in thread pool (blocking VLM calls)
    """
    task_label = f"{task.domain} ({task.id})"

    # Create Kernel + acquire browser + reset - all in thread pool for true parallelism
    # The browser reset happens in the thread, not on the event loop
    kernel, adapter, acquire_time = await asyncio.to_thread(
        _acquire_and_reset_browser_sync,
        cfg.pool_name,
        cfg.acquire_timeout_seconds,
        task.id,
    )

    console.print(f"    {ts()} [dim]{task_label}: browser acquired ({acquire_time:.1f}s)[/]")

    heartbeat_task: asyncio.Task | None = None
    try:
        # Start heartbeat as background task (don't await - it has random delays)
        heartbeat_task = asyncio.create_task(adapter.start_heartbeat())

        # Run the agent loop in thread pool (blocking VLM calls)
        trajectory, metadata = await asyncio.to_thread(
            _run_episode_sync,
            task,
            adapter,
            agent_config,
            cfg,
            convo_id,
        )

        return trajectory, metadata

    finally:
        # Cancel heartbeat task if still running, then stop heartbeat
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        await adapter.stop_heartbeat()

        # Release browser in thread pool (blocking HTTP call)
        # Don't reuse if browser is in bad state (e.g., reset failed)
        reuse = not adapter._should_not_reuse
        if not reuse:
            console.print(f"    {ts()} [yellow]Releasing browser {adapter.session_id} with reuse=False (browser in bad state)[/]")
        try:
            await asyncio.to_thread(
                kernel.browser_pools.release,
                cfg.pool_name,
                session_id=adapter.session_id,
                reuse=reuse,
            )
        except Exception as e:
            console.print(f"    {ts()} [red]Browser release failed for {adapter.session_id}: {e}[/]")


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

    # Initialize Raindrop (optional)
    raindrop_enabled = init_raindrop()
    if raindrop_enabled:
        console.print("  ✓ Raindrop tracking enabled")
    else:
        console.print("  [dim]ℹ Raindrop tracking disabled (no RAINDROP_WRITE_KEY)[/]")

    # Generate batch ID for Raindrop tracking
    batch_id = generate_id()

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

    # Query pool size from Kernel if not specified
    pool_size = cfg.pool_size
    if pool_size is None:
        try:
            pool_info = kernel.browser_pools.retrieve(cfg.pool_name)
            pool_size = pool_info.browser_pool_config.size
            console.print(f"  ✓ Pool size: {pool_size} (from pool config)")
        except Exception as e:
            console.print(f"[yellow]  ⚠ Could not query pool size: {e}[/]")
            pool_size = 1  # Fall back to sequential execution
            console.print(f"  → Falling back to pool size: {pool_size}")
    else:
        console.print(f"  ✓ Pool size: {pool_size} (from --pool-size)")

    webjudge = WebJudge(model=cfg.webjudge_model, api_key=openrouter_key)
    console.print(f"  ✓ WebJudge ({cfg.webjudge_model})")

    agent_config = AgentConfig(
        model=cfg.agent_model,
        api_key=openrouter_key,
        system_prompt=system_prompt,
        extra_actions=extra_actions,
    )
    console.print(f"  ✓ Agent config ({cfg.agent_model})")

    # Run evaluation
    console.print(f"\n[bold blue]Running evaluation ({pool_size} concurrent)...[/]")

    results: list[dict] = []
    results_lock = asyncio.Lock()
    success_count = 0
    completed_count = 0
    t_eval_start = time.perf_counter()

    # Semaphore to limit concurrency to pool size
    semaphore = asyncio.Semaphore(pool_size)

    async def run_and_evaluate(task, task_idx: int) -> dict:
        """Run a single episode and evaluate it, respecting the semaphore."""
        nonlocal success_count, completed_count

        async with semaphore:
            task_str = task.task[:60] + "..." if len(task.task) > 60 else task.task
            console.print(f"  {ts()} [dim]→ Starting:[/] {task.domain} ({task.id}) - {task_str}")
            t_start = time.perf_counter()

            # Start Raindrop interaction for this episode
            # Each task gets its own convo_id, with batch_id as a grouping property
            episode_convo_id = generate_id()
            interaction = begin_episode(
                task=task.task,
                convo_id=episode_convo_id,
                properties={
                    "batch_id": batch_id,
                    "task_id": task.id,
                    "task_idx": task_idx,
                    "domain": task.domain,
                    "initial_url": task.initial_url,
                    "env": cfg.env,
                    "model": cfg.agent_model,
                },
            )

            t_task_elapsed = 0.0
            t_webjudge_elapsed = 0.0

            try:
                # Run episode (blocking work runs in thread pool)
                trajectory, metadata = await run_single_episode(
                    task=task,
                    agent_config=agent_config,
                    cfg=cfg,
                    convo_id=episode_convo_id,
                )
                t_task_elapsed = time.perf_counter() - t_start

                # Evaluate with WebJudge
                t_wj_start = time.perf_counter()
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
                t_webjudge_elapsed = time.perf_counter() - t_wj_start

            except Exception as e:
                success = False
                score = 0.0
                reasoning = f"Episode error: {e}"
                metadata = {"error": str(e)}
                t_task_elapsed = time.perf_counter() - t_start

            t_elapsed = time.perf_counter() - t_start

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

            # Finish Raindrop interaction and track WebJudge signal
            finish_episode(
                interaction=interaction,
                output=f"{'Success' if success else 'Failure'}: {reasoning[:200]}",
                properties={
                    "success": success,
                    "score": score,
                    "steps": metadata.get("steps", 0),
                    "duration_seconds": round(t_elapsed, 2),
                },
            )
            track_webjudge_signal(
                interaction=interaction,
                success=success,
                score=score,
                reasoning=reasoning,
                webjudge_model=cfg.webjudge_model,
            )

            # Thread-safe update of counters and results
            async with results_lock:
                completed_count += 1
                if success:
                    success_count += 1
                results.append(result)

                status = "[green]✓[/]" if success else "[red]✗[/]"
                steps_info = f"{metadata.get('steps', 0)} steps"
                term_action = metadata.get('terminal_action')
                term_info = f", {term_action}" if term_action else ""
                # Show timing breakdown: task time + webjudge time = total
                timing_info = f"{t_task_elapsed:.1f}s task + {t_webjudge_elapsed:.1f}s judge"
                console.print(
                    f"  {ts()} {status} [{completed_count}/{len(tasks)}] {task.domain}: "
                    f"{steps_info}{term_info}, {timing_info}"
                )

            return result

    # Run all tasks concurrently with TaskGroup
    console.print(f"  {ts()} Starting {len(tasks)} tasks...")

    async with asyncio.TaskGroup() as tg:
        for i, task in enumerate(tasks):
            tg.create_task(run_and_evaluate(task, i))

    total_time = time.perf_counter() - t_eval_start

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Evaluation Summary[/]")
    console.print("=" * 60)

    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value")

    success_rate = success_count / len(tasks) * 100 if tasks else 0
    # Calculate avg from individual task durations (more accurate for parallel execution)
    avg_task_duration = (
        sum(r["duration_seconds"] for r in results) / len(results) if results else 0
    )

    summary_table.add_row("Total Tasks", str(len(tasks)))
    summary_table.add_row("Successful", str(success_count))
    summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
    summary_table.add_row("Total Time", f"{total_time:.1f}s (wall clock)")
    summary_table.add_row("Avg Time/Task", f"{avg_task_duration:.1f}s")

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

    # Flush and shutdown Raindrop
    if is_raindrop_enabled():
        console.print(f"\n[dim]ℹ Raindrop batch_id: {batch_id}[/]")
    shutdown_raindrop()

    return 0


def main() -> int:
    """Entry point."""
    cfg = parse_args()
    return asyncio.run(eval_main(cfg))


if __name__ == "__main__":
    sys.exit(main())


