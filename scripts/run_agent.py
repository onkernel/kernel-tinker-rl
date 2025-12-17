#!/usr/bin/env python3
"""
Run a computer use agent on a single task.

This script runs an OSWorld-style VLM agent against a real website
using Kernel browsers. Useful for testing and debugging agents.

Usage:
    # Basic usage
    uv run python -m scripts.run_agent \\
        --url https://github.com \\
        --task "Navigate to the sign in page"

    # With WebJudge evaluation
    uv run python -m scripts.run_agent \\
        --url https://github.com \\
        --task "Navigate to the sign in page" \\
        --webjudge

    # Use a different model
    uv run python -m scripts.run_agent \\
        --url https://example.com \\
        --task "Click the contact link" \\
        --model qwen/qwen3-vl-30b-a3b-instruct

Environment Variables:
    KERNEL_API_KEY: Required for Kernel browser API
    OPENROUTER_API_KEY: Required for VLM inference and WebJudge
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass

from kernel import Kernel
from rich.console import Console
from rich.table import Table

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from core import (
    AgentConfig,
    PoolBrowserAdapter,
    QwenAgent,
    Trajectory,
    WebJudge,
    setup_environment,
)

console = Console()

DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"


@dataclass
class RunConfig:
    """Configuration for agent run."""

    url: str
    task: str
    model: str = DEFAULT_MODEL
    max_steps: int = 20
    pool_name: str | None = None  # Use pool if set, else create browser
    headless: bool = False
    webjudge: bool = False
    webjudge_model: str = "openai/o4-mini"
    dry_run: bool = False


def parse_args() -> RunConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a computer use agent on a task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--url", required=True, help="Starting URL")
    parser.add_argument("--task", required=True, help="Task instruction for the agent")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps (default: 20)")
    parser.add_argument("--pool-name", default=None, help="Browser pool name (optional)")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--webjudge", action="store_true", help="Evaluate with WebJudge")
    parser.add_argument("--webjudge-model", default="openai/o4-mini", help="WebJudge model")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")

    args = parser.parse_args()

    return RunConfig(
        url=args.url,
        task=args.task,
        model=args.model,
        max_steps=args.max_steps,
        pool_name=args.pool_name,
        headless=args.headless,
        webjudge=args.webjudge,
        webjudge_model=args.webjudge_model,
        dry_run=args.dry_run,
    )


def print_config(cfg: RunConfig) -> None:
    """Print configuration table."""
    table = Table(title="Agent Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    table.add_row("URL", cfg.url)
    table.add_row("Task", cfg.task[:60] + "..." if len(cfg.task) > 60 else cfg.task)
    table.add_row("Model", cfg.model)
    table.add_row("Max Steps", str(cfg.max_steps))
    table.add_row("Browser Pool", cfg.pool_name or "create new")
    table.add_row("WebJudge", "enabled" if cfg.webjudge else "disabled")

    console.print(table)


async def run_agent(cfg: RunConfig) -> int:
    """Run the agent."""
    import os

    # Validate environment
    if not os.getenv("KERNEL_API_KEY"):
        console.print("[red]✗ KERNEL_API_KEY not set[/]")
        return 1

    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]✗ OPENROUTER_API_KEY not set[/]")
        return 1

    # Initialize Kernel
    kernel = Kernel()
    console.print("[green]✓[/] Connected to Kernel")

    # Create or acquire browser
    if cfg.pool_name:
        adapter = PoolBrowserAdapter(kernel, pool_name=cfg.pool_name)
        adapter.acquire()
        console.print(f"[green]✓[/] Acquired browser from pool: {cfg.pool_name}")
    else:
        browser = kernel.browsers.create(stealth=True, headless=cfg.headless)
        from core.browser import KernelBrowserAdapter
        adapter = KernelBrowserAdapter(kernel, browser.session_id)
        console.print(f"[green]✓[/] Created browser: {browser.session_id}")
        if hasattr(browser, "browser_live_view_url") and browser.browser_live_view_url:
            console.print(f"    Live view: {browser.browser_live_view_url}")

    # Initialize agent
    agent = QwenAgent(AgentConfig(model=cfg.model))
    console.print(f"[green]✓[/] Agent initialized with {cfg.model}")

    # Navigate to URL
    console.print(f"\n[bold]Navigating to {cfg.url}...[/]")
    adapter.navigate(cfg.url)

    # Run agent loop
    console.print(f"\n[bold]Running agent (max {cfg.max_steps} steps)...[/]\n")

    screenshots = []
    action_history = []

    for step in range(1, cfg.max_steps + 1):
        t0 = time.time()

        # Capture screenshot
        screenshot = adapter.capture_screenshot()
        screenshots.append(screenshot)

        # Get agent action
        action = agent.predict(cfg.task, screenshot)
        t_inference = time.time() - t0

        if action is None:
            console.print(f"[yellow]Step {step}: Failed to parse action[/]")
            break

        action_desc = action.to_description()
        if action.model_description:
            action_desc = f"{action.model_description} ({action_desc})"
        action_history.append(action_desc)

        console.print(f"[blue]Step {step}[/] ({t_inference:.1f}s): {action_desc}")

        # Check for terminal action
        if getattr(action, "is_terminal", False):
            console.print(f"\n[green]Agent terminated: {action_desc}[/]")
            break

        # Execute action
        t1 = time.time()
        baseline = adapter.capture_screenshot()
        should_continue = adapter.execute_action(action)
        if not should_continue:
            break

        # Wait for screen to settle
        if not getattr(action, "skip_screen_settle", False):
            adapter.wait_for_screen_settle(baseline=baseline)
        t_action = time.time() - t1

        console.print(f"         Action executed ({t_action:.1f}s)")
    else:
        console.print(f"\n[yellow]Max steps ({cfg.max_steps}) reached[/]")

    # Cleanup browser
    if cfg.pool_name:
        adapter.release()
    else:
        kernel.browsers.delete(id=adapter.session_id)

    # Print action history
    console.print("\n[bold]Action History:[/]")
    table = Table(show_header=True)
    table.add_column("Step", style="dim")
    table.add_column("Action")
    for i, action_desc in enumerate(action_history, 1):
        table.add_row(str(i), action_desc)
    console.print(table)

    # WebJudge evaluation
    if cfg.webjudge and screenshots:
        console.print("\n[bold]Running WebJudge evaluation...[/]")

        webjudge = WebJudge(model=cfg.webjudge_model)
        trajectory = Trajectory(
            task_id="run_agent",
            task=cfg.task,
            action_history=action_history,
            screenshots=screenshots,
            initial_url=cfg.url,
        )

        result = await webjudge.evaluate(trajectory)

        status = "[green]SUCCESS[/]" if result.success else "[red]FAILURE[/]"
        console.print(f"\nWebJudge Result: {status} (score={result.score})")
        console.print(f"\n[dim]Key Points:[/]\n{result.key_points}")
        console.print(f"\n[dim]Reasoning:[/]\n{result.reasoning[:500]}...")

    console.print("\n[green]Done![/]")
    return 0


def main() -> int:
    """Entry point."""
    cfg = parse_args()

    # Setup environment
    try:
        setup_environment()
    except EnvironmentError as e:
        console.print(f"[yellow]⚠ {e}[/]")

    print_config(cfg)

    if cfg.dry_run:
        console.print("\n[yellow]Dry run mode - not executing[/]")
        return 0

    return asyncio.run(run_agent(cfg))


if __name__ == "__main__":
    sys.exit(main())


