#!/usr/bin/env python3
"""
Run a computer use agent on a single task.

This script runs an OSWorld-style VLM agent against a real website
using Kernel browsers. Useful for testing and debugging agents.

Usage:
    # Run on a random task from the environment's tasks.jsonl
    uv run python -m scripts.run_agent --env osworld --random

    # Run on a specific task by ID
    uv run python -m scripts.run_agent --env agent_auth --id lknzafzfvcdtd72cq6eojgew

    # Ad-hoc mode (provide URL and task directly)
    uv run python -m scripts.run_agent \\
        --url https://github.com \\
        --task "Navigate to the sign in page"

    # Ad-hoc with specific environment config
    uv run python -m scripts.run_agent --env agent_auth \\
        --url https://github.com \\
        --task "Find the login form"

    # With WebJudge evaluation
    uv run python -m scripts.run_agent --env osworld --random --webjudge

    # Use a different model
    uv run python -m scripts.run_agent --env osworld --random \\
        --model qwen/qwen3-vl-30b-a3b-instruct

Environment Variables:
    KERNEL_API_KEY: Required for Kernel browser API
    OPENROUTER_API_KEY: Required for VLM inference and WebJudge
    RAINDROP_WRITE_KEY: Optional, enables Raindrop AI tracking
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Callable

# Suppress noisy httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

from kernel import Kernel
from PIL import Image
import raindrop.analytics as raindrop
from rich.console import Console
from rich.table import Table


# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from core import (
    Action,
    AgentConfig,
    KernelBrowserAdapter,
    QwenAgent,
    StepResult,
    Trajectory,
    WebJudge,
    run_agent_loop,
    setup_environment,
)
from core.tracking import (
    generate_id,
    init_raindrop,
    is_raindrop_enabled,
    make_image_attachment,
    run_webjudge_with_tracking,
    shutdown_raindrop,
)

console = Console()

DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"

# Available environments
AVAILABLE_ENVS = ["osworld", "agent_auth"]


@dataclass
class RunConfig:
    """Configuration for agent run."""

    # Environment selection
    env: str = "osworld"

    # Task selection (mutually exclusive modes)
    url: str | None = None  # Ad-hoc mode: direct URL
    task: str | None = None  # Ad-hoc mode: direct task
    task_id: str | None = None  # Task file mode: specific task ID
    random_task: bool = False  # Task file mode: pick random task
    task_file: str | None = None  # Override default task file

    # Agent config
    model: str = DEFAULT_MODEL
    max_steps: int = 10

    # Browser config
    pool_name: str | None = None  # Use pool if set, else create browser
    headless: bool = False

    # Evaluation
    webjudge: bool = False
    webjudge_model: str = "openai/gpt-5-mini"

    # Control
    dry_run: bool = False


def parse_args() -> RunConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a computer use agent on a task",
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

    # Task selection - ad-hoc mode
    parser.add_argument("--url", default=None, help="Starting URL (ad-hoc mode)")
    parser.add_argument("--task", default=None, help="Task instruction (ad-hoc mode)")

    # Task selection - task file mode
    parser.add_argument("--id", dest="task_id", default=None, help="Task ID from tasks.jsonl")
    parser.add_argument("--random", dest="random_task", action="store_true", help="Pick a random task")
    parser.add_argument("--task-file", default=None, help="Path to task JSONL file (overrides env default)")

    # Agent config
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Tinker checkpoint path (e.g., tinker://...sampler_weights/000010). Overrides --model.",
    )
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps (default: 10)")

    # Browser config
    parser.add_argument("--pool-name", default=None, help="Browser pool name (optional)")
    parser.add_argument("--headless", action="store_true", help="Run headless")

    # Evaluation
    parser.add_argument("--webjudge", action="store_true", help="Evaluate with WebJudge")
    parser.add_argument("--webjudge-model", default="openai/gpt-5-mini", help="WebJudge model")

    # Control
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")

    args = parser.parse_args()

    # Use checkpoint path as model if provided
    model = args.checkpoint if args.checkpoint else args.model

    return RunConfig(
        env=args.env,
        url=args.url,
        task=args.task,
        task_id=args.task_id,
        random_task=args.random_task,
        task_file=args.task_file,
        model=model,
        max_steps=args.max_steps,
        pool_name=args.pool_name,
        headless=args.headless,
        webjudge=args.webjudge,
        webjudge_model=args.webjudge_model,
        dry_run=args.dry_run,
    )


def get_env_config(cfg: RunConfig) -> tuple[str, list[type[Action]], str]:
    """
    Get environment-specific configuration.

    Returns:
        Tuple of (system_prompt, extra_actions, default_task_file)
    """
    if cfg.env == "osworld":
        from examples.osworld.config import get_osworld_system_prompt
        from examples.osworld.actions import OSWORLD_ACTIONS

        return (
            get_osworld_system_prompt(),
            OSWORLD_ACTIONS,
            "examples/osworld/tasks.jsonl",
        )
    elif cfg.env == "agent_auth":
        from examples.agent_auth.config import get_agent_auth_system_prompt
        from examples.agent_auth.actions import AGENT_AUTH_ACTIONS

        return (
            get_agent_auth_system_prompt(),
            AGENT_AUTH_ACTIONS,
            "examples/agent_auth/tasks.jsonl",
        )
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")


def load_task_from_file(
    cfg: RunConfig,
    default_task_file: str,
) -> tuple[str, str, str]:
    """
    Load a task from the task file.

    Returns:
        Tuple of (task_id, url, task_description)
    """
    if cfg.env == "osworld":
        from examples.osworld.dataset import load_tasks
    elif cfg.env == "agent_auth":
        from examples.agent_auth.dataset import load_tasks
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")

    task_file = cfg.task_file or default_task_file
    tasks = load_tasks(task_file)

    if not tasks:
        raise ValueError(f"No tasks found in {task_file}")

    if cfg.task_id:
        # Find specific task by ID
        matching = [t for t in tasks if t.id == cfg.task_id]
        if not matching:
            available_ids = [t.id for t in tasks[:10]]
            raise ValueError(
                f"Task ID '{cfg.task_id}' not found. "
                f"Available IDs (first 10): {available_ids}"
            )
        selected_task = matching[0]
    else:
        # Random task
        selected_task = random.choice(tasks)

    return selected_task.id, selected_task.initial_url, selected_task.task


def print_config(cfg: RunConfig, url: str, task: str, task_id: str | None = None) -> None:
    """Print configuration table."""
    table = Table(title="Agent Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    table.add_row("Environment", cfg.env)
    if task_id:
        table.add_row("Task ID", task_id)
    table.add_row("URL", url)
    table.add_row("Task", task[:60] + "..." if len(task) > 60 else task)
    table.add_row("Model", cfg.model)
    table.add_row("Max Steps", str(cfg.max_steps))
    table.add_row("Browser Pool", cfg.pool_name or "create new")
    table.add_row("WebJudge", "enabled" if cfg.webjudge else "disabled")

    console.print(table)


# --- Raindrop-decorated functions (no-op when tracing disabled) ---


@raindrop.tool("navigate")
def _navigate(adapter: KernelBrowserAdapter, url: str) -> Image.Image:
    """Navigate to URL."""
    return adapter.navigate(url)


def _create_step_callbacks(
    model: str,
    convo_id: str,
    nav_step_offset: int = 1,
) -> tuple[
    Callable[[int, Image.Image], None],
    Callable[[int, StepResult], None],
    Callable[[int, Action, Image.Image], None],
    dict,
]:
    """
    Create callbacks for the agent loop that handle logging and Raindrop integration.

    Returns:
        Tuple of (on_step_start, on_step_complete, on_action_overlay, shared_state)
    """
    # Shared state for passing data between callbacks
    shared_state: dict = {
        "step_interaction": None,
        "output_attachments": [],
    }

    def on_step_start(step: int, screenshot: Image.Image) -> None:
        """Called at the start of each step."""
        display_step = step + nav_step_offset

        if is_raindrop_enabled():
            input_attachments = [make_image_attachment(screenshot, f"step_{display_step}_input", "input")]
            shared_state["step_interaction"] = raindrop.begin(
                user_id=os.getenv("USER") or "system",
                event="agent_loop",
                input=f"Step {display_step}: Predict next action",
                convo_id=convo_id,
                attachments=input_attachments,
                properties={
                    "step": display_step,
                    "model": model,
                },
            )
        shared_state["output_attachments"] = []

    def on_action_overlay(step: int, action: Action, overlay: Image.Image) -> None:
        """Called when an action overlay image is generated."""
        display_step = step + nav_step_offset
        shared_state["output_attachments"].append(
            make_image_attachment(overlay, f"step_{display_step}_click_overlay", "output")
        )

    def on_step_complete(step: int, result: StepResult) -> None:
        """Called after each step completes."""
        display_step = step + nav_step_offset
        step_interaction = shared_state.get("step_interaction")
        output_attachments = shared_state.get("output_attachments", [])

        # Handle error/parse failure
        if result.error or result.action is None:
            console.print(f"[yellow]Step {display_step}: {result.error or 'Failed to parse action'}[/]")
            if step_interaction:
                step_interaction.finish(output=result.error or "Failed to parse action")
            return

        action_desc = result.action_desc

        # Log the step
        if result.is_terminal:
            console.print(
                f"[blue]Step {display_step}[/] [dim]total={result.total_time:.1f}s "
                f"predict={result.predict_time:.1f}s[/]: {action_desc}"
            )
            console.print(f"\n[green]Agent terminated: {action_desc}[/]")

            if step_interaction:
                step_interaction.finish(
                    output=action_desc,
                    properties={"is_terminal": True, "predict_time": result.predict_time},
                )
        else:
            console.print(
                f"[blue]Step {display_step}[/] [dim]total={result.total_time:.1f}s "
                f"predict={result.predict_time:.1f}s exec={result.exec_time:.1f}s[/]: {action_desc}"
            )

            if step_interaction:
                if output_attachments:
                    step_interaction.add_attachments(output_attachments)
                step_interaction.finish(
                    output=action_desc,
                    properties={
                        "predict_time": result.predict_time,
                        "exec_time": result.exec_time,
                        "total_time": result.total_time,
                    },
                )

    return on_step_start, on_step_complete, on_action_overlay, shared_state


async def run_agent(
    cfg: RunConfig,
    url: str,
    task: str,
    task_id: str | None,
    system_prompt: str,
    extra_actions: list[type[Action]],
) -> int:
    """Run the agent."""
    # Validate environment
    if not os.getenv("KERNEL_API_KEY"):
        console.print("[red]✗ KERNEL_API_KEY not set[/]")
        return 1

    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]✗ OPENROUTER_API_KEY not set[/]")
        return 1

    # Initialize Raindrop (optional)
    raindrop_enabled = init_raindrop()
    if raindrop_enabled:
        console.print("[green]✓[/] Raindrop tracking enabled")
    else:
        console.print("[dim]ℹ Raindrop tracking disabled (no RAINDROP_WRITE_KEY)[/]")

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig: int, frame: object) -> None:
        console.print("\n[yellow]Interrupted, cleaning up...[/]")
        shutdown_raindrop()
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Generate conversation ID for Raindrop tracking
    convo_id = generate_id()

    # Raindrop interaction (will be set if raindrop is enabled)
    interaction = None

    adapter: KernelBrowserAdapter | None = None

    try:
        # Initialize Kernel
        kernel = Kernel()
        console.print("[green]✓[/] Connected to Kernel")

        # Create or acquire browser
        if cfg.pool_name:
            browser = kernel.browser_pools.acquire(cfg.pool_name)
            adapter = KernelBrowserAdapter(kernel, browser)
            console.print(f"[green]✓[/] Acquired browser from pool: {cfg.pool_name}")
        else:
            browser = kernel.browsers.create(stealth=True, headless=cfg.headless)
            adapter = KernelBrowserAdapter(kernel, browser)
            console.print(f"[green]✓[/] Created browser: {browser.session_id}")

        # Start heartbeat to keep browser alive during long VLM inference
        adapter.start_heartbeat_sync()

        if adapter.live_view_url:
            console.print(f"    Live view: {adapter.live_view_url}")

        # Initialize agent with environment-specific config
        agent_config = AgentConfig(
            model=cfg.model,
            system_prompt=system_prompt,
            extra_actions=extra_actions,
        )
        agent = QwenAgent(config=agent_config)
        console.print(f"[green]✓[/] Agent initialized with {cfg.model} (env={cfg.env})")

        # Start Raindrop interaction
        if is_raindrop_enabled():
            interaction = raindrop.begin(
                user_id=os.getenv("USER") or "system",
                event="run_agent",
                input=task,
                convo_id=convo_id,
                properties={
                    "env": cfg.env,
                    "task_id": task_id,
                    "model": cfg.model,
                    "url": url,
                    "task": task,
                    "pool_name": cfg.pool_name,
                    "headless": cfg.headless,
                    "webjudge": cfg.webjudge,
                    "webjudge_model": cfg.webjudge_model,
                    "max_steps": cfg.max_steps,
                },
            )
            console.print(f"[dim]ℹ Raindrop convo_id: {convo_id}[/]")

        # Navigate to URL
        console.print(f"\n[bold]Navigating to {url}...[/]")
        t_nav_start = time.time()
        initial_screenshot = _navigate(adapter, url)
        t_nav = time.time() - t_nav_start
        console.print(f"[green]✓[/] Navigate: {t_nav:.1f}s")

        # Record the navigation action so the agent knows about it
        nav_action_desc = f"navigate({url})"
        agent.record_prior_action(nav_action_desc)

        # Create navigation step interaction with output screenshot
        if is_raindrop_enabled():
            nav_interaction = raindrop.begin(
                user_id=os.getenv("USER") or "system",
                event="agent_loop",
                input=f"Navigate to {url}",
                convo_id=convo_id,
                properties={"step": 1, "action": "navigate"},
            )
            nav_interaction.add_attachments([
                make_image_attachment(initial_screenshot, "step_1_output", "output")
            ])
            nav_interaction.finish(
                output=nav_action_desc,
                properties={"exec_time": t_nav},
            )

        # Run agent loop
        console.print(f"\n[bold]Running agent (max {cfg.max_steps} steps)...[/]\n")

        # Create callbacks for logging and Raindrop integration
        on_step_start, on_step_complete, on_action_overlay, _ = _create_step_callbacks(
            model=cfg.model,
            convo_id=convo_id,
            nav_step_offset=1,  # Navigation is step 1
        )

        # Run the shared agent loop in a thread pool so heartbeat can run
        loop_result = await asyncio.to_thread(
            run_agent_loop,
            agent=agent,
            adapter=adapter,
            task=task,
            initial_screenshot=initial_screenshot,
            max_steps=cfg.max_steps,
            on_step_start=on_step_start,
            on_step_complete=on_step_complete,
            on_action_overlay=on_action_overlay,
        )

        # Extract results
        screenshots = loop_result.screenshots
        termination_reason = loop_result.termination_reason
        final_action_desc = (
            loop_result.step_results[-1].action_desc
            if loop_result.step_results
            else nav_action_desc
        )

        # Log termination
        if termination_reason == "max_steps":
            console.print(f"\n[yellow]Max steps ({cfg.max_steps}) reached[/]")

        # Cleanup browser
        if cfg.pool_name:
            kernel.browser_pools.release(cfg.pool_name, session_id=adapter.session_id)
        else:
            kernel.browsers.delete_by_id(id=adapter.session_id)

        # Print action history with timings
        action_history = agent.get_action_history()
        console.print("\n[bold]Action History:[/]")
        table = Table(show_header=True)
        table.add_column("Step", style="dim")
        table.add_column("Action")
        table.add_column("Total", style="cyan", justify="right")
        table.add_column("Predict", style="yellow", justify="right")
        table.add_column("Exec", style="green", justify="right")

        # Add navigation step
        table.add_row("1", nav_action_desc, f"{t_nav:.1f}s", "-", f"{t_nav:.1f}s")

        # Add loop steps
        for result in loop_result.step_results:
            display_step = result.step + 1  # Offset for navigation
            table.add_row(
                str(display_step),
                result.action_desc,
                f"{result.total_time:.1f}s",
                f"{result.predict_time:.1f}s" if result.predict_time > 0 else "-",
                f"{result.exec_time:.1f}s" if result.exec_time > 0 else "-",
            )
        console.print(table)

        # Flush raindrop before WebJudge so agent loop data is visible immediately
        if is_raindrop_enabled():
            raindrop.flush()

        # WebJudge evaluation
        if cfg.webjudge and screenshots:
            console.print("\n[bold]Running WebJudge evaluation...[/]")

            webjudge = WebJudge(model=cfg.webjudge_model)
            trajectory = Trajectory(
                task_id=task_id or "run_agent",
                task=task,
                action_history=action_history,
                screenshots=screenshots,
                initial_url=url,
            )

            result = await run_webjudge_with_tracking(webjudge, trajectory)

            status = "[green]SUCCESS[/]" if result.success else "[red]FAILURE[/]"
            console.print(f"\nWebJudge Result: {status} (score={result.score})")
            console.print(f"\n[dim]Key Points:[/]\n{result.key_points}")
            console.print(f"\n[dim]Reasoning:[/]\n{result.reasoning[:500]}...")

            # Track webjudge result as a signal on the interaction
            if is_raindrop_enabled() and interaction is not None:
                signal_name = "webjudge_success" if result.success else "webjudge_failure"
                sentiment = "POSITIVE" if result.success else "NEGATIVE"
                raindrop.track_signal(
                    event_id=interaction.id,
                    name=signal_name,
                    signal_type="feedback",
                    sentiment=sentiment,
                    comment=f"Score: {result.score}. {result.key_points}",
                    properties={
                        "success": result.success,
                        "score": result.score,
                        "model": cfg.webjudge_model,
                    },
                )

        console.print("\n[green]Done![/]")

        # Finish Raindrop interaction
        if interaction is not None:
            interaction.finish(
                output=f"Completed: {final_action_desc}",
                properties={
                    "termination_reason": termination_reason,
                    "total_steps": loop_result.steps_completed + 1,  # +1 for navigation
                },
            )

        return 0

    finally:
        # Stop heartbeat thread
        if adapter is not None:
            adapter.stop_heartbeat_sync()
        shutdown_raindrop()


def main() -> int:
    """Entry point."""
    cfg = parse_args()

    # Setup environment
    try:
        setup_environment()
    except EnvironmentError as e:
        console.print(f"[yellow]⚠ {e}[/]")

    # Get environment config
    system_prompt, extra_actions, default_task_file = get_env_config(cfg)

    # Determine task source
    task_id: str | None = None

    if cfg.url and cfg.task:
        # Ad-hoc mode: URL and task provided directly
        url = cfg.url
        task = cfg.task
    elif cfg.task_id or cfg.random_task:
        # Task file mode: load from tasks.jsonl
        task_id, url, task = load_task_from_file(cfg, default_task_file)
    else:
        # Default to random task from env
        console.print("[dim]No task specified, picking random task from environment...[/]")
        cfg.random_task = True
        task_id, url, task = load_task_from_file(cfg, default_task_file)

    # Print config
    print_config(cfg, url, task, task_id)

    if cfg.dry_run:
        console.print("\n[yellow]Dry run mode - not executing[/]")
        return 0

    return asyncio.run(run_agent(cfg, url, task, task_id, system_prompt, extra_actions))


if __name__ == "__main__":
    sys.exit(main())
