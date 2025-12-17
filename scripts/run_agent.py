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
    RAINDROP_WRITE_KEY: Optional, enables Raindrop AI tracking
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Literal

# Suppress noisy httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

from cuid2 import cuid_wrapper
from kernel import Kernel
from PIL import Image
import raindrop.analytics as raindrop
from raindrop.models import Attachment
from rich.console import Console
from rich.table import Table


# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

# Generate cuid2 IDs
cuid2 = cuid_wrapper()

# Track whether raindrop is enabled
_raindrop_enabled = False

from core import (
    Action,
    AgentConfig,
    KernelBrowserAdapter,
    QwenAgent,
    Trajectory,
    WebJudge,
    WebJudgeResult,
    encode_image,
    setup_environment,
)

console = Console()

DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"


def _image_to_data_url(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64 data URL for Raindrop attachments."""
    b64 = encode_image(image, format=format)
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{b64}"


AttachmentRole = Literal["input", "output", "context"]


def _make_image_attachment(
    image: Image.Image,
    name: str,
    role: AttachmentRole = "input",
) -> Attachment:
    """Create a Raindrop image attachment from a PIL Image."""
    return Attachment(
        type="image",
        value=_image_to_data_url(image),
        name=name,
        role=role,
    )


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


def _init_raindrop() -> bool:
    """Initialize Raindrop if RAINDROP_WRITE_KEY is set. Returns True if enabled."""
    global _raindrop_enabled
    write_key = os.getenv("RAINDROP_WRITE_KEY")
    if not write_key:
        return False

    raindrop.init(write_key, tracing_enabled=True)
    _raindrop_enabled = True
    return True


def _shutdown_raindrop() -> None:
    """Flush and shutdown Raindrop if it was initialized."""
    if _raindrop_enabled:
        try:
            raindrop.flush()
            raindrop.shutdown()
        except Exception as e:
            console.print(f"[yellow]Warning: Raindrop shutdown error: {e}[/]")


# --- Raindrop-decorated functions (no-op when tracing disabled) ---


@raindrop.tool("navigate")
def _navigate(adapter: KernelBrowserAdapter, url: str) -> Image.Image:
    """Navigate to URL."""
    return adapter.navigate(url)


@raindrop.task("webjudge")
async def _run_webjudge(webjudge: WebJudge, trajectory: Trajectory) -> WebJudgeResult:
    """Run WebJudge evaluation."""
    return await webjudge.evaluate(trajectory)


@raindrop.task("agent_loop")
@raindrop.interaction("agent_loop")
def _run_agent_loop(
    cfg: RunConfig,
    agent: QwenAgent,
    adapter: KernelBrowserAdapter,
    screenshots: list[Image.Image],
    step_timings: list[tuple[int, str, float, float, float]],
    convo_id: str,
) -> tuple[str, str]:
    """Run the agent loop, returning (final_action_desc, termination_reason)."""
    final_action_desc = step_timings[0][1] if step_timings else "none"
    termination_reason = "max_steps"

    # Steps start at 2 since navigation is step 1
    for step in range(2, cfg.max_steps + 2):
        t_step_start = time.time()

        # Use the latest screenshot (result of previous action) for prediction
        screenshot = screenshots[-1]

        # Create input attachments for this step
        input_attachments = [_make_image_attachment(screenshot, f"step_{step}_input", "input")]

        # Begin a step interaction with the screenshot as input
        step_interaction = None
        if _raindrop_enabled:
            step_interaction = raindrop.begin(
                user_id=os.getenv("USER") or "system",
                event="agent_loop",
                input=f"Step {step}: Predict next action for task: {cfg.task}",
                convo_id=convo_id,
                attachments=input_attachments,
                properties={
                    "step": step,
                    "model": cfg.model,
                },
            )

        # Get agent action
        t_predict_start = time.time()
        action = agent.predict(cfg.task, screenshot)
        t_predict = time.time() - t_predict_start

        if action is None:
            console.print(f"[yellow]Step {step}: Failed to parse action[/]")
            if step_interaction:
                step_interaction.finish(output="Failed to parse action")
            termination_reason = "parse_failure"
            break

        action_desc = action.to_description()
        if action.model_description:
            action_desc = f"{action.model_description} ({action_desc})"
        final_action_desc = action_desc

        # Check for terminal action
        if getattr(action, "is_terminal", False):
            t_total = time.time() - t_step_start
            step_timings.append((step, action_desc, t_total, t_predict, 0.0))
            console.print(
                f"[blue]Step {step}[/] [dim]total={t_total:.1f}s predict={t_predict:.1f}s[/]: "
                f"{action_desc}"
            )
            console.print(f"\n[green]Agent terminated: {action_desc}[/]")
            # Terminal actions don't change state, so current screenshot is the result
            screenshots.append(screenshot)

            if step_interaction:
                step_interaction.finish(
                    output=action_desc,
                    properties={"is_terminal": True, "predict_time": t_predict},
                )

            termination_reason = "terminal_action"
            break

        # For click actions, create annotated image as output attachment
        output_attachments = []
        annotated_image = action.overlay_on_image(screenshot)
        if annotated_image is not None:
            output_attachments.append(
                _make_image_attachment(annotated_image, f"step_{step}_click_overlay", "output")
            )

        # Execute action and wait for UI to settle (wrapped in a span for tracing)
        t_exec_start = time.time()
        exec_failed = False

        with raindrop.tool_span("execute_action") as span:
            span.record_input({"action": action_desc, "step": step})

            baseline = adapter.capture_screenshot()
            should_continue = adapter.execute_action(action)

            if not should_continue:
                span.record_output({"success": False, "reason": "action_failed"})
                exec_failed = True
            else:
                # Wait for screen to settle
                if not getattr(action, "skip_screen_settle", False):
                    adapter.wait_for_screen_settle(baseline=baseline)

                new_screenshot = adapter.capture_screenshot()
                screenshots.append(new_screenshot)
                span.record_output({"success": True})

        t_exec = time.time() - t_exec_start

        if exec_failed:
            if step_interaction:
                step_interaction.finish(output=f"Action failed: {action_desc}")
            termination_reason = "action_failure"
            break
        t_total = time.time() - t_step_start
        step_timings.append((step, action_desc, t_total, t_predict, t_exec))

        # Finish the step interaction with output attachments
        if step_interaction:
            if output_attachments:
                step_interaction.add_attachments(output_attachments)
            step_interaction.finish(
                output=action_desc,
                properties={
                    "predict_time": t_predict,
                    "exec_time": t_exec,
                    "total_time": t_total,
                },
            )

        console.print(
            f"[blue]Step {step}[/] [dim]total={t_total:.1f}s predict={t_predict:.1f}s "
            f"exec={t_exec:.1f}s[/]: {action_desc}"
        )
    else:
        console.print(f"\n[yellow]Max steps ({cfg.max_steps}) reached[/]")

    return final_action_desc, termination_reason


async def run_agent(cfg: RunConfig) -> int:
    """Run the agent."""
    # Validate environment
    if not os.getenv("KERNEL_API_KEY"):
        console.print("[red]✗ KERNEL_API_KEY not set[/]")
        return 1

    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]✗ OPENROUTER_API_KEY not set[/]")
        return 1

    # Initialize Raindrop (optional)
    raindrop_enabled = _init_raindrop()
    if raindrop_enabled:
        console.print("[green]✓[/] Raindrop tracking enabled")
    else:
        console.print("[dim]ℹ Raindrop tracking disabled (no RAINDROP_WRITE_KEY)[/]")

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig: int, frame: object) -> None:
        console.print("\n[yellow]Interrupted, cleaning up...[/]")
        _shutdown_raindrop()
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Generate conversation ID for Raindrop tracking
    convo_id = cuid2()

    # Raindrop interaction (will be set if raindrop is enabled)
    interaction = None

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

        if adapter.live_view_url:
            console.print(f"    Live view: {adapter.live_view_url}")

        # Initialize agent
        agent = QwenAgent(AgentConfig(model=cfg.model))
        console.print(f"[green]✓[/] Agent initialized with {cfg.model}")

        # Start Raindrop interaction
        if _raindrop_enabled:
            interaction = raindrop.begin(
                user_id=os.getenv("USER") or "system",
                event="run_agent",
                input=cfg.task,
                convo_id=convo_id,
                properties={
                    "model": cfg.model,
                    "url": cfg.url,
                    "task": cfg.task,
                    "pool_name": cfg.pool_name,
                    "headless": cfg.headless,
                    "webjudge": cfg.webjudge,
                    "webjudge_model": cfg.webjudge_model,
                    "max_steps": cfg.max_steps,
                },
            )
            console.print(f"[dim]ℹ Raindrop convo_id: {convo_id}[/]")

        # Navigate to URL
        console.print(f"\n[bold]Navigating to {cfg.url}...[/]")
        t_nav_start = time.time()
        initial_screenshot = _navigate(adapter, cfg.url)
        t_nav = time.time() - t_nav_start
        console.print(f"[green]✓[/] Navigate: {t_nav:.1f}s")

        # Record the navigation action so the agent knows about it
        nav_action_desc = f"navigate({cfg.url})"
        agent.record_prior_action(nav_action_desc)

        # Create navigation step interaction with output screenshot
        if _raindrop_enabled:
            nav_interaction = raindrop.begin(
                user_id=os.getenv("USER") or "system",
                event="agent_loop",
                input=f"Navigate to {cfg.url}",
                convo_id=convo_id,
                properties={"step": 1, "action": "navigate"},
            )
            nav_interaction.add_attachments([
                _make_image_attachment(initial_screenshot, "step_1_output", "output")
            ])
            nav_interaction.finish(
                output=nav_action_desc,
                properties={"exec_time": t_nav},
            )

        # Run agent loop
        console.print(f"\n[bold]Running agent (max {cfg.max_steps} steps)...[/]\n")

        # Track screenshots for WebJudge evaluation
        screenshots = [initial_screenshot]

        # Track step timings for summary: (step, action_desc, total, predict, exec)
        # Navigation is step 1, no predict time (it's a setup action)
        step_timings: list[tuple[int, str, float, float, float]] = [
            (1, nav_action_desc, t_nav, 0.0, t_nav)
        ]

        # Run the agent loop (decorated with @raindrop.interaction)
        final_action_desc, termination_reason = _run_agent_loop(
            cfg, agent, adapter, screenshots, step_timings, convo_id
        )

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
        for step, action_desc, t_total, t_predict, t_exec in step_timings:
            table.add_row(
                str(step),
                action_desc,
                f"{t_total:.1f}s",
                f"{t_predict:.1f}s" if t_predict > 0 else "-",
                f"{t_exec:.1f}s" if t_exec > 0 else "-",
            )
        console.print(table)

        # Flush raindrop before WebJudge so agent loop data is visible immediately
        if _raindrop_enabled:
            raindrop.flush()

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

            result = await _run_webjudge(webjudge, trajectory)

            status = "[green]SUCCESS[/]" if result.success else "[red]FAILURE[/]"
            console.print(f"\nWebJudge Result: {status} (score={result.score})")
            console.print(f"\n[dim]Key Points:[/]\n{result.key_points}")
            console.print(f"\n[dim]Reasoning:[/]\n{result.reasoning[:500]}...")

            # Track webjudge result as a signal on the interaction
            if _raindrop_enabled and interaction is not None:
                sentiment = "POSITIVE" if result.success else "NEGATIVE"
                raindrop.track_signal(
                    event_id=interaction.id,
                    name="webjudge_result",
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
                    "total_steps": len(step_timings),
                },
            )

        return 0

    finally:
        _shutdown_raindrop()


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
