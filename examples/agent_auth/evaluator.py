"""
Agent Auth Sampling Evaluator for Tinker.

Uses Tinker's native SamplingClient for VLM inference, enabling evaluation
of finetuned checkpoints that don't work with the OpenAI-compatible endpoint.

This evaluator:
1. Uses tinker.SamplingClient for model inference (not OpenAI SDK)
2. Runs browser episodes with Kernel browser pools
3. Uses WebJudge for trajectory scoring
4. Returns aggregated metrics

Usage:
    from examples.agent_auth.evaluator import AgentAuthSamplingEvaluator

    evaluator = AgentAuthSamplingEvaluator(
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        tasks=tasks,
        pool_name="eval-browser-pool",
    )

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        model_path="tinker://...",
    )

    metrics = await evaluator(sampling_client)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import tinker
from kernel import Kernel
from PIL import Image
from rich.console import Console
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ImagePart, TextPart
from tinker_cookbook.tokenizer_utils import get_tokenizer

from core.actions import TerminateAction, parse_action_from_response
from core.browser import KernelBrowserAdapter
from core.reward_models.webjudge import Trajectory as WebJudgeTrajectory
from core.reward_models.webjudge import WebJudge
from core.utils import resize_image

from .actions import AGENT_AUTH_ACTIONS, FoundInputsAction
from .config import get_agent_auth_system_prompt
from .dataset import AgentAuthTask

logger = logging.getLogger(__name__)
console = Console()


def ts() -> str:
    """Return short timestamp [HH:MM:SS] for log lines."""
    return datetime.now().strftime("[%H:%M:%S]")


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    pool_name: str = "eval-browser-pool"
    acquire_timeout_seconds: int = 60
    max_steps: int = 10
    image_max_size: int = 512
    max_tokens: int = 512
    temperature: float = 0.0
    concurrency: int = 4  # Number of concurrent episodes


@dataclass
class EpisodeResult:
    """Result from a single evaluation episode."""

    task_id: str
    domain: str
    success: bool
    score: float
    reasoning: str
    steps: int
    terminal_action: str | None
    action_history: list[str]
    duration_seconds: float
    error: str | None = None


class AgentAuthSamplingEvaluator(SamplingClientEvaluator):
    """
    Evaluator for Agent Auth tasks using Tinker's native sampling.

    Uses SamplingClient for VLM inference instead of OpenAI SDK,
    enabling evaluation of finetuned checkpoints.
    """

    def __init__(
        self,
        model_name: str,
        tasks: list[AgentAuthTask],
        config: EvalConfig | None = None,
        webjudge_model: str = "openai/gpt-5-mini",
        webjudge_enabled: bool = True,
        renderer_name: str = "qwen3_vl_instruct",
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: Base model name (e.g., "Qwen/Qwen3-VL-30B-A3B-Instruct")
            tasks: List of tasks to evaluate
            config: Evaluation configuration
            webjudge_model: Model to use for WebJudge scoring
            webjudge_enabled: Whether to use WebJudge (vs heuristic)
            renderer_name: Tinker renderer name for prompt building
        """
        self.model_name = model_name
        self.tasks = tasks
        self.config = config or EvalConfig()
        self.webjudge_model = webjudge_model
        self.webjudge_enabled = webjudge_enabled
        self.renderer_name = renderer_name

        # Initialize renderer
        tokenizer = get_tokenizer(model_name)
        image_processor = get_image_processor(model_name)
        self.renderer = renderers.get_renderer(renderer_name, tokenizer, image_processor)

        # System prompt
        self.system_prompt = get_agent_auth_system_prompt()

        # Will be set when __call__ is invoked
        self._sampling_client: tinker.SamplingClient | None = None
        self._kernel: Kernel | None = None
        self._webjudge: WebJudge | None = None

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run evaluation on all tasks and return aggregated metrics.

        Args:
            sampling_client: Tinker sampling client for model inference

        Returns:
            Dictionary of metrics (success_rate, avg_score, etc.)
        """
        self._sampling_client = sampling_client
        self._kernel = Kernel()

        # Initialize WebJudge
        if self.webjudge_enabled:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                self._webjudge = WebJudge(
                    model=self.webjudge_model,
                    api_key=api_key,
                )
                console.print(f"  ✓ WebJudge initialized ({self.webjudge_model})")
            else:
                console.print("  [yellow]⚠ OPENROUTER_API_KEY not set, WebJudge disabled[/]")
                self._webjudge = None

        console.print(f"\n[bold blue]Running evaluation on {len(self.tasks)} tasks...[/]")
        console.print(f"  Concurrency: {self.config.concurrency}")
        console.print(f"  Max steps: {self.config.max_steps}")

        # Run episodes with concurrency limit
        semaphore = asyncio.Semaphore(self.config.concurrency)
        results: list[EpisodeResult] = []

        async def run_with_semaphore(task: AgentAuthTask, idx: int) -> EpisodeResult:
            async with semaphore:
                return await self._run_episode(task, idx)

        # Create tasks
        coros = [run_with_semaphore(task, i) for i, task in enumerate(self.tasks)]

        # Run all episodes
        t_start = time.perf_counter()
        results = await asyncio.gather(*coros)
        total_time = time.perf_counter() - t_start

        # Compute metrics
        success_count = sum(1 for r in results if r.success)
        total_score = sum(r.score for r in results)
        avg_steps = sum(r.steps for r in results) / len(results) if results else 0
        error_count = sum(1 for r in results if r.error)

        metrics = {
            "success_rate": success_count / len(results) if results else 0.0,
            "avg_score": total_score / len(results) if results else 0.0,
            "total_success": float(success_count),
            "total_tasks": float(len(results)),
            "avg_steps": avg_steps,
            "error_rate": error_count / len(results) if results else 0.0,
            "total_time_seconds": total_time,
        }

        # Print summary
        console.print("\n" + "=" * 60)
        console.print("[bold]Evaluation Summary[/]")
        console.print("=" * 60)
        console.print(f"  Total Tasks: {len(results)}")
        console.print(f"  Success: {success_count} ({metrics['success_rate']*100:.1f}%)")
        console.print(f"  Avg Score: {metrics['avg_score']:.3f}")
        console.print(f"  Avg Steps: {avg_steps:.1f}")
        console.print(f"  Errors: {error_count}")
        console.print(f"  Total Time: {total_time:.1f}s")

        # Store results for external access
        self.results = results

        return metrics

    async def _run_episode(self, task: AgentAuthTask, idx: int) -> EpisodeResult:
        """Run a single evaluation episode."""
        t_start = time.perf_counter()
        task_label = f"{task.domain} ({task.id})"

        console.print(f"  {ts()} [dim]→ Starting:[/] {task_label}")

        # Episode state
        action_history: list[str] = []
        screenshots: list[Image.Image] = []
        step_count = 0
        terminal_action: str | None = None
        error: str | None = None
        final_action = None

        # Acquire browser
        adapter: KernelBrowserAdapter | None = None
        try:
            browser = self._kernel.browser_pools.acquire(
                self.config.pool_name,
                acquire_timeout_seconds=self.config.acquire_timeout_seconds,
            )
            adapter = KernelBrowserAdapter(self._kernel, browser, reset_on_init=True)
            adapter.start_heartbeat_sync(task_label=task_label)

            # Navigate to initial URL
            adapter.navigate(task.initial_url)
            action_history.append(f"Navigate to {task.initial_url}")

            # Capture initial screenshot
            screenshot = adapter.capture_screenshot()
            screenshot = resize_image(screenshot, max_size=self.config.image_max_size)
            screenshots.append(screenshot.copy())

            # Build initial conversation
            conversation: list[renderers.Message] = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        ImagePart(type="image", image=screenshot),
                        TextPart(type="text", text=f"Task: {task.task}"),
                    ],
                },
            ]

            # Run agent loop
            for step in range(1, self.config.max_steps + 1):
                step_count = step

                # Build prompt and sample
                model_input = self.renderer.build_generation_prompt(conversation)
                sampling_params = types.SamplingParams(
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    stop=self.renderer.get_stop_sequences(),
                )

                response = await self._sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                )

                # Parse response
                tokens = response.sequences[0].tokens
                response_msg, _ = self.renderer.parse_response(tokens)
                response_content = renderers.ensure_text(response_msg.get("content", ""))

                # Parse action
                browser_action = parse_action_from_response(
                    response_content, AGENT_AUTH_ACTIONS
                )

                if browser_action is None:
                    console.print(
                        f"    {ts()} [red]{task_label}: step={step} error=parse_failed[/]"
                    )
                    error = "Failed to parse action"
                    break

                # Record action
                action_desc = browser_action.to_description()
                action_history.append(action_desc)

                # Check for terminal action
                if getattr(browser_action, "is_terminal", False):
                    final_action = browser_action
                    terminal_action = action_desc
                    console.print(
                        f"    {ts()} [dim]{task_label}: step={step} action={action_desc} [done][/]"
                    )
                    break

                # Execute action
                try:
                    baseline = adapter.capture_screenshot()
                    should_continue = adapter.execute_action(browser_action)

                    if not should_continue:
                        final_action = browser_action
                        terminal_action = action_desc
                        break

                    # Wait for settle and capture new screenshot
                    if not getattr(browser_action, "skip_screen_settle", False):
                        adapter.wait_for_screen_settle(baseline=baseline)

                    screenshot = adapter.capture_screenshot()
                    screenshot = resize_image(screenshot, max_size=self.config.image_max_size)
                    screenshots.append(screenshot.copy())

                except Exception as e:
                    error = f"Browser error: {e}"
                    console.print(f"    {ts()} [red]{task_label}: step={step} error={error}[/]")
                    break

                # Update conversation for next step
                conversation.append({"role": "assistant", "content": response_content})
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            ImagePart(type="image", image=screenshot),
                        ],
                    }
                )

                console.print(
                    f"    {ts()} [dim]{task_label}: step={step} action={action_desc}[/]"
                )

            # Max steps reached
            if step_count >= self.config.max_steps and terminal_action is None:
                console.print(
                    f"    {ts()} [yellow]{task_label}: max_steps reached[/]"
                )

        except Exception as e:
            error = f"Episode error: {e}"
            console.print(f"    {ts()} [red]{task_label}: error={error}[/]")

        finally:
            # Release browser
            if adapter is not None:
                try:
                    adapter.stop_heartbeat_sync()
                    self._kernel.browser_pools.release(
                        self.config.pool_name,
                        session_id=adapter.session_id,
                        reuse=error is None,
                    )
                except Exception as e:
                    logger.warning(f"Failed to release browser: {e}")

        # Evaluate with WebJudge
        success = False
        score = 0.0
        reasoning = ""

        if self._webjudge is not None and screenshots:
            try:
                wj_trajectory = WebJudgeTrajectory(
                    task_id=task.id,
                    task=task.task,
                    action_history=action_history,
                    screenshots=screenshots,
                )
                wj_result = await self._webjudge.evaluate(wj_trajectory)
                success = wj_result.success
                score = wj_result.score
                reasoning = wj_result.reasoning
            except Exception as e:
                reasoning = f"WebJudge error: {e}"
        else:
            # Heuristic scoring
            if isinstance(final_action, FoundInputsAction):
                success = len(final_action.fields) > 0
                score = 1.0 if success else 0.0
                reasoning = f"Found {len(final_action.fields)} input fields"
            elif isinstance(final_action, TerminateAction):
                status = getattr(final_action, "status", "unknown")
                success = status == "success"
                score = 1.0 if success else 0.0
                reasoning = f"Terminated with status: {status}"
            else:
                reasoning = "No terminal action"

        duration = time.perf_counter() - t_start

        status_icon = "[green]✓[/]" if success else "[red]✗[/]"
        console.print(
            f"  {ts()} {status_icon} {task_label}: steps={step_count} "
            f"score={score:.2f} time={duration:.1f}s"
        )

        return EpisodeResult(
            task_id=task.id,
            domain=task.domain,
            success=success,
            score=score,
            reasoning=reasoning,
            steps=step_count,
            terminal_action=terminal_action,
            action_history=action_history,
            duration_seconds=round(duration, 2),
            error=error,
        )

    def get_results(self) -> list[EpisodeResult]:
        """Get detailed results from the last evaluation run."""
        return getattr(self, "results", [])

