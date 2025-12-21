"""
Agent Auth RL Environment for Tinker.

Implements Tinker's Env, EnvGroupBuilder, and RLDataset interfaces
for training a VLM web agent to find login forms.

This serves as a reference implementation for building custom RL
environments with Kernel browsers and WebJudge reward computation.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Sequence

import time
from datetime import datetime

import chz
import tinker
from kernel import Kernel
from PIL import Image
from rich.console import Console
from tinker_cookbook import renderers

console = Console()


def ts() -> str:
    """Return short timestamp [HH:MM:SS] for log lines."""
    return datetime.now().strftime("[%H:%M:%S]")
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ImagePart, TextPart
from tinker_cookbook.rl.types import (
    Action as TinkerAction,
)
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

from core.actions import TerminateAction, parse_action_from_response
from core.browser import KernelBrowserAdapter
from core.reward_models.webjudge import Trajectory as WebJudgeTrajectory
from core.reward_models.webjudge import WebJudge
from core.tracking import (
    begin_episode,
    finish_episode,
    generate_id,
    is_raindrop_enabled,
    track_webjudge_signal,
)
from core.utils import resize_image

from .actions import AGENT_AUTH_ACTIONS, FoundInputsAction
from .config import get_agent_auth_system_prompt
from .dataset import AgentAuthTask, load_tasks

logger = logging.getLogger(__name__)

# Evaluation criteria for authentication/login discovery tasks
AGENT_AUTH_EVALUATION_CRITERIA = """1. The agent must have navigated to an authentication page (login, sign-up, register, create account).
2. The agent must have identified input fields that are actually visible on the final page.
3. Many sites use progressive disclosure (showing email first, then password on the next step) - this is valid and should be considered successful.
4. The reported fields should match what is visible in the screenshots.
5. Do not penalize for "missing" fields that would only appear in later steps of a multi-step auth flow.
6. If the task asks for "first" input fields, only the initially visible fields need to be reported.
7. The agent should not fill in or submit any forms - just identify the fields."""

# Default model for agent auth training
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
RENDERER_NAME = "qwen3_vl_instruct"


@dataclass
class AgentAuthEnvConfig:
    """Configuration for AgentAuthEnv."""

    pool_name: str = "rl-browser-pool"
    acquire_timeout_seconds: int = 60
    max_steps: int = 5  # Max actions per episode
    image_max_size: int = 512  # Max dimension for screenshots
    max_screenshots_in_context: int = 3  # Limit screenshots to prevent context explosion


class AgentAuthEnv(Env):
    """
    Browser environment for agent auth training.

    Each episode:
    1. Acquires a browser from the pool
    2. Navigates to the target domain
    3. Runs observation-action loop until terminal or max_steps
    4. Releases browser back to pool

    Error Handling:
    - If browser becomes corrupted (Kernel errors), sets _browser_corrupted flag
    - Corrupted browsers return terminal state immediately
    - Cleanup is always called to release resources
    """

    def __init__(
        self,
        task: AgentAuthTask,
        kernel: Kernel,
        renderer: renderers.Renderer,
        config: AgentAuthEnvConfig,
        system_prompt: str | None = None,
    ):
        self.task = task
        self.kernel = kernel
        self.renderer = renderer
        self.config = config
        self.system_prompt = system_prompt or get_agent_auth_system_prompt()

        # State
        self.adapter: KernelBrowserAdapter | None = None
        self.step_count = 0
        self.action_history: list[str] = []
        self.screenshots: list[Image.Image] = []
        self.conversation: list[renderers.Message] = []
        self.last_response: str | None = None
        self.final_action = None

        # Error tracking
        self._browser_corrupted = False

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """
        Initialize environment: acquire browser, navigate, capture screenshot.

        If browser acquisition or initial operations fail, marks browser as
        corrupted and raises an exception (which will cause the group to be skipped).
        """
        try:
            # Acquire browser from pool
            browser = self.kernel.browser_pools.acquire(
                self.config.pool_name,
                acquire_timeout_seconds=self.config.acquire_timeout_seconds,
            )
            self.adapter = KernelBrowserAdapter(self.kernel, browser, reset_on_init=True)

            # Start heartbeat to keep browser alive during long VLM inference
            self.adapter.start_heartbeat_sync(task_label=self.task.domain)

            # Navigate to domain
            url = self.task.initial_url
            try:
                self.adapter.navigate(url)
            except Exception as e:
                logger.warning(f"Navigation to {url} failed: {e}")
                # Continue anyway - capture whatever state we're in

            # Capture initial screenshot
            screenshot = self.adapter.capture_screenshot()
            screenshot_resized = resize_image(screenshot, max_size=self.config.image_max_size)
            self.screenshots.append(screenshot_resized.copy())
            self.action_history.append(f"Navigate to {url}")
            console.print(f"  {ts()} [dim]{self.task.domain}: step=0 action=navigate url={url}[/]")

        except Exception as e:
            # Browser acquisition failed - set flag and cleanup
            self._browser_corrupted = True
            logger.warning(f"Browser initialization failed for {self.task.initial_url}: {e}")
            console.print(f"  {ts()} [red]{self.task.domain}: error=init_failed[/]")
            # Cleanup any partial state
            self.cleanup()
            # Raise exception to signal Tinker to skip this environment
            # Returning ModelInput.empty() causes "Empty prompt provided" error
            # because Tinker tries to sample before calling step()
            raise RuntimeError(f"Browser initialization failed for {self.task.domain}: {e}") from e

        # Build conversation
        self.conversation = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    ImagePart(type="image", image=screenshot_resized),
                    TextPart(type="text", text=f"Task: {self.task.task}"),
                ],
            },
        ]

        return self.renderer.build_generation_prompt(self.conversation), self.stop_condition

    async def step(self, action: TinkerAction) -> StepResult:
        """
        Execute an action step.

        Args:
            action: Token IDs from the policy (Tinker's Action type)

        Returns:
            StepResult with reward, done flag, next observation

        If browser is corrupted, raises an exception to signal the group should be skipped.
        """
        self.step_count += 1

        # Check if browser is corrupted - return terminal result gracefully
        if self._browser_corrupted:
            return self._make_terminal_result(reward=0.0, metrics={"browser_corrupted": 1.0})

        # Decode tokens to text
        response_text, parse_success = self.renderer.parse_response(action)
        response_content = renderers.ensure_text(response_text.get("content", ""))
        self.last_response = response_content

        # Parse to our Action type
        browser_action = parse_action_from_response(response_content, AGENT_AUTH_ACTIONS)

        if browser_action is None:
            # Failed to parse - terminate with zero reward
            logtree.log_text(f"Failed to parse action from response: {response_content[:200]}")
            console.print(f"  {ts()} [red]{self.task.domain}: step={self.step_count} error=parse_failed[/]")
            return self._make_terminal_result(reward=0.0, metrics={"parse_error": 1.0})

        # Record action
        action_desc = browser_action.to_description()
        if browser_action.model_description:
            action_desc = f"{browser_action.model_description} ({action_desc})"
        self.action_history.append(action_desc)

        # Check for terminal action
        if getattr(browser_action, "is_terminal", False):
            self.final_action = browser_action
            # Terminal - reward computed by EnvGroupBuilder via WebJudge
            logtree.log_text(f"Terminal action: {action_desc}")
            console.print(f"  {ts()} [dim]{self.task.domain}: step={self.step_count} action={action_desc} [done][/]")
            return self._make_terminal_result(
                reward=0.0,  # Will be computed by compute_group_rewards
                metrics={"terminal": 1.0},
            )

        # Execute browser action
        if self.adapter is None:
            return self._make_terminal_result(reward=0.0, metrics={"no_adapter": 1.0})

        t_exec_start = time.perf_counter()
        try:
            # Capture baseline for settle detection
            baseline = self.adapter.capture_screenshot()

            should_continue = self.adapter.execute_action(browser_action)

            if not should_continue:
                self.final_action = browser_action
                t_exec = time.perf_counter() - t_exec_start
                console.print(f"  {ts()} [dim]{self.task.domain}: step={self.step_count} exec={t_exec:.1f}s action={action_desc} [done][/]")
                return self._make_terminal_result(reward=0.0, metrics={"terminal": 1.0})

            # Wait for screen to settle
            if not getattr(browser_action, "skip_screen_settle", False):
                self.adapter.wait_for_screen_settle(baseline=baseline)

            # Capture new screenshot
            screenshot = self.adapter.capture_screenshot()
            screenshot_resized = resize_image(screenshot, max_size=self.config.image_max_size)
            self.screenshots.append(screenshot_resized.copy())

        except Exception as e:
            # Browser operation failed - mark as corrupted and terminate gracefully
            self._browser_corrupted = True
            logger.warning(f"Browser operation failed for {self.task.initial_url}: {e}")
            console.print(f"  {ts()} [red]{self.task.domain}: step={self.step_count} error=browser_failed[/]")
            self.cleanup()
            return self._make_terminal_result(reward=0.0, metrics={"browser_error": 1.0})

        t_exec = time.perf_counter() - t_exec_start

        # Check max steps
        if self.step_count >= self.config.max_steps:
            logtree.log_text(f"Max steps ({self.config.max_steps}) reached")
            console.print(f"  {ts()} [yellow]{self.task.domain}: step={self.step_count} exec={t_exec:.1f}s action={action_desc} [max_steps][/]")
            return self._make_terminal_result(reward=0.0, metrics={"max_steps": 1.0})

        # Build next observation
        self.conversation.append({"role": "assistant", "content": response_content})
        self.conversation.append(
            {
                "role": "user",
                "content": [
                    ImagePart(type="image", image=screenshot_resized),
                ],
            }
        )

        # Prune old screenshots to limit context size
        self._prune_old_screenshots()

        logtree.log_text(f"Step {self.step_count}: {action_desc}")
        console.print(f"  {ts()} [dim]{self.task.domain}: step={self.step_count} exec={t_exec:.1f}s action={action_desc}[/]")

        return StepResult(
            reward=0.0,  # Intermediate rewards are 0
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(self.conversation),
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def _prune_old_screenshots(self) -> None:
        """
        Prune old screenshots from conversation to limit context size.

        Keeps:
        - System message (index 0)
        - First user message with task + initial screenshot (index 1)
        - Last N-1 pairs of (assistant response, user screenshot) to reach max_screenshots_in_context total

        This prevents context from growing unboundedly as steps accumulate,
        which dramatically slows down VLM inference.
        """
        max_screenshots = self.config.max_screenshots_in_context

        # Count current screenshots (user messages with images)
        screenshot_indices = []
        for i, msg in enumerate(self.conversation):
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list) and any(
                    isinstance(part, dict) and part.get("type") == "image" for part in content
                ):
                    screenshot_indices.append(i)

        if len(screenshot_indices) <= max_screenshots:
            return  # Nothing to prune

        # Keep first screenshot (index 1 - has task description) and last (max_screenshots-1) screenshots
        indices_to_keep = {screenshot_indices[0]}  # Always keep first (task + initial screenshot)
        indices_to_keep.update(screenshot_indices[-(max_screenshots - 1) :])  # Keep last N-1

        # Build pruned conversation
        pruned = [self.conversation[0]]  # System message

        i = 1
        while i < len(self.conversation):
            msg = self.conversation[i]
            if msg.get("role") == "user" and i in screenshot_indices:
                if i in indices_to_keep:
                    # Keep this user message and its preceding assistant response (if any)
                    if i > 1 and self.conversation[i - 1].get("role") == "assistant":
                        if pruned[-1] != self.conversation[i - 1]:
                            pruned.append(self.conversation[i - 1])
                    pruned.append(msg)
                i += 1
            elif msg.get("role") == "assistant":
                i += 1
            else:
                pruned.append(msg)
                i += 1

        self.conversation = pruned

    def _make_terminal_result(self, reward: float, metrics: Metrics) -> StepResult:
        """Create a terminal StepResult and cleanup."""
        # Release browser
        if self.adapter is not None:
            try:
                self.adapter.stop_heartbeat_sync()
                reuse = not self._browser_corrupted and not self.adapter._should_not_reuse
                self.kernel.browser_pools.release(
                    self.config.pool_name, session_id=self.adapter.session_id, reuse=reuse
                )
            except Exception as e:
                logger.warning(f"Failed to release browser: {e}")
            self.adapter = None

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    def to_webjudge_trajectory(self) -> WebJudgeTrajectory:
        """Convert to WebJudge trajectory for evaluation."""
        return WebJudgeTrajectory(
            task_id=self.task.id,
            task=self.task.task,
            action_history=self.action_history,
            screenshots=self.screenshots,
        )

    def cleanup(self) -> None:
        """Ensure browser is released (sync version)."""
        if self.adapter is not None:
            try:
                self.adapter.stop_heartbeat_sync()
                reuse = not self._browser_corrupted and not self.adapter._should_not_reuse
                self.kernel.browser_pools.release(
                    self.config.pool_name,
                    session_id=self.adapter.session_id,
                    reuse=reuse,
                )
            except Exception:
                pass
            self.adapter = None

    async def cleanup_async(self) -> None:
        """Ensure browser is released (async version - properly stops heartbeat)."""
        if self.adapter is not None:
            try:
                self.adapter.stop_heartbeat_sync()
                reuse = not self._browser_corrupted and not self.adapter._should_not_reuse
                self.kernel.browser_pools.release(
                    self.config.pool_name,
                    session_id=self.adapter.session_id,
                    reuse=reuse,
                )
            except Exception:
                pass
            self.adapter = None


@dataclass
class AgentAuthEnvGroupBuilder(EnvGroupBuilder):
    """
    Builder for a group of AgentAuthEnv instances.

    Creates multiple environments for the same task (for GRPO baseline).
    Computes rewards using WebJudge after trajectories are collected.
    """

    task: AgentAuthTask
    kernel: Kernel
    renderer: renderers.Renderer
    config: AgentAuthEnvConfig
    num_envs: int
    system_prompt: str | None = None

    # WebJudge for reward computation
    webjudge: WebJudge | None = None

    # Raindrop tracking
    raindrop_batch_id: str | None = None
    webjudge_model: str = "openai/gpt-5-mini"

    async def make_envs(self) -> Sequence[Env]:
        """Create num_envs environments for the same task."""
        return [
            AgentAuthEnv(
                task=self.task,
                kernel=self.kernel,
                renderer=self.renderer,
                config=self.config,
                system_prompt=self.system_prompt,
            )
            for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        """
        Compute rewards using WebJudge.

        Runs WebJudge on each trajectory and returns (reward, metrics) tuples.
        Uses finally blocks to ensure cleanup is always called.
        Tracks episodes and WebJudge signals to Raindrop if enabled.
        """
        results: list[tuple[float, Metrics]] = []

        for traj, env in zip(trajectory_group, env_group):
            assert isinstance(env, AgentAuthEnv)

            # Start Raindrop interaction for this episode
            interaction = None
            if self.raindrop_batch_id and is_raindrop_enabled():
                interaction = begin_episode(
                    task=env.task.task,
                    convo_id=self.raindrop_batch_id,
                    properties={
                        "task_id": env.task.id,
                        "domain": env.task.domain,
                        "initial_url": env.task.initial_url,
                        "steps": env.step_count,
                    },
                )

            try:
                # Build WebJudge trajectory
                wj_traj = env.to_webjudge_trajectory()

                # Evaluate with WebJudge
                if self.webjudge is not None and len(wj_traj.screenshots) > 0:
                    try:
                        t_wj_start = time.perf_counter()
                        wj_result = await self.webjudge.evaluate(wj_traj)
                        t_wj = time.perf_counter() - t_wj_start
                        reward = wj_result.score
                        metrics: Metrics = {
                            "webjudge_success": float(wj_result.success),
                            "webjudge_score": wj_result.score,
                        }

                        logtree.log_text(
                            f"WebJudge: {'SUCCESS' if wj_result.success else 'FAILURE'} "
                            f"(score={wj_result.score})"
                        )
                        status = "[green]✓[/]" if wj_result.success else "[red]✗[/]"
                        console.print(
                            f"  {ts()} {status} {env.task.domain}: "
                            f"steps={env.step_count} judge={t_wj:.1f}s reward={wj_result.score:.2f}"
                        )

                        # Track WebJudge signal to Raindrop
                        if interaction is not None:
                            track_webjudge_signal(
                                interaction=interaction,
                                success=wj_result.success,
                                score=wj_result.score,
                                reasoning=wj_result.reasoning,
                                webjudge_model=self.webjudge_model,
                            )

                    except Exception as e:
                        logger.warning(f"WebJudge evaluation failed: {e}")
                        reward = 0.0
                        metrics = {"webjudge_error": 1.0}
                else:
                    # No WebJudge - use heuristic based on terminal action
                    if isinstance(env.final_action, FoundInputsAction):
                        # Found login form - positive reward
                        reward = 1.0
                        metrics = {"found_form": 1.0}
                    elif isinstance(env.final_action, TerminateAction):
                        status = getattr(env.final_action, "status", "unknown")
                        reward = 1.0 if status == "success" else 0.0
                        metrics = {"terminate_status": reward}
                    else:
                        reward = 0.0
                        metrics = {"no_terminal": 1.0}

                # Finish Raindrop interaction
                if interaction is not None:
                    action_summary = " → ".join(env.action_history[-3:]) if env.action_history else "No actions"
                    finish_episode(
                        interaction=interaction,
                        output=action_summary,
                        properties={
                            "reward": reward,
                            "steps": env.step_count,
                            **metrics,
                        },
                    )

                results.append((reward, metrics))

            finally:
                # Always cleanup env, even if an error occurred
                await env.cleanup_async()

        return results

    def logging_tags(self) -> list[str]:
        return ["agent_auth"]


class AgentAuthRLDataset(RLDataset):
    """
    Dataset of agent auth tasks for RL training.

    Loads tasks and creates EnvGroupBuilders for each batch.
    """

    def __init__(
        self,
        tasks: list[AgentAuthTask],
        kernel: Kernel,
        renderer: renderers.Renderer,
        config: AgentAuthEnvConfig,
        batch_size: int,
        group_size: int,
        system_prompt: str | None = None,
        webjudge: WebJudge | None = None,
        webjudge_model: str = "openai/gpt-5-mini",
        seed: int = 42,
        raindrop_batch_id: str | None = None,
    ):
        self.tasks = tasks
        self.kernel = kernel
        self.renderer = renderer
        self.config = config
        self.batch_size = batch_size
        self.group_size = group_size
        self.system_prompt = system_prompt
        self.webjudge = webjudge
        self.webjudge_model = webjudge_model
        self.raindrop_batch_id = raindrop_batch_id

        # Shuffle tasks
        rng = random.Random(seed)
        self.shuffled_tasks = list(tasks)
        rng.shuffle(self.shuffled_tasks)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of EnvGroupBuilders for the given index."""
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.shuffled_tasks))
        batch_tasks = self.shuffled_tasks[start:end]

        return [
            AgentAuthEnvGroupBuilder(
                task=task,
                kernel=self.kernel,
                renderer=self.renderer,
                config=self.config,
                num_envs=self.group_size,
                system_prompt=self.system_prompt,
                webjudge=self.webjudge,
                raindrop_batch_id=self.raindrop_batch_id,
                webjudge_model=self.webjudge_model,
            )
            for task in batch_tasks
        ]

    def __len__(self) -> int:
        return len(self.shuffled_tasks) // self.batch_size


@chz.chz
class AgentAuthRLDatasetBuilder(RLDatasetBuilder):
    """
    Builder for AgentAuthRLDataset.

    Handles initialization of Kernel, renderer, WebJudge, etc.

    This is the main entry point for training - pass an instance of this
    to the train script to start training.
    """

    # Model config
    model_name: str = MODEL_NAME
    renderer_name: str = RENDERER_NAME

    # Dataset config
    batch_size: int = 4
    group_size: int = 2
    max_tasks: int | None = None  # Limit number of tasks (None = all)
    task_data_path: str = "examples/agent_auth/tasks.jsonl"
    seed: int = 42

    # Environment config
    pool_name: str = "rl-browser-pool"
    acquire_timeout_seconds: int = 60
    max_steps: int = 5
    image_max_size: int = 512
    max_screenshots_in_context: int = 3

    # WebJudge config
    webjudge_model: str = "openai/gpt-5-mini"
    webjudge_enabled: bool = True
    webjudge_criteria: str = AGENT_AUTH_EVALUATION_CRITERIA

    # Raindrop config
    raindrop_batch_id: str | None = None

    async def __call__(self) -> tuple[AgentAuthRLDataset, None]:
        """Build and return the dataset."""
        # Load tasks
        tasks = load_tasks(
            jsonl_path=self.task_data_path,
            limit=self.max_tasks,
        )
        logger.info(f"Loaded {len(tasks)} agent auth tasks")

        # Initialize Kernel
        kernel = Kernel()

        # Initialize renderer
        tokenizer = get_tokenizer(self.model_name)
        image_processor = get_image_processor(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer, image_processor)

        # Initialize WebJudge
        webjudge: WebJudge | None = None
        if self.webjudge_enabled:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                webjudge = WebJudge(
                    model=self.webjudge_model,
                    api_key=api_key,
                    evaluation_criteria=self.webjudge_criteria,
                )
                logger.info(f"WebJudge initialized with model: {self.webjudge_model}")
            else:
                logger.warning("OPENROUTER_API_KEY not set, WebJudge disabled")

        # Build config
        config = AgentAuthEnvConfig(
            pool_name=self.pool_name,
            acquire_timeout_seconds=self.acquire_timeout_seconds,
            max_steps=self.max_steps,
            image_max_size=self.image_max_size,
            max_screenshots_in_context=self.max_screenshots_in_context,
        )

        # Create dataset
        dataset = AgentAuthRLDataset(
            tasks=tasks,
            kernel=kernel,
            renderer=renderer,
            config=config,
            batch_size=self.batch_size,
            group_size=self.group_size,
            system_prompt=get_agent_auth_system_prompt(),
            webjudge=webjudge,
            webjudge_model=self.webjudge_model,
            seed=self.seed,
            raindrop_batch_id=self.raindrop_batch_id,
        )

        return dataset, None
