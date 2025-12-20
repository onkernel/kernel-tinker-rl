# Building Custom Environments

This guide walks you through creating a custom RL environment for your own use case.

## Overview

To create a custom environment, you need:

1. **Task dataset** - What tasks should the agent complete?
2. **Custom actions** (optional) - Any domain-specific actions?
3. **System prompt** - How should the agent approach the task?
4. **Environment class** - How does the episode work?
5. **Dataset builder** - How to integrate with Tinker?

Let's build a "form filling" environment as an example.

## Step 1: Create the Directory Structure

```bash
mkdir -p examples/form_filling
touch examples/form_filling/__init__.py
touch examples/form_filling/actions.py
touch examples/form_filling/config.py
touch examples/form_filling/dataset.py
touch examples/form_filling/environment.py
touch examples/form_filling/tasks.jsonl
```

## Step 2: Define Your Task Format

Create `examples/form_filling/dataset.py`:

```python
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class FormFillingTask:
    """A form filling task."""
    id: str
    initial_url: str
    task: str
    form_fields: dict[str, str]  # Field name -> expected value


def load_tasks(jsonl_path: str, limit: int | None = None) -> list[FormFillingTask]:
    """Load tasks from JSONL file."""
    tasks = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            tasks.append(FormFillingTask(
                id=data["id"],
                initial_url=data["initial_url"],
                task=data["task"],
                form_fields=data.get("form_fields", {}),
            ))
            if limit and len(tasks) >= limit:
                break
    return tasks
```

Create some sample tasks in `examples/form_filling/tasks.jsonl`:

```json
{
  "id": "form-001",
  "initial_url": "https://example.com/contact",
  "task": "Fill out the contact form with name 'John Doe' and email 'john@example.com'",
  "form_fields": { "name": "John Doe", "email": "john@example.com" }
}
```

## Step 3: Define Custom Actions (Optional)

If your use case needs custom actions, create `examples/form_filling/actions.py`:

```python
from dataclasses import dataclass
from typing import Any, ClassVar
from core.actions import Action


@dataclass
class SubmitFormAction(Action):
    """Submit the current form."""

    action_type: ClassVar[str] = "submit_form"
    description: ClassVar[str] = "Submit the form. Use when all fields are filled."
    is_terminal: ClassVar[bool] = True
    parameters: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def parse_args(cls, args: dict) -> "SubmitFormAction":
        return cls()

    def to_description(self) -> str:
        return "Submit form"

    def to_tool_args(self) -> dict:
        return {"action": self.action_type}


# List of custom actions for this environment
FORM_FILLING_ACTIONS: list[type[Action]] = [SubmitFormAction]
```

## Step 4: Create the System Prompt

Create `examples/form_filling/config.py`:

```python
from core.prompts import build_system_prompt
from .actions import FORM_FILLING_ACTIONS


FORM_FILLING_PROMPT = """You are a form-filling assistant. Your goal is to:

1. Navigate to forms on web pages
2. Fill in the required fields with the provided information
3. Submit the form when complete

Guidelines:
- Read field labels carefully before typing
- Use Tab or click to move between fields
- Verify your entries before submitting
- Use the submit_form action when all fields are filled
"""


def get_form_filling_system_prompt() -> str:
    """Build the system prompt for form filling agent."""
    from core.actions import STANDARD_ACTIONS

    all_actions = STANDARD_ACTIONS + FORM_FILLING_ACTIONS

    return build_system_prompt(
        task_description=FORM_FILLING_PROMPT,
        actions=all_actions,
    )
```

## Step 5: Implement the Environment

Create `examples/form_filling/environment.py`:

```python
from dataclasses import dataclass
import logging
from typing import Sequence

import chz
from kernel import Kernel
from PIL import Image
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import (
    Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder,
    Observation, StepResult, Trajectory, Metrics,
    Action as TinkerAction,
)

from core.actions import parse_action_from_response, TerminateAction
from core.browser import KernelBrowserAdapter
from core.reward_models import WebJudge, Trajectory as WebJudgeTrajectory
from core.utils import resize_image

from .actions import FORM_FILLING_ACTIONS, SubmitFormAction
from .config import get_form_filling_system_prompt
from .dataset import FormFillingTask, load_tasks


logger = logging.getLogger(__name__)


@dataclass
class FormFillingEnvConfig:
    pool_name: str = "rl-browser-pool"
    acquire_timeout_seconds: int = 60
    max_steps: int = 10
    image_max_size: int = 512


class FormFillingEnv(Env):
    """Environment for form filling tasks."""

    def __init__(
        self,
        task: FormFillingTask,
        kernel: Kernel,
        renderer: renderers.Renderer,
        config: FormFillingEnvConfig,
        system_prompt: str,
    ):
        self.task = task
        self.kernel = kernel
        self.renderer = renderer
        self.config = config
        self.system_prompt = system_prompt

        # State tracking
        self.adapter: KernelBrowserAdapter | None = None
        self.step_count = 0
        self.screenshots: list[Image.Image] = []
        self.action_history: list[str] = []
        self.conversation: list[dict] = []
        self.final_action = None

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    async def initial_observation(self):
        """Set up the environment and return first observation."""
        # Acquire browser from pool
        browser = self.kernel.browser_pools.acquire(
            self.config.pool_name,
            acquire_timeout_seconds=self.config.acquire_timeout_seconds,
        )
        self.adapter = KernelBrowserAdapter(self.kernel, browser)
        self.adapter.start_heartbeat_sync()

        # Navigate
        self.adapter.navigate(self.task.initial_url)

        # Capture screenshot
        screenshot = self.adapter.capture_screenshot()
        screenshot = resize_image(screenshot, max_size=self.config.image_max_size)
        self.screenshots.append(screenshot.copy())
        self.action_history.append(f"Navigate to {self.task.initial_url}")

        # Build initial conversation
        self.conversation = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": f"Task: {self.task.task}"},
                ],
            },
        ]

        return self.renderer.build_generation_prompt(self.conversation), self.stop_condition

    async def step(self, action: TinkerAction) -> StepResult:
        """Execute one step in the environment."""
        self.step_count += 1

        # Parse the action from tokens
        response_text, _ = self.renderer.parse_response(action)
        response_content = renderers.ensure_text(response_text.get("content", ""))

        browser_action = parse_action_from_response(
            response_content,
            extra_actions=FORM_FILLING_ACTIONS,
        )

        if browser_action is None:
            return self._terminate(0.0, {"parse_error": 1.0})

        # Record action
        action_desc = browser_action.to_description()
        self.action_history.append(action_desc)

        # Check for terminal
        if getattr(browser_action, "is_terminal", False):
            self.final_action = browser_action
            return self._terminate(0.0, {"terminal": 1.0})

        # Execute action
        baseline = self.adapter.capture_screenshot()
        should_continue = self.adapter.execute_action(browser_action)

        if not should_continue:
            return self._terminate(0.0, {"terminal": 1.0})

        # Wait for settle and capture new screenshot
        self.adapter.wait_for_screen_settle(baseline=baseline)
        screenshot = self.adapter.capture_screenshot()
        screenshot = resize_image(screenshot, max_size=self.config.image_max_size)
        self.screenshots.append(screenshot.copy())

        # Check max steps
        if self.step_count >= self.config.max_steps:
            return self._terminate(0.0, {"max_steps": 1.0})

        # Continue with next observation
        self.conversation.append({"role": "assistant", "content": response_content})
        self.conversation.append({
            "role": "user",
            "content": [{"type": "image", "image": screenshot}],
        })

        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(self.conversation),
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def _terminate(self, reward: float, metrics: Metrics) -> StepResult:
        """End the episode."""
        if self.adapter:
            try:
                self.adapter.stop_heartbeat_sync()
                self.kernel.browser_pools.release(
                    self.config.pool_name, session_id=self.adapter.session_id, reuse=True
                )
            except:
                pass
            self.adapter = None

        import tinker
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    def to_trajectory(self) -> WebJudgeTrajectory:
        return WebJudgeTrajectory(
            task_id=self.task.id,
            task=self.task.task,
            action_history=self.action_history,
            screenshots=self.screenshots,
        )

    async def cleanup_async(self):
        if self.adapter:
            self.adapter.stop_heartbeat_sync()
            self.kernel.browser_pools.release(
                self.config.pool_name, session_id=self.adapter.session_id, reuse=True
            )
            self.adapter = None


# ... EnvGroupBuilder, RLDataset, RLDatasetBuilder classes follow the same pattern
# as examples/agent_auth/environment.py
```

## Step 6: Register Your Environment

Update `scripts/train.py` to recognize your environment:

```python
AVAILABLE_ENVS = ["agent_auth", "form_filling"]

def get_dataset_builder(cfg: TrainConfig):
    if cfg.env == "form_filling":
        from examples.form_filling.environment import FormFillingRLDatasetBuilder
        return FormFillingRLDatasetBuilder(...)
    # ... existing environments
```

## Step 7: Train!

```bash
uv run python -m scripts.train \
  --env form_filling \
  --pool-name rl-browser-pool \
  --batch-size 4
```

## Tips for Custom Environments

### Task Design

- Start simple - can you solve it manually in 5-10 clicks?
- Include clear success criteria in the task description
- Provide variety - different websites, different form types

### Reward Shaping

WebJudge works well for general tasks. For specialized domains, consider:

```python
class CustomRewardModel(RewardModel):
    async def evaluate(self, trajectory: Trajectory) -> EvaluationResult:
        # Your custom logic
        if self._check_form_submitted(trajectory):
            return EvaluationResult(success=True, score=1.0)
        return EvaluationResult(success=False, score=0.0)
```

### Debugging

1. Use `scripts/run_agent.py` to test single episodes
2. Check the live view URL to watch the agent
3. Add logging to your environment:

```python
import logging
logger = logging.getLogger(__name__)

class MyEnv(Env):
    async def step(self, action):
        logger.info(f"Step {self.step_count}: {action}")
        # ...
```

## Reference Implementation

See `examples/agent_auth/` for a complete, production-ready example of:

- Custom actions (`FoundInputsAction`)
- Domain-specific prompts
- WebJudge integration
- Full Tinker RL dataset implementation
