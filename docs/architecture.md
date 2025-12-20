# Architecture Overview

This document explains how the Kernel + Tinker RL system works and how the components fit together.

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       Tinker Cloud                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   RL Training Loop                       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │  │
│  │  │   Policy    │───▶│  Rollout    │───▶│   GRPO      │   │  │
│  │  │   (VLM)     │    │  Collector  │    │   Update    │   │  │
│  │  └─────────────┘    └──────┬──────┘    └─────────────┘   │  │
│  └────────────────────────────┼─────────────────────────────┘  │
│                               │                                │
│                               ▼                                │
│                    ┌──────────────────┐                        │
│                    │   Environment    │                        │
│                    │   (Your Code)    │                        │
│                    └────────┬─────────┘                        │
└─────────────────────────────┼──────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                      Kernel Platform                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Browser Pool                           │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │  │
│  │  │Browser 1│ │Browser 2│ │Browser 3│ │Browser 4│  ...    │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘         │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent (`core/agent.py`)

The `QwenAgent` class handles VLM inference and action generation:

```python
from core import QwenAgent, AgentConfig

config = AgentConfig(
    model="qwen/qwen3-vl-8b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
agent = QwenAgent(config=config)

# Get next action from screenshot
action = agent.predict(task="Find the login button", screenshot=screenshot)
```

Key responsibilities:

- Building multi-turn conversations with images
- Calling the VLM API (via OpenRouter)
- Parsing structured action responses

### 2. Actions (`core/actions.py`)

OSWorld-compatible action types:

| Action            | Description                 |
| ----------------- | --------------------------- |
| `LeftClickAction` | Click at (x, y) coordinates |
| `TypeTextAction`  | Type text at current cursor |
| `KeyAction`       | Press keyboard keys         |
| `ScrollAction`    | Scroll in a direction       |
| `TerminateAction` | End the episode             |

Actions are parsed from VLM JSON responses:

```python
from core import parse_action_from_response

action = parse_action_from_response(
    response='{"action": "left_click", "x": 0.5, "y": 0.3}',
    extra_actions=[FoundInputsAction],  # Custom actions
)
```

### 3. Browser Adapters (`core/browser.py`)

Two adapter types for Kernel browsers:

**Direct Adapter** (for testing):

```python
from core import KernelBrowserAdapter

browser = kernel.browsers.create(stealth=True)
adapter = KernelBrowserAdapter(kernel, browser)
screenshot = adapter.capture_screenshot()
adapter.execute_action(action)
```

**Pool-based usage** (for training):

```python
from core import acquired_browser

# Using context manager (recommended)
with acquired_browser(kernel, "my-pool") as adapter:
    adapter.navigate("https://example.com")
    screenshot = adapter.capture_screenshot()
    # ... use browser ...
# Browser automatically released back to pool

# Or manual acquire/release
browser = kernel.browser_pools.acquire("my-pool")
adapter = KernelBrowserAdapter(kernel, browser)
# ... use browser ...
kernel.browser_pools.release("my-pool", session_id=adapter.session_id, reuse=True)
```

### 4. Reward Models (`core/reward_models/`)

WebJudge is an LLM-as-judge that evaluates trajectories:

```python
from core import WebJudge, Trajectory

webjudge = WebJudge(model="openai/gpt-5-mini")

trajectory = Trajectory(
    task_id="task-001",
    task="Find the login page",
    action_history=["Navigate to https://example.com", "Click on 'Sign In'"],
    screenshots=[screenshot1, screenshot2],
)

result = await webjudge.evaluate(trajectory)
print(f"Success: {result.success}, Score: {result.score}")
```

The 3-phase evaluation:

1. **Key Point Identification** - Extract requirements from task
2. **Key Screenshot Selection** - Score screenshots 1-5 for relevance
3. **Outcome Judgment** - Final success/failure determination

## Training Flow

### 1. Dataset Builder

Implements Tinker's `RLDatasetBuilder` interface:

```python
@chz.chz
class MyDatasetBuilder(RLDatasetBuilder):
    async def __call__(self) -> tuple[RLDataset, None]:
        tasks = load_tasks(...)
        return MyRLDataset(tasks=tasks, ...), None
```

### 2. Environment

Each task runs in an `Env` that:

1. Acquires a browser from the pool
2. Navigates to the initial URL
3. Runs the observation-action loop
4. Releases the browser back to the pool

```python
class MyEnv(Env):
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Setup browser, navigate, capture first screenshot
        ...

    async def step(self, action: TinkerAction) -> StepResult:
        # Parse action, execute in browser, capture screenshot
        ...
```

### 3. EnvGroupBuilder

Creates multiple environments for the same task (GRPO needs multiple rollouts):

```python
class MyEnvGroupBuilder(EnvGroupBuilder):
    async def make_envs(self) -> Sequence[Env]:
        return [MyEnv(task=self.task, ...) for _ in range(self.num_envs)]

    async def compute_group_rewards(self, trajectories, envs):
        # Run WebJudge on each trajectory
        return [(reward, metrics) for ...]
```

## Browser Pool Architecture

Browser pools are critical for RL training efficiency:

```
┌────────────────────────────────────────────────┐
│              Browser Pool "my-pool"            │
│                                                │
│  Ready Queue:    [B1] [B2] [B3] [B4]           │
│                                                │
│  acquire() ─────▶ Pop B1                       │
│  release(B1) ───▶ Push B1 back                 │
│                                                │
│  Pool Manager:                                 │
│  - Maintains pool size                         │
│  - Recycles browsers                           │
│  - Handles timeouts                            │
└────────────────────────────────────────────────┘
```

Benefits for RL:

- **Zero startup latency**: Browsers are pre-warmed
- **Efficient resource usage**: Browsers are reused
- **Automatic cleanup**: Pool handles crashes gracefully
- **Parallel rollouts**: Each env gets its own browser

## Customization Points

| Component     | Location                    | Customization              |
| ------------- | --------------------------- | -------------------------- |
| Actions       | `examples/*/actions.py`     | Add custom action types    |
| System Prompt | `examples/*/config.py`      | Modify agent instructions  |
| Environment   | `examples/*/environment.py` | Custom episode logic       |
| Dataset       | `examples/*/dataset.py`     | Task loading/preprocessing |
| Reward Model  | `core/reward_models/`       | Alternative evaluators     |

See [Custom Environments](./custom-environments.md) for a full walkthrough.
