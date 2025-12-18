# Getting Started

This guide walks you through setting up and running your first RL training job with Kernel + Tinker.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Kernel CLI](https://docs.onkernel.com/getting-started/cli) installed
- API keys for Kernel, OpenRouter, and optionally Weights & Biases

## Installation

```bash
# Clone the repository
git clone https://github.com/onkernel/kernel-tinker-rl.git
cd kernel-tinker-rl

# Install dependencies
uv sync
```

## Environment Setup

Create a `.env` file in the project root:

```bash
# Required
KERNEL_API_KEY=your_kernel_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional (for experiment tracking)
WANDB_API_KEY=your_wandb_api_key
```

## Step 1: Create a Browser Pool

Browser pools are essential for efficient RL training. They pre-warm browsers so your agent can acquire them instantly.

```bash
# Create a pool with 4 browsers
kernel browser-pool create \
  --name rl-browser-pool \
  --size 4 \
  --stealth \
  --timeout 300
```

See the [Kernel Browser Pools documentation](https://docs.onkernel.com/features/browser-pools) for more options.

## Step 2: Prepare Your Task Data

Tasks are stored in JSONL format with three required fields:

```json
{
  "id": "task-001",
  "initial_url": "https://example.com",
  "task": "Find the login page and identify the input fields"
}
```

For the Agent Auth example, a preprocessed task file is included at `examples/agent_auth/tasks.jsonl`.

## Step 3: Run a Quick Test

Before training, test that everything works with a single agent run:

```bash
uv run python -m scripts.run_agent \
  --url https://github.com \
  --task "Navigate to the sign in page" \
  --max-steps 5
```

## Step 4: Start Training

Run the RL training loop:

```bash
uv run python -m scripts.train \
  --env agent_auth \
  --pool-name rl-browser-pool \
  --batch-size 4 \
  --group-size 2 \
  --wandb-project my-rl-experiment
```

Training parameters:

| Parameter         | Description                  | Default |
| ----------------- | ---------------------------- | ------- |
| `--batch-size`    | Tasks per training batch     | 4       |
| `--group-size`    | Rollouts per task (for GRPO) | 2       |
| `--max-steps`     | Max actions per episode      | 5       |
| `--max-tasks`     | Limit total tasks            | all     |
| `--lora-rank`     | LoRA adapter rank            | 32      |
| `--learning-rate` | Learning rate                | 4e-5    |

## Step 5: Evaluate Your Model

After training, evaluate on a held-out test set:

```bash
uv run python -m scripts.evaluate \
  --env agent_auth \
  --pool-name rl-browser-pool \
  --max-tasks 50 \
  --output results.json
```

## Next Steps

- [Architecture Overview](./architecture.md) - Understand how the system works
- [Custom Environments](./custom-environments.md) - Build your own use cases
- [WebJudge Guide](./webjudge.md) - Configure and tune the reward model
