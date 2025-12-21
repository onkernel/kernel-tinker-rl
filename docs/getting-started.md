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

# Required for training and checkpoint evaluation
TINKER_API_KEY=your_tinker_api_key

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
  --batch-size 12 \
  --group-size 4 \
  --wandb-project my-rl-experiment
```

Training parameters:

| Parameter         | Description                  | Default |
| ----------------- | ---------------------------- | ------- |
| `--batch-size`    | Tasks per training batch     | 12      |
| `--group-size`    | Rollouts per task (for GRPO) | 4       |
| `--max-steps`     | Max actions per episode      | 5       |
| `--max-tasks`     | Limit total tasks            | all     |
| `--lora-rank`     | LoRA adapter rank            | 32      |
| `--learning-rate` | Learning rate                | 4e-5    |

## Step 5: Evaluate Your Model

After training, evaluate your checkpoints against baseline models.

### Basic Evaluation

Evaluate the baseline model on all tasks:

```bash
uv run python -m scripts.evaluate \
  --env agent_auth \
  --pool-name rl-browser-pool \
  --output results.json
```

### Evaluating on Held-Out Data

Use `--start-index` and `--end-index` to evaluate on a subset of tasks (0-based, inclusive).
This enables train/test splits for comparing checkpoints against baselines on unseen data:

```bash
# Evaluate checkpoint on held-out test set (last 20% of tasks, e.g., indices 80+)
uv run python -m scripts.evaluate \
  --env agent_auth \
  --pool-name rl-browser-pool \
  --model tinker://YOUR_RUN_ID:train:0/sampler_weights/000030 \
  --start-index 80

# Evaluate baseline on training set (first 80 tasks, indices 0-79)
uv run python -m scripts.evaluate \
  --env agent_auth \
  --pool-name rl-browser-pool \
  --end-index 79
```

### Evaluating Tinker Checkpoints

Point to a Tinker checkpoint's `sampler_path` from your `checkpoints.jsonl`:

```bash
# Find your checkpoint paths
cat /path/to/your/training/run/checkpoints.jsonl

# Example output:
# {"name": "000030", "batch": 30, "sampler_path": "tinker://488643ee-.../sampler_weights/000030", ...}

# Evaluate checkpoint 30 on held-out set
uv run python -m scripts.evaluate \
  --env agent_auth \
  --pool-name rl-browser-pool \
  --model tinker://488643ee-3be8-523e-9297-aecf5f8bb48f:train:0/sampler_weights/000030 \
  --start-index 80 \
  --output checkpoint_30_heldout.json
```

### Quick Checkpoint Testing

Test a single task with a checkpoint using `run_agent.py`:

```bash
uv run python -m scripts.run_agent \
  --env agent_auth \
  --checkpoint tinker://488643ee-.../sampler_weights/000030 \
  --random \
  --webjudge
```

## Next Steps

- [Architecture Overview](./architecture.md) - Understand how the system works
- [Custom Environments](./custom-environments.md) - Build your own use cases
- [WebJudge Guide](./webjudge.md) - Configure and tune the reward model
