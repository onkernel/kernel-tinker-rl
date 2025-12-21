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

### Creating Train/Test Splits

To enable proper evaluation against held-out data, split your dataset into train and eval sets:

```bash
# Split into 80% train / 20% eval with shuffling (seed=42 for reproducibility)
uv run python scripts/split_dataset.py examples/agent_auth/tasks.jsonl

# Creates:
#   examples/agent_auth/tasks_train.jsonl (375 tasks)
#   examples/agent_auth/tasks_eval.jsonl (94 tasks)
```

You can customize the split ratio and seed:

```bash
uv run python scripts/split_dataset.py examples/agent_auth/tasks.jsonl --train-ratio 0.9 --seed 123
```

## Step 3: Run a Quick Test

Before training, test that everything works with a single agent run:

```bash
uv run python -m scripts.run_agent \
  --url https://github.com \
  --task "Navigate to the sign in page" \
  --max-steps 5
```

## Step 4: Start Training

Run the RL training loop. Results are saved to `./results/<run_name>/` by default.

### Training on the Full Dataset

```bash
uv run python -m scripts.train \
  --env agent_auth \
  --pool-name rl-browser-pool \
  --batch-size 12 \
  --group-size 4 \
  --wandb-project my-rl-experiment
```

### Training on the Train Split Only

For proper train/eval separation, train only on the training split:

```bash
uv run python -m scripts.train \
  --env agent_auth \
  --pool-name rl-browser-pool \
  --task-file examples/agent_auth/tasks_train.jsonl \
  --wandb-project my-rl-experiment
```

### Continuing from a Checkpoint

Resume training from a previous checkpoint using the `state_path` from your `checkpoints.jsonl`:

```bash
uv run python -m scripts.train \
  --env agent_auth \
  --pool-name rl-browser-pool \
  --task-file examples/agent_auth/tasks_train.jsonl \
  --load-checkpoint "tinker://YOUR_RUN_ID:train:0/weights/final" \
  --wandb-project my-rl-experiment
```

Training parameters:

| Parameter          | Description                          | Default    |
| ------------------ | ------------------------------------ | ---------- |
| `--batch-size`     | Tasks per training batch             | 12         |
| `--group-size`     | Rollouts per task (for GRPO)         | 4          |
| `--max-steps`      | Max actions per episode              | 5          |
| `--max-tasks`      | Limit total tasks                    | all        |
| `--task-file`      | Path to task JSONL file              | (env default) |
| `--lora-rank`      | LoRA adapter rank                    | 32         |
| `--learning-rate`  | Learning rate                        | 4e-5       |
| `--load-checkpoint`| Tinker checkpoint path to resume from| None       |

Training outputs are saved to `./results/<run_name>/`:
- `config.json`: Training configuration
- `checkpoints.jsonl`: Checkpoint paths for each save
- `*.html`: Per-task rollout visualizations

## Step 5: Evaluate Your Model

After training, evaluate your checkpoints against baseline models on the held-out eval set.

### Evaluate Baseline Model

First, establish a baseline by evaluating the pre-trained model on your eval set:

```bash
uv run python -m scripts.evaluate \
  --env agent_auth \
  --pool-name eval-browser-pool \
  --pool-size 25 \
  --task-file examples/agent_auth/tasks_eval.jsonl \
  --model qwen/qwen3-vl-30b-a3b-instruct \
  --output results/baseline_eval.json
```

### Evaluate Fine-tuned Checkpoint

> **Note:** Tinker's OpenAI-compatible API currently only supports text-based inference and does not support VLM (vision) messages with images. To evaluate fine-tuned VLM checkpoints, you will need to use an alternative inference provider that can load LoRA weights and serve vision models (e.g., vLLM, TGI, or a custom deployment).

Once you have an inference endpoint for your fine-tuned model, evaluate it on the same eval set:

```bash
# Example with a custom inference endpoint
uv run python -m scripts.evaluate \
  --env agent_auth \
  --pool-name eval-browser-pool \
  --pool-size 25 \
  --task-file examples/agent_auth/tasks_eval.jsonl \
  --model your-finetuned-model-endpoint \
  --output results/checkpoint_eval.json
```

You can find your checkpoint paths in `results/YOUR_RUN_NAME/checkpoints.jsonl` to download the LoRA weights for deployment.

### Evaluation Parameters

| Parameter       | Description                           | Default       |
| --------------- | ------------------------------------- | ------------- |
| `--task-file`   | Path to eval task JSONL file          | (env default) |
| `--start-index` | Start index for task subset (0-based) | None (start)  |
| `--end-index`   | End index for task subset (inclusive) | None (end)    |
| `--pool-size`   | Number of concurrent evaluations      | (pool config) |
| `--model`       | Model to evaluate (OpenRouter or tinker://) | (default) |

### Using Index Ranges (Alternative)

If you prefer not to create separate task files, use index ranges:

```bash
# Evaluate on last 20% (indices 376-469 of 470 tasks)
uv run python -m scripts.evaluate \
  --env agent_auth \
  --pool-name eval-browser-pool \
  --start-index 376 \
  --end-index 469 \
  --output results/heldout_eval.json
```

### Quick Checkpoint Testing

Test a single task using `run_agent.py`:

```bash
# Test with the baseline model
uv run python -m scripts.run_agent \
  --env agent_auth \
  --random \
  --webjudge

# Test with a custom fine-tuned model endpoint (once deployed)
uv run python -m scripts.run_agent \
  --env agent_auth \
  --model your-finetuned-model-endpoint \
  --random \
  --webjudge
```

## Next Steps

- [Architecture Overview](./architecture.md) - Understand how the system works
- [Custom Environments](./custom-environments.md) - Build your own use cases
- [WebJudge Guide](./webjudge.md) - Configure and tune the reward model
