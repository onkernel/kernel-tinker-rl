# OSWorld Environment

Generic web navigation environment using standard OSWorld-style actions.

## Overview

This environment provides a baseline for web navigation agents using the standard
computer use action set (click, type, scroll, etc.) including the `terminate` action.

## Usage

```bash
# Run on a random task from tasks.jsonl
uv run python -m scripts.run_agent --env osworld --random

# Run on a specific task by ID
uv run python -m scripts.run_agent --env osworld --id github-signin

# Run with ad-hoc URL and task
uv run python -m scripts.run_agent --env osworld \
    --url https://example.com \
    --task "Click on the first link"
```

## Files

- `actions.py` - Empty action list (uses all standard OSWorld actions)
- `config.py` - System prompt configuration
- `dataset.py` - Task loading utilities
- `tasks.jsonl` - Example web navigation tasks

## Adding Tasks

Add tasks to `tasks.jsonl` in the format:

```jsonl
{"id": "unique-id", "initial_url": "https://example.com", "task": "Description of what to do"}
```

## Actions

This environment uses all standard OSWorld actions:
- `left_click`, `right_click`, `double_click`, `triple_click`, `middle_click`
- `mouse_move`, `left_click_drag`
- `type`, `key`
- `scroll`
- `wait`
- `terminate` (terminal action)


