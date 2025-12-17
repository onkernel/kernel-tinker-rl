#!/usr/bin/env python3
"""
Preprocess task data into the standardized format.

Converts task-data.jsonl (from the data generation pipeline) into the
standardized format used by this repository.

Input format:
    {"id": "...", "screenshot": "screenshots/example.com.png", "task": "..."}

Output format:
    {"id": "...", "initial_url": "https://example.com", "task": "..."}

Usage:
    # Convert all tasks
    uv run python -m scripts.preprocess_tasks \
        --input ../data/task-data.jsonl \
        --output data/tasks.jsonl

    # Filter to login-related tasks only (for agent auth)
    uv run python -m scripts.preprocess_tasks \
        --input ../data/task-data.jsonl \
        --output examples/agent_auth/tasks.jsonl \
        --filter-login \
        --dedupe-domains
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Patterns to match login-related tasks (with named groups for task generation)
LOGIN_PATTERN_GROUPS = [
    (r"sign\s*in", "sign in"),
    (r"log\s*in", "log in"),
    (r"login", "log in"),
    (r"register", "register"),
    (r"create.*account", "create an account"),
    (r"new\s*account", "create an account"),
    (r"sign\s*up", "sign up"),
]

# Combined regex for filtering
LOGIN_REGEX = re.compile(
    "|".join(f"({pattern})" for pattern, _ in LOGIN_PATTERN_GROUPS),
    re.IGNORECASE
)


def extract_domain_from_screenshot(screenshot_path: str) -> str:
    """Extract domain from screenshot filename like 'screenshots/github.com.png'."""
    filename = Path(screenshot_path).stem
    return filename


def detect_login_type(task: str) -> str | None:
    """
    Detect the type of login action from the task text.

    Returns a normalized action phrase like "sign in", "log in", "register", etc.
    Returns None if no login-related pattern is found.
    """
    task_lower = task.lower()
    for pattern, action_phrase in LOGIN_PATTERN_GROUPS:
        if re.search(pattern, task_lower, re.IGNORECASE):
            return action_phrase
    return None


def is_login_task(task: str) -> bool:
    """Check if a task is login-related."""
    return detect_login_type(task) is not None


def convert_task(
    data: dict,
    filter_login: bool = False,
) -> dict | None:
    """
    Convert a task to the standardized format.

    Args:
        data: Original task data
        filter_login: If True, skip non-login tasks and generate agent-auth format

    Returns:
        Converted task dict, or None if filtered out
    """
    task_text = data.get("task", "")

    # Extract domain from screenshot path
    screenshot_path = data.get("screenshot", "")
    domain = extract_domain_from_screenshot(screenshot_path)

    if not domain:
        return None

    # Build output
    output = {
        "id": data.get("id", ""),
        "initial_url": f"https://{domain}",
    }

    if filter_login:
        # Detect login type and generate appropriate task
        login_type = detect_login_type(task_text)
        if login_type is None:
            return None

        # Generate task with the detected action type
        output["task"] = (
            f"Navigate to {domain} and find the {login_type} page. "
            f"Identify the first input field(s) required to begin the '{login_type}' process."
        )
    else:
        output["task"] = task_text

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess task data into standardized format"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument(
        "--filter-login",
        action="store_true",
        help="Only include login-related tasks (generates agent-auth format)",
    )
    parser.add_argument(
        "--dedupe-domains", action="store_true", help="Only keep one task per domain"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_domains: set[str] = set()
    converted_count = 0
    skipped_count = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            data = json.loads(line)
            converted = convert_task(
                data,
                filter_login=args.filter_login,
            )

            if converted is None:
                skipped_count += 1
                continue

            # Dedupe by domain if requested
            if args.dedupe_domains:
                domain = extract_domain_from_screenshot(data.get("screenshot", ""))
                if domain in seen_domains:
                    skipped_count += 1
                    continue
                seen_domains.add(domain)

            f_out.write(json.dumps(converted) + "\n")
            converted_count += 1

    print(f"Converted {converted_count} tasks")
    print(f"Skipped {skipped_count} tasks")
    print(f"Output written to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

