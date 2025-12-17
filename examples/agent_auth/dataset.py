"""
Dataset utilities for Agent Auth tasks.

Loads tasks from the preprocessed JSONL file in the standardized format:
    - id: Unique task identifier
    - initial_url: Starting URL for the browser
    - task: Task description for the agent
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentAuthTask:
    """A single agent auth task."""

    id: str
    initial_url: str
    task: str
    original_task: str | None = None

    @property
    def domain(self) -> str:
        """Extract domain from initial_url."""
        url = self.initial_url
        if url.startswith("https://"):
            url = url[8:]
        elif url.startswith("http://"):
            url = url[7:]
        return url.split("/")[0]


def load_tasks(
    jsonl_path: str | Path,
    limit: int | None = None,
) -> list[AgentAuthTask]:
    """
    Load agent auth tasks from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file with tasks
        limit: Optional limit on number of tasks to load

    Returns:
        List of AgentAuthTask objects
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    tasks: list[AgentAuthTask] = []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            task = AgentAuthTask(
                id=data["id"],
                initial_url=data["initial_url"],
                task=data["task"],
                original_task=data.get("original_task"),
            )
            tasks.append(task)

            if limit is not None and len(tasks) >= limit:
                break

    return tasks
