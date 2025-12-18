"""
Dataset utilities for OSWorld tasks.

Loads tasks from JSONL files in the standardized format:
    - id: Unique task identifier
    - initial_url: Starting URL for the browser
    - task: Task description for the agent
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OsworldTask:
    """A single OSWorld task."""

    id: str
    initial_url: str
    task: str

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
) -> list[OsworldTask]:
    """
    Load OSWorld tasks from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file with tasks
        limit: Optional limit on number of tasks to load

    Returns:
        List of OsworldTask objects
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    tasks: list[OsworldTask] = []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            task = OsworldTask(
                id=data["id"],
                initial_url=data["initial_url"],
                task=data["task"],
            )
            tasks.append(task)

            if limit is not None and len(tasks) >= limit:
                break

    return tasks


