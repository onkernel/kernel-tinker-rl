"""
Configuration for OSWorld environment.

Provides the default system prompt for web navigation agents.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from core.prompts import CUA_BASE_PROMPT, build_system_prompt

from .actions import OSWORLD_ACTIONS


def get_osworld_system_prompt() -> str:
    """Get the default OSWorld system prompt."""
    return build_system_prompt(
        base_prompt=CUA_BASE_PROMPT,
        extra_actions=OSWORLD_ACTIONS,
        exclude_actions=None,  # Use all standard actions including terminate
    )


