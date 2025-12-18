"""
OSWorld - Generic web navigation environment.

This environment uses standard OSWorld actions for web navigation tasks.
It serves as the default environment for running agents on arbitrary web tasks.
"""

from .actions import OSWORLD_ACTIONS
from .config import get_osworld_system_prompt

__all__ = [
    "OSWORLD_ACTIONS",
    "get_osworld_system_prompt",
]


