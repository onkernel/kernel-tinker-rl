"""
Actions for OSWorld environment.

Uses all standard actions including TerminateAction.
No custom actions are added - this is the default OSWorld action set.
"""

from __future__ import annotations

from core.actions import Action

# OSWorld uses all standard actions, no custom additions
OSWORLD_ACTIONS: list[type[Action]] = []


