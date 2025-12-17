"""
Agent Auth - Login discovery example.

This example trains agents to find login pages and identify required input fields.
"""

from .actions import AGENT_AUTH_ACTIONS, InputField, RequestInputsAction
from .config import get_agent_auth_system_prompt, make_agent_auth_task

__all__ = [
    "RequestInputsAction",
    "InputField",
    "AGENT_AUTH_ACTIONS",
    "get_agent_auth_system_prompt",
    "make_agent_auth_task",
]


