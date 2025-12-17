"""
Custom actions for Agent Auth (login discovery).

Provides the RequestInputsAction for reporting discovered login form fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from core.actions import Action


@dataclass
class InputField:
    """A requested input field for login forms."""

    name: str
    description: str = ""


@dataclass
class RequestInputsAction(Action):
    """
    Request input fields from the user.

    Used by login discovery agents to indicate they found a login form
    and are reporting what inputs are needed.
    """

    fields: list[InputField] = field(default_factory=list)

    action_type: ClassVar[str] = "request_inputs"
    description: ClassVar[str] = (
        "Request input fields from the user (for login discovery). "
        "Use when you've found a form and want to report the required fields. "
        "This action completes the task."
    )
    is_terminal: ClassVar[bool] = True
    parameters: ClassVar[dict[str, dict[str, Any]]] = {
        "fields": {
            "type": "array",
            "description": "List of input fields to request.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the input field (e.g., 'Username', 'Password').",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of what this field is for.",
                    },
                },
                "required": ["name"],
            },
            "required": True,
        }
    }

    @classmethod
    def parse_args(cls, args: dict) -> "RequestInputsAction | None":
        fields_data = args.get("fields", [])
        fields = [
            InputField(
                name=f.get("name", ""),
                description=f.get("description", ""),
            )
            for f in fields_data
        ]
        return cls(fields=fields)

    def to_description(self) -> str:
        field_names = [f.name for f in self.fields]
        return f"Request inputs: {', '.join(field_names)}"

    def to_tool_args(self) -> dict:
        return {
            "action": self.action_type,
            "fields": [{"name": f.name, "description": f.description} for f in self.fields],
        }


# List of custom actions for Agent Auth
AGENT_AUTH_ACTIONS: list[type[Action]] = [RequestInputsAction]
