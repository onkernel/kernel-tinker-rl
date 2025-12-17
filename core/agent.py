"""
Qwen3-VL Computer Use Agent.

A VLM-based agent for computer use tasks, inspired by OSWorld's agent architecture.
Uses OpenRouter for model access and Qwen3-VL's native tool call format.

References:
    https://github.com/xlang-ai/OSWorld
    https://arxiv.org/abs/2404.07972 (OSWorld paper)
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

from openai import OpenAI
from PIL import Image

from .actions import Action, parse_action_from_response
from .prompts import get_system_prompt
from .utils import resize_image

# Available Qwen VLM models on OpenRouter
AVAILABLE_MODELS: list[str] = [
    "qwen/qwen3-vl-8b-instruct",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen/qwen3-vl-235b-a22b-instruct",
]

DEFAULT_MODEL: str = AVAILABLE_MODELS[0]


def encode_image(image: Image.Image) -> str:
    """Convert a PIL image to base64 JPEG string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@dataclass
class AgentConfig:
    """Configuration for the QwenAgent."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    history_n: int = 4  # Number of history steps to include in context
    max_tokens: int = 2048
    temperature: float = 0.0
    system_prompt: str | None = None  # Custom system prompt (uses default if None)
    extra_actions: list[type[Action]] = field(default_factory=list)


@dataclass
class AgentState:
    """Mutable state for the agent during a session."""

    screenshots: list[str] = field(default_factory=list)  # base64 encoded
    actions: list[str] = field(default_factory=list)  # action descriptions
    responses: list[str] = field(default_factory=list)  # raw LLM responses
    step_count: int = 0


class QwenAgent:
    """
    Qwen3-VL based computer use agent.

    Implements an observation-action loop for web navigation tasks.
    Uses normalized coordinates (0-999) for coordinate-independent actions.

    Usage:
        agent = QwenAgent(AgentConfig(model="qwen/qwen3-vl-8b-instruct"))

        while True:
            action = agent.predict(task, screenshot)
            if action is None or action.is_terminal:
                break
            execute_action(action)
            screenshot = capture_screenshot()
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        self.config = config or AgentConfig()
        self._system_prompt = self.config.system_prompt or get_system_prompt()
        self._extra_actions = self.config.extra_actions or None

        # Initialize OpenAI client for OpenRouter
        import os

        self.client = OpenAI(
            api_key=self.config.api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=self.config.base_url,
        )

        # Agent state
        self.state = AgentState()

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt."""
        return self._system_prompt

    def reset(self) -> None:
        """Reset agent state for a new task."""
        self.state = AgentState()

    def predict(self, instruction: str, screenshot: Image.Image) -> Action | None:
        """
        Generate the next action given a task instruction and screenshot.

        Screenshots are captured BEFORE each action is taken, following
        the Online-Mind2Web convention where screenshot[i] shows the state
        before action[i] was executed.

        Args:
            instruction: The task instruction (e.g., "Find the login page")
            screenshot: Current screenshot of the browser (state before action)

        Returns:
            The predicted Action, or None if parsing fails
        """
        # Process and store screenshot
        processed_screenshot = resize_image(screenshot, max_size=1024)
        screenshot_b64 = encode_image(processed_screenshot)
        self.state.screenshots.append(screenshot_b64)

        # Build messages with history
        messages = self._build_messages(instruction, screenshot_b64)

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        response_text = response.choices[0].message.content or ""
        self.state.responses.append(response_text)

        # Parse response to Action
        action = parse_action_from_response(response_text, self._extra_actions)

        if action:
            self.state.actions.append(action.to_description())

        self.state.step_count += 1

        return action

    def _build_messages(self, instruction: str, current_screenshot_b64: str) -> list[dict]:
        """
        Build the message list for the LLM, including history.

        Args:
            instruction: The task instruction
            current_screenshot_b64: Base64 encoded current screenshot

        Returns:
            List of message dicts for the OpenAI API
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

        # Build previous actions summary
        history_start = max(0, self.state.step_count - self.config.history_n)
        previous_actions = []
        for i in range(history_start):
            if i < len(self.state.actions):
                previous_actions.append(f"Step {i + 1}: {self.state.actions[i]}")
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        # Build instruction prompt
        instruction_prompt = f"""Please generate the next action according to the screenshot, instruction, and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions_str}"""

        # Add history with screenshots and responses
        history_len = min(self.config.history_n, len(self.state.responses))
        if history_len > 0:
            history_responses = self.state.responses[-history_len:]
            history_screenshots = self.state.screenshots[-history_len - 1 : -1]

            for idx in range(history_len):
                if idx < len(history_screenshots):
                    screenshot_b64 = history_screenshots[idx]
                    if idx == 0:
                        # First history item includes the full instruction
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{screenshot_b64}"
                                        },
                                    },
                                    {"type": "text", "text": instruction_prompt},
                                ],
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{screenshot_b64}"
                                        },
                                    }
                                ],
                            }
                        )

                # Add the assistant's response
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": history_responses[idx]}],
                    }
                )

            # Add current screenshot
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{current_screenshot_b64}"
                            },
                        }
                    ],
                }
            )
        else:
            # No history - first step
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{current_screenshot_b64}"
                            },
                        },
                        {"type": "text", "text": instruction_prompt},
                    ],
                }
            )

        return messages

    def get_action_history(self) -> list[str]:
        """Get the list of action descriptions taken so far."""
        return list(self.state.actions)

    def get_last_response(self) -> str | None:
        """Get the most recent LLM response."""
        if self.state.responses:
            return self.state.responses[-1]
        return None
