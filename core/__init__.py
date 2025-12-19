"""
Core infrastructure for computer use RL training.

This module provides the generic, reusable components:
- Agent: VLM-based computer use agent
- Actions: Structured action types (OSWorld-compatible)
- Browser: Kernel browser adapter and acquired_browser context manager
- Reward Models: WebJudge and base interfaces
- Prompts: System prompt utilities
- Utils: Image processing and environment setup
"""

from .actions import (
    Action,
    DoubleClickAction,
    KeyAction,
    LeftClickAction,
    LeftClickDragAction,
    MiddleClickAction,
    MouseMoveAction,
    RightClickAction,
    ScrollAction,
    TerminateAction,
    TripleClickAction,
    TypeTextAction,
    WaitAction,
    parse_action_from_response,
)
from .agent import AgentConfig, QwenAgent
from .agent_loop import AgentLoopResult, StepResult, run_agent_loop
from .browser import KernelBrowserAdapter, MockBrowserAdapter, acquired_browser
from .prompts import build_system_prompt, get_system_prompt
from .reward_models import EvaluationResult, RewardModel, Trajectory, WebJudge, WebJudgeResult
from .tracking import (
    begin_episode,
    create_step_callbacks,
    finish_episode,
    flush_raindrop,
    generate_id,
    image_to_data_url,
    init_raindrop,
    is_raindrop_enabled,
    make_image_attachment,
    run_webjudge_with_tracking,
    shutdown_raindrop,
    track_webjudge_signal,
)
from .utils import (
    encode_image,
    load_image,
    normalized_to_pixel,
    pixel_to_normalized,
    resize_image,
    setup_environment,
)

__all__ = [
    # Agent
    "QwenAgent",
    "AgentConfig",
    # Agent Loop
    "run_agent_loop",
    "AgentLoopResult",
    "StepResult",
    # Actions
    "Action",
    "LeftClickAction",
    "RightClickAction",
    "DoubleClickAction",
    "TripleClickAction",
    "MiddleClickAction",
    "MouseMoveAction",
    "LeftClickDragAction",
    "TypeTextAction",
    "KeyAction",
    "ScrollAction",
    "WaitAction",
    "TerminateAction",
    "parse_action_from_response",
    # Browser
    "KernelBrowserAdapter",
    "MockBrowserAdapter",
    "acquired_browser",
    # Reward Models
    "RewardModel",
    "EvaluationResult",
    "Trajectory",
    "WebJudge",
    "WebJudgeResult",
    # Prompts
    "build_system_prompt",
    "get_system_prompt",
    # Utils
    "resize_image",
    "load_image",
    "encode_image",
    "pixel_to_normalized",
    "normalized_to_pixel",
    "setup_environment",
    # Tracking
    "init_raindrop",
    "shutdown_raindrop",
    "is_raindrop_enabled",
    "flush_raindrop",
    "generate_id",
    "image_to_data_url",
    "make_image_attachment",
    "create_step_callbacks",
    "begin_episode",
    "finish_episode",
    "track_webjudge_signal",
    "run_webjudge_with_tracking",
]
