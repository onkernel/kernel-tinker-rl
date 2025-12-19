"""
Raindrop analytics integration for agent tracking.

Provides utilities for tracking agent runs at different granularities:
- Step-level: For interactive single-task runs (run_agent.py)
- Episode-level: For batch evaluation (evaluate.py)

Usage:
    from core.tracking import (
        init_raindrop, shutdown_raindrop, is_raindrop_enabled,
        generate_id, create_step_callbacks,
    )

    # Initialize at startup
    if init_raindrop():
        print("Raindrop enabled")

    # Generate conversation/batch IDs
    convo_id = generate_id()

    # Create callbacks for step-level tracking
    on_step_start, on_step_complete, on_action_overlay, _ = create_step_callbacks(
        model="qwen/qwen3-vl-8b-instruct",
        convo_id=convo_id,
    )

    # Shutdown at end
    shutdown_raindrop()
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Literal

from cuid2 import cuid_wrapper
from PIL import Image
import raindrop.analytics as raindrop
from raindrop.models import Attachment

from .utils import encode_image

if TYPE_CHECKING:
    from .actions import Action
    from .agent_loop import StepResult

# Generate cuid2 IDs
_cuid2 = cuid_wrapper()

# Module-level state
_raindrop_enabled = False


# =============================================================================
# Initialization and Lifecycle
# =============================================================================


def init_raindrop() -> bool:
    """
    Initialize Raindrop if RAINDROP_WRITE_KEY is set.

    Returns:
        True if Raindrop was initialized, False otherwise.
    """
    global _raindrop_enabled
    write_key = os.getenv("RAINDROP_WRITE_KEY")
    if not write_key:
        return False

    raindrop.init(write_key, tracing_enabled=True)
    _raindrop_enabled = True
    return True


def shutdown_raindrop() -> None:
    """Flush and shutdown Raindrop if it was initialized."""
    if _raindrop_enabled:
        try:
            raindrop.flush()
            raindrop.shutdown()
        except Exception:
            pass  # Silently ignore shutdown errors


def is_raindrop_enabled() -> bool:
    """Check if Raindrop is enabled."""
    return _raindrop_enabled


def generate_id() -> str:
    """Generate a unique ID for tracking (conversation/batch IDs)."""
    return _cuid2()


def flush_raindrop() -> None:
    """Flush pending Raindrop events."""
    if _raindrop_enabled:
        try:
            raindrop.flush()
        except Exception:
            pass


# =============================================================================
# Image Attachment Helpers
# =============================================================================

AttachmentRole = Literal["input", "output", "context"]


def image_to_data_url(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64 data URL."""
    b64 = encode_image(image, format=format)
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{b64}"


def make_image_attachment(
    image: Image.Image,
    name: str,
    role: AttachmentRole = "input",
) -> Attachment:
    """Create a Raindrop image attachment from a PIL Image."""
    return Attachment(
        type="image",
        value=image_to_data_url(image),
        name=name,
        role=role,
    )


# =============================================================================
# Step-Level Callbacks (for agent loop)
# =============================================================================


def create_step_callbacks(
    model: str,
    convo_id: str,
    nav_step_offset: int = 1,
) -> tuple[
    Callable[[int, Image.Image], None],
    Callable[[int, "StepResult"], None],
    Callable[[int, "Action", Image.Image], None],
    dict[str, Any],
]:
    """
    Create callbacks for the agent loop that handle Raindrop integration.

    These callbacks track each step of the agent loop as a separate interaction,
    including input screenshots and action overlays.

    Args:
        model: Model name for tracking properties
        convo_id: Conversation ID to group interactions
        nav_step_offset: Offset to add to step numbers (e.g., 1 if navigation is step 1)

    Returns:
        Tuple of (on_step_start, on_step_complete, on_action_overlay, shared_state)
    """
    # Shared state for passing data between callbacks
    shared_state: dict[str, Any] = {
        "step_interaction": None,
        "output_attachments": [],
    }

    def on_step_start(step: int, screenshot: Image.Image) -> None:
        """Called at the start of each step."""
        display_step = step + nav_step_offset

        if _raindrop_enabled:
            input_attachments = [
                make_image_attachment(screenshot, f"step_{display_step}_input", "input")
            ]
            shared_state["step_interaction"] = raindrop.begin(
                user_id=os.getenv("USER") or "system",
                event="agent_loop",
                input=f"Step {display_step}: Predict next action",
                convo_id=convo_id,
                attachments=input_attachments,
                properties={
                    "step": display_step,
                    "model": model,
                },
            )
        shared_state["output_attachments"] = []

    def on_action_overlay(step: int, action: "Action", overlay: Image.Image) -> None:
        """Called when an action overlay image is generated."""
        display_step = step + nav_step_offset
        shared_state["output_attachments"].append(
            make_image_attachment(overlay, f"step_{display_step}_click_overlay", "output")
        )

    def on_step_complete(step: int, result: "StepResult") -> None:
        """Called after each step completes."""
        step_interaction = shared_state.get("step_interaction")
        output_attachments = shared_state.get("output_attachments", [])

        # Handle error/parse failure
        if result.error or result.action is None:
            if step_interaction:
                step_interaction.finish(output=result.error or "Failed to parse action")
            return

        action_desc = result.action_desc

        if step_interaction:
            if output_attachments:
                step_interaction.add_attachments(output_attachments)
            step_interaction.finish(
                output=action_desc,
                properties={
                    "is_terminal": result.is_terminal,
                    "predict_time": result.predict_time,
                    "exec_time": result.exec_time,
                    "total_time": result.total_time,
                },
            )

    return on_step_start, on_step_complete, on_action_overlay, shared_state


# =============================================================================
# Episode-Level Tracking (for batch evaluation)
# =============================================================================


def begin_episode(
    task: str,
    convo_id: str,
    properties: dict[str, Any] | None = None,
) -> Any | None:
    """
    Begin tracking an episode.

    Args:
        task: Task description
        convo_id: Conversation/batch ID to group episodes
        properties: Additional properties to track

    Returns:
        Raindrop interaction object, or None if Raindrop is disabled.
    """
    if not _raindrop_enabled:
        return None
    return raindrop.begin(
        user_id=os.getenv("USER") or "system",
        event="evaluate_episode",
        input=task,
        convo_id=convo_id,
        properties=properties or {},
    )


def finish_episode(
    interaction: Any | None,
    output: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """
    Finish tracking an episode.

    Args:
        interaction: Raindrop interaction from begin_episode
        output: Output/result description
        properties: Additional properties to track
    """
    if interaction is None:
        return
    interaction.finish(output=output, properties=properties or {})


def track_webjudge_signal(
    interaction: Any | None,
    success: bool,
    score: float,
    reasoning: str,
    webjudge_model: str,
) -> None:
    """
    Track WebJudge result as a signal on an interaction.

    Args:
        interaction: Raindrop interaction to attach signal to
        success: Whether the task was successful
        score: WebJudge score
        reasoning: WebJudge reasoning
        webjudge_model: Model used for WebJudge
    """
    if interaction is None or not _raindrop_enabled:
        return
    sentiment = "POSITIVE" if success else "NEGATIVE"
    raindrop.track_signal(
        event_id=interaction.id,
        name="webjudge_result",
        signal_type="feedback",
        sentiment=sentiment,
        comment=f"Score: {score}. {reasoning[:200]}",
        properties={
            "success": success,
            "score": score,
            "webjudge_model": webjudge_model,
        },
    )


# =============================================================================
# Decorated Functions (for auto-tracking)
# =============================================================================


@raindrop.task("webjudge")
async def run_webjudge_with_tracking(webjudge, trajectory):
    """
    Run WebJudge evaluation with Raindrop tracking.

    This is a convenience wrapper that auto-tracks the WebJudge call.
    """
    return await webjudge.evaluate(trajectory)



