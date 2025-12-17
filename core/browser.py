"""
Kernel Browser Adapters for Computer Use Actions.

Provides adapters for executing agent actions via Kernel's browser API:
- KernelBrowserAdapter: Direct browser control via session ID
- PoolBrowserAdapter: Browser pool integration for scalable RL training

Browser Pools are a key feature for RL training, enabling efficient
browser acquisition/release across many parallel environments.

See: https://docs.onkernel.com/features/browser-pools
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import ssl
import time
from typing import TYPE_CHECKING, Callable

import websockets
from PIL import Image

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
)
from .utils import compute_image_similarity

if TYPE_CHECKING:
    from kernel import Kernel

logger = logging.getLogger(__name__)

# Type for custom action handlers: (adapter, action) -> should_continue
ActionHandler = Callable[["KernelBrowserAdapter", Action], bool]


# =============================================================================
# Base Browser Adapter
# =============================================================================


class KernelBrowserAdapter:
    """
    Adapter for executing computer use actions via Kernel's browser API.

    Handles:
    - Coordinate conversion from normalized (0-999) to pixel space
    - Screenshot capture
    - Action execution via Kernel's computer control API
    - Extensible custom action handlers

    Usage:
        kernel = Kernel()
        browser = kernel.browsers.create(stealth=True)
        adapter = KernelBrowserAdapter(kernel, browser.session_id)

        adapter.navigate("https://example.com")
        screenshot = adapter.capture_screenshot()
        adapter.execute_action(LeftClickAction(x=500, y=300))
    """

    def __init__(
        self,
        kernel: "Kernel",
        session_id: str,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
    ):
        """
        Initialize the adapter.

        Args:
            kernel: Kernel SDK client instance
            session_id: Browser session ID from kernel.browsers.create()
            viewport_width: Browser viewport width in pixels (default: 1920)
            viewport_height: Browser viewport height in pixels (default: 1080)
        """
        self.kernel = kernel
        self.session_id = session_id
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._custom_handlers: dict[str, ActionHandler] = {}

    def register_handler(
        self,
        action_type: str,
        handler: ActionHandler,
    ) -> None:
        """
        Register a custom handler for an action type.

        Args:
            action_type: The action_type string to handle
            handler: Callable taking (adapter, action) returning bool (should_continue)
        """
        self._custom_handlers[action_type] = handler

    def normalized_to_pixel(self, norm_x: int, norm_y: int) -> tuple[int, int]:
        """Convert normalized coordinates (0-999) to pixel coordinates."""
        pixel_x = int(norm_x * self.viewport_width / 999)
        pixel_y = int(norm_y * self.viewport_height / 999)
        return pixel_x, pixel_y

    def pixel_to_normalized(self, pixel_x: int, pixel_y: int) -> tuple[int, int]:
        """Convert pixel coordinates to normalized coordinates (0-999)."""
        norm_x = int(pixel_x * 999 / self.viewport_width)
        norm_y = int(pixel_y * 999 / self.viewport_height)
        return norm_x, norm_y

    def capture_screenshot(self) -> Image.Image:
        """Capture a screenshot of the current browser state."""
        image_data = self.kernel.browsers.computer.capture_screenshot(id=self.session_id)
        return Image.open(io.BytesIO(image_data.read()))

    def wait_for_screen_settle(
        self,
        baseline: Image.Image,
        change_threshold: float = 0.95,
        stability_threshold: float = 0.99,
        stability_count: int = 2,
        poll_interval: float = 0.3,
        change_timeout: float = 5.0,
        stability_timeout: float = 10.0,
    ) -> Image.Image:
        """
        Wait for screen to change from baseline, then stabilize.

        Two phases:
        1. CHANGE: Poll until screen differs significantly from baseline
        2. SETTLE: Poll until consecutive frames are similar

        Args:
            baseline: Screenshot to compare against (captured before action)
            change_threshold: Similarity below this = "changed" (default: 0.95)
            stability_threshold: Similarity above this = "stable" (default: 0.99)
            stability_count: Consecutive stable frames needed (default: 2)
            poll_interval: Seconds between captures (default: 0.3)
            change_timeout: Max seconds to wait for change (default: 5.0)
            stability_timeout: Max seconds to wait for stability (default: 10.0)

        Returns:
            The final stable screenshot
        """
        # Phase 1: Wait for change from baseline
        change_start = time.time()
        current = baseline

        while time.time() - change_start < change_timeout:
            time.sleep(poll_interval)
            current = self.capture_screenshot()
            similarity = compute_image_similarity(baseline, current)

            if similarity < change_threshold:
                break

        # Phase 2: Wait for stability
        stability_start = time.time()
        stable_frames = 0
        previous = current

        while time.time() - stability_start < stability_timeout:
            time.sleep(poll_interval)
            current = self.capture_screenshot()
            similarity = compute_image_similarity(previous, current)

            if similarity >= stability_threshold:
                stable_frames += 1
                if stable_frames >= stability_count:
                    break
            else:
                stable_frames = 0

            previous = current

        return current

    def navigate(self, url: str) -> Image.Image:
        """
        Navigate the browser to a URL and wait for the page to settle.

        Uses visual stability detection instead of unreliable networkidle.

        Args:
            url: URL to navigate to

        Returns:
            The stable screenshot after navigation completes
        """
        baseline = self.capture_screenshot()

        code = f'await page.goto("{url}", {{waitUntil: "domcontentloaded"}})'
        self.kernel.browsers.playwright.execute(id=self.session_id, code=code)

        return self.wait_for_screen_settle(baseline=baseline)

    def get_current_url(self) -> str:
        """Get the current page URL."""
        result = self.kernel.browsers.playwright.execute(
            id=self.session_id, code="return page.url()"
        )
        return result.result or ""

    def execute_action(self, action: Action) -> bool:
        """
        Execute an action via Kernel's computer control API.

        Custom handlers registered via register_handler() take precedence.

        Args:
            action: The Action to execute

        Returns:
            True if agent should continue, False if agent should stop
        """
        # Check for custom handler first
        action_type = getattr(action, "action_type", None)
        if action_type and action_type in self._custom_handlers:
            return self._custom_handlers[action_type](self, action)

        # Built-in handlers for standard OSWorld actions
        if isinstance(action, LeftClickAction):
            px, py = self.normalized_to_pixel(action.x, action.y)
            self.kernel.browsers.computer.click_mouse(id=self.session_id, x=px, y=py, button="left")

        elif isinstance(action, RightClickAction):
            px, py = self.normalized_to_pixel(action.x, action.y)
            self.kernel.browsers.computer.click_mouse(
                id=self.session_id, x=px, y=py, button="right"
            )

        elif isinstance(action, DoubleClickAction):
            px, py = self.normalized_to_pixel(action.x, action.y)
            self.kernel.browsers.computer.click_mouse(id=self.session_id, x=px, y=py, num_clicks=2)

        elif isinstance(action, TripleClickAction):
            px, py = self.normalized_to_pixel(action.x, action.y)
            self.kernel.browsers.computer.click_mouse(id=self.session_id, x=px, y=py, num_clicks=3)

        elif isinstance(action, MiddleClickAction):
            px, py = self.normalized_to_pixel(action.x, action.y)
            self.kernel.browsers.computer.click_mouse(
                id=self.session_id, x=px, y=py, button="middle"
            )

        elif isinstance(action, MouseMoveAction):
            px, py = self.normalized_to_pixel(action.x, action.y)
            self.kernel.browsers.computer.move_mouse(id=self.session_id, x=px, y=py)

        elif isinstance(action, LeftClickDragAction):
            start_px, start_py = self.normalized_to_pixel(action.start_x, action.start_y)
            end_px, end_py = self.normalized_to_pixel(action.end_x, action.end_y)
            self.kernel.browsers.computer.drag_mouse(
                id=self.session_id,
                path=[[start_px, start_py], [end_px, end_py]],
                button="left",
            )

        elif isinstance(action, TypeTextAction):
            self.kernel.browsers.computer.type_text(id=self.session_id, text=action.text)

        elif isinstance(action, KeyAction):
            # Convert key list to Kernel format: ["ctrl", "c"] -> "Ctrl+c"
            if len(action.keys) == 1:
                key_str = action.keys[0].capitalize()
            else:
                key_str = "+".join(k.capitalize() for k in action.keys)
            self.kernel.browsers.computer.press_key(id=self.session_id, keys=[key_str])

        elif isinstance(action, ScrollAction):
            px, py = self.normalized_to_pixel(action.x, action.y)
            self.kernel.browsers.computer.scroll(
                id=self.session_id,
                x=px,
                y=py,
                delta_x=action.delta_x,
                delta_y=action.delta_y,
            )

        elif isinstance(action, WaitAction):
            time.sleep(action.seconds)

        elif isinstance(action, TerminateAction):
            pass  # No browser action needed

        elif getattr(action, "is_terminal", False):
            pass  # Custom terminal action

        else:
            logger.warning(f"Unknown action type: {type(action).__name__}")

        return not getattr(action, "is_terminal", False)


# =============================================================================
# Browser Heartbeat (for long VLM inference)
# =============================================================================


class BrowserHeartbeat:
    """Keeps a browser session alive via periodic CDP WebSocket commands."""

    def __init__(self, session_id: str, cdp_ws_url: str, interval: int = 10):
        self.session_id = session_id
        self.cdp_ws_url = cdp_ws_url
        self.interval = interval
        self._task: asyncio.Task | None = None
        self._stopped = False
        self._ws = None
        self._cmd_id = 0
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    async def start(self) -> None:
        """Connect to CDP and start background heartbeat loop."""
        if self._ws is not None:
            return

        try:
            await asyncio.sleep(random.uniform(0, 2))
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.cdp_ws_url,
                    ssl=self._ssl_ctx,
                    ping_interval=None,
                    ping_timeout=None,
                ),
                timeout=10,
            )
            await self._send_heartbeat()
            self._stopped = False
            self._task = asyncio.create_task(self._heartbeat_loop())
        except Exception as e:
            logger.debug(f"Heartbeat connect failed for {self.session_id}: {e}")

    async def stop(self) -> None:
        """Stop heartbeat and close connection."""
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send_heartbeat(self) -> bool:
        if not self._ws:
            return False
        try:
            self._cmd_id += 1
            await self._ws.send(json.dumps({"id": self._cmd_id, "method": "Browser.getVersion"}))
            await asyncio.wait_for(self._ws.recv(), timeout=5)
            return True
        except Exception:
            return False

    async def _heartbeat_loop(self) -> None:
        while not self._stopped and self._ws:
            try:
                await asyncio.sleep(self.interval)
                if not self._stopped:
                    await self._send_heartbeat()
            except (asyncio.CancelledError, Exception):
                break


# =============================================================================
# Pool Browser Adapter (for RL training)
# =============================================================================


class PoolBrowserAdapter(KernelBrowserAdapter):
    """
    Browser adapter using Kernel browser pools for scalable RL training.

    Browser pools provide:
    - Pre-warmed browsers for fast acquisition
    - Automatic browser recycling and health checks
    - Efficient resource management across parallel environments

    Create a browser pool:
        kernel browser-pool create --name my-pool --size 10

    See: https://docs.onkernel.com/features/browser-pools

    Usage:
        kernel = Kernel()
        adapter = PoolBrowserAdapter(kernel, pool_name="my-pool")
        adapter.acquire()

        try:
            adapter.navigate("https://example.com")
            screenshot = adapter.capture_screenshot()
            # ... agent loop ...
        finally:
            adapter.release()
    """

    def __init__(
        self,
        kernel: "Kernel",
        pool_name: str,
        acquire_timeout_seconds: int = 60,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        heartbeat_interval: int = 10,
    ):
        """
        Initialize the pool adapter.

        Args:
            kernel: Kernel SDK client instance
            pool_name: Name of the browser pool to use
            acquire_timeout_seconds: Timeout for acquiring a browser (default: 60)
            viewport_width: Browser viewport width (default: 1920)
            viewport_height: Browser viewport height (default: 1080)
            heartbeat_interval: Seconds between CDP heartbeats (default: 10)
        """
        self.kernel = kernel
        self.pool_name = pool_name
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.heartbeat_interval = heartbeat_interval
        self._custom_handlers: dict = {}
        self.session_id: str | None = None
        self._browser_info: dict | None = None
        self._heartbeat: BrowserHeartbeat | None = None

    def acquire(self) -> None:
        """Acquire a browser from the pool."""
        if self.session_id is not None:
            raise RuntimeError("Browser already acquired. Call release() first.")

        browser = self.kernel.browser_pools.acquire(
            self.pool_name,
            acquire_timeout_seconds=self.acquire_timeout_seconds,
        )
        self.session_id = browser.session_id
        self._browser_info = {
            "session_id": browser.session_id,
            "cdp_ws_url": browser.cdp_ws_url,
            "live_view_url": getattr(browser, "browser_live_view_url", None),
        }

    async def start_heartbeat(self) -> None:
        """Start heartbeat to keep browser alive during long VLM inference."""
        if not self.session_id or self.heartbeat_interval <= 0:
            return
        cdp_ws_url = self._browser_info.get("cdp_ws_url") if self._browser_info else None
        if not cdp_ws_url:
            return
        self._heartbeat = BrowserHeartbeat(self.session_id, cdp_ws_url, self.heartbeat_interval)
        await self._heartbeat.start()

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat task."""
        if self._heartbeat:
            await self._heartbeat.stop()
            self._heartbeat = None

    def release(self, reuse: bool = True) -> None:
        """Release browser back to pool (sync)."""
        if self.session_id is None:
            return

        if self._heartbeat:
            self._heartbeat._stopped = True
            if self._heartbeat._task:
                self._heartbeat._task.cancel()
            self._heartbeat = None

        try:
            self.kernel.browser_pools.release(
                self.pool_name, session_id=self.session_id, reuse=reuse
            )
        finally:
            self.session_id = None
            self._browser_info = None

    async def release_async(self, reuse: bool = True) -> None:
        """Release browser back to pool (async)."""
        await self.stop_heartbeat()
        self.release(reuse=reuse)

    @property
    def live_view_url(self) -> str | None:
        """Live view URL for debugging."""
        return self._browser_info.get("live_view_url") if self._browser_info else None

    @property
    def cdp_ws_url(self) -> str | None:
        """CDP WebSocket URL."""
        return self._browser_info.get("cdp_ws_url") if self._browser_info else None

    def _ensure_acquired(self) -> None:
        if self.session_id is None:
            raise RuntimeError("Browser not acquired. Call acquire() first.")

    def capture_screenshot(self) -> Image.Image:
        self._ensure_acquired()
        return super().capture_screenshot()

    def navigate(self, url: str) -> Image.Image:
        self._ensure_acquired()
        return super().navigate(url)

    def execute_action(self, action: Action) -> bool:
        self._ensure_acquired()
        return super().execute_action(action)

    def get_current_url(self) -> str:
        self._ensure_acquired()
        return super().get_current_url()

    def wait_for_screen_settle(self, baseline: Image.Image, **kwargs) -> Image.Image:
        self._ensure_acquired()
        return super().wait_for_screen_settle(baseline, **kwargs)

    def __enter__(self) -> "PoolBrowserAdapter":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release(reuse=(exc_type is None))


# =============================================================================
# Mock Browser Adapter (for testing)
# =============================================================================


class MockBrowserAdapter:
    """
    Mock adapter for testing without Kernel.

    Useful for testing agent logic without spinning up real browsers.
    """

    def __init__(
        self,
        screenshot: Image.Image,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
    ):
        self.screenshot = screenshot
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.action_history: list[Action] = []
        self._custom_handlers: dict[str, ActionHandler] = {}

    def register_handler(self, action_type: str, handler: ActionHandler) -> None:
        self._custom_handlers[action_type] = handler

    def normalized_to_pixel(self, norm_x: int, norm_y: int) -> tuple[int, int]:
        pixel_x = int(norm_x * self.viewport_width / 999)
        pixel_y = int(norm_y * self.viewport_height / 999)
        return pixel_x, pixel_y

    def capture_screenshot(self) -> Image.Image:
        return self.screenshot

    def navigate(self, url: str) -> Image.Image:
        return self.screenshot

    def get_current_url(self) -> str:
        return "https://mock.example.com"

    def execute_action(self, action: Action) -> bool:
        self.action_history.append(action)

        action_type = getattr(action, "action_type", None)
        if action_type and action_type in self._custom_handlers:
            return self._custom_handlers[action_type](self, action)

        return not getattr(action, "is_terminal", False)
