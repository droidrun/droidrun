"""AndroidSSHDriver — SSH-based device driver.

Wraps ``SSHDevice`` + ``PortalClientHTTP`` to provide clean device I/O
via SSH shell commands and direct HTTP Portal communication.
No ADB required.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shlex
import subprocess
from typing import Any, Dict, List, Optional, Union

from droidrun.tools.android.portal_client_http import PortalClientHTTP
from droidrun.tools.driver.base import DeviceDriver

logger = logging.getLogger("droidrun")


def _list_to_cmdline(args: Union[list, tuple]) -> str:
    """Convert a list of arguments to a shell command string.

    Uses shlex.quote for proper escaping (unlike subprocess.list2cmdline).
    """
    return " ".join(map(shlex.quote, args))


class SSHDevice:
    """Thin wrapper around SSH that mimics the adbutils Device interface.

    Executes commands on the remote Android device via SSH using
    a configurable su binary path for root access.
    """

    def __init__(self, target: str, su_path: str = "/debug_ramdisk/su") -> None:
        """
        Args:
            target: SSH target, e.g. "redmi9" or "user@192.168.1.100"
            su_path: Path to su binary on device (default: "/debug_ramdisk/su")
        """
        self.target = target
        self.su_path = su_path

    def shell(
        self,
        cmdargs: Union[str, list, tuple],
        encoding: str | None = "utf-8",
    ) -> str | bytes:
        """Execute a shell command on the remote device and return output.

        Args:
            cmdargs: Command string or list of arguments
            encoding: Output encoding. Pass None to get raw bytes.

        Returns:
            Command stdout as str (if encoding set) or bytes (if encoding=None).
            Returns empty string/bytes on failure.
        """
        if isinstance(cmdargs, (list, tuple)):
            cmdargs = _list_to_cmdline(cmdargs)

        cmd = ["ssh", self.target, self.su_path, "-c", cmdargs]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as error:
            logger.debug(f"SSH shell command failed: {error}")
            return "" if encoding else b""

        if encoding:
            return result.stdout.decode(encoding)
        return result.stdout

    def click(self, x: int, y: int) -> None:
        logger.debug(f"SSHDevice click at ({x}, {y})")
        self.shell(["input", "tap", str(x), str(y)])

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_seconds: float = 1.0,
    ) -> None:
        logger.debug(
            f"SSHDevice swipe from ({start_x}, {start_y}) to ({end_x}, {end_y})"
        )
        duration_ms = str(int(duration_seconds * 1000))
        self.shell(
            [
                "input",
                "swipe",
                str(start_x),
                str(start_y),
                str(end_x),
                str(end_y),
                duration_ms,
            ]
        )

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_seconds: float = 1.0,
    ) -> None:
        logger.debug(
            f"SSHDevice drag from ({start_x}, {start_y}) to ({end_x}, {end_y})"
        )
        duration_ms = str(int(duration_seconds * 1000))
        self.shell(
            [
                "input",
                "draganddrop",
                str(start_x),
                str(start_y),
                str(end_x),
                str(end_y),
                duration_ms,
            ]
        )

    def keyevent(self, keycode: Union[int, str]) -> None:
        """Send a key event via ``input keyevent``."""
        self.shell(["input", "keyevent", str(keycode)])

    def app_start(self, package_name: str, activity: Optional[str] = None) -> None:
        """Start an app via ``am start`` or ``monkey``."""
        if activity:
            self.shell(["am", "start", "-n", f"{package_name}/{activity}"])
        else:
            self.shell(
                [
                    "monkey",
                    "-p",
                    package_name,
                    "-c",
                    "android.intent.category.LAUNCHER",
                    "1",
                ]
            )

    def screenshot_bytes(self) -> bytes:
        """Take a screenshot and return raw PNG bytes."""
        png_bytes = self.shell(["screencap", "-p"], encoding=None)
        return png_bytes

    def force_stop(self, package_name: str) -> None:
        """Force stop an app."""
        self.shell(["am", "force-stop", package_name])


class AndroidSSHDriver(DeviceDriver):
    """Android device driver using SSH for shell commands and HTTP for Portal."""

    supported = {
        "tap",
        "swipe",
        "input_text",
        "press_key",
        "start_app",
        "screenshot",
        "get_ui_tree",
        "get_date",
        "get_apps",
        "list_packages",
        "install_app",
        "drag",
    }

    def __init__(
        self,
        target: str,
        portal_url: str,
        portal_token: str,
        su_path: str = "/debug_ramdisk/su",
    ) -> None:
        """
        Args:
            target: SSH target, e.g. "redmi9" or "user@192.168.1.100"
            portal_url: Base URL of the Portal HTTP server, e.g. "http://192.168.1.100:8080"
            portal_token: Bearer token for Portal authentication
            su_path: Path to su binary on device (default: "/debug_ramdisk/su")
        """
        self._target = target
        self._portal_url = portal_url
        self._portal_token = portal_token
        self._su_path = su_path
        self.device: SSHDevice | None = None
        self.portal: PortalClientHTTP | None = None
        self._connected = False
        # Auto-connect on initialization
        asyncio.ensure_future(self.ensure_connected())

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return

        self.device = SSHDevice(self._target, self._su_path)
        self.portal = PortalClientHTTP(self._portal_url, self._portal_token)
        await self.portal.connect()

        self._connected = True
        logger.debug(
            f"AndroidSSHDriver connected: target={self._target}, portal={self._portal_url}, su_path={self._su_path}"
        )

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    # -- input actions -------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        await self.ensure_connected()
        self.device.click(x, y)

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        await self.ensure_connected()
        self.device.swipe(x1, y1, x2, y2, duration_ms / 1000)
        await asyncio.sleep(duration_ms / 1000)

    async def input_text(self, text: str, clear: bool = False) -> bool:
        await self.ensure_connected()
        return await self.portal.input_text(text, clear)

    async def press_key(self, keycode: int) -> None:
        await self.ensure_connected()
        self.device.keyevent(keycode)

    async def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 3.0,
    ) -> None:
        await self.ensure_connected()
        self.device.drag(x1, y1, x2, y2, duration)

    # -- app management ------------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        await self.ensure_connected()
        try:
            logger.debug(f"Starting app {package} with activity {activity}")
            if not activity:
                dumpsys_output = self.device.shell(
                    f"cmd package resolve-activity --brief {package}"
                )
                lines = dumpsys_output.strip().splitlines()
                if len(lines) < 2:
                    raise ValueError(
                        f"Unexpected resolve-activity output: {dumpsys_output!r}"
                    )
                activity = lines[1].split("/")[1]

            logger.debug(f"Activity: {activity}")
            self.device.app_start(package, activity)
            logger.debug(f"App started: {package} with activity {activity}")
            return f"App started: {package} with activity {activity}"
        except Exception as error:
            return f"Failed to start app {package}: {error}"

    async def install_app(self, path: str, **kwargs) -> str:
        """Install an APK from a local path by copying it to the device via SCP then installing.

        Args:
            path: Local path to the APK file
            kwargs:
                reinstall (bool): Reinstall if already installed (default False)
                grant_permissions (bool): Grant all permissions (default True)
        """
        await self.ensure_connected()
        if not os.path.exists(path):
            return f"Failed to install app: APK file not found at {path}"

        reinstall = kwargs.get("reinstall", False)
        grant_permissions = kwargs.get("grant_permissions", True)

        remote_path = f"/data/local/tmp/{os.path.basename(path)}"

        # Copy APK to device via SCP
        logger.debug(f"Copying APK to device: {path} -> {self._target}:{remote_path}")
        scp_cmd = ["scp", path, f"{self._target}:{remote_path}"]
        try:
            subprocess.run(
                scp_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as error:
            return f"Failed to copy APK to device: {error}"

        # Build pm install flags
        flags: List[str] = []
        if reinstall:
            flags.append("-r")
        if grant_permissions:
            flags.append("-g")

        flags_str = " ".join(flags)
        install_cmd = f"pm install {flags_str} {remote_path}".strip()

        logger.debug(f"Installing APK: {install_cmd}")
        result = self.device.shell(install_cmd)

        # Clean up remote APK
        self.device.shell(f"rm -f {remote_path}")

        logger.debug(f"Install result: {result}")
        return result.strip()

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        await self.ensure_connected()
        return await self.portal.get_apps(include_system)

    async def list_packages(self, include_system: bool = False) -> List[str]:
        await self.ensure_connected()
        filter_flag = "" if include_system else "-3"
        output = self.device.shell(f"pm list packages {filter_flag}")
        packages = []
        for line in output.strip().splitlines():
            line = line.strip()
            if line.startswith("package:"):
                packages.append(line[len("package:") :])
        return packages

    # -- state / observation -------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        await self.ensure_connected()
        return await self.portal.take_screenshot(hide_overlay)

    async def get_ui_tree(self) -> Dict[str, Any]:
        await self.ensure_connected()
        return await self.portal.get_state()

    async def get_date(self) -> str:
        await self.ensure_connected()
        result = self.device.shell("date")
        return result.strip()

    # -- element search -------------------------------------------------

    def _match_text(self, text: str, pattern: str, match_type: str) -> bool:
        """Match text against pattern with specified match type.

        Args:
            text: Text to match
            pattern: Pattern to match against
            match_type: Match type ("contains", "exact", "startswith", "endswith")

        Returns:
            True if match successful, False otherwise
        """
        if not text or not pattern:
            return False
        text = str(text) if text is not None else ""
        pattern = str(pattern) if pattern is not None else ""

        if match_type == "exact":
            return text == pattern
        elif match_type == "startswith":
            return text.startswith(pattern)
        elif match_type == "endswith":
            return text.endswith(pattern)
        else:  # contains (default)
            return pattern in text

    def _search_node(
        self, node: Dict[str, Any], search_conditions: Dict[str, Any], match_type: str
    ) -> List[Dict[str, Any]]:
        """Recursively search a single node and its children.

        Args:
            node: A11y node to search
            search_conditions: Dictionary of conditions to match
            match_type: Match type for string comparisons

        Returns:
            List of matching nodes
        """
        matched = []

        # Check if current node matches all conditions
        is_match = True
        for key, pattern in search_conditions.items():
            node_value = node.get(key)
            if node_value is None:
                is_match = False
                break

            if isinstance(node_value, str):
                if not self._match_text(node_value, pattern, match_type):
                    is_match = False
                    break
            else:
                # For non-string values, use exact match
                if node_value != pattern:
                    is_match = False
                    break

        if is_match:
            matched.append(node)

        # Recursively search children
        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    matched.extend(
                        self._search_node(child, search_conditions, match_type)
                    )

        return matched

    async def search_element(
        self,
        search_conditions: Dict[str, Any],
        match_type: str = "contains",
    ) -> List[Dict[str, Any]]:
        """Search for UI elements matching conditions.

        Args:
            search_conditions: Dictionary of conditions to match, e.g.:
                {"text": "Submit", "className": "android.widget.Button"}
            match_type: Match type for string comparisons:
                - "contains": Contains pattern (default)
                - "exact": Exact match
                - "startswith": Starts with pattern
                - "endswith": Ends with pattern

        Returns:
            List of matching UI elements
        """
        await self.ensure_connected()

        matched_elements: List[Dict[str, Any]] = []

        try:
            state = await self.portal.get_state()
            node = state.get("a11y_tree", {})
            # Search the single root node
            matched_elements.extend(
                self._search_node(node, search_conditions, match_type)
            )

            logger.info(f"Found {len(matched_elements)} matching elements")
            if not matched_elements:
                logger.debug(f"{node}")

        except KeyError:
            logger.error("错误：输入数据中缺少 'a11y_tree' 键")
        except Exception as e:
            logger.error(f"解析 a11y_tree 失败：{str(e)}")

        return matched_elements

    # -- element actions -----------------------------------------------------

    async def tap_element_relative(
        self,
        element: Dict[str, Any],
        position: str = "center",
    ) -> str:
        """Tap on a UI element at a specific relative position.

        Args:
            element: UI element dict containing boundsInScreen and other properties
            position: One of "center" (default), "top", "bottom", "left", "right"

        Returns:
            Result message describing the tap action
        """
        await self.ensure_connected()

        try:
            # Get bounds from boundsInScreen
            bounds_in_screen = element.get("boundsInScreen")
            if not bounds_in_screen:
                return f"Error: Element has no boundsInScreen and cannot be tapped."

            # Extract coordinates from boundsInScreen dict
            left = bounds_in_screen.get("left")
            top = bounds_in_screen.get("top")
            right = bounds_in_screen.get("right")
            bottom = bounds_in_screen.get("bottom")

            if any(v is None for v in [left, top, right, bottom]):
                return f"Error: Invalid boundsInScreen format for element: {bounds_in_screen}"

            # Calculate tap coordinates based on position
            if position == "center":
                x = (left + right) // 2
                y = (top + bottom) // 2
            elif position == "top":
                x = (left + right) // 2
                y = top + 2  # 2px offset to avoid edge
            elif position == "bottom":
                x = (left + right) // 2
                y = bottom - 2
            elif position == "left":
                x = left + 2
                y = (top + bottom) // 2
            elif position == "right":
                x = right - 2
                y = (top + bottom) // 2
            else:
                return f"Error: Unknown position '{position}', must be one of center/top/bottom/left/right"

            logger.debug(
                f"Tapping element at position '{position}' (coordinates: {x}, {y})"
            )

            # Tap the element
            await self.tap(x, y)
            await asyncio.sleep(0.5)

            response_parts = [
                f"Tapped element",
                f"Position: {position}",
                f"Coordinates: ({x}, {y})",
                f"Text: '{element.get('text', 'No text')}'",
                f"Class: {element.get('className', 'Unknown class')}",
            ]
            return " | ".join(response_parts)

        except Exception as e:
            return f"Error: {str(e)}"
