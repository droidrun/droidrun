"""
Portal Client - Unified communication layer for DroidRun Portal app.

This module provides automatic TCP/Content Provider fallback for Portal communication.
"""

import asyncio
import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx
from async_adbutils import AdbDevice

logger = logging.getLogger("droidrun")

PORTAL_REMOTE_PORT = 8080  # Port on device where Portal HTTP server runs


class PortalClient:
    """
    Unified client for DroidRun Portal communication.

    Automatically handles TCP vs Content Provider fallback with the following strategy:
    - On init, checks for existing port forward and reuses it
    - If no forward exists, creates new one
    - Tests connection and sets tcp_available flag
    - All methods auto-select TCP or content provider based on availability
    - Port forwards persist until device disconnect (no explicit cleanup needed)

    Key features:
    - Reuses existing port forwards (no cleanup needed)
    - Automatic fallback to content provider if TCP fails
    - Zero explicit resource management
    - Graceful degradation

    Note: TCP mode is significantly faster but requires ADB port forwarding.
    Content provider mode works without port forwarding but has higher latency.
    """

    def __init__(self, device: AdbDevice, prefer_tcp: bool = False):
        """
        Initialize Portal client.

        Args:
            device: ADB device instance
            prefer_tcp: Whether to prefer TCP communication (will fallback to content provider if unavailable)

        Note:
            Call `await client.connect()` after initialization to establish connection.
        """
        self.device = device
        self.prefer_tcp = prefer_tcp
        self.tcp_available = False
        self.tcp_base_url = None
        self.local_tcp_port = None
        self._connected = False
        self._auth_token: Optional[str] = None

    async def connect(self) -> None:
        """
        Establish connection...
        """
        if self._connected:
            return

        if self.prefer_tcp:
            await self._try_enable_tcp()

        self._connected = True

    async def _ensure_connected(self) -> None:
        """Check if connected, raise error if not."""
        if not self._connected:
            await self.connect()

    async def _fetch_auth_token(self) -> None:
        """
        Fetch Bearer token from Portal via content provider.

        Stores the token in self._auth_token for use in TCP requests.
        Fails silently — older Portal APKs without the auth endpoint
        continue to work without an Authorization header.
        """
        try:
            output = await self.device.shell(
                "content query --uri content://com.droidrun.portal/auth_token"
            )
            if "result=" not in output:
                logger.debug(
                    "No auth_token endpoint on this Portal version — proceeding without auth"
                )
                return
            json_str = output.split("result=", 1)[1].strip()
            data = json.loads(json_str)
            token = data.get("result") if isinstance(data, dict) else None
            if token:
                self._auth_token = token
                logger.debug("Auth token fetched from Portal")
            else:
                logger.debug("auth_token response present but empty — proceeding without auth")
        except Exception as e:
            logger.debug(f"Could not fetch auth token ({e}) — proceeding without auth")

    def _auth_headers(self) -> Dict[str, str]:
        """Return Authorization header dict if a token is available, else empty dict."""
        if self._auth_token:
            return {"Authorization": f"Bearer {self._auth_token}"}
        return {}

    async def _try_enable_tcp(self) -> None:
        """
        Try to enable TCP communication. Fails silently and falls back to content provider.

        Strategy:
        1. Check if forward already exists → reuse
        2. If not, create new forward
        3. Fetch auth token from Portal
        4. Test connection with authenticated endpoint (/version)
        5. Set tcp_available flag
        """
        try:
            # Step 1: Check for existing forward
            local_port = await self._find_existing_forward()

            # Step 2: If no forward exists, create one
            if local_port is None:
                logger.debug(
                    f"No existing forward found, creating new forward for port {PORTAL_REMOTE_PORT}"
                )
                local_port = await self.device.forward_port(PORTAL_REMOTE_PORT)
                logger.debug(
                    f"Created forward: localhost:{local_port} -> device:{PORTAL_REMOTE_PORT}"
                )
            else:
                logger.debug(
                    f"Reusing existing forward: localhost:{local_port} -> device:{PORTAL_REMOTE_PORT}"
                )

            # Store local port
            self.local_tcp_port = local_port
            self.tcp_base_url = f"http://localhost:{local_port}"

            # Step 3: Fetch auth token before testing connection
            await self._fetch_auth_token()

            # Step 4: Test connection using authenticated endpoint
            if await self._test_connection():
                self.tcp_available = True
                logger.debug(f"✓ TCP mode enabled: {self.tcp_base_url}")
            else:
                # Step 4b: Try enabling the HTTP server via content provider
                logger.debug("TCP auth probe failed, trying to enable Portal HTTP server...")
                await self.device.shell(
                    "content insert --uri content://com.droidrun.portal/toggle_socket_server --bind enabled:b:true"
                )
                await asyncio.sleep(1)
                # Re-fetch token in case server restart rotated it
                await self._fetch_auth_token()
                if await self._test_connection():
                    self.tcp_available = True
                    logger.debug(
                        f"✓ TCP mode enabled after starting server: {self.tcp_base_url}"
                    )
                else:
                    logger.warning("TCP unavailable, using content provider fallback")
                    self.tcp_available = False

        except Exception as e:
            logger.warning(f"TCP unavailable ({e}), using content provider fallback")
            self.tcp_available = False

    async def _find_existing_forward(self) -> Optional[int]:
        """
        Check if a forward already exists for the Portal remote port.

        Returns:
            Local port number if forward exists, None otherwise
        """
        try:
            forwards = []
            async for forward in self.device.forward_list():
                forwards.append(forward)
            # forwards is a list of ForwardItem objects with serial, local, remote attributes
            for forward in forwards:
                if (
                    forward.serial == self.device.serial
                    and forward.remote == f"tcp:{PORTAL_REMOTE_PORT}"
                ):
                    # Extract local port from "tcp:12345"
                    match = re.search(r"tcp:(\d+)", forward.local)
                    if match:
                        local_port = int(match.group(1))
                        logger.debug(
                            f"Found existing forward: localhost:{local_port} -> {PORTAL_REMOTE_PORT}"
                        )
                        return local_port
        except Exception as e:
            logger.debug(f"Failed to check existing forwards: {e}")

        return None

    async def _test_connection(self) -> bool:
        """
        Test if TCP connection to Portal is working.

        Uses /version (requires auth) instead of /ping (no auth required) so
        that tcp_available=True confirms both reachability and auth are working.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.tcp_base_url}/version",
                    headers=self._auth_headers(),
                    timeout=5,
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"TCP connection test failed: {e}")
            return False

    def _parse_content_provider_output(
        self, raw_output: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse the raw ADB content provider output and extract JSON data.

        Args:
            raw_output: Raw output from ADB content query command

        Returns:
            Parsed JSON data or None if parsing failed
        """
        lines = raw_output.strip().split("\n")

        # Try line-by-line parsing
        for line in lines:
            line = line.strip()

            # Look for "result=" pattern (common content provider format)
            if "result=" in line:
                result_start = line.find("result=") + 7
                json_str = line[result_start:]
                try:
                    json_data = json.loads(json_str)
                    # Handle nested "result" or "data" field with JSON string (backward compatible)
                    if isinstance(json_data, dict):
                        # Check for 'result' first (new portal format), then 'data' (legacy)
                        inner_key = (
                            "result"
                            if "result" in json_data
                            else "data" if "data" in json_data else None
                        )
                        if inner_key:
                            inner_value = json_data[inner_key]
                            if isinstance(inner_value, str):
                                try:
                                    return json.loads(inner_value)
                                except json.JSONDecodeError:
                                    return inner_value
                            return inner_value
                    return json_data
                except json.JSONDecodeError:
                    continue

            # Fallback: try lines starting with JSON
            elif line.startswith("{") or line.startswith("["):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        # Last resort: try parsing entire output
        try:
            return json.loads(raw_output.strip())
        except json.JSONDecodeError:
            return None

    async def get_state(self) -> Dict[str, Any]:
        """
        Get device state (accessibility tree + phone state).
        Auto-selects TCP or content provider.

        Returns:
            Dictionary containing 'a11y_tree' and 'phone_state' keys
        """
        await self._ensure_connected()
        if self.tcp_available:
            return await self._get_state_tcp()
        return await self._get_state_content_provider()

    async def _get_state_tcp(self) -> Dict[str, Any]:
        """Get state via TCP."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.tcp_base_url}/state_full",
                    headers=self._auth_headers(),
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()

                    # Handle nested "result" or "data" field (backward compatible)
                    if isinstance(data, dict):
                        # Check for 'result' first (new portal format), then 'data' (legacy)
                        inner_key = (
                            "result"
                            if "result" in data
                            else "data" if "data" in data else None
                        )
                        if inner_key:
                            inner_value = data[inner_key]
                            if isinstance(inner_value, str):
                                try:
                                    return json.loads(inner_value)
                                except json.JSONDecodeError:
                                    pass
                            elif isinstance(inner_value, dict):
                                return inner_value
                    return data
                else:
                    logger.debug(
                        f"TCP get_state failed ({response.status_code}), using fallback"
                    )
                    return await self._get_state_content_provider()
        except Exception as e:
            logger.debug(f"TCP get_state error: {e}, using fallback")
            return await self._get_state_content_provider()

    async def _get_state_content_provider(self) -> Dict[str, Any]:
        """Get state via content provider (fallback)."""
        try:
            output = await self.device.shell(
                "content query --uri content://com.droidrun.portal/state_full"
            )
            state_data = self._parse_content_provider_output(output)

            if state_data is None:
                return {
                    "error": "Parse Error",
                    "message": "Failed to parse state data from ContentProvider",
                }

            # Handle nested "result" or "data" field if present (backward compatible)
            if isinstance(state_data, dict):
                # Check for 'result' first (new portal format), then 'data' (legacy)
                inner_key = (
                    "result"
                    if "result" in state_data
                    else "data" if "data" in state_data else None
                )
                if inner_key:
                    inner_value = state_data[inner_key]
                    if isinstance(inner_value, str):
                        try:
                            return json.loads(inner_value)
                        except json.JSONDecodeError:
                            return {
                                "error": "Parse Error",
                                "message": "Failed to parse nested JSON data",
                            }
                    elif isinstance(inner_value, dict):
                        return inner_value

            return state_data

        except Exception as e:
            return {"error": "ContentProvider Error", "message": str(e)}

    async def input_text(self, text: str, clear: bool = False) -> bool:
        """
        Input text via keyboard.
        Auto-selects TCP or content provider.

        Args:
            text: Text to input
            clear: Whether to clear existing text first

        Returns:
            True if successful, False otherwise
        """
        await self._ensure_connected()
        if self.tcp_available:
            return await self._input_text_tcp(text, clear)
        return await self._input_text_content_provider(text, clear)

    async def _input_text_tcp(self, text: str, clear: bool) -> bool:
        """Input text via TCP."""
        try:
            encoded = base64.b64encode(text.encode()).decode()
            payload = {"base64_text": encoded, "clear": clear}
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.tcp_base_url}/keyboard/input",
                    json=payload,
                    headers={"Content-Type": "application/json", **self._auth_headers()},
                    timeout=10,
                )
                if response.status_code == 200:
                    logger.debug("TCP input_text successful")
                    return True
                else:
                    logger.debug(
                        f"TCP input_text failed ({response.status_code}), using fallback"
                    )
                    return await self._input_text_content_provider(text, clear)
        except Exception as e:
            logger.debug(f"TCP input_text error: {e}, using fallback")
            return await self._input_text_content_provider(text, clear)

    async def _input_text_content_provider(self, text: str, clear: bool) -> bool:
        """Input text via content provider (fallback)."""
        try:
            encoded = base64.b64encode(text.encode()).decode()
            clear_str = "true" if clear else "false"
            cmd = (
                f'content insert --uri "content://com.droidrun.portal/keyboard/input" '
                f'--bind base64_text:s:"{encoded}" '
                f"--bind clear:b:{clear_str}"
            )
            await self.device.shell(cmd)
            logger.debug("Content provider input_text successful")
            return True
        except Exception as e:
            logger.error(f"Content provider input_text error: {e}")
            return False

    async def take_screenshot(self, hide_overlay: bool = True) -> bytes:
        """
        Take screenshot of device.
        Auto-selects TCP or ADB screencap.

        Args:
            hide_overlay: Whether to hide Portal overlay during screenshot

        Returns:
            Screenshot image bytes (PNG format)
        """
        await self._ensure_connected()
        if self.tcp_available:
            return await self._take_screenshot_tcp(hide_overlay)
        return await self._take_screenshot_adb()

    async def _take_screenshot_tcp(self, hide_overlay: bool) -> bytes:
        """Take screenshot via TCP."""
        try:
            url = f"{self.tcp_base_url}/screenshot"
            if not hide_overlay:
                url += "?hideOverlay=false"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._auth_headers(),
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    # Check for 'result' first (new portal format), then 'data' (legacy)
                    if data.get("status") == "success":
                        inner_key = (
                            "result"
                            if "result" in data
                            else "data" if "data" in data else None
                        )
                        if inner_key:
                            logger.debug("Screenshot taken via TCP")
                            return base64.b64decode(data[inner_key])
                    logger.debug(
                        "TCP screenshot failed (invalid response), using fallback"
                    )
                    return await self._take_screenshot_adb()
                else:
                    logger.debug(
                        f"TCP screenshot failed ({response.status_code}), using fallback"
                    )
                    return await self._take_screenshot_adb()
        except Exception as e:
            logger.debug(f"TCP screenshot error: {e}, using fallback")
            return await self._take_screenshot_adb()

    async def _take_screenshot_adb(self) -> bytes:
        """Take screenshot via ADB screencap (fallback)."""
        data = await self.device.screenshot_bytes()
        logger.debug("Screenshot taken via ADB")
        return data

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get installed apps with package name and label.

        Note: Currently only supports content provider (no TCP endpoint exists yet)

        Args:
            include_system: Whether to include system apps

        Returns:
            List of dicts with 'package' and 'label' keys
        """
        await self._ensure_connected()
        try:
            logger.debug("Getting apps via content provider")

            # Query content provider
            output = await self.device.shell(
                "content query --uri content://com.droidrun.portal/packages"
            )
            packages_data = self._parse_content_provider_output(output)

            if not packages_data:
                logger.warning("No packages data found in content provider response")
                return []

            # Handle both formats:
            # - New format: array directly (via RawArray -> result: [...])
            # - Legacy format: wrapped in {"packages": [...]}
            packages_list = None
            if isinstance(packages_data, list):
                # New format: packages_data is already the list
                packages_list = packages_data
            elif isinstance(packages_data, dict):
                if "packages" in packages_data:
                    # Legacy format: wrapped in {"packages": [...]}
                    packages_list = packages_data["packages"]
                else:
                    # May be wrapped in result/data
                    inner_key = (
                        "result"
                        if "result" in packages_data
                        else "data" if "data" in packages_data else None
                    )
                    if inner_key:
                        inner_value = packages_data[inner_key]
                        if isinstance(inner_value, list):
                            packages_list = inner_value
                        elif (
                            isinstance(inner_value, dict) and "packages" in inner_value
                        ):
                            packages_list = inner_value["packages"]

            if not packages_list:
                logger.warning("Could not extract packages list from response")
                return []

            # Filter and format apps
            apps = []
            for package_info in packages_list:
                if not include_system and package_info.get("isSystemApp", False):
                    continue

                apps.append(
                    {
                        "package": package_info.get("packageName", ""),
                        "label": package_info.get("label", ""),
                    }
                )

            logger.debug(f"Found {len(apps)} apps")
            return apps

        except Exception as e:
            logger.error(f"Error getting apps: {e}")
            raise ValueError(f"Error getting apps: {e}") from e

    async def get_version(self) -> str:
        """Get Portal app version."""
        await self._ensure_connected()
        if self.tcp_available:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.tcp_base_url}/version",
                        headers=self._auth_headers(),
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # Check for 'result' first (new portal format), then 'data' (legacy)
                        inner_key = (
                            "result"
                            if "result" in data
                            else "data" if "data" in data else None
                        )
                        if inner_key:
                            return data[inner_key]
                        return data.get("status", "unknown")
            except Exception:
                pass

        # Fallback to content provider
        try:
            output = await self.device.shell(
                "content query --uri content://com.droidrun.portal/version"
            )
            result = self._parse_content_provider_output(output)
            if result:
                # Check for 'result' first (new portal format), then 'data' (legacy)
                inner_key = (
                    "result"
                    if "result" in result
                    else "data" if "data" in result else None
                )
                if inner_key:
                    return result[inner_key]
        except Exception:
            pass

        return "unknown"

    async def ping(self) -> Dict[str, Any]:
        """
        Test Portal connection and verify state availability.

        Returns:
            Dictionary with status and connection details
        """
        await self._ensure_connected()
        if self.tcp_available:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.tcp_base_url}/ping",
                        headers=self._auth_headers(),
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        try:
                            tcp_response = response.json() if response.content else {}
                            result = {
                                "status": "success",
                                "method": "tcp",
                                "url": self.tcp_base_url,
                                "response": tcp_response,
                            }
                        except json.JSONDecodeError:
                            result = {
                                "status": "success",
                                "method": "tcp",
                                "url": self.tcp_base_url,
                                "response": response.text,
                            }
                    else:
                        return {
                            "status": "error",
                            "method": "tcp",
                            "message": f"HTTP {response.status_code}: {response.text}",
                        }
            except Exception as e:
                return {"status": "error", "method": "tcp", "message": str(e)}
        else:
            # Test content provider
            try:
                output = await self.device.shell(
                    "content query --uri content://com.droidrun.portal/state"
                )
                if "Row: 0 result=" in output:
                    result = {"status": "success", "method": "content_provider"}
                else:
                    return {
                        "status": "error",
                        "method": "content_provider",
                        "message": "Invalid response",
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "method": "content_provider",
                    "message": str(e),
                }

        # Verify state has the required keys
        try:
            state = await self.get_state()
            required = ("a11y_tree", "phone_state", "device_context")
            missing = [k for k in required if k not in state]
            if missing:
                return {
                    "status": "error",
                    "method": result.get("method", "unknown"),
                    "message": f"incompatible portal — missing {', '.join(missing)}",
                }
        except Exception as e:
            return {
                "status": "error",
                "method": result.get("method", "unknown"),
                "message": f"state check failed: {e}",
            }

        return result
