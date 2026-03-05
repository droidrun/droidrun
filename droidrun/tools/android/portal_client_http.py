"""
Portal Client HTTP - Direct HTTP communication layer for DroidRun Portal app.

Simplified client that communicates directly via HTTP without ADB or TCP port forwarding.
Requires a Bearer token for authentication.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("droidrun")


class PortalClientHTTP:
    """
    HTTP-only client for DroidRun Portal communication.

    Communicates directly with the Portal HTTP server using a base URL and Bearer token.
    No ADB or TCP port forwarding required.

    Usage::

        client = PortalClientHTTP("http://192.168.1.100:8080", token="YOUR_TOKEN")
        await client.connect()
        state = await client.get_state()
    """

    def __init__(self, base_url: str, token: str) -> None:
        """
        Initialize Portal HTTP client.

        Args:
            base_url: Base URL of the Portal HTTP server, e.g. "http://192.168.1.100:8080"
            token: Bearer token for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._headers = {"Authorization": f"Bearer {token}"}
        self._connected = False

    async def connect(self) -> None:
        """Test connection to the Portal HTTP server."""
        if self._connected:
            return

        if not await self._test_connection():
            raise ConnectionError(
                f"Failed to connect to Portal at {self.base_url}. "
                "Check the URL and token."
            )

        self._connected = True
        logger.debug(f"✓ Connected to Portal HTTP server: {self.base_url}")

    async def _ensure_connected(self) -> None:
        """Connect if not already connected."""
        if not self._connected:
            await self.connect()

    async def _test_connection(self) -> bool:
        """Test if HTTP connection to Portal is working."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/ping",
                    headers=self._headers,
                    timeout=5,
                )
                return response.status_code == 200
        except Exception as error:
            logger.debug(f"Portal connection test failed: {error}")
            return False

    def _extract_inner_value(self, data: Dict[str, Any]) -> Any:
        """
        Extract the actual value from Portal response envelope.

        Portal wraps responses in either {"result": ...} (new format)
        or {"data": ...} (legacy format).
        """
        inner_key = "result" if "result" in data else "data" if "data" in data else None
        if inner_key is None:
            return data

        inner_value = data[inner_key]
        if isinstance(inner_value, str):
            try:
                return json.loads(inner_value)
            except json.JSONDecodeError:
                return inner_value
        return inner_value

    async def get_state(self) -> Dict[str, Any]:
        """
        Get device state (accessibility tree + phone state).

        Returns:
            Dictionary containing 'a11y_tree' and 'phone_state' keys
        """
        await self._ensure_connected()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/state_full",
                    headers=self._headers,
                    timeout=10,
                )
                if response.status_code != 200:
                    return {
                        "error": f"HTTP {response.status_code}",
                        "message": response.text,
                    }

                data = response.json()
                if isinstance(data, dict):
                    return self._extract_inner_value(data)
                return data

        except Exception as error:
            return {"error": "HTTP Error", "message": str(error)}

    async def input_text(self, text: str, clear: bool = False) -> bool:
        """
        Input text via Portal keyboard.

        Args:
            text: Text to input
            clear: Whether to clear existing text first

        Returns:
            True if successful, False otherwise
        """
        await self._ensure_connected()
        try:
            encoded = base64.b64encode(text.encode()).decode()
            payload = {"base64_text": encoded, "clear": clear}
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/keyboard/input",
                    json=payload,
                    headers=self._headers,
                    timeout=10,
                )
                if response.status_code == 200:
                    logger.debug("input_text successful")
                    return True
                logger.warning(f"input_text failed: HTTP {response.status_code}")
                return False
        except Exception as error:
            logger.error(f"input_text error: {error}")
            return False

    async def take_screenshot(self, hide_overlay: bool = True) -> bytes:
        """
        Take screenshot of device.

        Args:
            hide_overlay: Whether to hide Portal overlay during screenshot

        Returns:
            Screenshot image bytes (PNG format)
        """
        await self._ensure_connected()
        try:
            url = f"{self.base_url}/screenshot"
            if not hide_overlay:
                url += "?hideOverlay=false"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._headers,
                    timeout=10.0,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Screenshot failed: HTTP {response.status_code}"
                    )

                data = response.json()
                if data.get("status") == "success":
                    inner_key = (
                        "result"
                        if "result" in data
                        else "data" if "data" in data else None
                    )
                    if inner_key:
                        logger.debug("Screenshot taken via HTTP")
                        return base64.b64decode(data[inner_key])

                raise RuntimeError(f"Invalid screenshot response: {data}")

        except Exception as error:
            raise RuntimeError(f"take_screenshot error: {error}") from error

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get installed apps with package name and label.

        Args:
            include_system: Whether to include system apps

        Returns:
            List of dicts with 'package' and 'label' keys
        """
        await self._ensure_connected()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/packages",
                    headers=self._headers,
                    timeout=15,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"get_apps failed: HTTP {response.status_code}")

                data = response.json()
                packages_data = (
                    self._extract_inner_value(data) if isinstance(data, dict) else data
                )

            # Normalise to list
            packages_list: Optional[List] = None
            if isinstance(packages_data, list):
                packages_list = packages_data
            elif isinstance(packages_data, dict):
                if "packages" in packages_data:
                    packages_list = packages_data["packages"]

            if not packages_list:
                logger.warning("Could not extract packages list from response")
                return []

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

        except Exception as error:
            logger.error(f"get_apps error: {error}")
            raise ValueError(f"Error getting apps: {error}") from error

    async def get_version(self) -> str:
        """Get Portal app version."""
        await self._ensure_connected()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/version",
                    headers=self._headers,
                    timeout=5.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        inner = self._extract_inner_value(data)
                        if isinstance(inner, str):
                            return inner
                        return data.get("status", "unknown")
        except Exception as error:
            logger.debug(f"get_version error: {error}")

        return "unknown"

    async def ping(self) -> Dict[str, Any]:
        """
        Test Portal connection and verify state availability.

        Returns:
            Dictionary with status and connection details
        """
        await self._ensure_connected()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/ping",
                    headers=self._headers,
                    timeout=5.0,
                )
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "method": "http",
                        "message": f"HTTP {response.status_code}: {response.text}",
                    }

                try:
                    ping_response = response.json() if response.content else {}
                except json.JSONDecodeError:
                    ping_response = response.text

                result: Dict[str, Any] = {
                    "status": "success",
                    "method": "http",
                    "url": self.base_url,
                    "response": ping_response,
                }

        except Exception as error:
            return {"status": "error", "method": "http", "message": str(error)}

        # Verify state has the required keys
        try:
            state = await self.get_state()
            required = ("a11y_tree", "phone_state", "device_context")
            missing = [key for key in required if key not in state]
            if missing:
                return {
                    "status": "error",
                    "method": "http",
                    "message": f"incompatible portal — missing {', '.join(missing)}",
                }
        except Exception as error:
            return {
                "status": "error",
                "method": "http",
                "message": f"state check failed: {error}",
            }

        return result
