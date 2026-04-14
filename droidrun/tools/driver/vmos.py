"""VMOSDriver — VMOS Cloud device driver.

Controls VMOS Cloud phones via their HTTP API with HMAC-SHA256 authentication.
Uses ADB shell commands (via asyncCmd) for input actions instead of the
simulateClick/simulateSwipe endpoints, which are unreliable on VMOS phones.
"""

from __future__ import annotations

import asyncio
import binascii
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from droidrun.tools.driver.base import DeviceDisconnectedError, DeviceDriver

logger = logging.getLogger("droidrun")

_HOST = "api.vmoscloud.com"
_SERVICE = "armcloud-paas"


class VMOSDriver(DeviceDriver):
    """VMOS Cloud device I/O via their HTTP API."""

    platform = "Android"

    supported = {
        "tap",
        "swipe",
        "input_text",
        "press_button",
        "start_app",
        "screenshot",
        "get_date",
        "get_apps",
        "list_packages",
    }

    supported_buttons = {"back", "home", "enter"}

    _BUTTON_KEYCODES = {
        "back": 4,
        "home": 3,
        "enter": 66,
    }

    def __init__(
        self,
        pad_code: str,
        access_key: str,
        secret_key: str,
        base_url: str = "https://api.vmoscloud.com",
        task_poll_interval: float = 2.0,
        task_timeout: float = 60.0,
    ) -> None:
        self.pad_code = pad_code
        self._ak = access_key
        self._sk = secret_key
        self._base_url = base_url.rstrip("/")
        self._poll_interval = task_poll_interval
        self._task_timeout = task_timeout
        self._client = httpx.AsyncClient(timeout=30.0)

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        pass  # stateless HTTP — no persistent connection needed

    async def ensure_connected(self) -> None:
        pass

    # -- HMAC signing --------------------------------------------------------

    def _sign(self, body_json: str, x_date: str) -> str:
        """Compute HMAC-SHA256 signature for a VMOS API request."""
        content_type = "application/json;charset=UTF-8"
        signed_headers = "content-type;host;x-content-sha256;x-date"

        x_content_sha256 = hashlib.sha256(body_json.encode()).hexdigest()

        canonical = (
            f"host:{_HOST}\n"
            f"x-date:{x_date}\n"
            f"content-type:{content_type}\n"
            f"signedHeaders:{signed_headers}\n"
            f"x-content-sha256:{x_content_sha256}"
        )

        short_date = x_date[:8]
        credential_scope = f"{short_date}/{_SERVICE}/request"
        string_to_sign = (
            f"HMAC-SHA256\n"
            f"{x_date}\n"
            f"{credential_scope}\n"
            f"{hashlib.sha256(canonical.encode()).hexdigest()}"
        )

        # Derive signing key: SK -> date -> service -> "request"
        k1 = hmac.new(self._sk.encode(), short_date.encode(), hashlib.sha256).digest()
        k2 = hmac.new(k1, _SERVICE.encode(), hashlib.sha256).digest()
        signing_key = hmac.new(k2, b"request", hashlib.sha256).digest()

        return binascii.hexlify(
            hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).digest()
        ).decode()

    # -- HTTP transport ------------------------------------------------------

    async def _request(self, path: str, body: dict | None = None) -> Any:
        """POST to a VMOS API endpoint with HMAC signing."""
        body = body or {}
        body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        x_date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        signature = self._sign(body_json, x_date)

        headers = {
            "content-type": "application/json;charset=UTF-8",
            "x-date": x_date,
            "x-host": _HOST,
            "authorization": (
                f"HMAC-SHA256 Credential={self._ak}, "
                f"SignedHeaders=content-type;host;x-content-sha256;x-date, "
                f"Signature={signature}"
            ),
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}{path}",
                headers=headers,
                content=body_json,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise DeviceDisconnectedError(f"VMOS API request failed: {exc}") from exc

        result = resp.json()
        if result.get("code") != 200:
            raise DeviceDisconnectedError(
                f"VMOS API error: {result.get('msg', 'unknown')} (code={result.get('code')})"
            )

        return result.get("data", result)

    # -- ADB command helpers -------------------------------------------------

    async def _run_adb(self, cmd: str) -> str:
        """Run an ADB shell command via VMOS asyncCmd and return the result."""
        data = await self._request(
            "/vcpcloud/api/padApi/asyncCmd",
            {"padCodes": [self.pad_code], "scriptContent": cmd},
        )

        task_id: int | None = None
        if isinstance(data, list) and data:
            task_id = data[0].get("taskId") if isinstance(data[0], dict) else None
        elif isinstance(data, dict):
            task_id = data.get("taskId")

        if task_id is None:
            logger.warning("No taskId returned from asyncCmd for: %s", cmd)
            return ""

        return await self._wait_task(task_id)

    async def _wait_task(self, task_id: int) -> str:
        """Poll a VMOS async task until it completes, return its result string."""
        deadline = asyncio.get_running_loop().time() + self._task_timeout

        while asyncio.get_running_loop().time() < deadline:
            data = await self._request(
                "/vcpcloud/api/padApi/padTaskDetail",
                {"taskIds": [task_id]},
            )

            task: dict | None = None
            if isinstance(data, list) and data:
                task = data[0] if isinstance(data[0], dict) else None
            elif isinstance(data, dict):
                task = data

            if task:
                status = task.get("status")
                # status 0/1 = pending/running, 2 = success, 3+ = done/error
                if status not in (None, 0, 1, "pending", "running"):
                    return task.get("result", "")

            await asyncio.sleep(self._poll_interval)

        raise DeviceDisconnectedError(
            f"VMOS task {task_id} did not complete within {self._task_timeout}s"
        )

    # -- input actions -------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        await self._run_adb(f"input tap {x} {y}")

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        await self._run_adb(
            f"input swipe {x1} {y1} {x2} {y2} {int(duration_ms)}"
        )

    async def input_text(
        self,
        text: str,
        clear: bool = False,
        stealth: bool = False,
        wpm: int = 0,
    ) -> bool:
        if clear:
            # Move cursor to end, then delete 160 chars to clear the field
            await self._run_adb("input keyevent 123")  # KEYCODE_MOVE_END
            del_cmds = ";".join(["input keyevent 67"] * 160)  # KEYCODE_DEL
            await self._run_adb(del_cmds)

        await self._request(
            "/vcpcloud/api/padApi/inputText",
            {"padCodes": [self.pad_code], "text": text},
        )
        return True

    async def press_button(self, button: str) -> None:
        button_lower = button.lower()
        if button_lower not in self.supported_buttons:
            raise ValueError(
                f"Button '{button}' not supported. "
                f"Supported: {', '.join(sorted(self.supported_buttons))}"
            )
        keycode = self._BUTTON_KEYCODES[button_lower]
        await self._run_adb(f"input keyevent {keycode}")

    # -- app management ------------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        await self._request(
            "/vcpcloud/api/padApi/startApp",
            {"padCodes": [self.pad_code], "pkgName": package},
        )
        return f"App started: {package}"

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        data = await self._request(
            "/vcpcloud/api/padApi/listInstalledApp",
            {"padCodes": [self.pad_code]},
        )
        apps: list[dict[str, str]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    apps.append({
                        "package": item.get("packageName", item.get("pkgName", "")),
                        "label": item.get("appName", item.get("label", "")),
                    })
        return apps

    async def list_packages(self, include_system: bool = False) -> List[str]:
        cmd = "pm list packages" if include_system else "pm list packages -3"
        result = await self._run_adb(cmd)
        return [
            line.removeprefix("package:")
            for line in result.splitlines()
            if line.startswith("package:")
        ]

    # -- state / observation -------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        data = await self._request(
            "/vcpcloud/api/padApi/screenshot",
            {"padCodes": [self.pad_code], "definition": 80, "rotation": 0},
        )

        # VMOS returns either a list of dicts or a single dict depending on version
        url: str | None = None
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    url = item.get("accessUrl") or item.get("url")
                    if url:
                        break
        elif isinstance(data, dict):
            url = data.get("accessUrl") or data.get("url")

        if not url:
            raise DeviceDisconnectedError("VMOS screenshot API returned no image URL")

        try:
            resp = await self._client.get(url, timeout=15.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise DeviceDisconnectedError(
                f"Failed to download VMOS screenshot: {exc}"
            ) from exc

        if len(resp.content) < 1000:
            raise DeviceDisconnectedError(
                "VMOS screenshot download returned suspiciously small data"
            )

        return resp.content

    async def get_date(self) -> str:
        result = await self._run_adb("date")
        return result.strip()
