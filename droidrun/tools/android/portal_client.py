"""
Portal Client - Unified communication layer for DroidRun Portal app.

Design contract
---------------
The client is always in exactly one of three TCP states, managed exclusively
through the three state-transition helpers (_set_state_ready, _set_state_cp_only,
_set_state_degraded). Each helper owns its invariant: session lifetime, counter
resets, and logging are all co-located with the state change.

    TCP_READY   — persistent session alive, auth valid.  tcp_available == True.
    CP_ONLY     — TCP never attempted or permanently failed.
                  tcp_available == False, _session is None.
    DEGRADED    — was TCP_READY; hit a 401 that survived a forced token refresh.
                  tcp_available == False, _session is None.
                  Re-probe attempted every REPROBE_INTERVAL transport calls.

Concurrency model
-----------------
_session_lock serialises *session mutation only* — it is never held across
slow I/O (ADB shell calls). This eliminates the reentrant-lock deadlock that
would occur if _fetch_auth_token tried to re-acquire the lock while a caller
already held it.

Token TTL (TOKEN_TTL_SECONDS) is a *performance hint*, not a correctness
guarantee. A cached token is still considered valid until the server rejects
it with HTTP 401. The TTL merely reduces the frequency of proactive ADB
shell calls.
"""

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import httpx
from async_adbutils import AdbDevice

logger = logging.getLogger("droidrun")

PORTAL_REMOTE_PORT = 8080
REPROBE_INTERVAL = 10
TOKEN_TTL_SECONDS = 300


# ──────────────────────────────────────────────────────────────────────────────
# Supporting types
# ──────────────────────────────────────────────────────────────────────────────


class _TcpState(Enum):
    READY = auto()
    CP_ONLY = auto()
    DEGRADED = auto()


@dataclass(frozen=True)
class PortalTimeouts:
    """
    All timeouts in one place.

    Pass a custom instance to override defaults:
        PortalClient(device, timeouts=PortalTimeouts(request=20.0))
    """

    ping: float = 3.0
    auth_test: float = 5.0
    request: float = 10.0
    screenshot: float = 15.0
    adb_shell: float = 15.0


@dataclass
class TransportMetrics:
    """
    Per-session transport counters.

    Exposed as client.metrics for agent benchmarking and debugging.
    All increments happen at the call-site without a lock — CPython's
    GIL makes integer increments atomic, and the values are statistical
    counters rather than correctness-critical state.
    """

    tcp_requests: int = 0
    tcp_successes: int = 0
    fallback_count: int = 0
    auth_refresh_count: int = 0
    reprobe_attempts: int = 0
    reprobe_successes: int = 0

    def summary(self) -> Dict[str, Any]:
        total = self.tcp_requests or 1
        return {
            "tcp_requests": self.tcp_requests,
            "tcp_success_rate": round(self.tcp_successes / total, 3),
            "fallback_count": self.fallback_count,
            "auth_refresh_count": self.auth_refresh_count,
            "reprobe_attempts": self.reprobe_attempts,
            "reprobe_successes": self.reprobe_successes,
        }


# ──────────────────────────────────────────────────────────────────────────────
# PortalClient
# ──────────────────────────────────────────────────────────────────────────────


class PortalClient:
    """
    Unified client for DroidRun Portal communication.

    TCP mode: persistent httpx.AsyncClient, Bearer auth, ~3-5x lower latency.
    Content provider fallback: works without port forwarding, higher latency.

    Usage::

        client = PortalClient(device, prefer_tcp=True)
        await client.connect()
        state = await client.get_state()
        print(client.metrics.summary())
        await client.disconnect()
    """

    def __init__(
        self,
        device: AdbDevice,
        prefer_tcp: bool = False,
        timeouts: Optional[PortalTimeouts] = None,
    ):
        self.device = device
        self.prefer_tcp = prefer_tcp
        self.timeouts = timeouts or PortalTimeouts()
        self.metrics = TransportMetrics()

        # Internal state
        self._tcp_state: _TcpState = _TcpState.CP_ONLY
        self._connected: bool = False
        self._session: Optional[httpx.AsyncClient] = None

        # Auth — TTL is a fetch-frequency hint only; 401 always triggers refresh
        self._auth_token: Optional[str] = None
        self._token_fetched_at: float = 0.0

        # Lock guards session object mutation only — never held across ADB I/O
        self._session_lock = asyncio.Lock()

        # DEGRADED re-probe counter (read outside lock, write inside lock)
        self._degraded_call_count: int = 0

        # Degradation log gate — reset when client recovers to READY
        self._degradation_logged: bool = False

        # TCP coordinates
        self.tcp_base_url: Optional[str] = None
        self.local_tcp_port: Optional[int] = None

    # ── public derived property ────────────────────────────────────────────────

    @property
    def tcp_available(self) -> bool:
        return self._tcp_state == _TcpState.READY

    # ──────────────────────────────────────────────────────────────────────────
    # State-transition helpers  (Issue 1 fix)
    # Every transition lives here. Nothing else mutates _tcp_state directly.
    # ──────────────────────────────────────────────────────────────────────────

    async def _set_state_ready(self) -> None:
        """
        Transition to TCP_READY.

        Assumes the session has already been built by the caller.
        Resets DEGRADED counters and clears the degradation log gate so
        the next degradation event produces a fresh INFO log.
        """
        self._tcp_state = _TcpState.READY
        self._degraded_call_count = 0
        self._degradation_logged = False
        logger.debug(f"TCP state -> READY ({self.tcp_base_url})")

    async def _set_state_cp_only(self, reason: str) -> None:
        """
        Transition to CP_ONLY (permanent, no re-probe).

        Closes the session and clears auth state. Logged at WARNING.
        """
        async with self._session_lock:
            await self._close_session_unsafe()
        self._auth_token = None
        self._token_fetched_at = 0.0
        self._tcp_state = _TcpState.CP_ONLY
        logger.warning(f"TCP state -> CP_ONLY: {reason}")

    async def _set_state_degraded(self, reason: str) -> None:
        """
        Transition to DEGRADED (re-probe eligible).

        Closes the session but preserves auth token for re-probe attempt.
        Emits an INFO log exactly once per degradation episode.
        """
        async with self._session_lock:
            await self._close_session_unsafe()
        self._tcp_state = _TcpState.DEGRADED
        if not self._degradation_logged:
            logger.info(
                f"Portal TCP degraded ({reason}) -> content provider fallback. "
                f"Re-probing every {REPROBE_INTERVAL} requests."
            )
            self._degradation_logged = True

    # ──────────────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Prepare the client for use.

        Transport negotiation (_try_enable_tcp) is a distinct sub-concern.
        connect() is the *lifecycle* operation: it marks the client ready
        for callers and enforces post-condition invariants regardless of
        which transport was chosen.

        Post-conditions:
            TCP_READY  — prefer_tcp=True and all setup succeeded.
            CP_ONLY    — prefer_tcp=False, or TCP setup failed entirely.
        """
        if self._connected:
            return

        if self.prefer_tcp:
            await self._try_enable_tcp()

        # Enforce invariant: non-READY states must have no dangling session
        if self._tcp_state != _TcpState.READY and self._session is not None:
            async with self._session_lock:
                await self._close_session_unsafe()

        self._connected = True

    async def disconnect(self) -> None:
        """Release resources. Safe to call multiple times."""
        await self._set_state_cp_only("disconnect() called")
        self._connected = False

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    # ──────────────────────────────────────────────────────────────────────────
    # TCP setup
    # ──────────────────────────────────────────────────────────────────────────

    async def _try_enable_tcp(self) -> None:
        """
        Attempt full TCP setup. On any failure stays CP_ONLY.

        Steps:
        1. Find or create ADB port forward.
        2. Build unauthenticated session; ping to confirm Portal is alive.
        3. Optionally start Portal HTTP server if ping fails.
        4. Fetch auth token; rebuild session with Authorization header.
        5. Verify auth via /version (authenticated endpoint, not /ping).
        6. Transition to READY.
        """
        try:
            local_port = await self._find_existing_forward()
            if local_port is None:
                logger.debug(f"Creating port forward for {PORTAL_REMOTE_PORT}")
                local_port = await self.device.forward_port(PORTAL_REMOTE_PORT)
            else:
                logger.debug(f"Reusing existing forward on localhost:{local_port}")

            self.local_tcp_port = local_port
            self.tcp_base_url = f"http://localhost:{local_port}"

            # Build initial session (no token yet) for the unauthenticated ping
            async with self._session_lock:
                await self._rebuild_session_unsafe()

            if not await self._ping_portal():
                logger.debug("Ping failed, attempting to start Portal HTTP server...")
                await self.device.shell(
                    "content insert --uri content://com.droidrun.portal/toggle_socket_server"
                    " --bind enabled:b:true"
                )
                await asyncio.sleep(1)
                if not await self._ping_portal():
                    await self._set_state_cp_only(
                        "Portal not reachable after server start"
                    )
                    return

            # Fetch token (pure ADB, no lock) then rebuild session with it
            token = await self._request_token_from_device()
            if token:
                self._auth_token = token
                self._token_fetched_at = time.monotonic()
                async with self._session_lock:
                    await self._rebuild_session_unsafe()

            # Verify with authenticated endpoint
            if await self._test_authenticated_connection():
                await self._set_state_ready()
            else:
                await self._set_state_cp_only("Auth verification failed on /version")

        except Exception as e:
            await self._set_state_cp_only(f"TCP setup exception: {e}")

    async def _find_existing_forward(self) -> Optional[int]:
        try:
            async for fwd in self.device.forward_list():
                if (
                    fwd.serial == self.device.serial
                    and fwd.remote == f"tcp:{PORTAL_REMOTE_PORT}"
                ):
                    match = re.search(r"tcp:(\d+)", fwd.local)
                    if match:
                        return int(match.group(1))
        except Exception as e:
            logger.debug(f"Failed to list forwards: {e}")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Session management (lock-aware)
    # ──────────────────────────────────────────────────────────────────────────

    async def _rebuild_session_unsafe(self) -> None:
        """
        Replace the persistent AsyncClient.

        MUST be called while holding _session_lock.
        One AsyncClient per session lifetime — eliminates per-call TCP handshake
        overhead across 20-100 agent loop requests.
        """
        await self._close_session_unsafe()
        headers: Dict[str, str] = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        self._session = httpx.AsyncClient(
            headers=headers,
            timeout=self.timeouts.request,
        )

    async def _close_session_unsafe(self) -> None:
        """Close _session. MUST hold _session_lock."""
        if self._session is not None:
            try:
                await self._session.aclose()
            except Exception:
                pass
            self._session = None

    # ──────────────────────────────────────────────────────────────────────────
    # Auth management  (Deadlock fix)
    # ──────────────────────────────────────────────────────────────────────────

    def _token_is_fresh(self) -> bool:
        """
        Returns True if the cached token is younger than TOKEN_TTL_SECONDS.

        This is a *performance hint* only — it reduces proactive ADB shell
        calls. HTTP 401 always takes precedence as ground truth regardless
        of this value.
        """
        return (
            self._auth_token is not None
            and (time.monotonic() - self._token_fetched_at) < TOKEN_TTL_SECONDS
        )

    async def _request_token_from_device(self) -> Optional[str]:
        """
        Fetch the raw Bearer token string via ADB content provider.

        PURE I/O — no lock acquired, no session mutation. Designed to be
        called *before* acquiring _session_lock so there is no reentrant
        deadlock risk.

        Returns:
            Token string, or None if the Portal APK has no auth endpoint
            or the fetch failed.
        """
        try:
            output = await self.device.shell(
                "content query --uri content://com.droidrun.portal/auth_token"
            )
            if "result=" not in output:
                logger.debug("No auth_token endpoint on this Portal version")
                return None
            json_str = output.split("result=", 1)[1].strip()
            data = json.loads(json_str)
            token = data.get("result") if isinstance(data, dict) else None
            if not token:
                logger.debug("auth_token endpoint returned empty token")
            return token or None
        except Exception as e:
            logger.debug(f"Token fetch failed: {e}")
            return None

    async def _refresh_token_and_rebuild(self, force: bool = False) -> bool:
        """
        Fetch a fresh token (ADB, no lock) then rebuild the session (locked).

        This is the sole path that combines a token fetch with a session rebuild.
        Separating the I/O phase from the mutation phase is what prevents the
        asyncio.Lock reentrant deadlock.

        Args:
            force: Skip TTL check and always fetch from device.

        Returns:
            True if a token is available (fresh cached or newly fetched)
            and the session was rebuilt. False if no token could be obtained.
        """
        if not force and self._token_is_fresh():
            logger.debug("Token fresh — skipping device refetch")
            return True

        token = await self._request_token_from_device()
        if not token:
            return False

        self.metrics.auth_refresh_count += 1
        self._auth_token = token
        self._token_fetched_at = time.monotonic()

        async with self._session_lock:
            await self._rebuild_session_unsafe()

        logger.debug("Auth token refreshed, session rebuilt")
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Connection tests
    # ──────────────────────────────────────────────────────────────────────────

    async def _ping_portal(self) -> bool:
        """Unauthenticated liveness probe. Used only during connect()."""
        if not self._session:
            return False
        try:
            r = await self._session.get(
                f"{self.tcp_base_url}/ping", timeout=self.timeouts.ping
            )
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Portal ping failed: {e}")
            return False

    async def _test_authenticated_connection(self) -> bool:
        """
        Probe /version (requires auth) to confirm both reachability and auth.

        /ping is unauthenticated — a 200 there only proves the port is open.
        This call proves the token works, which is the real precondition for
        setting tcp_available = True.
        """
        if not self._session:
            return False
        try:
            r = await self._session.get(
                f"{self.tcp_base_url}/version", timeout=self.timeouts.auth_test
            )
            if r.status_code == 200:
                return True
            logger.debug(f"Auth test returned HTTP {r.status_code}")
            return False
        except Exception as e:
            logger.debug(f"Auth test failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # DEGRADED re-probe  (Issue 2 fix — fast path outside lock)
    # ──────────────────────────────────────────────────────────────────────────

    async def _maybe_reprobe(self) -> None:
        """
        Attempt TCP recovery from DEGRADED state every REPROBE_INTERVAL calls.

        Counter increment is outside the lock (statistical, not correctness-
        critical). State mutation is inside the lock via the state helpers.
        """
        # Fast path — no lock needed for the read or the increment
        self._degraded_call_count += 1
        if self._degraded_call_count % REPROBE_INTERVAL != 0:
            return

        self.metrics.reprobe_attempts += 1
        logger.debug(
            f"DEGRADED re-probe #{self._degraded_call_count // REPROBE_INTERVAL}"
        )

        refreshed = await self._refresh_token_and_rebuild(force=True)
        if not refreshed:
            logger.debug("Re-probe: token fetch failed")
            return

        if await self._test_authenticated_connection():
            self.metrics.reprobe_successes += 1
            await self._set_state_ready()
            logger.info("Portal TCP recovered from DEGRADED -> READY")
        else:
            logger.debug("Re-probe: auth test failed — staying DEGRADED")

    # ──────────────────────────────────────────────────────────────────────────
    # Core transport helpers  (Issue 4 fix — DRY gating)
    # ──────────────────────────────────────────────────────────────────────────

    async def _ensure_tcp_ready(self) -> bool:
        """
        Single gating check shared by _tcp_get and _tcp_post.

        Handles DEGRADED re-probe before returning the verdict.

        Returns:
            True if TCP is READY and _session is not None.
        """
        if self._tcp_state == _TcpState.DEGRADED:
            await self._maybe_reprobe()
        return self._tcp_state == _TcpState.READY and self._session is not None

    async def _tcp_get(self, path: str, **kwargs) -> Optional[httpx.Response]:
        """
        GET via persistent session with 401 self-healing.

        Returns Response on success, None when TCP is unavailable/unrecoverable.
        Callers always fall back to content provider on None.
        """
        if not await self._ensure_tcp_ready():
            return None

        self.metrics.tcp_requests += 1
        url = f"{self.tcp_base_url}{path}"
        try:
            response = await self._session.get(url, **kwargs)
            if response.status_code == 401:
                return await self._handle_401_and_retry("GET", url, kwargs, is_get=True)
            self.metrics.tcp_successes += 1
            return response
        except Exception as e:
            logger.debug(f"TCP GET {path} error: {e}")
            return None

    async def _tcp_post(self, path: str, **kwargs) -> Optional[httpx.Response]:
        """
        POST via persistent session with 401 self-healing.

        Same semantics as _tcp_get.
        """
        if not await self._ensure_tcp_ready():
            return None

        self.metrics.tcp_requests += 1
        url = f"{self.tcp_base_url}{path}"
        try:
            response = await self._session.post(url, **kwargs)
            if response.status_code == 401:
                return await self._handle_401_and_retry(
                    "POST", url, kwargs, is_get=False
                )
            self.metrics.tcp_successes += 1
            return response
        except Exception as e:
            logger.debug(f"TCP POST {path} error: {e}")
            return None

    async def _handle_401_and_retry(
        self,
        method: str,
        url: str,
        kwargs: dict,
        is_get: bool,
    ) -> Optional[httpx.Response]:
        """
        401-recovery path shared by _tcp_get and _tcp_post.

        Protocol (deadlock-free):
        1. _request_token_from_device() — pure ADB I/O, NO lock held.
        2. Acquire _session_lock, update token, rebuild session, release lock.
        3. Retry once with the new session.
        4. Second 401 -> DEGRADED.

        The lock is never held during ADB I/O. This prevents the reentrant
        deadlock that would occur if _fetch_auth_token (old design) tried to
        re-acquire a lock already held by this method.
        """
        logger.debug(f"401 on {method} {url} — refreshing auth token")

        # Phase 1: I/O (no lock)
        token = await self._request_token_from_device()
        if not token:
            self.metrics.fallback_count += 1
            await self._set_state_degraded(
                f"{method} {url} — token refresh returned nothing"
            )
            return None

        # Phase 2: Mutation (locked)
        self.metrics.auth_refresh_count += 1
        self._auth_token = token
        self._token_fetched_at = time.monotonic()
        async with self._session_lock:
            await self._rebuild_session_unsafe()

        if self._session is None:
            return None

        # Phase 3: Retry (lock released)
        try:
            response = (
                await self._session.get(url, **kwargs)
                if is_get
                else await self._session.post(url, **kwargs)
            )
            if response.status_code != 401:
                self.metrics.tcp_successes += 1
                return response

            self.metrics.fallback_count += 1
            await self._set_state_degraded(
                f"{method} {url} — refreshed token still rejected"
            )
            return None

        except Exception as e:
            logger.debug(f"Retry {method} {url} error: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Content provider helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_content_provider_output(self, raw: str) -> Optional[Any]:
        for line in raw.strip().split("\n"):
            line = line.strip()
            if "result=" in line:
                json_str = line[line.find("result=") + 7 :]
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        inner_key = (
                            "result"
                            if "result" in data
                            else "data" if "data" in data else None
                        )
                        if inner_key:
                            v = data[inner_key]
                            try:
                                return json.loads(v) if isinstance(v, str) else v
                            except json.JSONDecodeError:
                                return v
                    return data
                except json.JSONDecodeError:
                    continue
            elif line.startswith(("{", "[")):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _unwrap(data: Any) -> Any:
        """Strip the Portal response envelope (result/data key)."""
        if not isinstance(data, dict):
            return data
        key = "result" if "result" in data else "data" if "data" in data else None
        if not key:
            return data
        v = data[key]
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

    # ──────────────────────────────────────────────────────────────────────────
    # Public device API
    # ──────────────────────────────────────────────────────────────────────────

    async def get_state(self) -> Dict[str, Any]:
        """Get device state (accessibility tree + phone state)."""
        await self._ensure_connected()
        if self.tcp_available:
            return await self._get_state_tcp()
        return await self._get_state_content_provider()

    async def _get_state_tcp(self) -> Dict[str, Any]:
        r = await self._tcp_get("/state_full", timeout=self.timeouts.request)
        if r is not None and r.status_code == 200:
            unwrapped = self._unwrap(r.json())
            if isinstance(unwrapped, dict):
                return unwrapped
        self.metrics.fallback_count += 1
        return await self._get_state_content_provider()

    async def _get_state_content_provider(self) -> Dict[str, Any]:
        try:
            output = await self.device.shell(
                "content query --uri content://com.droidrun.portal/state_full"
            )
            data = self._parse_content_provider_output(output)
            if data is None:
                return {
                    "error": "Parse Error",
                    "message": "Failed to parse state from ContentProvider",
                }
            unwrapped = self._unwrap(data) if isinstance(data, dict) else data
            if isinstance(unwrapped, dict):
                return unwrapped
            return {"error": "Parse Error", "message": "Unexpected state format"}
        except Exception as e:
            return {"error": "ContentProvider Error", "message": str(e)}

    async def input_text(self, text: str, clear: bool = False) -> bool:
        """Input text via keyboard."""
        await self._ensure_connected()
        if self.tcp_available:
            return await self._input_text_tcp(text, clear)
        return await self._input_text_content_provider(text, clear)

    async def _input_text_tcp(self, text: str, clear: bool) -> bool:
        encoded = base64.b64encode(text.encode()).decode()
        r = await self._tcp_post(
            "/keyboard/input",
            json={"base64_text": encoded, "clear": clear},
            timeout=self.timeouts.request,
        )
        if r is not None and r.status_code == 200:
            return True
        self.metrics.fallback_count += 1
        return await self._input_text_content_provider(text, clear)

    async def _input_text_content_provider(self, text: str, clear: bool) -> bool:
        try:
            encoded = base64.b64encode(text.encode()).decode()
            cmd = (
                f'content insert --uri "content://com.droidrun.portal/keyboard/input" '
                f'--bind base64_text:s:"{encoded}" '
                f"--bind clear:b:{'true' if clear else 'false'}"
            )
            await self.device.shell(cmd)
            return True
        except Exception as e:
            logger.error(f"Content provider input_text error: {e}")
            return False

    async def take_screenshot(self, hide_overlay: bool = True) -> bytes:
        """Take screenshot of device."""
        await self._ensure_connected()
        if self.tcp_available:
            return await self._take_screenshot_tcp(hide_overlay)
        return await self._take_screenshot_adb()

    async def _take_screenshot_tcp(self, hide_overlay: bool) -> bytes:
        path = "/screenshot" + ("" if hide_overlay else "?hideOverlay=false")
        r = await self._tcp_get(path, timeout=self.timeouts.screenshot)
        if r is not None and r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                key = (
                    "result" if "result" in data else "data" if "data" in data else None
                )
                if key:
                    return base64.b64decode(data[key])
        self.metrics.fallback_count += 1
        return await self._take_screenshot_adb()

    async def _take_screenshot_adb(self) -> bytes:
        return await self.device.screenshot_bytes()

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get installed apps. Content provider only — no TCP endpoint exists yet."""
        await self._ensure_connected()
        try:
            output = await self.device.shell(
                "content query --uri content://com.droidrun.portal/packages"
            )
            data = self._parse_content_provider_output(output)
            if not data:
                logger.warning("No packages data from content provider")
                return []

            packages_list = None
            if isinstance(data, list):
                packages_list = data
            elif isinstance(data, dict):
                packages_list = data.get("packages")
                if packages_list is None:
                    inner = self._unwrap(data)
                    packages_list = (
                        inner
                        if isinstance(inner, list)
                        else (
                            inner.get("packages") if isinstance(inner, dict) else None
                        )
                    )

            if not packages_list:
                logger.warning("Could not extract packages list")
                return []

            return [
                {"package": p.get("packageName", ""), "label": p.get("label", "")}
                for p in packages_list
                if include_system or not p.get("isSystemApp", False)
            ]
        except Exception as e:
            logger.error(f"Error getting apps: {e}")
            raise ValueError(f"Error getting apps: {e}") from e

    async def get_version(self) -> str:
        """Get Portal app version."""
        await self._ensure_connected()
        if self.tcp_available:
            r = await self._tcp_get("/version", timeout=self.timeouts.auth_test)
            if r is not None and r.status_code == 200:
                v = self._unwrap(r.json())
                if isinstance(v, str):
                    return v

        try:
            output = await self.device.shell(
                "content query --uri content://com.droidrun.portal/version"
            )
            result = self._parse_content_provider_output(output)
            if result:
                v = self._unwrap(result)
                if isinstance(v, str):
                    return v
        except Exception:
            pass

        return "unknown"

    async def ping(self) -> Dict[str, Any]:
        """Test Portal connection and verify state availability."""
        await self._ensure_connected()
        result: Dict[str, Any] = {}

        if self.tcp_available:
            r = await self._tcp_get("/ping", timeout=self.timeouts.ping)
            if r is not None and r.status_code == 200:
                try:
                    body = r.json() if r.content else {}
                except json.JSONDecodeError:
                    body = r.text
                result = {
                    "status": "success",
                    "method": "tcp",
                    "url": self.tcp_base_url,
                    "tcp_state": self._tcp_state.name,
                    "metrics": self.metrics.summary(),
                    "response": body,
                }
            else:
                return {
                    "status": "error",
                    "method": "tcp",
                    "message": f"HTTP {r.status_code}" if r else "TCP request failed",
                }
        else:
            try:
                output = await self.device.shell(
                    "content query --uri content://com.droidrun.portal/state"
                )
                if "Row: 0 result=" in output:
                    result = {
                        "status": "success",
                        "method": "content_provider",
                        "tcp_state": self._tcp_state.name,
                        "metrics": self.metrics.summary(),
                    }
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

        try:
            state = await self.get_state()
            missing = [
                k
                for k in ("a11y_tree", "phone_state", "device_context")
                if k not in state
            ]
            if missing:
                return {
                    "status": "error",
                    "method": result.get("method"),
                    "message": f"incompatible portal — missing {', '.join(missing)}",
                }
        except Exception as e:
            return {
                "status": "error",
                "method": result.get("method"),
                "message": f"state check failed: {e}",
            }

        return result
