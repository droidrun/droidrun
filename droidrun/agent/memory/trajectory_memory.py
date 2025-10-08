"""HTTP backed trajectory memory client implementations"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import requests

from droidrun.agent.codeact.events import TaskThinkingEvent
from droidrun.agent.planner.events import PlanThinkingEvent
from droidrun.agent.droid.events import FinalizeEvent
from droidrun.agent.utils.trajectory import Trajectory
from droidrun.config import get_memory_config

from .base import (
    DisabledTrajectoryMemoryClient,
    TrajectoryManual,
    TrajectoryMemory,
)

logger = logging.getLogger("droidrun")

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 5
DEFAULT_THRESHOLD = 0.5


class _HttpTrajectoryMemoryBase(TrajectoryMemory):
    """Shared behaviour for HTTP based memory clients"""

    def __init__(
        self,
        *,
        base_url: str,
        enabled: bool,
        similarity_threshold: float,
        timeout: float,
        auth_token: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.similarity_threshold = similarity_threshold
        self.timeout = timeout
        self._auth_token = auth_token

    @property
    def _find_url(self) -> str:
        return f"{self.base_url}/find_trajectory"

    @property
    def _add_url(self) -> str:
        return f"{self.base_url}/add_trajectory"

    @property
    def _headers(self) -> Dict[str, str]:
        if self._auth_token:
            return {"Authorization": f"Bearer {self._auth_token}"}
        return {}

    def fetch_reference_manual(self, goal: str) -> Optional[str]:
        manual = self.find_trajectory(goal)
        if not manual:
            return None
        return format_manual_for_prompt(manual)

    def find_trajectory(self, goal: str) -> Optional[TrajectoryManual]:
        if not self.enabled or not goal:
            return None

        payload = {
            "goal": goal,
            "similarity_threshold": self.similarity_threshold,
        }

        try:
            response = requests.post(
                self._find_url,
                json=payload,
                timeout=self.timeout,
                headers=self._headers,
            )
            if response.status_code != 200:
                logger.debug(
                    "Trajectory lookup failed with status %s: %s",
                    response.status_code,
                    response.text,
                )
                return None
            data = response.json()
            if not data:
                logger.debug("Trajectory lookup returned no match for goal '%s'", goal)
            return data or None
        except requests.RequestException as exc:
            logger.debug("Trajectory lookup error: %s", exc)
            return None

    def add_trajectory(self, manual: TrajectoryManual) -> bool:
        if not self.enabled or not manual:
            return False

        status = manual.get("status")
        if not status:
            logger.debug("Trajectory manual missing status; skipping upload.")
            return False

        payload = dict(manual)
        payload["status"] = status.lower()

        try:
            response = requests.post(
                self._add_url,
                json=payload,
                timeout=self.timeout,
                headers=self._headers,
            )
            if response.status_code != 200:
                logger.debug(
                    "Trajectory upload failed with status %s: %s",
                    response.status_code,
                    response.text,
                )
                return False
            return True
        except requests.RequestException as exc:
            logger.debug("Trajectory upload error: %s", exc)
            return False


class LocalHttpTrajectoryMemoryClient(_HttpTrajectoryMemoryBase):
    """HTTP client for a locally hosted trajectory memory service"""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        similarity_threshold: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> None:
        resolved_base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        resolved_enabled = True if enabled is None else bool(enabled)
        resolved_threshold = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else DEFAULT_THRESHOLD
        )
        resolved_timeout = (
            float(timeout) if timeout is not None else float(DEFAULT_TIMEOUT)
        )
        super().__init__(
            base_url=resolved_base_url,
            enabled=resolved_enabled,
            similarity_threshold=resolved_threshold,
            timeout=resolved_timeout,
        )


class RemoteHttpTrajectoryMemoryClient(_HttpTrajectoryMemoryBase):
    """HTTP client for a remotely hosted trajectory memory service."""

    def __init__(
        self,
        *,
        base_url: str,
        enabled: Optional[bool] = None,
        similarity_threshold: Optional[float] = None,
        timeout: Optional[float] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        if not base_url:
            raise ValueError("RemoteHttpTrajectoryMemoryClient requires a base_url")

        resolved_base_url = base_url.rstrip("/")
        resolved_enabled = True if enabled is None else bool(enabled)
        resolved_threshold = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else DEFAULT_THRESHOLD
        )
        resolved_timeout = (
            float(timeout) if timeout is not None else float(DEFAULT_TIMEOUT)
        )

        super().__init__(
            base_url=resolved_base_url,
            enabled=resolved_enabled,
            similarity_threshold=resolved_threshold,
            timeout=resolved_timeout,
            auth_token=auth_token,
        )


class TrajectoryMemoryClient(LocalHttpTrajectoryMemoryClient):
    """Backward-compatible default client (local HTTP)"""


def _extract_section(config: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key, {})
    if isinstance(value, dict):
        return value
    logger.warning("Config section 'memory.%s' must be a table; ignoring value of type %s.", key, type(value))
    return {}


def _extract_options(source: Mapping[str, Any], mapping: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    for canonical, aliases in mapping.items():
        for alias in aliases:
            if alias in source and source[alias] is not None:
                options[canonical] = source[alias]
                break
    return options


def _split_kwargs(
    kwargs: Dict[str, Any], mapping: Dict[str, Tuple[str, ...]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    alias_lookup = {
        alias: canonical
        for canonical, aliases in mapping.items()
        for alias in aliases
    }
    overrides: Dict[str, Any] = {}
    remaining: Dict[str, Any] = {}
    for key, value in kwargs.items():
        canonical = alias_lookup.get(key)
        if canonical:
            overrides[canonical] = value
        else:
            remaining[key] = value
    return overrides, remaining


_LOCAL_KEYS: Dict[str, Tuple[str, ...]] = {
    "base_url": ("base_url", "url"),
    "enabled": ("enabled",),
    "similarity_threshold": ("similarity_threshold", "threshold"),
    "timeout": ("timeout",),
}

_REMOTE_KEYS: Dict[str, Tuple[str, ...]] = {
    **_LOCAL_KEYS,
    "auth_token": ("auth_token", "token"),
}


def create_memory_client(provider: Optional[str] = None, **kwargs: Any) -> TrajectoryMemory:
    """Factory that returns the appropriate trajectory memory client."""

    memory_config = get_memory_config()
    configured_provider = memory_config.get("provider")
    if isinstance(configured_provider, str):
        provider_choice = configured_provider
    else:
        provider_choice = None

    resolved_provider = (provider or provider_choice or "local_http").lower()

    if resolved_provider in {"local", "local_http", "local-http"}:
        section = _extract_section(memory_config, "local_http")
        config_options = _extract_options(section, _LOCAL_KEYS)
        override_options, remaining_kwargs = _split_kwargs(dict(kwargs), _LOCAL_KEYS)
        options = {**config_options, **override_options}
        return LocalHttpTrajectoryMemoryClient(**options, **remaining_kwargs)

    if resolved_provider in {"remote", "remote_http", "remote-http", "backend"}:
        section = _extract_section(memory_config, "remote_http")
        config_options = _extract_options(section, _REMOTE_KEYS)
        override_options, remaining_kwargs = _split_kwargs(dict(kwargs), _REMOTE_KEYS)
        options = {**config_options, **override_options}
        if not options.get("base_url"):
            logger.debug(
                "Remote memory provider selected but no base_url configured; disabling client.",
            )
            return DisabledTrajectoryMemoryClient()
        return RemoteHttpTrajectoryMemoryClient(**options, **remaining_kwargs)

    if resolved_provider in {"disabled", "none", "off"}:
        return DisabledTrajectoryMemoryClient()

    logger.debug(
        "Unknown trajectory memory provider '%s'. Falling back to disabled client.",
        resolved_provider,
    )
    return DisabledTrajectoryMemoryClient()


def parse_trajectory_for_manual(trajectory: Trajectory) -> TrajectoryManual:
    manual: TrajectoryManual = {
        "goal": trajectory.goal,
        "initial_plan": {},
        "execution_steps": [],
    }

    step_counter = 1
    status = "unknown"
    for event in trajectory.events:
        if isinstance(event, PlanThinkingEvent) and event.code:
            manual.setdefault("initial_plan", {})
            manual["initial_plan"]["reasoning"] = event.thoughts
            manual["initial_plan"]["plan_code"] = event.code
        if isinstance(event, TaskThinkingEvent) and event.code:
            manual.setdefault("execution_steps", [])
            manual["execution_steps"].append(
                {
                    "step_index": step_counter,
                    "reasoning": event.thoughts,
                    "action_code": event.code,
                }
            )
            step_counter += 1
        if isinstance(event, FinalizeEvent):
            status = "success" if getattr(event, "success", False) else "failure"

    manual["status"] = status

    return manual


def format_manual_for_prompt(manual: TrajectoryManual) -> str:
    status = (manual.get("status") or "unknown").lower()
    if status == "success":
        heading = "**Successful Trajectory Reference**"
    elif status == "failure":
        heading = "**Failed Trajectory Reference (learn from mistakes)**"
    else:
        heading = "**Trajectory Reference**"

    lines = [
        heading,
        f"**Goal:** {manual.get('goal', 'N/A')}",
        f"**Initial Plan:** {manual.get('initial_plan', {}).get('reasoning', 'N/A')}",
        "---",
        "**Execution Steps:**",
    ]

    for step in manual.get("execution_steps", []):
        lines.append(f"**Step {step.get('step_index', '?')}:**")
        lines.append(f"*   **Reasoning:** {step.get('reasoning', 'N/A')}")
        action = step.get("action_code", "")
        lines.append(f"*   **Action:** `{action}`")

    return "\n".join(lines)


__all__ = [
    "LocalHttpTrajectoryMemoryClient",
    "RemoteHttpTrajectoryMemoryClient",
    "TrajectoryMemoryClient",
    "create_memory_client",
    "parse_trajectory_for_manual",
    "format_manual_for_prompt",
]
