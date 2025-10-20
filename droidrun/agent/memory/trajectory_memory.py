"""HTTP-backed trajectory memory client implementations."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

from droidrun.agent.codeact.events import TaskThinkingEvent
from droidrun.agent.droid.events import FinalizeEvent
from droidrun.agent.executor.events import (
    ExecutorInternalActionEvent,
    ExecutorInternalResultEvent,
)
from droidrun.agent.manager.events import ManagerInternalPlanEvent
from droidrun.agent.utils.trajectory import Trajectory
from droidrun.config_manager import ConfigManager, DroidRunConfig, MemoryConfig

from .base import (
    DisabledTrajectoryMemoryClient,
    TrajectoryManual,
    TrajectoryMemory,
)

logger = logging.getLogger("droidrun")

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 5
DEFAULT_THRESHOLD = 0.5 # better to increase


class _HttpTrajectoryMemoryBase(TrajectoryMemory):
    """Shared behaviour for HTTP based memory clients."""

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

        status = (manual.get("status") or "").strip()
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
    """HTTP client for a locally hosted trajectory memory service."""

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
    """Backward-compatible default client (local HTTP)."""


def _normalize_memory_config(source: Any) -> MemoryConfig:
    """Convert various config shapes into a MemoryConfig instance."""

    if isinstance(source, MemoryConfig):
        return source

    if isinstance(source, dict):
        return MemoryConfig(**source)

    attrs: Dict[str, Any] = {}
    for key in ("enabled", "mode", "base_url", "similarity_threshold", "timeout", "auth_token"):
        if hasattr(source, key):
            attrs[key] = getattr(source, key)

    if not attrs:
        return MemoryConfig()

    return MemoryConfig(**attrs)


def _resolve_mode(choice: Optional[str], fallback: str) -> str:
    if not choice:
        return fallback

    normalized = str(choice).lower()
    if normalized in {"disabled", "none", "off"}:
        return "disabled"
    if normalized in {"remote", "remote_http", "remote-http", "backend"}:
        return "remote"
    if normalized in {"local", "local_http", "local-http"}:
        return "local"
    return fallback


def create_memory_client(
    config: Optional[DroidRunConfig] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> TrajectoryMemory:
    """Factory that returns the appropriate trajectory memory client."""

    resolved_config = config
    if resolved_config is None:
        try:
            resolved_config = ConfigManager().config
        except Exception:  # pragma: no cover - defensive fallback
            resolved_config = None

    if resolved_config is None:
        return DisabledTrajectoryMemoryClient()

    memory_source = getattr(resolved_config, "memory", None)
    memory_config = _normalize_memory_config(memory_source)

    provider_choice = memory_config.mode
    mode_kwarg = kwargs.pop("mode", None)
    provider_kwarg = kwargs.pop("provider", None)

    resolved_mode = _resolve_mode(provider, provider_choice or "local")
    resolved_mode = _resolve_mode(provider_kwarg, resolved_mode)
    resolved_mode = _resolve_mode(mode_kwarg, resolved_mode)

    enabled_override = kwargs.pop("enabled", None)
    is_enabled = memory_config.enabled
    if enabled_override is not None:
        is_enabled = bool(enabled_override)

    if resolved_mode == "disabled" or not is_enabled:
        return DisabledTrajectoryMemoryClient()

    base_url = kwargs.pop("base_url", memory_config.base_url)
    similarity_threshold = kwargs.pop(
        "similarity_threshold", memory_config.similarity_threshold
    )
    timeout = kwargs.pop("timeout", memory_config.timeout)
    auth_token = kwargs.pop("auth_token", memory_config.auth_token)

    if similarity_threshold is not None:
        similarity_threshold = float(similarity_threshold)
    if timeout is not None:
        timeout = float(timeout)

    if resolved_mode == "remote":
        if not base_url:
            logger.debug(
                "Remote memory mode selected but no base_url configured; disabling client.",
            )
            return DisabledTrajectoryMemoryClient()

        return RemoteHttpTrajectoryMemoryClient(
            base_url=base_url,
            similarity_threshold=similarity_threshold,
            timeout=timeout,
            auth_token=auth_token,
            enabled=True,
            **kwargs,
        )

    # Default to local HTTP client
    base_url = base_url or DEFAULT_BASE_URL
    return LocalHttpTrajectoryMemoryClient(
        base_url=base_url,
        similarity_threshold=similarity_threshold,
        timeout=timeout,
        enabled=True,
        **kwargs,
    )


def parse_trajectory_for_manual(trajectory: Trajectory) -> TrajectoryManual:
    manual: TrajectoryManual = {
        "goal": trajectory.goal,
        "initial_plan": {},
        "execution_steps": [],
        "memory_updates": [],
    }

    step_counter = 1
    status = "unknown"
    plan_captured = False

    for event in trajectory.events:
        if isinstance(event, ManagerInternalPlanEvent):
            manual.setdefault("initial_plan", {})

            if event.plan and not plan_captured:
                manual["initial_plan"]["plan_code"] = event.plan
                plan_captured = True

            if event.thought and not manual["initial_plan"].get("reasoning"):
                manual["initial_plan"]["reasoning"] = event.thought

            if event.manager_answer:
                manual["final_answer"] = event.manager_answer

            if event.memory_update:
                manual["memory_updates"].append(event.memory_update)

        elif isinstance(event, ExecutorInternalActionEvent):
            manual["execution_steps"].append(
                {
                    "step_index": step_counter,
                    "reasoning": event.thought or "",
                    "action_code": event.action_json,
                    "description": event.description,
                }
            )
            step_counter += 1

        elif isinstance(event, ExecutorInternalResultEvent):
            if manual["execution_steps"]:
                step = manual["execution_steps"][-1]
                step["summary"] = event.summary
                step["outcome"] = bool(event.outcome)
                if event.action_json and not step.get("action_code"):
                    step["action_code"] = event.action_json
                if event.thought and not step.get("reasoning"):
                    step["reasoning"] = event.thought

        elif isinstance(event, TaskThinkingEvent) and event.code:
            manual["execution_steps"].append(
                {
                    "step_index": step_counter,
                    "reasoning": event.thoughts or "",
                    "action_code": event.code,
                }
            )
            step_counter += 1

        elif isinstance(event, FinalizeEvent):
            status = "success" if getattr(event, "success", False) else "failure"

    manual["status"] = status

    if not manual["memory_updates"]:
        manual.pop("memory_updates")

    return manual


def format_manual_for_prompt(manual: TrajectoryManual) -> str:
    status = (manual.get("status") or "unknown").lower()
    if status == "success":
        heading = "**Successful Trajectory Reference**"
    elif status == "failure":
        heading = "**Failed Trajectory Reference (learn from mistakes)**"
    else:
        heading = "**Trajectory Reference**"

    lines = [heading, f"**Goal:** {manual.get('goal', 'N/A')}"]

    initial_plan = manual.get("initial_plan", {})
    if initial_plan:
        plan_reasoning = initial_plan.get("reasoning") or "N/A"
        plan_code = initial_plan.get("plan_code") or "N/A"
        lines.extend(
            [
                "---",
                "**Initial Plan:**",
                f"*   **Reasoning:** {plan_reasoning}",
                f"*   **Plan:** {plan_code}",
            ]
        )

    if manual.get("final_answer"):
        lines.extend(
            [
                "---",
                f"**Final Answer:** {manual['final_answer']}",
            ]
        )

    if manual.get("memory_updates"):
        lines.extend(["---", "**Memory Updates:**"])
        for update in manual["memory_updates"]:
            lines.append(f"*   {update}")

    execution_steps = manual.get("execution_steps", [])
    lines.extend(["---", "**Execution Steps:**"])

    if not execution_steps:
        lines.append("*   No execution steps recorded.")
    else:
        for step in execution_steps:
            lines.append(f"**Step {step.get('step_index', '?')}:**")
            lines.append(f"*   **Reasoning:** {step.get('reasoning', 'N/A')}")
            if step.get("description"):
                lines.append(f"*   **Description:** {step['description']}")
            action = step.get("action_code", "")
            if action:
                lines.append(f"*   **Action:** `{action}`")
            if step.get("summary"):
                lines.append(f"*   **Summary:** {step['summary']}")
            if "outcome" in step:
                outcome = "Success" if step.get("outcome") else "Failure"
                lines.append(f"*   **Outcome:** {outcome}")

    return "\n".join(lines)


__all__ = [
    "LocalHttpTrajectoryMemoryClient",
    "RemoteHttpTrajectoryMemoryClient",
    "TrajectoryMemoryClient",
    "create_memory_client",
    "parse_trajectory_for_manual",
    "format_manual_for_prompt",
]
