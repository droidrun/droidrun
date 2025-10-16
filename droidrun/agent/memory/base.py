"""Protocol definitions for trajectory memory clients."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TypedDict, runtime_checkable


class TrajectoryExecutionStep(TypedDict, total=False):
    step_index: int
    reasoning: str
    action_code: str
    description: str
    summary: str
    outcome: bool


class TrajectoryManual(TypedDict, total=False):
    goal: str
    initial_plan: Dict[str, Any]
    execution_steps: List[TrajectoryExecutionStep]
    status: str
    trajectory_id: str
    final_answer: str
    memory_updates: List[str]


@runtime_checkable
class TrajectoryMemory(Protocol):
    """Contract for trajectory memory providers."""

    enabled: bool

    def fetch_reference_manual(self, goal: str) -> Optional[str]:
        """Return a formatted manual string for a similar stored trajectory."""

    def find_trajectory(self, goal: str) -> Optional[TrajectoryManual]:
        """Return structured data for the closest stored trajectory."""

    def add_trajectory(self, manual: TrajectoryManual) -> bool:
        """Persist the provided trajectory manual and report success."""


class DisabledTrajectoryMemoryClient(TrajectoryMemory):
    """Fallback memory client used when trajectory memory is disabled."""

    def __init__(self) -> None:
        self.enabled = False

    def fetch_reference_manual(self, goal: str) -> Optional[str]:  # noqa: D401
        return None

    def find_trajectory(self, goal: str) -> Optional[TrajectoryManual]:  # noqa: D401
        return None

    def add_trajectory(self, manual: TrajectoryManual) -> bool:  # noqa: D401
        return False


__all__ = [
    "TrajectoryManual",
    "TrajectoryExecutionStep",
    "TrajectoryMemory",
    "DisabledTrajectoryMemoryClient",
]
