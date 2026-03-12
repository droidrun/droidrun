"""
DroidAgent coordination events.

These events route between DroidAgent and child agents.
For internal agent events, see each agent's events.py file.
"""

from typing import Any, Dict, List, Optional

from llama_index.core.workflow import Event, StopEvent
from pydantic import BaseModel


class FastAgentExecuteEvent(Event):
    instruction: str


class FastAgentResultEvent(Event):
    success: bool
    reason: str
    instruction: str


# ============================================================================
# Manager/Executor coordination events
# ============================================================================


class ManagerInputEvent(Event):
    """Trigger Manager workflow for planning"""

    pass


class ManagerPlanEvent(Event):
    """
    Coordination event from ManagerAgent to DroidAgent.

    Used for workflow step routing only (NOT streamed to frontend).
    For internal events with memory_update metadata, see ManagerPlanDetailsEvent.
    """

    plan: str
    current_subgoal: str
    thought: str
    answer: str = ""
    success: Optional[bool] = None  # True/False if complete, None if in progress


class ExecutorInputEvent(Event):
    """Trigger Executor workflow for action execution"""

    current_subgoal: str


class ExecutorResultEvent(Event):
    """Executor finished with action result(s)."""

    actions: List[Dict[str, Any]] = []
    thought: str = ""


# ============================================================================
# Script executor coordination events
# ============================================================================


class ScripterExecutorInputEvent(Event):
    """Trigger ScripterAgent workflow for off-device operations"""

    task: str


class ScripterExecutorResultEvent(Event):
    """Scripter finished."""

    task: str
    message: str
    success: bool
    code_executions: int


# ============================================================================
# TEXT MANIPULATOR WORKFLOW EVENTS
# ============================================================================


class TextManipulatorInputEvent(Event):
    """Trigger TextManipulatorAgent workflow for text manipulation"""

    task: str


class TextManipulatorResultEvent(Event):
    task: str
    text_to_type: str
    code_ran: str


# ============================================================================
# EXTERNAL USER MESSAGE EVENTS
# ============================================================================


class ExternalUserMessageAppliedEvent(Event):
    message_ids: List[str]
    consumer: str
    step_number: int


class ExternalUserMessageDroppedEvent(Event):
    message_ids: List[str]
    reason: str
    step_number: int


# ============================================================================
# FINALIZATION EVENTS
# ============================================================================


class FinalizeEvent(Event):
    """Trigger finalization."""

    success: bool
    reason: str


class ResultEvent(StopEvent):
    """
    Final result from DroidAgent.

    Returned by DroidAgent.run() with:
    - success: Whether the task completed successfully
    - reason: Explanation or answer
    - steps: Number of steps taken
    - structured_output: Extracted structured data (if output_model was provided)
    """

    success: bool
    reason: str
    steps: int
    structured_output: Optional[BaseModel] = None
