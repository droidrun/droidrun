"""
Events for the ExecutorAgent workflow.

Internal events for streaming to frontend/logging.
For DroidAgent coordination events, see droid/events.py
"""

from typing import List, Optional

from llama_index.core.workflow import Event

from droidrun.agent.action_result import ActionRecord
from droidrun.agent.usage import UsageResult


class ExecutorContextEvent(Event):
    """Context prepared, ready for LLM call."""

    subgoal: str


class ExecutorResponseEvent(Event):
    """LLM response received, ready for parsing."""

    response: str
    usage: Optional[UsageResult] = None


class ExecutorToolCallEvent(Event):
    """Tool calls parsed, ready to execute."""

    tool_calls_repr: Optional[str] = None
    thought: str = ""
    full_response: str = ""


class ExecutorActionResultEvent(Event):
    """Action execution results (all tool calls from one LLM response)."""

    actions: List[ActionRecord]
    thought: str = ""
    full_response: str = ""
